import os
import re
import logging

from deepsearch.components import question_decomposition_deep, question_decomposition_deep_kg, answer_generation, query_completer, kg_query_completer, text_summary, kg_summary, answer_generation_deep, evidence_verification, query_expansion
from utils import context_filter, format_history_context, extract_words_str, openai_complete, vdb_retrieve, normalize, parse_expanded_queries
from config import EMBED_MODEL_NAME, GRAG_MODE

def initialize_rag(grag_name:str, dataset:str):
    from sentence_transformers import SentenceTransformer

    working_dir = f"./graphkb/{grag_name}/{dataset}"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir, exist_ok=True)

    EMBED_MODEL = SentenceTransformer(f"{EMBED_MODEL_NAME}", trust_remote_code=True)

    if grag_name == "lightrag":
        from grag_initializers.lightrag import init_lightrag
        from lightrag import QueryParam
        grag = init_lightrag(working_dir, EMBED_MODEL)
    elif grag_name == "minirag":
        from grag_initializers.minirag import init_minirag
        from minirag import QueryParam
        grag = init_minirag(working_dir, EMBED_MODEL)
    elif grag_name == "hypergraphrag":
        from grag_initializers.hypergraphrag import init_hypergraphrag
        from hypergraphrag import QueryParam
        grag = init_hypergraphrag(working_dir, EMBED_MODEL)
    elif grag_name == "pathrag":
        from grag_initializers.pathrag import init_pathrag
        from PathRAG import QueryParam
        grag = init_pathrag(working_dir, EMBED_MODEL)

    grag_mode = GRAG_MODE[grag_name]
    return grag, grag_mode, QueryParam

async def vanilla_llm_reasoning(question:str):
    logging.info("Starting vanilla LLM reasoning...")
    logging.info(f"Question: {question}")
    answer = await openai_complete(prompt=question["question"])
    logging.info(f"Answer: {answer}")
async def naive_rag_reasoning(question, documents, index, embed_model, top_k):
    logging.info("Starting naive rag reasoning...")
    logging.info(f"Question: {question}")
    retrieved_context = vdb_retrieve(question, documents, index, embed_model, top_k)
    logging.info(f"Retrieved Context: {retrieved_context}")
    answer = await answer_generation(question, "\n".join(retrieved_context))
    logging.info(f"Answer: {answer}")

async def naive_grag_reasoning(question, grag, grag_mode, QueryParam, top_k):
    logging.info("Starting agent deep reasoning...")
    logging.info(f"Question: {question}")
    context = await grag.aquery(question, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
    logging.info(f"GraphRAG Context: {context}")
    answer = await grag.aquery(question, QueryParam(mode=grag_mode, only_need_context=False, top_k=top_k))
    logging.info(f"Answer: {answer}")

async def graph_search_reasoning(question, grag, grag_mode, QueryParam, top_k):
    logging.info("Starting graph search reasoning...")
    grag_context_data = await grag.aquery(question, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
    logging.info(f"Initial Context: {grag_context_data}")
    
    grag_context_text_summary = await text_summary(question, grag_context_data)
    logging.info(f"Initial Context Text Summary: {grag_context_text_summary}")
    grag_context_kg_summary = await kg_summary(question, grag_context_data)
    logging.info(f"Initial Context KG Summary: {grag_context_kg_summary}")
        
    # Question Decomposition
    decomposition_output = await question_decomposition_deep(question)
    kg_decomposition_output = await question_decomposition_deep_kg(question)
    
    sub_query_pattern = r'"Sub-query \d+":\s*"([^"]+)"'
    sub_kg_query_pattern = r'"Sub-query \d+":\s*(\[[^\]]+\])'
    sub_queries = re.findall(sub_query_pattern, decomposition_output)
    sub_kg_queries = re.findall(sub_kg_query_pattern, kg_decomposition_output)

    logging.info(f"Sub Queries: {sub_queries}")
    logging.info(f"Sub KG Queries: {sub_kg_queries}")

    text_query_history = []
    
    # Iterative Retrieval
    for i, sub_query in enumerate(sub_queries):
        text_query_history_str = format_history_context(text_query_history)
        if "#" in sub_query:
            sub_query = await query_completer(sub_query, decomposition_output + "\n\n" + text_query_history_str)
        logging.info(f"Sub Query: {sub_query}")
        # retrieve the graph database
        sub_query_context = await grag.aquery(sub_query, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
        logging.info(f"Sub Query Context: {sub_query_context}")

        # summarize the context for sub query, then try to answer use current context
        sub_query_context_summary = await text_summary(sub_query, sub_query_context)
        logging.info(f"Sub Query Context Summary: {sub_query_context_summary}")
        # answer sub query
        sub_query_context_data = text_query_history_str + "\n\n" + sub_query_context_summary
        sub_query_answer = await answer_generation(sub_query, sub_query_context_data)
        logging.info(f"Sub Query Answer: {sub_query_answer}")

        text_query_history.append((sub_query, sub_query_context_summary, sub_query_answer))

    # merge the history
    text_query_history_str = format_history_context(text_query_history)
    # Logic Drafting
    text_final_answer = await answer_generation_deep(question, text_query_history_str)
    logging.info(f"Logic Drafting: {text_final_answer}")

    # Self-Reflection, Optional
    # Evidence verification
    text_verification_result = await evidence_verification(question, text_query_history_str, text_final_answer)
    logging.info(f"Evidence Verification: {text_verification_result}")
    
    # Query Expansion
    if "yes" in normalize(text_verification_result):
        query_expansion_result = await query_expansion(question, text_query_history_str, text_final_answer, text_verification_result)
        expanded_queries = parse_expanded_queries(query_expansion_result)
    
        for expanded_query in expanded_queries:
            expanded_query_context = await grag.aquery(expanded_query, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
            logging.info(f"Expanded Query Context: {expanded_query_context}")
            
            expanded_query_context_summary = await text_summary(expanded_query, expanded_query_context)
            logging.info(f"Expanded Query Context Summary: {expanded_query_context_summary}")
            
            text_query_history.append((expanded_query, expanded_query_context_summary, ""))
            
        text_query_history_str = format_history_context(text_query_history)
        
    kg_query_history = []
    for i, sub_kg_query in enumerate(sub_kg_queries):
        kg_query_history_str = format_history_context(kg_query_history)
        if i > 0:
            sub_kg_query = await kg_query_completer(sub_kg_query, kg_decomposition_output + "\n\n" + kg_query_history_str)
        
        logging.info(f"Sub KG Query: {sub_kg_query}")
        sub_kg_query_cleaned = extract_words_str(sub_kg_query)
        sub_kg_query_context = await grag.aquery(sub_kg_query_cleaned, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
        # sub_kg_query_context = context_filter(grag_name, sub_kg_query_context, "graph")
        logging.info(f"Sub KG Query Context: {sub_kg_query_context}")
        
        sub_kg_query_context_summary = await kg_summary(sub_kg_query, sub_kg_query_context)
        logging.info(f"Sub KG Query Context Summary: {sub_kg_query_context_summary}")

        sub_kg_query_context_data = kg_query_history_str + "\n\n" + sub_kg_query_context_summary
        sub_kg_query_answer = await answer_generation(sub_kg_query, sub_kg_query_context_data)
        logging.info(f"Sub KG Query Answer: {sub_kg_query_answer}")

        kg_query_history.append((sub_kg_query, sub_kg_query_context_summary, sub_kg_query_answer))

    # merge the history
    kg_query_history_str = format_history_context(kg_query_history)
    # Logic Drafting
    kg_final_answer = await answer_generation_deep(question, kg_query_history_str)
    logging.info(f"KG Logic Drafting: {kg_final_answer}")

    # Self-Reflection, Optional
    # Evidence verification
    kg_verification_result = await evidence_verification(question, kg_query_history_str, kg_final_answer)
    logging.info(f"KG Evidence Verification: {kg_verification_result}")
    # Query Expansion
    if "yes" in normalize(kg_verification_result):
        query_expansion_result = await query_expansion(question, kg_query_history_str, kg_final_answer, kg_verification_result)
        expanded_queries = parse_expanded_queries(query_expansion_result)
    
        for expanded_query in expanded_queries:
            expanded_query_context = await grag.aquery(expanded_query, QueryParam(mode=grag_mode, only_need_context=True, top_k=top_k))
            logging.info(f"Expanded KG Query Context: {expanded_query_context}")
            
            expanded_query_context_summary = await kg_summary(expanded_query, expanded_query_context)
            logging.info(f"Expanded KG Query Context Summary: {expanded_query_context_summary}")
            
            kg_query_history.append((expanded_query, expanded_query_context_summary, ""))
            
        kg_query_history_str = format_history_context(kg_query_history)

    combined_query_history_str = "Background information:\n" + grag_context_text_summary + "\n" + grag_context_kg_summary + "\n\n" + text_query_history_str + "\n\n" + kg_query_history_str
    final_answer = await answer_generation_deep(question, combined_query_history_str)
    logging.info(f"Final Answer: {final_answer}")