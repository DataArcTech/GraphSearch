import asyncio
import json
from deepsearch.prompts import PROMPTS
from utils import openai_complete

async def keywords_extraction(query):
    try:
        keyword_prompt = PROMPTS["keywords_extraction"].format(query=query)
        keywords_response = await openai_complete(prompt=keyword_prompt)
        
        try:
            keywords_data = json.loads(keywords_response)
            high_level_keywords = keywords_data.get("high_level_keywords", [])
            high_level_keywords = ", ".join(high_level_keywords).strip()
            low_level_keywords = keywords_data.get("low_level_keywords", [])
            low_level_keywords = ", ".join(low_level_keywords).strip()
        except json.JSONDecodeError:
            high_level_keywords = keywords_response.strip()
            low_level_keywords = keywords_response.strip()
        
        return high_level_keywords, low_level_keywords
    except:
        return "", ""

async def question_decomposition_deep(query):
    try:
        decomp_prompt = PROMPTS["query_decomposition_deep"].format(query=query)
        sub_queries = await openai_complete(prompt=decomp_prompt)
        return sub_queries.strip()
    except:
        return ""

async def question_decomposition_deep_kg(query):
    try:
        decomp_prompt = PROMPTS["query_decomposition_deep_kg"].format(query=query)
        sub_queries = await openai_complete(prompt=decomp_prompt)
        return sub_queries.strip()
    except:
        return ""

async def query_completer(sub_query, context_data):
    try:
        completer_prompt = PROMPTS["query_completer"].format(
            sub_query=sub_query,
            context_data=context_data
        )
        completed_query = await openai_complete(prompt=completer_prompt)
        return completed_query.strip()
    except:
        return ""

async def kg_query_completer(sub_query, context_data):
    try:
        completer_prompt = PROMPTS["kg_query_completer"].format(
            sub_query=sub_query,
            context_data=context_data
        )
        completed_query = await openai_complete(prompt=completer_prompt)
        return completed_query.strip()
    except:
        return ""
async def text_summary(query, context_data):
    try:
        summary_prompt = PROMPTS["retrieval_text_summarization"].format(
            query=query,
            context_data=context_data
        )
        text_summary = await openai_complete(prompt=summary_prompt)
        return text_summary.strip()
    except:
        return ""

async def kg_summary(query, context_data):
    try:
        kg_summary_prompt = PROMPTS["knowledge_graph_summarization"].format(
            query=query,
            context_data=context_data
        )
        kg_summary = await openai_complete(prompt=kg_summary_prompt)
        return kg_summary.strip()
    except:
        return ""

async def answer_generation(query, context_data):
    try:
        answer_prompt = PROMPTS["answer_generation"].format(
            query=query,
            context_data=context_data
        )
        final_answer = await openai_complete(prompt=answer_prompt)
        return final_answer.strip()
    except:
        return ""

async def answer_generation_deep(query, context_data):
    try:
        answer_prompt = PROMPTS["answer_generation_deep"].format(
            query=query,
            context_data=context_data
        )
        final_answer = await openai_complete(prompt=answer_prompt)
        return final_answer.strip()
    except:
        return ""

async def evidence_verification(query, context_data, model_response):
    try:
        verify_prompt = PROMPTS["evidence_verification"].format(
            query=query,
            context_data=context_data,
            model_response=model_response
        )
        final_verification = await openai_complete(prompt=verify_prompt)
        return final_verification.strip()
    except:
        return ""

async def query_expansion(query, context_data, model_response, evidence_verification):
    try:
        query_expansion_prompt = PROMPTS["query_expansion"].format(
            query=query,
            context_data=context_data,
            model_response=model_response,
            evidence_verification=evidence_verification
        )
        expanded_queries = await openai_complete(prompt=query_expansion_prompt)
        return expanded_queries.strip()
    except:
        return ""

