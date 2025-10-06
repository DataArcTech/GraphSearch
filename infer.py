import argparse
import asyncio
from dotenv import load_dotenv

from graphrags import initialize_rag, naive_grag_reasoning, graph_search_reasoning, vanilla_llm_reasoning, naive_rag_reasoning
from utils import load_vdb

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="DeepGraphRAG CLI")
    parser.add_argument("-d", "--dataset", default="musique", choices=["hotpotqa", "musique", "2wikimultihopqa", "agriculture", "hypertension", "legal"], help="Dataset to use.")
    parser.add_argument("-m", "--method", default="graphsearch", choices=["graphsearch", "grag", "vanillallm", "naiverag"], help="Reasoning method.")
    parser.add_argument("-g", "--graphrag", default="lightrag", choices=["lightrag", "minirag", "nano", "pathrag", "hipporag", "hypergraphrag"], help="GraphRAG to use.")
    parser.add_argument("-k", "--top_k", default="30", type=int, help="top retrieved items.")
    args = parser.parse_args()

    if args.method in ["graphsearch", "grag"]:
        grag, grag_mode, QueryParam = initialize_rag(grag_name=args.graphrag, dataset=args.dataset)

    if args.method in ["naiverag"]:
        corpus_file = f"./datasets/contexts/{args.dataset}.txt"
        with open(corpus_file, "r", encoding="utf-8") as f:
            documents = f.readlines()
        index, embed_model = load_vdb(args.dataset, documents)

    question = "In what league did Jose Miranda's team compete?"
    if args.method == "vanillallm":
        asyncio.run(vanilla_llm_reasoning(question))
    elif args.method == "naiverag":
        asyncio.run(naive_rag_reasoning(question, documents, index, embed_model, args.top_k))
    elif args.method == "grag":
        asyncio.run(naive_grag_reasoning(question, grag, grag_mode, QueryParam, args.top_k))
    elif args.method == "graphsearch":
        asyncio.run(graph_search_reasoning(question, grag, grag_mode, QueryParam, args.top_k))
