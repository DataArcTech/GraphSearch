import argparse
from dotenv import load_dotenv

from graphrags import initialize_rag
from utils import load_vdb

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="DeepGraphRAG CLI")
    parser.add_argument("-d", "--dataset", default="musique", choices=["hotpotqa", "musique", "2wikimultihopqa", "agriculture", "hypertension", "legal"], help="Dataset to use.")
    parser.add_argument("-g", "--graphrag", default="lightrag", choices=["lightrag", "minirag", "nano", "pathrag", "hipporag", "hypergraphrag"], help="GraphRAG to use.")
    args = parser.parse_args()

    if args.method in ["graphsearch", "grag"]:
        grag, grag_mode, QueryParam = initialize_rag(grag_name=args.graphrag, dataset=args.dataset)

    if args.method in ["naiverag"]:
        corpus_file = f"./datasets/contexts/{args.dataset}.txt"
        with open(corpus_file, "r", encoding="utf-8") as f:
            documents = f.readlines()
        index, embed_model = load_vdb(args.dataset, documents)
    
    corpus_file = f"./datasets/contexts/{args.dataset}.txt"
    with open(corpus_file, "r", encoding="utf-8") as f:
        if args.dataset in ["hotpotqa", "musique", "2wikimultihopqa"]:
            lines = f.readlines()
            grag.insert(lines)
        elif args.dataset in ["agriculture", "hypertension", "legal"]:
            grag.insert(f.read())