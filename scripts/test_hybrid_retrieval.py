from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.storage.db import SessionLocal


def print_results(label: str, results) -> None:
    print("=" * 80)
    print(label)
    print("=" * 80)

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Method: {result.retrieval_method}")
        print(f"Score: {result.retrieval_score:.6f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Section: {result.section_title}")
        print(f"Content: {result.content}")
        print("-" * 80)


def main() -> None:
    queries = [
        "When should I use a DaemonSet?",
        "How does a DaemonSet update when nodes change?",
        "What is a DaemonSet?",
    ]

    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever()

    with SessionLocal() as session:
        for query in queries:
            print(f"\n\nQUERY: {query}\n")

            dense_results = dense_retriever.retrieve(session=session, query=query, top_k=3)
            bm25_results = bm25_retriever.retrieve(session=session, query=query, top_k=3)
            hybrid_results = hybrid_retriever.retrieve(
                session=session,
                query=query,
                top_k=3,
                candidate_k=10,
            )

            print_results("Dense Retrieval", dense_results)
            print_results("BM25 Retrieval", bm25_results)
            print_results("Hybrid Retrieval", hybrid_results)


if __name__ == "__main__":
    main()