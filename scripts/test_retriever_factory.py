from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


def run_method(method: str, query: str) -> None:
    retriever = get_retriever(method)

    with SessionLocal() as session:
        results = retriever.retrieve(session=session, query=query, top_k=3)

    print("=" * 80)
    print(f"Method: {method}")
    print("=" * 80)

    for result in results:
        print(f"Rank: {result.rank}")
        print(f"Score: {result.retrieval_score:.6f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Section: {result.section_title}")
        print("-" * 80)


def main() -> None:
    query = "When should I use a DaemonSet?"

    for method in ["dense", "bm25", "hybrid"]:
        run_method(method, query)


if __name__ == "__main__":
    main()