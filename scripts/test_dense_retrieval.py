from src.retrieval.dense import DenseRetriever
from src.storage.db import SessionLocal


def main() -> None:
    query = "When should I use a DaemonSet?"

    retriever = DenseRetriever()

    with SessionLocal() as session:
        results = retriever.retrieve(session=session, query=query, top_k=3)

    print(f"Query: {query}")
    print(f"Retrieved {len(results)} results")

    for result in results:
        print("-" * 60)
        print(f"Rank: {result.rank}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Section: {result.section_title}")
        print(f"Method: {result.retrieval_method}")
        print(f"Content: {result.content}")


if __name__ == "__main__":
    main()