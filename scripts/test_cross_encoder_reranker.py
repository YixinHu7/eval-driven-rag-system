from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


def main() -> None:
    retriever = get_retriever("hybrid_reranked")

    with SessionLocal() as session:
        results = retriever.retrieve(
            session=session,
            query="How do Kubernetes readiness probes work?",
            top_k=5,
        )

    for result in results:
        print(
            f"{result.rank}. "
            f"{result.doc_id} | "
            f"{result.section_title} | "
            f"score={result.retrieval_score:.4f} | "
            f"method={result.retrieval_method}"
        )


if __name__ == "__main__":
    main()