from src.generation.answer_generator import SimpleAnswerGenerator
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


def run_query(query: str, method: str = "hybrid") -> None:
    retriever = get_retriever(method)
    generator = SimpleAnswerGenerator()

    with SessionLocal() as session:
        chunks = retriever.retrieve(session=session, query=query, top_k=3)

    response = generator.generate(
        query=query,
        chunks=chunks,
        retrieval_strategy=method,
    )

    print("=" * 80)
    print(f"Query: {query}")
    print(f"Method: {method}")
    print(f"Abstained: {response.abstained}")
    print(f"Confidence: {response.confidence:.4f}")
    print(f"Answer: {response.answer}")

    if response.retrieved_chunks:
        top = response.retrieved_chunks[0]
        print(f"Top section: {top.section_title}")
        print(f"Top score: {top.retrieval_score:.6f}")


def main() -> None:
    supported_query = "When should I use a DaemonSet?"
    unsupported_query = "How do I bake a chocolate cake?"

    run_query(supported_query, method="hybrid")
    run_query(unsupported_query, method="hybrid")


if __name__ == "__main__":
    main()