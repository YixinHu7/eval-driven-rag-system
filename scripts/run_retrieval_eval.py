from src.core.config import settings
from src.evaluation.dataset import load_eval_questions
from src.evaluation.retrieval_metrics import (
    evaluate_retrieval_result,
    summarize_retrieval_results,
)
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


def print_method_results(method: str, results, summary) -> None:
    print("=" * 100)
    print(f"Method: {method}")
    print("=" * 100)

    print(f"Total questions: {summary.total_questions}")
    print(f"Supported questions: {summary.supported_questions}")
    print(f"Hit@k: {summary.hit_at_k:.3f}")
    print(f"Top-1 accuracy: {summary.top_1_accuracy:.3f}")

    print("\nPer-question results:")
    for result in results:
        print("-" * 100)
        print(f"ID: {result.question_id}")
        print(f"Type: {result.query_type}")
        print(f"Supported: {result.supported}")
        print(f"Query: {result.query}")
        print(f"Expected section: {result.expected_section}")
        print(f"Top-1 section: {result.top_1_section}")
        print(f"Hit@k: {result.hit_at_k}")
        print(f"Top-1 match: {result.top_1_match}")
        print(f"Retrieved sections: {result.retrieved_sections}")


def main() -> None:
    questions = load_eval_questions(settings.evaluation.eval_data_path)
    methods = ["dense", "bm25", "hybrid"]

    with SessionLocal() as session:
        for method in methods:
            retriever = get_retriever(method)
            method_results = []

            for question in questions:
                chunks = retriever.retrieve(
                    session=session,
                    query=question.query,
                    top_k=settings.retrieval.top_k,
                )

                result = evaluate_retrieval_result(
                    question=question,
                    method=method,
                    retrieved_chunks=chunks,
                )
                method_results.append(result)

            summary = summarize_retrieval_results(
                method=method,
                results=method_results,
            )

            print_method_results(method, method_results, summary)


if __name__ == "__main__":
    main()