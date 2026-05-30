import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import settings
from src.evaluation.answer_metrics import (
    AnswerEvalResult,
    AnswerEvalSliceSummary,
    AnswerEvalSummary,
    evaluate_answer_result,
    summarize_answer_by_query_type,
    summarize_answer_results,
)
from src.evaluation.dataset import load_eval_questions
from src.generation.answer_generator import SimpleAnswerGenerator
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


def print_method_results(
    method: str,
    summary: AnswerEvalSummary,
    slice_summaries: list[AnswerEvalSliceSummary],
    results: list[AnswerEvalResult],
) -> None:
    print("=" * 100)
    print(f"Method: {method}")
    print("=" * 100)

    print(f"Total questions: {summary.total_questions}")
    print(f"Abstention accuracy: {summary.abstention_accuracy:.3f}")
    print(f"Citation presence accuracy: {summary.citation_presence_accuracy:.3f}")
    print(f"Pass rate: {summary.pass_rate:.3f}")

    print("\nBy query type:")
    for item in slice_summaries:
        print(
            f"- {item.query_type}: "
            f"questions={item.total_questions}, "
            f"abstention={item.abstention_accuracy:.3f}, "
            f"citation={item.citation_presence_accuracy:.3f}, "
            f"pass={item.pass_rate:.3f}"
        )

    print("\nPer-question results:")
    for result in results:
        print("-" * 100)
        print(f"ID: {result.question_id}")
        print(f"Type: {result.query_type}")
        print(f"Supported: {result.supported}")
        print(f"Query: {result.query}")
        print(f"Abstained: {result.abstained}")
        print(f"Expected abstain: {result.expected_abstain}")
        print(f"Abstention correct: {result.abstention_correct}")
        print(f"Citation count: {result.citation_count}")
        print(f"Citation presence correct: {result.citation_presence_correct}")
        print(f"Passed: {result.passed}")


def write_json_report(
    output_path: Path,
    summaries: list[AnswerEvalSummary],
    slice_summaries: list[AnswerEvalSliceSummary],
    results: list[AnswerEvalResult],
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "top_k": settings.retrieval.top_k,
            "candidate_k": settings.retrieval.candidate_k,
            "embedding_model": settings.embedding.model_name,
            "embedding_dimensions": settings.embedding.embedding_dimensions,
        },
        "summaries": [summary.model_dump() for summary in summaries],
        "slice_summaries": [summary.model_dump() for summary in slice_summaries],
        "results": [result.model_dump() for result in results],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_report(
    output_path: Path,
    results: list[AnswerEvalResult],
) -> None:
    fieldnames = [
        "question_id",
        "method",
        "query_type",
        "supported",
        "query",
        "abstained",
        "expected_abstain",
        "abstention_correct",
        "citation_count",
        "citation_presence_correct",
        "passed",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.model_dump())


def main() -> None:
    questions = load_eval_questions(settings.evaluation.eval_data_path)
    methods = ["dense", "bm25", "hybrid"]

    generator = SimpleAnswerGenerator()

    all_results: list[AnswerEvalResult] = []
    summaries: list[AnswerEvalSummary] = []
    all_slice_summaries: list[AnswerEvalSliceSummary] = []

    with SessionLocal() as session:
        for method in methods:
            retriever = get_retriever(method)
            method_results: list[AnswerEvalResult] = []

            for question in questions:
                chunks = retriever.retrieve(
                    session=session,
                    query=question.query,
                    top_k=settings.retrieval.top_k,
                )

                response = generator.generate(
                    query=question.query,
                    chunks=chunks,
                    retrieval_strategy=method,
                )

                result = evaluate_answer_result(
                    question=question,
                    method=method,
                    response=response,
                )
                method_results.append(result)

            summary = summarize_answer_results(method, method_results)
            slice_summaries = summarize_answer_by_query_type(method, method_results)

            print_method_results(method, summary, slice_summaries, method_results)

            summaries.append(summary)
            all_slice_summaries.extend(slice_summaries)
            all_results.extend(method_results)

    output_dir = settings.data.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"answer_eval_{timestamp}.json"
    csv_path = output_dir / f"answer_eval_{timestamp}.csv"

    write_json_report(json_path, summaries, all_slice_summaries, all_results)
    write_csv_report(csv_path, all_results)

    print("\nSaved answer evaluation reports:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()