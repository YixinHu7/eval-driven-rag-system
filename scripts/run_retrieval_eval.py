import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.core.config import settings
from src.evaluation.dataset import load_eval_questions
from src.evaluation.retrieval_metrics import (
    RetrievalEvalResult,
    RetrievalEvalSummary,
    RetrievalEvalSliceSummary,
    evaluate_retrieval_result,
    summarize_by_query_type,
    summarize_retrieval_results,
)
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal

RetrievalMethod = Literal[
    "dense",
    "bm25",
    "hybrid",
    "dense_reranked",
    "hybrid_reranked",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-level evaluation.")

    parser.add_argument(
        "--method",
        choices=[
            "dense",
            "bm25",
            "hybrid",
            "dense_reranked",
            "hybrid_reranked",
            "all",
        ],
        default="all",
        help=(
            "Retrieval method to evaluate. "
            "Use 'all' for dense, bm25, and hybrid only."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of evaluation questions.",
    )

    return parser.parse_args()


def resolve_methods(method_arg: str) -> list[RetrievalMethod]:
    if method_arg == "all":
        return ["dense", "bm25", "hybrid"]

    return [method_arg]  # type: ignore[list-item]


def print_method_results(
    method: str,
    results: list[RetrievalEvalResult],
    summary: RetrievalEvalSummary,
    slice_summaries: list[RetrievalEvalSliceSummary],
) -> None:
    print("=" * 100)
    print(f"Method: {method}")
    print("=" * 100)

    print(f"Total questions: {summary.total_questions}")
    print(f"Supported questions: {summary.supported_questions}")
    print(f"Hit@k: {summary.hit_at_k:.3f}")
    print(f"Top-1 accuracy: {summary.top_1_accuracy:.3f}")

    print("\nBy query type:")
    for slice_summary in slice_summaries:
        print(
            f"- {slice_summary.query_type}: "
            f"questions={slice_summary.total_questions}, "
            f"supported={slice_summary.supported_questions}, "
            f"hit@k={slice_summary.hit_at_k:.3f}, "
            f"top1={slice_summary.top_1_accuracy:.3f}"
        )

    print("\nPer-question results:")
    for result in results:
        expected_doc_section = (
            f"{result.expected_doc_id}::{result.expected_section}"
            if result.expected_doc_id and result.expected_section
            else None
        )

        top_1_doc_section = (
            f"{result.top_1_doc_id}::{result.top_1_section}"
            if result.top_1_doc_id and result.top_1_section
            else None
        )

        print("-" * 100)
        print(f"ID: {result.question_id}")
        print(f"Type: {result.query_type}")
        print(f"Supported: {result.supported}")
        print(f"Query: {result.query}")
        print(f"Expected doc-section: {expected_doc_section}")
        print(f"Accepted doc-sections: {result.accepted_doc_sections}")
        print(f"Top-1 doc-section: {top_1_doc_section}")
        print(f"Hit@k: {result.hit_at_k}")
        print(f"Top-1 match: {result.top_1_match}")
        print(f"Retrieved doc-sections: {result.retrieved_doc_sections}")


def write_json_report(
    output_path: Path,
    methods: list[RetrievalMethod],
    summaries: list[RetrievalEvalSummary],
    slice_summaries: list[RetrievalEvalSliceSummary],
    results: list[RetrievalEvalResult],
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "methods": methods,
            "top_k": settings.retrieval.top_k,
            "candidate_k": settings.retrieval.candidate_k,
            "embedding_model": settings.embedding.model_name,
            "embedding_dimensions": settings.embedding.embedding_dimensions,
            "reranker_model": getattr(settings.reranker, "model_name", None),
            "reranker_candidate_k": getattr(settings.reranker, "candidate_k", None),
        },
        "summaries": [summary.model_dump() for summary in summaries],
        "slice_summaries": [summary.model_dump() for summary in slice_summaries],
        "results": [result.model_dump() for result in results],
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_csv_report(
    output_path: Path,
    results: list[RetrievalEvalResult],
) -> None:
    fieldnames = [
        "question_id",
        "method",
        "query_type",
        "supported",
        "query",
        "expected_doc_id",
        "expected_section",
        "accepted_doc_sections",
        "top_1_doc_id",
        "top_1_section",
        "hit_at_k",
        "top_1_match",
        "retrieved_doc_sections",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "question_id": result.question_id,
                    "method": result.method,
                    "query_type": result.query_type,
                    "supported": result.supported,
                    "query": result.query,
                    "expected_doc_id": result.expected_doc_id,
                    "expected_section": result.expected_section,
                    "accepted_doc_sections": " | ".join(result.accepted_doc_sections),
                    "top_1_doc_id": result.top_1_doc_id,
                    "top_1_section": result.top_1_section,
                    "hit_at_k": result.hit_at_k,
                    "top_1_match": result.top_1_match,
                    "retrieved_doc_sections": " | ".join(result.retrieved_doc_sections),
                }
            )


def main() -> None:
    args = parse_args()

    questions = load_eval_questions(settings.evaluation.eval_data_path)

    if args.limit is not None:
        questions = questions[: args.limit]

    methods = resolve_methods(args.method)

    all_results: list[RetrievalEvalResult] = []
    summaries: list[RetrievalEvalSummary] = []
    all_slice_summaries: list[RetrievalEvalSliceSummary] = []

    with SessionLocal() as session:
        for method in methods:
            retriever = get_retriever(method)
            method_results: list[RetrievalEvalResult] = []

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

            slice_summaries = summarize_by_query_type(
                method=method,
                results=method_results,
            )

            print_method_results(
                method=method,
                results=method_results,
                summary=summary,
                slice_summaries=slice_summaries,
            )

            summaries.append(summary)
            all_slice_summaries.extend(slice_summaries)
            all_results.extend(method_results)

    output_dir = settings.data.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    method_label = args.method
    json_path = output_dir / f"retrieval_eval_{method_label}_{timestamp}.json"
    csv_path = output_dir / f"retrieval_eval_{method_label}_{timestamp}.csv"

    write_json_report(
        output_path=json_path,
        methods=methods,
        summaries=summaries,
        slice_summaries=all_slice_summaries,
        results=all_results,
    )
    write_csv_report(csv_path, all_results)

    print("\nSaved evaluation reports:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
