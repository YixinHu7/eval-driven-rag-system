import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.core.models import AnswerResponse
from src.core.config import settings
from src.evaluation.answer_metrics import (
    evaluate_answer_result,
    summarize_answer_by_query_type,
    summarize_answer_results,
)
from src.evaluation.citation_metrics import evaluate_citations
from src.evaluation.dataset import EvalQuestion, load_eval_questions
from src.generation.answer_generator import SimpleAnswerGenerator
from src.generation.llm_answer_generator import LLMAnswerGenerator
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal
from src.routing.query_classifier import classify_query

RetrievalMethod = Literal["dense", "bm25", "hybrid"]
GeneratorName = Literal["simple", "llm"]


def serialize_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    if hasattr(model, "dict"):
        return model.dict()

    raise TypeError(f"Object is not serializable as a Pydantic model: {type(model)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer-level evaluation for simple or LLM generators."
    )

    parser.add_argument(
        "--generator",
        choices=["simple", "llm"],
        default="simple",
        help="Answer generator to evaluate.",
    )

    parser.add_argument(
        "--method",
        choices=["dense", "bm25", "hybrid", "all"],
        default="all",
        help="Retrieval method to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of evaluation questions.",
    )

    parser.add_argument(
        "--eval-path",
        type=str,
        default=str(settings.evaluation.eval_data_path),
        help="Path to evaluation questions JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.data.experiments_dir),
        help="Directory where evaluation reports are written.",
    )

    return parser.parse_args()


def resolve_methods(method_arg: str) -> list[RetrievalMethod]:
    if method_arg == "all":
        return ["dense", "bm25", "hybrid"]

    return [method_arg]  # type: ignore[list-item]


def build_generator(
    generator_name: GeneratorName,
) -> SimpleAnswerGenerator | LLMAnswerGenerator:
    if generator_name == "llm":
        return LLMAnswerGenerator()

    return SimpleAnswerGenerator()


def get_questions(eval_path: str, limit: int | None) -> list[EvalQuestion]:
    questions = load_eval_questions(Path(eval_path))

    if limit is not None:
        return questions[:limit]

    return questions


def evaluate_method(
    method: RetrievalMethod,
    generator_name: GeneratorName,
    questions: list[EvalQuestion],
) -> dict[str, Any]:
    retriever = get_retriever(method)
    generator = build_generator(generator_name)

    answer_eval_results = []
    citation_eval_results = []
    records: list[dict[str, Any]] = []

    for question in questions:
        with SessionLocal() as session:
            chunks = retriever.retrieve(
                session=session,
                query=question.query,
                top_k=settings.retrieval.top_k,
            )
        
        query_classification = classify_query(question.query)

        response: AnswerResponse = generator.generate(
            query=question.query,
            chunks=chunks,
            retrieval_strategy=method,
            query_type=query_classification.query_type,
        )

        answer_eval = evaluate_answer_result(
            question=question,
            method=method,
            response=response,
        )
        citation_eval = evaluate_citations(response)

        answer_eval_results.append(answer_eval)
        citation_eval_results.append(citation_eval)

        records.append(
            {
                "question": serialize_model(question),
                "response": serialize_model(response),
                "answer_eval": serialize_model(answer_eval),
                "citation_eval": serialize_model(citation_eval),
            }
        )

        status = "PASS" if getattr(answer_eval, "passed", False) else "FAIL"
        print(
            f"[{method} | {generator_name}] "
            f"{question.question_id}: {status} "
            f"abstained={response.abstained} "
            f"citations={len(response.citations)}"
        )

    answer_summary = summarize_answer_results(
        method=method,
        results=answer_eval_results,
    )
    slice_summary = summarize_answer_by_query_type(
        method=method,
        results=answer_eval_results,
    )
    citation_summary = summarize_citation_results(
        citation_eval_results=citation_eval_results,
        answer_eval_results=answer_eval_results,
    )

    return {
        "method": method,
        "generator": generator_name,
        "supported_questions": sum(1 for question in questions if question.supported),
        "answer_summary": serialize_model(answer_summary),
        "slice_summary": [serialize_model(item) for item in slice_summary],
        "citation_summary": citation_summary,
        "records": records,
    }


def summarize_citation_results(
    citation_eval_results: list[Any],
    answer_eval_results: list[Any],
) -> dict[str, Any]:
    if not citation_eval_results:
        return {
            "total": 0,
            "answered_total": 0,
            "abstained_total": 0,
            "citation_ids_valid_rate": 0.0,
            "citation_alignment_rate": 0.0,
            "average_citation_utilization": 0.0,
            "average_answered_citation_utilization": 0.0,
        }

    total = len(citation_eval_results)

    valid_count = sum(
        1 for result in citation_eval_results if result.citation_ids_valid
    )
    aligned_count = sum(
        1 for result in citation_eval_results if result.citation_alignment_correct
    )
    utilization_sum = sum(
        result.citation_utilization for result in citation_eval_results
    )

    answered_pairs = [
        (citation_result, answer_result)
        for citation_result, answer_result in zip(
            citation_eval_results,
            answer_eval_results,
            strict=True,
        )
        if not answer_result.abstained
    ]

    answered_total = len(answered_pairs)
    abstained_total = total - answered_total

    answered_utilization_sum = sum(
        citation_result.citation_utilization for citation_result, _ in answered_pairs
    )

    average_answered_citation_utilization = (
        answered_utilization_sum / answered_total if answered_total else 0.0
    )

    return {
        "total": total,
        "answered_total": answered_total,
        "abstained_total": abstained_total,
        "citation_ids_valid_rate": valid_count / total,
        "citation_alignment_rate": aligned_count / total,
        "average_citation_utilization": utilization_sum / total,
        "average_answered_citation_utilization": average_answered_citation_utilization,
    }


def write_json_report(
    report: dict[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Path:
    output_path = output_dir / f"answer_eval_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def write_csv_report(
    report: dict[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Path:
    output_path = output_dir / f"answer_eval_{timestamp}.csv"

    rows: list[dict[str, Any]] = []

    for method_result in report["method_results"]:
        method = method_result["method"]
        generator = method_result["generator"]

        for record in method_result["records"]:
            question = record["question"]
            response = record["response"]
            answer_eval = record["answer_eval"]
            citation_eval = record["citation_eval"]

            rows.append(
                {
                    "method": method,
                    "generator": generator,
                    "question_id": question["question_id"],
                    "query_type": question["query_type"],
                    "supported": question["supported"],
                    "abstained": response["abstained"],
                    "confidence": response["confidence"],
                    "answer": response["answer"],
                    "num_returned_citations": len(response["citations"]),
                    "abstention_correct": answer_eval["abstention_correct"],
                    "citation_presence_correct": answer_eval[
                        "citation_presence_correct"
                    ],
                    "passed": answer_eval["passed"],
                    "referenced_citation_ids": ";".join(
                        str(citation_id)
                        for citation_id in citation_eval["referenced_citation_ids"]
                    ),
                    "returned_citation_ids": ";".join(
                        str(citation_id)
                        for citation_id in citation_eval["returned_citation_ids"]
                    ),
                    "invalid_citation_ids": ";".join(
                        str(citation_id)
                        for citation_id in citation_eval["invalid_citation_ids"]
                    ),
                    "missing_returned_citation_ids": ";".join(
                        str(citation_id)
                        for citation_id in citation_eval[
                            "missing_returned_citation_ids"
                        ]
                    ),
                    "unreferenced_returned_citation_ids": ";".join(
                        str(citation_id)
                        for citation_id in citation_eval[
                            "unreferenced_returned_citation_ids"
                        ]
                    ),
                    "citation_ids_valid": citation_eval["citation_ids_valid"],
                    "citation_alignment_correct": citation_eval[
                        "citation_alignment_correct"
                    ],
                    "citation_utilization": citation_eval["citation_utilization"],
                }
            )

    if not rows:
        return output_path

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_summary(report: dict[str, Any]) -> None:
    print("\nAnswer Evaluation Summary")
    print("=" * 80)

    for method_result in report["method_results"]:
        method = method_result["method"]
        generator = method_result["generator"]
        answer_summary = method_result["answer_summary"]
        citation_summary = method_result["citation_summary"]

        print(f"\nMethod: {method}")
        print(f"Generator: {generator}")
        print(f"Total questions: {answer_summary['total_questions']}")
        print(f"Supported questions: {method_result['supported_questions']}")
        print(f"Abstention accuracy: {answer_summary['abstention_accuracy']:.3f}")
        print(
            "Citation presence accuracy: "
            f"{answer_summary['citation_presence_accuracy']:.3f}"
        )
        print(f"Pass rate: {answer_summary['pass_rate']:.3f}")
        print(
            "Citation ID validity rate: "
            f"{citation_summary['citation_ids_valid_rate']:.3f}"
        )
        print(
            "Citation alignment rate: "
            f"{citation_summary['citation_alignment_rate']:.3f}"
        )
        print(
            "Average citation utilization: "
            f"{citation_summary['average_citation_utilization']:.3f}"
        )


def main() -> None:
    args = parse_args()

    generator_name: GeneratorName = args.generator
    methods = resolve_methods(args.method)
    questions = get_questions(args.eval_path, args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    method_results = [
        evaluate_method(
            method=method,
            generator_name=generator_name,
            questions=questions,
        )
        for method in methods
    ]

    report = {
        "timestamp": timestamp,
        "generator": generator_name,
        "methods": methods,
        "limit": args.limit,
        "eval_path": args.eval_path,
        "method_results": method_results,
    }

    json_path = write_json_report(
        report=report,
        output_dir=output_dir,
        timestamp=timestamp,
    )
    csv_path = write_csv_report(
        report=report,
        output_dir=output_dir,
        timestamp=timestamp,
    )

    print_summary(report)

    print("\nWrote reports:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
