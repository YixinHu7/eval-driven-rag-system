import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.evaluation.dataset import load_eval_questions
from src.evaluation.query_classification_metrics import (
    evaluate_query_classification,
    summarize_query_classification_results,
)
from src.routing.query_classifier import classify_query


def serialize_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    if hasattr(model, "dict"):
        return model.dict()

    raise TypeError(f"Object is not serializable as a Pydantic model: {type(model)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run query classification evaluation."
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

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of evaluation questions.",
    )

    return parser.parse_args()


def write_json_report(
    report: dict[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Path:
    output_path = output_dir / f"query_classification_eval_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path


def write_csv_report(
    report: dict[str, Any],
    output_dir: Path,
    timestamp: str,
) -> Path:
    output_path = output_dir / f"query_classification_eval_{timestamp}.csv"

    rows: list[dict[str, Any]] = []

    for record in report["records"]:
        question = record["question"]
        classification = record["classification"]
        evaluation = record["evaluation"]

        rows.append(
            {
                "question_id": question["question_id"],
                "query": question["query"],
                "query_type": question["query_type"],
                "supported": question["supported"],
                "expected_domain": evaluation["expected_domain"],
                "predicted_domain": evaluation["predicted_domain"],
                "correct": evaluation["correct"],
                "confidence": classification["confidence"],
                "matched_terms": ";".join(classification["matched_terms"]),
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
    summary = report["summary"]

    print("\nQuery Classification Evaluation Summary")
    print("=" * 80)
    print(f"Total questions: {summary['total_questions']}")
    print(f"Accuracy: {summary['accuracy']:.3f}")
    print(f"Correct count: {summary['correct_count']}")
    print(f"Incorrect count: {summary['incorrect_count']}")
    print(f"False in-domain count: {summary['false_in_domain_count']}")
    print(f"False out-of-domain count: {summary['false_out_of_domain_count']}")
    print(f"Ambiguous count: {summary['ambiguous_count']}")

    print("\nConfusion counts:")
    for key, value in summary["confusion_counts"].items():
        print(f"- {key}: {value}")

    failed_records = [
        record
        for record in report["records"]
        if not record["evaluation"]["correct"]
    ]

    if failed_records:
        print("\nFailures:")
        for record in failed_records:
            evaluation = record["evaluation"]
            question = record["question"]
            classification = record["classification"]

            print(
                f"- {question['question_id']}: "
                f"expected={evaluation['expected_domain']} "
                f"predicted={evaluation['predicted_domain']} "
                f"confidence={classification['confidence']:.3f} "
                f"matched_terms={classification['matched_terms']} "
                f"query={question['query']}"
            )


def main() -> None:
    args = parse_args()

    questions = load_eval_questions(Path(args.eval_path))

    if args.limit is not None:
        questions = questions[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    eval_results = []

    for question in questions:
        classification = classify_query(question.query)
        evaluation = evaluate_query_classification(
            question=question,
            classification=classification,
        )

        eval_results.append(evaluation)

        records.append(
            {
                "question": serialize_model(question),
                "classification": serialize_model(classification),
                "evaluation": serialize_model(evaluation),
            }
        )

    summary = summarize_query_classification_results(eval_results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "eval_path": args.eval_path,
        "limit": args.limit,
        "summary": serialize_model(summary),
        "records": records,
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