import json
from pathlib import Path
from typing import Any

from src.core.config import settings


def find_latest_retrieval_report() -> Path:
    experiment_dir = settings.data.experiments_dir

    reports = sorted(
        experiment_dir.glob("retrieval_eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        raise FileNotFoundError(
            f"No retrieval evaluation reports found in {experiment_dir}"
        )

    return reports[0]


def classify_failure(result: dict[str, Any]) -> str:
    supported = result["supported"]
    hit_at_k = result["hit_at_k"]
    top_1_match = result["top_1_match"]

    expected_doc_id = result.get("expected_doc_id")
    expected_section = result.get("expected_section")
    top_1_doc_id = result.get("top_1_doc_id")
    top_1_section = result.get("top_1_section")

    if not supported:
        return "unsupported_query_retrieved"

    if hit_at_k and not top_1_match:
        return "expected_section_in_top_k_but_not_top_1"

    if expected_doc_id == top_1_doc_id and expected_section != top_1_section:
        return "right_document_wrong_section"

    if expected_doc_id != top_1_doc_id:
        return "wrong_document"

    if not hit_at_k:
        return "expected_section_missing_from_top_k"

    return "unknown"


def print_failure(result: dict[str, Any]) -> None:
    expected = (
        f"{result.get('expected_doc_id')}::{result.get('expected_section')}"
        if result.get("expected_doc_id") and result.get("expected_section")
        else None
    )

    top_1 = (
        f"{result.get('top_1_doc_id')}::{result.get('top_1_section')}"
        if result.get("top_1_doc_id") and result.get("top_1_section")
        else None
    )

    print("-" * 120)
    print(f"ID: {result['question_id']}")
    print(f"Method: {result['method']}")
    print(f"Type: {result['query_type']}")
    print(f"Query: {result['query']}")
    print(f"Failure type: {classify_failure(result)}")
    print(f"Expected: {expected}")
    print(f"Top-1: {top_1}")
    print("Retrieved:")
    for item in result.get("retrieved_doc_sections", []):
        print(f"  - {item}")


def main() -> None:
    report_path = find_latest_retrieval_report()
    print(f"Inspecting retrieval report: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload["results"]

    failures = [
        result
        for result in results
        if result["supported"] and not result["top_1_match"]
    ]

    print(f"Found {len(failures)} supported-query Top-1 failures")

    failures_by_method: dict[str, list[dict[str, Any]]] = {}

    for failure in failures:
        failures_by_method.setdefault(failure["method"], []).append(failure)

    for method, method_failures in failures_by_method.items():
        print("\n" + "=" * 120)
        print(f"Method: {method} | Failures: {len(method_failures)}")
        print("=" * 120)

        for failure in method_failures:
            print_failure(failure)


if __name__ == "__main__":
    main()