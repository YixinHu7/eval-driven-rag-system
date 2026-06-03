import json
from collections import Counter, defaultdict
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
        if result.get("retrieved_doc_sections"):
            return "unsupported_query_retrieved"
        return "unsupported_query_no_retrieval"

    if hit_at_k and not top_1_match:
        return "expected_section_in_top_k_but_not_top_1"

    if expected_doc_id == top_1_doc_id and expected_section != top_1_section:
        return "right_document_wrong_section"

    if expected_doc_id != top_1_doc_id:
        return "wrong_document"

    if not hit_at_k:
        return "expected_section_missing_from_top_k"

    return "unknown"


def is_failure(result: dict[str, Any], include_unsupported: bool = False) -> bool:
    if result["supported"]:
        return not result["top_1_match"]

    if include_unsupported:
        return bool(result.get("retrieved_doc_sections"))

    return False


def format_doc_section(doc_id: str | None, section: str | None) -> str | None:
    if doc_id and section:
        return f"{doc_id}::{section}"
    return None


def print_failure_summary(failures: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("Failure Summary")
    print("=" * 120)

    if not failures:
        print("No failures found.")
        return

    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    overall_counter: Counter[str] = Counter()

    for failure in failures:
        method = failure["method"]
        failure_type = classify_failure(failure)
        by_method[method][failure_type] += 1
        overall_counter[failure_type] += 1

    print("\nOverall failure counts:")
    for failure_type, count in overall_counter.most_common():
        print(f"- {failure_type}: {count}")

    print("\nFailure counts by method:")
    for method in sorted(by_method):
        total = sum(by_method[method].values())
        print(f"\n{method} total failures: {total}")
        for failure_type, count in by_method[method].most_common():
            print(f"  - {failure_type}: {count}")


def print_failure(result: dict[str, Any]) -> None:
    expected = format_doc_section(
        result.get("expected_doc_id"),
        result.get("expected_section"),
    )

    top_1 = format_doc_section(
        result.get("top_1_doc_id"),
        result.get("top_1_section"),
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


def print_failures_by_method(failures: list[dict[str, Any]]) -> None:
    failures_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for failure in failures:
        failures_by_method[failure["method"]].append(failure)

    for method, method_failures in sorted(failures_by_method.items()):
        print("\n" + "=" * 120)
        print(f"Method: {method} | Failures: {len(method_failures)}")
        print("=" * 120)

        for failure in method_failures:
            print_failure(failure)


def main() -> None:
    report_path = find_latest_retrieval_report()
    print(f"Inspecting retrieval report: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload["results"]

    failures = [
        result
        for result in results
        if is_failure(result, include_unsupported=False)
    ]

    print(f"Found {len(failures)} supported-query Top-1 failures")

    print_failure_summary(failures)
    print_failures_by_method(failures)


if __name__ == "__main__":
    main()