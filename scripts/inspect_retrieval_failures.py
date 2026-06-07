import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


def build_failure_records(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for failure in failures:
        expected = format_doc_section(
            failure.get("expected_doc_id"),
            failure.get("expected_section"),
        )
        top_1 = format_doc_section(
            failure.get("top_1_doc_id"),
            failure.get("top_1_section"),
        )

        records.append(
            {
                "question_id": failure["question_id"],
                "method": failure["method"],
                "query_type": failure["query_type"],
                "query": failure["query"],
                "failure_type": classify_failure(failure),
                "expected_doc_section": expected,
                "accepted_doc_sections": failure.get("accepted_doc_sections", []),
                "top_1_doc_section": top_1,
                "retrieved_doc_sections": failure.get("retrieved_doc_sections", []),
                "hit_at_k": failure["hit_at_k"],
                "top_1_match": failure["top_1_match"],
            }
        )

    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counter: Counter[str] = Counter()
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    by_query_type: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        failure_type = record["failure_type"]
        method = record["method"]
        query_type = record["query_type"]

        overall_counter[failure_type] += 1
        by_method[method][failure_type] += 1
        by_query_type[query_type][failure_type] += 1

    return {
        "total_failures": len(records),
        "overall": dict(overall_counter),
        "by_method": {
            method: dict(counter)
            for method, counter in sorted(by_method.items())
        },
        "by_query_type": {
            query_type: dict(counter)
            for query_type, counter in sorted(by_query_type.items())
        },
    }


def print_failure_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 120)
    print("Failure Summary")
    print("=" * 120)

    if summary["total_failures"] == 0:
        print("No failures found.")
        return

    print(f"Total failures: {summary['total_failures']}")

    print("\nOverall failure counts:")
    for failure_type, count in sorted(
        summary["overall"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"- {failure_type}: {count}")

    print("\nFailure counts by method:")
    for method, counts in summary["by_method"].items():
        total = sum(counts.values())
        print(f"\n{method} total failures: {total}")
        for failure_type, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  - {failure_type}: {count}")

    print("\nFailure counts by query type:")
    for query_type, counts in summary["by_query_type"].items():
        total = sum(counts.values())
        print(f"\n{query_type} total failures: {total}")
        for failure_type, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  - {failure_type}: {count}")


def print_failure(record: dict[str, Any]) -> None:
    print("-" * 120)
    print(f"ID: {record['question_id']}")
    print(f"Method: {record['method']}")
    print(f"Type: {record['query_type']}")
    print(f"Query: {record['query']}")
    print(f"Failure type: {record['failure_type']}")
    print(f"Expected: {record['expected_doc_section']}")
    print(f"Accepted: {record['accepted_doc_sections']}")
    print(f"Top-1: {record['top_1_doc_section']}")
    print("Retrieved:")
    for item in record["retrieved_doc_sections"]:
        print(f"  - {item}")


def print_failures_by_method(records: list[dict[str, Any]]) -> None:
    failures_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        failures_by_method[record["method"]].append(record)

    for method, method_failures in sorted(failures_by_method.items()):
        print("\n" + "=" * 120)
        print(f"Method: {method} | Failures: {len(method_failures)}")
        print("=" * 120)

        for failure in method_failures:
            print_failure(failure)


def write_json_report(
    output_path: Path,
    source_report: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(source_report),
        "summary": summary,
        "failures": records,
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_csv_report(output_path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "question_id",
        "method",
        "query_type",
        "query",
        "failure_type",
        "expected_doc_section",
        "accepted_doc_sections",
        "top_1_doc_section",
        "retrieved_doc_sections",
        "hit_at_k",
        "top_1_match",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    **record,
                    "accepted_doc_sections": " | ".join(
                        record["accepted_doc_sections"]
                    ),
                    "retrieved_doc_sections": " | ".join(
                        record["retrieved_doc_sections"]
                    ),
                }
            )


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

    records = build_failure_records(failures)
    summary = build_summary(records)

    print_failure_summary(summary)
    print_failures_by_method(records)

    output_dir = settings.data.experiments_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"retrieval_failures_{timestamp}.json"
    csv_path = output_dir / f"retrieval_failures_{timestamp}.csv"

    write_json_report(json_path, report_path, summary, records)
    write_csv_report(csv_path, records)

    print("\nSaved failure inspection reports:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()