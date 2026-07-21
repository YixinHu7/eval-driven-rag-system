import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_answer_eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final answer-evaluation experiment matrix."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.data.experiments_dir / "final_matrix"),
        help="Root directory for final matrix results.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment commands without running them.",
    )

    return parser.parse_args()


def build_common_args() -> list[str]:
    return [
        "--generator",
        "llm",
        "--context-expansion-window",
        str(settings.generation.context_expansion_window),
        "--max-expanded-context-chunks",
        str(settings.generation.max_expanded_context_chunks),
        "--max-neighbor-context-chars",
        str(settings.generation.max_neighbor_context_chars),
    ]


def build_scenarios() -> list[dict[str, Any]]:
    default_top_k = settings.retrieval.top_k
    common_args = build_common_args()

    return [
        {
            "name": "hybrid_default_expansion_enabled",
            "description": (
                "Default hybrid answer evaluation with " "context expansion enabled."
            ),
            "args": [
                "--method",
                "hybrid",
                "--top-k",
                str(default_top_k),
                "--enable-context-expansion",
                *common_args,
            ],
        },
        {
            "name": "hybrid_default_expansion_disabled",
            "description": (
                "Default hybrid answer evaluation with " "context expansion disabled."
            ),
            "args": [
                "--method",
                "hybrid",
                "--top-k",
                str(default_top_k),
                *common_args,
                "--disable-context-expansion",
            ],
        },
        {
            "name": "hybrid_top1_targeted_expansion_enabled",
            "description": (
                "Targeted neighbor-dependent evaluation with "
                "top-k 1 and expansion enabled."
            ),
            "args": [
                "--method",
                "hybrid",
                "--top-k",
                "1",
                "--question-ids",
                "q025",
                "q026",
                "--enable-context-expansion",
                *common_args,
            ],
        },
        {
            "name": "hybrid_top1_targeted_expansion_disabled",
            "description": (
                "Targeted neighbor-dependent evaluation with "
                "top-k 1 and expansion disabled."
            ),
            "args": [
                "--method",
                "hybrid",
                "--top-k",
                "1",
                "--question-ids",
                "q025",
                "q026",
                *common_args,
                "--disable-context-expansion",
            ],
        },
        {
            "name": "hybrid_reranked_default_expansion_enabled",
            "description": (
                "Hybrid reranked answer evaluation with "
                "default top-k and expansion enabled."
            ),
            "args": [
                "--method",
                "hybrid_reranked",
                "--top-k",
                str(default_top_k),
                "--enable-context-expansion",
                *common_args,
            ],
        },
    ]


def build_command(
    scenario: dict[str, Any],
    scenario_output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ANSWER_EVAL_SCRIPT),
        *scenario["args"],
        "--output-dir",
        str(scenario_output_dir),
    ]


def build_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()

    existing_pythonpath = environment.get(
        "PYTHONPATH",
        "",
    )

    pythonpath_entries = [
        str(PROJECT_ROOT),
    ]

    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return environment


def run_scenario(
    scenario: dict[str, Any],
    scenario_output_dir: Path,
    dry_run: bool,
) -> None:
    command = build_command(
        scenario=scenario,
        scenario_output_dir=scenario_output_dir,
    )

    print("\n" + "=" * 100)
    print(f"Scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print("Command:")
    print(" ".join(command))
    print("=" * 100)

    if dry_run:
        return

    scenario_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=build_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    console_output = completed_process.stdout + completed_process.stderr

    console_path = scenario_output_dir / "console_output.txt"
    console_path.write_text(
        console_output,
        encoding="utf-8",
    )

    print(console_output)

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Evaluation scenario failed: "
            f"{scenario['name']}. "
            f"See {console_path}."
        )


def find_report_path(
    scenario_output_dir: Path,
) -> Path:
    report_paths = sorted(scenario_output_dir.glob("answer_eval_*.json"))

    if len(report_paths) != 1:
        raise RuntimeError(
            "Expected exactly one JSON report in "
            f"{scenario_output_dir}, but found "
            f"{len(report_paths)}."
        )

    return report_paths[0]


def load_report(
    scenario_output_dir: Path,
) -> dict[str, Any]:
    report_path = find_report_path(scenario_output_dir)

    with report_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_nested_value(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def summarize_report(
    scenario: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    method_results = report.get(
        "method_results",
        [],
    )

    if len(method_results) != 1:
        raise RuntimeError(
            "Each final matrix scenario must contain " "exactly one method result."
        )

    method_result = method_results[0]
    answer_summary = method_result["answer_summary"]
    citation_summary = method_result["citation_summary"]
    expansion_summary = method_result.get(
        "context_expansion_summary",
        {},
    )

    return {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "method": method_result["method"],
        "generator": method_result["generator"],
        "top_k": method_result["top_k"],
        "expansion_enabled": get_nested_value(
            report,
            "context_expansion",
            "enabled",
        ),
        "expansion_window": get_nested_value(
            report,
            "context_expansion",
            "window",
        ),
        "max_expanded_context_chunks": (
            get_nested_value(
                report,
                "context_expansion",
                "max_expanded_context_chunks",
            )
        ),
        "max_neighbor_context_chars": (
            get_nested_value(
                report,
                "context_expansion",
                "max_neighbor_context_chars",
            )
        ),
        "question_ids": report.get("question_ids"),
        "total_questions": answer_summary["total_questions"],
        "supported_questions": method_result["supported_questions"],
        "abstention_accuracy": answer_summary["abstention_accuracy"],
        "citation_presence_accuracy": (answer_summary["citation_presence_accuracy"]),
        "pass_rate": answer_summary["pass_rate"],
        "required_evidence_questions": (
            answer_summary.get(
                "required_evidence_questions",
                0,
            )
        ),
        "required_evidence_accuracy": (
            answer_summary.get(
                "required_evidence_accuracy",
                0.0,
            )
        ),
        "average_required_evidence_coverage": (
            answer_summary.get(
                "average_required_evidence_coverage",
                0.0,
            )
        ),
        "citation_ids_valid_rate": (citation_summary["citation_ids_valid_rate"]),
        "citation_alignment_rate": (citation_summary["citation_alignment_rate"]),
        "average_citation_utilization": (
            citation_summary["average_citation_utilization"]
        ),
        "average_answered_citation_utilization": (
            citation_summary["average_answered_citation_utilization"]
        ),
        "questions_with_added_neighbors": (
            expansion_summary.get(
                "questions_with_added_neighbors",
                0,
            )
        ),
        "neighbor_addition_rate": (
            expansion_summary.get(
                "neighbor_addition_rate",
                0.0,
            )
        ),
        "average_initial_chunks": (
            expansion_summary.get(
                "average_initial_chunks",
                0.0,
            )
        ),
        "average_final_chunks": (
            expansion_summary.get(
                "average_final_chunks",
                0.0,
            )
        ),
        "average_added_neighbors": (
            expansion_summary.get(
                "average_added_neighbors",
                0.0,
            )
        ),
        "average_added_context_chars": (
            expansion_summary.get(
                "average_added_context_chars",
                0.0,
            )
        ),
        "max_added_neighbors": (
            expansion_summary.get(
                "max_added_neighbors",
                0,
            )
        ),
    }


def write_summary_json(
    rows: list[dict[str, Any]],
    matrix_output_dir: Path,
) -> Path:
    output_path = matrix_output_dir / "matrix_summary.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def write_summary_csv(
    rows: list[dict[str, Any]],
    matrix_output_dir: Path,
) -> Path:
    output_path = matrix_output_dir / "matrix_summary.csv"

    if not rows:
        return output_path

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    matrix_output_dir = Path(args.output_dir) / timestamp

    if not args.dry_run:
        matrix_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for scenario in scenarios:
        scenario_output_dir = matrix_output_dir / scenario["name"]

        run_scenario(
            scenario=scenario,
            scenario_output_dir=(scenario_output_dir),
            dry_run=args.dry_run,
        )

    if args.dry_run:
        return

    summary_rows = []

    for scenario in scenarios:
        scenario_output_dir = matrix_output_dir / scenario["name"]
        report = load_report(scenario_output_dir)
        summary_rows.append(
            summarize_report(
                scenario=scenario,
                report=report,
            )
        )

    json_path = write_summary_json(
        rows=summary_rows,
        matrix_output_dir=matrix_output_dir,
    )
    csv_path = write_summary_csv(
        rows=summary_rows,
        matrix_output_dir=matrix_output_dir,
    )

    print("\nFinal matrix complete.")
    print(f"Output directory: {matrix_output_dir}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
