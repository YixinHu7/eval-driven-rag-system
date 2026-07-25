import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.core.models import AnswerResponse, RetrievedChunk
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
from src.routing.policy import (
    build_routing_abstention_response,
    should_short_circuit_answer,
)
from src.retrieval.context_expander import expand_with_neighbor_chunks
from src.evaluation.context_expansion_metrics import (
    ContextExpansionEvalResult,
    evaluate_context_expansion,
    summarize_context_expansion_results,
)

RetrievalMethod = Literal[
    "dense", "bm25", "hybrid", "dense_reranked", "hybrid_reranked"
]
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
        choices=["dense", "bm25", "hybrid", "dense_reranked", "hybrid_reranked", "all"],
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
        "--question-ids",
        nargs="+",
        default=None,
        help=(
            "Optional question IDs to evaluate, such as " "--question-ids q025 q026."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval.top_k,
        help="Number of initial chunks retrieved before context expansion.",
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

    context_expansion_group = parser.add_mutually_exclusive_group()

    context_expansion_group.add_argument(
        "--enable-context-expansion",
        dest="enable_context_expansion",
        action="store_true",
        help="Enable section-neighbor context expansion.",
    )

    context_expansion_group.add_argument(
        "--disable-context-expansion",
        dest="enable_context_expansion",
        action="store_false",
        help="Disable section-neighbor context expansion.",
    )

    parser.set_defaults(
        enable_context_expansion=None,
    )

    parser.add_argument(
        "--context-expansion-window",
        type=int,
        default=settings.generation.context_expansion_window,
        help="Neighbor window size for context expansion.",
    )

    parser.add_argument(
        "--max-expanded-context-chunks",
        type=int,
        default=settings.generation.max_expanded_context_chunks,
        help="Maximum number of chunks after context expansion.",
    )

    parser.add_argument(
        "--max-neighbor-context-chars",
        type=int,
        default=(settings.generation.max_neighbor_context_chars),
        help=(
            "Maximum total characters contributed by newly " "added neighbor chunks."
        ),
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


def get_questions(
    eval_path: str,
    limit: int | None,
    question_ids: list[str] | None,
) -> list[EvalQuestion]:
    questions = load_eval_questions(Path(eval_path))

    if question_ids:
        requested_ids = set(question_ids)

        selected_questions = [
            question for question in questions if question.question_id in requested_ids
        ]

        found_ids = {question.question_id for question in selected_questions}
        missing_ids = sorted(requested_ids - found_ids)

        if missing_ids:
            missing_text = ", ".join(missing_ids)
            raise ValueError(f"Unknown evaluation question IDs: {missing_text}")

        questions = selected_questions

    if limit is not None:
        questions = questions[:limit]

    return questions


def evaluate_method(
    method: RetrievalMethod,
    generator_name: GeneratorName,
    questions: list[EvalQuestion],
    top_k: int,
    enable_context_expansion: bool,
    context_expansion_window: int,
    max_expanded_context_chunks: int,
    max_neighbor_context_chars: int,
) -> dict[str, Any]:
    retriever = get_retriever(method)
    generator = build_generator(generator_name)

    answer_eval_results = []
    citation_eval_results = []
    context_expansion_eval_results: list[ContextExpansionEvalResult] = []
    records: list[dict[str, Any]] = []

    for question in questions:
        query_classification = classify_query(question.query)

        retrieval_performed = False
        initial_chunks: list[RetrievedChunk] = []
        final_chunks: list[RetrievedChunk] = []

        if should_short_circuit_answer(query_classification):
            response = build_routing_abstention_response(
                classification=query_classification,
                retrieval_strategy=method,
            )
        else:
            retrieval_performed = True

            with SessionLocal() as session:
                initial_chunks = retriever.retrieve(
                    session=session,
                    query=question.query,
                    top_k=top_k,
                )

                final_chunks = initial_chunks

                if enable_context_expansion:
                    final_chunks = expand_with_neighbor_chunks(
                        session=session,
                        chunks=initial_chunks,
                        window=context_expansion_window,
                        max_chunks=max_expanded_context_chunks,
                        max_neighbor_context_chars=(max_neighbor_context_chars),
                    )

            response = generator.generate(
                query=question.query,
                chunks=final_chunks,
                retrieval_strategy=method,
                query_type=query_classification.query_type,
            )

        answer_eval = evaluate_answer_result(
            question=question,
            method=method,
            response=response,
        )
        citation_eval = evaluate_citations(response)

        context_expansion_eval = evaluate_context_expansion(
            initial_chunks=initial_chunks,
            final_chunks=final_chunks,
            expansion_enabled=enable_context_expansion,
            retrieval_performed=retrieval_performed,
        )

        answer_eval_results.append(answer_eval)
        citation_eval_results.append(citation_eval)
        context_expansion_eval_results.append(context_expansion_eval)

        records.append(
            {
                "question": serialize_model(question),
                "query_classification": serialize_model(query_classification),
                "response": serialize_model(response),
                "answer_eval": serialize_model(answer_eval),
                "citation_eval": serialize_model(citation_eval),
                "context_expansion_eval": serialize_model(context_expansion_eval),
            }
        )

        status = "PASS" if getattr(answer_eval, "passed", False) else "FAIL"

        evidence_output = ""

        if question.required_evidence_sections:
            evidence_output = (
                " "
                f"evidence_coverage="
                f"{answer_eval.required_evidence_coverage:.3f} "
                f"evidence_passed="
                f"{answer_eval.required_evidence_passed}"
            )

        print(
            f"[{method} | {generator_name}] "
            f"{question.question_id}: {status} "
            f"abstained={response.abstained} "
            f"citations={len(response.citations)}"
            f"{evidence_output}"
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
    context_expansion_summary = summarize_context_expansion_results(
        results=context_expansion_eval_results,
        expansion_enabled=enable_context_expansion,
    )

    return {
        "method": method,
        "generator": generator_name,
        "top_k": top_k,
        "supported_questions": sum(1 for question in questions if question.supported),
        "answer_summary": serialize_model(answer_summary),
        "slice_summary": [serialize_model(item) for item in slice_summary],
        "citation_summary": citation_summary,
        "context_expansion_summary": serialize_model(context_expansion_summary),
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
            query_classification = record["query_classification"]
            response = record["response"]
            answer_eval = record["answer_eval"]
            citation_eval = record["citation_eval"]
            context_expansion_eval = record["context_expansion_eval"]

            rows.append(
                {
                    "method": method,
                    "generator": generator,
                    "question_id": question["question_id"],
                    "query_type": question["query_type"],
                    "classified_query_type": query_classification["query_type"],
                    "classification_confidence": query_classification["confidence"],
                    "matched_terms": ";".join(query_classification["matched_terms"]),
                    "supported": question["supported"],
                    "abstained": response["abstained"],
                    "confidence": response["confidence"],
                    "answer": response["answer"],
                    "num_returned_citations": len(response["citations"]),
                    "abstention_correct": answer_eval["abstention_correct"],
                    "citation_presence_correct": answer_eval[
                        "citation_presence_correct"
                    ],
                    "required_evidence_sections": ";".join(
                        answer_eval["required_evidence_sections"]
                    ),
                    "cited_evidence_sections": ";".join(
                        answer_eval["cited_evidence_sections"]
                    ),
                    "required_evidence_coverage": answer_eval[
                        "required_evidence_coverage"
                    ],
                    "required_evidence_passed": answer_eval["required_evidence_passed"],
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
                    "retrieval_performed": context_expansion_eval[
                        "retrieval_performed"
                    ],
                    "context_expansion_enabled": context_expansion_eval[
                        "expansion_enabled"
                    ],
                    "initial_chunk_count": context_expansion_eval[
                        "initial_chunk_count"
                    ],
                    "final_chunk_count": context_expansion_eval["final_chunk_count"],
                    "added_neighbor_count": context_expansion_eval[
                        "added_neighbor_count"
                    ],
                    "initial_context_chars": context_expansion_eval[
                        "initial_context_chars"
                    ],
                    "final_context_chars": context_expansion_eval[
                        "final_context_chars"
                    ],
                    "added_context_chars": context_expansion_eval[
                        "added_context_chars"
                    ],
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
        context_expansion_summary = method_result["context_expansion_summary"]

        print(f"\nMethod: {method}")
        print(f"Generator: {generator}")
        print(f"Initial retrieval top-k: {method_result['top_k']}")
        print(
            "Maximum neighbor context characters: "
            f"{report['context_expansion']['max_neighbor_context_chars']}"
        )
        print(f"Total questions: {answer_summary['total_questions']}")
        print(f"Supported questions: {method_result['supported_questions']}")
        print(f"Abstention accuracy: {answer_summary['abstention_accuracy']:.3f}")
        print(
            "Citation presence accuracy: "
            f"{answer_summary['citation_presence_accuracy']:.3f}"
        )
        print(f"Pass rate: {answer_summary['pass_rate']:.3f}")

        required_evidence_questions = answer_summary["required_evidence_questions"]

        if required_evidence_questions:
            print("Required evidence questions: " f"{required_evidence_questions}")
            print(
                "Required evidence accuracy: "
                f"{answer_summary['required_evidence_accuracy']:.3f}"
            )
            print(
                "Average required evidence coverage: "
                f"{answer_summary['average_required_evidence_coverage']:.3f}"
            )

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
        print(
            "Average answered citation utilization: "
            f"{citation_summary['average_answered_citation_utilization']:.3f}"
        )
        print(
            "Context expansion enabled: "
            f"{context_expansion_summary['expansion_enabled']}"
        )
        print(
            "Retrieval questions: "
            f"{context_expansion_summary['retrieval_questions']}"
        )
        print(
            "Questions with added neighbors: "
            f"{context_expansion_summary['questions_with_added_neighbors']}"
        )
        print(
            "Neighbor addition rate: "
            f"{context_expansion_summary['neighbor_addition_rate']:.3f}"
        )
        print(
            "Average initial chunks: "
            f"{context_expansion_summary['average_initial_chunks']:.3f}"
        )
        print(
            "Average final chunks: "
            f"{context_expansion_summary['average_final_chunks']:.3f}"
        )
        print(
            "Average added neighbors: "
            f"{context_expansion_summary['average_added_neighbors']:.3f}"
        )
        print(
            "Average added context characters: "
            f"{context_expansion_summary['average_added_context_chars']:.1f}"
        )
        print(
            "Maximum added neighbors: "
            f"{context_expansion_summary['max_added_neighbors']}"
        )


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.max_neighbor_context_chars < 0:
        raise ValueError("--max-neighbor-context-chars must be at least 0.")

    generator_name: GeneratorName = args.generator
    methods = resolve_methods(args.method)
    questions = get_questions(
        eval_path=args.eval_path,
        limit=args.limit,
        question_ids=args.question_ids,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    enable_context_expansion = (
        settings.generation.enable_context_expansion
        if args.enable_context_expansion is None
        else args.enable_context_expansion
    )

    method_results = [
        evaluate_method(
            method=method,
            generator_name=generator_name,
            questions=questions,
            top_k=args.top_k,
            enable_context_expansion=enable_context_expansion,
            context_expansion_window=args.context_expansion_window,
            max_expanded_context_chunks=args.max_expanded_context_chunks,
            max_neighbor_context_chars=(args.max_neighbor_context_chars),
        )
        for method in methods
    ]

    report = {
        "timestamp": timestamp,
        "generator": generator_name,
        "methods": methods,
        "limit": args.limit,
        "question_ids": args.question_ids,
        "eval_path": args.eval_path,
        "top_k": args.top_k,
        "context_expansion": {
            "enabled": enable_context_expansion,
            "window": args.context_expansion_window,
            "max_expanded_context_chunks": args.max_expanded_context_chunks,
            "max_neighbor_context_chars": (args.max_neighbor_context_chars),
        },
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
