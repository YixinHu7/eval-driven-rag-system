from collections import defaultdict

from pydantic import BaseModel

from src.core.models import RetrievedChunk
from src.evaluation.dataset import EvalQuestion


class RetrievalEvalResult(BaseModel):
    question_id: str
    query: str
    method: str
    query_type: str
    supported: bool
    expected_doc_id: str | None
    expected_section: str | None
    accepted_doc_sections: list[str]
    top_1_doc_id: str | None
    top_1_section: str | None
    hit_at_k: bool
    top_1_match: bool
    retrieved_sections: list[str]
    retrieved_doc_sections: list[str]


class RetrievalEvalSummary(BaseModel):
    method: str
    total_questions: int
    supported_questions: int
    hit_at_k: float
    top_1_accuracy: float


class RetrievalEvalSliceSummary(BaseModel):
    method: str
    query_type: str
    total_questions: int
    supported_questions: int
    hit_at_k: float
    top_1_accuracy: float


def evaluate_retrieval_result(
    question: EvalQuestion,
    method: str,
    retrieved_chunks: list[RetrievedChunk],
) -> RetrievalEvalResult:
    retrieved_sections = [chunk.section_title for chunk in retrieved_chunks]
    retrieved_doc_sections = [
        f"{chunk.doc_id}::{chunk.section_title}" for chunk in retrieved_chunks
    ]

    top_1_chunk = retrieved_chunks[0] if retrieved_chunks else None
    top_1_doc_id = top_1_chunk.doc_id if top_1_chunk else None
    top_1_section = top_1_chunk.section_title if top_1_chunk else None
    top_1_doc_section = (
        f"{top_1_doc_id}::{top_1_section}" if top_1_chunk else None
    )

    accepted_doc_sections = question.get_accepted_doc_sections()

    if not question.supported:
        return RetrievalEvalResult(
            question_id=question.question_id,
            query=question.query,
            method=method,
            query_type=question.query_type,
            supported=question.supported,
            expected_doc_id=question.expected_doc_id,
            expected_section=question.expected_section,
            accepted_doc_sections=accepted_doc_sections,
            top_1_doc_id=top_1_doc_id,
            top_1_section=top_1_section,
            hit_at_k=False,
            top_1_match=False,
            retrieved_sections=retrieved_sections,
            retrieved_doc_sections=retrieved_doc_sections,
        )

    hit_at_k = any(
        accepted in retrieved_doc_sections for accepted in accepted_doc_sections
    )
    top_1_match = top_1_doc_section in accepted_doc_sections

    return RetrievalEvalResult(
        question_id=question.question_id,
        query=question.query,
        method=method,
        query_type=question.query_type,
        supported=question.supported,
        expected_doc_id=question.expected_doc_id,
        expected_section=question.expected_section,
        accepted_doc_sections=accepted_doc_sections,
        top_1_doc_id=top_1_doc_id,
        top_1_section=top_1_section,
        hit_at_k=hit_at_k,
        top_1_match=top_1_match,
        retrieved_sections=retrieved_sections,
        retrieved_doc_sections=retrieved_doc_sections,
    )


def summarize_retrieval_results(
    method: str,
    results: list[RetrievalEvalResult],
) -> RetrievalEvalSummary:
    supported_results = [result for result in results if result.supported]

    if not supported_results:
        return RetrievalEvalSummary(
            method=method,
            total_questions=len(results),
            supported_questions=0,
            hit_at_k=0.0,
            top_1_accuracy=0.0,
        )

    hit_at_k = sum(result.hit_at_k for result in supported_results) / len(supported_results)
    top_1_accuracy = sum(result.top_1_match for result in supported_results) / len(supported_results)

    return RetrievalEvalSummary(
        method=method,
        total_questions=len(results),
        supported_questions=len(supported_results),
        hit_at_k=hit_at_k,
        top_1_accuracy=top_1_accuracy,
    )


def summarize_by_query_type(
    method: str,
    results: list[RetrievalEvalResult],
) -> list[RetrievalEvalSliceSummary]:
    grouped: dict[str, list[RetrievalEvalResult]] = defaultdict(list)

    for result in results:
        grouped[result.query_type].append(result)

    summaries: list[RetrievalEvalSliceSummary] = []

    for query_type, group_results in sorted(grouped.items()):
        supported_results = [result for result in group_results if result.supported]

        if not supported_results:
            summaries.append(
                RetrievalEvalSliceSummary(
                    method=method,
                    query_type=query_type,
                    total_questions=len(group_results),
                    supported_questions=0,
                    hit_at_k=0.0,
                    top_1_accuracy=0.0,
                )
            )
            continue

        hit_at_k = sum(result.hit_at_k for result in supported_results) / len(
            supported_results
        )
        top_1_accuracy = sum(result.top_1_match for result in supported_results) / len(
            supported_results
        )

        summaries.append(
            RetrievalEvalSliceSummary(
                method=method,
                query_type=query_type,
                total_questions=len(group_results),
                supported_questions=len(supported_results),
                hit_at_k=hit_at_k,
                top_1_accuracy=top_1_accuracy,
            )
        )

    return summaries