from pydantic import BaseModel

from src.core.models import RetrievedChunk
from src.evaluation.dataset import EvalQuestion


class RetrievalEvalResult(BaseModel):
    question_id: str
    query: str
    method: str
    query_type: str
    supported: bool
    expected_section: str | None
    top_1_section: str | None
    hit_at_k: bool
    top_1_match: bool
    retrieved_sections: list[str]


class RetrievalEvalSummary(BaseModel):
    method: str
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
    top_1_section = retrieved_sections[0] if retrieved_sections else None

    if not question.supported:
        return RetrievalEvalResult(
            question_id=question.question_id,
            query=question.query,
            method=method,
            query_type=question.query_type,
            supported=question.supported,
            expected_section=question.expected_section,
            top_1_section=top_1_section,
            hit_at_k=False,
            top_1_match=False,
            retrieved_sections=retrieved_sections,
        )

    hit_at_k = question.expected_section in retrieved_sections
    top_1_match = top_1_section == question.expected_section

    return RetrievalEvalResult(
        question_id=question.question_id,
        query=question.query,
        method=method,
        query_type=question.query_type,
        supported=question.supported,
        expected_section=question.expected_section,
        top_1_section=top_1_section,
        hit_at_k=hit_at_k,
        top_1_match=top_1_match,
        retrieved_sections=retrieved_sections,
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