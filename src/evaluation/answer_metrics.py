from pydantic import BaseModel

from src.core.models import AnswerResponse
from src.evaluation.dataset import EvalQuestion


class AnswerEvalResult(BaseModel):
    question_id: str
    query: str
    method: str
    query_type: str
    supported: bool

    abstained: bool
    expected_abstain: bool
    abstention_correct: bool

    citation_count: int
    citation_presence_correct: bool

    passed: bool


class AnswerEvalSummary(BaseModel):
    method: str
    total_questions: int
    abstention_accuracy: float
    citation_presence_accuracy: float
    pass_rate: float


class AnswerEvalSliceSummary(BaseModel):
    method: str
    query_type: str
    total_questions: int
    abstention_accuracy: float
    citation_presence_accuracy: float
    pass_rate: float


def evaluate_answer_result(
    question: EvalQuestion,
    method: str,
    response: AnswerResponse,
) -> AnswerEvalResult:
    expected_abstain = not question.supported
    abstention_correct = response.abstained == expected_abstain

    citation_count = len(response.citations)

    if question.supported:
        citation_presence_correct = citation_count > 0
    else:
        citation_presence_correct = citation_count == 0

    passed = abstention_correct and citation_presence_correct

    return AnswerEvalResult(
        question_id=question.question_id,
        query=question.query,
        method=method,
        query_type=question.query_type,
        supported=question.supported,
        abstained=response.abstained,
        expected_abstain=expected_abstain,
        abstention_correct=abstention_correct,
        citation_count=citation_count,
        citation_presence_correct=citation_presence_correct,
        passed=passed,
    )


def summarize_answer_results(
    method: str,
    results: list[AnswerEvalResult],
) -> AnswerEvalSummary:
    if not results:
        return AnswerEvalSummary(
            method=method,
            total_questions=0,
            abstention_accuracy=0.0,
            citation_presence_accuracy=0.0,
            pass_rate=0.0,
        )

    total = len(results)

    return AnswerEvalSummary(
        method=method,
        total_questions=total,
        abstention_accuracy=sum(r.abstention_correct for r in results) / total,
        citation_presence_accuracy=sum(r.citation_presence_correct for r in results) / total,
        pass_rate=sum(r.passed for r in results) / total,
    )


def summarize_answer_by_query_type(
    method: str,
    results: list[AnswerEvalResult],
) -> list[AnswerEvalSliceSummary]:
    grouped: dict[str, list[AnswerEvalResult]] = {}

    for result in results:
        grouped.setdefault(result.query_type, []).append(result)

    summaries: list[AnswerEvalSliceSummary] = []

    for query_type, group in sorted(grouped.items()):
        total = len(group)

        summaries.append(
            AnswerEvalSliceSummary(
                method=method,
                query_type=query_type,
                total_questions=total,
                abstention_accuracy=sum(r.abstention_correct for r in group) / total,
                citation_presence_accuracy=sum(r.citation_presence_correct for r in group)
                / total,
                pass_rate=sum(r.passed for r in group) / total,
            )
        )

    return summaries