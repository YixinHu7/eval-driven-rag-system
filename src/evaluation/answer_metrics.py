from pydantic import BaseModel

from src.core.models import AnswerResponse
from src.evaluation.dataset import EvalQuestion
from src.generation.citation_builder import extract_citation_ids


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

    required_evidence_sections: list[str]
    cited_evidence_sections: list[str]
    required_evidence_coverage: float
    required_evidence_passed: bool

    passed: bool


class AnswerEvalSummary(BaseModel):
    method: str
    total_questions: int
    abstention_accuracy: float
    citation_presence_accuracy: float
    pass_rate: float

    required_evidence_questions: int
    required_evidence_accuracy: float
    average_required_evidence_coverage: float


class AnswerEvalSliceSummary(BaseModel):
    method: str
    query_type: str
    total_questions: int
    abstention_accuracy: float
    citation_presence_accuracy: float
    pass_rate: float

    required_evidence_questions: int
    required_evidence_accuracy: float
    average_required_evidence_coverage: float


def get_cited_evidence_sections(
    response: AnswerResponse,
) -> list[str]:
    referenced_citation_ids = set(extract_citation_ids(response.answer))

    cited_evidence_sections = {
        f"{citation.doc_id}::{citation.section_title}"
        for citation in response.citations
        if citation.citation_id in referenced_citation_ids
    }

    return sorted(cited_evidence_sections)


def evaluate_required_evidence(
    required_sections: list[str],
    cited_sections: list[str],
) -> tuple[float, bool]:
    if not required_sections:
        return 1.0, True

    required_set = set(required_sections)
    cited_set = set(cited_sections)

    matched_sections = required_set & cited_set
    coverage = len(matched_sections) / len(required_set)
    passed = required_set.issubset(cited_set)

    return coverage, passed


def summarize_required_evidence(
    results: list[AnswerEvalResult],
) -> tuple[int, float, float]:
    required_results = [
        result for result in results if result.required_evidence_sections
    ]

    if not required_results:
        return 0, 0.0, 0.0

    total = len(required_results)

    accuracy = (
        sum(result.required_evidence_passed for result in required_results) / total
    )
    average_coverage = (
        sum(result.required_evidence_coverage for result in required_results) / total
    )

    return total, accuracy, average_coverage


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

    cited_evidence_sections = get_cited_evidence_sections(response=response)

    (
        required_evidence_coverage,
        required_evidence_passed,
    ) = evaluate_required_evidence(
        required_sections=question.required_evidence_sections,
        cited_sections=cited_evidence_sections,
    )

    passed = (
        abstention_correct and citation_presence_correct and required_evidence_passed
    )

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
        required_evidence_sections=list(question.required_evidence_sections),
        cited_evidence_sections=cited_evidence_sections,
        required_evidence_coverage=required_evidence_coverage,
        required_evidence_passed=required_evidence_passed,
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
            required_evidence_questions=0,
            required_evidence_accuracy=0.0,
            average_required_evidence_coverage=0.0,
        )

    total = len(results)

    (
        required_evidence_questions,
        required_evidence_accuracy,
        average_required_evidence_coverage,
    ) = summarize_required_evidence(results)

    return AnswerEvalSummary(
        method=method,
        total_questions=total,
        abstention_accuracy=(
            sum(result.abstention_correct for result in results) / total
        ),
        citation_presence_accuracy=(
            sum(result.citation_presence_correct for result in results) / total
        ),
        pass_rate=(sum(result.passed for result in results) / total),
        required_evidence_questions=required_evidence_questions,
        required_evidence_accuracy=required_evidence_accuracy,
        average_required_evidence_coverage=(average_required_evidence_coverage),
    )


def summarize_answer_by_query_type(
    method: str,
    results: list[AnswerEvalResult],
) -> list[AnswerEvalSliceSummary]:
    grouped: dict[str, list[AnswerEvalResult]] = {}

    for result in results:
        grouped.setdefault(
            result.query_type,
            [],
        ).append(result)

    summaries: list[AnswerEvalSliceSummary] = []

    for query_type, group in sorted(grouped.items()):
        total = len(group)

        (
            required_evidence_questions,
            required_evidence_accuracy,
            average_required_evidence_coverage,
        ) = summarize_required_evidence(group)

        summaries.append(
            AnswerEvalSliceSummary(
                method=method,
                query_type=query_type,
                total_questions=total,
                abstention_accuracy=(
                    sum(result.abstention_correct for result in group) / total
                ),
                citation_presence_accuracy=(
                    sum(result.citation_presence_correct for result in group) / total
                ),
                pass_rate=(sum(result.passed for result in group) / total),
                required_evidence_questions=(required_evidence_questions),
                required_evidence_accuracy=(required_evidence_accuracy),
                average_required_evidence_coverage=(average_required_evidence_coverage),
            )
        )

    return summaries