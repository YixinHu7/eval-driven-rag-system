from collections import Counter
from typing import Literal

from pydantic import BaseModel

from src.evaluation.dataset import EvalQuestion
from src.routing.query_classifier import QueryClassification

ExpectedQueryDomain = Literal["in_domain", "out_of_domain"]


class QueryClassificationEvalResult(BaseModel):
    question_id: str
    query: str
    query_type: str
    supported: bool

    expected_domain: ExpectedQueryDomain
    predicted_domain: str
    confidence: float
    matched_terms: list[str]

    correct: bool


class QueryClassificationEvalSummary(BaseModel):
    total_questions: int
    accuracy: float
    correct_count: int
    incorrect_count: int
    false_in_domain_count: int
    false_out_of_domain_count: int
    ambiguous_count: int
    confusion_counts: dict[str, int]


def expected_domain_for_question(
    question: EvalQuestion,
) -> ExpectedQueryDomain:
    if question.query_type == "out_of_domain" or not question.supported:
        return "out_of_domain"

    return "in_domain"


def evaluate_query_classification(
    question: EvalQuestion,
    classification: QueryClassification,
) -> QueryClassificationEvalResult:
    expected_domain = expected_domain_for_question(question)
    predicted_domain = classification.query_type

    return QueryClassificationEvalResult(
        question_id=question.question_id,
        query=question.query,
        query_type=question.query_type,
        supported=question.supported,
        expected_domain=expected_domain,
        predicted_domain=predicted_domain,
        confidence=classification.confidence,
        matched_terms=classification.matched_terms,
        correct=expected_domain == predicted_domain,
    )


def summarize_query_classification_results(
    results: list[QueryClassificationEvalResult],
) -> QueryClassificationEvalSummary:
    if not results:
        return QueryClassificationEvalSummary(
            total_questions=0,
            accuracy=0.0,
            correct_count=0,
            incorrect_count=0,
            false_in_domain_count=0,
            false_out_of_domain_count=0,
            ambiguous_count=0,
            confusion_counts={},
        )

    total = len(results)
    correct_count = sum(result.correct for result in results)
    incorrect_count = total - correct_count

    false_in_domain_count = sum(
        1
        for result in results
        if result.expected_domain == "out_of_domain"
        and result.predicted_domain == "in_domain"
    )

    false_out_of_domain_count = sum(
        1
        for result in results
        if result.expected_domain == "in_domain"
        and result.predicted_domain == "out_of_domain"
    )

    ambiguous_count = sum(
        1 for result in results if result.predicted_domain == "ambiguous"
    )

    confusion_counter = Counter(
        f"{result.expected_domain}->{result.predicted_domain}" for result in results
    )

    return QueryClassificationEvalSummary(
        total_questions=total,
        accuracy=correct_count / total,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        false_in_domain_count=false_in_domain_count,
        false_out_of_domain_count=false_out_of_domain_count,
        ambiguous_count=ambiguous_count,
        confusion_counts=dict(sorted(confusion_counter.items())),
    )
