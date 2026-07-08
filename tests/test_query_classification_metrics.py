from src.evaluation.dataset import EvalQuestion
from src.evaluation.query_classification_metrics import (
    evaluate_query_classification,
    expected_domain_for_question,
    summarize_query_classification_results,
)
from src.routing.query_classifier import QueryClassification


def make_question(
    query_type: str,
    supported: bool,
) -> EvalQuestion:
    return EvalQuestion(
        question_id="q001",
        query="Test query",
        query_type=query_type,
        expected_section="Service" if supported else None,
        expected_doc_id="k8s_service" if supported else None,
        supported=supported,
        reference_notes="notes",
    )


def make_classification(
    query_type: str,
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        confidence=0.8,
        matched_terms=["service"] if query_type == "in_domain" else [],
    )


def test_supported_question_expected_in_domain() -> None:
    question = make_question(
        query_type="conceptual",
        supported=True,
    )

    assert expected_domain_for_question(question) == "in_domain"


def test_out_of_domain_question_expected_out_of_domain() -> None:
    question = make_question(
        query_type="out_of_domain",
        supported=False,
    )

    assert expected_domain_for_question(question) == "out_of_domain"


def test_evaluate_correct_classification() -> None:
    question = make_question(
        query_type="conceptual",
        supported=True,
    )
    classification = make_classification("in_domain")

    result = evaluate_query_classification(
        question=question,
        classification=classification,
    )

    assert result.expected_domain == "in_domain"
    assert result.predicted_domain == "in_domain"
    assert result.correct is True


def test_evaluate_incorrect_classification() -> None:
    question = make_question(
        query_type="out_of_domain",
        supported=False,
    )
    classification = make_classification("in_domain")

    result = evaluate_query_classification(
        question=question,
        classification=classification,
    )

    assert result.expected_domain == "out_of_domain"
    assert result.predicted_domain == "in_domain"
    assert result.correct is False


def test_summarize_query_classification_results() -> None:
    supported_question = make_question(
        query_type="conceptual",
        supported=True,
    )
    unsupported_question = make_question(
        query_type="out_of_domain",
        supported=False,
    )

    results = [
        evaluate_query_classification(
            supported_question,
            make_classification("in_domain"),
        ),
        evaluate_query_classification(
            unsupported_question,
            make_classification("in_domain"),
        ),
    ]

    summary = summarize_query_classification_results(results)

    assert summary.total_questions == 2
    assert summary.correct_count == 1
    assert summary.incorrect_count == 1
    assert summary.accuracy == 0.5
    assert summary.false_in_domain_count == 1
    assert summary.false_out_of_domain_count == 0
    assert summary.confusion_counts == {
        "in_domain->in_domain": 1,
        "out_of_domain->in_domain": 1,
    }