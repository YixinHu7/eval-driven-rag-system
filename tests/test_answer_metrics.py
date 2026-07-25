from src.core.models import AnswerResponse, Citation
from src.evaluation.answer_metrics import (
    evaluate_answer_result,
    evaluate_required_evidence,
)
from src.evaluation.dataset import EvalQuestion


def make_question(
    supported: bool,
    required_evidence_sections: list[str] | None = None,
) -> EvalQuestion:
    return EvalQuestion(
        question_id="q001",
        query="Test query",
        query_type="conceptual",
        expected_section="Service" if supported else None,
        expected_doc_id="k8s_service" if supported else None,
        required_evidence_sections=required_evidence_sections or [],
        supported=supported,
        reference_notes="notes",
    )


def make_citation(
    citation_id: int = 1,
    section_title: str = "Service",
) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=f"chunk_{citation_id}",
        doc_id="k8s_service",
        title=section_title,
        source_url="https://example.com",
        section_title=section_title,
        snippet=f"{section_title} content",
    )


def test_supported_answer_should_not_abstain_and_should_have_citation() -> None:
    question = make_question(supported=True)
    response = AnswerResponse(
        answer="Supported answer. [1]",
        citations=[make_citation()],
        retrieval_strategy="dense",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_answer_result(question, "dense", response)

    assert result.abstention_correct is True
    assert result.citation_presence_correct is True
    assert result.required_evidence_passed is True
    assert result.passed is True


def test_unsupported_answer_should_abstain_and_have_no_citations() -> None:
    question = make_question(supported=False)
    response = AnswerResponse(
        answer="I cannot answer.",
        citations=[],
        retrieval_strategy="dense",
        confidence=0.0,
        abstained=True,
        retrieved_chunks=[],
    )

    result = evaluate_answer_result(question, "dense", response)

    assert result.abstention_correct is True
    assert result.citation_presence_correct is True
    assert result.required_evidence_passed is True
    assert result.passed is True


def test_no_required_evidence_passes_by_default() -> None:
    coverage, passed = evaluate_required_evidence(
        required_sections=[],
        cited_sections=[],
    )

    assert coverage == 1.0
    assert passed is True


def test_full_required_evidence_coverage_passes() -> None:
    coverage, passed = evaluate_required_evidence(
        required_sections=[
            "k8s_service::Service",
            "k8s_service::Service Types",
        ],
        cited_sections=[
            "k8s_service::Service",
            "k8s_service::Service Types",
        ],
    )

    assert coverage == 1.0
    assert passed is True


def test_partial_required_evidence_coverage_fails() -> None:
    coverage, passed = evaluate_required_evidence(
        required_sections=[
            "k8s_service::Service",
            "k8s_service::Service Types",
        ],
        cited_sections=[
            "k8s_service::Service",
        ],
    )

    assert coverage == 0.5
    assert passed is False


def test_answer_passes_when_all_required_sections_are_referenced() -> None:
    question = make_question(
        supported=True,
        required_evidence_sections=[
            "k8s_service::Service",
            "k8s_service::Service Types",
        ],
    )
    response = AnswerResponse(
        answer=(
            "A Service exposes an application, and different "
            "Service types control how it is exposed. [1] [2]"
        ),
        citations=[
            make_citation(
                citation_id=1,
                section_title="Service",
            ),
            make_citation(
                citation_id=2,
                section_title="Service Types",
            ),
        ],
        retrieval_strategy="dense",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_answer_result(question, "dense", response)

    assert result.cited_evidence_sections == [
        "k8s_service::Service",
        "k8s_service::Service Types",
    ]
    assert result.required_evidence_coverage == 1.0
    assert result.required_evidence_passed is True
    assert result.passed is True


def test_answer_fails_when_required_section_is_not_referenced() -> None:
    question = make_question(
        supported=True,
        required_evidence_sections=[
            "k8s_service::Service",
            "k8s_service::Service Types",
        ],
    )
    response = AnswerResponse(
        answer="A Service exposes an application. [1]",
        citations=[
            make_citation(
                citation_id=1,
                section_title="Service",
            ),
            make_citation(
                citation_id=2,
                section_title="Service Types",
            ),
        ],
        retrieval_strategy="dense",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_answer_result(question, "dense", response)

    assert result.citation_presence_correct is True
    assert result.cited_evidence_sections == [
        "k8s_service::Service",
    ]
    assert result.required_evidence_coverage == 0.5
    assert result.required_evidence_passed is False
    assert result.passed is False
