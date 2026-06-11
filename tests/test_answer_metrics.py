from src.core.models import AnswerResponse, Citation
from src.evaluation.answer_metrics import evaluate_answer_result
from src.evaluation.dataset import EvalQuestion


def make_question(supported: bool) -> EvalQuestion:
    return EvalQuestion(
        question_id="q001",
        query="Test query",
        query_type="conceptual",
        expected_section="Service" if supported else None,
        expected_doc_id="k8s_service" if supported else None,
        supported=supported,
        reference_notes="notes",
    )


def make_citation() -> Citation:
    return Citation(
        citation_id=1,
        chunk_id="chunk_1",
        doc_id="k8s_service",
        title="Service",
        source_url="https://example.com",
        section_title="Service",
        snippet="Service content",
    )


def test_supported_answer_should_not_abstain_and_should_have_citation() -> None:
    question = make_question(supported=True)
    response = AnswerResponse(
        answer="Supported answer.",
        citations=[make_citation()],
        retrieval_strategy="dense",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_answer_result(question, "dense", response)

    assert result.abstention_correct is True
    assert result.citation_presence_correct is True
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
    assert result.passed is True