from src.core.models import AnswerResponse, Citation
from src.evaluation.citation_metrics import evaluate_citations


def make_citation(citation_id: int) -> Citation:
    return Citation(
        citation_id=citation_id,
        chunk_id=f"chunk_{citation_id}",
        doc_id="k8s_probes",
        title="Configure Probes",
        source_url="https://example.com",
        section_title="Readiness probes",
        snippet="Supporting evidence.",
    )


def test_citation_alignment_is_correct() -> None:
    response = AnswerResponse(
        answer="Readiness probes control service traffic [1][3].",
        citations=[make_citation(1), make_citation(3)],
        query_type=None,
        retrieval_strategy="hybrid",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_citations(response)

    assert result.citation_ids_valid is True
    assert result.citation_alignment_correct is True
    assert result.citation_utilization == 1.0


def test_detects_missing_returned_citation() -> None:
    response = AnswerResponse(
        answer="The answer is supported by [1][3].",
        citations=[make_citation(1)],
        query_type=None,
        retrieval_strategy="hybrid",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_citations(response)

    assert result.missing_returned_citation_ids == [3]
    assert result.citation_ids_valid is False
    assert result.citation_alignment_correct is False


def test_detects_unreferenced_returned_citation() -> None:
    response = AnswerResponse(
        answer="The answer is supported by [1].",
        citations=[make_citation(1), make_citation(2)],
        query_type=None,
        retrieval_strategy="hybrid",
        confidence=0.8,
        abstained=False,
        retrieved_chunks=[],
    )

    result = evaluate_citations(response)

    assert result.unreferenced_returned_citation_ids == [2]
    assert result.citation_alignment_correct is False
    assert result.citation_utilization == 0.5


def test_abstained_response_with_no_citations_is_aligned() -> None:
    response = AnswerResponse(
        answer="The answer is not supported by the retrieved context.",
        citations=[],
        query_type=None,
        retrieval_strategy="hybrid",
        confidence=0.0,
        abstained=True,
        retrieved_chunks=[],
    )

    result = evaluate_citations(response)

    assert result.citation_ids_valid is True
    assert result.citation_alignment_correct is True
    assert result.citation_utilization == 0.0