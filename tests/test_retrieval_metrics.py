from src.core.models import RetrievedChunk
from src.evaluation.dataset import EvalQuestion
from src.evaluation.retrieval_metrics import evaluate_retrieval_result


def make_chunk(
    doc_id: str,
    section_title: str,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}_{section_title}_{rank}",
        doc_id=doc_id,
        title="Test Doc",
        source_url="https://example.com",
        section_title=section_title,
        section_path=section_title,
        content="test content",
        retrieval_score=1.0,
        retrieval_method="dense",
        rank=rank,
    )


def test_retrieval_hit_when_accepted_doc_section_in_top_k() -> None:
    question = EvalQuestion(
        question_id="q001",
        query="What is a Service?",
        query_type="conceptual",
        expected_section="Service",
        expected_doc_id="k8s_service",
        accepted_doc_sections=["k8s_service::Service"],
        supported=True,
        reference_notes="Service definition.",
    )

    chunks = [
        make_chunk("k8s_configmap", "ConfigMaps", rank=1),
        make_chunk("k8s_service", "Service", rank=2),
    ]

    result = evaluate_retrieval_result(question, "dense", chunks)

    assert result.hit_at_k is True
    assert result.top_1_match is False


def test_retrieval_top_1_match_when_first_chunk_is_accepted() -> None:
    question = EvalQuestion(
        question_id="q001",
        query="What is a Service?",
        query_type="conceptual",
        expected_section="Service",
        expected_doc_id="k8s_service",
        accepted_doc_sections=["k8s_service::Service"],
        supported=True,
        reference_notes="Service definition.",
    )

    chunks = [make_chunk("k8s_service", "Service", rank=1)]

    result = evaluate_retrieval_result(question, "dense", chunks)

    assert result.hit_at_k is True
    assert result.top_1_match is True


def test_retrieval_does_not_match_same_section_in_wrong_document() -> None:
    question = EvalQuestion(
        question_id="q001",
        query="What should I consider before using Secrets?",
        query_type="constraint",
        expected_section="Caution:",
        expected_doc_id="k8s_secret",
        accepted_doc_sections=["k8s_secret::Caution:"],
        supported=True,
        reference_notes="Secret caution.",
    )

    chunks = [make_chunk("k8s_configmap", "Caution:", rank=1)]

    result = evaluate_retrieval_result(question, "dense", chunks)

    assert result.hit_at_k is False
    assert result.top_1_match is False