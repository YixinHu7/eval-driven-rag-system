from src.core.models import RetrievedChunk
from src.generation.abstention import should_abstain


def make_chunk(
    method: str,
    score: float,
    content: str = "A Kubernetes Service exposes applications running on Pods.",
    section_title: str = "Service",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk_1",
        doc_id="k8s_service",
        title="Services in Kubernetes",
        source_url="https://example.com",
        section_title=section_title,
        section_path=section_title,
        content=content,
        retrieval_score=score,
        retrieval_method=method,
        rank=1,
    )


def test_abstain_when_no_chunks() -> None:
    assert should_abstain("What is a Service?", []) is True


def test_dense_abstains_when_distance_is_too_high() -> None:
    chunks = [make_chunk("dense", 0.9)]

    assert should_abstain("What is a Service?", chunks) is True


def test_dense_does_not_abstain_when_distance_is_low() -> None:
    chunks = [make_chunk("dense", 0.2)]

    assert should_abstain("What is a Service?", chunks) is False


def test_hybrid_abstains_when_lexical_overlap_is_low() -> None:
    chunks = [
        make_chunk(
            "hybrid_rrf",
            0.04,
            content="A Kubernetes Service exposes applications running on Pods.",
            section_title="Service",
        )
    ]

    assert should_abstain("How do I bake a chocolate cake?", chunks) is True