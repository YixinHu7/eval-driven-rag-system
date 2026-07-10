from src.core.models import RetrievedChunk
from src.generation.context_selector import (
    is_generic_non_evidence_chunk,
    select_context_chunks,
)


def make_chunk(
    section_title: str,
    content: str,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{rank}",
        doc_id="k8s_probes",
        title="Configure Liveness, Readiness and Startup Probes",
        source_url="https://example.com",
        section_title=section_title,
        section_path=section_title,
        content=content,
        retrieval_score=0.1,
        retrieval_method="hybrid_rrf",
        rank=rank,
    )


def test_detects_feedback_chunk_as_generic_non_evidence() -> None:
    chunk = make_chunk(
        section_title="Feedback",
        content="Was this page helpful? Thanks for the feedback.",
        rank=1,
    )

    assert is_generic_non_evidence_chunk(chunk) is True


def test_detects_whats_next_chunk_as_generic_non_evidence() -> None:
    chunk = make_chunk(
        section_title="What's next",
        content="Learn more about Kubernetes probes.",
        rank=1,
    )

    assert is_generic_non_evidence_chunk(chunk) is True


def test_does_not_filter_caution_chunk() -> None:
    chunk = make_chunk(
        section_title="Caution:",
        content="Readiness and liveness probes do not depend on each other.",
        rank=1,
    )

    assert is_generic_non_evidence_chunk(chunk) is False


def test_select_context_chunks_filters_generic_sections() -> None:
    chunks = [
        make_chunk(
            section_title="Define readiness probes",
            content="Readiness probes detect whether a pod can receive traffic.",
            rank=1,
        ),
        make_chunk(
            section_title="What's next",
            content="Learn more about probes.",
            rank=2,
        ),
        make_chunk(
            section_title="Feedback",
            content="Was this page helpful?",
            rank=3,
        ),
        make_chunk(
            section_title="Caution:",
            content="Readiness and liveness probes do not depend on each other.",
            rank=4,
        ),
    ]

    selected = select_context_chunks(chunks)

    assert [chunk.section_title for chunk in selected] == [
        "Define readiness probes",
        "Caution:",
    ]


def test_select_context_chunks_falls_back_when_filtering_removes_everything() -> None:
    chunks = [
        make_chunk(
            section_title="Feedback",
            content="Was this page helpful?",
            rank=1,
        )
    ]

    selected = select_context_chunks(chunks)

    assert selected == chunks