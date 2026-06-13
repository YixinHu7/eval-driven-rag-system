from src.core.models import RetrievedChunk
from src.generation.citation_builder import (
    build_used_citations,
    extract_citation_ids,
)


def make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{index}",
        doc_id="k8s_probes",
        title="Configure Probes",
        source_url="https://example.com",
        section_title=f"Section {index}",
        section_path=f"Section {index}",
        content=f"Evidence content {index}",
        retrieval_score=0.1,
        retrieval_method="hybrid_rrf",
        rank=index,
    )


def test_extract_citation_ids() -> None:
    answer = "Readiness probes control traffic [1] and can use TCP checks [4][5]."

    assert extract_citation_ids(answer) == {1, 4, 5}


def test_build_used_citations_filters_unused_chunks() -> None:
    chunks = [make_chunk(index) for index in range(1, 6)]
    answer = "The answer is supported by [1][4][5]."

    citations = build_used_citations(answer, chunks)

    assert [citation.citation_id for citation in citations] == [1, 4, 5]


def test_build_used_citations_ignores_invalid_ids() -> None:
    chunks = [make_chunk(1), make_chunk(2)]
    answer = "This claim cites unavailable evidence [9]."

    citations = build_used_citations(answer, chunks)

    assert citations == []