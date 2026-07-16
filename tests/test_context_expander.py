from src.core.models import RetrievedChunk
from src.retrieval.context_expander import rerank_context_chunks


def make_chunk(
    chunk_id: str,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="k8s_probes",
        title="Configure Probes",
        source_url="https://example.com",
        section_title="Readiness probes",
        section_path="Configure Probes > Readiness probes",
        content="Readiness probe content.",
        retrieval_score=0.1,
        retrieval_method="hybrid_rrf",
        rank=rank,
    )


def test_rerank_context_chunks_assigns_sequential_ranks() -> None:
    chunks = [
        make_chunk("chunk_a", rank=10),
        make_chunk("chunk_b", rank=20),
        make_chunk("chunk_c", rank=30),
    ]

    reranked = rerank_context_chunks(chunks)

    assert [chunk.rank for chunk in reranked] == [1, 2, 3]
    assert [chunk.chunk_id for chunk in reranked] == [
        "chunk_a",
        "chunk_b",
        "chunk_c",
    ]