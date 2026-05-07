from src.core.models import RetrievedChunk


def should_abstain(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return True

    top_chunk = chunks[0]

    if top_chunk.retrieval_method == "dense":
        # Dense retrieval uses cosine distance.
        # Lower is better. This threshold is intentionally conservative
        # and should be tuned later with evaluation data.
        return top_chunk.retrieval_score > 0.45

    if top_chunk.retrieval_method == "bm25":
        # BM25 score is unbounded and corpus-dependent.
        # This is a temporary heuristic for small corpora.
        return top_chunk.retrieval_score < 0.1

    if top_chunk.retrieval_method == "hybrid_rrf":
        # RRF score is small by design.
        # For rrf_k=60, one rank-1 hit contributes around 0.0164.
        return top_chunk.retrieval_score < 0.025

    return False