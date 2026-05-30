import re

from src.core.models import RetrievedChunk


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "by",
    "how",
    "do",
    "i",
    "should",
    "when",
    "what",
    "why",
    "does",
    "did",
}

def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def _query_content_overlap(query: str, chunks: list[RetrievedChunk]) -> float:
    query_tokens = _content_tokens(query)

    if not query_tokens:
        return 0.0

    evidence_text = " ".join(
        f"{chunk.section_title} {chunk.content}" for chunk in chunks[:3]
    )
    evidence_tokens = _content_tokens(evidence_text)

    if not evidence_tokens:
        return 0.0

    overlap = query_tokens.intersection(evidence_tokens)
    return len(overlap) / len(query_tokens)


def should_abstain(query: str, chunks: list[RetrievedChunk]) -> bool:
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
        
        lexical_overlap = _query_content_overlap(query, chunks)
        
        # Hybrid needs both a minimally strong fused score and some lexical evidence.
        if top_chunk.retrieval_score < 0.025:
            return True

        if lexical_overlap < 0.25:
            return True
        
        return False

    return False