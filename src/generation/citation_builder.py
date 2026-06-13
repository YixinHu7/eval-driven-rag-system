import re

from src.core.models import Citation, RetrievedChunk


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []

    for idx, chunk in enumerate(chunks, start=1):
        snippet = chunk.content[:240].strip()
        if len(chunk.content) > 240:
            snippet += "..."

        citations.append(
            Citation(
                citation_id=idx,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_url=chunk.source_url,
                section_title=chunk.section_title,
                snippet=snippet,
            )
        )

    return citations


def extract_citation_ids(answer: str) -> set[int]:
    matches = re.findall(r"\[(\d+)\]", answer)
    return {int(match) for match in matches}


def build_used_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    all_citations = build_citations(chunks)
    used_ids = extract_citation_ids(answer)

    return [
        citation
        for citation in all_citations
        if citation.citation_id in used_ids
    ]