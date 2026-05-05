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