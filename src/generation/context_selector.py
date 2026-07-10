from src.core.models import RetrievedChunk


GENERIC_NON_EVIDENCE_SECTIONS = {
    "feedback",
    "what's next",
}


GENERIC_NON_EVIDENCE_PHRASES = {
    "was this page helpful",
    "thanks for the feedback",
    "open an issue in the github repository",
    "last modified",
}


def is_generic_non_evidence_chunk(chunk: RetrievedChunk) -> bool:
    section_title = chunk.section_title.strip().lower()
    content = chunk.content.strip().lower()

    if section_title in GENERIC_NON_EVIDENCE_SECTIONS:
        return True

    return any(
        phrase in content
        for phrase in GENERIC_NON_EVIDENCE_PHRASES
    )


def select_context_chunks(
    chunks: list[RetrievedChunk],
    max_chunks: int = 5,
    min_chunks_after_filtering: int = 1,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    filtered_chunks = [
        chunk
        for chunk in chunks
        if not is_generic_non_evidence_chunk(chunk)
    ]

    if len(filtered_chunks) >= min_chunks_after_filtering:
        return filtered_chunks[:max_chunks]

    return chunks[:max_chunks]