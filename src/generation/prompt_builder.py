from src.core.models import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        block = (
            f"[{idx}] {chunk.title}\n"
            f"Section: {chunk.section_title}\n"
            f"Source: {chunk.source_url}\n"
            f"Content: {chunk.content}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_grounded_qa_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)

    return (
        "You are answering a question using only the provided technical documentation context.\n"
        "If the context does not contain enough information, say that the answer is not supported by the retrieved context.\n"
        "Cite supporting sources using bracketed citation numbers like [1], [2].\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Answer:"
    )