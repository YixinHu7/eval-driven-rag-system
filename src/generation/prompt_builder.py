from src.core.models import RetrievedChunk

def build_context(chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        block = (
            f"[{idx}]\n"
            f"Document: {chunk.title}\n"
            f"Section: {chunk.section_title}\n"
            f"Section path: {chunk.section_path}\n"
            f"Source: {chunk.source_url}\n"
            f"Content:\n{chunk.content}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_grounded_qa_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    ) -> str:
    
    context = build_context(chunks)

    return (
        "Answer the question using only the retrieved technical documentation "
        "provided below.\n\n"
        "Requirements:\n"
        "1. Do not use outside knowledge or make unsupported assumptions.\n"
        "2. Cite each factual statement using one or more citation numbers, "
        "such as [1] or [2][3].\n"
        "3. Use only citation numbers that appear in the retrieved context.\n"
        "4. Cite only context blocks that directly support the statement.\n"
        "5. Do not include a references section or repeat the source URLs.\n"
        "6. If the retrieved context does not contain enough information, say: "
        "\"The answer is not supported by the retrieved context.\"\n"
        "7. Keep the answer clear, concise, and technically precise.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Grounded Answer:"
    )