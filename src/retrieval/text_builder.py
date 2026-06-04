from src.storage.schema import ChunkORM


def build_chunk_search_text(chunk: ChunkORM) -> str:
    document = chunk.document

    parts = [
        f"Document title: {document.title}",
        f"Category: {document.category}",
        f"Section path: {chunk.section_path}",
        f"Section title: {chunk.section_title}",
        f"Content: {chunk.chunk_text}",
    ]

    return "\n".join(part for part in parts if part)