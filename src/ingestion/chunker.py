from src.core.models import ChunkRecord


def split_markdown_by_headings(doc_id: str, text: str) -> list[ChunkRecord]:
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = stripped.lstrip("#").strip()
            current_lines = []
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[ChunkRecord] = []

    for idx, (heading, lines) in enumerate(sections):
        chunk_text = " ".join(lines).strip()
        if not chunk_text:
            continue

        token_count = len(chunk_text.split())

        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}_chunk_{idx:03d}",
                doc_id=doc_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                section_title=heading,
                section_path=heading,
                token_count=token_count,
                embedding=None,
            )
        )

    return chunks