from pathlib import Path

from src.core.models import DocumentRecord
from src.ingestion.cleaner import normalize_whitespace


def parse_markdown_document(
    file_path: str,
    doc_id: str,
    source_url: str,
    category: str,
) -> DocumentRecord:
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")
    cleaned_text = normalize_whitespace(raw_text)

    lines = cleaned_text.splitlines()
    title = path.stem
    section_path = path.stem

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            section_path = title
            break

    return DocumentRecord(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        category=category,
        section_path=section_path,
        parent_section=None,
        version=None,
        last_updated=None,
        raw_text=cleaned_text,
    )