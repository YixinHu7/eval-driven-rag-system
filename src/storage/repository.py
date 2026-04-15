from typing import Iterable, Optional

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from src.core.models import ChunkRecord, DocumentRecord
from src.storage.schema import ChunkORM, DocumentORM


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_document(self, record: DocumentRecord) -> DocumentORM:
        existing = self.session.get(DocumentORM, record.doc_id)
        if existing is not None:
            raise ValueError(f"Document with doc_id='{record.doc_id}' already exists.")

        document = DocumentORM(
            doc_id=record.doc_id,
            title=record.title,
            source_url=record.source_url,
            category=record.category,
            section_path=record.section_path,
            parent_section=record.parent_section,
            version=record.version,
            last_updated=record.last_updated,
            raw_text=record.raw_text,
        )
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentORM]:
        return self.session.get(DocumentORM, doc_id)

    def count_documents(self) -> int:
        stmt = select(func.count()).select_from(DocumentORM)
        return self.session.scalar(stmt) or 0


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def insert_chunks(self, records: Iterable[ChunkRecord]) -> list[ChunkORM]:
        chunk_orms: list[ChunkORM] = []

        for record in records:
            chunk = ChunkORM(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                chunk_index=record.chunk_index,
                chunk_text=record.chunk_text,
                section_title=record.section_title,
                section_path=record.section_path,
                token_count=record.token_count,
                embedding=record.embedding,
            )
            chunk_orms.append(chunk)

        self.session.add_all(chunk_orms)
        self.session.commit()

        for chunk in chunk_orms:
            self.session.refresh(chunk)

        return chunk_orms

    def list_chunks_for_doc(self, doc_id: str) -> list[ChunkORM]:
        stmt = (
            select(ChunkORM)
            .where(ChunkORM.doc_id == doc_id)
            .order_by(ChunkORM.chunk_index.asc())
        )
        return list(self.session.scalars(stmt).all())

    def count_chunks(self) -> int:
        stmt = select(func.count()).select_from(ChunkORM)
        return self.session.scalar(stmt) or 0