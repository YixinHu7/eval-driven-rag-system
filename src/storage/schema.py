from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from src.core.config import load_settings
from src.storage.db import Base


settings = load_settings()


class DocumentORM(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    parent_section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_updated: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    chunks: Mapped[list["ChunkORM"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class ChunkORM(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    section_title: Mapped[str] = mapped_column(String(256), nullable=False)
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding.embedding_dimensions),
        nullable=True,
    )

    document: Mapped["DocumentORM"] = relationship(back_populates="chunks")