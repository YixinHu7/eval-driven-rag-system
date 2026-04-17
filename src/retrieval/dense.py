from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.embedding.embedder import LocalEmbedder
from src.storage.schema import ChunkORM, DocumentORM


class DenseRetriever:
    def __init__(self) -> None:
        self.embedder = LocalEmbedder()

    def retrieve(self, session: Session, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or settings.retrieval.top_k

        query_embedding = self.embedder.embed_texts([query])[0]
        distance_expr = ChunkORM.embedding.cosine_distance(query_embedding).label("distance")

        stmt = (
            select(ChunkORM, DocumentORM, distance_expr)
            .join(DocumentORM, ChunkORM.doc_id == DocumentORM.doc_id)
            .where(ChunkORM.embedding.is_not(None))
            .order_by(distance_expr.asc())
            .limit(k)
        )

        rows = session.execute(stmt).all()

        results: list[RetrievedChunk] = []

        for rank, (chunk, document, distance) in enumerate(rows, start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=document.title,
                    source_url=document.source_url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    content=chunk.chunk_text,
                    retrieval_score=float(distance),
                    retrieval_method="dense",
                    rank=rank,
                )
            )

        return results