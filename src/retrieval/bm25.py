from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.storage.schema import ChunkORM, DocumentORM


def simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self) -> None:
        self.default_top_k = settings.retrieval.top_k

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.default_top_k

        stmt = (
            select(ChunkORM, DocumentORM)
            .join(DocumentORM, ChunkORM.doc_id == DocumentORM.doc_id)
            .order_by(ChunkORM.doc_id.asc(), ChunkORM.chunk_index.asc())
        )

        rows = session.execute(stmt).all()

        if not rows:
            return []

        corpus_tokens: list[list[str]] = []
        joined_rows: list[tuple[ChunkORM, DocumentORM]] = []

        for chunk, document in rows:
            search_text = f"{chunk.section_title} {chunk.chunk_text}"
            corpus_tokens.append(simple_tokenize(search_text))
            joined_rows.append((chunk, document))

        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = simple_tokenize(query)
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results: list[RetrievedChunk] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            chunk, document = joined_rows[idx]
            score = float(scores[idx])

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=document.title,
                    source_url=document.source_url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    content=chunk.chunk_text,
                    retrieval_score=score,
                    retrieval_method="bm25",
                    rank=rank,
                )
            )

        return results