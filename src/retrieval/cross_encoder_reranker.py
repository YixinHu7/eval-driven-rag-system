from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session
from typing import Protocol
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.retrieval.text_builder import build_chunk_search_text
from src.storage.schema import ChunkORM


class BaseRetriever(Protocol):
    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        ...
        

class CrossEncoderReranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(
            settings.reranker.model_name,
            device=settings.reranker.device,
        )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [
            (
                query,
                self._build_rerank_text(chunk),
            )
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        scored_chunks = [
            (
                chunk,
                float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

        scored_chunks.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_chunks: list[RetrievedChunk] = []

        for rank, (chunk, score) in enumerate(scored_chunks, start=1):
            reranked_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    content=chunk.content,
                    retrieval_score=score,
                    retrieval_method=f"{chunk.retrieval_method}_cross_encoder",
                    rank=rank,
                )
            )

        return reranked_chunks

    def _build_rerank_text(self, chunk: RetrievedChunk) -> str:
        parts = [
            f"Document title: {chunk.title}",
            f"Section path: {chunk.section_path}",
            f"Section title: {chunk.section_title}",
            f"Content: {chunk.content}",
        ]

        return "\n".join(part for part in parts if part)


class CrossEncoderRerankedRetriever:
    def __init__(
        self,
        base_retriever: BaseRetriever,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        candidate_k = max(
            top_k,
            settings.reranker.candidate_k,
        )

        candidates = self.base_retriever.retrieve(
            session=session,
            query=query,
            top_k=candidate_k,
        )

        reranked = self.reranker.rerank(
            query=query,
            chunks=candidates,
        )

        return reranked[:top_k]
