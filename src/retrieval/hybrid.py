from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(self, rrf_k: int = 60) -> None:
        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or settings.retrieval.top_k
        candidates = candidate_k or settings.retrieval.candidate_k

        dense_results = self.dense_retriever.retrieve(
            session=session,
            query=query,
            top_k=candidates,
        )
        bm25_results = self.bm25_retriever.retrieve(
            session=session,
            query=query,
            top_k=candidates,
        )

        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, RetrievedChunk] = {}

        self._accumulate_rrf_scores(
            results=dense_results,
            fused_scores=fused_scores,
            chunk_by_id=chunk_by_id,
            source_weight=1.0,
        )
        self._accumulate_rrf_scores(
            results=bm25_results,
            fused_scores=fused_scores,
            chunk_by_id=chunk_by_id,
            source_weight=1.0,
        )

        ranked_chunk_ids = sorted(
            fused_scores.keys(),
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        final_results: list[RetrievedChunk] = []

        for rank, chunk_id in enumerate(ranked_chunk_ids[:final_k], start=1):
            original = chunk_by_id[chunk_id]

            final_results.append(
                RetrievedChunk(
                    chunk_id=original.chunk_id,
                    doc_id=original.doc_id,
                    title=original.title,
                    source_url=original.source_url,
                    section_title=original.section_title,
                    section_path=original.section_path,
                    content=original.content,
                    retrieval_score=fused_scores[chunk_id],
                    retrieval_method="hybrid_rrf",
                    rank=rank,
                )
            )

        return final_results

    def _accumulate_rrf_scores(
        self,
        results: list[RetrievedChunk],
        fused_scores: dict[str, float],
        chunk_by_id: dict[str, RetrievedChunk],
        source_weight: float,
    ) -> None:
        for result in results:
            chunk_by_id[result.chunk_id] = result
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                source_weight / (self.rrf_k + result.rank)
            )