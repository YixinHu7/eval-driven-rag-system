from typing import Protocol

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


class Retriever(Protocol):
    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        ...


def get_retriever(method: str | None = None) -> Retriever:
    selected_method = method or settings.retrieval.default_method

    if selected_method == "dense":
        return DenseRetriever()

    if selected_method == "bm25":
        return BM25Retriever()

    if selected_method in {"hybrid", "hybrid_rrf"}:
        return HybridRetriever()

    raise ValueError(
        f"Unsupported retrieval method: {selected_method}. "
        "Expected one of: dense, bm25, hybrid."
    )