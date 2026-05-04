from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.models import SearchResponse
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal


router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: Literal["dense", "bm25", "hybrid"] = settings.retrieval.default_method
    top_k: int = Field(default=settings.retrieval.top_k, ge=1, le=20)


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    retriever = get_retriever(request.method)

    with SessionLocal() as session:
        results = retriever.retrieve(
            session=session,
            query=request.query,
            top_k=request.top_k,
        )

    return SearchResponse(
        query=request.query,
        retrieval_method=request.method,
        top_k=request.top_k,
        results=results,
    )