from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.models import AnswerResponse
from src.generation.answer_generator import SimpleAnswerGenerator
from src.retrieval.factory import get_retriever
from src.storage.db import SessionLocal
from src.generation.llm_answer_generator import LLMAnswerGenerator


router = APIRouter(prefix="/answer", tags=["answer"])


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: Literal["dense", "bm25", "hybrid"] = settings.retrieval.default_method
    top_k: int = Field(default=settings.retrieval.top_k, ge=1, le=20)
    generator: Literal["simple", "llm"] = "simple"

@router.post("", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    retriever = get_retriever(request.method)
    
    if request.generator == "llm":
        generator = LLMAnswerGenerator()
    else:
        generator = SimpleAnswerGenerator()

    with SessionLocal() as session:
        chunks = retriever.retrieve(
            session=session,
            query=request.query,
            top_k=request.top_k,
        )

    return generator.generate(
        query=request.query,
        chunks=chunks,
        retrieval_strategy=request.method,
    )