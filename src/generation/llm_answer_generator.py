from src.core.models import AnswerResponse, RetrievedChunk
from src.generation.abstention import should_abstain
from src.generation.citation_builder import build_used_citations
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_grounded_qa_prompt
from src.generation.context_selector import select_context_chunks


class LLMAnswerGenerator:
    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        retrieval_strategy: str,
        query_type: str | None = None,
    ) -> AnswerResponse:
        if should_abstain(query, chunks):
            return AnswerResponse(
                answer=(
                    "I could not find enough relevant documentation to answer this question "
                    "with confidence."
                ),
                citations=[],
                query_type=query_type,
                retrieval_strategy=retrieval_strategy,
                confidence=0.0,
                abstained=True,
                retrieved_chunks=chunks,
            )

        context_chunks = select_context_chunks(chunks=chunks, max_chunks=5)

        prompt = build_grounded_qa_prompt(query=query, chunks=context_chunks)
        answer = self.llm_client.generate(prompt)
        citations = build_used_citations(answer=answer, chunks=context_chunks)

        if not citations:
            return AnswerResponse(
                answer=(
                    "I found potentially relevant documentation, but I could not produce "
                    "a sufficiently grounded answer with verifiable citations."
                ),
                citations=[],
                query_type=query_type,
                retrieval_strategy=retrieval_strategy,
                confidence=0.0,
                abstained=True,
                retrieved_chunks=context_chunks,
            )

        return AnswerResponse(
            answer=answer,
            citations=citations,
            query_type=query_type,
            retrieval_strategy=retrieval_strategy,
            confidence=self._estimate_confidence(chunks),
            abstained=False,
            retrieved_chunks=context_chunks,
        )

    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0

        top_score = chunks[0].retrieval_score
        method = chunks[0].retrieval_method

        if method == "dense":
            return max(0.0, min(1.0, 1.0 - top_score))

        if method == "bm25":
            return max(0.0, min(1.0, top_score / 2.0))

        if method == "hybrid_rrf":
            return max(0.0, min(1.0, top_score * 20.0))

        return 0.5