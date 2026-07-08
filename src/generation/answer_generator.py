from src.core.models import AnswerResponse, RetrievedChunk
from src.generation.abstention import should_abstain
from src.generation.citation_builder import build_citations


class SimpleAnswerGenerator:
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
                confidence=self._estimate_confidence(chunks),
                abstained=True,
                retrieved_chunks=[],
            )

        citations = build_citations(chunks)

        answer_parts: list[str] = []
        answer_parts.append(
            "Based on the retrieved documentation, the most relevant information is:"
        )

        for idx, chunk in enumerate(chunks, start=1):
            answer_parts.append(
                f"[{idx}] {chunk.content}"
            )

        answer = "\n\n".join(answer_parts)

        confidence = self._estimate_confidence(chunks)

        return AnswerResponse(
            answer=answer,
            citations=citations,
            query_type=query_type,
            retrieval_strategy=retrieval_strategy,
            confidence=confidence,
            abstained=False,
            retrieved_chunks=chunks,
        )

    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> float:
        if not chunks:
            return 0.0

        # Temporary heuristic:
        # This is not a calibrated probability. It only gives us a structured
        # confidence field for later abstention and evaluation work.
        top_score = chunks[0].retrieval_score

        if chunks[0].retrieval_method == "dense":
            # Dense uses cosine distance, where lower is better.
            return max(0.0, min(1.0, 1.0 - top_score))

        if chunks[0].retrieval_method == "bm25":
            # BM25 scores are unbounded, so use a simple bounded heuristic.
            return max(0.0, min(1.0, top_score / 2.0))

        if chunks[0].retrieval_method == "hybrid_rrf":
            # RRF scores are usually small.
            return max(0.0, min(1.0, top_score * 20.0))

        return 0.5