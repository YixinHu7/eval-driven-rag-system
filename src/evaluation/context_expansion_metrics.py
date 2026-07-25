from pydantic import BaseModel

from src.core.models import RetrievedChunk


class ContextExpansionEvalResult(BaseModel):
    retrieval_performed: bool
    expansion_enabled: bool

    initial_chunk_count: int
    final_chunk_count: int
    added_neighbor_count: int

    initial_context_chars: int
    final_context_chars: int
    added_context_chars: int

    added_neighbor_chunk_ids: list[str]
    added_neighbor_doc_sections: list[str]


class ContextExpansionSummary(BaseModel):
    total_questions: int
    retrieval_questions: int
    expansion_enabled: bool

    questions_with_added_neighbors: int
    neighbor_addition_rate: float

    average_initial_chunks: float
    average_final_chunks: float
    average_added_neighbors: float
    average_added_context_chars: float

    max_added_neighbors: int


def evaluate_context_expansion(
    initial_chunks: list[RetrievedChunk],
    final_chunks: list[RetrievedChunk],
    expansion_enabled: bool,
    retrieval_performed: bool,
) -> ContextExpansionEvalResult:
    initial_chunk_ids = {chunk.chunk_id for chunk in initial_chunks}

    added_neighbor_chunks = [
        chunk for chunk in final_chunks if chunk.chunk_id not in initial_chunk_ids
    ]

    initial_context_chars = sum(len(chunk.content) for chunk in initial_chunks)
    final_context_chars = sum(len(chunk.content) for chunk in final_chunks)
    added_context_chars = sum(len(chunk.content) for chunk in added_neighbor_chunks)

    return ContextExpansionEvalResult(
        retrieval_performed=retrieval_performed,
        expansion_enabled=expansion_enabled,
        initial_chunk_count=len(initial_chunks),
        final_chunk_count=len(final_chunks),
        added_neighbor_count=len(added_neighbor_chunks),
        initial_context_chars=initial_context_chars,
        final_context_chars=final_context_chars,
        added_context_chars=added_context_chars,
        added_neighbor_chunk_ids=[chunk.chunk_id for chunk in added_neighbor_chunks],
        added_neighbor_doc_sections=[
            f"{chunk.doc_id}::{chunk.section_title}" for chunk in added_neighbor_chunks
        ],
    )


def summarize_context_expansion_results(
    results: list[ContextExpansionEvalResult],
    expansion_enabled: bool,
) -> ContextExpansionSummary:
    retrieval_results = [result for result in results if result.retrieval_performed]

    retrieval_questions = len(retrieval_results)

    if not retrieval_results:
        return ContextExpansionSummary(
            total_questions=len(results),
            retrieval_questions=0,
            expansion_enabled=expansion_enabled,
            questions_with_added_neighbors=0,
            neighbor_addition_rate=0.0,
            average_initial_chunks=0.0,
            average_final_chunks=0.0,
            average_added_neighbors=0.0,
            average_added_context_chars=0.0,
            max_added_neighbors=0,
        )

    questions_with_added_neighbors = sum(
        result.added_neighbor_count > 0 for result in retrieval_results
    )

    return ContextExpansionSummary(
        total_questions=len(results),
        retrieval_questions=retrieval_questions,
        expansion_enabled=expansion_enabled,
        questions_with_added_neighbors=questions_with_added_neighbors,
        neighbor_addition_rate=(questions_with_added_neighbors / retrieval_questions),
        average_initial_chunks=(
            sum(result.initial_chunk_count for result in retrieval_results)
            / retrieval_questions
        ),
        average_final_chunks=(
            sum(result.final_chunk_count for result in retrieval_results)
            / retrieval_questions
        ),
        average_added_neighbors=(
            sum(result.added_neighbor_count for result in retrieval_results)
            / retrieval_questions
        ),
        average_added_context_chars=(
            sum(result.added_context_chars for result in retrieval_results)
            / retrieval_questions
        ),
        max_added_neighbors=max(
            result.added_neighbor_count for result in retrieval_results
        ),
    )
