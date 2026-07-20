from src.core.models import RetrievedChunk
from src.evaluation.context_expansion_metrics import (
    evaluate_context_expansion,
    summarize_context_expansion_results,
)


def make_chunk(
    chunk_id: str,
    content: str,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="test_doc",
        title="Test Document",
        source_url="https://example.com",
        section_title=f"Section {chunk_id}",
        section_path=f"Test Document > Section {chunk_id}",
        content=content,
        retrieval_score=0.5,
        retrieval_method="hybrid",
        rank=rank,
    )


def test_context_expansion_identifies_added_neighbor_chunks() -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        content="Seed content",
        rank=1,
    )
    neighbor = make_chunk(
        chunk_id="neighbor_1",
        content="Neighbor content",
        rank=2,
    )

    result = evaluate_context_expansion(
        initial_chunks=[seed],
        final_chunks=[seed, neighbor],
        expansion_enabled=True,
        retrieval_performed=True,
    )

    assert result.initial_chunk_count == 1
    assert result.final_chunk_count == 2
    assert result.added_neighbor_count == 1
    assert result.added_neighbor_chunk_ids == ["neighbor_1"]
    assert result.added_neighbor_doc_sections == [
        "test_doc::Section neighbor_1"
    ]
    assert result.initial_context_chars == len("Seed content")
    assert result.final_context_chars == (
        len("Seed content")
        + len("Neighbor content")
    )
    assert result.added_context_chars == len("Neighbor content")


def test_disabled_expansion_adds_no_neighbor_chunks() -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        content="Seed content",
        rank=1,
    )

    result = evaluate_context_expansion(
        initial_chunks=[seed],
        final_chunks=[seed],
        expansion_enabled=False,
        retrieval_performed=True,
    )

    assert result.expansion_enabled is False
    assert result.initial_chunk_count == 1
    assert result.final_chunk_count == 1
    assert result.added_neighbor_count == 0
    assert result.added_context_chars == 0


def test_summary_excludes_questions_without_retrieval() -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        content="Seed content",
        rank=1,
    )
    neighbor = make_chunk(
        chunk_id="neighbor_1",
        content="Neighbor content",
        rank=2,
    )

    retrieval_result = evaluate_context_expansion(
        initial_chunks=[seed],
        final_chunks=[seed, neighbor],
        expansion_enabled=True,
        retrieval_performed=True,
    )
    short_circuit_result = evaluate_context_expansion(
        initial_chunks=[],
        final_chunks=[],
        expansion_enabled=True,
        retrieval_performed=False,
    )

    summary = summarize_context_expansion_results(
        results=[
            retrieval_result,
            short_circuit_result,
        ],
        expansion_enabled=True,
    )

    assert summary.total_questions == 2
    assert summary.retrieval_questions == 1
    assert summary.questions_with_added_neighbors == 1
    assert summary.neighbor_addition_rate == 1.0
    assert summary.average_initial_chunks == 1.0
    assert summary.average_final_chunks == 2.0
    assert summary.average_added_neighbors == 1.0
    assert summary.average_added_context_chars == len(
        "Neighbor content"
    )
    assert summary.max_added_neighbors == 1


def test_empty_retrieval_summary_returns_zero_metrics() -> None:
    result = evaluate_context_expansion(
        initial_chunks=[],
        final_chunks=[],
        expansion_enabled=True,
        retrieval_performed=False,
    )

    summary = summarize_context_expansion_results(
        results=[result],
        expansion_enabled=True,
    )

    assert summary.total_questions == 1
    assert summary.retrieval_questions == 0
    assert summary.questions_with_added_neighbors == 0
    assert summary.neighbor_addition_rate == 0.0
    assert summary.average_initial_chunks == 0.0
    assert summary.average_final_chunks == 0.0
    assert summary.average_added_neighbors == 0.0
    assert summary.average_added_context_chars == 0.0
    assert summary.max_added_neighbors == 0