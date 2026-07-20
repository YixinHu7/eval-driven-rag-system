from types import SimpleNamespace
from typing import Any

from src.core.models import RetrievedChunk
from src.retrieval import context_expander


def make_chunk(
    chunk_id: str,
    rank: int,
    retrieval_score: float = 0.1,
    retrieval_method: str = "hybrid_rrf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="k8s_probes",
        title="Configure Probes",
        source_url="https://example.com",
        section_title=f"Section {chunk_id}",
        section_path=f"Configure Probes > Section {chunk_id}",
        content=f"Content for {chunk_id}.",
        retrieval_score=retrieval_score,
        retrieval_method=retrieval_method,
        rank=rank,
    )


def make_orm_chunk(
    chunk_id: str,
    chunk_index: int,
    chunk_text: str | None = None,
) -> Any:
    content = chunk_text if chunk_text is not None else f"Content for {chunk_id}."

    return SimpleNamespace(
        chunk_id=chunk_id,
        doc_id="k8s_probes",
        chunk_index=chunk_index,
        chunk_text=content,
        section_title=f"Section {chunk_id}",
        section_path=(f"Configure Probes > Section {chunk_id}"),
        document=SimpleNamespace(
            title="Configure Probes",
            source_url="https://example.com",
        ),
    )


def test_rerank_context_chunks_assigns_sequential_ranks() -> None:
    chunks = [
        make_chunk("chunk_a", rank=10),
        make_chunk("chunk_b", rank=20),
        make_chunk("chunk_c", rank=30),
    ]

    reranked = context_expander.rerank_context_chunks(chunks)

    assert [chunk.rank for chunk in reranked] == [1, 2, 3]
    assert [chunk.chunk_id for chunk in reranked] == [
        "chunk_a",
        "chunk_b",
        "chunk_c",
    ]


def test_empty_chunk_list_returns_empty_result() -> None:
    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[],
    )

    assert result == []


def test_non_positive_max_chunks_returns_empty_result() -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed],
        max_chunks=0,
    )

    assert result == []


def test_retrieved_seeds_are_preserved_before_neighbors(
    monkeypatch,
) -> None:
    seed_1 = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )
    seed_2 = make_chunk(
        chunk_id="seed_2",
        rank=2,
        retrieval_score=0.8,
        retrieval_method="hybrid",
    )

    seed_1_orm = make_orm_chunk(
        chunk_id="seed_1",
        chunk_index=1,
    )
    seed_2_orm = make_orm_chunk(
        chunk_id="seed_2",
        chunk_index=2,
    )
    neighbor_orm = make_orm_chunk(
        chunk_id="neighbor_3",
        chunk_index=3,
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        assert window == 1

        return [
            seed_1_orm,
            seed_2_orm,
            neighbor_orm,
        ]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed_1, seed_2],
        window=1,
        max_chunks=3,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
        "seed_2",
        "neighbor_3",
    ]

    assert result[0].retrieval_method == "hybrid"
    assert result[0].retrieval_score == 0.9

    assert result[1].retrieval_method == "hybrid"
    assert result[1].retrieval_score == 0.8

    assert result[2].retrieval_method == "hybrid_neighbor"
    assert result[2].retrieval_score == 0.9

    assert [chunk.rank for chunk in result] == [1, 2, 3]


def test_neighbors_do_not_displace_retrieved_seeds(
    monkeypatch,
) -> None:
    seeds = [
        make_chunk(
            chunk_id="seed_1",
            rank=1,
            retrieval_score=0.9,
            retrieval_method="hybrid",
        ),
        make_chunk(
            chunk_id="seed_2",
            rank=2,
            retrieval_score=0.8,
            retrieval_method="hybrid",
        ),
        make_chunk(
            chunk_id="seed_3",
            rank=3,
            retrieval_score=0.7,
            retrieval_method="hybrid",
        ),
    ]

    neighbor_orm = make_orm_chunk(
        chunk_id="extra_neighbor",
        chunk_index=0,
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return [neighbor_orm]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=seeds,
        window=1,
        max_chunks=3,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
        "seed_2",
        "seed_3",
    ]

    assert all(chunk.retrieval_method == "hybrid" for chunk in result)


def test_duplicate_seed_chunks_are_removed(
    monkeypatch,
) -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return []

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed, seed],
        max_chunks=8,
    )

    assert len(result) == 1
    assert result[0].chunk_id == "seed_1"
    assert result[0].rank == 1


def test_zero_neighbor_character_budget_preserves_only_seeds(
    monkeypatch,
) -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )
    neighbor = make_orm_chunk(
        chunk_id="neighbor_1",
        chunk_index=2,
        chunk_text="Neighbor content",
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return [neighbor]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed],
        max_chunks=8,
        max_neighbor_context_chars=0,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
    ]


def test_neighbor_exceeding_character_budget_is_skipped(
    monkeypatch,
) -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )
    neighbor = make_orm_chunk(
        chunk_id="neighbor_1",
        chunk_index=2,
        chunk_text="123456",
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return [neighbor]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed],
        max_chunks=8,
        max_neighbor_context_chars=5,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
    ]


def test_neighbors_are_added_until_character_budget_is_reached(
    monkeypatch,
) -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )
    neighbor_1 = make_orm_chunk(
        chunk_id="neighbor_1",
        chunk_index=2,
        chunk_text="12345",
    )
    neighbor_2 = make_orm_chunk(
        chunk_id="neighbor_2",
        chunk_index=3,
        chunk_text="1234",
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return [
            neighbor_1,
            neighbor_2,
        ]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed],
        max_chunks=8,
        max_neighbor_context_chars=8,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
        "neighbor_1",
    ]


def test_neighbor_character_budget_allows_exact_total(
    monkeypatch,
) -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
        retrieval_score=0.9,
        retrieval_method="hybrid",
    )
    neighbor_1 = make_orm_chunk(
        chunk_id="neighbor_1",
        chunk_index=2,
        chunk_text="12345",
    )
    neighbor_2 = make_orm_chunk(
        chunk_id="neighbor_2",
        chunk_index=3,
        chunk_text="1234",
    )

    def fake_get_neighbor_chunks(
        session,
        seed_chunk,
        window,
    ):
        return [
            neighbor_1,
            neighbor_2,
        ]

    monkeypatch.setattr(
        context_expander,
        "get_neighbor_chunks",
        fake_get_neighbor_chunks,
    )

    result = context_expander.expand_with_neighbor_chunks(
        session=object(),
        chunks=[seed],
        max_chunks=8,
        max_neighbor_context_chars=9,
    )

    assert [chunk.chunk_id for chunk in result] == [
        "seed_1",
        "neighbor_1",
        "neighbor_2",
    ]


def test_negative_neighbor_character_budget_raises_error() -> None:
    seed = make_chunk(
        chunk_id="seed_1",
        rank=1,
    )

    try:
        context_expander.expand_with_neighbor_chunks(
            session=object(),
            chunks=[seed],
            max_neighbor_context_chars=-1,
        )
    except ValueError as error:
        assert str(error) == ("max_neighbor_context_chars must be at least 0.")
    else:
        raise AssertionError("Expected a ValueError for a negative budget.")
