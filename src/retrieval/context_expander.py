from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.models import RetrievedChunk
from src.storage.schema import ChunkORM


def orm_chunk_to_retrieved_chunk(
    chunk: ChunkORM,
    retrieval_score: float,
    retrieval_method: str,
    rank: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.document.title,
        source_url=chunk.document.source_url,
        section_title=chunk.section_title,
        section_path=chunk.section_path,
        content=chunk.chunk_text,
        retrieval_score=retrieval_score,
        retrieval_method=retrieval_method,
        rank=rank,
    )


def get_neighbor_chunks(
    session: Session,
    seed_chunk: RetrievedChunk,
    window: int = 1,
) -> list[ChunkORM]:
    seed_orm = session.execute(
        select(ChunkORM).where(ChunkORM.chunk_id == seed_chunk.chunk_id)
    ).scalar_one_or_none()

    if seed_orm is None:
        return []

    min_index = seed_orm.chunk_index - window
    max_index = seed_orm.chunk_index + window

    return list(
        session.execute(
            select(ChunkORM)
            .where(ChunkORM.doc_id == seed_orm.doc_id)
            .where(ChunkORM.chunk_index >= min_index)
            .where(ChunkORM.chunk_index <= max_index)
            .order_by(ChunkORM.chunk_index)
        )
        .scalars()
        .all()
    )


def expand_with_neighbor_chunks(
    session: Session,
    chunks: list[RetrievedChunk],
    window: int = 1,
    max_chunks: int = 8,
    max_neighbor_context_chars: int | None = None,
) -> list[RetrievedChunk]:
    if not chunks or max_chunks <= 0:
        return []

    if max_neighbor_context_chars is not None and max_neighbor_context_chars < 0:
        raise ValueError("max_neighbor_context_chars must be at least 0.")

    unique_seed_chunks: list[RetrievedChunk] = []
    seen_seed_ids: set[str] = set()

    for chunk in chunks:
        if chunk.chunk_id in seen_seed_ids:
            continue

        unique_seed_chunks.append(chunk)
        seen_seed_ids.add(chunk.chunk_id)

    # The existing max_chunks setting remains the hard limit on
    # the final number of context chunks.
    expanded = unique_seed_chunks[:max_chunks]
    seen_chunk_ids = {chunk.chunk_id for chunk in expanded}

    if len(expanded) >= max_chunks:
        return rerank_context_chunks(expanded)

    if max_neighbor_context_chars == 0:
        return rerank_context_chunks(expanded)

    added_neighbor_context_chars = 0

    for seed_chunk in unique_seed_chunks:
        neighbor_orms = get_neighbor_chunks(
            session=session,
            seed_chunk=seed_chunk,
            window=window,
        )

        for neighbor_orm in neighbor_orms:
            if len(expanded) >= max_chunks:
                break

            if neighbor_orm.chunk_id in seen_chunk_ids:
                continue

            neighbor_context_chars = len(neighbor_orm.chunk_text)

            if max_neighbor_context_chars is not None and (
                added_neighbor_context_chars + neighbor_context_chars
                > max_neighbor_context_chars
            ):
                continue

            expanded.append(
                orm_chunk_to_retrieved_chunk(
                    chunk=neighbor_orm,
                    retrieval_score=seed_chunk.retrieval_score,
                    retrieval_method=(f"{seed_chunk.retrieval_method}_neighbor"),
                    rank=len(expanded) + 1,
                )
            )
            seen_chunk_ids.add(neighbor_orm.chunk_id)
            added_neighbor_context_chars += neighbor_context_chars

        if len(expanded) >= max_chunks:
            break

    return rerank_context_chunks(expanded)


def rerank_context_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    reranked: list[RetrievedChunk] = []

    for rank, chunk in enumerate(chunks, start=1):
        reranked.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_url=chunk.source_url,
                section_title=chunk.section_title,
                section_path=chunk.section_path,
                content=chunk.content,
                retrieval_score=chunk.retrieval_score,
                retrieval_method=chunk.retrieval_method,
                rank=rank,
            )
        )

    return reranked
