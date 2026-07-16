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
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    expanded: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()

    for seed_chunk in chunks:
        if len(expanded) >= max_chunks:
            break

        neighbor_orms = get_neighbor_chunks(
            session=session,
            seed_chunk=seed_chunk,
            window=window,
        )

        # Keep the retrieved seed chunk first so its original ranking signal is preserved.
        if seed_chunk.chunk_id not in seen_chunk_ids:
            expanded.append(seed_chunk)
            seen_chunk_ids.add(seed_chunk.chunk_id)

        for neighbor_orm in neighbor_orms:
            if len(expanded) >= max_chunks:
                break

            if neighbor_orm.chunk_id in seen_chunk_ids:
                continue

            expanded.append(
                orm_chunk_to_retrieved_chunk(
                    chunk=neighbor_orm,
                    retrieval_score=seed_chunk.retrieval_score,
                    retrieval_method=f"{seed_chunk.retrieval_method}_neighbor",
                    rank=len(expanded) + 1,
                )
            )
            seen_chunk_ids.add(neighbor_orm.chunk_id)

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