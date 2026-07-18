import argparse

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from src.storage.db import SessionLocal
from src.storage.schema import ChunkORM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect matching chunks and their neighboring chunks."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search text in section titles, section paths, and chunk content.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optionally restrict results to one document.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Number of neighboring chunks to show on each side.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of matching seed chunks.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum number of characters printed per chunk.",
    )

    args = parser.parse_args()

    if not args.query and not args.doc_id:
        parser.error("At least one of --query or --doc-id is required.")

    if args.window < 0:
        parser.error("--window must be at least 0.")

    if args.limit <= 0:
        parser.error("--limit must be greater than 0.")

    if args.max_chars <= 0:
        parser.error("--max-chars must be greater than 0.")

    return args


def find_seed_chunks(
    session: Session,
    query: str | None,
    doc_id: str | None,
    limit: int,
) -> list[ChunkORM]:
    statement = select(ChunkORM)

    if doc_id:
        statement = statement.where(ChunkORM.doc_id == doc_id)

    if query:
        pattern = f"%{query.lower()}%"

        statement = statement.where(
            or_(
                func.lower(ChunkORM.section_title).like(pattern),
                func.lower(ChunkORM.section_path).like(pattern),
                func.lower(ChunkORM.chunk_text).like(pattern),
            )
        )

    statement = statement.order_by(
        ChunkORM.doc_id,
        ChunkORM.chunk_index,
    ).limit(limit)

    return list(session.execute(statement).scalars().all())


def find_neighbor_chunks(
    session: Session,
    seed_chunk: ChunkORM,
    window: int,
) -> list[ChunkORM]:
    min_index = seed_chunk.chunk_index - window
    max_index = seed_chunk.chunk_index + window

    statement = (
        select(ChunkORM)
        .where(ChunkORM.doc_id == seed_chunk.doc_id)
        .where(ChunkORM.chunk_index >= min_index)
        .where(ChunkORM.chunk_index <= max_index)
        .order_by(ChunkORM.chunk_index)
    )

    return list(session.execute(statement).scalars().all())


def truncate_content(content: str, max_chars: int) -> str:
    normalized = content.strip()

    if len(normalized) <= max_chars:
        return normalized

    return f"{normalized[:max_chars].rstrip()}..."


def print_neighbor_window(
    seed_chunk: ChunkORM,
    neighbors: list[ChunkORM],
    max_chars: int,
) -> None:
    print("\n" + "=" * 100)
    print(f"Seed chunk: {seed_chunk.chunk_id}")
    print(f"Document: {seed_chunk.doc_id}")
    print(f"Seed index: {seed_chunk.chunk_index}")
    print(f"Seed section: {seed_chunk.section_title}")
    print("=" * 100)

    for chunk in neighbors:
        offset = chunk.chunk_index - seed_chunk.chunk_index
        position = "SEED" if offset == 0 else f"{offset:+d}"

        print()
        print("-" * 100)
        print(f"Position: {position}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Chunk index: {chunk.chunk_index}")
        print(f"Section title: {chunk.section_title}")
        print(f"Section path: {chunk.section_path}")
        print(
            "Different section from seed: "
            f"{chunk.section_title != seed_chunk.section_title}"
        )
        print("-" * 100)
        print(truncate_content(chunk.chunk_text, max_chars))


def main() -> None:
    args = parse_args()

    with SessionLocal() as session:
        seed_chunks = find_seed_chunks(
            session=session,
            query=args.query,
            doc_id=args.doc_id,
            limit=args.limit,
        )

        if not seed_chunks:
            print("No matching chunks found.")
            return

        print(f"Found {len(seed_chunks)} matching seed chunk(s).")

        for seed_chunk in seed_chunks:
            neighbors = find_neighbor_chunks(
                session=session,
                seed_chunk=seed_chunk,
                window=args.window,
            )

            print_neighbor_window(
                seed_chunk=seed_chunk,
                neighbors=neighbors,
                max_chars=args.max_chars,
            )


if __name__ == "__main__":
    main()
