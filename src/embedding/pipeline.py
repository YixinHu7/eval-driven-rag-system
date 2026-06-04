from sqlalchemy import select
from sqlalchemy.orm import Session

from src.embedding.embedder import LocalEmbedder
from src.storage.schema import ChunkORM
from src.retrieval.text_builder import build_chunk_search_text


def embed_chunks_without_vectors(session: Session, batch_size: int = 16) -> None:
    embedder = LocalEmbedder()

    stmt = select(ChunkORM).where(ChunkORM.embedding.is_(None))
    chunks = list(session.scalars(stmt).all())

    print(f"Found {len(chunks)} chunks without embeddings")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts = [build_chunk_search_text(chunk) for chunk in batch]
        embeddings = embedder.embed_texts(texts)

        for chunk, emb in zip(batch, embeddings):
            chunk.embedding = emb

        session.commit()
        print(f"Processed batch {i // batch_size + 1}")