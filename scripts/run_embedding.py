from src.storage.db import SessionLocal
from src.embedding.pipeline import embed_chunks_without_vectors


def main() -> None:
    with SessionLocal() as session:
        embed_chunks_without_vectors(session)


if __name__ == "__main__":
    main()