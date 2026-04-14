from sqlalchemy import text

from src.storage.db import Base, engine
from src.storage.schema import ChunkORM, DocumentORM


def main() -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    print("Tables created successfully.")
    print(f"Registered ORM models: {DocumentORM.__tablename__}, {ChunkORM.__tablename__}")


if __name__ == "__main__":
    main()