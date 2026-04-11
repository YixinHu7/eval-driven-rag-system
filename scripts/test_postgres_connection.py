import os

from sqlalchemy import create_engine, text


def main() -> None:
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/eval_driven_rag",
    )

    engine = create_engine(database_url)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.scalar_one()
        print("Connected to PostgreSQL successfully.")
        print(version)

        ext_result = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        )
        extension = ext_result.scalar_one_or_none()
        print(f"pgvector enabled: {extension == 'vector'}")


if __name__ == "__main__":
    main()