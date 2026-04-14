from sqlalchemy import text

from src.storage.db import SessionLocal


def main() -> None:
    with SessionLocal() as session:
        result = session.execute(text("SELECT current_database(), current_user;"))
        row = result.one()

        print("Connected through SessionLocal successfully.")
        print(f"database={row[0]}, user={row[1]}")


if __name__ == "__main__":
    main()