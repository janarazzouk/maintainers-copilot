from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_database(database_url: str) -> None:
    """Initialize SQLAlchemy using the database URL loaded from Vault."""
    global _engine, _SessionLocal

    _engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    if _SessionLocal is None:
        raise RuntimeError("Database has not been initialized.")

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> None:
    """Fail fast if the API cannot reach Postgres."""
    if _engine is None:
        raise RuntimeError("Database has not been initialized.")

    with _engine.connect() as connection:
        connection.execute(text("SELECT 1"))