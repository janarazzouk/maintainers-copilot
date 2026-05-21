import os
from pathlib import Path

from app.infra.config import get_settings
from app.infra.database import get_db, init_database
from app.infra.vault import VaultClient, VaultError
from app.services.rag_ingestion_service import RagIngestionService


def main() -> None:
    settings = get_settings()

    database_url = _resolve_database_url(settings)
    init_database(database_url)

    api_root = Path(__file__).resolve().parents[2]

    corpus_path = _resolve_path(api_root, settings.rag_corpus_path)
    chunks_path = _resolve_path(api_root, settings.rag_chunks_path)

    db_generator = get_db()
    db = next(db_generator)

    try:
        service = RagIngestionService(db)
        result = service.ingest(
            corpus_path=corpus_path,
            chunks_path=chunks_path,
        )

        print("RAG ingestion completed successfully.")
        print(f"Documents loaded this run: {result.documents_loaded}")
        print(f"Chunks loaded this run: {result.chunks_loaded}")
        print(f"Total documents in DB: {result.total_documents}")
        print(f"Total chunks in DB: {result.total_chunks}")

    finally:
        db_generator.close()


def _resolve_database_url(settings) -> str:
    try:
        vault = VaultClient(
            addr=settings.vault_addr,
            token=settings.vault_dev_root_token_id,
        )
        secrets = vault.read_app_secrets()
        database_url = secrets.get("database_url")

        if database_url:
            return str(database_url)

    except VaultError:
        # Local fallback for development scripts only.
        pass

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL was not found in Vault or environment."
        )

    return database_url


def _resolve_path(api_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return api_root / path


if __name__ == "__main__":
    main()