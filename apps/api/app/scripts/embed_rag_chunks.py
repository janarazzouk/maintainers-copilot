import os

from app.infra.config import get_settings
from app.infra.database import get_db, init_database
from app.infra.embeddings import EmbeddingModel
from app.infra.vault import VaultClient, VaultError
from app.services.rag_embedding_service import RagEmbeddingService


def main() -> None:
    settings = get_settings()

    database_url = _resolve_database_url(settings)
    init_database(database_url)

    print(f"Loading embedding model: {settings.embedding_model_name}")
    embedding_model = EmbeddingModel(settings.embedding_model_name)

    db_generator = get_db()
    db = next(db_generator)

    try:
        service = RagEmbeddingService(
            db=db,
            embedding_model=embedding_model,
            batch_size=settings.embedding_batch_size,
        )

        result = service.embed_missing_chunks()

        print("RAG embedding completed successfully.")
        print(f"Chunks embedded this run: {result.chunks_embedded}")
        print(f"Chunks with embeddings: {result.chunks_with_embeddings}")
        print(f"Chunks missing embeddings: {result.chunks_missing_embeddings}")

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
        pass

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL was not found in Vault or environment."
        )

    return database_url


if __name__ == "__main__":
    main()