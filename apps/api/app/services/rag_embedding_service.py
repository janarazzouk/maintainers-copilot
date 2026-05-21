from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.repositories.rag_repository import RagRepository


@dataclass(frozen=True)
class RagEmbeddingResult:
    chunks_embedded: int
    chunks_with_embeddings: int
    chunks_missing_embeddings: int


class RagEmbeddingService:
    """Generates and stores embeddings for RAG chunks."""

    def __init__(
        self,
        db: Session,
        embedding_model: EmbeddingModel,
        batch_size: int,
    ) -> None:
        self.db = db
        self.repository = RagRepository(db)
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed_missing_chunks(self, limit: int = 10_000) -> RagEmbeddingResult:
        chunks = self.repository.list_chunks_missing_embeddings(limit=limit)

        if not chunks:
            return RagEmbeddingResult(
                chunks_embedded=0,
                chunks_with_embeddings=self.repository.count_chunks_with_embeddings(),
                chunks_missing_embeddings=self.repository.count_chunks_missing_embeddings(),
            )

        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(
            texts,
            batch_size=self.batch_size,
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count: "
                f"{len(embeddings)} embeddings for {len(chunks)} chunks."
            )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != 384:
                raise RuntimeError(
                    f"Expected 384-dimensional embedding for {chunk.chunk_id}, "
                    f"got {len(embedding)}."
                )

            self.repository.update_chunk_embedding(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
            )

        self.db.commit()

        return RagEmbeddingResult(
            chunks_embedded=len(chunks),
            chunks_with_embeddings=self.repository.count_chunks_with_embeddings(),
            chunks_missing_embeddings=self.repository.count_chunks_missing_embeddings(),
        )