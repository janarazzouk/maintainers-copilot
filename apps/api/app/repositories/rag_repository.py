from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.rag import RagChunk, RagDocument


class RagRepository:
    """Database access for RAG documents and chunks.

    This layer owns SQLAlchemy operations only.
    It should not contain retrieval logic or LLM logic.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_document(self, data: dict) -> None:
        document = self.db.scalar(
            select(RagDocument).where(RagDocument.doc_id == data["doc_id"])
        )

        if document is None:
            document = RagDocument(doc_id=data["doc_id"])

        document.source_type = data["source_type"]
        document.repo = data["repo"]
        document.issue_id = data["issue_id"]
        document.title = data["title"]
        document.url = data["url"]
        document.final_label = data["final_label"]
        document.state = data["state"]
        document.issue_created_at = data["issue_created_at"]
        document.issue_closed_at = data["issue_closed_at"]
        document.problem_summary = data["problem_summary"]
        document.maintainer_answer = data["maintainer_answer"]
        document.resolution_type = data["resolution_type"]
        document.text = data["text"]
        document.raw_metadata = data["raw_metadata"]

        self.db.add(document)

    def upsert_chunk(self, data: dict) -> None:
        chunk = self.db.scalar(
            select(RagChunk).where(RagChunk.chunk_id == data["chunk_id"])
        )

        if chunk is None:
            chunk = RagChunk(chunk_id=data["chunk_id"])

        chunk.doc_id = data["doc_id"]
        chunk.source_type = data["source_type"]
        chunk.title = data["title"]
        chunk.url = data["url"]
        chunk.chunk_index = data["chunk_index"]
        chunk.chunk_text = data["chunk_text"]
        chunk.final_label = data["final_label"]
        chunk.issue_id = data["issue_id"]
        chunk.raw_metadata = data["raw_metadata"]

        # Embeddings will be added in the next step.
        if "embedding" in data:
            chunk.embedding = data["embedding"]

        self.db.add(chunk)

    def document_exists(self, doc_id: str) -> bool:
        result = self.db.scalar(
            select(RagDocument.doc_id).where(RagDocument.doc_id == doc_id)
        )
        return result is not None

    def count_documents(self) -> int:
        return self.db.scalar(select(func.count()).select_from(RagDocument)) or 0

    def count_chunks(self) -> int:
        return self.db.scalar(select(func.count()).select_from(RagChunk)) or 0
    

    def list_chunks_missing_embeddings(self, limit: int = 500) -> list[RagChunk]:
        statement = (
            select(RagChunk)
            .where(RagChunk.embedding.is_(None))
            .order_by(RagChunk.id)
            .limit(limit)
        )

        return list(self.db.scalars(statement).all())

    def update_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> None:
        chunk = self.db.scalar(
            select(RagChunk).where(RagChunk.chunk_id == chunk_id)
        )

        if chunk is None:
            raise ValueError(f"Chunk not found: {chunk_id}")

        chunk.embedding = embedding
        self.db.add(chunk)

    def count_chunks_missing_embeddings(self) -> int:
        return (
            self.db.scalar(
                select(func.count())
                .select_from(RagChunk)
                .where(RagChunk.embedding.is_(None))
            )
            or 0
        )

    def count_chunks_with_embeddings(self) -> int:
        return (
            self.db.scalar(
                select(func.count())
                .select_from(RagChunk)
                .where(RagChunk.embedding.is_not(None))
            )
            or 0
        )
    
    def search_chunks_by_vector(
        self,
        query_embedding: list[float],
        limit: int = 5,
        final_label: str | None = None,
    ) -> list[tuple[RagChunk, float]]:
        """Search embedded chunks using cosine distance.

        Lower distance is better.
        """

        distance = RagChunk.embedding.cosine_distance(query_embedding).label("distance")

        statement = (
            select(RagChunk, distance)
            .where(RagChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(limit)
        )

        if final_label:
            statement = statement.where(RagChunk.final_label == final_label)

        rows = self.db.execute(statement).all()

        return [(row[0], float(row[1])) for row in rows]

    def get_documents_by_doc_ids(
        self,
        doc_ids: list[str],
    ) -> dict[str, RagDocument]:
        if not doc_ids:
            return {}

        statement = select(RagDocument).where(RagDocument.doc_id.in_(doc_ids))
        documents = self.db.scalars(statement).all()

        return {document.doc_id: document for document in documents}