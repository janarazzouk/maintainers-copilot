from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.repositories.rag_repository import RagRepository
from app.schemas.rag import RagQueryRequest, RagQueryResponse, RagSource, RagTrace


class RagService:
    """Dense retrieval service for the RAG pipeline.

    Current version:
    - embeds the question
    - searches pgvector
    - returns top chunks with parent document context

    Later versions will add:
    - BM25 hybrid search
    - cross-encoder reranking
    - multi-query rewriting
    - LLM generation
    """

    def __init__(
        self,
        db: Session,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.repository = RagRepository(db)
        self.embedding_model = embedding_model

    def query(self, payload: RagQueryRequest) -> RagQueryResponse:
        query_embedding = self.embedding_model.embed_query(payload.question)

        retrieved = self.repository.search_chunks_by_vector(
            query_embedding=query_embedding,
            limit=payload.top_k,
            final_label=payload.label_filter,
        )

        doc_ids = self._unique_doc_ids([chunk.doc_id for chunk, _distance in retrieved])
        documents_by_id = self.repository.get_documents_by_doc_ids(doc_ids)

        sources: list[RagSource] = []

        for chunk, distance in retrieved:
            document = documents_by_id.get(chunk.doc_id)

            # pgvector cosine distance: lower is better.
            # Convert to a rough similarity score for display.
            score = max(0.0, 1.0 - distance)

            sources.append(
                RagSource(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    issue_id=chunk.issue_id,
                    title=chunk.title,
                    url=chunk.url,
                    final_label=chunk.final_label,
                    resolution_type=document.resolution_type if document else None,
                    score=round(score, 4),
                    chunk_text=chunk.chunk_text,
                    problem_summary=document.problem_summary if document else None,
                    maintainer_answer=document.maintainer_answer if document else None,
                )
            )

        return RagQueryResponse(
            answer=self._build_retrieval_answer(sources),
            sources=sources,
            trace=RagTrace(
                original_question=payload.question,
                retrieved_chunk_ids=[source.chunk_id for source in sources],
                retrieved_doc_ids=self._unique_doc_ids(
                    [source.doc_id for source in sources]
                ),
            ),
        )

    def _build_retrieval_answer(self, sources: list[RagSource]) -> str:
        if not sources:
            return (
                "I could not find relevant resolved Node.js issues for this question."
            )

        top = sources[0]

        issue_part = (
            f"issue #{top.issue_id}" if top.issue_id is not None else top.doc_id
        )

        return (
            f"I found {len(sources)} relevant RAG source(s). "
            f"The top match is {issue_part}: {top.title}. "
            "This endpoint is currently returning retrieved context only; "
            "LLM answer generation will be added after retrieval evaluation."
        )

    def _unique_doc_ids(self, doc_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []

        for doc_id in doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc_id)

        return unique