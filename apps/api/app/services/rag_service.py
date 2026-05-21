from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.repositories.rag_repository import RagRepository
from app.schemas.rag import RagQueryRequest, RagQueryResponse, RagSource, RagTrace


class RagService:
    """Dense retrieval service for the RAG pipeline.

    Current version:
    - embeds the question
    - searches pgvector
    - deduplicates repeated chunks by parent doc_id
    - returns clean parent-document context previews

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

        # Retrieve more candidates than we display because many top chunks
        # may come from the same parent issue.
        candidate_limit = max(payload.top_k * 6, 20)

        candidates = self.repository.search_chunks_by_vector(
            query_embedding=query_embedding,
            limit=candidate_limit,
            final_label=payload.label_filter,
        )

        diversified = self._deduplicate_by_doc_id(
            candidates=candidates,
            top_k=payload.top_k,
        )

        doc_ids = self._unique_doc_ids(
            [chunk.doc_id for chunk, _distance in diversified]
        )
        documents_by_id = self.repository.get_documents_by_doc_ids(doc_ids)

        sources: list[RagSource] = []

        for chunk, distance in diversified:
            document = documents_by_id.get(chunk.doc_id)

            # pgvector cosine distance: lower is better.
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
                    chunk_excerpt=self._excerpt(chunk.chunk_text, max_chars=650),
                    problem_summary_excerpt=(
                        self._excerpt(document.problem_summary, max_chars=650)
                        if document
                        else None
                    ),
                    maintainer_answer_excerpt=(
                        self._excerpt(document.maintainer_answer, max_chars=650)
                        if document
                        else None
                    ),
                )
            )

        return RagQueryResponse(
            answer=self._build_retrieval_answer(sources),
            sources=sources,
            trace=RagTrace(
                original_question=payload.question,
                candidate_chunk_count=len(candidates),
                retrieved_chunk_ids=[source.chunk_id for source in sources],
                retrieved_doc_ids=self._unique_doc_ids(
                    [source.doc_id for source in sources]
                ),
            ),
        )

    def _deduplicate_by_doc_id(
        self,
        candidates: list[tuple[object, float]],
        top_k: int,
    ) -> list[tuple[object, float]]:
        """Keep only the best chunk per parent document.

        Without this, one issue with many similar chunks can dominate
        the whole response.
        """

        selected: list[tuple[object, float]] = []
        seen_doc_ids: set[str] = set()

        for chunk, distance in candidates:
            if chunk.doc_id in seen_doc_ids:
                continue

            selected.append((chunk, distance))
            seen_doc_ids.add(chunk.doc_id)

            if len(selected) >= top_k:
                break

        return selected

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
            f"I found {len(sources)} relevant parent issue(s). "
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

    def _excerpt(self, text: str | None, max_chars: int = 650) -> str:
        if not text:
            return ""

        cleaned = " ".join(text.split())

        if len(cleaned) <= max_chars:
            return cleaned

        return cleaned[: max_chars - 3].rstrip() + "..."