from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.infra.embeddings import EmbeddingModel
from app.models.rag import RagChunk
from app.repositories.rag_repository import RagRepository
from app.schemas.rag import RagQueryRequest, RagQueryResponse, RagSource, RagTrace
from app.services.rag_reranker import (
    LightweightTechnicalReranker,
    RerankedCandidate,
)


@dataclass(frozen=True)
class HybridCandidate:
    chunk: RagChunk
    score: float
    dense_score: float
    keyword_score: float


class RagService:
    """Hybrid retrieval service for the RAG pipeline.

    Current version:
    - dense vector search
    - keyword search
    - hybrid merge
    - lightweight technical reranking
    - parent-document retrieval

    Later:
    - multi-query rewriting
    - LLM answer generation
    """

    DENSE_WEIGHT = 0.60
    KEYWORD_WEIGHT = 0.40

    def __init__(
        self,
        db: Session,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.repository = RagRepository(db)
        self.embedding_model = embedding_model
        self.reranker = LightweightTechnicalReranker()

    def query(self, payload: RagQueryRequest) -> RagQueryResponse:
        query_embedding = self.embedding_model.embed_query(payload.question)

        candidate_limit = max(payload.top_k * 6, 20)

        dense_candidates = self.repository.search_chunks_by_vector(
            query_embedding=query_embedding,
            limit=candidate_limit,
            final_label=payload.label_filter,
        )

        keyword_candidates = self.repository.search_chunks_by_keyword(
            query_text=payload.question,
            limit=candidate_limit,
            final_label=payload.label_filter,
        )

        hybrid_candidates = self._merge_hybrid_candidates(
            dense_candidates=dense_candidates,
            keyword_candidates=keyword_candidates,
        )

        reranked_candidates = self.reranker.rerank(
            question=payload.question,
            candidates=hybrid_candidates,
        )

        diversified = self._deduplicate_by_doc_id(
            candidates=reranked_candidates,
            top_k=payload.top_k,
        )

        doc_ids = self._unique_doc_ids(
            [candidate.chunk.doc_id for candidate in diversified]
        )
        documents_by_id = self.repository.get_documents_by_doc_ids(doc_ids)

        sources: list[RagSource] = []

        for candidate in diversified:
            chunk = candidate.chunk
            document = documents_by_id.get(chunk.doc_id)

            sources.append(
                RagSource(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    issue_id=chunk.issue_id,
                    title=chunk.title,
                    url=chunk.url,
                    final_label=chunk.final_label,
                    resolution_type=document.resolution_type if document else None,
                    score=round(candidate.score, 4),
                    base_hybrid_score=round(candidate.base_score, 4),
                    rerank_bonus=round(candidate.rerank_bonus, 4),
                    matched_terms=candidate.matched_terms,
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
                retrieval_mode=(
                    "hybrid_dense_keyword_with_lightweight_technical_reranking"
                ),
                candidate_chunk_count=len(reranked_candidates),
                retrieved_chunk_ids=[source.chunk_id for source in sources],
                retrieved_doc_ids=self._unique_doc_ids(
                    [source.doc_id for source in sources]
                ),
            ),
        )

    def _merge_hybrid_candidates(
        self,
        dense_candidates: list[tuple[RagChunk, float]],
        keyword_candidates: list[tuple[RagChunk, float]],
    ) -> list[HybridCandidate]:
        """Merge dense and keyword candidates into one ranked list.

        Dense search returns cosine distance, where lower is better.
        Keyword search returns full-text rank, where higher is better.
        """

        dense_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RagChunk] = {}

        for chunk, distance in dense_candidates:
            chunks_by_id[chunk.chunk_id] = chunk
            dense_scores[chunk.chunk_id] = max(0.0, 1.0 - float(distance))

        max_keyword_rank = max(
            [rank for _chunk, rank in keyword_candidates],
            default=0.0,
        )

        for chunk, rank in keyword_candidates:
            chunks_by_id[chunk.chunk_id] = chunk

            if max_keyword_rank > 0:
                keyword_scores[chunk.chunk_id] = float(rank) / max_keyword_rank
            else:
                keyword_scores[chunk.chunk_id] = 0.0

        merged: list[HybridCandidate] = []

        for chunk_id, chunk in chunks_by_id.items():
            dense_score = dense_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)

            final_score = (
                self.DENSE_WEIGHT * dense_score
                + self.KEYWORD_WEIGHT * keyword_score
            )

            merged.append(
                HybridCandidate(
                    chunk=chunk,
                    score=final_score,
                    dense_score=dense_score,
                    keyword_score=keyword_score,
                )
            )

        return sorted(merged, key=lambda item: item.score, reverse=True)

    def _deduplicate_by_doc_id(
        self,
        candidates: list[RerankedCandidate],
        top_k: int,
    ) -> list[RerankedCandidate]:
        selected: list[RerankedCandidate] = []
        seen_doc_ids: set[str] = set()

        for candidate in candidates:
            if candidate.chunk.doc_id in seen_doc_ids:
                continue

            selected.append(candidate)
            seen_doc_ids.add(candidate.chunk.doc_id)

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
            "This endpoint is currently using hybrid retrieval with "
            "lightweight technical reranking. "
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