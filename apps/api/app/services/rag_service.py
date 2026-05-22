from dataclasses import dataclass

from sqlalchemy.orm import Session
from app.infra.groq_client import GroqLLMClient

from app.infra.embeddings import EmbeddingModel
from app.models.rag import RagChunk
from app.repositories.rag_repository import RagRepository
from app.schemas.rag import RagQueryRequest, RagQueryResponse, RagSource, RagTrace
from app.services.rag_query_rewriter import RuleBasedMultiQueryRewriter
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
    """RAG retrieval service.

    Current version:
    - rule-based multi-query rewriting
    - dense vector search
    - keyword search
    - hybrid merge
    - lightweight technical reranking
    - parent-document retrieval
    """

    DENSE_WEIGHT = 0.60
    KEYWORD_WEIGHT = 0.40

    def __init__(
        self,
        db: Session,
        embedding_model: EmbeddingModel,
        llm_client: GroqLLMClient | None = None,
    ) -> None:
        self.repository = RagRepository(db)
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.query_rewriter = RuleBasedMultiQueryRewriter()
        self.reranker = LightweightTechnicalReranker()

    def query(self, payload: RagQueryRequest) -> RagQueryResponse:
        generated_queries = self.query_rewriter.rewrite(payload.question)

        candidate_limit = max(payload.top_k * 6, 20)

        all_hybrid_candidates: list[HybridCandidate] = []

        for search_query in generated_queries:
            all_hybrid_candidates.extend(
                self._retrieve_hybrid_candidates(
                    search_query=search_query,
                    candidate_limit=candidate_limit,
                    label_filter=payload.label_filter,
                )
            )

        merged_candidates = self._merge_candidates_across_queries(
            all_hybrid_candidates
        )

        reranked_candidates = self.reranker.rerank(
            question=payload.question,
            candidates=merged_candidates,
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


        answer = self._build_retrieval_answer(sources)
        llm_model = None
        llm_usage = None

        if payload.generate_answer:
            if self.llm_client is None:
                raise RuntimeError(
                    "LLM answer generation was requested, but Groq is not configured."
                )

            context_sources = self._select_generation_sources(sources)
            context_blocks = self._build_context_blocks(context_sources)
            llm_response = self.llm_client.generate_rag_answer(
                question=payload.question,
                context_blocks=context_blocks,
            )
            answer = llm_response.content
            llm_model = llm_response.model
            llm_usage = llm_response.usage
        return RagQueryResponse(
            answer=answer,
            sources=sources,
            trace=RagTrace(
                original_question=payload.question,
                generated_queries=generated_queries,
                retrieval_mode=(
                    "multi_query_hybrid_dense_keyword_with_lightweight_technical_reranking"
                ),
                candidate_chunk_count=len(merged_candidates),
                retrieved_chunk_ids=[source.chunk_id for source in sources],
                retrieved_doc_ids=self._unique_doc_ids(
                    [source.doc_id for source in sources]
                ),
            ),
            llm_model=llm_model,
            llm_usage=llm_usage,
        )

    def _retrieve_hybrid_candidates(
        self,
        search_query: str,
        candidate_limit: int,
        label_filter: str | None,
    ) -> list[HybridCandidate]:
        query_embedding = self.embedding_model.embed_query(search_query)

        dense_candidates = self.repository.search_chunks_by_vector(
            query_embedding=query_embedding,
            limit=candidate_limit,
            final_label=label_filter,
        )

        keyword_candidates = self.repository.search_chunks_by_keyword(
            query_text=search_query,
            limit=candidate_limit,
            final_label=label_filter,
        )

        return self._merge_hybrid_candidates(
            dense_candidates=dense_candidates,
            keyword_candidates=keyword_candidates,
        )

    def _merge_hybrid_candidates(
        self,
        dense_candidates: list[tuple[RagChunk, float]],
        keyword_candidates: list[tuple[RagChunk, float]],
    ) -> list[HybridCandidate]:
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

    def _merge_candidates_across_queries(
        self,
        candidates: list[HybridCandidate],
    ) -> list[HybridCandidate]:
        """Merge duplicated chunks retrieved by multiple rewritten queries.

        If the same chunk appears from several queries, keep its best score.
        """

        best_by_chunk_id: dict[str, HybridCandidate] = {}

        for candidate in candidates:
            current = best_by_chunk_id.get(candidate.chunk.chunk_id)

            if current is None or candidate.score > current.score:
                best_by_chunk_id[candidate.chunk.chunk_id] = candidate

        return sorted(
            best_by_chunk_id.values(),
            key=lambda item: item.score,
            reverse=True,
        )

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
            "This endpoint is currently using multi-query hybrid retrieval "
            "with lightweight technical reranking. "
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
    
    def _build_context_blocks(self, sources: list[RagSource]) -> list[str]:
        blocks: list[str] = []

        for index, source in enumerate(sources, start=1):
            issue_part = (
                f"#{source.issue_id}" if source.issue_id is not None else source.doc_id
            )

            block = (
                f"Source {index}\n"
                f"Issue: {issue_part}\n"
                f"Title: {source.title}\n"
                f"URL: {source.url}\n"
                f"Label: {source.final_label}\n"
                f"Resolution type: {source.resolution_type}\n"
                f"Problem summary: {source.problem_summary_excerpt or ''}\n"
                f"Maintainer answer: {source.maintainer_answer_excerpt or ''}\n"
                f"Relevant chunk: {source.chunk_excerpt}\n"
            )

            blocks.append(block)

        return blocks
    
    def _select_generation_sources(self, sources: list[RagSource]) -> list[RagSource]:
        """Choose sources for LLM generation.

        The API can return several retrieved sources, but the LLM should not be
        asked to merge unrelated issues into one answer.
        """

        if not sources:
            return []

        top_source = sources[0]
        selected: list[RagSource] = [top_source]

        for source in sources[1:]:
            same_issue_strength = (
                source.matched_terms
                and source.score >= top_source.score * 0.80
            )

            close_score = source.score >= top_source.score * 0.90

            if same_issue_strength or close_score:
                selected.append(source)

        return selected[:3]