import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.models.rag import RagChunk


class CandidateLike(Protocol):
    chunk: RagChunk
    score: float


@dataclass(frozen=True)
class RerankedCandidate:
    chunk: RagChunk
    score: float
    base_score: float
    rerank_bonus: float
    matched_terms: list[str]


class LightweightTechnicalReranker:
    """Small local reranker for technical GitHub issue retrieval.

    It boosts candidates that contain exact technical terms from the query:
    - error codes
    - function names
    - dotted APIs
    - versions
    - file/path-like strings

    This avoids heavy PyTorch/cross-encoder dependencies for now.
    """

    def rerank(
        self,
        question: str,
        candidates: Sequence[CandidateLike],
    ) -> list[RerankedCandidate]:
        query_terms = self._extract_query_terms(question)
        query_words = self._tokenize_words(question)

        reranked: list[RerankedCandidate] = []

        for candidate in candidates:
            chunk = candidate.chunk
            candidate_text = self._candidate_text(chunk)

            matched_terms = self._matched_terms(query_terms, candidate_text)
            entity_score = (
                len(matched_terms) / len(query_terms)
                if query_terms
                else 0.0
            )

            candidate_words = self._tokenize_words(candidate_text)
            word_overlap_score = (
                len(query_words.intersection(candidate_words)) / len(query_words)
                if query_words
                else 0.0
            )

            rerank_bonus = min(
                1.0,
                (0.75 * entity_score) + (0.25 * word_overlap_score),
            )

            # Keep the existing hybrid score dominant, but let exact technical
            # matches move good candidates upward.
            final_score = (0.75 * candidate.score) + (0.25 * rerank_bonus)

            reranked.append(
                RerankedCandidate(
                    chunk=chunk,
                    score=final_score,
                    base_score=candidate.score,
                    rerank_bonus=rerank_bonus,
                    matched_terms=matched_terms,
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _candidate_text(self, chunk: RagChunk) -> str:
        return f"{chunk.title}\n{chunk.chunk_text}"

    def _extract_query_terms(self, question: str) -> list[str]:
        patterns = [
            # Error constants: ERR_BUFFER_OUT_OF_BOUNDS, ERR_INVALID_STATE
            r"\b[A-Z][A-Z0-9_]{3,}\b",

            # JS errors: RangeError, TypeError, SyntaxError
            r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b",

            # Dotted APIs: Buffer.toString, fs.readFile, StatementSync.iterate
            r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b",

            # Function calls: readFile(), toString(), iterate()
            r"\b[A-Za-z_$][\w$]*\(\)",

            # Node versions: v22.7.0, 22.7.0
            r"\bv?\d+\.\d+(?:\.\d+)?\b",

            # Path-like/code-like terms: node:internal/buffer, lib/buffer.js
            r"\b[A-Za-z0-9_@.-]+[:/][A-Za-z0-9_@./:-]+\b",
        ]

        terms: list[str] = []

        for pattern in patterns:
            for match in re.findall(pattern, question):
                terms.append(match)

        # Also keep meaningful quoted/backticked phrases.
        for match in re.findall(r"[`'\"]([^`'\"]{3,80})[`'\"]", question):
            terms.append(match)

        return self._unique_preserve_order(terms)

    def _matched_terms(self, terms: list[str], text: str) -> list[str]:
        lower_text = text.lower()
        matched: list[str] = []

        for term in terms:
            if term.lower() in lower_text:
                matched.append(term)

        return matched

    def _tokenize_words(self, text: str) -> set[str]:
        stopwords = {
            "what",
            "happened",
            "with",
            "the",
            "and",
            "for",
            "from",
            "this",
            "that",
            "node",
            "nodejs",
            "issue",
            "error",
            "bug",
            "does",
            "did",
            "was",
            "were",
            "are",
            "how",
            "why",
            "when",
            "where",
        }

        words = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_.$:-]+", text)
            if len(token) >= 3
        }

        return words - stopwords

    def _unique_preserve_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []

        for value in values:
            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            unique.append(value)

        return unique