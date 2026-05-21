from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    # Optional metadata filter.
    # Example: "bug", "docs", "feature", "question"
    label_filter: str | None = None


class RagSource(BaseModel):
    doc_id: str
    chunk_id: str
    issue_id: int | None
    title: str
    url: str
    final_label: str | None
    resolution_type: str | None

    score: float
    base_hybrid_score: float
    rerank_bonus: float
    matched_terms: list[str]

    chunk_excerpt: str
    problem_summary_excerpt: str | None
    maintainer_answer_excerpt: str | None


class RagTrace(BaseModel):
    original_question: str
    retrieval_mode: str
    candidate_chunk_count: int
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSource]
    trace: RagTrace