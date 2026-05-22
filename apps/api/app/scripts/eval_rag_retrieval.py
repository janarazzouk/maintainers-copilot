import json
import os
import sys

import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.infra.config import get_settings
from app.infra.database import get_db, init_database
from app.infra.embeddings import EmbeddingModel
from app.infra.vault import VaultClient, VaultError
from app.models.rag import RagChunk
from app.repositories.rag_repository import RagRepository
from app.schemas.rag import RagQueryRequest
from app.services.rag_service import RagService
from app.infra.vault import resolve_vault_token

DENSE_WEIGHT = 0.60
KEYWORD_WEIGHT = 0.40


@dataclass(frozen=True)
class RetrievedItem:
    chunk_id: str
    doc_id: str
    score: float


@dataclass(frozen=True)
class RagEvalExampleResult:
    mode: str
    question_id: str
    label: str
    question: str
    ground_truth_doc_ids: list[str]
    ground_truth_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit_at_5: bool
    hit_at_10: bool
    mrr_at_10: float
    doc_recall_at_5: float
    chunk_recall_at_10: float
    top_doc_id: str | None
    top_chunk_id: str | None


@dataclass(frozen=True)
class RagEvalSummary:
    mode: str
    total_examples: int
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    doc_recall_at_5: float
    chunk_recall_at_10: float
    by_label: dict[str, dict[str, float]]


def main() -> None:
    settings = get_settings()

    database_url = _resolve_database_url(settings)
    init_database(database_url)

    api_root = Path(__file__).resolve().parents[2]
    golden_path = _resolve_path(api_root, settings.rag_golden_path)

    examples = _read_jsonl(golden_path)

    print(f"Loaded {len(examples)} RAG golden examples from {golden_path}")
    print(f"Loading embedding model: {settings.embedding_model_name}")

    embedding_model = EmbeddingModel(
                        model_name=settings.embedding_model_name,
                        cache_dir=settings.embedding_cache_dir,
                    )

    db_generator = get_db()
    db = next(db_generator)

    try:
        repository = RagRepository(db)
        final_service = RagService(
            db=db,
            embedding_model=embedding_model,
        )

        modes = [
            "dense_only",
            "hybrid_only",
            "final_multi_query_hybrid_rerank",
        ]

        all_results: list[RagEvalExampleResult] = []

        for mode in modes:
            print(f"\nRunning RAG eval mode: {mode}")

            mode_results = [
                _evaluate_one_example(
                    mode=mode,
                    repository=repository,
                    final_service=final_service,
                    embedding_model=embedding_model,
                    example=example,
                )
                for example in examples
            ]

            all_results.extend(mode_results)

        summaries = [
            _summarize(mode, [result for result in all_results if result.mode == mode])
            for mode in modes
        ]

        report = {
            "summaries": [asdict(summary) for summary in summaries],
            "examples": [asdict(result) for result in all_results],
        }

        output_path = api_root / "data" / "rag" / "rag_eval_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        _print_comparison(summaries)
        threshold_path = api_root / "eval_thresholds.yaml"
        _validate_thresholds(
            summaries=summaries,
            threshold_path=threshold_path,
        )
        print(f"\nWrote eval report to: {output_path}")

    finally:
        db_generator.close()


def _evaluate_one_example(
    mode: str,
    repository: RagRepository,
    final_service: RagService,
    embedding_model: EmbeddingModel,
    example: dict[str, Any],
) -> RagEvalExampleResult:
    question_id = str(example.get("question_id") or example.get("id") or "")
    label = str(example.get("label") or "unknown")
    question = str(example["question"])

    ground_truth_doc_ids = [
        str(value) for value in example.get("ground_truth_doc_ids", [])
    ]
    ground_truth_chunk_ids = [
        str(value) for value in example.get("ground_truth_chunk_ids", [])
    ]

    retrieved_items = _retrieve_for_mode(
        mode=mode,
        question=question,
        repository=repository,
        final_service=final_service,
        embedding_model=embedding_model,
        top_k=10,
    )

    retrieved_doc_ids = [item.doc_id for item in retrieved_items]
    retrieved_chunk_ids = [item.chunk_id for item in retrieved_items]

    top_5_doc_ids = retrieved_doc_ids[:5]
    top_10_doc_ids = retrieved_doc_ids[:10]

    hit_at_5 = _has_overlap(ground_truth_doc_ids, top_5_doc_ids)
    hit_at_10 = _has_overlap(ground_truth_doc_ids, top_10_doc_ids)

    return RagEvalExampleResult(
        mode=mode,
        question_id=question_id,
        label=label,
        question=question,
        ground_truth_doc_ids=ground_truth_doc_ids,
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        hit_at_5=hit_at_5,
        hit_at_10=hit_at_10,
        mrr_at_10=_mrr_at_k(
            ground_truth_ids=ground_truth_doc_ids,
            retrieved_ids=top_10_doc_ids,
            k=10,
        ),
        doc_recall_at_5=_recall_at_k(
            ground_truth_ids=ground_truth_doc_ids,
            retrieved_ids=top_5_doc_ids,
            k=5,
        ),
        chunk_recall_at_10=_recall_at_k(
            ground_truth_ids=ground_truth_chunk_ids,
            retrieved_ids=retrieved_chunk_ids[:10],
            k=10,
        ),
        top_doc_id=retrieved_doc_ids[0] if retrieved_doc_ids else None,
        top_chunk_id=retrieved_chunk_ids[0] if retrieved_chunk_ids else None,
    )


def _retrieve_for_mode(
    mode: str,
    question: str,
    repository: RagRepository,
    final_service: RagService,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[RetrievedItem]:
    if mode == "dense_only":
        return _retrieve_dense_only(
            question=question,
            repository=repository,
            embedding_model=embedding_model,
            top_k=top_k,
        )

    if mode == "hybrid_only":
        return _retrieve_hybrid_only(
            question=question,
            repository=repository,
            embedding_model=embedding_model,
            top_k=top_k,
        )

    if mode == "final_multi_query_hybrid_rerank":
        response = final_service.query(
            RagQueryRequest(
                question=question,
                top_k=top_k,
                label_filter=None,
            )
        )

        return [
            RetrievedItem(
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                score=source.score,
            )
            for source in response.sources
        ]

    raise ValueError(f"Unknown eval mode: {mode}")


def _retrieve_dense_only(
    question: str,
    repository: RagRepository,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[RetrievedItem]:
    candidate_limit = max(top_k * 6, 20)
    query_embedding = embedding_model.embed_query(question)

    dense_candidates = repository.search_chunks_by_vector(
        query_embedding=query_embedding,
        limit=candidate_limit,
        final_label=None,
    )

    items = [
        RetrievedItem(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            score=max(0.0, 1.0 - float(distance)),
        )
        for chunk, distance in dense_candidates
    ]

    return _deduplicate_by_doc_id(items, top_k=top_k)


def _retrieve_hybrid_only(
    question: str,
    repository: RagRepository,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> list[RetrievedItem]:
    candidate_limit = max(top_k * 6, 20)
    query_embedding = embedding_model.embed_query(question)

    dense_candidates = repository.search_chunks_by_vector(
        query_embedding=query_embedding,
        limit=candidate_limit,
        final_label=None,
    )

    keyword_candidates = repository.search_chunks_by_keyword(
        query_text=question,
        limit=candidate_limit,
        final_label=None,
    )

    merged = _merge_dense_and_keyword_candidates(
        dense_candidates=dense_candidates,
        keyword_candidates=keyword_candidates,
    )

    return _deduplicate_by_doc_id(merged, top_k=top_k)


def _merge_dense_and_keyword_candidates(
    dense_candidates: list[tuple[RagChunk, float]],
    keyword_candidates: list[tuple[RagChunk, float]],
) -> list[RetrievedItem]:
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

    merged: list[RetrievedItem] = []

    for chunk_id, chunk in chunks_by_id.items():
        dense_score = dense_scores.get(chunk_id, 0.0)
        keyword_score = keyword_scores.get(chunk_id, 0.0)

        final_score = (
            DENSE_WEIGHT * dense_score
            + KEYWORD_WEIGHT * keyword_score
        )

        merged.append(
            RetrievedItem(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                score=final_score,
            )
        )

    return sorted(merged, key=lambda item: item.score, reverse=True)


def _deduplicate_by_doc_id(
    items: list[RetrievedItem],
    top_k: int,
) -> list[RetrievedItem]:
    selected: list[RetrievedItem] = []
    seen_doc_ids: set[str] = set()

    for item in items:
        if item.doc_id in seen_doc_ids:
            continue

        selected.append(item)
        seen_doc_ids.add(item.doc_id)

        if len(selected) >= top_k:
            break

    return selected


def _summarize(mode: str, results: list[RagEvalExampleResult]) -> RagEvalSummary:
    if not results:
        raise ValueError(f"Cannot summarize empty RAG eval results for mode: {mode}")

    labels = sorted({result.label for result in results})
    by_label: dict[str, dict[str, float]] = {}

    for label in labels:
        label_results = [result for result in results if result.label == label]
        by_label[label] = _metric_dict(label_results)

    return RagEvalSummary(
        mode=mode,
        total_examples=len(results),
        hit_at_5=_mean([1.0 if result.hit_at_5 else 0.0 for result in results]),
        hit_at_10=_mean([1.0 if result.hit_at_10 else 0.0 for result in results]),
        mrr_at_10=_mean([result.mrr_at_10 for result in results]),
        doc_recall_at_5=_mean([result.doc_recall_at_5 for result in results]),
        chunk_recall_at_10=_mean([result.chunk_recall_at_10 for result in results]),
        by_label=by_label,
    )


def _metric_dict(results: list[RagEvalExampleResult]) -> dict[str, float]:
    return {
        "count": float(len(results)),
        "hit_at_5": _mean([1.0 if result.hit_at_5 else 0.0 for result in results]),
        "hit_at_10": _mean([1.0 if result.hit_at_10 else 0.0 for result in results]),
        "mrr_at_10": _mean([result.mrr_at_10 for result in results]),
        "doc_recall_at_5": _mean([result.doc_recall_at_5 for result in results]),
        "chunk_recall_at_10": _mean(
            [result.chunk_recall_at_10 for result in results]
        ),
    }


def _has_overlap(expected: list[str], actual: list[str]) -> bool:
    return bool(set(expected).intersection(actual))


def _mrr_at_k(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    ground_truth = set(ground_truth_ids)

    for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        if retrieved_id in ground_truth:
            return 1.0 / rank

    return 0.0


def _recall_at_k(
    ground_truth_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    ground_truth = set(ground_truth_ids)

    if not ground_truth:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    return len(ground_truth.intersection(retrieved)) / len(ground_truth)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def _print_comparison(summaries: list[RagEvalSummary]) -> None:
    print("\nRAG Retrieval Eval Comparison")
    print("=============================")
    print(
        f"{'Mode':45} "
        f"{'Hit@5':>8} "
        f"{'Hit@10':>8} "
        f"{'MRR@10':>8} "
        f"{'DocR@5':>8} "
        f"{'ChunkR@10':>10}"
    )
    print("-" * 95)

    for summary in summaries:
        print(
            f"{summary.mode:45} "
            f"{summary.hit_at_5:8.4f} "
            f"{summary.hit_at_10:8.4f} "
            f"{summary.mrr_at_10:8.4f} "
            f"{summary.doc_recall_at_5:8.4f} "
            f"{summary.chunk_recall_at_10:10.4f}"
        )

    print("\nBy label")
    print("--------")

    for summary in summaries:
        print(f"\nMode: {summary.mode}")

        for label, metrics in summary.by_label.items():
            print(
                f"  {label}: "
                f"count={int(metrics['count'])}, "
                f"Hit@5={metrics['hit_at_5']:.4f}, "
                f"MRR@10={metrics['mrr_at_10']:.4f}, "
                f"DocRecall@5={metrics['doc_recall_at_5']:.4f}, "
                f"ChunkRecall@10={metrics['chunk_recall_at_10']:.4f}"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

    return records


def _resolve_database_url(settings) -> str:
    try:
        vault = VaultClient(
            addr=settings.vault_addr,
            token=resolve_vault_token(settings),
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


def _resolve_path(api_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return api_root / path

def _validate_thresholds(
    summaries: list[RagEvalSummary],
    threshold_path: Path,
) -> None:
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with threshold_path.open("r", encoding="utf-8") as file:
        thresholds = yaml.safe_load(file) or {}

    rag_thresholds = thresholds.get("rag") or {}
    mode = rag_thresholds.get("mode", "final_multi_query_hybrid_rerank")

    summary_by_mode = {summary.mode: summary for summary in summaries}

    if mode not in summary_by_mode:
        raise ValueError(
            f"Threshold mode {mode!r} was not found in eval summaries. "
            f"Available modes: {sorted(summary_by_mode)}"
        )

    summary = summary_by_mode[mode]

    checks = [
        ("hit_at_5", summary.hit_at_5, rag_thresholds.get("hit_at_5_min")),
        ("hit_at_10", summary.hit_at_10, rag_thresholds.get("hit_at_10_min")),
        ("mrr_at_10", summary.mrr_at_10, rag_thresholds.get("mrr_at_10_min")),
        (
            "doc_recall_at_5",
            summary.doc_recall_at_5,
            rag_thresholds.get("doc_recall_at_5_min"),
        ),
    ]

    failures: list[str] = []

    for metric_name, actual, minimum in checks:
        if minimum is None:
            continue

        minimum_float = float(minimum)

        if actual < minimum_float:
            failures.append(
                f"{metric_name}: actual={actual:.4f}, required>={minimum_float:.4f}"
            )

    if failures:
        print("\nRAG eval threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("\nRAG eval threshold check passed.")


if __name__ == "__main__":
    main()