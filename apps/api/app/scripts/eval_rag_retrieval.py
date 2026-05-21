import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.infra.config import get_settings
from app.infra.database import get_db, init_database
from app.infra.embeddings import EmbeddingModel
from app.infra.vault import VaultClient, VaultError
from app.schemas.rag import RagQueryRequest
from app.services.rag_service import RagService


@dataclass(frozen=True)
class RagEvalExampleResult:
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
    retrieval_mode: str
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

    embedding_model = EmbeddingModel(settings.embedding_model_name)

    db_generator = get_db()
    db = next(db_generator)

    try:
        service = RagService(
            db=db,
            embedding_model=embedding_model,
        )

        results = [
            _evaluate_one_example(service=service, example=example)
            for example in examples
        ]

        summary = _summarize(results)

        report = {
            "summary": asdict(summary),
            "examples": [asdict(result) for result in results],
        }

        output_path = api_root / "data" / "rag" / "rag_eval_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        _print_summary(summary)
        print(f"\nWrote eval report to: {output_path}")

    finally:
        db_generator.close()


def _evaluate_one_example(
    service: RagService,
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

    response = service.query(
        RagQueryRequest(
            question=question,
            top_k=10,
            # Do not use label_filter during eval.
            # In real usage, the user may not know the label.
            label_filter=None,
        )
    )

    retrieved_doc_ids = response.trace.retrieved_doc_ids
    retrieved_chunk_ids = response.trace.retrieved_chunk_ids

    top_5_doc_ids = retrieved_doc_ids[:5]
    top_10_doc_ids = retrieved_doc_ids[:10]

    hit_at_5 = _has_overlap(ground_truth_doc_ids, top_5_doc_ids)
    hit_at_10 = _has_overlap(ground_truth_doc_ids, top_10_doc_ids)

    return RagEvalExampleResult(
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


def _summarize(results: list[RagEvalExampleResult]) -> RagEvalSummary:
    if not results:
        raise ValueError("Cannot summarize empty RAG eval results.")

    labels = sorted({result.label for result in results})

    by_label: dict[str, dict[str, float]] = {}

    for label in labels:
        label_results = [result for result in results if result.label == label]
        by_label[label] = _metric_dict(label_results)

    return RagEvalSummary(
        retrieval_mode="multi_query_hybrid_dense_keyword_with_lightweight_technical_reranking",
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


def _print_summary(summary: RagEvalSummary) -> None:
    print("\nRAG Retrieval Eval Summary")
    print("==========================")
    print(f"Mode: {summary.retrieval_mode}")
    print(f"Examples: {summary.total_examples}")
    print(f"Hit@5: {summary.hit_at_5:.4f}")
    print(f"Hit@10: {summary.hit_at_10:.4f}")
    print(f"MRR@10: {summary.mrr_at_10:.4f}")
    print(f"Doc Recall@5: {summary.doc_recall_at_5:.4f}")
    print(f"Chunk Recall@10: {summary.chunk_recall_at_10:.4f}")

    print("\nBy label")
    print("--------")

    for label, metrics in summary.by_label.items():
        print(
            f"{label}: "
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


def _resolve_path(api_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return api_root / path


if __name__ == "__main__":
    main()