"""RoBERTa model loading utilities."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from app.infra.config import Settings


@dataclass(frozen=True)
class ModelBundle:
    """Loaded classifier artifacts used at inference time."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: torch.device
    id_to_label: dict[int, str]
    label_to_id: dict[str, int]
    max_length: int


def _resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_device = torch.device(device_setting)

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return requested_device


def _resolve_model_dir(model_dir: Path) -> Path:
    resolved = model_dir.resolve()

    if not resolved.exists():
        raise RuntimeError(f"Model directory does not exist: {resolved}")

    return resolved


def _load_label_mapping(model_dir: Path) -> tuple[dict[int, str], dict[str, int]]:
    mapping_path = model_dir / "label_mapping.json"

    if not mapping_path.exists():
        raise RuntimeError(f"Missing label mapping file: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as file:
        mapping: dict[str, Any] = json.load(file)

    raw_id_to_label = mapping.get("id2label")
    raw_label_to_id = mapping.get("label2id")

    if not isinstance(raw_id_to_label, dict):
        raise RuntimeError("label_mapping.json must contain an 'id2label' object.")

    if not isinstance(raw_label_to_id, dict):
        raise RuntimeError("label_mapping.json must contain a 'label2id' object.")

    id_to_label = {
        int(label_id): str(label)
        for label_id, label in raw_id_to_label.items()
    }

    label_to_id = {
        str(label): int(label_id)
        for label, label_id in raw_label_to_id.items()
    }

    if set(id_to_label.values()) != set(label_to_id.keys()):
        raise RuntimeError("label2id and id2label do not contain the same labels.")

    return id_to_label, label_to_id


def load_roberta_classifier(settings: Settings) -> ModelBundle:
    """Load tokenizer, RoBERTa classifier, and label mapping once at startup."""

    model_dir = _resolve_model_dir(settings.model_dir)
    device = _resolve_device(settings.device)

    id_to_label, label_to_id = _load_label_mapping(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        max_length=settings.max_length,
    )