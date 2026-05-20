"""Issue classification service using fine-tuned RoBERTa."""

import torch

from app.infra.model_loader import ModelBundle
from app.schemas.nlp import ClassificationResponse, IssueTextRequest
from app.services.text_preprocessing import build_issue_text


def classify_issue(
    request: IssueTextRequest,
    model_bundle: ModelBundle,
) -> ClassificationResponse:
    """Classify a GitHub issue into bug / feature / docs / question."""

    text = build_issue_text(request.title, request.body)

    inputs = model_bundle.tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=model_bundle.max_length,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model_bundle.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model_bundle.model(**inputs)
        logits = outputs.logits
        probabilities_tensor = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities_tensor).item())
    confidence = float(probabilities_tensor[predicted_id].item())

    probabilities = {
        label: float(probabilities_tensor[label_id].item())
        for label_id, label in model_bundle.id_to_label.items()
    }

    predicted_label = model_bundle.id_to_label[predicted_id]

    return ClassificationResponse(
        label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
    )