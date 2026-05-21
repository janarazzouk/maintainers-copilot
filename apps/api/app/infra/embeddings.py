import numpy as np
from fastembed import TextEmbedding


class EmbeddingModel:
    """Local embedding model adapter using FastEmbed.

    This avoids PyTorch/CUDA downloads in Docker.
    It uses small ONNX-based embedding models.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []

        embeddings = list(
            self._model.embed(
                texts,
                batch_size=batch_size,
            )
        )

        return [self._normalize(vector).tolist() for vector in embeddings]

    def embed_query(self, query: str) -> list[float]:
        embeddings = list(self._model.embed([query]))

        if not embeddings:
            raise RuntimeError("Embedding model returned no query embedding.")

        return self._normalize(embeddings[0]).tolist()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector

        return vector / norm