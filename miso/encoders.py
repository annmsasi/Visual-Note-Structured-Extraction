"""Real embedder + reranker wrappers, both backed by sentence-transformers.

We use sentence-transformers' `CrossEncoder` for reranking rather than
FlagEmbedding's `FlagReranker`; FlagEmbedding ≤ 1.4 is incompatible with
transformers 5.x. Same model weights, one fewer dependency.
"""
from __future__ import annotations

import math
from typing import Any


class STEmbedder:
    """Sentence-transformers embedder. Default `BAAI/bge-base-en-v1.5` (768-dim).

    Returns L2-normalised vectors so cosine similarity equals dot product, per
    the bge model card.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self._model: Any = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return arr.tolist()


class BGEReranker:
    """Cross-encoder reranker. Default `BAAI/bge-reranker-base`.

    Raw model scores are sigmoided into [0, 1] so the reranker threshold
    reads as a probability.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self._model: Any = CrossEncoder(model_name)
        self.model_name = model_name

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(s))) for s in logits]
