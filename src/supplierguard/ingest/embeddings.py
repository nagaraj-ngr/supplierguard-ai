"""Embedders. All return L2-normalised float32 arrays so dot product == cosine similarity."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import List, Optional, Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray: ...


def _normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class HashEmbedder:
    """Offline, deterministic bag-of-words embedder (hashing trick, unigrams + bigrams).

    This is LEXICAL, not semantic: it matches shared words, not meaning. It exists so the
    pipeline and its tests run without an API key. Use OpenAIEmbedder for real experiments.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
            feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat, count in Counter(feats).items():
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                out[i, h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(count))
        return _normalise(out)


class OpenAIEmbedder:
    """OpenAI embeddings. Needs OPENAI_API_KEY and network access."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None,
                 batch_size: int = 96):
        from openai import OpenAI

        self.model = model
        self.name = f"openai-{model}"
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key)
        self.dim = 1536 if "small" in model else 3072 if "large" in model else 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            vectors.extend(d.embedding for d in resp.data)
        arr = np.asarray(vectors, dtype=np.float32)
        self.dim = arr.shape[1]
        return _normalise(arr)


def get_embedder(kind: str = "auto", model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None) -> Embedder:
    if kind not in ("auto", "openai", "hash"):
        raise ValueError(f"unknown embedder kind {kind!r}")
    if kind == "hash" or (kind == "auto" and not api_key):
        return HashEmbedder()
    if not api_key:
        raise ValueError("embedder 'openai' requires OPENAI_API_KEY")
    return OpenAIEmbedder(model=model, api_key=api_key)
