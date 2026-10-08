"""A small vector store with metadata filtering.

Uses FAISS (IndexFlatIP, exact search) when installed, otherwise a NumPy matrix product.
Both return identical rankings for normalised vectors. At this project's scale (about 500
chunks) exact search is the right choice, so no approximate index is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised only where faiss is installed
    faiss = None


class VectorStore:
    def __init__(self, dim: int, use_faiss: Optional[bool] = None):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self.metas: List[dict] = []
        want = (faiss is not None) if use_faiss is None else use_faiss
        if want and faiss is None:
            raise RuntimeError("use_faiss=True but faiss is not installed (pip install faiss-cpu)")
        self._index = faiss.IndexFlatIP(dim) if want else None

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def __len__(self) -> int:
        return len(self.metas)

    def add(self, vectors: np.ndarray, metas: Sequence[dict]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(vectors) != len(metas):
            raise ValueError("vectors and metas must have the same length")
        self._vectors = np.vstack([self._vectors, vectors])
        self.metas.extend(dict(m) for m in metas)
        if self._index is not None:
            self._index.add(vectors)

    def search(self, query: np.ndarray, k: int = 5, supplier_id: Optional[str] = None,
               doc_types: Optional[Sequence[str]] = None) -> List[Tuple[float, dict]]:
        """Top-k chunks by cosine similarity, optionally restricted to one supplier / doc types."""
        n = len(self.metas)
        if n == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query has dim {q.shape[1]}, store has dim {self.dim}")
        if self._index is not None:
            scores, ids = self._index.search(q, n)
            ranked = list(zip(ids[0].tolist(), scores[0].tolist()))
        else:
            sims = self._vectors @ q[0]
            order = np.argsort(-sims, kind="stable")
            ranked = [(int(i), float(sims[i])) for i in order]
        hits: List[Tuple[float, dict]] = []
        for i, score in ranked:
            if i < 0:
                continue
            meta = self.metas[i]
            if supplier_id is not None and meta["supplier_id"] != supplier_id:
                continue
            if doc_types is not None and meta["doc_type"] not in doc_types:
                continue
            hits.append((float(score), meta))
            if len(hits) == k:
                break
        return hits

    def save(self, directory, extra: Optional[Dict] = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._vectors)
        payload = {"dim": self.dim, "metas": self.metas, **(extra or {})}
        (d / "meta.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory, use_faiss: Optional[bool] = None) -> Tuple["VectorStore", Dict]:
        d = Path(directory)
        payload = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        store = cls(payload["dim"], use_faiss=use_faiss)
        vectors = np.load(d / "vectors.npy")
        if len(vectors):
            store.add(vectors, payload["metas"])
        return store, {k: v for k, v in payload.items() if k not in ("dim", "metas")}
