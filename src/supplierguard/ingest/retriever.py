"""Query-time retrieval over a saved index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .embeddings import Embedder
from .vectorstore import VectorStore


@dataclass(frozen=True)
class Hit:
    score: float
    chunk_id: str
    supplier_id: str
    supplier_name: str
    doc_type: str
    text: str


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    @classmethod
    def load(cls, index_dir, embedder: Embedder, use_faiss: Optional[bool] = None) -> "Retriever":
        store, info = VectorStore.load(index_dir, use_faiss=use_faiss)
        built_with = info.get("embedder")
        if built_with != embedder.name:
            # Mixing embedding models silently produces meaningless rankings, so fail loudly.
            raise ValueError(f"index was built with embedder {built_with!r} but {embedder.name!r} was supplied")
        return cls(store, embedder)

    def search(self, query: str, supplier_id: Optional[str] = None, k: int = 5,
               doc_types: Optional[Sequence[str]] = None) -> List[Hit]:
        qvec = self.embedder.embed([query])[0]
        return [Hit(score, m["chunk_id"], m["supplier_id"], m["supplier_name"], m["doc_type"], m["text"])
                for score, m in self.store.search(qvec, k=k, supplier_id=supplier_id, doc_types=doc_types)]


def format_context(hits: Sequence[Hit]) -> str:
    """Numbered context block for prompts, with chunk ids so answers can cite their sources."""
    return "\n\n".join(f"[{i}] ({h.chunk_id})\n{h.text}" for i, h in enumerate(hits, start=1))
