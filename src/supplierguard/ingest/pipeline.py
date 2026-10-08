"""Ingestion pipeline: load -> scrub PII -> chunk -> embed -> index -> save."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .chunking import chunk_text
from .documents import load_documents
from .embeddings import Embedder, get_embedder
from .pii import TOKENS, scrub
from .vectorstore import VectorStore


def build_index(data_dir, index_dir, embedder: Optional[Embedder] = None, scrub_pii: bool = True,
                max_chars: int = 800, overlap: int = 120, use_faiss: Optional[bool] = None) -> dict:
    embedder = embedder or get_embedder("hash")
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"no documents found under {Path(data_dir) / 'documents'}")

    pii_counts = {k: 0 for k in TOKENS}
    texts, metas = [], []
    for doc in docs:
        body = doc.text
        if scrub_pii:
            result = scrub(body)
            body = result.text
            for k, v in result.counts.items():
                pii_counts[k] += v
        for i, chunk in enumerate(chunk_text(body, max_chars=max_chars, overlap=overlap)):
            text = f"[{doc.supplier_name} | {doc.doc_type}]\n{chunk}"
            texts.append(text)
            metas.append({
                "chunk_id": f"{doc.doc_id}#{i}",
                "supplier_id": doc.supplier_id,
                "supplier_name": doc.supplier_name,
                "doc_type": doc.doc_type,
                "chunk_index": i,
                "text": text,
            })

    vectors = embedder.embed(texts)
    store = VectorStore(vectors.shape[1], use_faiss=use_faiss)
    store.add(vectors, metas)
    report = {
        "documents": len(docs),
        "chunks": len(metas),
        "pii_scrubbed": scrub_pii,
        "pii_redactions": pii_counts,
        "embedder": embedder.name,
        "dim": int(vectors.shape[1]),
        "backend": store.backend,
        "max_chars": max_chars,
        "overlap": overlap,
    }
    store.save(index_dir, extra={"embedder": embedder.name, "report": report})
    (Path(index_dir) / "ingest_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
