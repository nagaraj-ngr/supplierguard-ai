"""CLI: python -m supplierguard.ingest [--query "SOC 2 expired" --supplier SUP-001]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import get_settings
from .embeddings import get_embedder
from .pipeline import build_index
from .retriever import Retriever


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Build the PII-scrubbed vector index.")
    ap.add_argument("--data", type=Path, default=s.data_dir)
    ap.add_argument("--index", type=Path, default=s.index_dir)
    ap.add_argument("--embedder", choices=["auto", "openai", "hash"], default=s.embedder_kind)
    ap.add_argument("--no-scrub", action="store_true", help="disable PII scrubbing (not recommended)")
    ap.add_argument("--query", help="run a test query after building")
    ap.add_argument("--supplier", help="restrict the test query to one supplier id")
    args = ap.parse_args()

    embedder = get_embedder(args.embedder, s.embedding_model, s.openai_api_key)
    report = build_index(args.data, args.index, embedder=embedder, scrub_pii=not args.no_scrub)
    print(json.dumps(report, indent=2))
    if embedder.name.startswith("hash"):
        print("\nNote: the hash embedder is lexical only. Set OPENAI_API_KEY for semantic retrieval.")
    if args.query:
        for hit in Retriever.load(args.index, embedder).search(args.query, supplier_id=args.supplier, k=3):
            print(f"\n{hit.score:.3f}  {hit.chunk_id}\n{hit.text}")


if __name__ == "__main__":
    main()
