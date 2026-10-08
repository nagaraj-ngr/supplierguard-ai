"""Shared test helpers (no fixtures, so they work under any runner)."""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from supplierguard.data_gen import generate_dataset


def fresh_dir(prefix: str = "sg_") -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


@lru_cache(maxsize=None)
def dataset_dir(seed: int = 42, n: int = 50) -> Path:
    d = fresh_dir(f"sg_data_{seed}_{n}_")
    generate_dataset(d, seed=seed, n=n)
    return d


def truth(d: Path) -> dict:
    return json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
