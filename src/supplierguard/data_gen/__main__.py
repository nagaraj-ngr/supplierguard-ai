"""CLI: python -m supplierguard.data_gen --out data --seed 42 [--pdf]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import get_settings
from .generator import generate_dataset


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Generate the synthetic SupplierGuard dataset.")
    ap.add_argument("--out", type=Path, default=s.data_dir)
    ap.add_argument("--seed", type=int, default=s.seed)
    ap.add_argument("-n", "--suppliers", type=int, default=50)
    ap.add_argument("--pdf", action="store_true", help="write documents as PDF instead of .txt")
    args = ap.parse_args()
    summary = generate_dataset(args.out, seed=args.seed, n=args.suppliers, pdf=args.pdf)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
