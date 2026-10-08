"""Read-only access to supplier data for the agents.

Agents see only what these methods return. ground_truth.json is deliberately not read here:
it is for evaluation only and must never reach a prompt.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from .guardrails import resolve_supplier
from .ingest.retriever import Retriever, format_context


class DataUnavailable(LookupError):
    """No data of this kind exists for the supplier. Agents turn this into an insufficient-data gap."""


class SupplierSources:
    def __init__(self, data_dir, retriever: Optional[Retriever] = None):
        self.root = Path(data_dir)
        self.retriever = retriever
        self.suppliers: Dict[str, dict] = {
            s["id"]: s for s in json.loads((self.root / "suppliers.json").read_text(encoding="utf-8"))
        }
        self._financials = self._csv_by_supplier("financials.csv")
        self._delivery = self._csv_by_supplier("delivery.csv")
        self._dependency = {
            d["supplier_id"]: d for d in json.loads((self.root / "dependency.json").read_text(encoding="utf-8"))
        }

    def _csv_by_supplier(self, name: str) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        with (self.root / name).open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out.setdefault(row["supplier_id"], []).append(row)
        return out

    def resolve(self, query: str) -> str:
        names = [s["name"] for s in self.suppliers.values()]
        name = resolve_supplier(query, names)
        return next(sid for sid, s in self.suppliers.items() if s["name"] == name)

    def name(self, supplier_id: str) -> str:
        return self.suppliers[supplier_id]["name"]

    def financials(self, supplier_id: str) -> str:
        rows = self._financials.get(supplier_id)
        if not rows:
            raise DataUnavailable("no financial records")
        lines = ["quarter | revenue_musd | gross_margin_pct | debt_to_equity | current_ratio | credit_score"]
        lines += [
            f"{r['quarter']} | {r['revenue_musd']} | {r['gross_margin_pct']} | {r['debt_to_equity']} | "
            f"{r['current_ratio']} | {r['credit_score']}"
            for r in rows
        ]
        return "\n".join(lines)

    def delivery(self, supplier_id: str) -> str:
        rows = self._delivery.get(supplier_id)
        if not rows:
            raise DataUnavailable("no delivery records")
        lines = ["month | orders | on_time_pct | defect_rate_pct | sla_breaches | quality_score"]
        lines += [
            f"{r['month']} | {r['orders']} | {r['on_time_pct']} | {r['defect_rate_pct']} | "
            f"{r['sla_breaches']} | {r['quality_score']}"
            for r in rows
        ]
        return "\n".join(lines)

    def news(self, supplier_id: str) -> str:
        path = self.root / "news" / f"{supplier_id}.json"
        if not path.exists():
            raise DataUnavailable("no news articles")
        articles = sorted(json.loads(path.read_text(encoding="utf-8")), key=lambda a: a["date"], reverse=True)
        if not articles:
            raise DataUnavailable("no news articles")
        return "\n\n".join(f"{a['date']} | {a['source']} | {a['headline']}\n{a['body']}" for a in articles)

    def dependency(self, supplier_id: str) -> str:
        d = self._dependency.get(supplier_id)
        if d is None:
            raise DataUnavailable("no dependency record")
        return "\n".join(f"{k}: {v}" for k, v in d.items())

    def documents(self, supplier_id: str, query: str, k: int = 6) -> str:
        if self.retriever is None:
            raise DataUnavailable("no document index loaded")
        hits = self.retriever.search(query, supplier_id=supplier_id, k=k)
        if not hits:
            raise DataUnavailable("no matching document chunks")
        return format_context(hits)
