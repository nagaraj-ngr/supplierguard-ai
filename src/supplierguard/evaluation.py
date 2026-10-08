"""Evaluation: run an approach on the golden suppliers and compare against ground truth.

ground_truth.json is read here only. Agents never receive it.

CLI: python -m supplierguard.evaluation --approach multi [--limit 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .approaches import APPROACHES, Assessment
from .config import get_settings
from .dimensions import DIMENSIONS
from .llm import LLMError
from .schemas import SupplierRiskReport
from .service import default_model, load_sources


FAILURE_MARKERS = ("failed", "Unable to assess", "did not finish", "could not write")


def load_ground_truth(data_dir) -> Dict:
    return json.loads((Path(data_dir) / "ground_truth.json").read_text(encoding="utf-8"))


def score_report(report: SupplierRiskReport, truth: Dict) -> Dict:
    """Metrics for one supplier. Pure: no model or file access."""
    scored = [s for s in report.dimension_scores if not s.insufficient_data]
    errors = [abs(s.score - truth["dimension_scores"][s.dimension.value]) for s in scored]
    overall_error = None
    if report.overall_risk_score is not None and truth.get("overall") is not None:
        overall_error = abs(report.overall_risk_score - truth["overall"])
    return {
        "coverage": len(scored) / len(DIMENSIONS),
        "dimension_mae": statistics.fmean(errors) if errors else None,
        "overall_abs_error": overall_error,
        "risk_level": report.risk_level.value,
        "failed": any(marker in f for f in report.flags for marker in FAILURE_MARKERS),
    }


def aggregate(rows: List[Dict]) -> Dict:
    def mean_of(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    return {
        "suppliers": len(rows),
        "mean_coverage": mean_of("coverage"),
        "dimension_mae": mean_of("dimension_mae"),
        "overall_mae": mean_of("overall_abs_error"),
        "failed_runs": sum(1 for r in rows if r["failed"]),
        "unsupported_evidence": sum(r["unsupported_evidence"] for r in rows),
        "llm_calls_per_run": mean_of("llm_calls"),
        "input_tokens_per_run": mean_of("input_tokens"),
        "output_tokens_per_run": mean_of("output_tokens"),
        "seconds_per_run": mean_of("seconds"),
    }


def evaluate(approach: str, limit: Optional[int] = None, results_dir: Optional[Path] = None) -> Dict:
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}")
    settings = get_settings()
    sources = load_sources(settings)
    truth_file = load_ground_truth(settings.data_dir)
    model = default_model(settings)

    golden = truth_file["golden_ids"][:limit] if limit else truth_file["golden_ids"]
    rows: List[Dict] = []
    details: List[Dict] = []
    for sid in golden:
        assessment: Assessment = APPROACHES[approach](sid, sources, model)
        metrics = score_report(assessment.report, truth_file["suppliers"][sid])
        st = assessment.stats
        row = {**metrics, "unsupported_evidence": assessment.unsupported_evidence, "llm_calls": st.llm_calls,
               "input_tokens": st.input_tokens, "output_tokens": st.output_tokens, "seconds": round(st.seconds, 3)}
        rows.append(row)
        details.append({"supplier_id": sid, **row, "errors": st.errors})

    result = {"approach": approach, "model": model.name, "summary": aggregate(rows), "suppliers": details}
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{approach}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score an approach against the golden suppliers.")
    ap.add_argument("--approach", choices=list(APPROACHES), required=True)
    ap.add_argument("--limit", type=int, help="only the first N golden suppliers")
    args = ap.parse_args(argv)
    try:
        result = evaluate(args.approach, args.limit, results_dir=get_settings().data_dir.parent / "results")
    except (LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
