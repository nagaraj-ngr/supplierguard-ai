"""Pure scoring logic (no Pydantic): risk bands, weighted aggregation, report fields.

Kept separate from schemas.py so it can be tested in isolation and reused by every
approach (vanilla RAG, ReAct, multi-agent) so they are all scored identically.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from .dimensions import DIMENSIONS, Dimension

LOW, MEDIUM, HIGH, CRITICAL, UNKNOWN = "LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"


def risk_level_for(score: float) -> str:
    """Band a 1..10 score using its integer part: 1-3 LOW, 4-6 MEDIUM, 7-8 HIGH, 9-10 CRITICAL.

    These are the bands from the design document. For fractional scores the integer part
    decides, so 3.9 is LOW and 4.0 is MEDIUM.
    """
    if score is None or (isinstance(score, float) and math.isnan(score)):
        raise ValueError("score must be a number")
    if not 1 <= score <= 10:
        raise ValueError(f"score must be between 1 and 10, got {score}")
    whole = math.floor(score)
    if whole <= 3:
        return LOW
    if whole <= 6:
        return MEDIUM
    if whole <= 8:
        return HIGH
    return CRITICAL


def weighted_overall(scores: Mapping[Dimension, Optional[int]],
                     weights: Optional[Mapping[Dimension, float]] = None) -> Optional[float]:
    """Weighted mean over dimensions that have a score. None if nothing is scored.

    Dimensions without data are excluded from both numerator and denominator so a data
    gap never silently drags the overall score toward zero or toward the middle.
    """
    total = weight_sum = 0.0
    for dim, score in scores.items():
        if score is None:
            continue
        w = 1.0 if weights is None else float(weights.get(dim, 1.0))
        if w < 0:
            raise ValueError("weights must be non-negative")
        total += w * score
        weight_sum += w
    if weight_sum == 0:
        return None
    return round(total / weight_sum, 2)


def top_concerns(scores: Mapping[Dimension, Optional[int]], n: int = 3, min_score: int = 4) -> List[str]:
    """Highest-scoring dimensions, most severe first. Dimensions below min_score are not concerns."""
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    ranked = sorted(((d, s) for d, s in scores.items() if s is not None and s >= min_score),
                    key=lambda ds: (-ds[1], order[ds[0]]))
    return [f"{d.value} ({s}/10)" for d, s in ranked[:n]]


def _value(x):
    return getattr(x, "value", x)


def compute_report_fields(dimension_scores: Sequence, weights: Optional[Mapping[Dimension, float]] = None) -> Dict:
    """Derive every aggregate field of a report from per-dimension scores.

    ``dimension_scores`` items need ``.dimension``, ``.score``, ``.confidence`` and
    ``.insufficient_data``. Returns kwargs for SupplierRiskReport.
    """
    by_dim = {s.dimension: s for s in dimension_scores}
    mapping = {d: (None if s.insufficient_data else s.score) for d, s in by_dim.items()}
    overall = weighted_overall(mapping, weights)
    gaps = [d.value for d in DIMENSIONS if d not in by_dim or by_dim[d].insufficient_data]
    flags = [f"low confidence: {d.value}" for d in DIMENSIONS
             if d in by_dim and not by_dim[d].insufficient_data and _value(by_dim[d].confidence) == "low"]
    if gaps:
        flags.append(f"partial assessment: {len(DIMENSIONS) - len(gaps)} of {len(DIMENSIONS)} dimensions scored")
    return {
        "overall_risk_score": overall,
        "risk_level": risk_level_for(overall) if overall is not None else UNKNOWN,
        "top_concerns": top_concerns(mapping),
        "data_gaps": gaps,
        "flags": flags,
    }
