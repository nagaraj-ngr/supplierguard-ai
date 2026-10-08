"""Pieces shared by all three approaches: output contract, parsing, grounding, report assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from ..dimensions import Dimension
from ..guardrails import EVIDENCE_ONLY_INSTRUCTION, check_evidence_grounded, retry_until_valid
from ..llm import Meter, RunStats
from ..schemas import Confidence, RiskScore, SupplierRiskReport

DIMENSION_NAMES = ", ".join(d.value for d in Dimension)

ASSESSMENT_FORMAT = (
    '{"dimensions": [{"dimension": "<one of the names listed>", "score": <integer 1-10 or null>, '
    '"confidence": "low" | "medium" | "high", "evidence": ["<fact from the context, cite its [id]>"], '
    '"reasoning": "<one or two sentences>", "insufficient_data": true | false}], '
    '"recommendations": ["<action for the procurement team>"]}'
)

ASSESSMENT_RULES = (
    "Scores run from 1 (lowest risk) to 10 (highest risk). "
    "If the context does not contain the data needed for a dimension, set insufficient_data to true "
    "and score to null. Otherwise give an integer score and at least one evidence item. "
    "Each evidence item must quote or closely paraphrase the context."
)


def assessment_system_prompt(dimensions: Sequence[Dimension]) -> str:
    names = ", ".join(d.value for d in dimensions)
    return (
        "You are a supplier risk analyst assessing one supplier for a procurement team.\n"
        f"{EVIDENCE_ONLY_INSTRUCTION}\n{ASSESSMENT_RULES}\n"
        f"Assess these dimensions only: {names}.\n"
        f"Respond with JSON only, in this form: {ASSESSMENT_FORMAT}"
    )


def parse_assessments(text: str, required: Sequence[Dimension]) -> Tuple[List[RiskScore], List[str]]:
    """Strict parser. Raises ValueError with a message that is fed back to the model on retry."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("dimensions"), list):
        raise ValueError('output must be a JSON object with a "dimensions" list')
    recs = data.get("recommendations", [])
    if not isinstance(recs, list) or not all(isinstance(r, str) for r in recs):
        raise ValueError('"recommendations" must be a list of strings')
    allowed = set(required)
    out: List[RiskScore] = []
    seen = set()
    for item in data["dimensions"]:
        try:
            score = RiskScore.model_validate(item)
        except ValidationError as exc:
            raise ValueError(f"invalid dimension entry: {exc.errors()[0]['msg']}") from exc
        if score.dimension not in allowed:
            raise ValueError(f"dimension {score.dimension.value} is not one of the requested dimensions")
        if score.dimension in seen:
            raise ValueError(f"dimension {score.dimension.value} appears more than once")
        seen.add(score.dimension)
        out.append(score)
    missing = [d.value for d in required if d not in seen]
    if missing:
        raise ValueError(f"missing dimensions: {', '.join(missing)}")
    return out, [r.strip() for r in recs if r.strip()]


def ground(scores: Iterable[RiskScore], context: str) -> Tuple[List[RiskScore], int]:
    """Lower confidence to low where evidence is not found in the context. Returns the unsupported count."""
    out: List[RiskScore] = []
    unsupported = 0
    for s in scores:
        if s.insufficient_data or not s.evidence:
            out.append(s)
            continue
        bad = check_evidence_grounded(s.evidence, context)
        if bad:
            unsupported += len(bad)
            s = s.model_copy(update={"confidence": Confidence.LOW})
        out.append(s)
    return out, unsupported


def ask_for_assessments(meter: Meter, system: str, user: str,
                        required: Sequence[Dimension]) -> Tuple[List[RiskScore], List[str]]:
    def call(feedback: Optional[str]) -> str:
        suffix = f"\n\nYour previous answer was rejected: {feedback}. Reply with corrected JSON only." if feedback else ""
        return meter.complete(system, user + suffix)

    return retry_until_valid(call, lambda text: parse_assessments(text, required))


def gap(dimension: Dimension, reason: str) -> RiskScore:
    return RiskScore(dimension=dimension, confidence=Confidence.LOW, insufficient_data=True, reasoning=reason)


@dataclass
class Assessment:
    approach: str
    report: SupplierRiskReport
    stats: RunStats
    unsupported_evidence: int = 0
    notes: Dict[str, str] = field(default_factory=dict)


def build_report(supplier_name: str, scores: Sequence[RiskScore], recommendations: Optional[Sequence[str]] = None,
                 extra_flags: Sequence[str] = ()) -> SupplierRiskReport:
    report = SupplierRiskReport.from_scores(supplier_name, scores, recommendations=recommendations)
    if not extra_flags:
        return report
    return report.model_copy(update={"flags": list(report.flags) + list(extra_flags)})


def unavailable_report(supplier_name: str, reason: str) -> SupplierRiskReport:
    """Used when no assessment could be made at all. Every dimension is an explicit gap."""
    return build_report(supplier_name, [], recommendations=[], extra_flags=[reason])
