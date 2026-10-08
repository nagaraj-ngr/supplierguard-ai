"""Pydantic models for agent outputs and the final report.

Every approach (vanilla RAG, ReAct, multi-agent) must return these types, which is what
makes the three comparable. Validation here is the "score range enforcement" guardrail:
an out-of-range or evidence-free score raises ValidationError, and the caller retries.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import scoring
from .dimensions import Dimension

Score = Annotated[int, Field(strict=True, ge=1, le=10)]


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class RiskScore(BaseModel):
    """One agent's assessment of one risk dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    score: Optional[Score] = None
    confidence: Confidence
    evidence: List[str] = Field(default_factory=list)
    reasoning: str = ""
    insufficient_data: bool = False

    @field_validator("evidence")
    @classmethod
    def _drop_blank_evidence(cls, v: List[str]) -> List[str]:
        return [e.strip() for e in v if e and e.strip()]

    @model_validator(mode="after")
    def _evidence_or_gap(self) -> "RiskScore":
        if self.insufficient_data:
            if self.score is not None:
                raise ValueError("insufficient_data=True requires score to be null")
        else:
            if self.score is None:
                raise ValueError("score is required unless insufficient_data=True")
            if not self.evidence:
                raise ValueError("a score must cite at least one piece of evidence")
        return self


class SupplierRiskReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_name: str = Field(min_length=1)
    overall_risk_score: Optional[float] = Field(default=None, ge=1, le=10)
    risk_level: RiskLevel
    dimension_scores: List[RiskScore]
    top_concerns: List[str] = Field(default_factory=list, max_length=3)
    recommendations: List[str] = Field(default_factory=list)
    data_gaps: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "SupplierRiskReport":
        dims = [s.dimension for s in self.dimension_scores]
        if len(dims) != len(set(dims)):
            raise ValueError("each dimension may appear only once")
        scored = [s for s in self.dimension_scores if not s.insufficient_data]
        if not scored:
            if self.overall_risk_score is not None or self.risk_level != RiskLevel.UNKNOWN:
                raise ValueError("with no scored dimension, overall score must be null and level UNKNOWN")
        else:
            if self.overall_risk_score is None:
                raise ValueError("overall_risk_score is required when any dimension is scored")
            if self.risk_level.value != scoring.risk_level_for(self.overall_risk_score):
                raise ValueError("risk_level does not match overall_risk_score")
        expected_gaps = {s.dimension.value for s in self.dimension_scores if s.insufficient_data}
        if set(self.data_gaps) != expected_gaps:
            raise ValueError("data_gaps must list exactly the dimensions marked insufficient_data")
        return self

    @classmethod
    def from_scores(cls, supplier_name: str, scores: Sequence[RiskScore],
                    recommendations: Optional[Sequence[str]] = None,
                    weights: Optional[Mapping[Dimension, float]] = None) -> "SupplierRiskReport":
        """Build a consistent report. Any dimension with no score becomes an explicit data gap."""
        by_dim = {}
        for s in scores:
            if s.dimension in by_dim:
                raise ValueError(f"duplicate score for dimension {s.dimension.value}")
            by_dim[s.dimension] = s
        full = [
            by_dim.get(d) or RiskScore(dimension=d, confidence=Confidence.LOW, insufficient_data=True,
                                       reasoning="No assessment was produced for this dimension.")
            for d in Dimension
        ]
        return cls(supplier_name=supplier_name, dimension_scores=full,
                   recommendations=list(recommendations or []),
                   **scoring.compute_report_fields(full, weights))
