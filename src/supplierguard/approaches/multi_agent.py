"""Approach 3: four specialist agents in parallel, coordinated by an orchestrator (LangGraph)."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Dict, List, Optional, Sequence, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from ..dimensions import Dimension
from ..guardrails import GuardrailError, retry_until_valid
from ..llm import ChatModel, LLMError, Meter
from ..schemas import RiskScore, SupplierRiskReport
from ..sources import DataUnavailable, SupplierSources
from .common import (Assessment, assessment_system_prompt, ask_for_assessments, build_report, gap, ground,
                     unavailable_report)

APPROACH = "multi_agent"

D = Dimension


@dataclass(frozen=True)
class AgentSpec:
    name: str
    dimensions: Tuple[Dimension, ...]
    sources: Tuple[str, ...]
    focus: str


AGENTS: Tuple[AgentSpec, ...] = (
    AgentSpec("document_intelligence", (D.CYBERSECURITY, D.COMPLIANCE_LEGAL), ("documents",),
              "security certifications, breach notification, audits, legal filings and contract terms"),
    AgentSpec("financial_operations", (D.FINANCIAL, D.DELIVERY_QUALITY), ("financials", "delivery"),
              "financial health trends and on-time delivery and quality"),
    AgentSpec("external_signals", (D.NEWS_REPUTATION,), ("news",),
              "news events, reputation and sentiment"),
    AgentSpec("strategic_risk", (D.CONCENTRATION,), ("dependency",),
              "spend share, qualified alternatives and switching time"),
)

DOCUMENT_QUERY = "security certification breach audit findings legal filings contract compliance"


class State(TypedDict):
    supplier_id: str
    supplier_name: str
    assessments: Annotated[List[Dict], operator.add]
    agent_flags: Annotated[List[str], operator.add]
    unsupported: Annotated[int, operator.add]
    report: Optional[SupplierRiskReport]


def gather(spec: AgentSpec, supplier_id: str, sources: SupplierSources) -> Tuple[str, List[str], bool]:
    """Context block for one agent, the sources that had no data, and whether any source had data."""
    blocks: List[str] = []
    missing: List[str] = []
    for src in spec.sources:
        try:
            if src == "documents":
                text = sources.documents(supplier_id, DOCUMENT_QUERY)
            else:
                text = {"financials": sources.financials, "delivery": sources.delivery,
                        "news": sources.news, "dependency": sources.dependency}[src](supplier_id)
            blocks.append(f"[{src}]\n{text}")
        except DataUnavailable as exc:
            missing.append(f"{src}: {exc}")
            blocks.append(f"[{src}]\nNo data available ({exc}).")
    return "\n\n".join(blocks), missing, len(missing) < len(spec.sources)


def _agent_node(spec: AgentSpec, sources: SupplierSources, meter: Meter):
    def run(state: State) -> Dict:
        context, missing, any_data = gather(spec, state["supplier_id"], sources)
        if not any_data:
            scores = [gap(d, f"No data for {spec.name}: {', '.join(missing)}") for d in spec.dimensions]
            return {"assessments": [{"agent": spec.name, "scores": scores}]}

        user = f"Supplier: {state['supplier_name']}\nFocus: {spec.focus}\n\nContext:\n{context}"
        try:
            scores, _ = ask_for_assessments(meter, assessment_system_prompt(spec.dimensions), user, spec.dimensions)
        except (GuardrailError, LLMError) as exc:
            meter.error(f"{spec.name}: {exc}")
            scores = [gap(d, f"{spec.name} failed: {exc}") for d in spec.dimensions]
            return {"assessments": [{"agent": spec.name, "scores": scores}],
                    "agent_flags": [f"agent failed: {spec.name}"]}

        scores, unsupported = ground(scores, context)
        return {"assessments": [{"agent": spec.name, "scores": scores}], "unsupported": unsupported}

    return run


def _recommendation_prompt(scores: Sequence[RiskScore]) -> str:
    lines = []
    for s in scores:
        value = "no data" if s.insufficient_data else f"{s.score}/10, confidence {s.confidence.value}"
        lines.append(f"- {s.dimension.value}: {value}. {s.reasoning}")
    return "\n".join(lines)


def _parse_recommendations(text: str) -> List[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {exc}") from exc
    recs = data.get("recommendations") if isinstance(data, dict) else None
    if not isinstance(recs, list) or not all(isinstance(r, str) for r in recs):
        raise ValueError('output must be {"recommendations": [strings]}')
    return [r.strip() for r in recs if r.strip()]


def _orchestrator_node(meter: Meter):
    system = (
        "You are the coordinator of a supplier risk review. Given the specialist assessments below, "
        "write 3 to 5 concrete actions for the procurement team, grounded only in those assessments. "
        'Reply with JSON only: {"recommendations": ["..."]}'
    )

    def run(state: State) -> Dict:
        scores = [s for a in state["assessments"] for s in a["scores"]]
        recs: List[str] = []
        flags = list(state["agent_flags"])
        scored = [s for s in scores if not s.insufficient_data]
        if scored:
            user = f"Supplier: {state['supplier_name']}\n\nAssessments:\n{_recommendation_prompt(scores)}"
            try:
                recs = retry_until_valid(
                    lambda fb: meter.complete(system, user + (f"\n\nRejected: {fb}" if fb else "")),
                    _parse_recommendations,
                )
            except (GuardrailError, LLMError) as exc:
                meter.error(f"orchestrator: {exc}")
                flags.append("orchestrator could not write recommendations")
        report = build_report(state["supplier_name"], scores, recommendations=recs, extra_flags=flags)
        return {"report": report}

    return run


def build_graph(sources: SupplierSources, meter: Meter):
    builder = StateGraph(State)
    for spec in AGENTS:
        builder.add_node(spec.name, _agent_node(spec, sources, meter))
    builder.add_node("orchestrator", _orchestrator_node(meter))
    for spec in AGENTS:
        builder.add_edge(START, spec.name)
    builder.add_edge([spec.name for spec in AGENTS], "orchestrator")
    builder.add_edge("orchestrator", END)
    return builder.compile()


def assess_multi(supplier_id: str, sources: SupplierSources, model: ChatModel) -> Assessment:
    meter = Meter(model)
    name = sources.name(supplier_id)
    graph = build_graph(sources, meter)
    try:
        final = graph.invoke({"supplier_id": supplier_id, "supplier_name": name, "assessments": [],
                              "agent_flags": [], "unsupported": 0, "report": None})
    except (GuardrailError, LLMError) as exc:
        meter.error(str(exc))
        return Assessment(APPROACH, unavailable_report(name, f"Multi-agent run failed: {exc}"), meter.stats)
    return Assessment(APPROACH, final["report"], meter.stats, unsupported_evidence=final["unsupported"])
