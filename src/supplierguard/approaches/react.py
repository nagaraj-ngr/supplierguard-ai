"""Approach 2: ReAct single agent. One agent chooses tools in a loop until it submits a report."""

from __future__ import annotations

import json
from typing import List, Tuple

from ..dimensions import DIMENSIONS
from ..guardrails import GuardrailError, retry_until_valid
from ..llm import ChatModel, LLMError, Meter
from ..sources import DataUnavailable, SupplierSources
from .common import (ASSESSMENT_RULES, DIMENSION_NAMES, Assessment, build_report, ground, parse_assessments,
                     unavailable_report)

APPROACH = "react"
MAX_STEPS = 10

TOOL_HELP = {
    "search_documents": 'Search contracts, audits, security and legal documents. Input: {"query": "<text>"}',
    "get_financials": "Eight quarters of revenue, gross margin, debt to equity, current ratio, credit score. Input: {}",
    "get_delivery": "Twelve months of on-time %, defect rate, SLA breaches, quality score. Input: {}",
    "get_news": "Recent news articles about the supplier. Input: {}",
    "get_dependency": "Spend share, qualified alternatives, switching time, single-source flag. Input: {}",
}

SYSTEM = (
    "You are a supplier risk analyst. Work step by step: call one tool per turn to gather evidence, "
    "then submit your assessment. Assess all six dimensions: " + DIMENSION_NAMES + ".\n"
    + ASSESSMENT_RULES + "\n"
    "Reply with JSON only, in one of these two forms.\n"
    'To call a tool: {"action": "<tool name>", "input": {...}}\n'
    'To finish: {"action": "final", "dimensions": [...], "recommendations": [...]} where each dimension '
    'entry has the fields dimension, score, confidence, evidence, reasoning, insufficient_data.\n'
    "Tools:\n" + "\n".join(f"- {name}: {desc}" for name, desc in TOOL_HELP.items())
)

def parse_step(text: str) -> Tuple:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("output must be a JSON object")
    action = data.get("action")
    if action == "final":
        scores, recs = parse_assessments(text, DIMENSIONS)
        return ("final", scores, recs)
    if action in TOOL_HELP:
        args = data.get("input", {})
        if not isinstance(args, dict):
            raise ValueError('"input" must be a JSON object')
        return ("tool", action, args)
    raise ValueError(f'"action" must be "final" or one of: {", ".join(TOOL_HELP)}')


def run_tool(tool: str, args: dict, supplier_id: str, sources: SupplierSources) -> str:
    try:
        if tool == "search_documents":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return "error: search_documents needs a non-empty string query"
            return sources.documents(supplier_id, query)
        method = {"get_financials": sources.financials, "get_delivery": sources.delivery,
                  "get_news": sources.news, "get_dependency": sources.dependency}[tool]
        return method(supplier_id)
    except DataUnavailable as exc:
        return f"no data: {exc}"


def assess_react(supplier_id: str, sources: SupplierSources, model: ChatModel) -> Assessment:
    meter = Meter(model)
    name = sources.name(supplier_id)
    transcript: List[str] = [f"Supplier: {name} (id {supplier_id})"]
    observations: List[str] = []
    steps_taken = 0

    for _ in range(MAX_STEPS):
        user = "\n\n".join(transcript) + "\n\nNext action (JSON only):"
        try:
            step = retry_until_valid(
                lambda feedback: meter.complete(
                    SYSTEM, user + (f"\n\nYour previous reply was rejected: {feedback}" if feedback else "")),
                parse_step,
            )
        except (GuardrailError, LLMError) as exc:
            meter.error(str(exc))
            return Assessment(APPROACH, unavailable_report(name, f"ReAct agent failed: {exc}"), meter.stats)
        steps_taken += 1

        if step[0] == "final":
            scores, recs = step[1], step[2]
            context = "\n\n".join(observations)
            scores, unsupported = ground(scores, context)
            report = build_report(name, scores, recommendations=recs)
            return Assessment(APPROACH, report, meter.stats, unsupported_evidence=unsupported,
                              notes={"steps": str(steps_taken)})

        _, tool, args = step
        observation = run_tool(tool, args, supplier_id, sources)
        observations.append(f"[{tool}]\n{observation}")
        transcript.append(f"Action: {tool} {json.dumps(args)}\nObservation [{tool}]:\n{observation}")

    meter.error(f"no final report after {MAX_STEPS} steps")
    return Assessment(APPROACH,
                      unavailable_report(name, f"ReAct agent did not finish within {MAX_STEPS} steps"),
                      meter.stats, notes={"steps": str(steps_taken)})
