"""Approach 1: vanilla RAG. One retrieval over the document store, one LLM call, one report."""

from __future__ import annotations

from ..dimensions import DIMENSIONS
from ..llm import ChatModel, LLMError, Meter
from ..guardrails import GuardrailError
from ..sources import DataUnavailable, SupplierSources
from .common import (Assessment, assessment_system_prompt, ask_for_assessments, build_report, ground,
                     unavailable_report)

APPROACH = "vanilla_rag"

RETRIEVAL_QUERY = (
    "security certifications audit findings compliance legal filings contract terms "
    "service levels delivery performance financial risk"
)


def assess_rag(supplier_id: str, sources: SupplierSources, model: ChatModel, k: int = 12) -> Assessment:
    meter = Meter(model)
    name = sources.name(supplier_id)
    try:
        context = sources.documents(supplier_id, RETRIEVAL_QUERY, k=k)
    except DataUnavailable as exc:
        return Assessment(APPROACH, unavailable_report(name, f"Unable to assess: {exc}"), meter.stats)

    user = f"Supplier: {name}\n\nContext:\n{context}"
    try:
        scores, recs = ask_for_assessments(meter, assessment_system_prompt(DIMENSIONS), user, DIMENSIONS)
    except (GuardrailError, LLMError) as exc:
        meter.error(str(exc))
        return Assessment(APPROACH, unavailable_report(name, f"RAG assessment failed: {exc}"), meter.stats)

    scores, unsupported = ground(scores, context)
    report = build_report(name, scores, recommendations=recs)
    return Assessment(APPROACH, report, meter.stats, unsupported_evidence=unsupported)
