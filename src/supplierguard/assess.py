"""CLI: python -m supplierguard.assess --supplier "Quarry Chemicals" --approach multi"""

from __future__ import annotations

import argparse
import json
import sys

from .approaches import APPROACHES, Assessment
from .guardrails import InvalidSupplierQuery
from .llm import LLMError
from .schemas import SupplierRiskReport
from .service import assess_supplier


def render_markdown(assessment: Assessment) -> str:
    r: SupplierRiskReport = assessment.report
    score = "n/a" if r.overall_risk_score is None else f"{r.overall_risk_score:.2f}/10"
    lines = [
        f"## {r.supplier_name}",
        f"**Approach:** {assessment.approach}  |  **Overall:** {score}  |  **Risk level:** {r.risk_level.value}",
        "",
        "| Dimension | Score | Confidence | Reasoning |",
        "|---|---|---|---|",
    ]
    for s in r.dimension_scores:
        value = "gap" if s.insufficient_data else str(s.score)
        lines.append(f"| {s.dimension.value} | {value} | {s.confidence.value} | {s.reasoning} |")
    if r.top_concerns:
        lines += ["", "**Top concerns:** " + "; ".join(r.top_concerns)]
    if r.recommendations:
        lines += ["", "**Recommendations**"] + [f"- {x}" for x in r.recommendations]
    if r.flags:
        lines += ["", "**Flags**"] + [f"- {x}" for x in r.flags]
    st = assessment.stats
    lines += ["", f"LLM calls: {st.llm_calls}, tokens in/out: {st.input_tokens}/{st.output_tokens}, "
                  f"seconds: {st.seconds:.1f}, unsupported evidence items: {assessment.unsupported_evidence}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assess one supplier's risk.")
    ap.add_argument("--supplier", required=True, help="supplier name (typos are corrected)")
    ap.add_argument("--approach", choices=list(APPROACHES), default="multi")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    try:
        assessment = assess_supplier(args.approach, args.supplier)
    except (InvalidSupplierQuery, LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(assessment.report.model_dump(mode="json"), indent=2))
    else:
        print(render_markdown(assessment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
