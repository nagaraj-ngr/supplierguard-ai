"""Gradio UI: python -m supplierguard.ui  (needs the 'ui' extra: pip install -e ".[ui]")"""

from __future__ import annotations

from functools import lru_cache

from .assess import render_markdown
from .guardrails import InvalidSupplierQuery
from .llm import LLMError
from .service import assess_supplier, load_sources

APPROACH_CHOICES = [("Multi-agent (LangGraph)", "multi"), ("ReAct single agent", "react"),
                    ("Vanilla RAG baseline", "rag")]


@lru_cache(maxsize=1)
def _sources():
    return load_sources()


def assess_for_ui(supplier: str, approach: str) -> str:
    if not supplier or not supplier.strip():
        return "Enter a supplier name."
    try:
        assessment = assess_supplier(approach, supplier, sources=_sources())
    except (InvalidSupplierQuery, LLMError, ValueError) as exc:
        return f"**Could not assess:** {exc}"
    return render_markdown(assessment)


def build_app():
    import gradio as gr

    with gr.Blocks(title="SupplierGuard AI") as app:
        gr.Markdown("# SupplierGuard AI\nEvidence-backed supplier risk scorecards. Scores run 1 (low) to 10 (high).")
        with gr.Row():
            supplier = gr.Textbox(label="Supplier name", placeholder="e.g. Quarry Chemicals", scale=3)
            approach = gr.Dropdown(choices=APPROACH_CHOICES, value="multi", label="Approach", scale=2)
        run = gr.Button("Assess supplier", variant="primary")
        output = gr.Markdown()
        run.click(assess_for_ui, inputs=[supplier, approach], outputs=output)
    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
