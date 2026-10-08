"""The three approaches compared in the project. Each takes (supplier_id, sources, model)."""

from .common import Assessment
from .multi_agent import assess_multi
from .rag import assess_rag
from .react import assess_react

APPROACHES = {
    "rag": assess_rag,
    "react": assess_react,
    "multi": assess_multi,
}

__all__ = ["APPROACHES", "Assessment", "assess_multi", "assess_rag", "assess_react"]
