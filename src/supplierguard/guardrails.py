"""Guardrails that sit around the LLM. Pure Python so they are cheap to test exhaustively.

  * resolve_supplier: only known suppliers are assessed; injection-style input is rejected.
  * check_evidence_grounded: heuristic check that cited evidence appears in retrieved context.
  * retry_until_valid: reject-and-retry loop for invalid model output.

check_evidence_grounded is a token-overlap heuristic, not a proof of faithfulness. The
evaluation step measures faithfulness properly with RAGAs and an LLM judge.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")

# Instruction injected into every agent system prompt (from the design document).
EVIDENCE_ONLY_INSTRUCTION = (
    "Score only based on the provided context. If the information is not in the context, "
    "state that data is unavailable and set insufficient_data to true. Never guess a score."
)

MAX_QUERY_CHARS = 100

INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier)",
    r"(?:reveal|show|print|repeat|leak) (?:your |the )?(?:system )?(?:prompt|instructions)",
    r"system prompt",
    r"you are now\b",
    r"\bact as\b",
    r"list (?:all )?(?:the )?files",
    r"<\s*/?\s*script",
    r"(?:drop|delete|truncate)\s+table",
    r"[;&|`$]\s*(?:rm|curl|wget|cat|ls|sudo)\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


class GuardrailError(Exception):
    """Base class for guardrail violations."""


class InvalidSupplierQuery(GuardrailError, ValueError):
    """The input is not a usable supplier name."""


class UnknownSupplier(InvalidSupplierQuery):
    """The input looks like a name, but no known supplier matches it."""


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def resolve_supplier(query: str, known_names: Sequence[str], cutoff: float = 0.85) -> str:
    """Map user input to exactly one known supplier name, or raise.

    Accepts exact matches (case and punctuation insensitive) and close typos. Rejects empty,
    over-long or injection-like input before any LLM call is made.
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidSupplierQuery("supplier name is empty")
    if len(query) > MAX_QUERY_CHARS or "\n" in query:
        raise InvalidSupplierQuery("input is not a plausible supplier name")
    if _INJECTION.search(query):
        raise InvalidSupplierQuery("input looks like an instruction, not a supplier name")
    by_norm = {_norm(n): n for n in known_names}
    q = _norm(query)
    if q in by_norm:
        return by_norm[q]
    close = difflib.get_close_matches(q, list(by_norm), n=2, cutoff=cutoff)
    if len(close) == 1:
        return by_norm[close[0]]
    if len(close) > 1:
        raise UnknownSupplier(f"ambiguous supplier name; did you mean one of: {', '.join(by_norm[c] for c in close)}?")
    raise UnknownSupplier(f"no supplier matches {query!r}")


_WORD = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) > 2 or t.isdigit()]


def check_evidence_grounded(evidence: Iterable[str], context: str, min_overlap: float = 0.7) -> List[str]:
    """Return the evidence items that are NOT supported by the context.

    An item is supported if it appears verbatim (case-insensitive) or if at least
    ``min_overlap`` of its content tokens occur in the context. Numbers must match exactly,
    so a fabricated figure fails even when the surrounding words are real.
    """
    ctx_norm = " ".join(_WORD.findall(context.lower()))
    ctx_tokens = set(_content_tokens(context))
    unsupported = []
    for item in evidence:
        norm = " ".join(_WORD.findall(item.lower()))
        if not norm:
            unsupported.append(item)
            continue
        if norm in ctx_norm:
            continue
        tokens = _content_tokens(item)
        numbers = [t for t in tokens if t.isdigit()]
        if not tokens or any(n not in ctx_tokens for n in numbers):
            unsupported.append(item)
            continue
        if sum(t in ctx_tokens for t in tokens) / len(tokens) < min_overlap:
            unsupported.append(item)
    return unsupported


def retry_until_valid(call: Callable[[Optional[str]], object], parse: Callable[[object], T],
                      max_attempts: int = 3) -> T:
    """Call the model, parse its output, and on ValueError retry with the error as feedback.

    ``call`` receives the previous validation error (None on the first attempt) so it can
    append it to the prompt. pydantic.ValidationError subclasses ValueError, so schema
    failures are retried too. After max_attempts the last error is raised as GuardrailError.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    feedback: Optional[str] = None
    for _ in range(max_attempts):
        raw = call(feedback)
        try:
            return parse(raw)
        except ValueError as exc:
            feedback = str(exc)
    raise GuardrailError(f"output still invalid after {max_attempts} attempts: {feedback}")
