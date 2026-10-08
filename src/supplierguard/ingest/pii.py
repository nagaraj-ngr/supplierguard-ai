"""PII scrubbing applied to every document before it is chunked, embedded or indexed.

Detection is rule-based and deliberately conservative:
  * emails, international and US-style phone numbers, IBANs, labelled bank account and
    routing numbers: regular expressions.
  * person names: (a) the name that follows a role label such as "Account Manager:" and
    (b) honorific + name ("Dr. Smith"). An optional spaCy PERSON detector can be switched on.

Known limitations (documented, not hidden): free-text names without a label or honorific are
not caught by the default rules, and a label followed by a non-person capitalised phrase
would be redacted. The synthetic corpus only places personal names in labelled slots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict

EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,6}(?:\s?[A-Z0-9]{1,4})?\b")
ACCOUNT = re.compile(
    r"(?i)\b(account(?:\s+(?:no\.?|number))?|acct\.?|routing\s+number)(\s*[:#]\s*)(\d[\d\s-]{6,22}\d)"
)
PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{1,4}\)?(?:[\s.-]?\d{2,4}){2,4}(?!\w)")
PHONE_US = re.compile(r"(?<![\w+])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)")
# Names must stay on one line: [ \t] instead of \s, so a match can never run into the next label.
LABEL_NAME = re.compile(
    r"(\b(?:Account Manager|Contract Signatory|Prepared by|Lead Auditor|Counsel of record|"
    r"Security Contact|Authorized Signatory|Signed by)[ \t]*:[ \t]*)"
    r"([A-Z][a-z'\u2019-]+(?:[ \t]+[A-Z][a-z'\u2019-]+){1,2})"
)
HONORIFIC_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?[ \t]+[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?")

TOKENS = {
    "email": "[EMAIL_REDACTED]",
    "phone": "[PHONE_REDACTED]",
    "account": "[ACCOUNT_REDACTED]",
    "name": "[NAME_REDACTED]",
}


@dataclass
class ScrubResult:
    text: str
    counts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in TOKENS})

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _spacy_person_spans(text: str):
    """Optional NER backend. Returns [] when spaCy or its model is not installed."""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []
    return [(e.start_char, e.end_char) for e in nlp(text).ents if e.label_ == "PERSON"]


def scrub(text: str, use_spacy: bool = False) -> ScrubResult:
    result = ScrubResult(text=text)
    counts = result.counts

    def sub(pattern: re.Pattern, repl, kind: str, s: str) -> str:
        s, n = pattern.subn(repl, s)
        counts[kind] += n
        return s

    s = text
    s = sub(EMAIL, TOKENS["email"], "email", s)
    s = sub(IBAN, TOKENS["account"], "account", s)
    s = sub(ACCOUNT, lambda m: f"{m.group(1)}{m.group(2)}{TOKENS['account']}", "account", s)
    s = sub(PHONE_INTL, TOKENS["phone"], "phone", s)
    s = sub(PHONE_US, TOKENS["phone"], "phone", s)
    s = sub(LABEL_NAME, lambda m: f"{m.group(1)}{TOKENS['name']}", "name", s)
    s = sub(HONORIFIC_NAME, TOKENS["name"], "name", s)
    if use_spacy:
        for start, end in sorted(_spacy_person_spans(s), reverse=True):
            s = s[:start] + TOKENS["name"] + s[end:]
            counts["name"] += 1
    result.text = s
    return result
