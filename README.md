# SupplierGuard AI

Agentic supplier risk intelligence. Given a supplier name, produce a structured, evidence-backed
risk scorecard across six dimensions: cybersecurity, financial, compliance/legal,
news/reputation, delivery/quality and concentration.

The project compares three approaches on the same data and metrics: vanilla RAG (baseline),
a ReAct single agent, and a LangGraph multi-agent system. See
[`docs/SupplierGuard_AI_Design_Doc.docx`](docs/SupplierGuard_AI_Design_Doc.docx) for the problem
statement, architecture, evaluation plan and framework justification.

## Status

| Step | Content | State |
|---|---|---|
| 1 | Scaffold, config | done |
| 2 | Deterministic synthetic data (50 suppliers) | done |
| 3 | PII scrubbing, chunking, embeddings, vector store, retrieval | done |
| 4 | Pydantic schemas, scoring, guardrails | done |
| 5 | Approach 1: vanilla RAG | not started |
| 6 | Approach 2: ReAct single agent | not started |
| 7 | Approach 3: LangGraph multi-agent | not started |
| 8 | Evaluation harness and comparison | not started |
| 9 | Error-handling tests | not started |
| 10 | Gradio UI, failure analysis | not started |

## Results

None yet. No approach has been implemented or evaluated, so this repository reports no
accuracy, faithfulness, cost or latency numbers. They will be added from real runs in step 8.

## Prerequisites

- **Python 3.10 or newer.** `pyproject.toml` sets `requires-python = ">=3.10"`, and
  `pip install -e` refuses older versions. The macOS system `python3` is 3.9 and will not work.
  Check with `python3 --version`. Install a newer one with `brew install python@3.12`, or with
  [`uv`](https://docs.astral.sh/uv/) (`uv python install 3.12`).
- **Git**, to clone the repository.
- **An OpenAI API key**, optional. Steps 1 to 4 work without one.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # any Python >= 3.10
python -m pip install --upgrade pip
pip install -e ".[dev,faiss,pdf]"
cp .env.example .env                 # add OPENAI_API_KEY once you have one
python -m supplierguard.data_gen     # optional: regenerates ./data identically
python -m supplierguard.ingest --query "SOC 2 certification expired" --supplier SUP-001
pytest
```

Steps 1 to 4 run without an API key. Without one, embeddings use an offline hash embedder.

The data generator is deterministic for a given seed, so running it should leave `git status`
clean. If it changes tracked files, the environment differs from the one that produced the data.

### Verifying the setup

1. The ingest command prints ranked chunks from `SUP-001` only. Personal details appear as
   placeholders such as `[NAME_REDACTED]` and `[EMAIL_REDACTED]`.
2. `pytest` reports all tests passing.

## What exists so far

**Data** (`src/supplierguard/data_gen`). Each supplier gets a latent risk score per dimension;
financials, delivery logs, news, dependency data and four documents are derived from it. The
tests check that the derived data really tracks the scores (for example, correlation between the
financial score and mean debt-to-equity is above 0.9).

**PII handling** (`ingest/pii.py`). Every document is scrubbed before chunking, so the vector
store and any LLM prompt never contain raw PII. Emails, phone numbers (international and US
formats), IBANs, labelled account and routing numbers, and names in labelled slots
(`Account Manager:`, `Prepared by:`, and so on) or after honorifics are replaced with
`[EMAIL_REDACTED]`, `[PHONE_REDACTED]`, `[ACCOUNT_REDACTED]` and `[NAME_REDACTED]`. An optional
spaCy PERSON detector can be enabled with `scrub(text, use_spacy=True)`.

**Retrieval** (`ingest/`). Paragraph-aware chunking with word-aligned overlap, a FAISS or NumPy
exact-search store with supplier and document-type filters, and a `Retriever` that refuses to
load an index built with a different embedder.

**Schemas and guardrails** (`schemas.py`, `scoring.py`, `guardrails.py`). Every approach must
return `SupplierRiskReport`, which is what makes them comparable. Guardrails: scores must be
integers from 1 to 10, a score without evidence is rejected, a dimension without data is an
explicit gap rather than a guess, unknown or injection-like supplier input is rejected before any
LLM call, and invalid model output is retried.

## Known limitations

- Ground truth comes from the generator's parameters. It is not human expert labelling, and
  results should be reported with that caveat.
- The hash embedder is lexical, not semantic. Retrieval quality numbers must come from the
  OpenAI embedder.
- Name detection is rule-based. Names that are not in a labelled slot or after an honorific are
  not caught by default, and a capitalised non-name after a label would be redacted.
- `check_evidence_grounded` is a token-overlap heuristic and deliberately conservative. It flags
  paraphrases and acronyms. Faithfulness is measured separately in step 8.
- Documents are `.txt` by default. `python -m supplierguard.data_gen --pdf` writes PDFs, which the
  ingestion step reads with `pypdf`.

## Layout

```
src/supplierguard/
  config.py  dimensions.py  scoring.py  schemas.py  guardrails.py
  data_gen/    synthetic data generator
  ingest/      pii, chunking, embeddings, vectorstore, pipeline, retriever
tests/         pytest suite
data/          generated dataset (see data/README.md)
docs/          design document
```
