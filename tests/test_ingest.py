import json
import re

import numpy as np
import pytest

from _support import dataset_dir, fresh_dir
from supplierguard.ingest import HashEmbedder, Retriever, VectorStore, build_index, format_context

_CACHE = {}


def _built():
    if "idx" not in _CACHE:
        idx = fresh_dir("sg_index_")
        report = build_index(dataset_dir(), idx, embedder=HashEmbedder(), use_faiss=False)
        _CACHE["idx"], _CACHE["report"] = idx, report
    return _CACHE["idx"], _CACHE["report"]


def _all_chunk_text(idx):
    return "\n".join(m["text"] for m in json.loads((idx / "meta.json").read_text())["metas"])


def test_report_counts():
    _, r = _built()
    assert r["documents"] == 200 and r["chunks"] >= 200
    assert r["embedder"] == "hash-384" and r["backend"] == "numpy" and r["pii_scrubbed"]
    assert r["pii_redactions"] == {"email": 100, "phone": 100, "account": 100, "name": 300}


def test_no_pii_reaches_the_index():
    idx, _ = _built()
    text = _all_chunk_text(idx)
    assert not re.search(r"[\w.]+@[\w-]+\.example", text)
    assert not re.search(r"\+\d{1,3}[\s.-]?\d", text) and not re.search(r"\(\d{3}\) \d{3}-\d{4}", text)
    assert not re.search(r"(?:Account Number|IBAN): [A-Z0-9]", text)
    assert not re.search(
        r"(?:Prepared by|Security Contact|Account Manager|Lead Auditor|Counsel of record|Contract Signatory):"
        r"[ \t]*(?!\[NAME_REDACTED\])[A-Z][a-z]+ [A-Z]", text)


def test_business_facts_survive_scrubbing():
    text = _all_chunk_text(_built()[0])
    assert "SOC 2 Type II" in text and "Active lawsuits:" in text and "Open major findings:" in text


def test_every_chunk_carries_supplier_metadata():
    idx, _ = _built()
    metas = json.loads((idx / "meta.json").read_text())["metas"]
    for m in metas:
        assert re.fullmatch(r"SUP-\d{3}/\w+#\d+", m["chunk_id"])
        assert m["text"].startswith(f"[{m['supplier_name']} | {m['doc_type']}]")


def test_supplier_filter_isolates_one_supplier():
    idx, _ = _built()
    ret = Retriever.load(idx, HashEmbedder(), use_faiss=False)
    hits = ret.search("security breaches certifications", supplier_id="SUP-007", k=10)
    assert hits and all(h.supplier_id == "SUP-007" for h in hits)
    assert ret.search("anything", supplier_id="SUP-999") == []


def test_doc_type_filter():
    idx, _ = _built()
    ret = Retriever.load(idx, HashEmbedder(), use_faiss=False)
    hits = ret.search("lawsuits", supplier_id="SUP-003", k=10, doc_types=["legal_filings"])
    assert hits and {h.doc_type for h in hits} == {"legal_filings"}


def test_lexical_retrieval_finds_the_right_document_type():
    idx, _ = _built()
    ret = Retriever.load(idx, HashEmbedder(), use_faiss=False)
    queries = {
        "security_assessment": "SOC 2 Type II certification and security breaches",
        "legal_filings": "active lawsuits regulatory fines investigations",
        "compliance_audit": "audit findings corrective actions ISO 9001",
        "contract": "payment terms service levels on-time delivery commitment",
    }
    total = hits_ok = 0
    for sid in [f"SUP-{i:03d}" for i in range(1, 51)]:
        for doc_type, q in queries.items():
            total += 1
            top = ret.search(q, supplier_id=sid, k=3)
            hits_ok += any(h.doc_type == doc_type for h in top)
    assert hits_ok / total >= 0.95, f"hit@3 = {hits_ok}/{total}"


def test_scores_are_sorted_and_bounded():
    idx, _ = _built()
    hits = Retriever.load(idx, HashEmbedder(), use_faiss=False).search("security", k=8)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True) and all(-1.0001 <= s <= 1.0001 for s in scores)


def test_embedder_mismatch_is_rejected():
    idx, _ = _built()
    with pytest.raises(ValueError, match="built with embedder"):
        Retriever.load(idx, HashEmbedder(dim=128), use_faiss=False)


def test_format_context_is_numbered_and_cites_chunk_ids():
    idx, _ = _built()
    hits = Retriever.load(idx, HashEmbedder(), use_faiss=False).search("breaches", supplier_id="SUP-002", k=2)
    ctx = format_context(hits)
    assert ctx.startswith("[1] (SUP-002/") and "\n\n[2] (SUP-002/" in ctx


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder()
    a, b = e.embed(["SOC 2 expired", "lawsuit filed"]), e.embed(["SOC 2 expired", "lawsuit filed"])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert float(a[0] @ a[0]) > float(a[0] @ a[1])


def test_save_and_load_round_trip():
    idx, _ = _built()
    store, info = VectorStore.load(idx, use_faiss=False)
    assert info["embedder"] == "hash-384" and len(store) == _built()[1]["chunks"]
    out = fresh_dir()
    store.save(out, extra=info)
    again, _ = VectorStore.load(out, use_faiss=False)
    q = HashEmbedder().embed(["lawsuits"])[0]
    assert [m["chunk_id"] for _, m in store.search(q, 5)] == [m["chunk_id"] for _, m in again.search(q, 5)]


def test_store_validates_shapes():
    s = VectorStore(4, use_faiss=False)
    with pytest.raises(ValueError):
        s.add(np.zeros((2, 3), dtype=np.float32), [{}, {}])
    with pytest.raises(ValueError):
        s.add(np.zeros((2, 4), dtype=np.float32), [{}])
    assert s.search(np.zeros(4), 3) == []


def test_faiss_backend_matches_numpy():
    pytest.importorskip("faiss")
    idx, _ = _built()
    np_store, _ = VectorStore.load(idx, use_faiss=False)
    fa_store, _ = VectorStore.load(idx, use_faiss=True)
    assert fa_store.backend == "faiss"
    q = HashEmbedder().embed(["SOC 2 certification expired"])[0]
    assert [m["chunk_id"] for _, m in np_store.search(q, 10)] == [m["chunk_id"] for _, m in fa_store.search(q, 10)]


def test_unscrubbed_index_keeps_pii_so_the_flag_is_meaningful():
    idx = fresh_dir("sg_raw_")
    build_index(dataset_dir(), idx, embedder=HashEmbedder(), scrub_pii=False, use_faiss=False)
    assert re.search(r"[\w.]+@[\w-]+\.example", _all_chunk_text(idx))


def test_embedder_selection_never_needs_a_key_by_default():
    from supplierguard.ingest import get_embedder
    assert get_embedder("hash").name == "hash-384"
    assert get_embedder("auto", api_key=None).name == "hash-384"


def test_openai_embedder_requires_a_key():
    from supplierguard.ingest import get_embedder
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embedder("openai", api_key=None)
    with pytest.raises(ValueError, match="unknown embedder"):
        get_embedder("word2vec")
