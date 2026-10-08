import csv
import json
import re
from collections import defaultdict

import pytest

from _support import dataset_dir, fresh_dir, pearson, tree_hashes, truth
from supplierguard.data_gen import generate_dataset, negative_article_count
from supplierguard.data_gen.generator import DOC_TYPES, _rng
from supplierguard.dimensions import DIMENSIONS


def _csv(d, name):
    with (d / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scores(d, dim):
    return {sid: v["dimension_scores"][dim] for sid, v in truth(d)["suppliers"].items()}


def _doc(d, sid, doc_type):
    return (d / "documents" / sid / f"{doc_type}.txt").read_text(encoding="utf-8")


def _per_supplier_mean(rows, field):
    acc = defaultdict(list)
    for r in rows:
        acc[r["supplier_id"]].append(float(r[field]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _aligned(d, dim, per_supplier):
    s = _scores(d, dim)
    ids = sorted(s)
    return [s[i] for i in ids], [per_supplier[i] for i in ids]


def test_supplier_count_and_tier_mix():
    d = dataset_dir()
    t = truth(d)["suppliers"]
    assert len(t) == 50
    tiers = [v["tier"] for v in t.values()]
    assert (tiers.count("high"), tiers.count("medium"), tiers.count("low")) == (15, 20, 15)
    names = [v["name"] for v in t.values()]
    assert len(set(names)) == 50


def test_golden_split_is_stratified_and_complete():
    t = truth(dataset_dir())
    golden = set(t["golden_ids"])
    assert len(golden) == 20
    by_tier = defaultdict(int)
    for sid in golden:
        by_tier[t["suppliers"][sid]["tier"]] += 1
    assert dict(by_tier) == {"high": 6, "medium": 8, "low": 6}
    assert all(v["split"] in ("golden", "holdout") for v in t["suppliers"].values())
    assert sum(v["split"] == "golden" for v in t["suppliers"].values()) == 20


def test_scores_cover_all_six_dimensions_in_range():
    t = truth(dataset_dir())
    assert t["dimensions"] == [d.value for d in DIMENSIONS] and len(DIMENSIONS) == 6
    for v in t["suppliers"].values():
        assert set(v["dimension_scores"]) == set(t["dimensions"])
        assert all(isinstance(s, int) and 1 <= s <= 10 for s in v["dimension_scores"].values())


def test_tiers_are_ordered_by_mean_risk():
    t = truth(dataset_dir())["suppliers"]
    mean = {tier: sum(v["overall"] for v in t.values() if v["tier"] == tier) / sum(
        v["tier"] == tier for v in t.values()) for tier in ("high", "medium", "low")}
    assert mean["high"] > mean["medium"] > mean["low"]
    assert mean["high"] - mean["low"] > 3


def test_every_supplier_has_every_source():
    d = dataset_dir()
    ids = sorted(truth(d)["suppliers"])
    fin, dlv = _csv(d, "financials.csv"), _csv(d, "delivery.csv")
    dep = json.loads((d / "dependency.json").read_text())
    for sid in ids:
        for doc_type in DOC_TYPES:
            assert (d / "documents" / sid / f"{doc_type}.txt").is_file()
        news = json.loads((d / "news" / f"{sid}.json").read_text())
        assert len(news) == 8 and all({"article_id", "date", "headline", "body"} <= set(a) for a in news)
        assert sum(r["supplier_id"] == sid for r in fin) == 8
        assert sum(r["supplier_id"] == sid for r in dlv) == 12
        assert sum(r["supplier_id"] == sid for r in dep) == 1


def test_same_seed_is_byte_identical_and_different_seed_differs():
    a, b, c = fresh_dir(), fresh_dir(), fresh_dir()
    generate_dataset(a, seed=7, n=12)
    generate_dataset(b, seed=7, n=12)
    generate_dataset(c, seed=8, n=12)
    assert tree_hashes(a) == tree_hashes(b)
    assert tree_hashes(a) != tree_hashes(c)


def test_ground_truth_does_not_leak_into_source_data():
    d = dataset_dir()
    blob = "".join(p.read_text(encoding="utf-8") for p in d.rglob("*")
                   if p.is_file() and p.name != "ground_truth.json")
    assert "dimension_scores" not in blob and "risk_tier" not in blob and '"tier"' not in blob


def test_financial_data_tracks_ground_truth():
    d = dataset_dir()
    rows = _csv(d, "financials.csv")
    x, y = _aligned(d, "financial", _per_supplier_mean(rows, "debt_to_equity"))
    assert pearson(x, y) > 0.9
    x, y = _aligned(d, "financial", _per_supplier_mean(rows, "credit_score"))
    assert pearson(x, y) < -0.9
    growth = {}
    for sid in {r["supplier_id"] for r in rows}:
        rev = [float(r["revenue_musd"]) for r in rows if r["supplier_id"] == sid]
        growth[sid] = rev[-1] / rev[0] - 1
    x, y = _aligned(d, "financial", growth)
    assert pearson(x, y) < -0.9


def test_delivery_data_tracks_ground_truth():
    d = dataset_dir()
    rows = _csv(d, "delivery.csv")
    x, y = _aligned(d, "delivery_quality", _per_supplier_mean(rows, "on_time_pct"))
    assert pearson(x, y) < -0.9
    x, y = _aligned(d, "delivery_quality", _per_supplier_mean(rows, "defect_rate_pct"))
    assert pearson(x, y) > 0.9
    x, y = _aligned(d, "delivery_quality", _per_supplier_mean(rows, "sla_breaches"))
    assert pearson(x, y) > 0.85


def test_dependency_data_tracks_ground_truth():
    d = dataset_dir()
    dep = {r["supplier_id"]: r for r in json.loads((d / "dependency.json").read_text())}
    x, y = _aligned(d, "concentration", {k: v["share_of_category_spend_pct"] for k, v in dep.items()})
    assert pearson(x, y) > 0.9
    x, y = _aligned(d, "concentration", {k: v["qualified_alternative_suppliers"] for k, v in dep.items()})
    assert pearson(x, y) < -0.85
    assert all(v["single_source"] == (v["qualified_alternative_suppliers"] == 0) for v in dep.values())


def test_security_documents_track_ground_truth():
    d = dataset_dir()
    breaches = {sid: int(re.search(r"Confirmed security breaches \(last 3 years\): (\d+)",
                                   _doc(d, sid, "security_assessment")).group(1)) for sid in truth(d)["suppliers"]}
    x, y = _aligned(d, "cybersecurity", breaches)
    assert pearson(x, y) > 0.8
    open_crit = {sid: int(re.search(r"Open critical penetration-test findings: (\d+)",
                                    _doc(d, sid, "security_assessment")).group(1)) for sid in breaches}
    x, y = _aligned(d, "cybersecurity", open_crit)
    assert pearson(x, y) > 0.8


def test_compliance_and_legal_documents_track_ground_truth():
    d = dataset_dir()
    ids = truth(d)["suppliers"]
    major = {sid: int(re.search(r"Open major findings: (\d+)", _doc(d, sid, "compliance_audit")).group(1)) for sid in ids}
    x, y = _aligned(d, "compliance_legal", major)
    assert pearson(x, y) > 0.8
    active = {sid: int(re.search(r"Active lawsuits: (\d+)", _doc(d, sid, "legal_filings")).group(1)) for sid in ids}
    x, y = _aligned(d, "compliance_legal", active)
    assert pearson(x, y) > 0.8


def test_negative_news_volume_increases_with_score():
    means = []
    for score in range(1, 11):
        rng = _rng(0, "news-test", str(score))
        means.append(sum(negative_article_count(rng, score) for _ in range(400)) / 400)
    assert all(b >= a for a, b in zip(means, means[1:])), means
    assert means[0] < 1.5 and means[-1] > 5


def test_contracts_contain_pii_for_the_scrubber_to_find():
    text = _doc(dataset_dir(), "SUP-001", "contract")
    assert re.search(r"[\w.]+@[\w.-]+\.example", text)
    assert re.search(r"Account Number: \d{10,12}", text)
    assert re.search(r"IBAN: [A-Z]{2}\d{2}", text)
    assert re.search(r"Account Manager: [A-Z][a-z]+ [A-Z][a-z]+", text)


def test_rejects_more_suppliers_than_unique_names():
    with pytest.raises(ValueError, match="unique supplier names"):
        generate_dataset(fresh_dir(), n=10_000)


def test_pdf_output_is_extractable():
    pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    from supplierguard.ingest.documents import read_pdf
    d = fresh_dir()
    generate_dataset(d, seed=1, n=3, pdf=True)
    text = read_pdf(d / "documents" / "SUP-001" / "security_assessment.pdf")
    assert "Confirmed security breaches" in text and "SUP-001" in text
