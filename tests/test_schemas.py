import pytest

pydantic = pytest.importorskip("pydantic")

from supplierguard.dimensions import DIMENSIONS, Dimension  # noqa: E402
from supplierguard.schemas import Confidence, RiskLevel, RiskScore, SupplierRiskReport  # noqa: E402

D = Dimension


def good(dim=D.CYBERSECURITY, score=8, **kw):
    return RiskScore(dimension=dim, score=score, confidence=Confidence.HIGH,
                     evidence=["SOC 2 Type II expired on 2026-04-24"], reasoning="Expired certification.", **kw)


def gap(dim):
    return RiskScore(dimension=dim, confidence=Confidence.LOW, insufficient_data=True, reasoning="No data.")


def test_valid_score_accepted():
    s = good()
    assert s.score == 8 and s.dimension is D.CYBERSECURITY


@pytest.mark.parametrize("bad", [0, 11, -1, 100])
def test_out_of_range_scores_rejected(bad):
    with pytest.raises(pydantic.ValidationError):
        good(score=bad)


@pytest.mark.parametrize("bad", [7.5, "7", True, None])
def test_non_integer_scores_rejected(bad):
    with pytest.raises(pydantic.ValidationError):
        good(score=bad)


def test_score_without_evidence_rejected():
    with pytest.raises(pydantic.ValidationError, match="evidence"):
        RiskScore(dimension=D.FINANCIAL, score=5, confidence=Confidence.MEDIUM, evidence=["  ", ""])


def test_insufficient_data_must_not_carry_a_score():
    with pytest.raises(pydantic.ValidationError):
        RiskScore(dimension=D.FINANCIAL, score=5, confidence=Confidence.LOW, insufficient_data=True)
    assert gap(D.FINANCIAL).score is None


def test_unknown_fields_and_dimensions_rejected():
    with pytest.raises(pydantic.ValidationError):
        RiskScore(dimension="geopolitical", score=5, confidence="low", evidence=["x"])
    with pytest.raises(pydantic.ValidationError):
        RiskScore(dimension=D.FINANCIAL, score=5, confidence="low", evidence=["x"], surprise=1)


def test_model_validates_json_from_an_llm():
    s = RiskScore.model_validate_json(
        '{"dimension":"financial","score":9,"confidence":"high","evidence":["debt to equity 2.9"],"reasoning":"r"}')
    assert s.score == 9
    with pytest.raises(pydantic.ValidationError):
        RiskScore.model_validate_json('{"dimension":"financial","score":"nine","confidence":"high","evidence":["x"]}')


def test_full_report_from_scores():
    scores = [good(d, v) for d, v in zip(DIMENSIONS, [8, 6, 7, 9, 5, 7])]
    r = SupplierRiskReport.from_scores("Quarry Chemicals", scores, recommendations=["Schedule a review"])
    assert r.overall_risk_score == 7.0 and r.risk_level is RiskLevel.HIGH
    assert r.data_gaps == [] and len(r.dimension_scores) == 6 and len(r.top_concerns) == 3


def test_missing_dimensions_become_explicit_gaps():
    r = SupplierRiskReport.from_scores("X", [good(D.CYBERSECURITY, 8), good(D.FINANCIAL, 4)])
    assert set(r.data_gaps) == {"compliance_legal", "news_reputation", "delivery_quality", "concentration"}
    assert r.overall_risk_score == 6.0 and any(f.startswith("partial assessment") for f in r.flags)


def test_nothing_assessable_gives_unknown():
    r = SupplierRiskReport.from_scores("X", [gap(d) for d in DIMENSIONS])
    assert r.overall_risk_score is None and r.risk_level is RiskLevel.UNKNOWN


def test_duplicate_dimension_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        SupplierRiskReport.from_scores("X", [good(D.CYBERSECURITY, 8), good(D.CYBERSECURITY, 3)])


def test_inconsistent_report_rejected():
    scores = [good(d, 8) for d in DIMENSIONS]
    with pytest.raises(pydantic.ValidationError, match="risk_level"):
        SupplierRiskReport(supplier_name="X", overall_risk_score=8.0, risk_level=RiskLevel.LOW,
                           dimension_scores=scores, data_gaps=[])
    with pytest.raises(pydantic.ValidationError, match="data_gaps"):
        SupplierRiskReport(supplier_name="X", overall_risk_score=8.0, risk_level=RiskLevel.HIGH,
                           dimension_scores=scores, data_gaps=["financial"])


def test_json_round_trip():
    r = SupplierRiskReport.from_scores("Quarry Chemicals", [good(d, 6) for d in DIMENSIONS])
    assert SupplierRiskReport.model_validate_json(r.model_dump_json()) == r
