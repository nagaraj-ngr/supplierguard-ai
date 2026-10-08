from types import SimpleNamespace

import pytest

from supplierguard import scoring
from supplierguard.dimensions import DIMENSIONS, Dimension

D = Dimension


def _s(dim, score, conf="high", gap=False):
    return SimpleNamespace(dimension=dim, score=None if gap else score, confidence=conf, insufficient_data=gap)


@pytest.mark.parametrize("score,level", [
    (1, "LOW"), (3, "LOW"), (3.9, "LOW"), (4, "MEDIUM"), (6.99, "MEDIUM"),
    (7, "HIGH"), (8.9, "HIGH"), (9, "CRITICAL"), (10, "CRITICAL"),
])
def test_risk_bands_follow_the_design_document(score, level):
    assert scoring.risk_level_for(score) == level


@pytest.mark.parametrize("bad", [0, 0.99, 10.01, 11, -3, None, float("nan")])
def test_risk_level_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        scoring.risk_level_for(bad)


def test_equal_weight_mean():
    scores = {D.CYBERSECURITY: 8, D.FINANCIAL: 4, D.CONCENTRATION: 6}
    assert scoring.weighted_overall(scores) == 6.0


def test_custom_weights():
    scores = {D.CYBERSECURITY: 10, D.FINANCIAL: 2}
    assert scoring.weighted_overall(scores, {D.CYBERSECURITY: 3, D.FINANCIAL: 1}) == 8.0


def test_missing_dimensions_are_excluded_not_counted_as_zero():
    scores = {D.CYBERSECURITY: 8, D.FINANCIAL: None, D.CONCENTRATION: 6}
    assert scoring.weighted_overall(scores) == 7.0


def test_nothing_scored_gives_none():
    assert scoring.weighted_overall({D.CYBERSECURITY: None}) is None
    assert scoring.weighted_overall({}) is None
    assert scoring.weighted_overall({D.CYBERSECURITY: 5}, {D.CYBERSECURITY: 0}) is None


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        scoring.weighted_overall({D.CYBERSECURITY: 5}, {D.CYBERSECURITY: -1})


def test_top_concerns_sorted_by_severity_with_stable_ties():
    scores = {D.CYBERSECURITY: 7, D.FINANCIAL: 9, D.COMPLIANCE_LEGAL: 7, D.NEWS_REPUTATION: 5,
              D.DELIVERY_QUALITY: 2, D.CONCENTRATION: None}
    assert scoring.top_concerns(scores) == ["financial (9/10)", "cybersecurity (7/10)", "compliance_legal (7/10)"]


def test_low_risk_supplier_has_no_concerns():
    assert scoring.top_concerns({D.CYBERSECURITY: 2, D.FINANCIAL: 3}) == []


def test_report_fields_for_a_full_assessment():
    full = [_s(d, v) for d, v in zip(DIMENSIONS, [8, 6, 7, 9, 5, 7])]
    f = scoring.compute_report_fields(full)
    assert f["overall_risk_score"] == 7.0 and f["risk_level"] == "HIGH"
    assert f["data_gaps"] == [] and f["flags"] == []
    assert f["top_concerns"] == ["news_reputation (9/10)", "cybersecurity (8/10)", "compliance_legal (7/10)"]


def test_report_fields_flag_gaps_and_low_confidence():
    scores = [_s(D.CYBERSECURITY, 8, conf="low"), _s(D.FINANCIAL, 0, gap=True),
              _s(D.COMPLIANCE_LEGAL, 6), _s(D.NEWS_REPUTATION, 4), _s(D.DELIVERY_QUALITY, 2)]
    f = scoring.compute_report_fields(scores)
    assert f["data_gaps"] == ["financial", "concentration"]  # explicit gap + missing agent output
    assert "low confidence: cybersecurity" in f["flags"]
    assert "partial assessment: 4 of 6 dimensions scored" in f["flags"]
    assert f["overall_risk_score"] == 5.0 and f["risk_level"] == "MEDIUM"


def test_report_fields_when_nothing_could_be_assessed():
    f = scoring.compute_report_fields([_s(d, 0, gap=True) for d in DIMENSIONS])
    assert f["overall_risk_score"] is None and f["risk_level"] == "UNKNOWN"
    assert len(f["data_gaps"]) == 6 and f["top_concerns"] == []
