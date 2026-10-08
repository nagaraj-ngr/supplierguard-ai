import pytest

from supplierguard.ingest.pii import scrub


def test_email_redacted():
    r = scrub("Write to priya.raman@brightline-components.example today.")
    assert "@" not in r.text and "[EMAIL_REDACTED]" in r.text and r.counts["email"] == 1


@pytest.mark.parametrize("phone", [
    "+1-415-555-0142", "+44 20 7946 0958", "+91 80 4155 0142", "(415) 555-0142", "415-555-0142",
])
def test_phone_formats_redacted(phone):
    r = scrub(f"Call {phone} for details.")
    assert "[PHONE_REDACTED]" in r.text and "555" not in r.text and r.counts["phone"] == 1


@pytest.mark.parametrize("text", [
    "Audit date: 2026-09-29",
    "Revenue grew in 2025 2026 2027 projections",
    "SOC 2 Type II and ISO/IEC 27001:2022",
    "Claimed damages: USD 1,250,000.",
    "Mean time to patch: 43 days, MFA coverage 61%",
    "Order 1234567890 was shipped",
])
def test_no_false_positive_phone(text):
    r = scrub(text)
    assert r.text == text and r.total == 0


def test_bank_details_redacted():
    r = scrub("Account Number: 4839201748\nIBAN: GB29 NWBK 6016 1331 9268 19\nRouting Number: 021000021")
    assert "4839201748" not in r.text and "NWBK" not in r.text and "021000021" not in r.text
    assert r.counts["account"] == 3


def test_label_keeps_its_prefix_when_account_redacted():
    assert scrub("Account Number: 4839201748").text == "Account Number: [ACCOUNT_REDACTED]"


def test_labelled_names_redacted_without_touching_neighbours():
    text = "Prepared by: Omar Okafor\nSecurity Contact: Priya Mehta (x)"
    r = scrub(text)
    assert "Omar" not in r.text and "Priya" not in r.text
    assert r.text.splitlines()[0] == "Prepared by: [NAME_REDACTED]"
    assert r.text.splitlines()[1].startswith("Security Contact: [NAME_REDACTED]")
    assert r.counts["name"] == 2


def test_name_match_never_crosses_a_line():
    # Regression: a name pattern using \s+ swallowed the next line's label and skipped a name.
    r = scrub("Prepared by: Omar Okafor\nSecurity Contact: Priya Mehta")
    assert "Priya" not in r.text and r.counts["name"] == 2


def test_honorific_names_redacted():
    assert "Okafor" not in scrub("Meeting with Dr. Amara Okafor on Monday").text


def test_company_names_and_business_facts_are_preserved():
    text = "Supplier: Quarry Chemicals (SUP-001)\nSOC 2 Type II: Expired on 2026-04-24.\nActive lawsuits: 2"
    assert scrub(text).text == text


def test_scrub_is_idempotent():
    once = scrub("Account Manager: Priya Raman (priya@x.example, +1-415-555-0142)").text
    twice = scrub(once)
    assert twice.text == once and twice.total == 0


def test_counts_cover_every_kind():
    r = scrub("Account Manager: Priya Raman (priya@x.example, +1-415-555-0142)\nAccount Number: 4839201748")
    assert r.counts == {"email": 1, "phone": 1, "account": 1, "name": 1} and r.total == 4
