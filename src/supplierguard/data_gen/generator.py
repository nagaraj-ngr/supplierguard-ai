"""Deterministic synthetic data generator for SupplierGuard AI.

Design: every supplier first receives a latent risk score (1 to 10) for each of the six
dimensions. All source data (financials, delivery logs, news, dependency data and the four
documents) is then derived from those scores with a seeded random generator. The latent
scores are the ground truth used for evaluation, so they are known by construction rather
than guessed. They are NOT human expert labels and the README says so.

Every random stream is seeded from (seed, supplier id, purpose), so the same seed always
produces byte-identical files, and changing one generator does not shift the others.
"""

from __future__ import annotations

import csv
import json
import random
import re
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from ..dimensions import DIMENSIONS, Dimension

GENERATOR_VERSION = "1.0"
ASOF = date(2026, 9, 28)  # fixed "today" so output never depends on the wall clock

PREFIXES = [
    "Altair", "Brightline", "Cobalt", "Dunmore", "Everpeak", "Fairhaven", "Granite",
    "Halcyon", "Ironwood", "Juniper", "Kestrel", "Lumen", "Meridian", "Novara", "Orchard",
    "Pinnacle", "Quarry", "Redwood", "Silverton", "Tidewater", "Umber", "Vantage",
    "Westmere", "Xenon", "Yardley", "Zephyr",
]
SUFFIX_INDUSTRY = {
    "Components": "Electronic components",
    "Logistics": "Freight and logistics",
    "Metals": "Metal fabrication",
    "Plastics": "Injection-molded plastics",
    "Electronics": "Electronics assembly",
    "Packaging": "Industrial packaging",
    "Textiles": "Technical textiles",
    "Chemicals": "Specialty chemicals",
    "Systems": "Embedded systems",
    "Industries": "Industrial equipment",
}
COUNTRIES = [
    ("United States", "North America"), ("Mexico", "North America"),
    ("Germany", "Europe"), ("Poland", "Europe"), ("India", "Asia"),
    ("Vietnam", "Asia"), ("Malaysia", "Asia"), ("Brazil", "South America"),
    ("Turkey", "Europe"), ("Canada", "North America"),
]
CITIES = [
    "Dayton", "Leipzig", "Pune", "Monterrey", "Gdansk", "Penang", "Curitiba",
    "Izmir", "Hanoi", "Windsor", "Rotterdam", "Austin",
]
FIRST_NAMES = [
    "Priya", "Marcus", "Elena", "Tomas", "Amara", "Jonas", "Mei", "Rafael", "Ingrid",
    "Kofi", "Sofia", "Dmitri", "Lucia", "Omar", "Hannah", "Arjun", "Beatriz", "Callum",
    "Dalia", "Emil", "Farah", "Gustav", "Hana", "Ivan", "Jada", "Kenji",
]
LAST_NAMES = [
    "Raman", "Holloway", "Petrova", "Novak", "Okafor", "Lindqvist", "Zhang", "Castillo",
    "Berg", "Mensah", "Rossi", "Volkov", "Navarro", "Haddad", "Fischer", "Mehta",
    "Almeida", "Brennan", "Saleh", "Kowalski", "Tanaka", "Dubois", "Ahmed", "Larsen",
]

QUARTERS = ["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
MONTHS = [
    "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03",
    "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09",
]

DOC_TYPES = ("contract", "security_assessment", "compliance_audit", "legal_filings")

# (headline, body) templates. Placeholders: {n} supplier name, {city}, {pct}, {amt}.
POSITIVE = [
    ("{n} opens new production line in {city}",
     "{n} commissioned a new line at its {city} site, adding capacity for existing customers. Management said the expansion was funded from operating cash flow."),
    ("{n} wins quality award from industry association",
     "{n} received a supplier excellence award citing low defect rates and strong on-time delivery. The company plans to publish its audit results later this year."),
    ("{n} signs multi-year framework agreement",
     "{n} announced a framework agreement with a large manufacturer. Terms were not disclosed, and the company expects stable volumes through the contract period."),
    ("{n} reports record quarterly revenue",
     "{n} reported revenue growth of {pct}% year over year, driven by repeat orders. Analysts described the results as ahead of expectations."),
]
NEUTRAL = [
    ("{n} publishes annual sustainability report",
     "{n} released its annual sustainability report covering energy use and waste reduction targets. The report was prepared in line with standard reporting guidelines."),
    ("{n} to present at regional manufacturing conference",
     "{n} will present on supply chain practices at a regional conference in {city}. Executives are scheduled to take questions from attendees."),
    ("{n} appoints new regional sales director",
     "{n} announced a leadership appointment in its sales organization. The company said the change is part of a planned succession."),
]
NEGATIVE = [
    [  # mild
        ("{n} reports minor shipment delays at {city} plant",
         "{n} confirmed shipment delays of several days at its {city} plant due to a equipment maintenance backlog. The company said deliveries should normalize within weeks."),
        ("{n} faces customer complaints over packaging defects",
         "Several customers reported damaged goods linked to packaging changes at {n}. The company said it is reviewing its packing process."),
        ("Analysts trim outlook for {n} on softer demand",
         "Two analysts lowered their outlook for {n}, citing softer demand and rising input costs. No change to guidance has been announced."),
    ],
    [  # moderate
        ("{n} announces layoffs affecting {pct}% of workforce",
         "{n} said it will cut {pct}% of its workforce as part of a cost reduction program. Unions criticized the lack of consultation."),
        ("Regulators open inquiry into {n} labeling practices",
         "A regulator opened an inquiry into product labeling at {n}. The company said it is cooperating and that no violations have been established."),
        ("Labor union calls strike at {n} {city} facility",
         "Workers at the {city} facility of {n} voted to strike over pay and safety conditions. Production at the site may be disrupted."),
        ("{n} cuts guidance after margin squeeze",
         "{n} lowered full-year guidance, citing margin pressure from raw material costs and weaker pricing power."),
    ],
    [  # severe
        ("{n} under fraud investigation over inflated invoices",
         "Prosecutors are investigating {n} over allegations of inflated invoicing to customers. The company denies wrongdoing, but two customers have paused orders."),
        ("Lenders flag covenant breach at {n}, bankruptcy rumors grow",
         "Lenders to {n} have flagged a covenant breach and are negotiating a standstill. Market commentary has raised the possibility of insolvency proceedings."),
        ("Data breach at {n} exposes customer records",
         "{n} disclosed unauthorized access to systems holding customer records. Regulators have been notified and affected customers are being contacted."),
        ("{n} executives arrested in bribery probe",
         "Authorities detained senior executives of {n} as part of a bribery investigation. The board has suspended the individuals pending the outcome."),
        ("Court freezes assets of {n} amid insolvency petition",
         "A court froze assets of {n} after creditors filed an insolvency petition. Operations continue under court supervision."),
    ],
]
OUTLETS = ["Meridian Business Wire", "Harbor Trade Journal", "Continental Industry Review",
           "Northgate Financial Daily", "Supply Chain Ledger"]

PHONE_FORMATS = [
    lambda r: f"+1-{r.choice([212, 415, 312, 617, 206])}-555-01{r.randint(0, 99):02d}",
    lambda r: f"+44 20 7946 0{r.randint(100, 999)}",
    lambda r: f"+91 80 4155 0{r.randint(100, 999)}",
    lambda r: f"({r.choice([212, 415, 312])}) 555-01{r.randint(0, 99):02d}",
]

# Latent score choices per tier: (values, weights)
SCORE_CHOICES = {
    "high": ([6, 7, 8, 9, 10], [1, 2, 3, 2, 2]),
    "medium": ([3, 4, 5, 6, 7], [1, 2, 3, 2, 1]),
    "low": ([1, 2, 3, 4], [2, 3, 2, 1]),
}


# ---------------------------------------------------------------- helpers

def sev(score: int) -> float:
    """Map a 1..10 risk score to a severity in [0, 1]."""
    return (score - 1) / 9.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rng(seed: int, *parts: str) -> random.Random:
    # String seeding is hashed with sha512 by Python, so it is stable across runs and platforms.
    return random.Random(":".join([str(seed), *parts]))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def tier_counts(n: int) -> Dict[str, int]:
    high = round(n * 0.3)
    low = round(n * 0.3)
    return {"high": high, "medium": n - high - low, "low": low}


def golden_counts(n: int) -> Dict[str, int]:
    total = min(n, round(n * 0.4))
    high = round(total * 0.3)
    low = round(total * 0.3)
    return {"high": high, "medium": total - high - low, "low": low}


def _iso(d: date) -> str:
    return d.isoformat()


def _person(rng: random.Random, domain: str) -> Tuple[str, str, str]:
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    return f"{first} {last}", f"{first.lower()}.{last.lower()}@{domain}", rng.choice(PHONE_FORMATS)(rng)


def _account_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([10, 11, 12])))


def _iban(rng: random.Random) -> str:
    cc = rng.choice(["GB", "DE", "NL", "FR"])
    bank = "".join(rng.choices(string.ascii_uppercase, k=4))
    groups = " ".join(str(rng.randint(1000, 9999)) for _ in range(3))
    return f"{cc}{rng.randint(10, 99)} {bank} {groups} {rng.randint(10, 99)}"


# ---------------------------------------------------------------- tabular data

def financial_rows(rng: random.Random, sid: str, score: int) -> List[dict]:
    t = sev(score)
    revenue = rng.uniform(25, 900)
    growth = 0.03 - 0.07 * t  # per-quarter growth: +3% (safe) down to -4% (distressed)
    rows = []
    for q in QUARTERS:
        revenue *= 1 + growth + rng.gauss(0, 0.006)
        rows.append({
            "supplier_id": sid,
            "quarter": q,
            "revenue_musd": round(revenue, 1),
            "gross_margin_pct": round(clamp(32 - 22 * t + rng.gauss(0, 1.2), 2, 60), 1),
            "debt_to_equity": round(max(0.05, 0.3 + 2.7 * t + rng.gauss(0, 0.08)), 2),
            "current_ratio": round(max(0.3, 2.4 - 1.5 * t + rng.gauss(0, 0.08)), 2),
            "credit_score": int(clamp(800 - 400 * t + rng.gauss(0, 12), 300, 850)),
        })
    return rows


def delivery_rows(rng: random.Random, sid: str, score: int) -> List[dict]:
    t = sev(score)
    rows = []
    for i, m in enumerate(MONTHS):
        progress = i / (len(MONTHS) - 1)
        rows.append({
            "supplier_id": sid,
            "month": m,
            "orders": rng.randint(80, 400),
            "on_time_pct": round(clamp(99.0 - 28 * t - 4.0 * t * progress + rng.gauss(0, 1.2), 40, 100), 1),
            "defect_rate_pct": round(max(0.05, 0.3 + 6.0 * t + 1.0 * t * progress + rng.gauss(0, 0.25)), 2),
            "sla_breaches": max(0, int(round(0.2 + 4.5 * t + rng.gauss(0, 0.6)))),
            "quality_score": round(clamp(98 - 33 * t + rng.gauss(0, 1.5), 30, 100), 1),
        })
    return rows


def dependency_record(rng: random.Random, sid: str, score: int) -> dict:
    t = sev(score)
    category_share = round(clamp(8 + 78 * t + rng.gauss(0, 3), 2, 95), 1)
    alternatives = max(0, int(round(5 - 5 * t + rng.gauss(0, 0.4))))
    category_weight = rng.uniform(4, 22)
    return {
        "supplier_id": sid,
        "category": rng.choice(["Raw materials", "Sub-assemblies", "Packaging", "Logistics", "Contract manufacturing"]),
        "annual_spend_musd": round(rng.uniform(1.5, 60), 1),
        "share_of_category_spend_pct": category_share,
        "share_of_total_spend_pct": round(category_share * category_weight / 100, 2),
        "qualified_alternative_suppliers": alternatives,
        "estimated_switching_time_months": round(clamp(1 + 16 * t + rng.gauss(0, 0.8), 0.5, 24), 1),
        "single_source": alternatives == 0,
    }


def negative_article_count(rng: random.Random, score: int) -> int:
    """Number of negative news articles (out of 8) for a given news/reputation score."""
    return int(clamp(round(0.5 + 5.5 * sev(score) + rng.gauss(0, 0.4)), 0, 7))


def news_articles(rng: random.Random, sid: str, name: str, city: str, score: int) -> List[dict]:
    t = sev(score)
    n_neg = negative_article_count(rng, score)
    probs = (0.8, 0.2, 0.0) if t < 0.34 else (0.3, 0.5, 0.2) if t < 0.67 else (0.1, 0.3, 0.6)
    picked = []
    for _ in range(n_neg):
        severity_bin = rng.choices([0, 1, 2], weights=probs)[0]
        picked.append(rng.choice(NEGATIVE[severity_bin]))
    for _ in range(8 - n_neg):
        picked.append(rng.choice(POSITIVE if rng.random() < 0.65 else NEUTRAL))
    articles = []
    for headline, body in picked:
        fields = {"n": name, "city": city, "pct": rng.randint(5, 25), "amt": rng.randint(2, 40)}
        articles.append({
            "date": _iso(ASOF - timedelta(days=rng.randint(3, 350))),
            "source": rng.choice(OUTLETS),
            "headline": headline.format(**fields),
            "body": body.format(**fields),
        })
    articles.sort(key=lambda a: (a["date"], a["headline"]), reverse=True)
    return [{"article_id": f"{sid}-N{i:02d}", **a} for i, a in enumerate(articles, start=1)]


# ---------------------------------------------------------------- documents

def contract_text(rng: random.Random, sid: str, name: str, domain: str) -> str:
    mgr, mgr_email, mgr_phone = _person(rng, domain)
    signer, _, _ = _person(rng, domain)
    start = ASOF - timedelta(days=rng.randint(200, 1200))
    term = rng.choice([24, 36, 48])
    commit = rng.choice([95, 96, 97, 98])
    penalty = rng.choice([1, 2, 3])
    net = rng.choice([30, 45, 60])
    return f"""SUPPLY AND SERVICES AGREEMENT
Supplier: {name} ({sid})
Buyer: Harborline Manufacturing Group (the "Buyer")

1. Parties and contacts
Account Manager: {mgr} ({mgr_email}, {mgr_phone})
Contract Signatory: {signer}

2. Term
This agreement took effect on {_iso(start)} and runs for {term} months. It renews automatically for 12 months unless either party gives 90 days written notice.

3. Service levels
The Supplier commits to an on-time delivery rate of at least {commit}% measured monthly. Each month below the commitment earns the Buyer a service credit of {penalty}% of that month's invoiced value. Defects above the agreed cap must be corrected at the Supplier's cost.

4. Security and compliance requirements
The Supplier shall hold a current SOC 2 Type II report and ISO/IEC 27001 certification, notify the Buyer of any security breach within 72 hours, and permit an audit of its controls once per year.

5. Payment
Payment terms are net {net} days.
Account Name: {name}
Account Number: {_account_number(rng)}
IBAN: {_iban(rng)}
"""


def security_assessment_text(rng: random.Random, sid: str, name: str, domain: str, score: int) -> str:
    t = sev(score)
    author, _, _ = _person(rng, domain)
    contact, contact_email, contact_phone = _person(rng, domain)
    assessed = ASOF - timedelta(days=rng.randint(20, 120))
    if t < 0.3:
        soc2 = f"Valid. Report dated {_iso(assessed - timedelta(days=90))}, expires {_iso(assessed + timedelta(days=270))}."
    elif t < 0.6:
        soc2 = f"Valid but expires within 90 days ({_iso(assessed + timedelta(days=60))}). Renewal audit in progress."
    elif t < 0.8:
        soc2 = f"Expired on {_iso(assessed - timedelta(days=120))}. Renewal audit not yet scheduled."
    else:
        soc2 = "Not held. The Supplier has never completed a SOC 2 audit."
    if t < 0.5:
        iso = f"Certified (ISO/IEC 27001:2022), valid until {_iso(assessed + timedelta(days=500))}."
    elif t < 0.75:
        iso = "Certification lapsed. Re-certification pending."
    else:
        iso = "Not certified."
    breaches = int(clamp(round(3.2 * t + rng.gauss(0, 0.35)), 0, 4))
    mfa = int(clamp(round(100 - 55 * t + rng.gauss(0, 3)), 20, 100))
    crit = max(0, int(round(7 * t + rng.gauss(0, 0.6))))
    patch_days = int(clamp(round(5 + 55 * t + rng.gauss(0, 3)), 2, 90))
    ir_plan = "Yes" if t < 0.8 else "No"
    ir_tested = "Yes" if t < 0.5 else "No"
    breach_detail = ""
    if breaches > 0:
        kind = rng.choice(["phishing-driven credential theft", "ransomware on a file server",
                           "misconfigured cloud storage", "compromise of a third-party vendor"])
        breach_detail = f"\nMost recent breach: {(ASOF - timedelta(days=rng.randint(30, 700))).strftime('%Y-%m')}, caused by {kind}."
    if t < 0.34:
        verdict = "The assessor found a mature security program with timely patching and tested response procedures."
    elif t < 0.67:
        verdict = "The assessor found a partly mature program. Several controls need improvement and remediation is under way."
    else:
        verdict = "The assessor found significant control weaknesses. Critical findings remain open and several expected certifications are missing."
    return f"""INFORMATION SECURITY ASSESSMENT
Supplier: {name} ({sid})
Assessment date: {_iso(assessed)}
Prepared by: {author}
Security Contact: {contact} ({contact_email}, {contact_phone})

Certifications
SOC 2 Type II: {soc2}
ISO/IEC 27001: {iso}

Security posture
Confirmed security breaches (last 3 years): {breaches}{breach_detail}
Multi-factor authentication coverage: {mfa}%
Open critical penetration-test findings: {crit}
Mean time to patch critical vulnerabilities: {patch_days} days
Documented incident response plan: {ir_plan}
Incident response plan tested within 12 months: {ir_tested}

Assessor summary
{verdict}
"""


def compliance_audit_text(rng: random.Random, sid: str, name: str, domain: str, score: int) -> str:
    t = sev(score)
    auditor, _, _ = _person(rng, domain)
    audited = ASOF - timedelta(days=rng.randint(30, 300))
    major = int(clamp(round(4.5 * t + rng.gauss(0, 0.4)), 0, 5))
    minor = int(clamp(round(1 + 9 * t + rng.gauss(0, 0.8)), 0, 12))
    overdue = int(clamp(round(4 * t + rng.gauss(0, 0.5)), 0, 5))
    violations = int(clamp(round(3 * t + rng.gauss(0, 0.4)), 0, 3))
    if major == 0 and minor <= 3:
        result = "Pass"
    elif major <= 1 and minor <= 7:
        result = "Pass with observations"
    else:
        result = "Fail"
    iso9001 = "Valid" if t < 0.6 else "Suspended pending corrective action"
    iso14001 = "Valid" if t < 0.7 else "Not held"
    bribery = "Policy in place and training current" if t < 0.7 else "Gaps identified in policy and training records"
    if t < 0.34:
        narrative = "Management responded promptly to prior findings and closed corrective actions on schedule."
    elif t < 0.67:
        narrative = "Some corrective actions are behind schedule. Management has committed to a remediation plan."
    else:
        narrative = "Repeat findings were observed and several corrective actions are overdue. Management oversight of compliance is weak."
    return f"""COMPLIANCE AUDIT REPORT
Supplier: {name} ({sid})
Audit date: {_iso(audited)}
Auditing body: Halverson Assurance Partners
Lead Auditor: {auditor}
Scope: quality management, labor practices, environmental controls, anti-bribery

Findings
Overall result: {result}
Open major findings: {major}
Open minor findings: {minor}
Overdue corrective actions: {overdue}
Regulatory violations (last 24 months): {violations}

Certifications and policies
ISO 9001: {iso9001}
ISO 14001: {iso14001}
Anti-bribery program: {bribery}

Auditor narrative
{narrative}
"""


def legal_filings_text(rng: random.Random, sid: str, name: str, domain: str, score: int) -> str:
    t = sev(score)
    counsel, _, _ = _person(rng, domain)
    active = int(clamp(round(3.5 * t + rng.gauss(0, 0.5)), 0, 4))
    settled = int(clamp(round(1 + 3 * t + rng.gauss(0, 0.7)), 0, 5))
    fines = int(round((t ** 2) * 4_000_000 * rng.uniform(0.5, 1.5))) if t > 0.3 else 0
    fine_actions = 0 if fines == 0 else 1 + (1 if t > 0.7 else 0)
    investigations = 2 if t > 0.9 else 1 if t > 0.65 else 0
    kinds = ["Contract dispute", "Intellectual property claim", "Labor claim", "Product liability claim", "Fraud allegation"]
    lines = []
    for i in range(active):
        lines.append(f"{i + 1}. [Active] {rng.choice(kinds)}. Claimed damages: USD {rng.randint(2, 90) * 50_000:,}.")
    for j in range(settled):
        lines.append(f"{active + j + 1}. [Settled] {rng.choice(kinds)}. Settlement: USD {rng.randint(1, 30) * 25_000:,}.")
    cases = "\n".join(lines) if lines else "No lawsuits on record."
    return f"""LEGAL AND REGULATORY FILINGS SUMMARY
Supplier: {name} ({sid})
Search date: {_iso(ASOF - timedelta(days=rng.randint(5, 60)))}
Counsel of record: {counsel}

Summary
Active lawsuits: {active}
Settled lawsuits (last 5 years): {settled}
Regulatory fines (last 5 years): USD {fines:,} across {fine_actions} actions
Open regulatory investigations: {investigations}

Case details
{cases}
"""


# ---------------------------------------------------------------- orchestration

def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def generate_dataset(out_dir, seed: int = 42, n: int = 50, pdf: bool = False) -> dict:
    """Write the full synthetic dataset to ``out_dir`` and return a summary."""
    out = Path(out_dir)
    (out / "news").mkdir(parents=True, exist_ok=True)
    (out / "documents").mkdir(parents=True, exist_ok=True)

    master = _rng(seed, "master")
    combos = [(p, s) for p in PREFIXES for s in SUFFIX_INDUSTRY]
    if n > len(combos):
        raise ValueError(f"n={n} exceeds the {len(combos)} available unique supplier names")
    master.shuffle(combos)
    tiers: List[str] = []
    for tier, count in tier_counts(n).items():
        tiers += [tier] * count
    master.shuffle(tiers)

    suppliers, truth = [], {}
    fin_rows: List[dict] = []
    del_rows: List[dict] = []
    dependency: List[dict] = []

    for i in range(n):
        sid = f"SUP-{i + 1:03d}"
        prefix, suffix = combos[i]
        name = f"{prefix} {suffix}"
        tier = tiers[i]
        country, region = master.choice(COUNTRIES)
        city = master.choice(CITIES)
        values, weights = SCORE_CHOICES[tier]
        scores = {d.value: master.choices(values, weights=weights)[0] for d in DIMENSIONS}
        suppliers.append({
            "id": sid, "name": name, "industry": SUFFIX_INDUSTRY[suffix], "country": country,
            "region": region, "headquarters_city": city,
            "employees": master.randint(80, 9000), "founded": master.randint(1952, 2016),
        })
        overall = round(sum(scores.values()) / len(scores), 1)
        truth[sid] = {"name": name, "tier": tier, "dimension_scores": scores, "overall": overall}

        fin_rows += financial_rows(_rng(seed, sid, "financial"), sid, scores[Dimension.FINANCIAL.value])
        del_rows += delivery_rows(_rng(seed, sid, "delivery"), sid, scores[Dimension.DELIVERY_QUALITY.value])
        dependency.append(dependency_record(_rng(seed, sid, "dependency"), sid, scores[Dimension.CONCENTRATION.value]))
        _write_json(out / "news" / f"{sid}.json", news_articles(
            _rng(seed, sid, "news"), sid, name, city, scores[Dimension.NEWS_REPUTATION.value]))

        domain = f"{_slug(name)}.example"
        docs = {
            "contract": contract_text(_rng(seed, sid, "contract"), sid, name, domain),
            "security_assessment": security_assessment_text(
                _rng(seed, sid, "security"), sid, name, domain, scores[Dimension.CYBERSECURITY.value]),
            "compliance_audit": compliance_audit_text(
                _rng(seed, sid, "compliance"), sid, name, domain, scores[Dimension.COMPLIANCE_LEGAL.value]),
            "legal_filings": legal_filings_text(
                _rng(seed, sid, "legal"), sid, name, domain, scores[Dimension.COMPLIANCE_LEGAL.value]),
        }
        doc_dir = out / "documents" / sid
        doc_dir.mkdir(parents=True, exist_ok=True)
        for doc_type, text in docs.items():
            if pdf:
                from .pdf import write_text_pdf
                write_text_pdf(text, doc_dir / f"{doc_type}.pdf")
            else:
                (doc_dir / f"{doc_type}.txt").write_text(text, encoding="utf-8")

    # Stratified golden split: the evaluation set is chosen across all three risk tiers.
    golden_ids: List[str] = []
    for tier, count in golden_counts(n).items():
        pool = sorted(sid for sid, t in truth.items() if t["tier"] == tier)
        golden_ids += master.sample(pool, count)
    golden_ids.sort()
    for sid in truth:
        truth[sid]["split"] = "golden" if sid in golden_ids else "holdout"

    _write_json(out / "suppliers.json", suppliers)
    _write_csv(out / "financials.csv", fin_rows)
    _write_csv(out / "delivery.csv", del_rows)
    _write_json(out / "dependency.json", dependency)
    _write_json(out / "ground_truth.json", {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "note": ("Scores are the latent parameters the data was generated from. They are "
                 "ground truth by construction, not human expert labels."),
        "dimensions": [d.value for d in DIMENSIONS],
        "golden_ids": golden_ids,
        "suppliers": truth,
    })
    tier_summary = {t: sum(1 for v in truth.values() if v["tier"] == t) for t in ("high", "medium", "low")}
    return {"suppliers": n, "tiers": tier_summary, "golden": len(golden_ids),
            "documents": "pdf" if pdf else "txt", "seed": seed, "out": str(out)}
