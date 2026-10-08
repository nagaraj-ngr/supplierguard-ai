"""The six risk dimensions assessed for every supplier."""

from __future__ import annotations

from enum import Enum


class Dimension(str, Enum):
    CYBERSECURITY = "cybersecurity"
    FINANCIAL = "financial"
    COMPLIANCE_LEGAL = "compliance_legal"
    NEWS_REPUTATION = "news_reputation"
    DELIVERY_QUALITY = "delivery_quality"
    CONCENTRATION = "concentration"


DIMENSIONS = list(Dimension)
