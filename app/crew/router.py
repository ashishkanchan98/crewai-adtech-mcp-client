"""
Query classifier: routes incoming support queries to the correct crew.
Uses keyword scoring for speed; falls back to 'delivery_failure' if no match.
"""
import logging
import re

logger = logging.getLogger(__name__)

USE_CASE_META = {
    "delivery_failure": {
        "label": "Campaign Not Delivering",
        "keywords": [
            "not delivering", "zero impressions", "no impressions", "not running",
            "not spending", "no spend", "paused", "delivery stopped", "delivery issue",
            "campaign stopped", "impressions dropped", "no traffic",
        ],
    },
    "reporting_discrepancy": {
        "label": "Reporting Discrepancy Investigation",
        "keywords": [
            "discrepancy", "gap", "mismatch", "dsp shows", "gam shows",
            "impression count", "different numbers", "reporting gap",
            "120k", "84k", "30%", "reconcile", "third-party vs",
        ],
    },
    "ivt_fraud": {
        "label": "IVT / Fraud Spike",
        "keywords": [
            "ivt", "fraud", "bot traffic", "invalid traffic", "click fraud",
            "fraud spike", "ivt rate", "sivt", "givt", "ias", "doubleverify",
            "suspicious clicks", "placement quality",
        ],
    },
    "pmp_deal": {
        "label": "PMP Deal Zero Bids",
        "keywords": [
            "deal", "pmp", "private marketplace", "zero bids", "no bid requests",
            "bid stream", "seat id", "seat mapping", "deal-", "deal_", "zero bid",
            "deal sync", "deal not", "0 bid",
        ],
    },
    "pixel_attribution": {
        "label": "Pixel & Conversion Attribution Drop",
        "keywords": [
            "pixel", "conversion", "attribution", "itp", "tracking",
            "not converting", "conversion drop", "ios 17", "cookie", "match rate",
            "server-side", "first-party", "view-through", "click-through",
        ],
    },
    "pre_launch_audit": {
        "label": "Pre-Launch Campaign Audit",
        "keywords": [
            "audit", "pre-launch", "pre launch", "before launch", "go live",
            "launch tomorrow", "launch today", "ready to launch", "launch checklist",
            "campaign audit", "ready for launch",
        ],
    },
    "budget_pacing": {
        "label": "Budget Pacing Optimization",
        "keywords": [
            "pacing", "budget", "overspend", "burning", "depleting",
            "10am", "morning", "asap pacing", "budget exhausted", "daily budget",
            "overpacing", "budget hit", "budget ran out", "front-loaded",
        ],
    },
}


def classify_query(query: str) -> str:
    """
    Score each use case by keyword hits in the query (case-insensitive).
    Returns the use case key with the highest score.
    Ties broken by order in USE_CASE_META.
    """
    q = query.lower()
    scores: dict[str, int] = {}

    for use_case, meta in USE_CASE_META.items():
        score = sum(1 for kw in meta["keywords"] if kw in q)
        scores[use_case] = score

    best_use_case = max(scores, key=lambda k: scores[k])
    best_score = scores[best_use_case]

    if best_score == 0:
        logger.info("No keyword match — defaulting to delivery_failure")
        return "delivery_failure"

    logger.info(
        "Query classified as '%s' (score=%d)  query='%.80s'",
        best_use_case, best_score, query,
    )
    return best_use_case


def get_use_case_label(use_case: str) -> str:
    return USE_CASE_META.get(use_case, {}).get("label", use_case)


ALL_USE_CASES = list(USE_CASE_META.keys())
