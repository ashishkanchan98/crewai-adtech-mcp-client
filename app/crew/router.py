"""
Query classifier: routes incoming support queries to the correct crew.
classifier_mode=0 (local)  : keyword scoring — fast, no API call.
classifier_mode=1 (production): Groq llama-3.1-8b-instant — understands full query context.
"""
import logging

from groq import Groq

from app.config import settings

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


def _classify_keyword(query: str) -> str:
    """classifier_mode=0: keyword hit-count scoring."""
    q = query.lower()
    scores = {
        uc: sum(1 for kw in meta["keywords"] if kw in q)
        for uc, meta in USE_CASE_META.items()
    }
    best, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score == 0:
        logger.info("No keyword match — defaulting to delivery_failure")
        return "delivery_failure"
    logger.info("Keyword classifier → '%s' (score=%d)  query='%.80s'", best, best_score, query)
    return best


def _classify_llm(query: str) -> str:
    """classifier_mode=1: single Groq llama-3.1-8b-instant call."""
    use_cases = "\n".join(
        f"- {key}: {meta['label']}" for key, meta in USE_CASE_META.items()
    )
    prompt = (
        "You are an adtech support router. Given a support query, return ONLY the single "
        "most appropriate use-case key from the list below — no explanation, no punctuation.\n\n"
        f"Use cases:\n{use_cases}\n\n"
        f"Query: {query}"
    )
    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip().lower()
        if result in USE_CASE_META:
            logger.info("LLM classifier → '%s'  query='%.80s'", result, query)
            return result
        logger.warning("LLM returned unknown key '%s' — falling back to keyword", result)
    except Exception as exc:
        logger.warning("LLM classifier error (%s) — falling back to keyword", exc)
    return _classify_keyword(query)


def classify_query(query: str) -> str:
    if settings.classifier_mode == 1:
        return _classify_llm(query)
    return _classify_keyword(query)


def get_use_case_label(use_case: str) -> str:
    return USE_CASE_META.get(use_case, {}).get("label", use_case)


ALL_USE_CASES = list(USE_CASE_META.keys())
