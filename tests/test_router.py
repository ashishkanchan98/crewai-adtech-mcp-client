"""
Tests for app/crew/router.py — both classifier modes.
Mode 0 (keyword): runs fully offline, no mocking needed.
Mode 1 (LLM):     Groq client is mocked — no real API call.
"""
import types
from unittest.mock import MagicMock, patch

import pytest

from app.crew.router import _classify_keyword, _classify_llm, classify_query, ALL_USE_CASES


# ── Mode 0: keyword classifier ────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("campaign not delivering, zero impressions since morning",  "delivery_failure"),
    ("DSP shows 120k but GAM shows 84k, 30% discrepancy",       "reporting_discrepancy"),
    ("fraud spike detected, IVT rate 40%, SIVT flagged",        "ivt_fraud"),
    ("PMP deal zero bids, seat ID not mapped",                  "pmp_deal"),
    ("pixel not firing, conversion drop after iOS 17",          "pixel_attribution"),
    ("ready to launch tomorrow, need a campaign audit",         "pre_launch_audit"),
    ("budget exhausted by 10am, severe overpacing",             "budget_pacing"),
])
def test_keyword_classifier_known_queries(query, expected):
    assert _classify_keyword(query) == expected


def test_keyword_classifier_zero_hits_defaults_to_delivery_failure():
    assert _classify_keyword("nothing useful here xyz") == "delivery_failure"


def test_keyword_classifier_case_insensitive():
    assert _classify_keyword("FRAUD SPIKE DETECTED IVT RATE HIGH") == "ivt_fraud"


def test_keyword_classifier_tie_returns_first_in_dict():
    # "deal" hits pmp_deal; ensure it does not bleed into another category
    result = _classify_keyword("deal")
    assert result == "pmp_deal"


# ── Mode 1: LLM classifier (Groq mocked) ──────────────────────────────────────

def _make_groq_response(content: str):
    """Build a minimal fake Groq response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.parametrize("llm_reply,expected", [
    ("delivery_failure",       "delivery_failure"),
    ("reporting_discrepancy",  "reporting_discrepancy"),
    ("ivt_fraud",              "ivt_fraud"),
    ("pmp_deal",               "pmp_deal"),
    ("pixel_attribution",      "pixel_attribution"),
    ("pre_launch_audit",       "pre_launch_audit"),
    ("budget_pacing",          "budget_pacing"),
])
def test_llm_classifier_valid_responses(llm_reply, expected):
    with patch("app.crew.router.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_groq_response(llm_reply)
        )
        assert _classify_llm("any query") == expected


def test_llm_classifier_unknown_key_falls_back_to_keyword():
    with patch("app.crew.router.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_groq_response("completely_unknown_key")
        )
        # fallback runs keyword scoring on "campaign not delivering"
        result = _classify_llm("campaign not delivering zero impressions")
        assert result == "delivery_failure"


def test_llm_classifier_api_error_falls_back_to_keyword():
    with patch("app.crew.router.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.side_effect = Exception("API timeout")
        result = _classify_llm("fraud spike ivt rate high")
        assert result == "ivt_fraud"


def test_llm_classifier_strips_whitespace_and_lowercases():
    with patch("app.crew.router.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_groq_response("  Budget_Pacing\n")
        )
        assert _classify_llm("budget burning fast") == "budget_pacing"


# ── classify_query() dispatch ─────────────────────────────────────────────────

def test_classify_query_mode0_uses_keyword(monkeypatch):
    monkeypatch.setattr("app.crew.router.settings.classifier_mode", 0)
    assert classify_query("campaign not delivering zero impressions") == "delivery_failure"


def test_classify_query_mode1_uses_llm(monkeypatch):
    monkeypatch.setattr("app.crew.router.settings.classifier_mode", 1)
    with patch("app.crew.router.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = (
            _make_groq_response("ivt_fraud")
        )
        assert classify_query("bot traffic spike") == "ivt_fraud"


# ── sanity: ALL_USE_CASES is complete ─────────────────────────────────────────

def test_all_use_cases_has_seven_entries():
    assert len(ALL_USE_CASES) == 7
