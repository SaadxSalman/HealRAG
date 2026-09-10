"""Unit tests for LLM-output JSON coercion and graph routing logic."""

from __future__ import annotations

import pytest

from app.agents.graph import route_after_grade, route_after_hallucination, route_after_rewrite
from app.core.llm import _coerce_json


# ---------------------------------------------------------------------- #
# JSON coercion
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"score": 7, "verdict": "accept"}', {"score": 7, "verdict": "accept"}),
        (
            '```json\n{"score": 3, "verdict": "reject"}\n```',
            {"score": 3, "verdict": "reject"},
        ),
        ('prefix text {"score": 9} suffix', {"score": 9}),
    ],
)
def test_coerce_json(raw, expected):
    assert _coerce_json(raw) == expected


def test_coerce_json_rejects_garbage():
    with pytest.raises(ValueError):
        _coerce_json("not json at all")


# ---------------------------------------------------------------------- #
# Graph routing decisions
# ---------------------------------------------------------------------- #
def test_route_after_grade_generates_when_enough_accepted():
    state = {"chunks": [{"accepted": True}, {"accepted": False}], "rewrite_count": 0}
    assert route_after_grade(state) == "generate"


def test_route_after_grade_rewrites_when_none_accepted():
    state = {"chunks": [{"accepted": False}], "rewrite_count": 0}
    assert route_after_grade(state) == "rewrite"


def test_route_after_grade_forces_generate_when_rewrites_exhausted():
    state = {"chunks": [{"accepted": False}], "rewrite_count": 2}
    assert route_after_grade(state) == "generate"


def test_route_after_rewrite_loops_within_budget():
    state = {"rewrite_count": 1, "accepted_chunks": []}
    assert route_after_rewrite(state) == "rewrite"


def test_route_after_hallucination_retries_on_fail():
    state = {
        "hallucination_result": {"verdict": "fail"},
        "retries": 0,
    }
    assert route_after_hallucination(state) == "corrective_rerefetch"


def test_route_after_hallucination_ends_after_max_retries():
    state = {
        "hallucination_result": {"verdict": "fail"},
        "retries": 2,
    }
    assert route_after_hallucination(state) == "end"


def test_route_after_hallucination_ends_on_pass():
    state = {
        "hallucination_result": {"verdict": "pass"},
        "retries": 0,
    }
    assert route_after_hallucination(state) == "end"


# ---------------------------------------------------------------------- #
# Grader normalization of off-spec SLM JSON
# ---------------------------------------------------------------------- #
from app.agents.grader import (  # noqa: E402
    RelevanceGrader,
    HallucinationGrader,
    _derive_verdict,
    _extract_score,
)


class _FakeClient:
    """Stands in for the Ollama client, replaying scripted JSON payloads."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list = []

    def chat_json(self, prompt, **kwargs):
        self.calls.append(prompt)
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def test_extract_score_handles_alternate_keys_and_scales():
    assert _extract_score({"score": 8}) == 0.8
    assert _extract_score({"relevance": 3}) == 0.3
    assert _extract_score({"score": "7"}) == 0.7
    assert _extract_score({"score": None, "verdict": "accept"}) is None
    assert _extract_score({"unrelated": 1}) is None


def test_derive_verdict_infers_from_score_when_missing():
    assert _derive_verdict({"score": 8}, 0.8, "accept", "reject", 0.5) == "accept"
    assert _derive_verdict({"score": 1}, 0.1, "accept", "reject", 0.5) == "reject"
    assert _derive_verdict({"verdict": "weird"}, 0.9, "pass", "fail", 0.7) == "pass"
    assert _derive_verdict({"verdict": "weird"}, None, "pass", "fail", 0.7) is None


def test_relevance_grader_accepts_off_spec_but_usable_json():
    g = RelevanceGrader(model="fake")
    g.client = _FakeClient([{"relevance": 9, "explanation": "on topic"}])
    out = g.grade_chunk("q", "id", "text")
    assert out["verdict"] == "accept"
    assert out["relevance"] == 0.9
    assert out["reason"] == "on topic"


def test_relevance_grader_accept_without_score_defaults_positive():
    g = RelevanceGrader(model="fake")
    g.client = _FakeClient([{"verdict": "accept"}])
    out = g.grade_chunk("q", "id", "text")
    assert out["verdict"] == "accept"
    assert out["relevance"] == 0.7


def test_relevance_grader_unusable_json_is_rejected_not_fatal():
    g = RelevanceGrader(model="fake")
    g.client = _FakeClient([{"foo": "bar"}, {"foo": 1}])
    out = g.grade_batch("q", [{"id": "a", "text": "t"}, {"id": "b", "text": "u"}])
    assert all(c["accepted"] is False for c in out)
    assert all(c["verdict"] == "reject" for c in out)
    assert "unusable grader JSON" in out[0]["reason"]


def test_hallucination_grader_fails_closed_on_garbage():
    g = HallucinationGrader(model="fake")
    g.client = _FakeClient([{"nope": True}])
    out = g.verify("q", "answer", ["context"])
    assert out["verdict"] == "fail"
    assert out["score"] == 0.0


def test_hallucination_grader_infers_pass_from_high_score():
    g = HallucinationGrader(model="fake")
    g.client = _FakeClient([{"score": 9, "unsupported_claims": []}])
    out = g.verify("q", "answer", ["context"])
    assert out["verdict"] == "pass"