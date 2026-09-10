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