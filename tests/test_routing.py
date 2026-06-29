"""Tests for auto-routing: deciding when a turn should escalate to Claude."""

from __future__ import annotations

import pytest

from local_ai.routing import route_request, RouteDecision


def test_simple_request_stays_local():
    decision = route_request("fix this typo in the README", current_backend="local")
    assert isinstance(decision, RouteDecision)
    assert decision.escalate is False


def test_simple_question_stays_local():
    assert route_request("what does this function do?", current_backend="local").escalate is False


@pytest.mark.parametrize(
    "text",
    [
        "refactor the auth module across the whole codebase",
        "help me redesign the architecture of this service",
        "trace the root cause of this intermittent race condition",
        "why is the build failing only sometimes? debug it",
        "rewrite this to fix the memory leak",
        "do a security audit of the payment flow",
    ],
)
def test_complex_requests_escalate(text):
    decision = route_request(text, current_backend="local")
    assert decision.escalate is True
    assert decision.reason  # non-empty explanation


def test_very_long_request_escalates():
    long_text = "please update " + "and ".join(f"file{i}.py" for i in range(60))
    assert route_request(long_text, current_backend="local").escalate is True


def test_never_escalates_when_already_on_claude():
    # No point routing to Claude if we're already there.
    decision = route_request(
        "refactor the entire architecture", current_backend="claude"
    )
    assert decision.escalate is False


def test_reason_names_the_trigger():
    decision = route_request("refactor this", current_backend="local")
    assert "refactor" in decision.reason.lower()
