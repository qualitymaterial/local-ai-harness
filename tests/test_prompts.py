"""Tests for prompt/message assembly."""

from __future__ import annotations

from local_ai.prompts import build_messages


def test_no_consecutive_same_role():
    # Consecutive same-role messages make some local models (e.g. qwen2.5-coder)
    # return empty completions. Roles must alternate.
    msgs = build_messages("SYS", "CONTEXT_BODY", "REQUEST")
    roles = [m.role for m in msgs]
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"consecutive same-role messages: {roles}"


def test_starts_with_system():
    msgs = build_messages("SYSTEM_PROMPT", "ctx", "req")
    assert msgs[0].role == "system"
    assert msgs[0].content == "SYSTEM_PROMPT"


def test_only_system_and_user_roles():
    msgs = build_messages("s", "c", "r")
    assert all(m.role in ("system", "user") for m in msgs)


def test_context_and_request_both_reach_the_model():
    msgs = build_messages("SYS", "MY_CONTEXT_BLOB", "MY_USER_REQUEST")
    user_text = "\n".join(m.content for m in msgs if m.role == "user")
    assert "MY_CONTEXT_BLOB" in user_text
    assert "MY_USER_REQUEST" in user_text
