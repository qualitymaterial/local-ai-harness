"""Tests for in-chat backend-switch slash commands (/claude, /local, etc.)."""

from __future__ import annotations

import pytest

from local_ai.cli import parse_backend_command


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/claude", "claude"),
        ("/opus", "claude"),
        ("/local", "local"),
        ("/qwen", "local"),
    ],
)
def test_recognizes_backend_commands(text, expected):
    assert parse_backend_command(text) == expected


def test_is_case_insensitive():
    assert parse_backend_command("/Claude") == "claude"
    assert parse_backend_command("/LOCAL") == "local"


def test_tolerates_surrounding_whitespace():
    assert parse_backend_command("  /claude  ") == "claude"


@pytest.mark.parametrize(
    "text",
    [
        "claude",            # missing slash — it's a normal message
        "/exit",             # a different slash command
        "tell me about /claude",  # mention inside a sentence, not a command
        "",
        "/claudex",          # not an exact match
    ],
)
def test_ignores_non_commands(text):
    assert parse_backend_command(text) is None
