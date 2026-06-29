"""Tests for CLI argv rewriting (bare/slash → subcommand)."""

from __future__ import annotations

from local_ai.cli import _rewrite_cli_args

_CMDS = {"index", "review", "ask", "plan", "patch", "apply", "diff", "run", "config", "chat", "embed", "cmd"}


def test_bare_goes_to_chat():
    assert _rewrite_cli_args([], _CMDS) == ["chat"]


def test_known_command_unchanged():
    assert _rewrite_cli_args(["chat"], _CMDS) == ["chat"]
    assert _rewrite_cli_args(["ask", "hello"], _CMDS) == ["ask", "hello"]


def test_unknown_word_becomes_ask():
    assert _rewrite_cli_args(["what", "is", "this"], _CMDS) == ["ask", "what", "is", "this"]


def test_slash_becomes_cmd():
    assert _rewrite_cli_args(["/review", "the", "auth"], _CMDS) == ["cmd", "review", "the", "auth"]


def test_slash_with_leading_flag():
    assert _rewrite_cli_args(["-C", "/tmp", "/review", "x"], _CMDS) == ["-C", "/tmp", "cmd", "review", "x"]
