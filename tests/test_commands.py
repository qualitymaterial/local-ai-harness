"""Tests for custom slash commands."""

from __future__ import annotations

from local_ai import commands
from local_ai.commands import (
    expand_command,
    find_command,
    list_commands,
    load_command,
    resolve_slash,
)


# ---- expand_command ----
def test_expand_replaces_single():
    assert expand_command("do $ARGUMENTS now", "X") == "do X now"


def test_expand_replaces_multiple():
    assert expand_command("$ARGUMENTS and $ARGUMENTS", "Y") == "Y and Y"


def test_expand_appends_when_no_placeholder():
    assert expand_command("Review the code.", "auth.py") == "Review the code.\n\nauth.py"


def test_expand_no_placeholder_empty_args_unchanged():
    assert expand_command("Review.", "") == "Review."


def test_expand_placeholder_empty_args_becomes_empty():
    assert expand_command("do $ARGUMENTS", "") == "do "


# ---- find_command ----
def test_find_prefers_project(tmp_path, monkeypatch):
    g = tmp_path / "g"
    g.mkdir()
    (g / "review.md").write_text("global review")
    monkeypatch.setattr(commands, "_global_commands_dir", lambda: g)
    proj = tmp_path / "repo"
    pc = proj / ".local-ai" / "commands"
    pc.mkdir(parents=True)
    (pc / "review.md").write_text("project review")
    assert find_command("review", proj) == pc / "review.md"


def test_find_global_when_only_global(tmp_path, monkeypatch):
    g = tmp_path / "g"
    g.mkdir()
    (g / "x.md").write_text("g")
    monkeypatch.setattr(commands, "_global_commands_dir", lambda: g)
    assert find_command("x", tmp_path / "repo") == g / "x.md"


def test_find_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "_global_commands_dir", lambda: tmp_path / "nope")
    assert find_command("ghost", tmp_path / "repo") is None


# ---- load_command ----
def test_load_parses_frontmatter_description(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("---\ndescription: Do a thing\n---\nThe body $ARGUMENTS\n")
    desc, body = load_command(p)
    assert desc == "Do a thing"
    assert body == "The body $ARGUMENTS"


def test_load_description_falls_back_to_first_line(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("Summarize the diff.\nMore detail.\n")
    desc, body = load_command(p)
    assert desc == "Summarize the diff."
    assert body == "Summarize the diff.\nMore detail."


# ---- list_commands ----
def test_list_merges_project_over_global_sorted(tmp_path, monkeypatch):
    g = tmp_path / "g"
    g.mkdir()
    (g / "review.md").write_text("global review")
    (g / "alpha.md").write_text("alpha desc")
    monkeypatch.setattr(commands, "_global_commands_dir", lambda: g)
    proj = tmp_path / "repo"
    pc = proj / ".local-ai" / "commands"
    pc.mkdir(parents=True)
    (pc / "review.md").write_text("project review")
    infos = list_commands(proj)
    assert [c.name for c in infos] == ["alpha", "review"]
    review = next(c for c in infos if c.name == "review")
    assert review.scope == "project"
    assert review.description == "project review"


# ---- resolve_slash ----
def _isolate_global(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "_global_commands_dir", lambda: tmp_path / "no_global")


def test_resolve_message_for_non_slash(tmp_path):
    assert resolve_slash("hello there", {"claude"}, tmp_path) == ("message", "hello there")


def test_resolve_builtin(tmp_path):
    assert resolve_slash("/claude", {"claude"}, tmp_path) == ("builtin", "claude")


def test_resolve_expand(tmp_path, monkeypatch):
    _isolate_global(tmp_path, monkeypatch)
    pc = tmp_path / ".local-ai" / "commands"
    pc.mkdir(parents=True)
    (pc / "myreview.md").write_text("Review this: $ARGUMENTS")
    kind, payload = resolve_slash("/myreview the auth code", {"claude"}, tmp_path)
    assert kind == "expand"
    assert payload == "Review this: the auth code"


def test_resolve_unknown(tmp_path, monkeypatch):
    _isolate_global(tmp_path, monkeypatch)
    assert resolve_slash("/nope", {"claude"}, tmp_path) == ("unknown", "nope")


def test_resolve_empty(tmp_path, monkeypatch):
    _isolate_global(tmp_path, monkeypatch)
    pc = tmp_path / ".local-ai" / "commands"
    pc.mkdir(parents=True)
    (pc / "blank.md").write_text("   \n  ")
    assert resolve_slash("/blank", {"claude"}, tmp_path) == ("empty", "blank")
