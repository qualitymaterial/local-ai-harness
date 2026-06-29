"""Tests for per-repo project instructions loading."""

from __future__ import annotations

from local_ai.instructions import (
    LoadedInstructions,
    augment_system,
    find_instructions_file,
    load_instructions,
)


def test_agents_md_takes_precedence(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents rules")
    (tmp_path / "CLAUDE.md").write_text("claude rules")
    assert find_instructions_file(tmp_path).name == "AGENTS.md"


def test_falls_back_to_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("claude rules")
    assert find_instructions_file(tmp_path).name == "CLAUDE.md"


def test_none_when_no_file(tmp_path):
    assert find_instructions_file(tmp_path) is None
    assert load_instructions(tmp_path) is None


def test_load_returns_stripped_text_and_source(tmp_path):
    (tmp_path / "AGENTS.md").write_text("  use tabs, not spaces  ")
    loaded = load_instructions(tmp_path)
    assert loaded.text == "use tabs, not spaces"
    assert loaded.source == "AGENTS.md"


def test_empty_or_whitespace_file_is_none(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n   \n")
    assert load_instructions(tmp_path) is None


def test_oversized_file_is_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 9000)
    loaded = load_instructions(tmp_path, max_chars=6000)
    assert "truncated to fit context" in loaded.text
    assert len(loaded.text) < 9000


def test_augment_with_none_returns_base_unchanged():
    assert augment_system("BASE_SYSTEM", None) == "BASE_SYSTEM"


def test_augment_includes_base_and_instructions():
    loaded = LoadedInstructions(text="ALWAYS BRITISH SPELLING", source="AGENTS.md")
    out = augment_system("BASE_SYSTEM", loaded)
    assert "BASE_SYSTEM" in out
    assert "ALWAYS BRITISH SPELLING" in out
    assert "# Project Instructions (from AGENTS.md)" in out
