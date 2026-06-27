"""Tests for ranking and context building."""

from __future__ import annotations

from pathlib import Path

from local_ai import context_builder as ctx
from local_ai import scanner
from local_ai.file_ranker import rank_files


def _write(root: Path, rel: str, content: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    _write(tmp_path, "package.json", '{"dependencies": {"react": "18"}}')
    _write(tmp_path, "src/pages/Home.tsx", "export function Home() { return null }")
    _write(tmp_path, "src/components/Footer.tsx", "export const Footer = () => null")
    _write(tmp_path, "src/styles/mobile.css", "@media (max-width: 600px) { body {} }")
    _write(tmp_path, "src/auth/login.ts", "export function login() {}")
    return tmp_path


def test_ranking_prefers_relevant_filenames(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    index = scanner.scan_repo(root)
    ranked = rank_files(index, "Where is the homepage component?")
    top = [r.entry.rel_path for r in ranked[:3]]
    assert "src/pages/Home.tsx" in top


def test_ranking_concept_expansion_mobile(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    index = scanner.scan_repo(root)
    ranked = rank_files(index, "Fix the mobile layout issue")
    top = [r.entry.rel_path for r in ranked[:3]]
    assert "src/styles/mobile.css" in top


def test_context_respects_budget(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    # A big file that should be summarized / possibly skipped under a tiny budget.
    _write(root, "src/huge.ts", "// big\n" + ("const a = 1;\n" * 5000))
    index = scanner.scan_repo(root)
    packet = ctx.build_context(index, "huge file", max_context_tokens=1200)
    assert packet.approx_tokens <= 1200
    # Repo map is always present.
    assert "Repo Map" in packet.body


def test_context_includes_repo_map_and_files(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    index = scanner.scan_repo(root)
    packet = ctx.build_context(index, "homepage", max_context_tokens=8000)
    assert "Repo Map" in packet.body
    assert "src/pages/Home.tsx" in packet.included_files
    # Included files are listed in the body as fenced blocks.
    assert "src/pages/Home.tsx" in packet.body


def test_estimate_tokens_monotonic() -> None:
    assert ctx.estimate_tokens("a" * 400) > ctx.estimate_tokens("a" * 40)
