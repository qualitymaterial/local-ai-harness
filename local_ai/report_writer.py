"""Persist reports, plans, patches, and run logs under .local-ai/ with timestamps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import workspace_dir

# Subdirectory names under .local-ai/
REPORTS = "reports"
PLANS = "plans"
PATCHES = "patches"
RUNS = "runs"


def timestamp() -> str:
    """Filesystem-safe local timestamp, e.g. 20260627_143015."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _subdir(repo_root: Path, name: str) -> Path:
    path = workspace_dir(repo_root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_repo_map(repo_root: Path, content: str) -> Path:
    ws = workspace_dir(repo_root)
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / "repo_map.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_repo_index(repo_root: Path, content: str) -> Path:
    ws = workspace_dir(repo_root)
    ws.mkdir(parents=True, exist_ok=True)
    path = ws / "repo_index.json"
    path.write_text(content, encoding="utf-8")
    return path


def write_report(repo_root: Path, content: str, ts: str | None = None) -> Path:
    ts = ts or timestamp()
    path = _subdir(repo_root, REPORTS) / f"repo_review_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_plan(repo_root: Path, content: str, ts: str | None = None) -> Path:
    ts = ts or timestamp()
    path = _subdir(repo_root, PLANS) / f"plan_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_patch(
    repo_root: Path, diff_text: str | None, explanation: str, ts: str | None = None
) -> tuple[Path | None, Path]:
    ts = ts or timestamp()
    patch_dir = _subdir(repo_root, PATCHES)
    diff_path: Path | None = None
    if diff_text:
        diff_path = patch_dir / f"patch_{ts}.diff"
        diff_path.write_text(diff_text, encoding="utf-8")
    md_path = patch_dir / f"patch_{ts}.md"
    md_path.write_text(explanation, encoding="utf-8")
    return diff_path, md_path


def write_run(repo_root: Path, content: str, ts: str | None = None) -> Path:
    ts = ts or timestamp()
    path = _subdir(repo_root, RUNS) / f"run_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return path


def context_footer(included: list[str], skipped: list[str], approx_tokens: int) -> str:
    """A standard section documenting which files were sent to the model."""
    lines = ["", "---", "", "## Context used", "", f"Approx. context tokens: ~{approx_tokens}", ""]
    lines.append("Files included in the model context:")
    if included:
        lines.extend(f"- `{rel}`" for rel in included)
    else:
        lines.append("- _(none)_")
    if skipped:
        lines.append("")
        lines.append("Relevant files dropped for budget (consider raising max_context_tokens):")
        lines.extend(f"- `{rel}`" for rel in skipped)
    lines.append("")
    return "\n".join(lines)
