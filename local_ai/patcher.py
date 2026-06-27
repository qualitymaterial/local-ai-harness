"""Extract, inspect, and apply unified diffs produced by the model."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_DIFF_GIT_RE = re.compile(r"^diff --git ", re.MULTILINE)
_FILE_HEADER_RE = re.compile(r"^\+\+\+ [ab]/(.+)$", re.MULTILINE)
_OLD_FILE_RE = re.compile(r"^--- [ab]/(.+)$", re.MULTILINE)


@dataclass
class PatchStats:
    files: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


def extract_diff(model_output: str) -> Optional[str]:
    """Pull a unified diff out of model output.

    Strategy: prefer a fenced ```diff block that actually contains `diff --git` or
    `---`/`+++` headers; otherwise fall back to scanning the raw text for a diff body.
    """
    candidates: list[str] = []
    for match in _DIFF_FENCE_RE.finditer(model_output):
        candidates.append(match.group(1))
    # Also consider the raw text in case the model didn't fence it.
    candidates.append(model_output)

    for cand in candidates:
        text = cand.strip("\n")
        if _looks_like_diff(text):
            return _normalize(text)
    return None


def _looks_like_diff(text: str) -> bool:
    if _DIFF_GIT_RE.search(text):
        return True
    # A minimal valid hunk needs ---/+++ and an @@ marker.
    return ("\n+++ " in text or text.startswith("+++ ")) and "@@" in text


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def diff_stats(diff_text: str) -> PatchStats:
    stats = PatchStats()
    files = _FILE_HEADER_RE.findall(diff_text)
    if not files:
        files = _OLD_FILE_RE.findall(diff_text)
    # Filter /dev/null and dedup.
    seen: set[str] = set()
    for f in files:
        if f == "/dev/null":
            continue
        if f not in seen:
            seen.add(f)
            stats.files.append(f)

    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            stats.additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats.deletions += 1
    return stats


def git_available() -> bool:
    return shutil.which("git") is not None


def _is_git_repo(repo_root: Path) -> bool:
    if not git_available():
        return False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except OSError:
        return False


@dataclass
class ApplyResult:
    ok: bool
    method: str  # "git-apply" | "git-apply-check" | "none"
    message: str


def check_apply(repo_root: Path, diff_path: Path) -> ApplyResult:
    """Dry-run the patch with `git apply --check` without modifying files."""
    if not _is_git_repo(repo_root):
        return ApplyResult(
            ok=False,
            method="none",
            message="Not a git repository (or git unavailable); cannot safely apply.",
        )
    result = subprocess.run(
        ["git", "apply", "--check", "--verbose", str(diff_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return ApplyResult(ok=True, method="git-apply-check", message="Patch applies cleanly.")
    return ApplyResult(
        ok=False,
        method="git-apply-check",
        message=(result.stderr or result.stdout or "git apply --check failed").strip(),
    )


def apply_patch(repo_root: Path, diff_path: Path) -> ApplyResult:
    """Apply the patch atomically with git apply. Never partially modifies files."""
    if not _is_git_repo(repo_root):
        return ApplyResult(
            ok=False,
            method="none",
            message="Not a git repository (or git unavailable); refusing to apply.",
        )
    # git apply is atomic: it validates all hunks before writing.
    result = subprocess.run(
        ["git", "apply", "--verbose", str(diff_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return ApplyResult(ok=True, method="git-apply", message="Patch applied successfully.")
    return ApplyResult(
        ok=False,
        method="git-apply",
        message=(result.stderr or result.stdout or "git apply failed").strip(),
    )
