"""Per-repo project instructions (AGENTS.md / CLAUDE.md) for local-ai.

Auto-injects a repo's instructions file into the model's system prompt, like
Claude Code's CLAUDE.md. AGENTS.md is preferred; CLAUDE.md is the fallback.
The text is capped so it doesn't crowd the context window (RAG already fills
most of it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Discovery order: first match wins.
INSTRUCTION_FILENAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
DEFAULT_MAX_CHARS = 6000


@dataclass
class LoadedInstructions:
    text: str
    source: str  # filename, e.g. "AGENTS.md"


def find_instructions_file(repo_root: Path) -> Optional[Path]:
    """Return the repo's instructions file by discovery order, or None."""
    for name in INSTRUCTION_FILENAMES:
        candidate = repo_root / name
        if candidate.is_file():
            return candidate
    return None


def load_instructions(
    repo_root: Path, *, max_chars: int = DEFAULT_MAX_CHARS
) -> Optional[LoadedInstructions]:
    """Load the repo's instructions, capped at max_chars. None if absent/empty/unreadable."""
    path = find_instructions_file(repo_root)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + f"\n\n[... {path.name} truncated to fit context ...]"
    return LoadedInstructions(text=text, source=path.name)


def augment_system(base_system: str, loaded: Optional[LoadedInstructions]) -> str:
    """Append project instructions to a base system prompt (unchanged if None)."""
    if loaded is None:
        return base_system
    return (
        base_system
        + "\n\n# Project Instructions (from "
        + loaded.source
        + ")\n"
        + loaded.text
    )
