# Project Instructions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-inject a repo's `AGENTS.md` (or `CLAUDE.md`) into the model's system prompt for every `local-ai` command.

**Architecture:** A new pure module `local_ai/instructions.py` discovers and loads the repo instructions file (capped for context), and an `augment_system` helper appends it to the base system prompt. Two call sites in `cli.py` (`_call_model` and the `chat` command) apply it, covering every command.

**Tech Stack:** Python 3.14, pytest, existing `local-ai` package (no new dependencies).

## Global Constraints

- Discovery order is fixed: `AGENTS.md` preferred, then `CLAUDE.md`; first match wins.
- Per-repo only (repo root = the resolved `-C`/`--dir` path). No global file, no parent-dir walk.
- Instructions text capped at 6000 chars (~1500 tokens) to protect the context window.
- No new config keys in v1.
- TDD: write the failing test first; commit after each green task.
- Run tests with the venv: `cd ~/local-ai && .venv/bin/python -m pytest`.

---

### Task 1: `instructions.py` module (pure core)

**Files:**
- Create: `local_ai/instructions.py`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `LoadedInstructions` dataclass with `text: str`, `source: str`.
  - `find_instructions_file(repo_root: Path) -> Optional[Path]`
  - `load_instructions(repo_root: Path, *, max_chars: int = 6000) -> Optional[LoadedInstructions]`
  - `augment_system(base_system: str, loaded: Optional[LoadedInstructions]) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_instructions.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/local-ai && .venv/bin/python -m pytest tests/test_instructions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_ai.instructions'`

- [ ] **Step 3: Write the module**

Create `local_ai/instructions.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/local-ai && .venv/bin/python -m pytest tests/test_instructions.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
cd ~/local-ai
git add local_ai/instructions.py tests/test_instructions.py
git commit -m "feat: project instructions loader (AGENTS.md/CLAUDE.md)"
```

---

### Task 2: Wire instructions into the CLI

**Files:**
- Modify: `local_ai/cli.py` — `_call_model` (covers `ask`/`plan`/`patch`/`review`/`diff`/`run`) and the `chat` command's new-session system assembly.
- Test: live verification (the augment/load logic is already unit-tested in Task 1; this task is thin wiring).

**Interfaces:**
- Consumes: `load_instructions`, `augment_system` from `local_ai.instructions` (Task 1).
- Produces: no new public API; behavior change only.

- [ ] **Step 1: Augment the system prompt in `_call_model`**

In `local_ai/cli.py`, find the start of `_call_model` (the function body begins by building `messages = prompts.build_messages(system, context_body, request)`). Insert BEFORE that line, at the top of the function body:

```python
    # Inject per-repo project instructions (AGENTS.md / CLAUDE.md) when we have a repo.
    if repo_root is not None:
        from .instructions import augment_system, load_instructions

        _loaded = load_instructions(repo_root)
        if _loaded:
            system = augment_system(system, _loaded)
            console.print(f"[dim][project instructions: {_loaded.source}][/dim]")
```

- [ ] **Step 2: Augment the system prompt in the `chat` command (new sessions)**

In `local_ai/cli.py`, find the `chat` command's new-session branch where it builds:

```python
        system_content = prompts.CHAT_SYSTEM + "\n\n# Repository Context\n\n" + packet.body
```

Replace that single line with:

```python
        from .instructions import augment_system, load_instructions

        _base_system = prompts.CHAT_SYSTEM
        _loaded = load_instructions(root)
        if _loaded:
            _base_system = augment_system(_base_system, _loaded)
            console.print(f"[dim][project instructions: {_loaded.source}][/dim]")
        system_content = _base_system + "\n\n# Repository Context\n\n" + packet.body
```

- [ ] **Step 3: Run the full suite to confirm nothing broke**

Run: `cd ~/local-ai && .venv/bin/python -m pytest -q`
Expected: PASS (all existing tests + 8 new from Task 1)

- [ ] **Step 4: Confirm the CLI imports cleanly**

Run: `cd ~/local-ai && .venv/bin/python -c "from local_ai import cli; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Live end-to-end check (the model actually obeys the file)**

```bash
cd ~/local-ai
printf '# Project Instructions\n\nAlways answer using British spelling (e.g. "colour", "optimise").\n' > /tmp/aitest_AGENTS.md
mkdir -p /tmp/aitest && cp /tmp/aitest_AGENTS.md /tmp/aitest/AGENTS.md
echo "def f(): pass" > /tmp/aitest/sample.py
.venv/bin/python -m local_ai.cli ask -C /tmp/aitest "Write one sentence about code colour and optimisation in this project."
```
Expected: a `[project instructions: AGENTS.md]` status line appears, and the answer uses British spelling ("colour", "optimise"). Clean up: `rm -rf /tmp/aitest /tmp/aitest_AGENTS.md`.

- [ ] **Step 6: Commit**

```bash
cd ~/local-ai
git add local_ai/cli.py
git commit -m "feat: inject project instructions into ask/chat system prompts"
```

---

## Self-Review

- **Spec coverage:** discovery order (Task 1 `find_instructions_file`), cap (Task 1 `load_instructions`), `augment_system` (Task 1), both wiring spots + visibility line (Task 2 steps 1–2), all edge cases (Task 1 tests: precedence, fallback, none, empty, oversized), live check (Task 2 step 5). All spec sections covered.
- **Placeholders:** none — every code/test step contains complete code and exact commands.
- **Type consistency:** `LoadedInstructions(text, source)`, `find_instructions_file`, `load_instructions`, `augment_system` names/signatures match between Task 1 definitions and Task 2 usage.
