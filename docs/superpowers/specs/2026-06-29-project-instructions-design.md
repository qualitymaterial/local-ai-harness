# Design: Project Instructions for local-ai

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending implementation plan

## Goal

Give `local-ai` a Claude-Code-style project-instructions mechanism: a per-repo
instructions file that is automatically injected into the model's system prompt
for every command. Today `local-ai` ignores `CLAUDE.md`/`AGENTS.md` entirely;
project guidance only reaches the model if RAG happens to rank the file relevant,
and even then it's treated as ordinary source, not instructions.

This is the foundation for a possible future "skills" feature; skills are out of
scope here.

## Scope decisions (from brainstorm)

- **Per-repo only.** No global/user-level instructions file in v1 (the existing
  `~/.claude/CLAUDE.md` is vault/ingestion-focused and would add noise).
- **File discovery order:** `AGENTS.md` (preferred) → `CLAUDE.md` (fallback) →
  none. If `AGENTS.md` exists, `CLAUDE.md` is ignored. This auto-works in
  existing repos that already have a `CLAUDE.md`, while letting a repo add an
  `AGENTS.md` to give `local-ai` different instructions than Claude Code.
- **No new config** in v1 — discovery order is fixed (easy to make configurable
  later).

## Architecture

### New module: `local_ai/instructions.py` (pure, no I/O beyond reading the file)

- `find_instructions_file(repo_root: Path) -> Path | None`
  Returns `repo_root/AGENTS.md` if it exists, else `repo_root/CLAUDE.md` if it
  exists, else `None`. Only looks at the repo root (no parent-dir walk in v1).

- `load_instructions(repo_root: Path, *, max_chars: int = 6000) -> LoadedInstructions | None`
  where `LoadedInstructions` is a small dataclass `(text: str, source: str)`
  (`source` is the filename, e.g. `"AGENTS.md"`). Finds the file, reads it as
  UTF-8, strips, and returns `LoadedInstructions(text, filename)`. Returns
  `None` when there is no file, the file is empty/whitespace-only, or it cannot
  be read/decoded. If the text exceeds `max_chars`, `text` is the head truncated
  to `max_chars` plus a trailing marker:
  `\n\n[... <filename> truncated to fit context ...]`.
  Rationale for the cap: RAG + repo context already consume ~10–12k of the 16k
  loaded context window, so instructions must stay bounded (~1500 tokens).

- `augment_system(base_system: str, loaded: LoadedInstructions | None) -> str`
  If `loaded` is `None`, returns `base_system` unchanged. Otherwise returns:
  ```
  <base_system>

  # Project Instructions (from <loaded.source>)
  <loaded.text>
  ```
  The section header names the source file via `loaded.source`. This is the
  single function both wiring spots call.

### Wiring (two spots cover all commands)

1. **`_call_model(config, system, context_body, request, *, repo_root)`** in
   `cli.py` — when `repo_root` is provided, load instructions for that root and
   wrap `system` via `augment_system` before building messages. This covers
   `ask`, `plan`, `patch`, `review`, `diff`, and `run` (diagnosis).
2. **`chat` command** — when assembling the session's system content, apply
   `augment_system` to `CHAT_SYSTEM` (before appending the repo context section).

Both spots call the same `augment_system`, so behavior is consistent and the
logic is unit-tested once.

### Visibility

When instructions are loaded, print a dim status line, e.g.:
`[project instructions: AGENTS.md, ~1.2k tokens]`
so the user knows project guidance is active for the session/command.

## Edge cases

- Neither file exists → no change; system prompt unchanged.
- Empty / whitespace-only file → treated as none (no empty section injected).
- Unreadable / binary / non-UTF-8 file → skipped silently (defensive, mirrors
  the context reader's behavior).
- Oversized file → head kept up to `max_chars` + truncation marker.

## Testing (TDD)

`tests/test_instructions.py`, all using `tmp_path` (no model calls):

1. `AGENTS.md` present (alongside `CLAUDE.md`) → `find_instructions_file` returns
   `AGENTS.md` (precedence).
2. Only `CLAUDE.md` present → returns `CLAUDE.md`.
3. Neither present → `find_instructions_file` returns `None`; `load_instructions`
   returns `None`.
4. Oversized file → `load_instructions` result length ≤ cap region and ends with
   the truncation marker.
5. Empty / whitespace-only file → `load_instructions` returns `None`.
6. `augment_system`:
   - with a `LoadedInstructions(text, source)` → result contains both
     `base_system` and `text` under a `# Project Instructions (from <source>)`
     header;
   - with `None` → returns `base_system` unchanged.

### Post-implementation live check (manual, not in suite)

Drop an `AGENTS.md` containing a distinctive directive (e.g. "Always answer in
British spelling") into a test repo, run `ai ask "..."`, and confirm the model
complies — proving the instruction reaches the model end to end.

## Out of scope

- Skills / progressive disclosure (separate future feature).
- Global/user-level instructions file.
- Nested/monorepo instruction merging (parent-dir walk).
- Configurable discovery order or filenames.
