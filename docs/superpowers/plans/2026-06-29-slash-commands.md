# Custom Slash Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Claude-Code-style custom slash commands to `local-ai` — markdown prompt templates invoked as `/name args` in chat or `ai /name args` on the CLI.

**Architecture:** A new pure module `local_ai/commands.py` discovers/loads/expands command files (project `.local-ai/commands/` over global `~/.local-ai/commands/`) and provides a `resolve_slash` router. The `chat` loop and a new `cmd` CLI command wire it in; `ask`'s core is extracted into a shared `_answer_prompt` helper.

**Tech Stack:** Python 3.14, pytest, Typer (existing), Rich (existing). No new dependencies.

## Global Constraints

- Discovery: project `.local-ai/commands/<name>.md` wins over global `~/.local-ai/commands/<name>.md`. Command name = filename stem.
- `$ARGUMENTS` is replaced everywhere it appears; if absent, non-empty args are appended on a new line; empty args → `$ARGUMENTS` becomes `""`.
- Built-ins always win over command files (cannot shadow `/claude`, `/local`, etc.).
- Commands read fresh on each invocation.
- Two entry points: chat REPL `/name` and CLI `ai /name`.
- TDD; commit after each green task. Run tests: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest`.

---

### Task 1: `commands.py` pure module

**Files:**
- Create: `local_ai/commands.py`
- Test: `tests/test_commands.py`

**Interfaces:**
- Consumes: stdlib only.
- Produces:
  - `CommandInfo(name: str, description: str, scope: str)`
  - `_global_commands_dir() -> Path`
  - `command_dirs(repo_root: Path) -> list[tuple[str, Path]]`
  - `find_command(name: str, repo_root: Path) -> Optional[Path]`
  - `load_command(path: Path) -> tuple[str, str]`  # (description, body)
  - `expand_command(body: str, arguments: str) -> str`
  - `list_commands(repo_root: Path) -> list[CommandInfo]`
  - `resolve_slash(user_input: str, builtins: set[str], repo_root: Path) -> tuple[str, str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_commands.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest tests/test_commands.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'local_ai.commands'`

- [ ] **Step 3: Write the module**

Create `local_ai/commands.py`:

```python
"""Custom slash commands for local-ai (Claude-Code-style prompt expansion).

A command is a markdown file whose body is a prompt template. `/name [args]`
in chat — or `ai /name [args]` on the CLI — expands the template (substituting
$ARGUMENTS) and runs it as the user's turn. Files live in a repo's
.local-ai/commands/ (project, wins) and ~/.local-ai/commands/ (global).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ARGUMENTS_PLACEHOLDER = "$ARGUMENTS"


@dataclass
class CommandInfo:
    name: str
    description: str
    scope: str  # "project" or "global"


def _global_commands_dir() -> Path:
    return Path.home() / ".local-ai" / "commands"


def command_dirs(repo_root: Path) -> list[tuple[str, Path]]:
    """Command dirs in precedence order: project first, then global."""
    return [
        ("project", repo_root / ".local-ai" / "commands"),
        ("global", _global_commands_dir()),
    ]


def find_command(name: str, repo_root: Path) -> Optional[Path]:
    """Path to <name>.md in the first dir (project, then global) that has it."""
    for _scope, directory in command_dirs(repo_root):
        candidate = directory / f"{name}.md"
        if candidate.is_file():
            return candidate
    return None


def load_command(path: Path) -> tuple[str, str]:
    """Return (description, body). Parses optional leading YAML frontmatter."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ("", "")
    description = ""
    body = raw
    lines = raw.split("\n")
    if lines and lines[0].strip() == "---":
        end = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end = i
                break
        if end is not None:
            for line in lines[1:end]:
                if line.strip().lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"').strip("'")
                    break
            body = "\n".join(lines[end + 1:])
    body = body.strip()
    if not description:
        for line in body.split("\n"):
            if line.strip():
                description = line.strip()
                break
    return (description, body)


def expand_command(body: str, arguments: str) -> str:
    """Substitute $ARGUMENTS; if absent, append non-empty args on a new line."""
    if ARGUMENTS_PLACEHOLDER in body:
        return body.replace(ARGUMENTS_PLACEHOLDER, arguments)
    if arguments:
        return body + "\n\n" + arguments
    return body


def list_commands(repo_root: Path) -> list[CommandInfo]:
    """All commands, project overriding global on name, sorted by name."""
    seen: dict[str, CommandInfo] = {}
    for scope, directory in command_dirs(repo_root):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            name = path.stem
            if name in seen:
                continue  # project (scanned first) wins
            description, _body = load_command(path)
            seen[name] = CommandInfo(name=name, description=description, scope=scope)
    return sorted(seen.values(), key=lambda c: c.name)


def resolve_slash(
    user_input: str, builtins: set[str], repo_root: Path
) -> tuple[str, str]:
    """Route a chat input.

    Returns (kind, payload):
      ("message", user_input) — not a slash command
      ("builtin", name)       — a built-in (checked before command files)
      ("expand", expanded)    — a command file matched; payload is the prompt
      ("unknown", name)       — slash but no built-in / command file
      ("empty", name)         — command file exists but body is empty
    """
    if not user_input.startswith("/"):
        return ("message", user_input)
    rest = user_input[1:]
    parts = rest.split(None, 1)
    name = parts[0] if parts else ""
    arguments = parts[1] if len(parts) > 1 else ""
    if name.lower() in builtins:
        return ("builtin", name.lower())
    path = find_command(name, repo_root)
    if path is None:
        return ("unknown", name)
    _description, body = load_command(path)
    if not body.strip():
        return ("empty", name)
    return ("expand", expand_command(body, arguments))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest tests/test_commands.py -q`
Expected: PASS (17 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/briananderson/local-ai
git add local_ai/commands.py tests/test_commands.py
git commit -m "feat: custom slash command module (discovery, load, expand, resolve)"
```

---

### Task 2: Wire commands into the `chat` REPL

**Files:**
- Modify: `local_ai/cli.py` — add `_BUILTIN_SLASH_NAMES`; in the `chat` loop add custom-command resolution; extend `/help`.

**Interfaces:**
- Consumes: `resolve_slash`, `list_commands` from `local_ai.commands` (Task 1).
- Produces: no new public API; behavior change only.

- [ ] **Step 1: Add the built-in names set**

In `local_ai/cli.py`, immediately after the line `console = Console()` (top-level, near `err_console = Console(stderr=True)`), add:

```python
# Chat built-in slash command names (without the slash). A command file cannot
# shadow these — they are handled directly by the chat loop.
_BUILTIN_SLASH_NAMES = {
    "claude", "opus", "local", "qwen", "cost", "spend",
    "help", "h", "?", "exit", "quit", "q",
}
```

- [ ] **Step 2: Extend `/help` to list custom commands**

In the `chat` command, find the `/help` handler block:

```python
        if user_input.lower() in ("/help", "/?", "/h"):
            console.print(
                "[dim]Commands:[/dim]\n"
                "  [bold]/claude[/bold] or [bold]/opus[/bold]  — escalate this session to Claude (cloud)\n"
                "  [bold]/local[/bold] or [bold]/qwen[/bold]   — switch back to the local model\n"
                "  [bold]/cost[/bold]               — show Claude spend so far this session\n"
                "  [bold]/help[/bold]               — show this help\n"
                "  [bold]exit[/bold]                — quit (session is saved)\n"
            )
            continue
```

Replace it with (adds the custom-command listing before `continue`):

```python
        if user_input.lower() in ("/help", "/?", "/h"):
            console.print(
                "[dim]Commands:[/dim]\n"
                "  [bold]/claude[/bold] or [bold]/opus[/bold]  — escalate this session to Claude (cloud)\n"
                "  [bold]/local[/bold] or [bold]/qwen[/bold]   — switch back to the local model\n"
                "  [bold]/cost[/bold]               — show Claude spend so far this session\n"
                "  [bold]/help[/bold]               — show this help\n"
                "  [bold]exit[/bold]                — quit (session is saved)\n"
            )
            from .commands import list_commands

            _cmds = list_commands(root)
            if _cmds:
                console.print("[dim]Custom commands:[/dim]")
                for _c in _cmds:
                    console.print(
                        f"  [bold]/{_c.name}[/bold]  — {_c.description}  [dim]({_c.scope})[/dim]"
                    )
            continue
```

- [ ] **Step 3: Add custom-command resolution before message handling**

In the `chat` loop, find the auto-routing block that begins with this comment:

```python
        # Auto-routing: offer to escalate complex turns to Claude (opt-in via auto_route).
```

Insert the following block IMMEDIATELY BEFORE that comment line (so built-ins handled above still win, and an expanded command then flows through auto-route + generation):

```python
        # Custom slash commands (project/global .local-ai/commands/*.md).
        if user_input.startswith("/"):
            from .commands import resolve_slash

            _kind, _payload = resolve_slash(user_input, _BUILTIN_SLASH_NAMES, root)
            if _kind == "unknown":
                console.print(f"[yellow]Unknown command: /{_payload} (try /help)[/yellow]\n")
                continue
            if _kind == "empty":
                console.print(f"[yellow]Command '/{_payload}' is empty.[/yellow]\n")
                continue
            if _kind == "expand":
                user_input = _payload  # fall through to normal message handling

```

- [ ] **Step 4: Run the full suite + import check**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest -q`
Expected: PASS (all existing + Task 1's 17).
Run: `cd /Users/briananderson/local-ai && .venv/bin/python -c "from local_ai import cli; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
cd /Users/briananderson/local-ai
git add local_ai/cli.py
git commit -m "feat: custom slash commands in chat REPL + /help listing"
```

---

### Task 3: CLI one-shot (`ai /name`) + extract `_answer_prompt`

**Files:**
- Modify: `local_ai/cli.py` — extract `_answer_prompt` from `ask`; add `cmd` command; add `_rewrite_cli_args` and use it in `main()`.
- Test: `tests/test_cli_args.py` (pure argv-rewrite logic).

**Interfaces:**
- Consumes: `find_command`, `load_command`, `expand_command` from `local_ai.commands` (Task 1).
- Produces: `_rewrite_cli_args(args: list[str], known_commands: set[str]) -> list[str]` (pure, tested); `_answer_prompt(root, config, prompt) -> str`; a new Typer `cmd` command.

- [ ] **Step 1: Write the failing argv-rewrite tests**

Create `tests/test_cli_args.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest tests/test_cli_args.py -q`
Expected: FAIL — `ImportError: cannot import name '_rewrite_cli_args'`

> Note on `test_slash_with_leading_flag`: `-C /tmp` here is a flag plus its value; the first non-flag token is `/review`. `_rewrite_cli_args` finds the first token starting with `/` and rewrites at that index, leaving the leading flags in place. The `cmd` Typer command's own `-C/--dir` option still works when the flag comes before the slash because Typer parses options anywhere; this test only pins the rewrite output.

- [ ] **Step 3: Add `_rewrite_cli_args` and rewrite `main()`**

In `local_ai/cli.py`, find the existing `main` function:

```python
def main() -> None:
    """Entry point — bare `ai` drops into conversational chat REPL."""
    import sys
    _COMMANDS = {
        "index", "review", "ask", "plan", "patch",
        "apply", "diff", "run", "config", "chat", "embed",
    }
    non_flags = [a for a in sys.argv[1:] if not a.startswith("-")]
    help_requested = bool({"--help", "-h"} & set(sys.argv[1:]))

    if not non_flags and not help_requested:
        # Bare `ai` → drop into conversational chat
        sys.argv.insert(1, "chat")
    elif non_flags and non_flags[0] not in _COMMANDS:
        # Words but no known command → treat as a one-shot question
        idx = next(i for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("-"))
        sys.argv.insert(idx, "ask")
    app()
```

Replace the ENTIRE function with:

```python
def _rewrite_cli_args(args: list[str], known_commands: set[str]) -> list[str]:
    """Map bare/slash invocations to subcommands. Operates on argv[1:].

    - no positional args → `chat`
    - first non-flag starts with `/` → `cmd <name> ...` (custom command)
    - first non-flag is not a known command → `ask ...`
    - otherwise unchanged
    """
    non_flags = [a for a in args if not a.startswith("-")]
    help_requested = bool({"--help", "-h"} & set(args))

    if not non_flags and not help_requested:
        return ["chat", *args]
    if non_flags and non_flags[0].startswith("/"):
        out = list(args)
        idx = next(i for i, a in enumerate(out) if a.startswith("/"))
        out[idx] = out[idx][1:]  # strip leading slash → command name
        out.insert(idx, "cmd")
        return out
    if non_flags and non_flags[0] not in known_commands:
        out = list(args)
        idx = next(i for i, a in enumerate(out) if not a.startswith("-"))
        out.insert(idx, "ask")
        return out
    return args


def main() -> None:
    """Entry point — bare `ai` → chat; `ai /name` → custom command; words → ask."""
    import sys

    _COMMANDS = {
        "index", "review", "ask", "plan", "patch",
        "apply", "diff", "run", "config", "chat", "embed", "cmd",
    }
    sys.argv[1:] = _rewrite_cli_args(sys.argv[1:], _COMMANDS)
    app()
```

- [ ] **Step 4: Run argv tests to verify pass**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest tests/test_cli_args.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Extract `_answer_prompt` from `ask`**

In `local_ai/cli.py`, find the current `ask` command:

```python
@app.command()
def ask(
    path: str = typer.Option(".", "-C", "--dir", help="Repo directory (default: current)."),
    question: list[str] = typer.Argument(..., help="Your question about the repo."),
) -> None:
    """Answer a question about the repo, selecting relevant files automatically."""
    question = " ".join(question)
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    index_result = _get_index(root)

    force = _semantic_files(root, config, question)
    packet = ctx.build_context(
        index_result, question,
        max_context_tokens=config.max_context_tokens,
        force_include=force,
    )
    _print_context_summary(packet)

    output = _call_model(config, prompts.ASK_SYSTEM, packet.body, question, repo_root=root)

    mem = Memory.load(root)
    mem.record("ask", question=question[:200])
    mem.save()

    if not config.stream:
        console.print(Panel(output, title="Answer", border_style="cyan"))
```

Replace it with a shared helper plus the slimmed `ask`:

```python
def _answer_prompt(root: Path, config: Config, prompt: str) -> str:
    """Run a prompt through the standard ask pipeline (RAG context + model)."""
    index_result = _get_index(root)
    force = _semantic_files(root, config, prompt)
    packet = ctx.build_context(
        index_result, prompt,
        max_context_tokens=config.max_context_tokens,
        force_include=force,
    )
    _print_context_summary(packet)
    output = _call_model(config, prompts.ASK_SYSTEM, packet.body, prompt, repo_root=root)
    if not config.stream:
        console.print(Panel(output, title="Answer", border_style="cyan"))
    return output


@app.command()
def ask(
    path: str = typer.Option(".", "-C", "--dir", help="Repo directory (default: current)."),
    question: list[str] = typer.Argument(..., help="Your question about the repo."),
) -> None:
    """Answer a question about the repo, selecting relevant files automatically."""
    question = " ".join(question)
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    _answer_prompt(root, config, question)
    mem = Memory.load(root)
    mem.record("ask", question=question[:200])
    mem.save()
```

- [ ] **Step 6: Add the `cmd` command**

In `local_ai/cli.py`, add a new command. Place it immediately AFTER the `ask` command definition (after the slimmed `ask` from Step 5):

```python
@app.command()
def cmd(
    name: str = typer.Argument(..., help="Command name (without the slash)."),
    args: Optional[list[str]] = typer.Argument(None, help="Arguments ($ARGUMENTS)."),
    path: str = typer.Option(".", "-C", "--dir", help="Repo directory (default: current)."),
) -> None:
    """Run a custom slash command from .local-ai/commands/ (project or global)."""
    from .commands import expand_command, find_command, load_command

    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)

    cmd_path = find_command(name, root)
    if cmd_path is None:
        err_console.print(f"[red]Unknown command:[/red] /{name}")
        raise typer.Exit(code=2)
    _description, body = load_command(cmd_path)
    if not body.strip():
        err_console.print(f"[red]Command '/{name}' is empty.[/red]")
        raise typer.Exit(code=1)

    prompt = expand_command(body, " ".join(args or []))
    _answer_prompt(root, config, prompt)
    mem = Memory.load(root)
    mem.record("cmd", name=name)
    mem.save()
```

- [ ] **Step 7: Run the full suite + import check**

Run: `cd /Users/briananderson/local-ai && .venv/bin/python -m pytest -q`
Expected: PASS (all existing + Task 1's 17 + Task 3's 5).
Run: `cd /Users/briananderson/local-ai && .venv/bin/python -c "from local_ai import cli; print('ok')"`
Expected: `ok`

- [ ] **Step 8: Live end-to-end check (both paths)**

LM Studio must be running with `qwen/qwen2.5-coder-14b` loaded.

```bash
mkdir -p ~/.local-ai/commands
printf 'Answer in British spelling: $ARGUMENTS\n' > ~/.local-ai/commands/British.md
mkdir -p /tmp/cmdtest && echo "def f(): pass" > /tmp/cmdtest/sample.py
cd /Users/briananderson/local-ai
# CLI one-shot:
.venv/bin/python -m local_ai.cli /British -C /tmp/cmdtest "describe colour and optimization"
# Unknown command errors:
.venv/bin/python -m local_ai.cli /bogus -C /tmp/cmdtest "x"; echo "exit=$?"
```
Expected: the British command expands and runs (an answer prints); `/bogus` prints "Unknown command: /bogus" and exits non-zero. Clean up: `rm -rf /tmp/cmdtest`. (Leave `~/.local-ai/commands/British.md` or remove it — your choice.)

- [ ] **Step 9: Commit**

```bash
cd /Users/briananderson/local-ai
git add local_ai/cli.py tests/test_cli_args.py
git commit -m "feat: ai /name one-shot CLI commands; extract _answer_prompt"
```

---

## Self-Review

- **Spec coverage:** module functions (Task 1), `resolve_slash` routing incl. precedence/unknown/empty (Task 1 tests), chat wiring + `/help` listing (Task 2), CLI `ai /name` + `cmd` + `_answer_prompt` refactor + `main()` slash routing (Task 3), `$ARGUMENTS` semantics (Task 1 `expand_command` tests), project-over-global (Task 1 `find_command`/`list_commands` tests), both live paths (Task 3 step 8). All spec sections covered.
- **Placeholders:** none — every code/test step has complete code and exact commands.
- **Type consistency:** `CommandInfo(name, description, scope)`, `find_command`, `load_command` (→ `(description, body)`), `expand_command`, `list_commands`, `resolve_slash` (→ `(kind, payload)`), `_rewrite_cli_args`, `_answer_prompt` — names/signatures consistent between Task 1 definitions and Task 2/3 usage. `_BUILTIN_SLASH_NAMES` (Task 2) passed to `resolve_slash` (Task 1). `cmd` added to `_COMMANDS` (Task 3) matching the `cmd` Typer command name.
