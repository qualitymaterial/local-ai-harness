# Design: Custom Slash Commands for local-ai

**Date:** 2026-06-29
**Status:** Approved (brainstorm), pending implementation plan

## Goal

Add Claude-Code-style custom slash commands to `ai chat`. A command is a
markdown file whose body is a prompt template; typing `/<name> [args]` expands
the template (substituting `$ARGUMENTS`) and runs the result as the user's turn,
through the normal RAG-context + model flow. Explicit invocation means no
reliance on the 14B's weak tool-calling.

This builds on the project-instructions feature (`instructions.py`) but is
independent of it.

## Scope decisions (from brainstorm)

- **Semantics:** one-shot prompt expansion (Claude Code style), not persistent
  toggles/modes.
- **Storage:** global `~/.local-ai/commands/*.md` (available in every repo) +
  per-repo `.local-ai/commands/*.md`; **project wins** on name conflict.
- **Command name** = filename stem (`review.md` → `/review`).
- **Arguments:** `$ARGUMENTS` placeholder; if absent from the template, args are
  appended on a new line; no args → `$ARGUMENTS` becomes "".
- **Built-ins always win** — a command file cannot shadow `/claude`, `/local`,
  `/qwen`, `/opus`, `/cost`, `/spend`, `/help`, `exit`/`quit`.
- **Two entry points:** the chat REPL (`/name args`) **and** one-shot from the
  terminal as `ai /name args` (e.g. `ai /review the auth module`). Both share the
  same command files and resolution.
- Commands are read fresh on each invocation (edit a file → effect is immediate).

## Architecture

### New module: `local_ai/commands.py` (pure; only file reads)

Dataclasses:
- `CommandInfo(name: str, description: str, scope: str)` — `scope` is
  `"project"` or `"global"`; used for `/help` listing.

Functions:
- `_global_commands_dir() -> Path`
  Returns `Path.home() / ".local-ai" / "commands"`. Isolated as its own function
  so tests can monkeypatch it to a `tmp_path` instead of touching the real home.
- `command_dirs(repo_root: Path) -> list[tuple[str, Path]]`
  Returns `[("project", repo_root/".local-ai/commands"), ("global", _global_commands_dir())]`
  in precedence order (project first).

- `find_command(name: str, repo_root: Path) -> Optional[Path]`
  Returns the path to `<name>.md` in the first dir (project, then global) that
  has it; `None` if not found. Missing dirs are skipped.

- `load_command(path: Path) -> tuple[str, str]`
  Returns `(description, body)`. Parses optional leading YAML frontmatter
  delimited by `---` lines; reads `description:` if present. `body` is the
  content after the frontmatter (or the whole file if no frontmatter), stripped.
  If no `description` in frontmatter, `description` falls back to the first
  non-empty line of the body.

- `expand_command(body: str, arguments: str) -> str`
  If `body` contains `$ARGUMENTS`, replace **all** occurrences with `arguments`.
  Otherwise return `body + "\n\n" + arguments` when `arguments` is non-empty, or
  `body` unchanged when `arguments` is empty.

- `list_commands(repo_root: Path) -> list[CommandInfo]`
  Scans both dirs, builds `CommandInfo` per `*.md`, **project overrides global**
  for same name, sorted by name. Used by `/help`.

### Routing helper (testable, lives in `commands.py`)

- `resolve_slash(user_input: str, builtins: set[str], repo_root: Path) -> tuple[str, str]`
  Pure decision function the chat loop calls. Returns one of:
  - `("message", user_input)` — does not start with `/` → ordinary chat message.
  - `("builtin", name)` — `/name` (lowercased, no args) is in `builtins` → caller
    handles via existing built-in logic.
  - `("expand", expanded_text)` — `/name` matches a command file → returns the
    expanded prompt.
  - `("unknown", name)` — starts with `/` but is neither built-in nor a command.
  - `("empty", name)` — command file exists but its body is empty/whitespace.

  Built-ins are checked **before** command files (precedence). `name` is parsed
  as the first whitespace-delimited token after `/`; the remainder is `arguments`.

### Wiring in `cli.py` `chat` loop

The loop already handles built-ins (`exit`, `/help`, `/cost`, `/claude`,
`/local`, …) each with `continue`. Add, after those built-in handlers and
before the auto-route + `session.messages.append(...)` block:

```python
if user_input.startswith("/"):
    from .commands import resolve_slash
    kind, payload = resolve_slash(user_input, _BUILTIN_SLASH_NAMES, root)
    if kind == "unknown":
        console.print(f"[yellow]Unknown command: /{payload} (try /help)[/yellow]\n")
        continue
    if kind == "empty":
        console.print(f"[yellow]Command '/{payload}' is empty.[/yellow]\n")
        continue
    if kind == "expand":
        user_input = payload  # fall through to normal message handling
    # kind == "builtin" can't happen here (handled above); kind == "message" impossible (starts with /)
```

`_BUILTIN_SLASH_NAMES` is a module-level set of the existing built-in command
names (without the slash): `{"claude","opus","local","qwen","cost","spend","help","h","?","exit","quit","q"}`.
When `kind == "expand"`, `user_input` is reassigned and execution falls through
to the existing append-to-session + generate code, so the expanded prompt is
treated exactly like a typed message (RAG context, cost meter, etc. all apply).

`/help` is extended: after listing built-ins, call `list_commands(root)` and
print each custom command as `  /<name>  — <description>  (<scope>)`.

### CLI one-shot invocation (`ai /name args`)

A new Typer command runs a command from the terminal without entering chat:

```python
@app.command()
def cmd(
    name: str = typer.Argument(..., help="Command name (without the slash)."),
    args: list[str] = typer.Argument(None, help="Arguments ($ARGUMENTS)."),
    path: str = typer.Option(".", "-C", "--dir"),
) -> None:
    ...
```

`cmd` resolves the command via `find_command(name, root)` (project then global),
errors with exit code 2 if not found, errors if the body is empty, otherwise
`expand_command(body, " ".join(args or []))` and runs the expanded prompt
through the **same path as `ask`** (RAG retrieval + context + `_call_model`).

To avoid duplicating `ask`'s body, extract its core into a shared helper
`_answer_prompt(root: Path, config: Config, prompt: str) -> None` (does
`_get_index` → `_semantic_files` → `ctx.build_context` → `_print_context_summary`
→ `_call_model(config, prompts.ASK_SYSTEM, packet.body, prompt, repo_root=root)`).
Both `ask` and `cmd` call `_answer_prompt`. This is a small, in-scope refactor of
the existing `ask` command.

**Sugar in `main()`:** the existing `main()` rewrites `sys.argv` (bare `ai` →
`chat`; unknown leading word → `ask`). Add one rule **first**: if the first
non-flag token starts with `/`, rewrite `["/review", "x", "y"]` →
`["cmd", "review", "x", "y"]` (strip the slash, insert `cmd`). Add `"cmd"` to the
`_COMMANDS` set so it's recognized. `ai /name` invokes custom commands only
(chat built-ins like `/claude` are not CLI commands; an unknown name errors).

## Edge cases

- Unknown `/foo` → warning, nothing sent to the model.
- Name collides with a built-in → built-in wins (command file ignored for that name).
- Project + global both define a name → project wins (`find_command` / `list_commands`).
- Empty/whitespace body → `("empty", name)` → warning, nothing sent.
- `$ARGUMENTS` present but no args typed → replaced with "".
- Multiple `$ARGUMENTS` → all replaced.
- Commands dir absent → no custom commands, no error.
- Unreadable/bad file → treated as not found (defensive).

## Testing (TDD)

`tests/test_commands.py`, pure functions with `tmp_path` (no model calls). For
the global dir, tests pass an explicit `repo_root` and create command files
under it / a fake home via monkeypatch of the global dir helper, OR test the
project dir directly; `command_dirs` takes `repo_root` so the project dir is
fully controllable. (Global-dir resolution uses `Path.home()`; tests monkeypatch
`commands._global_commands_dir` to a `tmp_path` to avoid touching the real home.)

1. `expand_command`:
   - single `$ARGUMENTS` replaced;
   - multiple occurrences all replaced;
   - no placeholder + non-empty args → appended on a new line;
   - no placeholder + empty args → body unchanged;
   - `$ARGUMENTS` + empty args → becomes "".
2. `find_command`: project path returned when both project and global have it
   (precedence); global when only global; `None` when neither.
3. `load_command`:
   - frontmatter `description:` parsed and body excludes the frontmatter;
   - no frontmatter → description = first non-empty line; body = whole file.
4. `list_commands`: merges global + project, project overrides same name, sorted
   by name, `scope` set correctly.
5. `resolve_slash`:
   - non-slash input → `("message", input)`;
   - `/claude` (a built-in) → `("builtin", "claude")`;
   - `/myreview extra args` where `myreview.md` exists with `$ARGUMENTS` →
     `("expand", <expanded with "extra args">)`;
   - `/nope` with no file → `("unknown", "nope")`;
   - empty command file → `("empty", name)`.

**CLI path:** `cmd` reuses the already-tested `find_command` / `load_command` /
`expand_command` plus the shared `_answer_prompt` helper, so it needs no new pure
tests. The `main()` slash rewrite is argv manipulation covered by the live check
below; keep the rewrite logic to the single documented rule so it stays trivially
correct.

### Post-implementation live check (manual)

Create `~/.local-ai/commands/British.md` with body
`Answer this in British spelling: $ARGUMENTS`, then:
- **Chat:** run `ai chat`, type `/British what is colour optimization`, confirm
  it expands and the answer respects the instruction; confirm `/help` lists it
  and `/bogus` is rejected.
- **CLI:** run `ai /British "what is colour optimization"` from the terminal and
  confirm the same expansion/behavior one-shot; confirm `ai /bogus` errors.

## Out of scope

- Persistent toggle/mode commands (you chose one-shot expansion; AGENTS.md
  already covers always-on instructions).
- Positional args (`$1`, `$2`) — only `$ARGUMENTS` (YAGNI; easy to add later).
- Semantic/keyword auto-selection of commands (rejected alternative; reintroduces
  guessing we're avoiding on the 14B).
- Namespacing/subdirectories of commands (YAGNI until the folder gets crowded).
