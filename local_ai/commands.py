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
