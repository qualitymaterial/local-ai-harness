"""Tool definitions and executor for the agentic loop."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_file",
        "description": (
            "Read the full contents of a file in the repository. "
            "Use when you need to inspect source code, configs, or data files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repository root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search for a text pattern across the repository using grep. "
            "Returns matching file paths, line numbers, and the matched lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for.",
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Subdirectory to limit the search (optional). "
                        "Defaults to the entire repository."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories at a path within the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repo root. Defaults to root.",
                }
            },
            "required": [],
        },
    },
]

_MAX_FILE_BYTES = 60_000  # 60 KB per single read
_MAX_SEARCH_LINES = 60


def execute_tool(name: str, tool_input: dict, repo_root: Path) -> str:
    """Dispatch a tool call and return its output as a string."""
    try:
        if name == "read_file":
            return _read_file(tool_input.get("path", ""), repo_root)
        if name == "search_code":
            return _search_code(
                tool_input.get("pattern", ""),
                tool_input.get("directory"),
                repo_root,
            )
        if name == "list_directory":
            return _list_directory(tool_input.get("path", ""), repo_root)
        return f"Unknown tool: {name!r}"
    except Exception as exc:
        return f"Tool error ({name}): {exc}"


# ----------------------------------------------------------------- tool impls

def _read_file(rel_path: str, repo_root: Path) -> str:
    if not rel_path:
        return "Error: no path provided."
    resolved_root = repo_root.resolve()
    path = (repo_root / rel_path).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError:
        return f"Error: '{rel_path}' is outside the repository."
    if not path.exists():
        return f"File not found: {rel_path}"
    if not path.is_file():
        return f"Not a file: {rel_path}"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"Could not read file: {exc}"
    if len(raw) > _MAX_FILE_BYTES:
        text = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        return (
            f"[First {_MAX_FILE_BYTES} bytes of {rel_path} — "
            f"file is {len(raw)} bytes total]\n\n{text}\n\n[...truncated]"
        )
    return raw.decode("utf-8", errors="replace")


def _search_code(pattern: str, directory: Optional[str], repo_root: Path) -> str:
    if not pattern:
        return "Error: no pattern provided."
    resolved_root = repo_root.resolve()
    search_root = resolved_root
    if directory:
        candidate = (repo_root / directory).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            return f"Error: '{directory}' is outside the repository."
        if not candidate.is_dir():
            return f"Directory not found: {directory}"
        search_root = candidate

    _EXCLUDE_DIRS = [
        "node_modules", ".git", ".venv", "venv", "__pycache__",
        "dist", "build", ".next", "target", ".local-ai",
    ]
    cmd = ["grep", "-rn", "--binary-files=without-match"]
    for d in _EXCLUDE_DIRS:
        cmd.extend(["--exclude-dir", d])
    cmd.extend([pattern, str(search_root)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return "Error: grep is not available on this system."
    except subprocess.TimeoutExpired:
        return f"Search timed out for pattern '{pattern}'."

    if proc.returncode not in (0, 1):
        return f"grep error: {proc.stderr.strip() or 'non-zero exit'}"

    output = proc.stdout.strip()
    if not output:
        return f"No matches found for '{pattern}'."

    lines = output.splitlines()
    truncated = len(lines) > _MAX_SEARCH_LINES
    if truncated:
        lines = lines[:_MAX_SEARCH_LINES]

    # Make paths relative to repo root for readability
    rel_lines = []
    for line in lines:
        # grep output: /abs/path/to/file:42:matched content
        parts = line.split(":", 2)
        if len(parts) >= 1:
            try:
                rel = Path(parts[0]).relative_to(resolved_root)
                parts[0] = str(rel)
                line = ":".join(parts)
            except ValueError:
                pass
        rel_lines.append(line)

    result = "\n".join(rel_lines)
    if truncated:
        result += f"\n[...truncated at {_MAX_SEARCH_LINES} lines]"
    return result


def _list_directory(rel_path: str, repo_root: Path) -> str:
    resolved_root = repo_root.resolve()
    target = resolved_root
    if rel_path:
        candidate = (repo_root / rel_path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            return f"Error: '{rel_path}' is outside the repository."
        target = candidate

    if not target.exists():
        return f"Path not found: {rel_path or '.'}"
    if not target.is_dir():
        return f"Not a directory: {rel_path or '.'}"

    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        return f"Could not list directory: {exc}"

    lines = []
    for entry in entries[:120]:
        indicator = "/" if entry.is_dir() else ""
        try:
            rel = str(entry.relative_to(resolved_root))
        except ValueError:
            rel = entry.name
        lines.append(f"{rel}{indicator}")

    if len(entries) > 120:
        lines.append(f"[...{len(entries) - 120} more entries not shown]")

    return "\n".join(lines) if lines else "(empty directory)"
