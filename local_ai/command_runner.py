"""Run shell commands with capture and destructive-command guarding."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Patterns that indicate a potentially destructive or dangerous command.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[rf]", re.IGNORECASE), "recursive/forced rm"),
    (re.compile(r"\brm\s+-rf?\b", re.IGNORECASE), "rm -rf"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bchmod\s+-R\b"), "recursive chmod"),
    (re.compile(r"\bchown\s+-R\b"), "recursive chown"),
    (re.compile(r"\bmkfs\b"), "filesystem format"),
    (re.compile(r"\bdd\s+if="), "raw dd"),
    (re.compile(r":\(\)\s*\{.*\}\s*;"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b.*\|\s*(sudo\s+)?(bash|sh|zsh)\b"), "curl|wget pipe to shell"),
    (re.compile(r">\s*/dev/sd[a-z]"), "write to raw disk"),
    (re.compile(r"\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)"), "destructive git"),
    (re.compile(r"\bshutdown\b|\breboot\b"), "shutdown/reboot"),
    (re.compile(r">\s*/dev/null\s*2>&1\s*;\s*rm"), "chained rm"),
]


@dataclass
class CommandResult:
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def check_dangerous(command: str) -> list[str]:
    """Return a list of human-readable reasons the command looks dangerous (empty if safe)."""
    reasons: list[str] = []
    for pattern, label in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            reasons.append(label)
    return reasons


def run_command(command: str, cwd: Path, *, timeout: int = 1800) -> CommandResult:
    """Run ``command`` via the shell from ``cwd``, capturing stdout/stderr/exit code."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            command=command,
            cwd=str(cwd),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            cwd=str(cwd),
            exit_code=124,
            stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=(exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
            + f"\n[command timed out after {timeout}s]",
            timed_out=True,
        )


def tail(text: str, max_chars: int = 6000) -> str:
    """Return the last ``max_chars`` of text (errors usually live at the end)."""
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]
