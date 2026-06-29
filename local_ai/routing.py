"""Auto-routing: decide when a chat turn should escalate to the Claude backend.

This is intentionally a small, deterministic, explainable heuristic — not a model
call. The local model handles routine work; turns that look like heavy reasoning
(architecture, root-cause debugging, large refactors, security) are candidates for
Claude. The decision is advisory: the chat loop confirms with the user before
spending on the cloud backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Intent phrases that signal a task likely beyond a ~30B local model.
# Kept as word-ish patterns so "refactor" matches but "refactoring" still works,
# while avoiding accidental substring hits.
_ESCALATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brefactor", "refactor"),
    (r"\bre-?architect|\barchitecture\b|\bredesign", "architecture/redesign"),
    (r"\broot[\s-]?cause", "root-cause analysis"),
    (r"\brace condition|\bdeadlock|\bconcurren|\bthread[\s-]?saf", "concurrency"),
    (r"\bmemory leak", "memory leak"),
    (r"\bdebug\b|\bwhy is\b.*\bfail|\bintermittent", "hard debugging"),
    (r"\brewrite\b|\bmigrate\b|\bport\b.*\bto\b", "rewrite/migration"),
    (r"\bsecurity (audit|review|vuln)", "security review"),
    (r"\boptimi[sz]e (the )?performance|\bperformance bottleneck", "performance"),
    (r"\bdesign (a|the|this)\b", "design"),
)

# Above this length, a request is probably a big, multi-part ask worth Claude.
_LONG_REQUEST_CHARS = 600


@dataclass
class RouteDecision:
    """Whether to escalate a turn to Claude, with a human-readable reason."""

    escalate: bool
    reason: str


def route_request(text: str, *, current_backend: str) -> RouteDecision:
    """Decide whether `text` should be handled by Claude instead of the local model.

    Never escalates when already on the Claude backend. Otherwise escalates when the
    request matches a heavy-reasoning intent pattern or is unusually long.
    """
    if current_backend == "claude":
        return RouteDecision(False, "already on claude")

    lowered = text.lower()
    for pattern, label in _ESCALATION_PATTERNS:
        if re.search(pattern, lowered):
            return RouteDecision(True, f"looks like {label}")

    if len(text) >= _LONG_REQUEST_CHARS:
        return RouteDecision(True, "large multi-part request")

    return RouteDecision(False, "routine — staying local")
