"""Rank files by relevance to a question or task using cheap heuristics.

No embeddings or model calls — purely lexical + structural scoring so it stays fast
and fully local. Signals combined:
  * keyword overlap between the query and the filename / path
  * keyword overlap between the query and extracted symbols
  * folder importance (src/app/components/routes/api/...)
  * config / entry-point importance
  * framework-convention bonuses (e.g. "homepage" -> index/page files)
"""

from __future__ import annotations

import re
from pathlib import Path

from .types import FileEntry, RankedFile, RepoIndex

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "to", "of", "in", "on", "for", "and", "or",
        "where", "what", "how", "why", "this", "that", "it", "be", "do", "does",
        "with", "into", "from", "fix", "add", "make", "refactor", "issue", "problem",
        "app", "code", "file", "files", "function", "component", "please", "can",
        "should", "would", "find", "show", "me", "my", "i", "we", "us", "project",
    }
)

# Folder weights: deeper-meaning source dirs rank higher.
_FOLDER_WEIGHTS: dict[str, float] = {
    "src": 1.5,
    "app": 1.8,
    "pages": 1.8,
    "components": 1.6,
    "routes": 1.6,
    "api": 1.5,
    "lib": 1.2,
    "server": 1.4,
    "hooks": 1.2,
    "store": 1.2,
    "styles": 1.0,
    "tests": 0.8,
    "test": 0.8,
    "__tests__": 0.8,
}

# Concept -> filename hints, to map intent to framework conventions.
_CONCEPT_HINTS: dict[str, tuple[str, ...]] = {
    "homepage": ("index", "home", "page", "app", "main", "landing"),
    "home": ("index", "home", "page", "app", "main"),
    "layout": ("layout", "app", "_app", "template", "shell"),
    "route": ("route", "router", "routes", "pages", "app"),
    "router": ("route", "router", "routes"),
    "auth": ("auth", "login", "session", "user", "signin", "signup"),
    "login": ("auth", "login", "session", "signin"),
    "api": ("api", "route", "controller", "handler", "server", "endpoint"),
    "style": ("css", "style", "styles", "tailwind", "theme", "scss"),
    "mobile": ("css", "style", "responsive", "media", "tailwind", "layout"),
    "responsive": ("css", "style", "media", "tailwind", "layout"),
    "config": ("config", "settings", "env"),
    "test": ("test", "spec"),
    "database": ("db", "database", "models", "schema", "prisma", "migration"),
    "model": ("model", "models", "schema", "entity"),
    "build": ("vite", "webpack", "rollup", "config", "package"),
}


def _tokenize(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) > 1 and w.lower() not in _STOPWORDS
    }


def _split_identifier(name: str) -> set[str]:
    """Split a path/identifier into lowercase word tokens (camelCase + delimiters)."""
    # Insert spaces at camelCase boundaries, then split on non-alnum.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return {t.lower() for t in _WORD_RE.findall(spaced) if t}


def _expand_query(query_tokens: set[str]) -> set[str]:
    expanded = set(query_tokens)
    for tok in query_tokens:
        if tok in _CONCEPT_HINTS:
            expanded.update(_CONCEPT_HINTS[tok])
    return expanded


def score_file(entry: FileEntry, query_tokens: set[str], expanded: set[str]) -> RankedFile:
    score = 0.0
    reasons: list[str] = []

    path_tokens = _split_identifier(entry.rel_path)
    direct = query_tokens & path_tokens
    if direct:
        gain = 3.0 * len(direct)
        score += gain
        reasons.append(f"path matches: {', '.join(sorted(direct))}")

    concept = (expanded - query_tokens) & path_tokens
    if concept:
        score += 1.2 * len(concept)
        reasons.append(f"convention match: {', '.join(sorted(concept))}")

    symbol_tokens: set[str] = set()
    for sym in entry.symbols:
        symbol_tokens |= _split_identifier(sym)
    sym_hits = (query_tokens | expanded) & symbol_tokens
    if sym_hits:
        score += 1.5 * len(sym_hits)
        reasons.append(f"symbol matches: {', '.join(sorted(sym_hits))}")

    # Folder importance.
    top = entry.rel_path.split("/")[0] if "/" in entry.rel_path else ""
    parts = entry.rel_path.split("/")
    folder_bonus = 0.0
    for part in parts[:-1]:
        folder_bonus = max(folder_bonus, _FOLDER_WEIGHTS.get(part, 0.0))
    if folder_bonus:
        score += folder_bonus
        reasons.append(f"in significant folder ({top or parts[0]})")

    if entry.is_important:
        score += 2.0
        reasons.append("important config/entry file")

    # Slight penalty for very large files (we'll summarize them anyway).
    if entry.size_bytes > 80_000:
        score -= 0.5

    # Mild bonus for source code over docs/markup when query is technical.
    if entry.language and entry.language not in ("Markdown", "JSON", "YAML", "TOML"):
        score += 0.3

    return RankedFile(entry=entry, score=score, reasons=reasons)


def rank_files(
    index: RepoIndex,
    query: str,
    *,
    top_k: int | None = None,
    include_binary: bool = False,
) -> list[RankedFile]:
    query_tokens = _tokenize(query)
    expanded = _expand_query(query_tokens)

    ranked: list[RankedFile] = []
    for entry in index.files:
        if entry.is_binary and not include_binary:
            continue
        ranked.append(score_file(entry, query_tokens, expanded))

    # Stable sort: score desc, then important first, then shorter path.
    ranked.sort(
        key=lambda r: (-r.score, not r.entry.is_important, len(r.entry.rel_path))
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked
