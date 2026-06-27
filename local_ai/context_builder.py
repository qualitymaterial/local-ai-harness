"""Assemble a token-budgeted context packet from a repo index + ranked files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import scanner
from .file_ranker import rank_files
from .types import ContextPacket, FileEntry, RepoIndex

# Rough heuristic: ~4 characters per token for code/English. Good enough for budgeting
# without pulling in a tokenizer dependency.
CHARS_PER_TOKEN = 4

# When a file is too big to include whole, include this many chars from the head.
LARGE_FILE_HEAD_CHARS = 2400


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _read_text(entry: FileEntry, limit: int = 200_000) -> Optional[str]:
    try:
        data = entry.path.read_bytes()[:limit]
    except OSError:
        return None
    if b"\x00" in data[:1024]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


def _fence_lang(entry: FileEntry) -> str:
    ext = entry.path.suffix.lower().lstrip(".")
    return ext or ""


def _summarize_large_file(entry: FileEntry, text: str) -> str:
    """For oversized files, include head + structural outline instead of full body."""
    head = text[:LARGE_FILE_HEAD_CHARS]
    parts = [head.rstrip()]
    outline = []
    if entry.imports:
        outline.append("imports: " + ", ".join(entry.imports[:25]))
    if entry.symbols:
        outline.append("symbols: " + ", ".join(entry.symbols[:40]))
    if outline:
        parts.append("\n# --- structural outline (file truncated for context budget) ---\n")
        parts.append("\n".join(outline))
    return "\n".join(parts)


def _render_file_block(entry: FileEntry, content: str, *, truncated: bool) -> str:
    lang = _fence_lang(entry)
    note = "  (truncated)" if truncated else ""
    lines_info = f", {entry.lines} lines" if entry.lines else ""
    header = f"### `{entry.rel_path}`  ({entry.size_bytes} bytes{lines_info}){note}"
    return f"{header}\n\n```{lang}\n{content}\n```\n"


def build_repo_map(index: RepoIndex, *, max_tree_entries: int = 400) -> str:
    """Produce the markdown repo map (also persisted to .local-ai/repo_map.md)."""
    lines: list[str] = []
    lines.append(f"# Repo Map: {index.root.name}")
    lines.append("")
    lines.append(f"- Root: `{index.root}`")
    lines.append(f"- Files scanned: {index.file_count}")
    lines.append(f"- Total size: {index.total_size_bytes / 1024:.1f} KB")
    if index.frameworks:
        lines.append(f"- Detected stack: {', '.join(index.frameworks)}")
    if index.languages:
        lang_str = ", ".join(f"{k} ({v})" for k, v in index.languages.items())
        lines.append(f"- Languages: {lang_str}")
    lines.append("")

    if index.important_files:
        lines.append("## Important files")
        for rel in index.important_files:
            lines.append(f"- `{rel}`")
        lines.append("")

    lines.append("## File tree")
    lines.append("")
    lines.append("```")
    lines.append(scanner.build_file_tree(index, max_entries=max_tree_entries))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_context(
    index: RepoIndex,
    query: str,
    *,
    max_context_tokens: int,
    repo_map: Optional[str] = None,
    force_include: Optional[list[str]] = None,
    reserve_for_response: int = 1500,
) -> ContextPacket:
    """Build a context packet for a query within ``max_context_tokens``.

    Always includes the repo map and important config files first, then fills the
    remaining budget with the highest-ranked relevant source files.
    """
    if repo_map is None:
        repo_map = build_repo_map(index)

    budget = max(1000, max_context_tokens - reserve_for_response)
    used = estimate_tokens(repo_map)

    included: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []

    ranked = rank_files(index, query)
    rank_by_rel = {r.entry.rel_path: r for r in ranked}

    # Determine inclusion order: forced files, then important config (small), then ranked.
    ordered: list[FileEntry] = []
    seen: set[str] = set()

    def _add(entry: FileEntry) -> None:
        if entry.rel_path not in seen and not entry.is_binary:
            seen.add(entry.rel_path)
            ordered.append(entry)

    by_rel = {f.rel_path: f for f in index.files}
    for rel in force_include or []:
        if rel in by_rel:
            _add(by_rel[rel])

    # Small important files (configs, readme) give the model grounding cheaply.
    for f in index.files:
        if f.is_important and f.size_bytes <= 12_000:
            _add(f)

    for r in ranked:
        if r.score <= 0:
            continue
        _add(r.entry)

    for entry in ordered:
        content = _read_text(entry)
        if content is None:
            continue
        truncated = False
        if len(content) > LARGE_FILE_HEAD_CHARS * 3:
            content = _summarize_large_file(entry, content)
            truncated = True
        block = _render_file_block(entry, content, truncated=truncated)
        cost = estimate_tokens(block)
        if used + cost > budget:
            # Try a truncated version if not already truncated and it might fit.
            if not truncated:
                short = _summarize_large_file(entry, content)
                block = _render_file_block(entry, short, truncated=True)
                cost = estimate_tokens(block)
            if used + cost > budget:
                skipped.append(entry.rel_path)
                continue
        blocks.append(block)
        included.append(entry.rel_path)
        used += cost

    body_parts = [repo_map, "\n---\n", "## Included file contents\n"]
    if blocks:
        body_parts.append("\n".join(blocks))
    else:
        body_parts.append("_No source files included (none matched or all over budget)._")
    body = "\n".join(body_parts)

    return ContextPacket(
        repo_map=repo_map,
        included_files=included,
        skipped_files=skipped,
        body=body,
        approx_tokens=used,
    )
