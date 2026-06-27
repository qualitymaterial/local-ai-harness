"""Shared dataclasses and type definitions for local-ai."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FileEntry:
    """A single scanned file with metadata used for ranking and context building."""

    path: Path  # absolute path
    rel_path: str  # path relative to the repo root, POSIX-style
    size_bytes: int
    language: Optional[str] = None
    is_binary: bool = False
    is_important: bool = False
    # Lightweight structural signals extracted during scan (best-effort, cheap).
    imports: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)  # function / class / export names

    @property
    def lines(self) -> int:
        return self._lines

    _lines: int = 0


@dataclass
class RepoIndex:
    """The result of scanning a repository."""

    root: Path
    files: list[FileEntry]
    languages: dict[str, int]  # language -> file count
    frameworks: list[str]
    important_files: list[str]  # rel paths
    total_size_bytes: int

    @property
    def file_count(self) -> int:
        return len(self.files)


@dataclass
class RankedFile:
    """A file with a relevance score for a given question/task."""

    entry: FileEntry
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ContextPacket:
    """A bundle of content prepared to send to the model, within a token budget."""

    repo_map: str
    included_files: list[str]  # rel paths actually included
    skipped_files: list[str]  # rel paths considered but dropped for budget
    body: str  # the assembled context text
    approx_tokens: int


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ModelResponse:
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
