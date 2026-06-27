"""Helpers for respecting .gitignore and ignoring common junk directories/files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pathspec

# Directories we always ignore regardless of .gitignore.
ALWAYS_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "coverage",
        ".coverage",
        "target",
        "vendor",
        ".idea",
        ".vscode",
        ".local-ai",
        ".turbo",
        ".cache",
        "out",
    }
)

# Files we ignore by name. Lockfiles are heavy and rarely useful as model context;
# they can be force-included by the caller if ever needed.
ALWAYS_IGNORE_FILES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "poetry.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
    }
)

# Extensions treated as binary/non-text and skipped for content.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff", ".svg",
        ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".7z", ".rar",
        ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac", ".webm", ".mkv",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".so", ".dylib", ".dll", ".a", ".o", ".class", ".jar", ".pyc",
        ".wasm", ".bin", ".exe", ".lock", ".db", ".sqlite", ".sqlite3",
        ".parquet", ".npy", ".onnx", ".pt", ".pth", ".gguf", ".safetensors",
    }
)


def load_gitignore_spec(repo_root: Path) -> Optional[pathspec.PathSpec]:
    """Load a combined PathSpec from .gitignore (and .git/info/exclude if present)."""
    patterns: list[str] = []
    for candidate in (repo_root / ".gitignore", repo_root / ".git" / "info" / "exclude"):
        if candidate.exists():
            try:
                patterns.extend(candidate.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue
    if not patterns:
        return None
    # 'gitignore' is the newer factory name; fall back for older pathspec versions.
    try:
        return pathspec.PathSpec.from_lines("gitignore", patterns)
    except (KeyError, ValueError):
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def is_ignored_dir(name: str) -> bool:
    return name in ALWAYS_IGNORE_DIRS


def is_ignored_file(name: str) -> bool:
    return name in ALWAYS_IGNORE_FILES


def is_binary_path(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def is_gitignored(spec: Optional[pathspec.PathSpec], rel_path: str) -> bool:
    if spec is None:
        return False
    return spec.match_file(rel_path)
