"""Repository scanner: walks a repo, respects ignores, extracts lightweight metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from . import gitignore_utils as gi
from .types import FileEntry, RepoIndex

# Map of file extension -> human language label.
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript (React)",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript (React)",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".php": "PHP",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cpp": "C++",
    ".cc": "C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".m": "Objective-C",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".sql": "SQL",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    ".html": "HTML",
    ".md": "Markdown",
    ".mdx": "MDX",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
}

# Glob-ish patterns (matched against rel path or basename) that mark "important" files.
IMPORTANT_BASENAMES: frozenset[str] = frozenset(
    {
        "readme", "readme.md", "package.json", "pyproject.toml", "requirements.txt",
        "setup.py", "setup.cfg", "tsconfig.json", "go.mod", "cargo.toml", "gemfile",
        "dockerfile", "docker-compose.yml", "docker-compose.yaml", "makefile",
        ".env.example", "manifest.json",
    }
)

IMPORTANT_PREFIXES: tuple[str, ...] = (
    "vite.config",
    "next.config",
    "tailwind.config",
    "svelte.config",
    "nuxt.config",
    "astro.config",
    "webpack.config",
    "rollup.config",
    "babel.config",
    "jest.config",
    "vitest.config",
    "eslint.config",
)

# Directory names whose presence signals project structure importance.
IMPORTANT_DIRS: frozenset[str] = frozenset(
    {"src", "app", "pages", "components", "routes", "api", "lib", "tests", "test", "server"}
)

MAX_BYTES_FOR_STRUCTURE_SCAN = 400_000  # don't parse huge files for symbols

# Cheap regexes for structural signals. Best-effort, language-agnostic-ish.
_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+.+?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),  # JS/TS
    re.compile(r"^\s*import\s+['\"]([^'\"]+)['\"]", re.MULTILINE),  # JS side-effect
    re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", re.MULTILINE),  # CJS
    re.compile(r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE),  # Python from
    re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),  # Python import
]

_SYMBOL_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", re.MULTILINE),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE),
    re.compile(r"^\s*export\s+const\s+([A-Za-z_$][\w$]*)", re.MULTILINE),
    re.compile(r"^\s*def\s+([A-Za-z_][\w]*)", re.MULTILINE),  # Python
    re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)", re.MULTILINE),  # Rust
    re.compile(r"^\s*func\s+([A-Za-z_][\w]*)", re.MULTILINE),  # Go
]


def detect_language(path: Path) -> Optional[str]:
    return LANGUAGE_BY_EXT.get(path.suffix.lower())


def _is_important(rel_path: str) -> bool:
    parts = rel_path.split("/")
    basename = parts[-1].lower()
    if basename in IMPORTANT_BASENAMES:
        return True
    if basename.startswith("readme"):
        return True
    for prefix in IMPORTANT_PREFIXES:
        if basename.startswith(prefix):
            return True
    # A top-level config-ish or a file directly under an important dir.
    if len(parts) >= 2 and parts[0] in IMPORTANT_DIRS:
        return False  # being inside src/ alone isn't "important", but the dir is tracked
    return False


def _read_text_safely(path: Path, max_bytes: int) -> Optional[str]:
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return None
    # Heuristic binary sniff: NUL byte in the head.
    if b"\x00" in data[:1024]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None


def _extract_structure(text: str) -> tuple[list[str], list[str]]:
    imports: list[str] = []
    for pat in _IMPORT_PATTERNS:
        imports.extend(pat.findall(text))
    symbols: list[str] = []
    for pat in _SYMBOL_PATTERNS:
        symbols.extend(pat.findall(text))
    # Dedup while preserving order, cap to keep metadata small.
    return _dedup(imports)[:40], _dedup(symbols)[:60]


def _dedup(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _detect_frameworks(files: list[FileEntry], root: Path) -> list[str]:
    frameworks: set[str] = set()
    names = {f.rel_path for f in files}

    pkg = root / "package.json"
    if pkg.exists():
        text = _read_text_safely(pkg, 200_000) or ""
        lowered = text.lower()
        checks = {
            "next": "Next.js",
            "react": "React",
            "vue": "Vue",
            "svelte": "Svelte",
            "@angular/core": "Angular",
            "astro": "Astro",
            "express": "Express",
            "fastify": "Fastify",
            "nestjs": "NestJS",
            "vite": "Vite",
            "tailwindcss": "Tailwind CSS",
        }
        for needle, label in checks.items():
            if f'"{needle}"' in lowered or f"'{needle}'" in lowered:
                frameworks.add(label)

    if any(n in names for n in ("pyproject.toml", "requirements.txt", "setup.py")):
        frameworks.add("Python")
    if "go.mod" in names:
        frameworks.add("Go")
    if "Cargo.toml" in names:
        frameworks.add("Rust")
    if any(n.startswith("next.config") for n in names):
        frameworks.add("Next.js")
    if any(n.startswith("vite.config") for n in names):
        frameworks.add("Vite")
    if any(n.startswith("tailwind.config") for n in names):
        frameworks.add("Tailwind CSS")
    # Python web frameworks by content of requirements/pyproject.
    for cfg in ("requirements.txt", "pyproject.toml"):
        p = root / cfg
        if p.exists():
            txt = (_read_text_safely(p, 100_000) or "").lower()
            for needle, label in {
                "fastapi": "FastAPI",
                "flask": "Flask",
                "django": "Django",
                "typer": "Typer",
                "click": "Click",
            }.items():
                if needle in txt:
                    frameworks.add(label)

    return sorted(frameworks)


def scan_repo(
    root: Path,
    *,
    respect_gitignore: bool = True,
    extract_structure: bool = True,
    max_files: int = 5000,
) -> RepoIndex:
    """Walk ``root`` and build a :class:`RepoIndex`."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    spec = gi.load_gitignore_spec(root) if respect_gitignore else None
    files: list[FileEntry] = []
    languages: dict[str, int] = {}
    total_size = 0

    for path in _walk(root, spec):
        if len(files) >= max_files:
            break
        rel = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue

        binary = gi.is_binary_path(path)
        language = detect_language(path)
        entry = FileEntry(
            path=path,
            rel_path=rel,
            size_bytes=size,
            language=language,
            is_binary=binary,
            is_important=_is_important(rel),
        )

        if not binary and extract_structure and size <= MAX_BYTES_FOR_STRUCTURE_SCAN:
            text = _read_text_safely(path, MAX_BYTES_FOR_STRUCTURE_SCAN)
            if text is not None:
                entry._lines = text.count("\n") + 1
                imports, symbols = _extract_structure(text)
                entry.imports = imports
                entry.symbols = symbols
            else:
                entry.is_binary = True

        files.append(entry)
        total_size += size
        if language:
            languages[language] = languages.get(language, 0) + 1

    files.sort(key=lambda f: f.rel_path)
    frameworks = _detect_frameworks(files, root)
    important = sorted(f.rel_path for f in files if f.is_important)

    return RepoIndex(
        root=root,
        files=files,
        languages=dict(sorted(languages.items(), key=lambda kv: (-kv[1], kv[0]))),
        frameworks=frameworks,
        important_files=important,
        total_size_bytes=total_size,
    )


def _walk(root: Path, spec) -> Iterable[Path]:
    """Yield files under root, pruning ignored directories early."""
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except (OSError, PermissionError):
            continue
        for child in entries:
            name = child.name
            rel = child.relative_to(root).as_posix()
            if child.is_symlink():
                continue
            if child.is_dir():
                if gi.is_ignored_dir(name):
                    continue
                if gi.is_gitignored(spec, rel + "/") or gi.is_gitignored(spec, rel):
                    continue
                stack.append(child)
            elif child.is_file():
                if gi.is_ignored_file(name):
                    continue
                if gi.is_gitignored(spec, rel):
                    continue
                yield child


def build_file_tree(index: RepoIndex, max_entries: int = 400) -> str:
    """Render an indented ASCII file tree (truncated for very large repos)."""
    lines: list[str] = [index.root.name + "/"]
    rels = [f.rel_path for f in index.files]
    shown = rels[:max_entries]
    # Build a nested dict for tree rendering.
    tree: dict = {}
    for rel in shown:
        node = tree
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part + "/", {})
        node[parts[-1]] = None

    def render(node: dict, prefix: str) -> None:
        items = sorted(node.items(), key=lambda kv: (not kv[0].endswith("/"), kv[0]))
        for i, (key, child) in enumerate(items):
            last = i == len(items) - 1
            connector = "└── " if last else "├── "
            lines.append(prefix + connector + key)
            if isinstance(child, dict):
                extension = "    " if last else "│   "
                render(child, prefix + extension)

    render(tree, "")
    if len(rels) > max_entries:
        lines.append(f"... ({len(rels) - max_entries} more files not shown)")
    return "\n".join(lines)
