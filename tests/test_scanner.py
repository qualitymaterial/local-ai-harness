"""Tests for the repository scanner."""

from __future__ import annotations

from pathlib import Path

from local_ai import scanner


def _write(root: Path, rel: str, content: str = "") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_scan_basic_and_ignores(tmp_path: Path) -> None:
    _write(tmp_path, "package.json", '{"dependencies": {"react": "^18.0.0", "next": "14"}}')
    _write(tmp_path, "src/App.tsx", "export function App() { return null }\nimport x from 'y'\n")
    _write(tmp_path, "src/components/Header.tsx", "export const Header = () => null\n")
    # Junk that must be ignored.
    _write(tmp_path, "node_modules/foo/index.js", "module.exports = 1")
    _write(tmp_path, "dist/bundle.js", "console.log(1)")
    _write(tmp_path, ".DS_Store", "junk")
    _write(tmp_path, "package-lock.json", "{}")

    index = scanner.scan_repo(tmp_path)
    rels = {f.rel_path for f in index.files}

    assert "package.json" in rels
    assert "src/App.tsx" in rels
    assert "src/components/Header.tsx" in rels
    # Ignored entries absent.
    assert not any(r.startswith("node_modules/") for r in rels)
    assert not any(r.startswith("dist/") for r in rels)
    assert ".DS_Store" not in rels
    assert "package-lock.json" not in rels


def test_framework_detection(tmp_path: Path) -> None:
    _write(tmp_path, "package.json", '{"dependencies": {"next": "14", "tailwindcss": "3"}}')
    _write(tmp_path, "next.config.js", "module.exports = {}")
    index = scanner.scan_repo(tmp_path)
    assert "Next.js" in index.frameworks
    assert "Tailwind CSS" in index.frameworks


def test_structure_extraction(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/util.ts",
        "import { thing } from './thing'\nexport function doStuff() {}\nexport const VALUE = 1\n",
    )
    index = scanner.scan_repo(tmp_path)
    entry = next(f for f in index.files if f.rel_path == "src/util.ts")
    assert "./thing" in entry.imports
    assert "doStuff" in entry.symbols
    assert "VALUE" in entry.symbols


def test_gitignore_respected(tmp_path: Path) -> None:
    _write(tmp_path, ".gitignore", "secret.txt\nbuildout/\n")
    _write(tmp_path, "secret.txt", "shh")
    _write(tmp_path, "buildout/thing.js", "x")
    _write(tmp_path, "keep.txt", "ok")

    index = scanner.scan_repo(tmp_path)
    rels = {f.rel_path for f in index.files}
    assert "keep.txt" in rels
    assert "secret.txt" not in rels
    assert not any(r.startswith("buildout/") for r in rels)


def test_important_files_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# hi")
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'")
    _write(tmp_path, "vite.config.ts", "export default {}")
    index = scanner.scan_repo(tmp_path)
    important = set(index.important_files)
    assert "README.md" in important
    assert "pyproject.toml" in important
    assert "vite.config.ts" in important


def test_file_tree_renders(tmp_path: Path) -> None:
    _write(tmp_path, "src/a.ts", "x")
    _write(tmp_path, "src/b.ts", "y")
    index = scanner.scan_repo(tmp_path)
    tree = scanner.build_file_tree(index)
    assert "src/" in tree
    assert "a.ts" in tree
