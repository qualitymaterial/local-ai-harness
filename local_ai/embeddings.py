"""Semantic codebase retrieval (RAG) for local-ai.

Chunks source files, embeds them via an OpenAI-compatible /v1/embeddings endpoint
(LM Studio's nomic-embed-text by default), and stores the vectors in a small SQLite
index at .local-ai/embeddings.db. At query time it embeds the query and returns the
most cosine-similar chunks — sharper, smaller context than dumping a flat repo map.

The embedding call is injected as ``embed_fn`` so the pure logic (chunking, cosine,
ranking, persistence) is testable without a running server.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx

from .config import Config
from .model_client import ModelError, ServerUnreachableError

EmbedFn = Callable[[list[str]], list[list[float]]]

# Files larger than this are skipped during indexing (likely data/minified).
_MAX_FILE_BYTES = 200_000
_DEFAULT_MAX_LINES = 60
_DEFAULT_OVERLAP = 10


@dataclass
class Chunk:
    rel_path: str
    start_line: int
    end_line: int
    text: str


@dataclass
class SearchHit:
    rel_path: str
    start_line: int
    end_line: int
    text: str
    score: float


# --------------------------------------------------------------------------- math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector has zero magnitude."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- chunking


def chunk_file(
    rel_path: str,
    text: str,
    *,
    max_lines: int = _DEFAULT_MAX_LINES,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping line-window chunks.

    Each chunk covers up to ``max_lines`` lines; consecutive chunks overlap by
    ``overlap`` lines so context that straddles a boundary isn't lost.
    """
    lines = text.splitlines()
    if not lines:
        return []
    if overlap >= max_lines:
        overlap = 0
    step = max_lines - overlap

    chunks: list[Chunk] = []
    start = 0
    n = len(lines)
    while start < n:
        end = min(start + max_lines, n)
        body = "\n".join(lines[start:end])
        chunks.append(Chunk(rel_path, start + 1, end, body))
        if end == n:
            break
        start += step
    return chunks


def chunk_repo(root: Path, rel_paths: Iterable[str]) -> list[Chunk]:
    """Chunk a set of repo files (by relative path) into embeddable pieces."""
    out: list[Chunk] = []
    for rel in rel_paths:
        p = root / rel
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if len(data) > _MAX_FILE_BYTES or b"\x00" in data[:1024]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out.extend(chunk_file(rel, text))
    return out


# --------------------------------------------------------------------------- index


class EmbeddingIndex:
    """SQLite-backed store of chunk embeddings with cosine search."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                rel_path   TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line   INTEGER NOT NULL,
                text       TEXT NOT NULL,
                vector     TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def build(self, chunks: list[Chunk], embed_fn: EmbedFn, *, batch_size: int = 32) -> int:
        """(Re)build the index from chunks. Returns the number of chunks indexed."""
        self._conn.execute("DELETE FROM chunks")
        count = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = embed_fn([c.text for c in batch])
            rows = [
                (c.rel_path, c.start_line, c.end_line, c.text, json.dumps(v))
                for c, v in zip(batch, vectors)
            ]
            self._conn.executemany(
                "INSERT INTO chunks (rel_path, start_line, end_line, text, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            count += len(rows)
        self._conn.commit()
        return count

    def search(self, query: str, embed_fn: EmbedFn, *, top_k: int = 8) -> list[SearchHit]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        query_vec = embed_fn([query])[0]
        rows = self._conn.execute(
            "SELECT rel_path, start_line, end_line, text, vector FROM chunks"
        ).fetchall()
        scored: list[SearchHit] = []
        for rel_path, start_line, end_line, text, vector_json in rows:
            score = cosine_similarity(query_vec, json.loads(vector_json))
            scored.append(SearchHit(rel_path, start_line, end_line, text, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------- live embedder


class EmbeddingClient:
    """Calls an OpenAI-compatible /v1/embeddings endpoint (LM Studio by default)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = getattr(config, "embedding_model", "") or "text-embedding-nomic-embed-text-v1.5"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.config.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                resp = client.post(
                    url, headers=headers, json={"model": self.model, "input": texts}
                )
        except httpx.ConnectError as exc:
            raise ServerUnreachableError(
                f"Could not connect to {self.config.base_url} for embeddings.",
                hint="Is LM Studio running with an embedding model available?",
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"HTTP error during embeddings: {exc}") from exc

        if resp.status_code != 200:
            raise ModelError(
                f"Embeddings endpoint returned HTTP {resp.status_code}: {resp.text[:300]}",
                hint="Load an embedding model (e.g. nomic-embed-text) in LM Studio.",
            )
        data = resp.json().get("data") or []
        return [item["embedding"] for item in data]
