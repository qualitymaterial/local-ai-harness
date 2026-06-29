"""Tests for the embeddings / semantic-retrieval core.

A deterministic fake embedder maps text → a small bag-of-words vector so search
behavior is testable without hitting the LM Studio embeddings endpoint.
"""

from __future__ import annotations

import math

from local_ai.embeddings import (
    Chunk,
    EmbeddingIndex,
    chunk_file,
    cosine_similarity,
)

_VOCAB = ["add", "subtract", "multiply", "divide", "database", "render"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Vector = count of each vocab word in the (lowercased) text."""
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count(word)) for word in _VOCAB])
    return out


# --------------------------------------------------------------- cosine_similarity


def test_cosine_identical_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_handles_zero_vector():
    # No division-by-zero blowup; just returns 0 similarity.
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------- chunk_file


def test_chunk_file_splits_long_files():
    text = "\n".join(f"line {i}" for i in range(1, 151))  # 150 lines
    chunks = chunk_file("foo.py", text, max_lines=60, overlap=0)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 150


def test_chunk_file_small_file_single_chunk():
    chunks = chunk_file("tiny.py", "a\nb\nc", max_lines=60)
    assert len(chunks) == 1
    assert chunks[0].rel_path == "tiny.py"


# --------------------------------------------------------------- EmbeddingIndex


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk("math.py", 1, 2, "def add(a, b): return a + b"),
        Chunk("math.py", 4, 5, "def multiply(a, b): return a * b"),
        Chunk("db.py", 1, 3, "connect to the database and query"),
        Chunk("ui.py", 1, 2, "render the page"),
    ]


def test_search_returns_most_relevant_first(tmp_path):
    idx = EmbeddingIndex(tmp_path / "emb.db")
    idx.build(_sample_chunks(), fake_embed)

    hits = idx.search("how do I add two numbers", fake_embed, top_k=2)
    assert hits[0].rel_path == "math.py"
    assert "add" in hits[0].text
    assert len(hits) == 2


def test_search_distinguishes_topics(tmp_path):
    idx = EmbeddingIndex(tmp_path / "emb.db")
    idx.build(_sample_chunks(), fake_embed)

    hits = idx.search("query the database connection", fake_embed, top_k=1)
    assert hits[0].rel_path == "db.py"


def test_index_persists_to_disk(tmp_path):
    db = tmp_path / "emb.db"
    EmbeddingIndex(db).build(_sample_chunks(), fake_embed)

    # A fresh instance reads the same db without rebuilding.
    reopened = EmbeddingIndex(db)
    hits = reopened.search("multiply numbers", fake_embed, top_k=1)
    assert hits[0].rel_path == "math.py"
    assert "multiply" in hits[0].text


def test_search_top_k_limits_results(tmp_path):
    idx = EmbeddingIndex(tmp_path / "emb.db")
    idx.build(_sample_chunks(), fake_embed)
    assert len(idx.search("add", fake_embed, top_k=3)) == 3
