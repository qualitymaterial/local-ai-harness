"""Tests for Claude spend tracking (cost meter)."""

from __future__ import annotations

import math

from local_ai.pricing import SessionCost, compute_cost, model_pricing


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# ------------------------------------------------------------------ model_pricing


def test_known_model_pricing():
    # Opus 4.8: $5 / $25 per 1M tokens (input / output)
    assert model_pricing("claude-opus-4-8") == (5.0, 25.0)


def test_pricing_matches_by_prefix():
    # Date-suffixed or aliased IDs still resolve.
    assert model_pricing("claude-haiku-4-5-20251001") == (1.0, 5.0)


def test_local_model_is_free():
    # Local models (Ollama/LM Studio ids) aren't in the table → no price.
    assert model_pricing("qwen3-coder-30b") is None
    assert model_pricing("text-embedding-nomic-embed-text-v1.5") is None


# ------------------------------------------------------------------ compute_cost


def test_compute_cost_opus():
    # 1M input + 1M output on Opus 4.8 = $5 + $25 = $30
    assert _close(compute_cost("claude-opus-4-8", 1_000_000, 1_000_000), 30.0)


def test_compute_cost_partial():
    # 200k input + 50k output on Opus 4.8 = 0.2*5 + 0.05*25 = 1.0 + 1.25 = 2.25
    assert _close(compute_cost("claude-opus-4-8", 200_000, 50_000), 2.25)


def test_compute_cost_local_is_zero():
    assert compute_cost("qwen3-coder-30b", 500_000, 500_000) == 0.0


# ------------------------------------------------------------------ SessionCost


def test_session_cost_accumulates():
    sc = SessionCost()
    sc.add("claude-opus-4-8", 200_000, 50_000)   # $2.25
    sc.add("claude-opus-4-8", 200_000, 50_000)   # $2.25
    assert _close(sc.total_usd, 4.50)


def test_session_cost_ignores_free_models():
    sc = SessionCost()
    sc.add("qwen3-coder-30b", 1_000_000, 1_000_000)
    assert sc.total_usd == 0.0


def test_session_cost_tracks_token_totals():
    sc = SessionCost()
    sc.add("claude-opus-4-8", 100, 200)
    sc.add("claude-opus-4-8", 300, 400)
    assert sc.prompt_tokens == 400
    assert sc.completion_tokens == 600
