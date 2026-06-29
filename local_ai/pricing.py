"""Claude API spend tracking for the hybrid local/cloud workflow.

Local models (Ollama / LM Studio) are free, so only the Claude backend accrues
cost. Prices are USD per 1 million tokens (input, output), sourced from the
Anthropic pricing reference. Update here when prices change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# USD per 1,000,000 tokens: model_id -> (input_price, output_price)
_PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def model_pricing(model: str) -> Optional[tuple[float, float]]:
    """Return (input, output) price per 1M tokens for a model, or None if unknown/free.

    Matches an exact id first, then the longest known id that is a prefix of the
    given model (so date-suffixed ids like ``claude-haiku-4-5-20251001`` resolve).
    """
    if model in _PRICING:
        return _PRICING[model]
    best: Optional[str] = None
    for known in _PRICING:
        if model.startswith(known) and (best is None or len(known) > len(best)):
            best = known
    return _PRICING[best] if best else None


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost in USD for a single call. Returns 0.0 for free/local/unknown models."""
    pricing = model_pricing(model)
    if pricing is None:
        return 0.0
    input_price, output_price = pricing
    return (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price


@dataclass
class SessionCost:
    """Running tally of Claude spend across a chat session."""

    total_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record one call's usage and return its cost in USD."""
        cost = compute_cost(model, prompt_tokens or 0, completion_tokens or 0)
        if cost > 0:
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0
            self.total_usd += cost
        return cost
