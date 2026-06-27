"""Backend factory for local-ai."""

from __future__ import annotations

from ..config import Config
from .base import BaseBackend


def get_backend(config: Config) -> BaseBackend:
    """Return the appropriate backend based on config.backend."""
    if config.backend == "claude":
        from .claude import ClaudeBackend
        return ClaudeBackend(config)
    # Default: LM Studio (OpenAI-compatible)
    from .lmstudio import LMStudioBackend
    return LMStudioBackend(config)
