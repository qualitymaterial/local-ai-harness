"""Abstract base for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class AgentResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "end_turn"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class BaseBackend(ABC):
    """Common interface for model backends (LM Studio, Claude, etc.)."""

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> AgentResponse:
        """Return a complete response (non-streaming), including any tool calls."""

    @abstractmethod
    def append_tool_messages(
        self,
        messages: list[dict],
        response: AgentResponse,
        results: list[tuple[str, str, str]],
    ) -> None:
        """
        Append the assistant's tool-call message and tool results to `messages` in-place.

        `results` is a list of (tool_call_id, tool_name, result_text) tuples.
        Each backend formats these in its own wire format (Anthropic vs OpenAI).
        """

    def health_check(self) -> bool:
        return True
