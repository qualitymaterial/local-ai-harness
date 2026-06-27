"""Anthropic Claude backend using the official SDK."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from ..config import Config, ConfigError
from ..model_client import ModelError, ServerUnreachableError
from .base import BaseBackend, AgentResponse, ToolCall


def _resolve_api_key(config: Config) -> str:
    """Resolve the Anthropic API key from env → ~/.claude/.env → config."""
    # 1. Standard env var — the SDK picks this up automatically, but we verify it exists.
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    # 2. ~/.claude/.env (Brian's global secret store per CLAUDE.md).
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    # 3. Explicit config value (least preferred — avoid putting keys in repo files).
    if config.claude_api_key:
        return config.claude_api_key
    raise ConfigError(
        "No Anthropic API key found. "
        "Set the ANTHROPIC_API_KEY environment variable, "
        "add it to ~/.claude/.env, "
        "or set 'claude_api_key' in .local-ai/config.toml."
    )


class ClaudeBackend(BaseBackend):
    def __init__(self, config: Config) -> None:
        self.config = config
        self._api_key = _resolve_api_key(config)

    def _get_client(self):  # type: ignore[return]
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Claude backend. "
                "Run: pip install anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=self._api_key)

    # ------------------------------------------------------------------ stream

    def stream(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        system, user_messages = _split_system(messages)

        kwargs: dict = {
            "model": self.config.claude_model,
            "max_tokens": 8192,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        else:
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise _translate_exc(exc) from exc

    # ---------------------------------------------------------------- complete

    def complete(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> AgentResponse:
        client = self._get_client()
        system, user_messages = _split_system(messages)

        kwargs: dict = {
            "model": self.config.claude_model,
            "max_tokens": 8192,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)
        else:
            kwargs["thinking"] = {"type": "adaptive"}

        try:
            with client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except Exception as exc:
            raise _translate_exc(exc) from exc

        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in msg.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        usage = msg.usage
        return AgentResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=msg.stop_reason or "end_turn",
            prompt_tokens=usage.input_tokens if usage else None,
            completion_tokens=usage.output_tokens if usage else None,
        )

    # ------------------------------------------------------ tool message helpers

    def append_tool_messages(
        self,
        messages: list[dict],
        response: AgentResponse,
        results: list[tuple[str, str, str]],
    ) -> None:
        """Append in Anthropic tool_use / tool_result format."""
        # Assistant message: content blocks
        assistant_content: list[dict] = []
        if response.content:
            assistant_content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
            )
        messages.append({"role": "assistant", "content": assistant_content})

        # User message: tool_result blocks
        result_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": result_text,
            }
            for call_id, _name, result_text in results
        ]
        messages.append({"role": "user", "content": result_blocks})

    # ---------------------------------------------------------------- health

    def health_check(self) -> bool:
        try:
            self._get_client()
            return bool(self._api_key)
        except Exception:
            return False


# ------------------------------------------------------------------ helpers

def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate the system message from user/assistant messages."""
    system = ""
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            system = content if isinstance(content, str) else ""
        else:
            rest.append(m)
    return system, rest


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


def _translate_exc(exc: Exception) -> ModelError:
    """Map Anthropic SDK exceptions to ModelError so callers don't need to import anthropic."""
    name = type(exc).__name__
    msg = str(exc)
    if "APIConnectionError" in name or "ConnectError" in name:
        return ServerUnreachableError(
            "Cannot connect to the Anthropic API.",
            hint="Check your internet connection.",
        )
    if "AuthenticationError" in name:
        return ModelError(
            "Anthropic API authentication failed.",
            hint=(
                "Check that ANTHROPIC_API_KEY is set correctly in your environment "
                "or ~/.claude/.env."
            ),
        )
    if "RateLimitError" in name:
        return ModelError(
            "Anthropic API rate limit exceeded.",
            hint="Wait a moment and try again, or check your usage limits.",
        )
    if "APIStatusError" in name or "APIError" in name:
        return ModelError(f"Anthropic API error: {msg}")
    return ModelError(f"Unexpected error from Claude backend: {msg}")
