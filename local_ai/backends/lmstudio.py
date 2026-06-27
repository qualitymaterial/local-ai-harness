"""LM Studio / OpenAI-compatible backend with SSE streaming."""

from __future__ import annotations

import json
import re
from typing import Iterator, Optional

import httpx

from ..model_client import (
    ModelClient,
    ModelError,
    ServerUnreachableError,
    EmptyResponseError,
)
from ..config import Config
from ..types import ChatMessage
from .base import BaseBackend, AgentResponse, ToolCall


class LMStudioBackend(BaseBackend):
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = ModelClient(config)

    # ------------------------------------------------------------------ stream

    def stream(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
        }
        if tools:
            payload["tools"] = _to_openai_tools(tools)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.config.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        _raise_for_status(resp, self.config)
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
        except httpx.ConnectError as exc:
            raise ServerUnreachableError(
                f"Could not connect to {self.config.base_url}.",
                hint=(
                    "Is LM Studio running with the local server started?\n"
                    "In LM Studio: Developer tab → Start Server (default port 1234)."
                ),
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ModelError(
                f"The model did not respond within {self.config.request_timeout}s.",
                hint="Increase 'request_timeout' in .local-ai/config.toml.",
            ) from exc

    # ---------------------------------------------------------------- complete

    def complete(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
    ) -> AgentResponse:
        if not tools:
            # Fast path: delegate to the robust ModelClient (which has full error handling).
            chat_msgs = [
                ChatMessage(
                    role=m["role"],
                    content=m["content"] if isinstance(m["content"], str) else json.dumps(m["content"]),
                )
                for m in messages
                if isinstance(m.get("content"), (str, list))
            ]
            resp = self._client.chat(chat_msgs)
            return AgentResponse(
                content=resp.content,
                finish_reason=resp.finish_reason or "stop",
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
            )

        return self._complete_with_tools(messages, tools)

    def _complete_with_tools(
        self, messages: list[dict], tools: list[dict]
    ) -> AgentResponse:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False,
            "tools": _to_openai_tools(tools),
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.config.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise ServerUnreachableError(
                f"Could not connect to {self.config.base_url}.",
                hint="Is LM Studio running?",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ModelError(
                f"Model did not respond within {self.config.request_timeout}s.",
                hint="Increase 'request_timeout' in .local-ai/config.toml.",
            ) from exc

        _raise_for_status(resp, self.config)

        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelError("Model returned a non-JSON response.") from exc

        choices = data.get("choices") or []
        if not choices:
            raise EmptyResponseError("Model returned no choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()
        tool_calls: list[ToolCall] = []

        # Standard OpenAI tool_calls array
        for tc in message.get("tool_calls") or []:
            func = tc.get("function") or {}
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    input=args,
                )
            )

        # Fallback: parse DeepSeek's embedded tool-call marker format
        if not tool_calls and content:
            content, tool_calls = _parse_deepseek_tool_calls(content)

        usage = data.get("usage") or {}
        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    # ------------------------------------------------------ tool message helpers

    def append_tool_messages(
        self,
        messages: list[dict],
        response: AgentResponse,
        results: list[tuple[str, str, str]],
    ) -> None:
        """Append in OpenAI tool-call / tool-result format."""
        # Assistant message with tool_calls array
        tool_calls_payload = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.input),
                },
            }
            for tc in response.tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": tool_calls_payload,
        })
        # One tool-role message per result
        for call_id, _name, result_text in results:
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_text,
            })

    # ---------------------------------------------------------------- health

    def health_check(self) -> bool:
        return self._client.health_check()


# ------------------------------------------------------------------ helpers

# DeepSeek uses special Unicode markers for tool calls embedded in text content
# rather than the standard OpenAI tool_calls array.
_DS_TOOL_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>(\w+)\s*```(?:json)?\s*(\{.*?\})\s*```<｜tool▁call▁end｜>",
    re.DOTALL,
)
_DS_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_TOOL_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"


def _parse_deepseek_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """
    Parse DeepSeek's embedded tool-call markers from text content.

    Returns (cleaned_content, tool_calls) where cleaned_content has the tool
    call block and any hallucinated output section stripped.
    """
    if _DS_TOOL_CALLS_BEGIN not in content:
        return content, []

    # Strip the tool-calls block and everything after (model often hallucinates
    # its own tool outputs section which we don't want).
    pre = content[: content.index(_DS_TOOL_CALLS_BEGIN)]
    tool_block = content[content.index(_DS_TOOL_CALLS_BEGIN):]
    # Also strip any hallucinated <｜tool▁outputs▁begin｜> section
    if _DS_TOOL_OUTPUTS_BEGIN in tool_block:
        tool_block = tool_block[: tool_block.index(_DS_TOOL_OUTPUTS_BEGIN)]

    tool_calls: list[ToolCall] = []
    for i, match in enumerate(_DS_TOOL_CALL_RE.finditer(tool_block)):
        func_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(ToolCall(id=f"call_{i}", name=func_name, input=args))

    return pre.strip(), tool_calls


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _raise_for_status(resp: httpx.Response, config: Config) -> None:
    if resp.status_code == 200:
        return
    from ..model_client import ModelNotLoadedError, ContextTooLargeError

    if resp.status_code == 404:
        raise ModelNotLoadedError(
            f"Model endpoint returned 404 for model '{config.model}'.",
            hint=(
                "The model name may not match what's loaded in LM Studio.\n"
                "Check the exact model identifier in LM Studio's server panel."
            ),
        )
    if resp.status_code == 400:
        body = resp.text.lower()
        if "context" in body or "token" in body or "length" in body:
            raise ContextTooLargeError(
                "The request exceeded the model's context window.",
                hint="Lower 'max_context_tokens' in .local-ai/config.toml.",
            )
        raise ModelError(f"Model returned 400 Bad Request: {resp.text[:500]}")
    raise ModelError(
        f"Model returned HTTP {resp.status_code}: {resp.text[:500]}",
        hint="Check the LM Studio server logs for details.",
    )
