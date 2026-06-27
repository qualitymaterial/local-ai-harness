"""Agentic multi-step loop: call model → execute tools → repeat until done."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .backends.base import BaseBackend, AgentResponse
from .tools import TOOL_DEFINITIONS, execute_tool


def run_agent(
    backend: BaseBackend,
    messages: list[dict],
    *,
    repo_root: Path,
    max_iterations: int = 5,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
) -> AgentResponse:
    """
    Run the model in a tool-use loop until it stops calling tools or `max_iterations`
    is reached. Each backend handles its own message format via `append_tool_messages`.

    `on_tool_call(tool_name, tool_input)` is called before each tool execution (for UI).

    Returns the final AgentResponse (the last response that had no tool calls, or the
    last response if the iteration budget ran out).
    """
    msgs = list(messages)  # local copy — don't mutate the caller's list
    last_response: Optional[AgentResponse] = None

    for _iteration in range(max_iterations):
        response = backend.complete(msgs, tools=TOOL_DEFINITIONS)
        last_response = response

        if not response.tool_calls:
            return response

        # Collect tool results
        results: list[tuple[str, str, str]] = []
        for tc in response.tool_calls:
            if on_tool_call:
                on_tool_call(tc.name, tc.input)
            result_text = execute_tool(tc.name, tc.input, repo_root)
            results.append((tc.id, tc.name, result_text))

        # Append in backend-native format (Anthropic vs OpenAI differ here)
        backend.append_tool_messages(msgs, response, results)

    # Budget exhausted — do one final call without tools so the model can summarize
    if last_response and last_response.tool_calls:
        final = backend.complete(msgs)
        return final

    return last_response or AgentResponse(content="")
