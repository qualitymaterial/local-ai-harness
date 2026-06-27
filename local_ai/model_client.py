"""OpenAI-compatible chat client for LM Studio (and similar local servers)."""

from __future__ import annotations

from typing import Iterable, Optional

import httpx

from .config import Config
from .types import ChatMessage, ModelResponse


class ModelError(Exception):
    """Base class for model client errors, carrying a user-facing hint."""

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.hint = hint


class ServerUnreachableError(ModelError):
    pass


class ModelNotLoadedError(ModelError):
    pass


class ContextTooLargeError(ModelError):
    pass


class EmptyResponseError(ModelError):
    pass


class ModelClient:
    """Thin wrapper over POST {base_url}/chat/completions with robust error handling."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _endpoint(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                resp = client.post(self._endpoint(), json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise ServerUnreachableError(
                f"Could not connect to {self.config.base_url}.",
                hint=(
                    "Is LM Studio running with the local server started?\n"
                    "In LM Studio: Developer tab → Start Server (default port 1234).\n"
                    f"Or set LOCAL_AI_BASE_URL to the correct address."
                ),
            ) from exc
        except httpx.ConnectTimeout as exc:
            raise ServerUnreachableError(
                f"Connection to {self.config.base_url} timed out.",
                hint="Confirm LM Studio's server is started and reachable.",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ModelError(
                f"The model did not respond within {self.config.request_timeout}s.",
                hint=(
                    "Local models can be slow. Increase 'request_timeout' in "
                    ".local-ai/config.toml, or use a smaller/faster model."
                ),
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelError(f"HTTP error talking to the model: {exc}") from exc

        return self._parse_response(resp)

    def _parse_response(self, resp: httpx.Response) -> ModelResponse:
        if resp.status_code == 404:
            raise ModelNotLoadedError(
                f"Model endpoint returned 404 for model '{self.config.model}'.",
                hint=(
                    "The model name may not match what's loaded in LM Studio.\n"
                    "Check the exact model identifier in LM Studio's server panel "
                    "and set it via 'model' in .local-ai/config.toml or LOCAL_AI_MODEL."
                ),
            )
        if resp.status_code == 400:
            body = _safe_text(resp).lower()
            if "context" in body or "token" in body or "length" in body:
                raise ContextTooLargeError(
                    "The request exceeded the model's context window.",
                    hint=(
                        "Lower 'max_context_tokens' in .local-ai/config.toml (or "
                        "LOCAL_AI_MAX_CONTEXT_TOKENS), or load a model with a larger "
                        "context length in LM Studio."
                    ),
                )
            raise ModelError(f"Model returned 400 Bad Request: {_safe_text(resp)[:500]}")
        if resp.status_code != 200:
            raise ModelError(
                f"Model returned HTTP {resp.status_code}: {_safe_text(resp)[:500]}",
                hint="Check the LM Studio server logs for details.",
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ModelError(
                "Model returned a non-JSON response.",
                hint="The server may have crashed mid-generation. Check LM Studio logs.",
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise EmptyResponseError(
                "Model returned no choices.",
                hint="The model produced no output. Try again or reload the model.",
            )
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise EmptyResponseError(
                "Model returned an empty message.",
                hint="Try increasing temperature slightly or reloading the model.",
            )

        usage = data.get("usage") or {}
        return ModelResponse(
            content=content,
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choices[0].get("finish_reason"),
        )

    def health_check(self) -> bool:
        """Return True if the server is reachable (best-effort GET /models)."""
        url = f"{self.config.base_url}/models"
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(
                    url, headers={"Authorization": f"Bearer {self.config.api_key}"}
                )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _safe_text(resp: httpx.Response) -> str:
    try:
        return resp.text
    except Exception:  # pragma: no cover - defensive
        return "<unreadable response body>"
