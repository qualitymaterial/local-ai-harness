"""Configuration loading for local-ai.

Precedence (highest first):
  1. Environment variables (LOCAL_AI_*)
  2. .local-ai/config.toml in the target repo
  3. Built-in defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

try:  # Python 3.11+ has tomllib in the stdlib.
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import tomli_w

WORKSPACE_DIRNAME = ".local-ai"
CONFIG_FILENAME = "config.toml"


DEFAULTS: dict[str, object] = {
    # Local backend (LM Studio, OpenAI-compatible endpoint)
    "model": "qwen3-coder-30b",
    "base_url": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "embedding_model": "text-embedding-nomic-embed-text-v1.5",  # for semantic codebase search
    "max_context_tokens": 24000,
    "temperature": 0.2,
    "top_p": 0.9,
    "request_timeout": 600,
    # Backend selection
    "backend": "local",           # "local" (Ollama/LM Studio) | "claude"
    "claude_model": "claude-opus-4-8",
    "claude_api_key": "",         # prefer ANTHROPIC_API_KEY env var or ~/.claude/.env
    # Feature flags
    "stream": True,               # token-by-token streaming output
    "agentic": True,              # enable multi-step tool-use loop (reads files itself)
    "max_agent_iterations": 6,
    "auto_route": False,          # in chat, offer to escalate complex turns to Claude
}


@dataclass
class Config:
    model: str
    base_url: str
    api_key: str
    embedding_model: str
    max_context_tokens: int
    temperature: float
    top_p: float
    request_timeout: int
    backend: str
    claude_model: str
    claude_api_key: str
    stream: bool
    agentic: bool
    max_agent_iterations: int
    auto_route: bool

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Config":
        merged = {**DEFAULTS, **{k: v for k, v in data.items() if v is not None}}
        return cls(
            model=str(merged["model"]),
            base_url=str(merged["base_url"]).rstrip("/"),
            api_key=str(merged["api_key"]),
            embedding_model=str(merged["embedding_model"]),
            max_context_tokens=int(merged["max_context_tokens"]),  # type: ignore[arg-type]
            temperature=float(merged["temperature"]),  # type: ignore[arg-type]
            top_p=float(merged["top_p"]),  # type: ignore[arg-type]
            request_timeout=int(merged["request_timeout"]),  # type: ignore[arg-type]
            backend=str(merged["backend"]),
            claude_model=str(merged["claude_model"]),
            claude_api_key=str(merged["claude_api_key"]),
            stream=bool(merged["stream"]),
            agentic=bool(merged["agentic"]),
            max_agent_iterations=int(merged["max_agent_iterations"]),  # type: ignore[arg-type]
            auto_route=bool(merged["auto_route"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def active_model(self) -> str:
        return self.claude_model if self.backend == "claude" else self.model


def workspace_dir(repo_root: Path) -> Path:
    return repo_root / WORKSPACE_DIRNAME


def config_path(repo_root: Path) -> Path:
    return workspace_dir(repo_root) / CONFIG_FILENAME


def _load_file(repo_root: Path) -> dict[str, object]:
    path = config_path(repo_root)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError) as exc:  # type: ignore[attr-defined]
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc


def _env_overrides() -> dict[str, object]:
    out: dict[str, object] = {}
    if v := os.environ.get("LOCAL_AI_BASE_URL"):
        out["base_url"] = v
    if v := os.environ.get("LOCAL_AI_MODEL"):
        out["model"] = v
    if v := os.environ.get("LOCAL_AI_API_KEY"):
        out["api_key"] = v
    if v := os.environ.get("LOCAL_AI_MAX_CONTEXT_TOKENS"):
        try:
            out["max_context_tokens"] = int(v)
        except ValueError:
            raise ConfigError("LOCAL_AI_MAX_CONTEXT_TOKENS must be an integer")
    if v := os.environ.get("LOCAL_AI_BACKEND"):
        out["backend"] = v
    if v := os.environ.get("LOCAL_AI_STREAM"):
        out["stream"] = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("LOCAL_AI_AGENTIC"):
        out["agentic"] = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("LOCAL_AI_AUTO_ROUTE"):
        out["auto_route"] = v.lower() in ("1", "true", "yes")
    return out


def load_config(repo_root: Path) -> Config:
    file_data = _load_file(repo_root)
    env_data = _env_overrides()
    return Config.from_dict({**file_data, **env_data})


def ensure_default_config(repo_root: Path) -> Path:
    """Write a default config.toml if one does not already exist. Returns its path."""
    workspace_dir(repo_root).mkdir(parents=True, exist_ok=True)
    path = config_path(repo_root)
    if not path.exists():
        with path.open("wb") as fh:
            tomli_w.dump(DEFAULTS, fh)
    return path


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or parsed."""
