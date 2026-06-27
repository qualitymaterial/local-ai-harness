"""Persistent multi-turn chat session management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Session:
    """A persisted conversation with the model, stored as JSON."""

    session_id: str
    repo_root: str
    created_at: str
    updated_at: str
    # Full message list in backend-wire format (role+content dicts).
    # The first entry is the system message when a system prompt is included.
    messages: list[dict] = field(default_factory=list)

    # ---------------------------------------------------------------- factories

    @classmethod
    def new(cls, repo_root: Path, *, system: str = "") -> "Session":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        now_iso = datetime.now().isoformat()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        return cls(
            session_id=f"session_{ts}",
            repo_root=str(repo_root),
            created_at=now_iso,
            updated_at=now_iso,
            messages=messages,
        )

    @classmethod
    def load(cls, path: Path) -> "Session":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load session from {path}: {exc}") from exc
        return cls(
            session_id=data.get("session_id", path.stem),
            repo_root=data.get("repo_root", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            messages=data.get("messages", []),
        )

    # ---------------------------------------------------------------- persistence

    def save(self, sessions_dir: Path) -> Path:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        path = sessions_dir / f"{self.session_id}.json"
        path.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "repo_root": self.repo_root,
                    "created_at": self.created_at,
                    "updated_at": self.updated_at,
                    "messages": self.messages,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    # ---------------------------------------------------------------- helpers

    @property
    def turn_count(self) -> int:
        """Number of user turns in this session."""
        return sum(1 for m in self.messages if m.get("role") == "user")

    def user_messages(self) -> list[str]:
        """Return just the text of user messages (for display)."""
        out = []
        for m in self.messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    out.append(content)
        return out
