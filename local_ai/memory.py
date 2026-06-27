"""Lightweight per-repo memory: an audit log of actions plus a cached index pointer.

Stored at .local-ai/memory.json. This is intentionally simple — it records what
local-ai did and when, so there is a clear audit trail inside .local-ai/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import workspace_dir

MEMORY_FILENAME = "memory.json"


def _memory_path(repo_root: Path) -> Path:
    return workspace_dir(repo_root) / MEMORY_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Memory:
    repo_root: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, repo_root: Path) -> "Memory":
        path = _memory_path(repo_root)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        data.setdefault("events", [])
        data.setdefault("last_index", None)
        return cls(repo_root=repo_root, data=data)

    def save(self) -> None:
        workspace_dir(self.repo_root).mkdir(parents=True, exist_ok=True)
        path = _memory_path(self.repo_root)
        path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def record(self, action: str, **details: Any) -> None:
        event = {"ts": _now_iso(), "action": action, **details}
        self.data["events"].append(event)
        # Keep the log bounded.
        if len(self.data["events"]) > 500:
            self.data["events"] = self.data["events"][-500:]

    def note_index(self, file_count: int, frameworks: list[str]) -> None:
        self.data["last_index"] = {
            "ts": _now_iso(),
            "file_count": file_count,
            "frameworks": frameworks,
        }

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        return list(self.data.get("events", []))[-n:]
