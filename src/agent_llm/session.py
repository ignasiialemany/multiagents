import json
import os
from pathlib import Path
from typing import Any

class SessionStore:
    """Stores and retrieves agent message histories by session_id."""

    def __init__(self, sessions_dir: str | Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Load messages for a given session. Returns empty list if not found."""
        path = self._get_path(session_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Save messages for a given session."""
        path = self._get_path(session_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a single message to a session."""
        messages = self.load(session_id)
        messages.append(message)
        self.save(session_id, messages)

    def get_all_sessions(self) -> list[str]:
        """Return a list of all session IDs."""
        if not self.sessions_dir.exists():
            return []
        return [p.stem for p in self.sessions_dir.glob("*.json")]
