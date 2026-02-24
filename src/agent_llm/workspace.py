import json
from pathlib import Path
from typing import Any, Protocol

from agent_llm.tools import _tool, Tool

class WorkspaceLike(Protocol):
    def read_all(self) -> Any:
        ...
    def write_key(self, key: str, value: Any) -> Any:
        ...

class Workspace:
    """
    A shared structured workspace (e.g., task board, common facts).
    Persists as a JSON file.
    """
    def __init__(self, workspace_file: str | Path):
        self.workspace_file = Path(workspace_file)
        # Initialize an empty dict if it doesn't exist
        if not self.workspace_file.exists():
            self.workspace_file.parent.mkdir(parents=True, exist_ok=True)
            self._save({})

    def _load(self) -> dict[str, Any]:
        try:
            with self.workspace_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        with self.workspace_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def read_all(self) -> str:
        data = self._load()
        return json.dumps(data, indent=2)

    def write_key(self, key: str, value: Any) -> str:
        data = self._load()
        data[key] = value
        self._save(data)
        return f"Successfully updated workspace key '{key}'."

    def append_task(self, task: str) -> str:
        data = self._load()
        if "tasks" not in data or not isinstance(data["tasks"], list):
            data["tasks"] = []
        data["tasks"].append({"task": task, "status": "pending"})
        self._save(data)
        return "Task appended to workspace."

def create_workspace_tools(workspace: WorkspaceLike) -> dict[str, Tool]:
    """Return dictionary of tools to interact with the workspace."""
    
    def read_workspace() -> str:
        data = workspace.read_all()
        if isinstance(data, str):
            return data
        return json.dumps(data, indent=2)

    def update_workspace(key: str, value: str) -> str:
        # Simple string for value, though could be json-parsed
        result = workspace.write_key(key, value)
        return str(result) if result is not None else f"Successfully updated workspace key '{key}'."

    return {
        "read_workspace": _tool(
            name="read_workspace",
            description="Read the entire shared workspace state (JSON format). Useful to see shared tasks or facts.",
            parameters={"type": "object", "properties": {}},
            execute_fn=read_workspace
        ),
        "update_workspace": _tool(
            name="update_workspace",
            description="Update a key in the shared workspace state.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to update in the shared workspace."},
                    "value": {"type": "string", "description": "The string value to set for the key."}
                },
                "required": ["key", "value"]
            },
            execute_fn=update_workspace
        )
    }
