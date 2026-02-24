import json
import logging
from typing import Any, Dict, List
import redis

logger = logging.getLogger(__name__)


class RedisState:
    """Redis-backed state for tasks, workspace, and results."""

    def __init__(self, url: str):
        self.client = redis.from_url(url, decode_responses=True)

    def push_task(self, to_agent: str, msg: Dict[str, Any]) -> None:
        """Push a message task to an agent's queue."""
        key = f"agent_llm:tasks:{to_agent}"
        self.client.rpush(key, json.dumps(msg))
        logger.debug("Pushed task to %s: %s", to_agent, msg)

    def get_pending_task_count(self, agent_id: str) -> int:
        """Return number of pending tasks for an agent (does not clear)."""
        key = f"agent_llm:tasks:{agent_id}"
        return self.client.llen(key)

    def get_and_clear_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all pending tasks for an agent and clear the queue."""
        key = f"agent_llm:tasks:{agent_id}"
        # Pipeline to get all items and clear the list atomically
        pipe = self.client.pipeline(transaction=True)
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = pipe.execute()

        items = results[0] or []
        return [json.loads(item) for item in items]


class RedisWorkspace:
    """A Redis-backed workspace."""

    def __init__(self, state: RedisState, workspace_id: str = "default"):
        self.state = state
        self.workspace_key = f"agent_llm:workspace:{workspace_id}"

    def read_all(self) -> Dict[str, Any]:
        """Read all keys from the Redis hash."""
        data = self.state.client.hgetall(self.workspace_key)
        return {k: json.loads(v) for k, v in data.items()}

    def write_key(self, key: str, value: Any) -> None:
        """Write a JSON-serializable value to the workspace."""
        self.state.client.hset(self.workspace_key, key, json.dumps(value))
        logger.debug("Workspace key written: %s", key)

    def delete_key(self, key: str) -> None:
        """Delete a key from the workspace."""
        self.state.client.hdel(self.workspace_key, key)
        logger.debug("Workspace key deleted: %s", key)

    def clear(self) -> None:
        """Clear the entire workspace."""
        self.state.client.delete(self.workspace_key)
