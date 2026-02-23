from typing import Any, Callable

class MessageBus:
    """A simple in-memory message bus for agent-to-agent communication."""

    def __init__(self):
        # Maps agent_id -> list of messages in their inbox
        self._inboxes: dict[str, list[dict[str, Any]]] = {}
        # Simple broadcast subscribers (callbacks)
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []

    def _ensure_inbox(self, agent_id: str):
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = []

    def post(self, msg: dict[str, Any]) -> None:
        """
        Post a message to the bus.
        msg should have: 'from', 'to', 'content', and optionally 'type' (request, response, announce).
        """
        to_agent = msg.get("to")
        if not to_agent:
            return  # Invalid message

        if to_agent == "broadcast":
            # Call all subscribers or put in everyone's inbox?
            # For simplicity, we just trigger callbacks for now.
            for sub in self._subscribers:
                try:
                    sub(msg)
                except Exception as e:
                    pass
        else:
            self._ensure_inbox(to_agent)
            self._inboxes[to_agent].append(msg)

    def get_inbox(self, agent_id: str) -> list[dict[str, Any]]:
        """Return all messages for this agent, and clear the inbox."""
        self._ensure_inbox(agent_id)
        messages = self._inboxes[agent_id]
        self._inboxes[agent_id] = []
        return messages

    def peek_inbox(self, agent_id: str) -> list[dict[str, Any]]:
        """Return all messages without clearing the inbox."""
        self._ensure_inbox(agent_id)
        return list(self._inboxes[agent_id])

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to broadcast messages."""
        self._subscribers.append(callback)
