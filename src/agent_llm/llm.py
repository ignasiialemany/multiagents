"""
Thin wrapper around OpenRouter (OpenAI-compatible) chat completions API.
Exposes: complete(messages, tools) -> (content, tool_calls).
"""

import json
from typing import Any

from openai import OpenAI


# Type for a single tool call returned to the agent
ToolCall = dict[str, Any]  # {"id": str, "name": str, "input": dict}


class OpenRouterLLMClient:
    """Stateless client: call with messages + optional tools; returns content and tool_calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-sonnet-4",
        max_tokens: int = 4096,
    ):
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """
        Send messages (and optional tool definitions) to the model.
        Returns (text_content, tool_calls).
        - text_content: message content string.
        - tool_calls: list of {"id", "name", "input"} for each tool call.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        content = (msg.content or "").strip()
        tool_calls: list[ToolCall] = []

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                raw_args = getattr(tc.function, "arguments", None) if hasattr(tc, "function") else None
                try:
                    inp = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "name": name,
                    "input": inp,
                })

        return (content, tool_calls)
