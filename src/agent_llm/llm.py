"""
Thin wrapper around OpenRouter (OpenAI-compatible) chat completions API.
Exposes: complete(messages, tools) -> (content, tool_calls).
Also provides embed() for embeddings and score_importance() for memory scoring.
"""

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Type for a single tool call returned to the agent
ToolCall = dict[str, Any]  # {"id": str, "name": str, "input": dict}

_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class OpenRouterLLMClient:
    """Stateless client: call with messages + optional tools; returns content and tool_calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "minimax/minimax-m2.5",
        max_tokens: int = 4096,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ):
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.embedding_model = embedding_model

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
        if not response.choices:
            return ("", [])
        msg = response.choices[0].message

        content = (msg.content or "").strip()
        tool_calls: list[ToolCall] = []

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.function.name or ""
                raw_args = tc.function.arguments
                try:
                    inp = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                tool_calls.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "name": name,
                        "input": inp,
                    }
                )

        return (content, tool_calls)

    # ── Embeddings ───────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text* using the configured embedding model."""
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    # ── Importance scoring ───────────────────────────────────────────────

    def score_importance(self, text: str) -> int:
        """
        Ask the LLM to rate the importance of a piece of text on a 1-10 scale.
        Returns an integer in [1, 10].
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory importance scorer. Given a piece of text, "
                    "rate its importance on a scale of 1 (mundane, routine) to "
                    "10 (critical, life-changing, core insight). "
                    "Reply with ONLY a single integer."
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            content, _ = self.complete(messages, tools=None)
            score = int(content.strip())
            return max(1, min(10, score))
        except (ValueError, TypeError):
            logger.debug("Importance scoring returned non-integer, defaulting to 5")
            return 5
