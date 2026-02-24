"""
Agent wrapper: takes llm_client and tools, exposes run(messages) with a tool-call loop.
"""

import json
import logging
from typing import Any

from agent_llm.llm import OpenRouterLLMClient

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

# Max chars of previous result to return when skipping a duplicate (full result up to this)
_DEDUP_PREVIOUS_RESULT_MAX_CHARS = 2000


def _normalize_path_for_key(val: Any) -> str:
    """Normalize path-like args for dedup key so '.' and '' and 'foo/' match."""
    if val is None:
        return "."
    s = str(val).strip()
    if not s or s == ".":
        return "."
    return s.rstrip("/") or "."


def _args_for_key(n: str, i: dict) -> dict:
    """Normalise path-like args so dedup keys are canonical."""
    out = dict(i)
    for key in ("path", "root"):
        if key in out and out[key] is not None:
            out[key] = _normalize_path_for_key(out[key])
    return out


def call_key(n: str, i: dict) -> tuple[str, str]:
    """Stable dedup key for a tool call: (name, canonical-json-args)."""
    return (n, json.dumps(_args_for_key(n, i), sort_keys=True))


def _to_openrouter_tools(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert internal tools dict to OpenRouter/OpenAI API format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools.values()
    ]


class Agent:
    """Single agent that calls the LLM and executes tools until a final answer."""

    def __init__(
        self,
        llm_client: OpenRouterLLMClient,
        tools: dict[str, Any],
        agent_id: str = "agent",
    ):
        """
        Args:
            llm_client: Object with complete(messages, tools) -> (content, tool_calls).
            tools: dict[name -> Tool] where Tool has name, description, parameters, execute.
            agent_id: Human-readable identifier used in log messages.
        """
        self.llm_client = llm_client
        self.tools = tools
        self.agent_id = agent_id

    def run(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """
        Run the agent: call LLM with messages and tool definitions; if the model
        returns tool calls, execute them and loop until a final answer.

        Returns:
            (final_text, updated_messages).
        """
        messages = list(messages)

        # Build dedup map once from incoming history; updated incrementally inside the loop.
        seen_tool_results: dict[tuple[str, str], str] = {}
        for _i, _m in enumerate(messages):
            if _m.get("role") != "assistant" or not _m.get("tool_calls"):
                continue
            _tool_results: list[str] = []
            for _j in range(_i + 1, len(messages)):
                if messages[_j].get("role") == "tool":
                    _tool_results.append(messages[_j].get("content", ""))
                elif messages[_j].get("role") == "assistant":
                    break
            for _k, _tc in enumerate(_m.get("tool_calls", [])):
                _fn = _tc.get("function") or {}
                _n = _fn.get("name", "?")
                _raw = _fn.get("arguments") or "{}"
                try:
                    _inp = json.loads(_raw) if isinstance(_raw, str) else {}
                except (json.JSONDecodeError, TypeError):
                    _inp = {}
                _key = call_key(_n, _inp)
                if _k < len(_tool_results):
                    seen_tool_results[_key] = _tool_results[_k]

        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            # so is the LLM that receives message and tools
            current_openrouter_tools = (
                _to_openrouter_tools(self.tools) if self.tools else None
            )
            content, tool_calls = self.llm_client.complete(
                messages,
                current_openrouter_tools,
            )

            # Build assistant message (OpenRouter/OpenAI style)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content if content else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(
                                tc["input"] if tc.get("input") is not None else {}
                            ),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                return (content, messages)

            # Execute each tool call and append one "tool" message per result
            for tc in tool_calls:
                tool_id = tc["id"]
                name = tc["name"]
                inp = tc["input"] if tc["input"] is not None else {}
                logger.info(
                    "[%s] tool selected: name=%s, args=%s", self.agent_id, name, inp
                )

                dup_key = call_key(name, inp)
                if dup_key in seen_tool_results:
                    prev = seen_tool_results[dup_key]
                    cap = _DEDUP_PREVIOUS_RESULT_MAX_CHARS
                    msg = prev if len(prev) <= cap else prev[:cap] + "..."
                    logger.info(
                        "[%s] skipping duplicate tool call: %s(%s)",
                        self.agent_id,
                        name,
                        inp,
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_id, "content": msg}
                    )
                    continue

                tool = self.tools.get(name)
                if tool is None:
                    msg = f"Error: unknown tool '{name}'"
                    logger.warning("[%s] unknown tool: %s", self.agent_id, name)
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_id, "content": msg}
                    )
                    continue

                # Restrict to schema properties to avoid "unexpected keyword argument"
                params_schema = tool.get("parameters") or {}
                allowed = set((params_schema.get("properties") or {}).keys())
                if allowed:
                    inp = {k: v for k, v in inp.items() if k in allowed}

                try:
                    out = tool["execute"](**inp)
                    result_str = out if isinstance(out, str) else str(out)
                    seen_tool_results[dup_key] = result_str
                    logger.info(
                        "[%s] tool result: %s",
                        self.agent_id,
                        result_str[:200] + "..."
                        if len(result_str) > 200
                        else result_str,
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_id, "content": result_str}
                    )
                except Exception as e:
                    msg = f"Error: {e}"
                    logger.exception(
                        "[%s] tool execution failed: name=%s", self.agent_id, name
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_id, "content": msg}
                    )

        return (
            "Agent stopped: maximum iterations reached without a final answer.",
            messages,
        )
