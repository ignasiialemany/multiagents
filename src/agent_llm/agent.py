"""
Agent wrapper: takes llm_client and tools, exposes run(messages) with a tool-call loop.
"""

import json
import logging
from typing import Any
from agent_llm.llm import OpenRouterLLMClient

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


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

    def __init__(self, llm_client: OpenRouterLLMClient, tools: dict[str, Any]):
        """
        Args:
            llm_client: Object with complete(messages, tools) -> (content, tool_calls).
            tools: dict[name -> Tool] where Tool has name, description, parameters, execute.
        """
        self.llm_client = llm_client
        self.tools = tools
        self._openrouter_tools = _to_openrouter_tools(tools) if tools else []

    def run(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """
        Run the agent: call LLM with messages and tool definitions; if the model
        returns tool calls, execute them and loop until a final answer.

        Returns:
            (final_text, updated_messages).
        """
        messages = list(messages)
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            #so is the LLM that receives message and tools
            current_openrouter_tools = _to_openrouter_tools(self.tools) if self.tools else None
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
                            "arguments": json.dumps(tc["input"] if tc.get("input") else {}),
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
                inp = tc["input"] or {}
                logger.info("tool selected: name=%s, args=%s", name, inp)

                tool = self.tools.get(name)
                if tool is None:
                    msg = f"Error: unknown tool '{name}'"
                    logger.warning("unknown tool: %s", name)
                    messages.append({"role": "tool", "tool_call_id": tool_id, "content": msg})
                    continue

                try:
                    out = tool["execute"](**inp)
                    result_str = out if isinstance(out, str) else str(out)
                    logger.info("tool result: %s", result_str[:200] + "..." if len(result_str) > 200 else result_str)
                    messages.append({"role": "tool", "tool_call_id": tool_id, "content": result_str})
                except Exception as e:
                    msg = f"Error: {e}"
                    logger.exception("tool execution failed: name=%s", name)
                    messages.append({"role": "tool", "tool_call_id": tool_id, "content": msg})

        return (
            "Agent stopped: maximum iterations reached without a final answer.",
            messages,
        )
