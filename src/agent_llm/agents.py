import json
from pathlib import Path
from typing import Any, Callable

from agent_llm.tools import _tool, Tool
from agent_llm.state_redis import RedisState


def load_agent_registry(registry_path: str | Path) -> list[dict[str, str]]:
    """Load the agent registry JSON which lists available agents and their capabilities."""
    path = Path(registry_path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_delegate_tool(
    bus,
    sender_agent_id: str,
    agent_registry: list[dict[str, str]],
    task_store: RedisState | None = None,
) -> Tool:
    """
    Create a tool that allows an agent to send messages (delegate) to other agents via the bus or task_store.
    The agent_registry provides the context of who is available.
    """
    known_agents = [agent["agent_id"] for agent in agent_registry]
    agent_descriptions = "\n".join(
        f"- {a['agent_id']}: {a['description']}"
        for a in agent_registry
        if a["agent_id"] != sender_agent_id
    )

    def delegate(to_agent: str, content: str) -> str:
        if to_agent not in known_agents:
            return f"Error: Unknown agent '{to_agent}'. Known agents are: {', '.join(known_agents)}"

        msg = {
            "from": sender_agent_id,
            "to": to_agent,
            "content": content,
            "type": "request",
        }

        if task_store is not None:
            try:
                task_store.push_task(to_agent, msg)
            except Exception as e:
                return f"Error: failed to enqueue task for {to_agent}: {e}"
        elif bus is not None:
            bus.post(msg)

        return f"Message sent successfully to {to_agent}."

    desc = f"Delegate a task or send a message to another agent. Available agents:\n{agent_descriptions}"

    return _tool(
        name="send_message",
        description=desc,
        parameters={
            "type": "object",
            "properties": {
                "to_agent": {
                    "type": "string",
                    "description": "The ID of the agent to send the message to.",
                },
                "content": {
                    "type": "string",
                    "description": "The message or task description.",
                },
            },
            "required": ["to_agent", "content"],
        },
        execute_fn=delegate,
    )


def create_assign_tool(
    agent_tools: dict[str, dict[str, Tool]],
    assignable_toolbox: dict[str, Tool],
    agent_registry: list[dict[str, str]],
    bus=None,
    sender_agent_id: str = "tool_designer",
    task_store: RedisState | None = None,
) -> Tool:
    """
    Create a tool that allows an agent (e.g. tool_designer) to assign predefined tools to other agents.
    If bus is provided, a message is posted to the target agent's inbox so they get a turn next loop.
    """
    known_agents = [agent["agent_id"] for agent in agent_registry]
    known_tools = list(assignable_toolbox.keys())

    def assign_tool(agent_id: str, tool_name: str) -> str:
        if agent_id not in known_agents:
            return f"Error: Unknown agent '{agent_id}'. Known agents are: {', '.join(known_agents)}"
        if tool_name not in assignable_toolbox:
            return f"Error: Unknown tool '{tool_name}'. Assignable tools are: {', '.join(known_tools)}"
        if agent_id not in agent_tools:
            return f"Error: Agent '{agent_id}' tool dictionary not initialized."

        agent_tools[agent_id][tool_name] = assignable_toolbox[tool_name]
        if task_store is not None:
            task_store.push_task(
                agent_id,
                {
                    "from": sender_agent_id,
                    "to": agent_id,
                    "content": (
                        f"You have been granted the '{tool_name}' tool. "
                        "Use it to continue your task."
                    ),
                    "type": "tool_grant",
                },
            )
        elif bus is not None:
            bus.post(
                {
                    "from": sender_agent_id,
                    "to": agent_id,
                    "content": f"You have been granted the '{tool_name}' tool. You can use it on your next turn.",
                    "type": "request",
                }
            )
        return f"Successfully assigned tool '{tool_name}' to agent '{agent_id}'."

    desc = (
        f"Assign a new tool to another agent so they can use it on their next turn.\n"
        f"Available agents to assign to: {', '.join(known_agents)}\n"
        f"Assignable tools: {', '.join(known_tools)}"
    )

    return _tool(
        name="assign_tool_to_agent",
        description=desc,
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent to receive the tool.",
                },
                "tool_name": {
                    "type": "string",
                    "description": "The name of the tool to assign.",
                },
            },
            "required": ["agent_id", "tool_name"],
        },
        execute_fn=assign_tool,
    )
