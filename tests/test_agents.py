"""Tests for src/agent_llm/agents.py"""

import pytest

from agent_llm.agents import create_delegate_tool, create_assign_tool


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

REGISTRY = [
    {"agent_id": "architect", "description": "Plans tasks"},
    {"agent_id": "coder", "description": "Writes code"},
]


def make_fake_tool(name: str = "write_file") -> dict:
    """Return a minimal tool dict that satisfies the Tool type alias."""
    return {
        "name": name,
        "description": "A tool",
        "parameters": {},
        "execute": lambda: "ok",
    }


class FakeTaskStore:
    """Minimal stand-in for RedisState that records push_task calls."""

    def __init__(self):
        self.calls = []

    def push_task(self, agent_id: str, msg: dict) -> None:
        self.calls.append((agent_id, msg))


# ---------------------------------------------------------------------------
# create_delegate_tool tests
# ---------------------------------------------------------------------------


def test_delegate_tool_sends_via_task_store():
    """Calling execute routes the message to the correct agent via task_store."""
    fake = FakeTaskStore()
    tool = create_delegate_tool(
        bus=None,
        sender_agent_id="architect",
        agent_registry=REGISTRY,
        task_store=fake,
    )

    result = tool["execute"](to_agent="coder", content="do it")

    assert len(fake.calls) == 1
    agent_id, msg = fake.calls[0]
    assert agent_id == "coder"
    assert msg["from"] == "architect"
    assert msg["content"] == "do it"
    assert result == "Message sent successfully to coder."


def test_delegate_tool_rejects_unknown_agent():
    """execute returns an error string when the target agent is not in the registry."""
    fake = FakeTaskStore()
    tool = create_delegate_tool(
        bus=None,
        sender_agent_id="architect",
        agent_registry=REGISTRY,
        task_store=fake,
    )

    result = tool["execute"](to_agent="nobody", content="x")

    assert result.startswith("Error:")
    assert len(fake.calls) == 0


def test_delegate_tool_silent_when_no_bus_no_store():
    """execute succeeds without crashing when both bus and task_store are None."""
    tool = create_delegate_tool(
        bus=None,
        task_store=None,
        sender_agent_id="a",
        agent_registry=REGISTRY,
    )

    result = tool["execute"](to_agent="coder", content="hi")

    assert result == "Message sent successfully to coder."


# ---------------------------------------------------------------------------
# create_assign_tool tests
# ---------------------------------------------------------------------------


def test_assign_tool_adds_to_agent_tools():
    """execute inserts the tool into the target agent's tool dict."""
    agent_tools = {"coder": {}}
    assignable = {"write_file": make_fake_tool("write_file")}

    tool = create_assign_tool(
        agent_tools,
        assignable,
        REGISTRY,
        bus=None,
    )

    result = tool["execute"](agent_id="coder", tool_name="write_file")

    assert "write_file" in agent_tools["coder"]
    assert agent_tools["coder"]["write_file"] is assignable["write_file"]


def test_assign_tool_notifies_via_task_store():
    """execute pushes a tool_grant message to the target agent via task_store."""
    fake = FakeTaskStore()
    agent_tools = {"coder": {}}
    assignable = {"write_file": make_fake_tool("write_file")}

    tool = create_assign_tool(
        agent_tools,
        assignable,
        REGISTRY,
        bus=None,
        task_store=fake,
    )

    tool["execute"](agent_id="coder", tool_name="write_file")

    assert len(fake.calls) == 1
    agent_id, msg = fake.calls[0]
    assert agent_id == "coder"
    assert msg["type"] == "tool_grant"


def test_assign_tool_rejects_unknown_agent():
    """execute returns an error string when the target agent is not in the registry."""
    agent_tools = {"coder": {}}
    assignable = {"write_file": make_fake_tool("write_file")}

    tool = create_assign_tool(agent_tools, assignable, REGISTRY, bus=None)

    result = tool["execute"](agent_id="nobody", tool_name="write_file")

    assert result.startswith("Error:")


def test_assign_tool_rejects_unknown_tool():
    """execute returns an error string when the requested tool is not assignable."""
    agent_tools = {"coder": {}}
    assignable = {"write_file": make_fake_tool("write_file")}

    tool = create_assign_tool(agent_tools, assignable, REGISTRY, bus=None)

    result = tool["execute"](agent_id="coder", tool_name="fly")

    assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# Regression: descriptions must contain real newlines, not literal \n sequences
# ---------------------------------------------------------------------------


def test_tool_descriptions_contain_real_newlines():
    """Tool descriptions use real newline characters, not the two-char literal \\n.

    Regression test: an earlier bug serialised descriptions with the escaped
    string '\\n' instead of an actual newline, breaking LLM prompt formatting.
    """
    agent_tools = {"coder": {}}
    assignable = {"write_file": make_fake_tool("write_file")}

    delegate = create_delegate_tool(
        bus=None,
        sender_agent_id="architect",
        agent_registry=REGISTRY,
        task_store=None,
    )
    assign = create_assign_tool(
        agent_tools,
        assignable,
        REGISTRY,
        bus=None,
    )

    for tool in (delegate, assign):
        desc = tool["description"]
        assert "\n" in desc, f"No real newline in description of {tool['name']!r}"
        # The two-character literal backslash-n must not appear
        assert "\\n" not in desc, (
            f"Literal '\\\\n' found in description of {tool['name']!r}"
        )
