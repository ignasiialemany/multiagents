"""Tests for src/agent_llm/state_redis.py — uses fakeredis (no live Redis needed)"""

import fakeredis
from agent_llm.state_redis import RedisState, RedisWorkspace


def make_state() -> RedisState:
    """Return a RedisState instance backed by an in-memory FakeRedis client."""
    state = RedisState.__new__(RedisState)
    state.client = fakeredis.FakeRedis(decode_responses=True)
    return state


def make_workspace(state: RedisState) -> RedisWorkspace:
    """Return a RedisWorkspace instance wired to the given state."""
    ws = RedisWorkspace.__new__(RedisWorkspace)
    ws.state = state
    ws.workspace_key = "agent_llm:workspace:test"
    return ws


# ---------------------------------------------------------------------------
# RedisState tests
# ---------------------------------------------------------------------------


def test_push_and_get_tasks():
    """Push two different tasks; get_and_clear_tasks returns both in order."""
    state = make_state()

    task_a = {"from": "agent_a", "to": "agent_b", "content": "hello"}
    task_b = {"from": "agent_a", "to": "agent_b", "content": "world"}

    state.push_task("agent_b", task_a)
    state.push_task("agent_b", task_b)

    tasks = state.get_and_clear_tasks("agent_b")

    assert len(tasks) == 2
    assert tasks[0] == task_a
    assert tasks[1] == task_b


def test_get_and_clear_is_destructive():
    """After get_and_clear_tasks returns items, a second call returns an empty list."""
    state = make_state()

    state.push_task("agent_x", {"from": "src", "to": "agent_x", "content": "ping"})

    first = state.get_and_clear_tasks("agent_x")
    assert len(first) == 1

    second = state.get_and_clear_tasks("agent_x")
    assert second == []


def test_get_pending_task_count():
    """Pushing 3 tasks yields count 3; count drops to 0 after get_and_clear_tasks."""
    state = make_state()

    for i in range(3):
        state.push_task("agent_c", {"from": "src", "to": "agent_c", "content": i})

    assert state.get_pending_task_count("agent_c") == 3

    state.get_and_clear_tasks("agent_c")

    assert state.get_pending_task_count("agent_c") == 0


def test_push_task_stores_valid_json():
    """A task with a nested dict round-trips through push/get_and_clear identically."""
    state = make_state()

    original = {"from": "a", "to": "b", "content": {"key": 1}}
    state.push_task("b", original)

    tasks = state.get_and_clear_tasks("b")

    assert len(tasks) == 1
    assert tasks[0] == original


# ---------------------------------------------------------------------------
# RedisWorkspace tests
# ---------------------------------------------------------------------------


def test_redis_workspace_write_and_read():
    """Writing a key to the workspace and calling read_all returns it with the correct value."""
    state = make_state()
    ws = make_workspace(state)

    ws.write_key("color", "blue")

    data = ws.read_all()

    assert "color" in data
    assert data["color"] == "blue"


def test_redis_workspace_delete_key():
    """Deleting a key from the workspace removes it from read_all."""
    state = make_state()
    ws = make_workspace(state)

    ws.write_key("temp", 42)
    assert "temp" in ws.read_all()

    ws.delete_key("temp")

    assert "temp" not in ws.read_all()
