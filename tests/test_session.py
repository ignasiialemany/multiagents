"""Tests for src/agent_llm/session.py"""

import json

import pytest

from agent_llm.session import SessionStore


def test_load_returns_empty_for_missing_session(tmp_path):
    """Loading a non-existent session ID returns an empty list without raising."""
    store = SessionStore(tmp_path)
    result = store.load("nonexistent")
    assert result == []


def test_save_and_load_roundtrip(tmp_path):
    """Messages saved under a session ID are returned unchanged by load()."""
    store = SessionStore(tmp_path)
    messages = [{"role": "user", "content": "hello"}]
    store.save("s1", messages)
    assert store.load("s1") == messages


def test_save_is_atomic_on_disk(tmp_path):
    """The file written by save() contains valid JSON (atomic write succeeded)."""
    store = SessionStore(tmp_path)
    messages = [{"role": "assistant", "content": "hi"}]
    store.save("s2", messages)
    path = tmp_path / "s2.json"
    # Must not raise — file must be parseable JSON
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == messages


def test_get_all_sessions_returns_ids(tmp_path):
    """get_all_sessions() includes every session ID that has been saved."""
    store = SessionStore(tmp_path)
    store.save("alice", [{"role": "user", "content": "a"}])
    store.save("bob", [{"role": "user", "content": "b"}])
    sessions = store.get_all_sessions()
    assert "alice" in sessions
    assert "bob" in sessions


def test_append_adds_message(tmp_path):
    """append() adds a message to an existing session, giving two messages in total."""
    store = SessionStore(tmp_path)
    first = {"role": "user", "content": "first"}
    second = {"role": "assistant", "content": "second"}
    store.save("s3", [first])
    store.append("s3", second)
    assert store.load("s3") == [first, second]


def test_load_returns_empty_on_corrupt_file(tmp_path):
    """load() returns [] without raising when the session file contains invalid JSON."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    corrupt = sessions_dir / "bad.json"
    corrupt.write_text("NOT JSON {{", encoding="utf-8")
    store = SessionStore(sessions_dir)
    assert store.load("bad") == []
