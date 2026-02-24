"""Tests for cross-repo portability features (cli.py, _runner.py, tools_custom.py)"""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# cli._init
# ---------------------------------------------------------------------------


def test_init_creates_expected_files(tmp_path):
    """_init should create agents.json, tools.json, .env.example, and a notes/ dir."""
    from agent_llm.cli import _init

    _init(tmp_path)

    assert (tmp_path / ".agent-llm" / "agents.json").exists()
    assert (tmp_path / ".agent-llm" / "tools.json").exists()
    assert (tmp_path / ".agent-llm" / ".env.example").exists()
    assert (tmp_path / "notes").is_dir()

    agents = json.loads((tmp_path / ".agent-llm" / "agents.json").read_text())
    assert isinstance(agents, list) and len(agents) > 0
    assert "agent_id" in agents[0]


def test_init_does_not_overwrite_existing(tmp_path):
    """_init must leave pre-existing config files untouched."""
    from agent_llm.cli import _init

    agents_file = tmp_path / ".agent-llm" / "agents.json"
    agents_file.parent.mkdir(parents=True, exist_ok=True)
    agents_file.write_text("CUSTOM_CONTENT")

    _init(tmp_path)

    assert agents_file.read_text() == "CUSTOM_CONTENT"


def test_init_is_idempotent(tmp_path):
    """Calling _init twice on the same directory must not raise."""
    from agent_llm.cli import _init

    _init(tmp_path)
    _init(tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# _runner._resolve_config_path
# ---------------------------------------------------------------------------


def test_resolve_config_path_uses_override(tmp_path):
    """An explicit override string should be returned (resolved to absolute Path)."""
    from agent_llm._runner import _resolve_config_path

    result = _resolve_config_path(
        tmp_path,
        "/custom/override.json",
        tmp_path / "primary.json",
        tmp_path / "fallback.json",
    )
    # _resolve_config_path calls Path(override).resolve(), so compare resolved form.
    assert result == Path("/custom/override.json").resolve()


def test_resolve_config_path_uses_primary_when_exists(tmp_path):
    """When no override is given and the primary path exists, it should be returned."""
    from agent_llm._runner import _resolve_config_path

    primary = tmp_path / "primary.json"
    primary.touch()

    result = _resolve_config_path(
        tmp_path,
        None,
        primary,
        tmp_path / "fallback.json",
    )
    assert result == primary


def test_resolve_config_path_falls_back_when_primary_missing(tmp_path):
    """When no override is given and the primary path does not exist, return fallback."""
    from agent_llm._runner import _resolve_config_path

    fallback = tmp_path / "fallback.json"

    result = _resolve_config_path(
        tmp_path,
        None,
        tmp_path / "missing.json",
        fallback,
    )
    assert result == fallback


# ---------------------------------------------------------------------------
# tools_custom.grep_repo
# ---------------------------------------------------------------------------


def test_grep_repo_requires_explicit_path():
    """grep_repo called without a path/root argument should return an error string."""
    from agent_llm.tools_custom import grep_repo

    result = grep_repo(pattern="hello")

    assert result.startswith("Error:")
    assert "path" in result.lower()


# ---------------------------------------------------------------------------
# _runner._parse_args
# ---------------------------------------------------------------------------


def test_parse_args_work_dir_flag():
    """--work-dir should be captured in args.work_dir; remaining words become args.task."""
    from agent_llm._runner import _parse_args

    args = _parse_args(["--work-dir", "/tmp/myproject", "do", "something"])

    assert args.work_dir == "/tmp/myproject"
    assert args.task == ["do", "something"]


def test_parse_args_agents_registry_flag():
    """--agents-registry should be stored in args.agents_registry."""
    from agent_llm._runner import _parse_args

    args = _parse_args(["--agents-registry", "/tmp/agents.json", "task"])

    assert args.agents_registry == "/tmp/agents.json"


def test_parse_args_model_flag():
    """--model should be stored verbatim in args.model."""
    from agent_llm._runner import _parse_args

    args = _parse_args(["--model", "anthropic/claude-3", "task"])

    assert args.model == "anthropic/claude-3"
