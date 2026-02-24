"""Tests for src/agent_llm/tools.py"""

import json

import pytest

from agent_llm.tools import (
    _resolve_safe,
    create_assignable_tools,
    create_default_tools,
    load_registry_tools,
)


# ---------------------------------------------------------------------------
# _resolve_safe
# ---------------------------------------------------------------------------


def test_resolve_safe_allows_valid_path(tmp_path):
    """_resolve_safe returns a Path inside root for a legitimate relative path."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file.txt").write_text("content")

    result = _resolve_safe(tmp_path, "subdir/file.txt")

    assert result is not None
    assert result.is_relative_to(tmp_path)


def test_resolve_safe_blocks_traversal(tmp_path):
    """_resolve_safe returns None when the path escapes root via '..'."""
    result = _resolve_safe(tmp_path, "../../etc/passwd")

    assert result is None


def test_resolve_safe_blocks_prefix_confusion(tmp_path):
    """_resolve_safe returns None for a sibling dir whose name is a prefix of root.

    This guards against the path-traversal bug where 'notes_extended' could be
    mistaken for a subdirectory of 'notes' via a naive string prefix check.
    """
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    notes_extended = tmp_path / "notes_extended"
    notes_extended.mkdir()
    secret = notes_extended / "secret"
    secret.write_text("sensitive")

    result = _resolve_safe(notes_root, "../notes_extended/secret")

    assert result is None


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_names(tmp_path):
    """list_files('.') returns a string containing both filenames in the root."""
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.txt").write_text("b")
    tools = create_default_tools(tmp_path)
    list_files = tools["list_files"]["execute"]

    result = list_files(".")

    assert "alpha.txt" in result
    assert "beta.txt" in result


def test_list_files_blocks_traversal(tmp_path):
    """list_files returns an error string when the path escapes the root."""
    tools = create_default_tools(tmp_path)
    list_files = tools["list_files"]["execute"]

    result = list_files("../..")

    assert "Error" in result


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_content(tmp_path):
    """read_file returns the exact text content of an existing file."""
    target = tmp_path / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    tools = create_default_tools(tmp_path)
    read_file = tools["read_file"]["execute"]

    result = read_file("hello.txt")

    assert result == "hello world"


def test_read_file_blocks_traversal(tmp_path):
    """read_file returns a string starting with 'Error' for traversal paths."""
    tools = create_default_tools(tmp_path)
    read_file = tools["read_file"]["execute"]

    result = read_file("../../etc/passwd")

    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_file(tmp_path):
    """write_file creates the file on disk with the supplied content."""
    tools = create_assignable_tools(tmp_path)
    write_file = tools["write_file"]["execute"]

    write_file("out.txt", "hello")

    written = tmp_path / "out.txt"
    assert written.exists()
    assert written.read_text(encoding="utf-8") == "hello"


def test_write_file_blocks_traversal(tmp_path):
    """write_file returns a string containing 'Error' for traversal paths."""
    tools = create_assignable_tools(tmp_path)
    write_file = tools["write_file"]["execute"]

    result = write_file("../../pwned", "x")

    assert "Error" in result


# ---------------------------------------------------------------------------
# load_registry_tools
# ---------------------------------------------------------------------------


def test_load_registry_tools_skips_bad_entry_with_warning(tmp_path):
    """load_registry_tools returns an empty dict when the module cannot be imported.

    A bad entry (nonexistent module) must be silently skipped — no exception
    should propagate to the caller.
    """
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(
        json.dumps(
            [
                {
                    "name": "bad_tool",
                    "module": "nonexistent.module",
                    "function": "fn",
                    "description": "x",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = load_registry_tools(registry_file)

    assert result == {}
