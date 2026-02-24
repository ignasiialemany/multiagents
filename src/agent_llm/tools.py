"""
Tool model and implementations: list_files, read_file, search_docs.
All tools are restricted to a designated root directory (no path traversal).
Registry loading for Phase 2 custom tools.
"""

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Tool representation: name, description, JSON Schema parameters, execute callable
Tool = dict[str, Any]  # name, description, parameters, execute


def _resolve_safe(root: Path, path_str: str) -> Path | None:
    """Resolve path_str relative to root. Return None if outside root (e.g. '..')."""
    try:
        path_str = path_str.strip() or "."
        resolved = (root / path_str).resolve()
        root_resolved = root.resolve()
        if not resolved.is_relative_to(root_resolved):
            return None
        return resolved
    except Exception:
        return None


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    execute_fn: Callable[..., str],
) -> Tool:
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
        "execute": execute_fn,
    }


def create_default_tools(docs_root: str | Path) -> dict[str, Tool]:
    """Build the default tool registry: list_files, read_file, search_docs."""
    root = Path(docs_root).resolve()
    if not root.is_dir():
        raise ValueError(f"docs_root must be an existing directory: {root}")

    def list_files(path: str) -> str:
        p = _resolve_safe(root, path)
        if p is None:
            return "Error: path is outside the allowed directory."
        if not p.exists():
            return f"Error: path does not exist: {path}"
        if not p.is_dir():
            return "Error: path is not a directory."
        try:
            names = sorted(os.listdir(p))
            return "\n".join(names) if names else "(empty directory)"
        except OSError as e:
            return f"Error: {e}"

    def read_file(path: str) -> str:
        p = _resolve_safe(root, path)
        if p is None:
            return "Error: path is outside the allowed directory."
        if not p.exists():
            return f"Error: path does not exist: {path}"
        if not p.is_dir():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Error: {e}"
        return "Error: path is a directory, not a file."

    def search_docs(query: str) -> str:
        if not query or not query.strip():
            return "Error: query must be non-empty."
        query = query.strip()
        results: list[str] = []
        try:
            for fp in root.rglob("*"):
                if not fp.is_file():
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    if query.lower() in text.lower():
                        # Include a short snippet
                        idx = text.lower().find(query.lower())
                        start = max(0, idx - 40)
                        end = min(len(text), idx + len(query) + 40)
                        snippet = text[start:end].replace("\n", " ")
                        results.append(f"{fp.relative_to(root)}: ...{snippet}...")
                except (OSError, UnicodeDecodeError):
                    continue
        except OSError as e:
            return f"Error: {e}"
        if not results:
            return "No results found."
        return "\n".join(results[:20])  # cap at 20 matches

    return {
        "list_files": _tool(
            "list_files",
            "List files and directories at the given path (relative to the notes/docs root).",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under the notes/docs root, e.g. '.' or 'subfolder'.",
                    }
                },
                "required": ["path"],
            },
            list_files,
        ),
        "read_file": _tool(
            "read_file",
            "Read the full text content of a file at the given path (relative to the notes/docs root).",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file under the notes/docs root.",
                    }
                },
                "required": ["path"],
            },
            read_file,
        ),
        "search_docs": _tool(
            "search_docs",
            "Search for a query string in the contents of text files under the notes/docs folder.",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find in file contents.",
                    }
                },
                "required": ["query"],
            },
            search_docs,
        ),
    }


def load_registry_tools(registry_path: str | Path) -> dict[str, Tool]:
    """
    Load custom tools from tools/registry.json.
    Each entry must have: name, description, parameters, module, function.
    Returns dict[name -> Tool] (same shape as create_default_tools).
    """
    path = Path(registry_path)
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        entries = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(entries, list):
        return {}
    result: dict[str, Tool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        module_name = entry.get("module")
        function_name = entry.get("function")
        description = entry.get("description", "")
        parameters = entry.get("parameters")
        if (
            not name
            or not module_name
            or not function_name
            or not isinstance(parameters, dict)
        ):
            continue
        try:
            mod = importlib.import_module(module_name)
            execute_fn = getattr(mod, function_name)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "load_registry_tools: skipping tool %r — could not load %s.%s: %s",
                name,
                module_name,
                function_name,
                exc,
            )
            continue
        result[name] = _tool(name, description, parameters, execute_fn)
    return result


def create_assignable_tools(docs_root: str | Path) -> dict[str, Tool]:
    """Build a registry of tools that can be assigned dynamically (e.g. by a tool designer)."""
    root = Path(docs_root).resolve()
    if not root.is_dir():
        raise ValueError(f"docs_root must be an existing directory: {root}")

    def write_file(path: str, content: str) -> str:
        p = _resolve_safe(root, path)
        if p is None:
            return "Error: path is outside the allowed directory."
        if p.is_dir():
            return "Error: path is a directory, not a file."
        try:
            # create parent directories if needed
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {path}."
        except OSError as e:
            return f"Error writing file: {e}"

    return {
        "write_file": _tool(
            "write_file",
            "Write text content to a file at the given path (relative to the notes/docs root). Overwrites if exists.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file under the notes/docs root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
            write_file,
        ),
    }
