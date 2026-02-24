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


# ── Spawn-subagent tool ─────────────────────────────────────────────────────


def create_spawn_tool(
    agents: dict,
    store,
    display,
    agent_registry: list[dict[str, str]],
) -> Tool:
    """
    Create a tool that lets the interactive agent spawn a subagent to handle
    a task synchronously, returning the subagent's final response.

    Parameters
    ----------
    agents : dict[str, Agent]
    store  : SessionStore
    display: RunDisplay
    agent_registry : list of agent info dicts
    """
    from agent_llm.agent import Agent

    agent_desc = "\n".join(
        f"- {a['agent_id']}: {a['description']}" for a in agent_registry
    )

    def spawn_subagent(agent_id: str, task: str) -> str:
        # Compute live so newly created agents are always recognised.
        known = [a["agent_id"] for a in agent_registry]
        if agent_id not in known:
            return f"Error: Unknown agent '{agent_id}'. Available: {', '.join(known)}"
        agent = agents.get(agent_id)
        if agent is None:
            return f"Error: Agent '{agent_id}' not initialised."

        messages = store.load(agent_id)
        if not messages:
            messages.append({
                "role": "system",
                "content": f"You are agent {agent_id}.",
            })

        display.agent_turn_start(agent_id, "interactive_agent", task)
        messages.append({
            "role": "user",
            "content": f"[Task from interactive agent]: {task}",
        })

        try:
            final_text, updated = agent.run(messages)
        except Exception as exc:
            display.agent_error(agent_id, str(exc))
            return f"Error: subagent failed: {exc}"

        store.save(agent_id, updated)
        display.agent_turn_end(agent_id, final_text or "")
        return final_text or "(no response from subagent)"

    return _tool(
        name="spawn_subagent",
        description=(
            "Spawn a subagent to handle a specific task. The subagent runs "
            "synchronously and returns its response.\n"
            f"Available agents:\n{agent_desc}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "ID of the subagent to spawn.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for the subagent.",
                },
            },
            "required": ["agent_id", "task"],
        },
        execute_fn=spawn_subagent,
    )


# ── Meeting tool: subagents discuss a topic and produce a plan ───────────────


def create_meeting_tool(
    agent_registry: list[dict],
    llm_client,
    display,
    run_meeting_fn,
    meeting_dir=None,
) -> Tool:
    """
    Create a tool that runs a meeting: subagents prepare, discuss, and produce a plan.
    If meeting_dir is set, the transcript and plan are saved to a file there.

    Parameters
    ----------
    agent_registry : list of agent info dicts
    llm_client : LLM client (for run_meeting_fn)
    display : RunDisplay (for run_meeting_fn)
    run_meeting_fn : callable(..., meeting_dir=...) -> str
    meeting_dir : optional path to save meeting files
    """
    from pathlib import Path

    agent_desc = "\n".join(
        f"- {a['agent_id']}: {a['description']}" for a in agent_registry
    )
    _meeting_dir = Path(meeting_dir) if meeting_dir else None

    def create_meeting(
        topic: str,
        agent_ids: str,
        max_rounds: int = 4,
    ) -> str:
        # Compute live so newly created agents are always recognised.
        known = [a["agent_id"] for a in agent_registry]
        ids = [x.strip() for x in agent_ids.split(",") if x.strip()]
        for a_id in ids:
            if a_id not in known:
                return f"Error: unknown agent '{a_id}'. Available: {', '.join(known)}"
        return run_meeting_fn(
            topic, ids, agent_registry, llm_client, display, max_rounds,
            meeting_dir=_meeting_dir,
        )

    return _tool(
        name="create_meeting",
        description=(
            "Create a meeting where multiple subagents discuss a topic together. "
            "They take turns speaking; after the discussion a summary/plan is produced. "
            "Use this to have the team align on a plan before executing.\n"
            f"Available agents:\n{agent_desc}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or question for the meeting.",
                },
                "agent_ids": {
                    "type": "string",
                    "description": "Comma-separated list of agent IDs to include (e.g. architect,coder,reviewer).",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Number of discussion rounds (default 4).",
                },
            },
            "required": ["topic", "agent_ids"],
        },
        execute_fn=create_meeting,
    )
