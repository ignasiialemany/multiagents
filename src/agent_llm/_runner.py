#!/usr/bin/env python3
"""
Supervisor-routed parallel runner — importable module.

All agents use Redis for shared state (tasks, workspace).
The coordinator pops tasks from Redis and runs agents in parallel.

Usage (via shim)
----------------
    python scripts/run_supervisor_parallel.py "Build a hello world module"
    python scripts/run_supervisor_parallel.py --verbose "..."
    python scripts/run_supervisor_parallel.py --quiet   "..."
    python scripts/run_supervisor_parallel.py --transcript "..."
    python scripts/run_supervisor_parallel.py --work-dir /path/to/project "..."
    python scripts/run_supervisor_parallel.py --help

Usage (direct, from any directory)
-----------------------------------
    python -m agent_llm._runner "Build a hello world module"
    python -m agent_llm._runner --work-dir /path/to/project "..."
"""

import argparse
import json
import logging
import os
import sys
import concurrent.futures
import time
import traceback
from pathlib import Path

# NOTE: No sys.path manipulation — this file lives inside the package.
# NOTE: No module-level load_dotenv — deferred to _run_supervisor() so work_dir is known.

from agent_llm.agent import Agent
from agent_llm.llm import OpenRouterLLMClient
from agent_llm.tools import (
    create_default_tools,
    load_registry_tools,
    create_assignable_tools,
)
from agent_llm.session import SessionStore
from agent_llm.workspace import create_workspace_tools
from agent_llm.agents import (
    load_agent_registry,
    create_delegate_tool,
    create_assign_tool,
)
from agent_llm.state_redis import RedisState, RedisWorkspace
from agent_llm.tools_sandbox import create_sandbox_tool
from agent_llm.cli_output import RunDisplay

logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Config-path resolver ────────────────────────────────────────────────────


def _resolve_config_path(
    work_dir: Path,
    override: str | None,
    primary: Path,
    fallback: str | Path,
) -> Path:
    """
    Resolve a configuration path with a three-tier precedence:

    1. ``override``  — explicit CLI value (returned as-is, resolved to absolute).
    2. ``primary``   — preferred default location; used only when it already exists.
    3. ``fallback``  — last-resort default (always returned if primary absent).

    Parameters
    ----------
    work_dir:
        The resolved working directory.  Unused in path construction here
        (callers pass fully-constructed Path objects), but kept for API clarity.
    override:
        Raw string from a CLI flag, or ``None`` when the flag was not supplied.
    primary:
        Preferred path (e.g. ``work_dir / ".agent-llm" / "agents.json"``).
    fallback:
        Legacy / alternate path (e.g. ``work_dir / "agents" / "registry.json"``).

    Returns
    -------
    Path
        An absolute Path.
    """
    if override:
        return Path(override).resolve()
    if primary.exists():
        return primary
    return Path(fallback)


# ── CLI argument parsing ────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="agent_llm._runner",
        description="Supervisor-routed parallel multi-agent runner (Redis-backed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_supervisor_parallel.py "Write a hello world module"
  python scripts/run_supervisor_parallel.py --verbose "Refactor the notes/ directory"
  python scripts/run_supervisor_parallel.py --quiet --transcript "Long running task"
  python scripts/run_supervisor_parallel.py --work-dir /path/to/project "..."

Output modes (mutually exclusive):
  default     Compact: per-turn headers, tool calls with moderate truncation, snippets.
  --verbose   Full tool arguments and complete agent responses, no truncation.
  --quiet     Suppress per-turn detail; only round headers and final summary.

Path resolution (--agents-registry / --tools-registry / --sessions-dir):
  If not supplied the runner looks for a .agent-llm/ sub-directory inside
  --work-dir first, then falls back to the legacy agents/ / tools/ / sessions/
  layout.  Supply explicit paths to override both.
""",
    )
    p.add_argument(
        "task",
        nargs="*",
        help="Task description passed to the first agent (architect).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full tool args and agent responses.",
    )
    p.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress per-turn detail output."
    )
    p.add_argument(
        "-t",
        "--transcript",
        action="store_true",
        help="Write full transcript to last_supervisor_run.txt (inside --work-dir).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        metavar="N",
        help="Maximum supervisor rounds (default: 10).",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (default: WARNING).",
    )
    # ── Portability flags ──────────────────────────────────────────────────
    p.add_argument(
        "--work-dir",
        default=os.environ.get("AGENT_LLM_WORK_DIR", str(Path.cwd())),
        metavar="DIR",
        help=(
            "Working directory for the run.  Agents operate on files inside this "
            "directory.  Default: CWD.  Env: AGENT_LLM_WORK_DIR."
        ),
    )
    p.add_argument(
        "--agents-registry",
        default=None,
        metavar="FILE",
        help=(
            "Path to the agents registry JSON.  "
            "Default: <work-dir>/.agent-llm/agents.json if it exists, "
            "otherwise <work-dir>/agents/registry.json."
        ),
    )
    p.add_argument(
        "--tools-registry",
        default=None,
        metavar="FILE",
        help=(
            "Path to the tools registry JSON.  "
            "Default: <work-dir>/.agent-llm/tools.json if it exists, "
            "otherwise <work-dir>/tools/registry.json."
        ),
    )
    p.add_argument(
        "--sessions-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for agent session persistence.  "
            "Default: <work-dir>/.agent-llm/sessions if it exists, "
            "otherwise <work-dir>/sessions."
        ),
    )
    p.add_argument(
        "--model",
        default="minimax/minimax-m2.5",
        metavar="MODEL",
        help=(
            "OpenRouter model string passed to OpenRouterLLMClient.  "
            "Default: minimax/minimax-m2.5."
        ),
    )
    return p.parse_args(argv)


# ── Agent turn helpers ──────────────────────────────────────────────────────


def _extract_tool_events(
    updated_messages: list[dict],
) -> list[tuple[str, dict, str, bool]]:
    """
    Walk updated_messages and return a list of (tool_name, args_dict, result, is_error)
    tuples for every tool call found in the conversation.
    """
    events = []
    for i, msg in enumerate(updated_messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue
        # Collect the corresponding "tool" role results
        results: list[str] = []
        for j in range(i + 1, len(updated_messages)):
            m = updated_messages[j]
            if m.get("role") == "tool":
                results.append(m.get("content", ""))
            elif m.get("role") == "assistant":
                break
        for k, tc in enumerate(tool_calls):
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = fn.get("name", "?") if isinstance(fn, dict) else "?"
            raw_args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            result = results[k] if k < len(results) else ""
            is_error = result.lower().startswith("error:")
            events.append((name, args, result, is_error))
    return events


def run_agent_turn(
    agent: Agent,
    session_id: str,
    store: SessionStore,
    new_message: dict,
    display: RunDisplay,
) -> str:
    """
    Load the agent's session, run one turn, persist, and emit display events.

    Returns the agent's final text response.
    """
    messages = store.load(session_id)
    if not messages:
        system_content = f"You are agent {session_id}."
        if session_id == "tool_designer":
            system_content += (
                " You can assign tools like write_file to other agents when they need"
                " them. When assigning a tool, the assignee will automatically receive"
                " a notification task so they can continue. Assign tools promptly when"
                " requested."
            )
        if session_id == "architect":
            system_content += (
                " After delegating a task to another agent, provide a brief final reply"
                " (e.g. that you have delegated the task) instead of continuing to use tools."
            )
        if session_id in ("coder", "architect"):
            system_content += (
                " When you need a tool you don't have, request it from tool_designer via send_message."
                " If you CANNOT proceed without that tool (critical need), request it and then END your turn"
                " with a short message saying you are waiting."
                " If you CAN do other useful work first (e.g. read files, list dirs), continue doing that work"
                " and request the tool when you need it; on your next turn after the assignment you will have"
                " the tool available."
            )
        system_content += (
            " Avoid calling the same tool with the same arguments twice;"
            " check the conversation for existing results."
        )
        messages.append({"role": "system", "content": system_content})

    sender = new_message.get("_from", "unknown")
    content = new_message.get("content", "")

    display.agent_turn_start(session_id, sender, content)

    messages.append({"role": "user", "content": f"[Message from {sender}]: {content}"})

    final_text, updated_messages = agent.run(messages)
    store.save(session_id, updated_messages)

    # Emit tool-call events
    for tool_name, args, result, is_error in _extract_tool_events(updated_messages):
        display.tool_call(session_id, tool_name, args, result, error=is_error)

    display.agent_turn_end(session_id, final_text or "")
    return final_text or ""


# ── Supervisor orchestration loop ───────────────────────────────────────────


def _has_any_pending_tasks(redis_state: RedisState, agent_registry: list) -> bool:
    """Return True if any agent has pending tasks in the Redis queue."""
    return any(
        redis_state.get_pending_task_count(info["agent_id"]) > 0
        for info in agent_registry
    )


def _run_supervisor(args: argparse.Namespace, display: RunDisplay) -> None:
    # ── Working directory ────────────────────────────────────────────────────
    work_dir = Path(args.work_dir).resolve()

    # ── .env loading (deferred so work_dir is known) ─────────────────────────
    # Try work_dir/.env first; fall back to ~/.agent-llm/.env.
    # python-dotenv's load_dotenv does NOT override existing env vars by default,
    # so calling both is safe: the first file wins for each variable.
    try:
        from dotenv import load_dotenv  # local import — optional dependency

        _local_env = work_dir / ".env"
        _global_env = Path.home() / ".agent-llm" / ".env"
        if _local_env.is_file():
            load_dotenv(_local_env)
        elif _global_env.is_file():
            load_dotenv(_global_env)
    except ImportError:
        pass  # python-dotenv not installed; rely on environment variables

    # ── API key ──────────────────────────────────────────────────────────────
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        display.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)
    if not api_key.startswith("sk-or-"):
        display.warn(
            "OPENROUTER_API_KEY should start with 'sk-or-'. An OpenAI key here will cause 401."
        )

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url = os.environ.get("AGENT_LLM_REDIS_URL") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379"
    )
    try:
        redis_state = RedisState(redis_url)
        redis_state.client.ping()
    except Exception as e:
        display.error(f"Could not connect to Redis at {redis_url}: {e}")
        sys.exit(1)

    # ── Config path resolution ───────────────────────────────────────────────
    sessions_dir = _resolve_config_path(
        work_dir,
        args.sessions_dir,
        primary=work_dir / ".agent-llm" / "sessions",
        fallback=work_dir / "sessions",
    )
    agents_registry_path = _resolve_config_path(
        work_dir,
        args.agents_registry,
        primary=work_dir / ".agent-llm" / "agents.json",
        fallback=work_dir / "agents" / "registry.json",
    )
    tools_registry_path = _resolve_config_path(
        work_dir,
        args.tools_registry,
        primary=work_dir / ".agent-llm" / "tools.json",
        fallback=work_dir / "tools" / "registry.json",
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
    # notes_root == work_dir: agents operate directly on the project directory.
    notes_root = work_dir

    store = SessionStore(sessions_dir)
    redis_workspace = RedisWorkspace(redis_state, workspace_id="supervisor_run")

    default_tools = create_default_tools(notes_root)
    custom_tools = load_registry_tools(tools_registry_path)
    workspace_tools = create_workspace_tools(redis_workspace)
    assignable_tools = create_assignable_tools(notes_root)
    sandbox_tool = create_sandbox_tool(notes_root)

    agent_registry = load_agent_registry(agents_registry_path)
    if not agent_registry:
        display.error(f"No agents configured. Checked: {agents_registry_path}")
        sys.exit(1)

    # ── LLM client ───────────────────────────────────────────────────────────
    llm_client = OpenRouterLLMClient(api_key=api_key, model=args.model)

    # ── Build per-agent tool registries and Agent instances ──────────────────
    agent_tools: dict[str, dict] = {}
    agents: dict[str, Agent] = {}

    for agent_info in agent_registry:
        a_id = agent_info["agent_id"]
        base = {**default_tools, **custom_tools, **workspace_tools}
        delegate_tool = create_delegate_tool(
            bus=None,
            sender_agent_id=a_id,
            agent_registry=agent_registry,
            task_store=redis_state,
        )
        base[delegate_tool["name"]] = delegate_tool
        # Coder has write_file (and other assignable tools) by default;
        # does not depend on tool_designer for this.
        if a_id == "coder":
            base.update(assignable_tools)
            base[sandbox_tool["name"]] = sandbox_tool
        agent_tools[a_id] = base

    if "tool_designer" in agent_tools:
        assign_tool = create_assign_tool(
            agent_tools,
            assignable_tools,
            agent_registry,
            bus=None,
            sender_agent_id="tool_designer",
            task_store=redis_state,
        )
        agent_tools["tool_designer"][assign_tool["name"]] = assign_tool

    for a_id, tools in agent_tools.items():
        agents[a_id] = Agent(llm_client, tools, agent_id=a_id)

    # ── Display startup banner ───────────────────────────────────────────────
    display.run_start(agent_ids=[info["agent_id"] for info in agent_registry])

    # ── Seed initial task ────────────────────────────────────────────────────
    user_content = " ".join(args.task).strip()
    if user_content:
        first_agent = agent_registry[0]["agent_id"]
        redis_state.push_task(
            first_agent, {"from": "user", "to": first_agent, "content": user_content}
        )

    max_cycles = args.max_rounds
    cycles = 0

    def run_agent_job(a_id: str, tasks: list) -> int:
        count = 0
        for msg in tasks:
            # Normalise key: the push_task caller uses "from", but we want "_from"
            normalised = {**msg, "_from": msg.get("from", "unknown")}
            try:
                run_agent_turn(agents[a_id], a_id, store, normalised, display)
            except Exception:
                display.agent_error(a_id, traceback.format_exc())
            count += 1
        return count

    agent_turn_count = 0
    t0 = time.monotonic()

    while _has_any_pending_tasks(redis_state, agent_registry) and cycles < max_cycles:
        # Collect tasks for this round
        agent_tasks: dict[str, list] = {}
        for agent_info in agent_registry:
            a_id = agent_info["agent_id"]
            tasks = redis_state.get_and_clear_tasks(a_id)
            if tasks:
                agent_tasks[a_id] = tasks

        if not agent_tasks:
            break  # Nothing to do

        cycles += 1
        task_counts = {a: len(t) for a, t in agent_tasks.items()}
        display.round_start(cycles, max_cycles, task_counts)

        # tool_designer must run first (serial) so tool assignments are in place
        # before other agents (e.g. coder) run in parallel.
        tool_designer_id = "tool_designer"
        if tool_designer_id in agent_tasks:
            agent_turn_count += run_agent_job(
                tool_designer_id, agent_tasks.pop(tool_designer_id)
            )

        if agent_tasks:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(agent_tasks)
            ) as executor:
                futures = {
                    executor.submit(run_agent_job, a_id, tasks): a_id
                    for a_id, tasks in agent_tasks.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        agent_turn_count += future.result()
                    except Exception:
                        a_id = futures[future]
                        display.agent_error(a_id, traceback.format_exc())

        # Show pending counts for next round
        pending = {}
        for agent_info in agent_registry:
            a_id = agent_info["agent_id"]
            n = redis_state.get_pending_task_count(a_id)
            if n:
                pending[a_id] = n
        display.round_end(cycles, pending)

    display.run_end(
        rounds=cycles, turns=agent_turn_count, elapsed=time.monotonic() - t0
    )


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args(sys.argv[1:])

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # transcript_path is relative to work_dir (resolved in _run_supervisor,
    # but work_dir isn't available yet here — resolve directly).
    work_dir = Path(args.work_dir).resolve()
    transcript_path = (
        (work_dir / "last_supervisor_run.txt") if args.transcript else None
    )

    with RunDisplay(
        verbose=args.verbose,
        quiet=args.quiet,
        transcript_path=transcript_path,
    ) as display:
        _run_supervisor(args, display)


if __name__ == "__main__":
    main()
