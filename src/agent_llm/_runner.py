#!/usr/bin/env python3
"""
Supervisor-routed parallel runner — importable module.

Supports two modes:

1. **Batch mode** (existing): pass a task string; agents loop via Redis.
2. **Interactive mode** (new): REPL with memory stream, retrieval, reflection,
   planning, and on-demand subagent spawning (generative agent architecture).

Usage (via shim)
----------------
    python scripts/run_supervisor_parallel.py "Build a hello world module"
    python scripts/run_supervisor_parallel.py --verbose "..."
    python scripts/run_supervisor_parallel.py --quiet   "..."
    python scripts/run_supervisor_parallel.py --transcript "..."
    python scripts/run_supervisor_parallel.py --work-dir /path/to/project "..."
    python scripts/run_supervisor_parallel.py --help

Interactive mode
----------------
    agent-llm -i
    agent-llm                    # (no task args → interactive by default)
    python -m agent_llm._runner --interactive
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

from agent_llm.agent import Agent
from agent_llm.llm import OpenRouterLLMClient
from agent_llm.tools import (
    create_default_tools,
    load_registry_tools,
    create_assignable_tools,
    create_spawn_tool,
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
from agent_llm.memory import MemoryStream
from agent_llm.reflection import maybe_reflect, make_plan

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
  agent-llm -i                                          # interactive REPL
  agent-llm "Write a hello world module"                # batch mode
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
        "-i",
        "--interactive",
        action="store_true",
        help="Start in interactive REPL mode (generative agent).",
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
    p.add_argument(
        "--reflection-threshold",
        type=int,
        default=50,
        metavar="N",
        help="Cumulative importance before auto-reflection triggers (default: 50).",
    )
    return p.parse_args(argv)


# ── Common infrastructure setup ─────────────────────────────────────────────


def _setup_infrastructure(args: argparse.Namespace, display: RunDisplay):
    """
    Shared bootstrap for both batch and interactive modes.

    Returns a dict with all initialised components.
    """
    work_dir = Path(args.work_dir).resolve()

    try:
        from dotenv import load_dotenv

        _local_env = work_dir / ".env"
        _global_env = Path.home() / ".agent-llm" / ".env"
        if _local_env.is_file():
            load_dotenv(_local_env)
        elif _global_env.is_file():
            load_dotenv(_global_env)
    except ImportError:
        pass

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        display.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)
    if not api_key.startswith("sk-or-"):
        display.warn(
            "OPENROUTER_API_KEY should start with 'sk-or-'. An OpenAI key here will cause 401."
        )

    redis_url = os.environ.get("AGENT_LLM_REDIS_URL") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379"
    )
    try:
        redis_state = RedisState(redis_url)
        redis_state.client.ping()
    except Exception as e:
        display.error(f"Could not connect to Redis at {redis_url}: {e}")
        sys.exit(1)

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

    llm_client = OpenRouterLLMClient(api_key=api_key, model=args.model)

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

    return {
        "work_dir": work_dir,
        "store": store,
        "redis_state": redis_state,
        "redis_workspace": redis_workspace,
        "agent_registry": agent_registry,
        "agents": agents,
        "agent_tools": agent_tools,
        "llm_client": llm_client,
        "default_tools": default_tools,
        "custom_tools": custom_tools,
        "workspace_tools": workspace_tools,
        "assignable_tools": assignable_tools,
        "sandbox_tool": sandbox_tool,
        "sessions_dir": sessions_dir,
    }


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

    for tool_name, args, result, is_error in _extract_tool_events(updated_messages):
        display.tool_call(session_id, tool_name, args, result, error=is_error)

    display.agent_turn_end(session_id, final_text or "")
    return final_text or ""


# ── Supervisor orchestration loop (batch mode) ──────────────────────────────


def _has_any_pending_tasks(redis_state: RedisState, agent_registry: list) -> bool:
    """Return True if any agent has pending tasks in the Redis queue."""
    return any(
        redis_state.get_pending_task_count(info["agent_id"]) > 0
        for info in agent_registry
    )


def _run_supervisor(args: argparse.Namespace, display: RunDisplay) -> None:
    infra = _setup_infrastructure(args, display)
    redis_state = infra["redis_state"]
    agent_registry = infra["agent_registry"]
    agents = infra["agents"]
    store = infra["store"]

    display.run_start(agent_ids=[info["agent_id"] for info in agent_registry])

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
        agent_tasks: dict[str, list] = {}
        for agent_info in agent_registry:
            a_id = agent_info["agent_id"]
            tasks = redis_state.get_and_clear_tasks(a_id)
            if tasks:
                agent_tasks[a_id] = tasks

        if not agent_tasks:
            break

        cycles += 1
        task_counts = {a: len(t) for a, t in agent_tasks.items()}
        display.round_start(cycles, max_cycles, task_counts)

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


# ── Interactive REPL mode ────────────────────────────────────────────────────

_INTERACTIVE_SYSTEM_PROMPT = """\
You are an interactive generative agent. You perceive user messages, retrieve \
relevant memories from your experience, and respond thoughtfully.

You have access to tools for reading files, searching, and spawning subagents \
to handle specialised tasks (coding, reviewing, etc.).

When the user asks you to do something complex, break it down and use \
spawn_subagent to delegate parts to specialised agents.

Be concise but helpful. You remember previous conversations through your \
memory stream.\
"""


def _build_context_messages(
    system_prompt: str,
    relevant_memories: list,
    recent_conversation: list[dict],
) -> list[dict]:
    """
    Build the message list for the interactive agent's LLM call.
    Injects relevant retrieved memories into the system context.
    """
    memory_block = ""
    if relevant_memories:
        lines = []
        for m in relevant_memories:
            lines.append(f"[{m.kind}, importance={m.importance}] {m.content}")
        memory_block = (
            "\n\nRelevant memories from your experience:\n"
            + "\n".join(lines)
        )

    messages = [
        {"role": "system", "content": system_prompt + memory_block},
    ]
    messages.extend(recent_conversation)
    return messages


def _handle_slash_command(
    cmd: str,
    memory: MemoryStream,
    llm_client: OpenRouterLLMClient,
    display: RunDisplay,
    agent_registry: list[dict],
    agents: dict[str, Agent],
    store: SessionStore,
    reflection_threshold: int,
) -> bool:
    """
    Handle a slash command. Returns True if the command was recognised.
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit"):
        raise _QuitInteractive()

    if command == "/help":
        display.slash_help()
        return True

    if command == "/agents":
        for a in agent_registry:
            display.command_output(f"  {a['agent_id']:>14s}  {a['description']}")
        return True

    if command == "/memory":
        if arg:
            results = memory.retrieve(arg, k=10)
        else:
            results = memory.get_recent(15)
        display.memory_display([e.to_dict() for e in results])
        return True

    if command == "/reflect":
        display.command_output("Running reflection cycle...")
        reflections = maybe_reflect(memory, llm_client, threshold=0)
        if reflections:
            for r in reflections:
                display.command_output(f"  reflection: {r}")
        else:
            display.command_output("  (no reflections produced)")
        return True

    if command == "/plan":
        if not arg:
            display.command_output("Usage: /plan <goal description>")
            return True
        display.command_output("Creating plan...")
        plan = make_plan(memory, llm_client, arg)
        display.command_output(plan)
        return True

    if command == "/spawn":
        spawn_parts = arg.split(maxsplit=1)
        if len(spawn_parts) < 2:
            display.command_output("Usage: /spawn <agent_id> <task>")
            return True
        agent_id, task = spawn_parts
        known = [a["agent_id"] for a in agent_registry]
        if agent_id not in known:
            display.command_output(
                f"Unknown agent '{agent_id}'. Available: {', '.join(known)}"
            )
            return True
        display.command_output(f"Spawning {agent_id}...")
        agent = agents.get(agent_id)
        if agent is None:
            display.command_output(f"Agent '{agent_id}' not initialised.")
            return True
        result = _run_subagent_sync(agent, agent_id, store, task, display)
        memory.add(
            f"Subagent {agent_id} result: {result}",
            kind="observation",
            importance=llm_client.score_importance(result),
        )
        display.command_output(f"[{agent_id} result]: {result}")
        return True

    if command == "/clear":
        display.command_output("Conversation cleared (memory stream preserved).")
        return True  # caller checks for /clear separately

    if command == "/save":
        display.command_output("Session saved.")
        return True

    display.command_output(f"Unknown command: {command}. Type /help for usage.")
    return True


class _QuitInteractive(Exception):
    """Raised to break out of the interactive loop."""


def _run_subagent_sync(
    agent: Agent,
    agent_id: str,
    store: SessionStore,
    task: str,
    display: RunDisplay,
) -> str:
    """Run a single subagent turn synchronously and return its response."""
    messages = store.load(agent_id)
    if not messages:
        messages.append({"role": "system", "content": f"You are agent {agent_id}."})

    display.agent_turn_start(agent_id, "interactive_agent", task)
    messages.append({
        "role": "user",
        "content": f"[Task from interactive agent]: {task}",
    })

    try:
        final_text, updated = agent.run(messages)
    except Exception as exc:
        display.agent_error(agent_id, str(exc))
        return f"(subagent error: {exc})"

    store.save(agent_id, updated)

    for tool_name, args, result, is_error in _extract_tool_events(updated):
        display.tool_call(agent_id, tool_name, args, result, error=is_error)

    display.agent_turn_end(agent_id, final_text or "")
    return final_text or "(no response)"


def _run_interactive(args: argparse.Namespace, display: RunDisplay) -> None:
    """Interactive REPL with generative agent architecture."""
    infra = _setup_infrastructure(args, display)
    agents = infra["agents"]
    store = infra["store"]
    agent_registry = infra["agent_registry"]
    llm_client = infra["llm_client"]
    sessions_dir = infra["sessions_dir"]

    # Set up the interactive agent's memory stream
    memory_path = Path(sessions_dir) / "interactive_agent.memory.jsonl"
    embedder = llm_client  # OpenRouterLLMClient implements embed()
    memory = MemoryStream(persist_path=memory_path, embedder=embedder)

    # Build interactive agent's tool set: default tools + spawn_subagent
    interactive_tools = {
        **infra["default_tools"],
        **infra["custom_tools"],
        **infra["workspace_tools"],
        **infra["assignable_tools"],
    }
    spawn_tool = create_spawn_tool(agents, store, display, agent_registry)
    interactive_tools[spawn_tool["name"]] = spawn_tool
    interactive_tools[infra["sandbox_tool"]["name"]] = infra["sandbox_tool"]

    interactive_agent = Agent(llm_client, interactive_tools, agent_id="interactive")

    display.interactive_banner(
        agent_ids=[a["agent_id"] for a in agent_registry],
    )

    conversation: list[dict] = []
    reflection_threshold = args.reflection_threshold

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                is_clear = user_input.strip().lower() == "/clear"
                try:
                    _handle_slash_command(
                        user_input,
                        memory,
                        llm_client,
                        display,
                        agent_registry,
                        agents,
                        store,
                        reflection_threshold,
                    )
                except _QuitInteractive:
                    break
                if is_clear:
                    conversation.clear()
                continue

            # 1. Perceive — store user message in memory stream
            importance = llm_client.score_importance(user_input)
            memory.add(user_input, kind="observation", importance=importance)

            # 2. Retrieve — find relevant memories
            relevant = memory.retrieve(user_input, k=10)

            # 3. Build context and respond
            conversation.append({"role": "user", "content": user_input})
            context = _build_context_messages(
                _INTERACTIVE_SYSTEM_PROMPT, relevant, conversation,
            )
            response_text, updated = interactive_agent.run(context)

            # Keep only the new messages added during this turn
            # (context already has the full history; we track raw conversation separately)
            if response_text:
                conversation.append({"role": "assistant", "content": response_text})

            # Emit tool events from this turn
            for tool_name, t_args, result, is_error in _extract_tool_events(updated):
                display.tool_call("interactive", tool_name, t_args, result, error=is_error)

            # 4. Store response in memory
            if response_text:
                resp_importance = llm_client.score_importance(response_text)
                memory.add(response_text, kind="observation", importance=resp_importance)

            # 5. Maybe reflect
            reflections = maybe_reflect(memory, llm_client, threshold=reflection_threshold)
            if reflections:
                for r in reflections:
                    display.command_output(f"  [reflection] {r}")

            # 6. Display response
            display.agent_response(response_text or "(no response)")

            # Persist conversation to session store
            store.save("interactive_agent", conversation)

    except KeyboardInterrupt:
        pass

    display.command_output("\nGoodbye.")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args(sys.argv[1:])

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    work_dir = Path(args.work_dir).resolve()
    transcript_path = (
        (work_dir / "last_supervisor_run.txt") if args.transcript else None
    )

    # Default to interactive mode when no task is provided
    is_interactive = args.interactive or not args.task

    with RunDisplay(
        verbose=args.verbose,
        quiet=args.quiet,
        transcript_path=transcript_path,
    ) as display:
        if is_interactive:
            _run_interactive(args, display)
        else:
            _run_supervisor(args, display)


if __name__ == "__main__":
    main()
