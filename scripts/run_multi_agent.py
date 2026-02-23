#!/usr/bin/env python3
"""
Multi-Agent runner.
Starts multiple agents using a shared message bus, a shared workspace, and persistent sessions.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
load_dotenv(_repo_root / ".env.example")
load_dotenv(_repo_root / ".env")

from agent_llm.agent import Agent
from agent_llm.llm import OpenRouterLLMClient
from agent_llm.tools import create_default_tools, load_registry_tools, create_assignable_tools

from agent_llm.session import SessionStore
from agent_llm.bus import MessageBus
from agent_llm.workspace import Workspace, create_workspace_tools
from agent_llm.agents import load_agent_registry, create_delegate_tool, create_assign_tool

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_flags(argv: list[str]) -> tuple[list[str], bool, bool, bool]:
    """Remove --quiet/-q, --transcript/-t, --verbose/-v from argv; return (remaining_args, quiet, transcript, verbose)."""
    args = list(argv)
    quiet = False
    transcript = False
    verbose = False
    if "--quiet" in args:
        args.remove("--quiet")
        quiet = True
    if "-q" in args:
        args.remove("-q")
        quiet = True
    if "--transcript" in args:
        args.remove("--transcript")
        transcript = True
    if "-t" in args:
        args.remove("-t")
        transcript = True
    if "--verbose" in args:
        args.remove("--verbose")
        verbose = True
    if "-v" in args:
        args.remove("-v")
        verbose = True
    return (args, quiet, transcript, verbose)


def _short_arg(val: str | int | bool, max_len: int = 28) -> str:
    """Short display for a single argument value."""
    s = str(val)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."

def _tool_summary_lines(updated_messages: list[dict]) -> list[str]:
    """From updated_messages, extract all assistant tool calls and their results; return formatted lines."""
    lines = []
    result_max = 52
    for i, msg in enumerate(updated_messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue
        tool_results: list[str] = []
        for j in range(i + 1, len(updated_messages)):
            m = updated_messages[j]
            if m.get("role") == "tool":
                tool_results.append(m.get("content", ""))
            elif m.get("role") == "assistant":
                break
        for k, tc in enumerate(tool_calls):
            func = tc.get("function") if isinstance(tc, dict) else None
            name = func.get("name", "?") if isinstance(func, dict) else "?"
            raw_args = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
            try:
                inp = json.loads(raw_args) if isinstance(raw_args, str) else {}
            except (json.JSONDecodeError, TypeError):
                inp = {}
            # Abbreviate args: for send_message highlight to_agent; keep others short
            parts = []
            for key, val in inp.items():
                if name == "send_message" and key == "content":
                    parts.append(f"content={_short_arg(val, 20)}")
                else:
                    parts.append(f"{key}={_short_arg(val)}")
            args_str = ", ".join(parts)
            result = tool_results[k] if k < len(tool_results) else ""
            result = result.replace("\n", " ").strip()
            if len(result) > result_max:
                result = result[: result_max - 3].rstrip() + "..."
            lines.append(f"  tools: {name}({args_str}) → {result}")
    return lines


def run_agent_turn(
    agent: Agent,
    session_id: str,
    store: SessionStore,
    new_message: dict | None = None,
    out: Callable[[str], None] = print,
    transcript_out: Callable[[str], None] | None = None,
    verbose: bool = False,
) -> None:
    """Load session, optionally append a message, run agent, save result."""
    messages = store.load(session_id)

    # If this is a fresh session, add system prompt or description
    if not messages:
        system_content = f"You are agent {session_id}."
        if session_id == "tool_designer":
            system_content += " You can assign tools like write_file to other agents when they need them."
        messages.append({"role": "system", "content": system_content})

    if new_message:
        messages.append(new_message)

    out(f"\n--- {session_id} ---")
    final_text, updated_messages = agent.run(messages)

    store.save(session_id, updated_messages)
    for line in _tool_summary_lines(updated_messages):
        out(line)
    snippet = (final_text or "").replace("\n", " ").strip()[:80]
    if len((final_text or "").replace("\n", " ").strip()) > 80:
        snippet += "..."
    out(f"  snippet: {snippet}")

    full_response = (final_text or "").strip()
    if full_response:
        if verbose:
            out(full_response)
        elif transcript_out is not None:
            transcript_out("  response:")
            transcript_out(full_response)

def main() -> None:
    argv_rest, quiet, transcript, verbose = _parse_flags(sys.argv[1:])
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    transcript_file = None
    if transcript:
        transcript_path = _repo_root / "last_run.txt"
        transcript_file = open(transcript_path, "w", encoding="utf-8")

    def out(s: str) -> None:
        print(s)
        if transcript_file:
            transcript_file.write(s + "\n")
            transcript_file.flush()

    def transcript_out(s: str) -> None:
        if transcript_file:
            transcript_file.write(s + "\n")
            transcript_file.flush()

    try:
        _run(argv_rest, out, transcript_out if transcript else None, verbose)
    finally:
        if transcript_file:
            transcript_file.close()


def _run(
    argv_rest: list[str],
    out: Callable[[str], None],
    transcript_out: Callable[[str], None] | None = None,
    verbose: bool = False,
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    # Initialize shared components
    bus = MessageBus()
    store = SessionStore(_repo_root / "sessions")
    workspace = Workspace(_repo_root / "workspace.json")
    
    # Tools
    notes_root = _repo_root / "notes"
    default_tools = create_default_tools(notes_root)
    custom_tools = load_registry_tools(_repo_root / "tools" / "registry.json")
    workspace_tools = create_workspace_tools(workspace)
    assignable_toolbox = create_assignable_tools(notes_root)
    
    # Agent Registry
    agent_registry = load_agent_registry(_repo_root / "agents" / "registry.json")
    if not agent_registry:
        out("No agents configured in agents/registry.json. Exiting.")
        sys.exit(1)

    llm_client = OpenRouterLLMClient(api_key=api_key)

    # Build agent instances
    agents = {}
    agent_tools = {}
    
    for agent_info in agent_registry:
        a_id = agent_info["agent_id"]
        # Everyone gets default + custom + workspace + delegate tool
        base = {**default_tools, **custom_tools, **workspace_tools}
        delegate_tool = create_delegate_tool(bus, a_id, agent_registry)
        base[delegate_tool["name"]] = delegate_tool
        agent_tools[a_id] = base
        
    if "tool_designer" in agent_tools:
        assign_tool = create_assign_tool(
            agent_tools, assignable_toolbox, agent_registry, bus=bus, sender_agent_id="tool_designer"
        )
        agent_tools["tool_designer"][assign_tool["name"]] = assign_tool
        
    for a_id, tools in agent_tools.items():
        agents[a_id] = Agent(llm_client, tools)

    # If there's a CLI arg, inject it to the first agent (architect by default)
    user_content = " ".join(argv_rest).strip()
    if user_content:
        first_agent = agent_registry[0]["agent_id"]
        bus.post({"from": "user", "to": first_agent, "content": user_content})

    # Simple Orchestrator Loop
    max_turns = 10
    turns = 0
    agent_turn_count = 0
    active = True

    while active and turns < max_turns:
        active = False
        turns += 1
        out(f"\n========== Loop Iteration {turns} ==========")

        for agent_info in agent_registry:
            a_id = agent_info["agent_id"]
            inbox = bus.get_inbox(a_id)
            if inbox:
                active = True
                for msg in inbox:
                    sender = msg.get("from", "unknown")
                    content = msg.get("content", "")
                    user_msg = {
                        "role": "user",
                        "content": f"[Message from {sender}]: {content}"
                    }
                    run_agent_turn(agents[a_id], a_id, store, user_msg, out=out, transcript_out=transcript_out, verbose=verbose)
                    agent_turn_count += 1

    out("\nNo more messages on the bus.")
    out(f"Run finished: {turns} iterations, {agent_turn_count} agent turns.")

if __name__ == "__main__":
    main()
