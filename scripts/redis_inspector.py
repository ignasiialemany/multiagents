#!/usr/bin/env python3
"""
Redis Inspector - Development utility for querying Redis state.

This script provides commands to inspect the Redis-backed state of the
agent_llm system for debugging and monitoring purposes.

Usage:
    python scripts/redis_inspector.py keys                    # List all agent_llm:* keys
    python scripts/redis_inspector.py tasks                   # Show pending tasks per agent
    python scripts/redis_inspector.py tasks <agent_id>        # Show tasks for specific agent
    python scripts/redis_inspector.py workspace               # Dump workspace hash
    python scripts/redis_inspector.py stats                   # Quick system stats
    python scripts/redis_inspector.py dump                    # Full state dump
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))
load_dotenv(_repo_root / ".env.example")
load_dotenv(_repo_root / ".env")

import redis


def get_redis_client() -> redis.Redis:
    """Create Redis client using environment config."""
    redis_url = os.environ.get("AGENT_LLM_REDIS_URL") or os.environ.get(
        "REDIS_URL", "redis://localhost:6379"
    )
    try:
        return redis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        print(f"Error connecting to Redis at {redis_url}: {e}")
        sys.exit(1)


def cmd_keys(client: redis.Redis, args: argparse.Namespace) -> None:
    """List all keys matching a pattern."""
    pattern = args.pattern or "agent_llm:*"
    keys = sorted(client.keys(pattern))
    if not keys:
        print(f"No keys found matching pattern: {pattern}")
        return
    print(f"Keys matching '{pattern}' ({len(keys)} total):")
    for key in keys:
        key_type = client.type(key)
        if key_type == "list":
            count = client.llen(key)
            print(f"  {key} (list, {count} items)")
        elif key_type == "hash":
            count = client.hlen(key)
            print(f"  {key} (hash, {count} fields)")
        elif key_type == "string":
            print(f"  {key} (string)")
        else:
            print(f"  {key} ({key_type})")


def cmd_tasks(client: redis.Redis, args: argparse.Namespace) -> None:
    """Show pending tasks for agents."""
    agent_id = args.agent_id

    if agent_id:
        # Show tasks for specific agent
        key = f"agent_llm:tasks:{agent_id}"
        tasks = client.lrange(key, 0, -1)
        if not tasks:
            print(f"No pending tasks for agent: {agent_id}")
            return
        print(f"Pending tasks for '{agent_id}' ({len(tasks)} total):")
        for i, task in enumerate(tasks):
            try:
                task_data = json.loads(task)
                print(f"  [{i}] {json.dumps(task_data)}")
            except json.JSONDecodeError:
                print(f"  [{i}] {task}")
    else:
        # Show all task queues
        pattern = "agent_llm:tasks:*"
        keys = sorted(client.keys(pattern))
        if not keys:
            print("No task queues found.")
            return

        print(f"Task queues ({len(keys)} total):")
        for key in keys:
            agent = key.replace("agent_llm:tasks:", "")
            count = client.llen(key)
            print(f"  {agent}: {count} pending task(s)")


def cmd_workspace(client: redis.Redis, args: argparse.Namespace) -> None:
    """Dump workspace hash contents."""
    workspace_id = args.workspace_id or "default"
    key = f"agent_llm:workspace:{workspace_id}"

    data = client.hgetall(key)
    if not data:
        print(f"Workspace '{workspace_id}' is empty.")
        return

    print(f"Workspace '{workspace_id}' ({len(data)} keys):")
    for field, value in sorted(data.items()):
        try:
            parsed = json.loads(value)
            print(f"  {field}:")
            print(f"    {json.dumps(parsed, indent=4)}")
        except json.JSONDecodeError:
            print(f"  {field}: {value}")


def cmd_stats(client: redis.Redis, args: argparse.Namespace) -> None:
    """Show quick system statistics."""
    # Task queue stats
    task_keys = client.keys("agent_llm:tasks:*")
    total_tasks = sum(client.llen(k) for k in task_keys)

    # Workspace stats
    workspace_keys = client.keys("agent_llm:workspace:*")
    total_workspace_fields = sum(client.hlen(k) for k in workspace_keys)

    # All agent_llm keys
    all_keys = client.keys("agent_llm:*")

    print("=== Redis State Statistics ===")
    print(f"Total 'agent_llm:*' keys: {len(all_keys)}")
    print(f"  - Task queues: {len(task_keys)}")
    print(f"  - Total pending tasks: {total_tasks}")
    print(f"  - Workspaces: {len(workspace_keys)}")
    print(f"  - Total workspace fields: {total_workspace_fields}")

    if task_keys:
        print("\nTasks per agent:")
        for key in sorted(task_keys):
            agent = key.replace("agent_llm:tasks:", "")
            count = client.llen(key)
            print(f"  {agent}: {count}")


def cmd_dump(client: redis.Redis, args: argparse.Namespace) -> None:
    """Full dump of all agent_llm state."""
    print("=== Full Redis State Dump ===\n")

    # All keys
    all_keys = sorted(client.keys("agent_llm:*"))
    print(f"Total keys: {len(all_keys)}\n")

    for key in all_keys:
        key_type = client.type(key)
        print(f"--- {key} ({key_type}) ---")

        if key_type == "list":
            items = client.lrange(key, 0, -1)
            for i, item in enumerate(items):
                try:
                    parsed = json.loads(item)
                    print(f"  [{i}] {json.dumps(parsed, indent=4)}")
                except json.JSONDecodeError:
                    print(f"  [{i}] {item}")
        elif key_type == "hash":
            data = client.hgetall(key)
            for field, value in sorted(data.items()):
                try:
                    parsed = json.loads(value)
                    print(f"  {field}: {json.dumps(parsed, indent=4)}")
                except json.JSONDecodeError:
                    print(f"  {field}: {value}")
        elif key_type == "string":
            value = client.get(key)
            print(f"  {value}")
        else:
            print(f"  (unhandled type)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redis Inspector - Query Redis state for agent_llm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # keys command
    keys_parser = subparsers.add_parser("keys", help="List keys matching pattern")
    keys_parser.add_argument(
        "-p", "--pattern", help="Key pattern (default: agent_llm:*)", default=None
    )

    # tasks command
    tasks_parser = subparsers.add_parser("tasks", help="Show pending tasks")
    tasks_parser.add_argument(
        "agent_id", nargs="?", help="Specific agent ID (optional)"
    )

    # workspace command
    workspace_parser = subparsers.add_parser("workspace", help="Dump workspace hash")
    workspace_parser.add_argument(
        "workspace_id", nargs="?", help="Workspace ID (default: default)", default=None
    )

    # stats command
    subparsers.add_parser("stats", help="Quick system statistics")

    # dump command
    subparsers.add_parser("dump", help="Full state dump")

    args = parser.parse_args()

    client = get_redis_client()

    # Route to command handler
    if args.command == "keys":
        cmd_keys(client, args)
    elif args.command == "tasks":
        cmd_tasks(client, args)
    elif args.command == "workspace":
        cmd_workspace(client, args)
    elif args.command == "stats":
        cmd_stats(client, args)
    elif args.command == "dump":
        cmd_dump(client, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
