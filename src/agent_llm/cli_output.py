"""
Rich-based CLI output layer for the supervisor/multi-agent runner.

Provides a thread-safe, colored, timestamped event stream suitable for
parallel agent execution.  Callers emit structured events; this module
handles all formatting and printing.

Usage
-----
    from agent_llm.cli_output import RunDisplay

    display = RunDisplay(verbose=False, transcript_path=Path("last_run.txt"))
    with display:
        display.run_start(agent_ids=["architect", "coder", "tool_designer"])
        display.round_start(round_num=1, max_rounds=10)
        display.agent_message(agent="architect", direction="in",  peer="user",    content="Build hello world")
        display.tool_call(agent="architect", tool="send_message",  args={"to": "coder", "content": "Write hello_world.py"}, result="Message sent.")
        display.agent_message(agent="architect", direction="out", peer="coder",   content="I have delegated...")
        display.round_end(round_num=1, pending_counts={"coder": 1})
        display.run_end(rounds=1, turns=2, elapsed=12.4)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

# ── Colour palette ──────────────────────────────────────────────────────────
# Each agent gets a stable colour by cycling through this list.
_AGENT_COLOURS = [
    "cyan",
    "magenta",
    "green",
    "yellow",
    "blue",
    "bright_red",
    "bright_cyan",
    "bright_magenta",
]

_TOOL_COLOUR = "bright_black"  # muted — tools are detail, not headline
_IN_COLOUR = "dim"  # incoming message prefix
_OUT_COLOUR = "italic dim"  # outgoing (snippet/response)

_THEME = Theme(
    {
        "round": "bold white on dark_blue",
        "header": "bold bright_white",
        "ts": "bright_black",
        "ok": "green",
        "warn": "yellow",
        "error": "bold red",
        "tool": _TOOL_COLOUR,
        "in": _IN_COLOUR,
        "out": _OUT_COLOUR,
    }
)


class RunDisplay:
    """
    Thread-safe Rich console wrapper for a supervisor run.

    All public methods are safe to call from multiple threads simultaneously.
    Output is serialised through a single ``rich.Console`` instance (which
    holds its own internal lock).

    Parameters
    ----------
    verbose:
        If True, print full tool arguments and full agent responses to the
        terminal.  If False, truncate long values.
    transcript_path:
        If given, every line is also written to this file.
    quiet:
        If True, suppress per-tool and per-turn detail; only show round
        headers and the final summary.
    """

    def __init__(
        self,
        verbose: bool = False,
        transcript_path: Path | None = None,
        quiet: bool = False,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self._transcript_path = transcript_path
        self._transcript_file = None
        self._lock = threading.Lock()

        # Main console (stdout, with markup & colours)
        self._console = Console(theme=_THEME, highlight=False)

        # Plain console for transcript (no markup, no colour codes)
        self._tc: Console | None = None

        self._agent_colour: dict[str, str] = {}
        self._colour_idx = 0
        self._start_time = time.monotonic()

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "RunDisplay":
        if self._transcript_path:
            self._transcript_file = open(self._transcript_path, "w", encoding="utf-8")
            self._tc = Console(
                file=self._transcript_file, no_color=True, highlight=False
            )
        return self

    def __exit__(self, *_: Any) -> None:
        if self._transcript_file:
            self._transcript_file.close()
            self._transcript_file = None
            self._tc = None

    # ── internal helpers ─────────────────────────────────────────────────────

    def _elapsed(self) -> str:
        secs = int(time.monotonic() - self._start_time)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _agent_col(self, agent_id: str) -> str:
        if agent_id not in self._agent_colour:
            self._agent_colour[agent_id] = _AGENT_COLOURS[
                self._colour_idx % len(_AGENT_COLOURS)
            ]
            self._colour_idx += 1
        return self._agent_colour[agent_id]

    def _agent_tag(self, agent_id: str) -> str:
        col = self._agent_col(agent_id)
        return f"[{col}]{escape(agent_id):>14s}[/{col}]"

    def _print(self, markup: str, plain: str | None = None) -> None:
        """Print to console (markup) and transcript (plain text)."""
        self._console.print(markup, highlight=False)
        if self._tc:
            self._tc.print(plain if plain is not None else markup, highlight=False)

    @staticmethod
    def _trunc(s: str, n: int) -> str:
        s = s.replace("\n", " ").strip()
        if len(s) <= n:
            return s
        return s[: n - 3].rstrip() + "..."

    # ── public API ───────────────────────────────────────────────────────────

    def run_start(self, agent_ids: list[str]) -> None:
        """Print a startup banner listing all agents."""
        # Pre-assign colours so they're stable
        for a in agent_ids:
            self._agent_col(a)

        agent_list = "  ".join(
            f"[{self._agent_col(a)}]{escape(a)}[/{self._agent_col(a)}]"
            for a in agent_ids
        )
        plain_list = "  ".join(agent_ids)
        self._print(
            f"\n[header]Supervisor run[/header]  [ts]{self._ts()}[/ts]\n"
            f"Agents: {agent_list}\n",
            plain=f"\nSupervisor run  {self._ts()}\nAgents: {plain_list}\n",
        )

    def round_start(
        self, round_num: int, max_rounds: int, task_counts: dict[str, int] | None = None
    ) -> None:
        """Print a round header."""
        tasks_str = ""
        plain_tasks = ""
        if task_counts:
            parts = [f"{a}:{n}" for a, n in task_counts.items() if n]
            if parts:
                tasks_str = "  tasks: " + ", ".join(parts)
                plain_tasks = tasks_str
        line = (
            f"[round] Round {round_num}/{max_rounds} [/round]"
            f"[ts]  {self._elapsed()} elapsed[/ts]{escape(tasks_str)}"
        )
        plain = f"=== Round {round_num}/{max_rounds}  {self._elapsed()} elapsed{plain_tasks} ==="
        self._print(line, plain=plain)

    def round_end(
        self, round_num: int, pending_counts: dict[str, int] | None = None
    ) -> None:
        """Optionally print pending task info after a round."""
        if not pending_counts:
            return
        for agent_id, n in pending_counts.items():
            if n:
                tag = self._agent_tag(agent_id)
                self._print(
                    f"  {tag} [ts]→ {n} task(s) pending next round[/ts]",
                    plain=f"  {agent_id:>14s} → {n} task(s) pending next round",
                )

    def agent_turn_start(self, agent_id: str, sender: str, content: str) -> None:
        """Log that an agent is starting to process a message."""
        if self.quiet:
            return
        tag = self._agent_tag(agent_id)
        preview = escape(self._trunc(content, 80))
        self._print(
            f"  {tag} [in]← {escape(sender)}: {preview}[/in]",
            plain=f"  {agent_id:>14s} ← {sender}: {self._trunc(content, 80)}",
        )

    def tool_call(
        self,
        agent_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        error: bool = False,
    ) -> None:
        """Log a single tool call and its result."""
        if self.quiet:
            return
        tag = self._agent_tag(agent_id)
        result_col = "error" if error else "ok"

        if self.verbose:
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            result_display = result.strip()
        else:
            # Special case: show send_message content fully (it's the delegation intent)
            parts = []
            for k, v in args.items():
                val_s = str(v)
                limit = 120 if (tool_name == "send_message" and k == "content") else 40
                parts.append(f"{k}={self._trunc(val_s, limit)!r}")
            args_str = ", ".join(parts)
            result_display = self._trunc(result, 80)

        arrow = "[error]✗[/error]" if error else "[ok]✓[/ok]"

        if self.verbose and "\n" in result_display:
            self._print(
                f"  {tag} [tool]  {escape(tool_name)}({escape(args_str)}) {arrow}[/tool]",
                plain=f"  {agent_id:>14s}   {tool_name}({args_str})",
            )
            for rline in result_display.splitlines():
                self._print(
                    f"  {'':>14s}   [tool]  → {escape(rline)}[/tool]",
                    plain=f"  {'':>14s}     → {rline}",
                )
        else:
            self._print(
                f"  {tag} [tool]  {escape(tool_name)}({escape(args_str)}) {arrow} {escape(result_display)}[/tool]",
                plain=f"  {agent_id:>14s}   {tool_name}({args_str}) {'✗' if error else '✓'} {result_display}",
            )

    def agent_turn_end(self, agent_id: str, response: str) -> None:
        """Log the agent's final textual response for this turn."""
        if self.quiet:
            return
        tag = self._agent_tag(agent_id)
        if self.verbose:
            self._print(
                f"  {tag} [out]{escape(response.strip())}[/out]",
                plain=f"  {agent_id:>14s} {response.strip()}",
            )
        else:
            snippet = self._trunc(response, 100)
            self._print(
                f"  {tag} [out]↩ {escape(snippet)}[/out]",
                plain=f"  {agent_id:>14s} ↩ {snippet}",
            )

    def agent_error(self, agent_id: str, error: str) -> None:
        """Log an unhandled exception in an agent thread."""
        tag = self._agent_tag(agent_id)
        self._print(
            f"  {tag} [error]ERROR: {escape(error)}[/error]",
            plain=f"  {agent_id:>14s} ERROR: {error}",
        )

    def run_end(self, rounds: int, turns: int, elapsed: float) -> None:
        """Print final summary."""
        mm = int(elapsed) // 60
        ss = int(elapsed) % 60
        self._print(
            f"\n[ok]✓ Finished:[/ok] {rounds} round(s), {turns} agent turn(s), "
            f"[ts]{mm:02d}:{ss:02d} elapsed[/ts]\n",
            plain=f"\n✓ Finished: {rounds} round(s), {turns} agent turn(s), {mm:02d}:{ss:02d} elapsed\n",
        )

    def info(self, message: str) -> None:
        """Print a plain informational message (setup errors, warnings, etc.)."""
        self._print(
            f"[ts]{self._ts()}[/ts] {escape(message)}",
            plain=f"{self._ts()} {message}",
        )

    def warn(self, message: str) -> None:
        self._print(
            f"[warn]WARN  {escape(message)}[/warn]",
            plain=f"WARN  {message}",
        )

    def error(self, message: str) -> None:
        self._print(
            f"[error]ERROR {escape(message)}[/error]",
            plain=f"ERROR {message}",
        )

    # ── Interactive REPL display ─────────────────────────────────────────

    def agent_response(self, text: str) -> None:
        """Display the agent's conversational response."""
        self._print(
            f"[bold bright_white]agent>[/bold bright_white] {escape(text)}",
            plain=f"agent> {text}",
        )

    def memory_display(self, entries: list[dict]) -> None:
        """Display a list of memory entries in a compact table."""
        if not entries:
            self._print("[ts]  (no memories)[/ts]", plain="  (no memories)")
            return
        for e in entries:
            kind_col = {"observation": "cyan", "reflection": "magenta", "plan": "yellow"}.get(
                e.get("kind", ""), "white"
            )
            imp = e.get("importance", "?")
            kind = e.get("kind", "?")
            content = e.get("content", "")
            snippet = self._trunc(content, 100)
            self._print(
                f"  [{kind_col}]{kind:>12s}[/{kind_col}] "
                f"[ts](imp={imp})[/ts] {escape(snippet)}",
                plain=f"  {kind:>12s} (imp={imp}) {snippet}",
            )

    def command_output(self, text: str) -> None:
        """Display output of a slash command."""
        self._print(
            f"[ts]{escape(text)}[/ts]",
            plain=text,
        )

    def slash_help(self) -> None:
        """Print the help message for interactive slash commands."""
        lines = [
            "",
            "[header]Slash commands:[/header]",
            "  /memory [query]         Show recent memories or search by query",
            "  /reflect                Force a reflection cycle",
            "  /plan <goal>            Create a plan for the given goal",
            "  /meeting <topic>        Run a meeting: subagents discuss topic and produce a plan",
            "  /spawn <agent> <task>   Delegate a task to a subagent",
            "  /agents                 List available subagents",
            "  /clear                  Clear conversation context (memory persists)",
            "  /save                   Save session now",
            "  /help                   Show this help message",
            "  /quit or /exit          Exit interactive mode",
            "",
        ]
        plain_lines = [line.replace("[header]", "").replace("[/header]", "") for line in lines]
        self._print("\n".join(lines), plain="\n".join(plain_lines))

    def interactive_banner(self, agent_ids: list[str]) -> None:
        """Print the welcome banner for interactive mode."""
        agent_list = ", ".join(
            f"[{self._agent_col(a)}]{escape(a)}[/{self._agent_col(a)}]"
            for a in agent_ids
        )
        plain_list = ", ".join(agent_ids)
        self._print(
            f"\n[header]Interactive agent[/header]  [ts]{self._ts()}[/ts]\n"
            f"Available subagents: {agent_list}\n"
            f"Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.\n",
            plain=(
                f"\nInteractive agent  {self._ts()}\n"
                f"Available subagents: {plain_list}\n"
                f"Type /help for commands, /quit to exit.\n"
            ),
        )

    # ── Meeting display ─────────────────────────────────────────────────────

    def meeting_start(self, topic: str, agent_ids: list[str]) -> None:
        """Print meeting start banner."""
        agents_str = ", ".join(agent_ids)
        self._print(
            f"\n[header]Meeting[/header]  [ts]{self._ts()}[/ts]\n"
            f"Topic: {escape(topic)}\n"
            f"Participants: {agents_str}\n",
            plain=f"\nMeeting  {self._ts()}\nTopic: {topic}\nParticipants: {agents_str}\n",
        )

    def meeting_prep(self, agent_id: str, content: str) -> None:
        """Print one agent's preparation (questions/points) before the meeting."""
        tag = self._agent_tag(agent_id)
        snippet = self._trunc(content, 200)
        self._print(
            f"  {tag} [in](prep)[/in] {escape(snippet)}",
            plain=f"  {agent_id:>14s} (prep) {snippet}",
        )

    def meeting_turn(self, agent_id: str, content: str) -> None:
        """Print one agent's contribution in the meeting."""
        tag = self._agent_tag(agent_id)
        snippet = self._trunc(content, 200)
        self._print(
            f"  {tag} [out]{escape(snippet)}[/out]",
            plain=f"  {agent_id:>14s} {snippet}",
        )

    def meeting_end(self, summary: str) -> None:
        """Print meeting summary/plan."""
        self._print(
            f"[header]Meeting summary[/header]\n[out]{escape(summary)}[/out]\n",
            plain=f"Meeting summary\n{summary}\n",
        )
