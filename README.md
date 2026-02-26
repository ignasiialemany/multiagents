# agent-llm

A Redis-backed parallel multi-agent runner with supervisor orchestration. Agents can read/write files, run sandboxed code, and delegate to sub-agents.

## Interactive vs Batch Mode

`agent-llm` supports two operational modes:
- **Interactive Mode**: An interactive REPL where you converse with a single generative agent that maintains a memory stream, reflects, plans, and can spawn subagents. (Default when no task is provided). For a full description, see [docs/interactive_mode.md](docs/interactive_mode.md).
- **Batch Mode**: A one-shot task execution where the supervisor distributes work among agents until the task is complete.

## Two Modes

### Standalone (this repo)
Run agents on this repository's code and notes.

### Portable (any repo)
Run agents on **your** project — agent-llm operates on the calling project's directory, not its own. This is the recommended way to use agent-llm.

---

## Quick Start (Portable Mode)

```bash
# 1. Install agent-llm globally
pip install agent-llm

# 2. Initialize your project
cd /path/to/your/project
agent-llm init

# 3. Run the supervisor
agent-llm
# Or with explicit model:
agent-llm --model anthropic/claude-opus-4-2025-02-05
```

The `init` command creates:
- `.agent-llm/agents.json` — customize agent definitions
- `.agent-llm/tools.json` — add custom tools
- `.agent-llm/.env` — copy API keys here
- `notes/` — default sandbox for file operations

---

## Setup (Development / Standalone)

1. Create a virtual environment and install:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and set your OpenRouter API key:

   ```bash
   cp .env.example .env
   # Edit .env and set OPENROUTER_API_KEY=sk-or-...
   ```

   Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys). Do not commit `.env` or put keys in code. The app uses OpenRouter (default model: `anthropic/claude-sonnet-4-2025-02-05`; you can pick any model that supports tools on OpenRouter).

3. Run Redis (required for task queue):

   ```bash
   docker run -d -p 6379:6379 redis
   # Or on macOS: brew install redis && redis-server
   ```

---

## Usage

Run the optional unit tests (no API key needed): `pip install -e ".[dev]"` then `pytest tests/`.

### Portable mode (recommended)

```bash
agent-llm                    # Uses current directory as work-dir
agent-llm --work-dir /path/to/project
agent-llm --model anthropic/claude-opus-4-2025-02-05
agent-llm --agents-registry .my-agents/tools.json
```

### Standalone mode (this repo)

```bash
python scripts/run_supervisor_parallel.py
```

Logs will show each agent's tool calls and messages per round.

**Phase 2 – Tool proposals (human-in-the-loop):**

Tool proposals are managed directly. Implement the function in `src/agent_llm/tools_custom.py`, then add the entry to `tools/registry.json` — the tool will be available to all agents on the next run.

See `tools/README.md` for registry and log format.

## Web UI

You can interact with the agent through a minimal web interface:

```bash
# Ensure optional web dependencies are installed
pip install -e ".[web]"

# Run the web server
python -m agent_llm.serve_web
```

Then open http://127.0.0.1:5000 in your browser to chat with the interactive agent.

---

## Project layout

- `src/agent_llm/` – agent package
  - `llm.py` – OpenRouter client wrapper (OpenAI-compatible API)
  - `tools.py` – tool definitions + `load_registry_tools()`
  - `tools_custom.py` – custom tools (Phase 2)
  - `agent.py` – agent wrapper and tool-call loop
  - `_runner.py` – core runner logic (importable module)
  - `cli.py` – CLI entry point
- `scripts/run_supervisor_parallel.py` – thin shim for standalone mode
- `agents/registry.json` – agent definitions (architect, coder, reviewer)
- `tools/registry.json` – custom tool definitions
- `notes/` – default root for file tools (safe sandbox)
- `tests/` – unit tests (47 passing)
