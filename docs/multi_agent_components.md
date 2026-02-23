# Multi-Agent Components: Bus, Sessions, and Workspace

The multi-agent runner uses three shared components. Each serves a different kind of state and communication.

---

## Output files (what the multi-agent runner generates)

When you run `scripts/run_multi_agent.py`, the following files are created or updated:

| Path | Produced by | Description |
|------|--------------|-------------|
| **`sessions/{agent_id}.json`** | SessionStore | One JSON file per agent (e.g. `sessions/architect.json`, `sessions/coder.json`). Contains that agent’s full conversation history (system, user, assistant, tool messages). Created or updated after each agent turn. |
| **`workspace.json`** | Workspace | Single JSON file at the repo root. Shared key–value state that agents read/update via `read_workspace` and `update_workspace`. Created on first write; updated whenever any agent calls `update_workspace`. |

The **message bus** is in-memory only; it does not write any files. Inbox contents are consumed each round and only affect what gets appended to session files when agents run.

---

## Tools: what each agent has, and the tool designer

Each agent in the multi-agent run gets the **same** combined tool set:

| Source | Tools | Where it comes from |
|--------|--------|----------------------|
| **Default** | `list_files`, `read_file`, `search_docs` | `create_default_tools(notes_root)` — scoped to the `notes/` directory. |
| **Registry** | Custom tools (e.g. `grep_repo`) | `load_registry_tools(tools/registry.json)`. Every entry in the registry is loaded and available to all agents. |
| **Workspace** | `read_workspace`, `update_workspace` | `create_workspace_tools(workspace)` — same shared workspace instance. |
| **Delegate** | `send_message(to_agent, content)` | `create_delegate_tool(bus, agent_id, agent_registry)` — one per agent (each knows its own id and the registry). |

So: **default + registry + workspace + send_message** are all active in `run_multi_agent.py`.

**Tool designer: is it still there?**  
Yes. The tool designer flow is **unchanged** and **separate** from the multi-agent runner:

1. **Design a tool:** `python scripts/design_tool.py "description of what you need"` — runs the agent with **no** tools and the designer system prompt; the agent proposes a tool (JSON); the proposal is written to `tools/proposals/` and logged in `tools/proposals_log.jsonl`.
2. **Implement and register:** You implement the function (e.g. in `src/agent_llm/tools_custom.py`), then `python scripts/register_tool.py tools/proposals/<name>_proposal.json` — adds the tool to `tools/registry.json`.
3. **Use in multi-agent (or single-agent):** Both `run_multi_agent.py` and `run_with_registry.py` call `load_registry_tools(tools/registry.json)`. So any tool you register is available to **all** agents in the multi-agent run.

So the tool designer is still how you add new tools; the multi-agent runner simply uses whatever is in the registry (plus default, workspace, and delegate tools).

---

## 1. Message Bus (`MessageBus`)

**What it is:** A central place where messages are sent and delivered. It holds **one inbox per agent** (`agent_id` → list of messages).

**How agents use it:**
- When Agent A wants to talk to Agent B, A uses the **`send_message`** tool (which calls `bus.post(msg)`). The message goes into B’s inbox.
- The runner loop calls `bus.get_inbox(agent_id)` for each agent, takes those messages, and injects them as user messages into that agent’s conversation before running a turn.

**So:** The bus is for **direct, addressed communication** between agents (and from the user to an agent). Each agent only sees messages that were sent **to** them. Optionally, messages with `"to": "broadcast"` can notify **subscribers** (e.g. an orchestrator); that path is not used by the current runner.

**In code:** `bus = MessageBus()` in `run_multi_agent.py`; tools get the same `bus` instance so that `send_message` posts to the right inboxes.

---

## 2. Session Store (`SessionStore`)

**What it is:** Persisted **conversation history per agent**. Each agent has its own session: a list of messages (system, user, assistant, tool) stored in a file like `sessions/{agent_id}.json`.

**How agents use it:**
- The runner **loads** that agent’s messages before a turn, optionally **appends** new user message(s) (e.g. from the bus), runs `agent.run(messages)`, then **saves** the updated messages back.
- Agents never see each other’s session; they only see their own history plus whatever is injected (e.g. “[Message from architect]: …”) from the bus.

**So:** Sessions are **private, per-agent state**. They give each agent a stable memory across turns. Interaction between agents happens only via the **bus** (and optionally the workspace), not by sharing session history.

**In code:** `store = SessionStore(_repo_root / "sessions")`; the runner uses `store.load(session_id)` and `store.save(session_id, messages)` around each agent turn.

---

## 3. Workspace (`Workspace`)

**What it is:** A **shared, persistent key–value store** backed by a single JSON file (e.g. `workspace.json`). All agents can read and write it via tools.

**How agents use it:**
- Each agent gets **workspace tools**: `read_workspace` (see the whole JSON) and `update_workspace(key, value)` (set a key). When an agent calls these tools, the underlying `Workspace` instance loads/saves the same file.
- Typical uses: shared task list, “current plan”, specs or results that multiple agents need to see or update.

**So:** The workspace is **shared, visible state** that agents use to coordinate and build together. Unlike the bus, it is not “who said what to whom” but “what is the current shared context/artifacts.” Unlike sessions, it is not private; every agent with workspace tools sees the same data.

**In code:** `workspace = Workspace(_repo_root / "workspace.json")`; `create_workspace_tools(workspace)` gives each agent the same tools backed by that one `Workspace` instance.

---

## How they relate

| Component   | Scope       | Purpose                                      |
|------------|-------------|----------------------------------------------|
| **Bus**    | Per-message | Who is talking to whom (inboxes, send_message). |
| **Sessions** | Per-agent   | Each agent’s own conversation history.      |
| **Workspace** | Global     | Shared context and artifacts (read/update).  |

- **Bus + Sessions:** Agents keep **separate sessions** but **interact** by sending messages through the bus; the runner turns bus messages into new user messages in the right agent’s session.
- **Workspace:** Agents **coordinate** by reading and writing the same workspace (e.g. “here’s the API spec,” “task X is done”); no addressing, just shared state.

Together they give you: private memory (sessions), directed communication (bus), and shared context (workspace).

---

## The two loops (orchestrator vs agent)

There are two levels of looping: the **runner’s loop** (which checks inboxes and runs agents) and, inside each run, the **agent’s loop** (LLM + tools until a final answer).

**1. Runner loop (one pass = one “round”)**

```
┌─────────────────────────────────────────────────────────────────┐
│  iteration = 1 .. max_turns (e.g. 10)                            │
│  Stop early if no agent had any mail this round.                 │
│                                                                  │
│  For each agent in order (e.g. architect, then coder):           │
│    inbox = bus.get_inbox(agent_id)                              │
│    if inbox not empty:                                           │
│      for each message in inbox:                                  │
│        run_agent_turn(agent, session_id, store, user_msg)  ──────┼──► 2. Agent loop (below)
│                                                                  │
│  If nobody had mail this round → exit. Else → next iteration.    │
└─────────────────────────────────────────────────────────────────┘
```

**2. Agent loop (inside `run_agent_turn` → `agent.run`)**

```
┌─────────────────────────────────────────────────────────────────┐
│  One “turn” for one agent (one or more new user messages).       │
│                                                                  │
│  Load session → append new message(s) → then:                     │
│                                                                  │
│  repeat (up to MAX_ITERATIONS):                                  │
│    call LLM(messages, tools)                                     │
│    if LLM returns tool_calls:                                    │
│      for each tool_call: execute tool (e.g. send_message,        │
│        read_workspace) → append tool result to messages          │
│      → loop again                                                 │
│    else:                                                          │
│      return final answer → save session → done for this turn    │
└─────────────────────────────────────────────────────────────────┘
```

**How they connect**

- Runner loop **decides who runs**: each round it looks at every agent’s **inbox**. If there are messages, it runs that agent once per message (each run is one “turn”).
- One turn = one **agent loop**: load session, add the new user message, then keep calling the LLM and running tools until the LLM stops with a final answer. If the agent uses `send_message(coder, "…")`, that goes into **coder’s inbox** and will be seen in the **next round** when the runner checks coder’s inbox.

So: **outer loop** = “which agents have mail this round, run them”; **inner loop** = “for this agent, keep doing LLM + tools until it answers.”
