# Improving Multi-Agent Meeting Flow and Prior Context

This doc summarizes what the current meeting workflow does, what agents know (and don’t) before and during the meeting, and research-backed changes to improve **flow** and **exactly what they need to know prior**.

---

## Current behavior (quick recap)

**Where it lives:** `run_meeting()` in [src/agent_llm/_runner.py](src/agent_llm/_runner.py); `create_meeting_tool` in [src/agent_llm/tools.py](src/agent_llm/tools.py).

**Flow today:**

1. **Preparation**  
   For each agent, in isolation:
   - **System:** “You are \<agent_id\>. Your role: \<description\>. You are preparing for a meeting. Prepare 1–2 questions or discussion points…”
   - **User:** “Meeting topic: \<topic\>. List 1–2 questions or points…”
   - No list of other participants, no other roles, no shared brief. Each agent only sees the topic and its own role.

2. **Discussion**  
   For each round, for each agent in turn:
   - **System:** “You are \<agent_id\>. \<description\>. You are in a meeting. You see the preparation and discussion so far. Reply in 1–2 short paragraphs. Do not use tools.”
   - **User:** “Meeting topic: \<topic\”. Preparation and discussion so far: \<full transcript\”. What do you want to say?”
   - So during the meeting they see the **full transcript** (everyone’s prep + all previous turns). They still do **not** get an explicit “who else is here and what they care about” in the system message.

3. **Plan**  
   A single LLM call over the full transcript produces the plan (decisions, next steps, open questions).

**What’s missing for “prior context” and “flow”:**

- Agents don’t know **who else is in the meeting** or **their roles** before preparing or speaking.
- There is **no shared brief or pre-read** (e.g. a short doc or workspace summary everyone sees).
- There is **no explicit meeting goal** (e.g. “align on architecture” vs “decide go/no-go”).
- There is **no structure** (e.g. phases like “divergent → convergent → critical” that your own meeting transcript already discussed).
- **Turn-taking is fixed** (round-robin); there’s no “who should speak next” or “answer this question from the architect.”

---

## What agents should know before the meeting

So that there’s **more flow** and they have **exactly what they need to know prior**, they should get:

| Prior context | Purpose |
|---------------|--------|
| **Participant list + roles** | So each agent knows who they’re addressing and can prepare questions for the right role (e.g. “I’ll ask the architect about X, the coder about Y”). |
| **Meeting goal (optional)** | One line: “Align on architecture,” “Decide scope,” “Retrospective.” Keeps the conversation focused. |
| **Brief / pre-read (optional)** | Short shared text (or pointer to workspace key) everyone sees before prep. E.g. “Current state: we have X; constraint: Y.” Reduces repeated clarification. |

**Implementation idea:**  
Before the prep phase, build a **context block** that includes:

- “Meeting topic: …”
- “Meeting goal: …” (if provided)
- “Participants: \<agent_id\>: \<description\>” for every participant (including self).
- “Brief / pre-read: …” (if provided).

Then use this same block in the **system** (or the first user message) for both **prep** and **discussion**, so every agent sees it in every call. Prep and discussion prompts can then say “you know who is in the meeting and their roles; use that to prepare / address the right people.”

---

## Improving flow in the conversation

“More flow” can mean:

1. **Prep that aligns with the room**  
   Once agents see who’s in the meeting and their roles, prep can naturally target “questions for the architect” vs “points for the reviewer,” so the discussion feels more connected from the first turn.

2. **Structured phases (optional)**  
   Your transcript already agreed on phases (e.g. divergent → convergent → critical). We could:
   - Add an optional `phase` or `phase_plan` to the meeting (e.g. “Round 1–2: divergent; Round 3–4: convergent”).
   - In the system prompt for each round, tell the agent which phase we’re in and what’s expected (e.g. “Phase: divergent. Generate ideas; do not criticize yet.”). That creates more flow by giving each turn a clear purpose.

3. **Optional brief parameter**  
   Allow the caller (e.g. interactive agent or user) to pass a **brief** or **pre_read** string (or “use workspace key X”). Run_meeting includes it in the context block above so everyone has the same starting picture. That reduces “what’s the context?” and increases “building on the same facts.”

4. **Optional “answer this” targeting**  
   A more advanced option: when the transcript shows “\[architect\]: … I’d like the coder’s view on X,” the next turn for the coder could include “The architect asked for your view on: X.” So the model is explicitly invited to answer, which can improve flow. (Simple version: in the user message for the current speaker, append “If someone’s question is directed at you, address it.”)

---

## Concrete implementation sketch

**1. Extend `run_meeting()` signature (e.g.):**

- `topic: str`
- `agent_ids: list[str]`
- `meeting_goal: str | None = None`
- `brief: str | None = None`  
  (or `brief_workspace_key: str | None = None` and resolve from workspace)
- `max_rounds: int = 4`
- `phase_plan: list[str] | None = None`  
  (e.g. `["divergent", "divergent", "convergent", "critical"]` for 4 rounds)
- `meeting_dir: Path | None = None`
- plus existing `agent_registry`, `llm_client`, `display`.

**2. Build a shared context string once:**

- Topic, goal (if any), participants (id + description), brief (if any). Use it in **prep** and **discussion** system/user messages so “what they need to know prior” is always present.

**3. Prep phase:**

- System (or user) includes the shared context and: “You know who is in the meeting and their roles. Prepare 1–2 questions or points you want to raise, including who you’re addressing if relevant.”

**4. Discussion rounds:**

- For each round, if `phase_plan` is set, add to the system: “Current phase: \<phase\>. \<short instruction for that phase\>.”
- User message continues to include the full transcript; optionally append one line: “If someone asked you a direct question, address it in your reply.”

**5. Meeting tool (`create_meeting`):**

- Add optional parameters: `meeting_goal`, `brief`, and optionally `phase_plan` (or a single “phase” string for the whole meeting). Pass them through to `run_meeting_fn`.

**6. Optional: workspace integration**

- If `brief_workspace_key` is set, run_meeting (or the tool) reads that key from the workspace and uses it as the brief. Then the interactive agent (or user) can set “meeting_brief” in the workspace before calling `/meeting`, and everyone sees the same prior context.

---

## Summary

- **Prior context:** Give every agent, before prep and during discussion, a **participant list + roles**, an optional **meeting goal**, and an optional **brief/pre-read**. That’s “exactly what they need to know prior.”
- **Flow:** Use that context so prep is aligned with the room; optionally add a **brief** and **phase_plan** so the conversation has a shared starting point and a clear purpose per round. Optionally add light “answer this” targeting so replies feel more connected.

Implementing the shared context block and participant list in prep/discussion is the highest-impact, smallest change; adding `meeting_goal` and `brief` (and later phases or targeting) can build on that.

**(Status: Implemented)** The `create_meeting_tool` now supports `meeting_goal`, `brief`, `phase_plan`, and `brief_workspace_key`. The `run_meeting` function uses them to build a robust prior context block for each agent.

---

## Research: How to improve the meeting flow further

This section summarizes findings from your internal meeting ([sessions/meetings/meeting_Research other multi-agent frameworks...](sessions/meetings/)) and from recent literature on multi-agent meetings and conversation structure.

### 1. What your meeting already agreed (from the transcript)

- **Structured output templates** (priority 1): Enhance the synthesis phase with defined schemas (summary, action items, decisions). Low risk, clear value; no change to the meeting model.
- **Memory persistence** (priority 2): Store context between meetings (e.g. recurring standups) and inject it as pre-read for the next meeting. Differentiator; also a dependency for role-based agents.
- **Role-based AI agents** (priority 3): Add dedicated AI participants (e.g. Devil’s Advocate, Synthesis Agent) with explicit personas and injection points. Highest complexity.

The “adopt vs. adapt” frame was used: adopt when a feature fills a clear workflow gap; adapt when it needs customization for meeting context.

### 2. Turn-taking and “who speaks next”

**Current behaviour:** Fixed round-robin; every agent speaks once per round in registry order.

**Research:** Multi-agent debate work shows that *who speaks next* can matter less than *reasoning strength* and *diversity* of views. Process design (e.g. “debate only when necessary”) can improve efficiency without hurting quality. Options to improve flow:

- **Keep round-robin but add phase awareness** — Already in place via `phase_plan` (divergent → convergent → critical). Phases give each turn a clear purpose even with fixed order.
- **Optional “answer this” targeting** — When the transcript shows a direct question to an agent (e.g. “I’d like the coder’s view on X”), the next time that agent is due to speak, prepend: “The architect asked for your view on: X. Address it in your reply.” Improves relevance without full dynamic speaker selection.
- **Dynamic “who speaks next” (later)** — A lightweight facilitator or router could choose the next speaker from the transcript (e.g. “who was asked a question” or “who hasn’t spoken this round”). More complex; worth doing after structured output and memory.

### 3. Structured synthesis (summary, action items, decisions)

**Research:** Meeting summarization works well when treated as **action-item-driven** and **fact-based**: extract salient facts and action items, then build the summary from them. This reduces hallucination and aligns with how humans use meeting notes. LLMs can produce structured outputs (summary, action items, decisions) via prompt engineering without fine-tuning.

**For our tool:**

- Define **output templates** for the plan phase: e.g. “Summary” (paragraph), “Decisions” (list), “Action items” (owner + description + due), “Open questions” (list). The synthesis prompt would ask for these sections explicitly.
- Optionally run **two steps**: (1) extract facts/action items from the transcript, (2) generate the narrative summary from those. Reduces drift and keeps actionability.

### 4. Memory and context for recurring meetings

**Research:** Recurring meetings with no memory force repetition and lose thread. Injected prior context (e.g. “Last time we decided X; action item Y was owned by Z”) improves continuity.

**For our tool:**

- **Session/series identity:** Link meetings by a “series” or “recurring meeting id” (e.g. “weekly standup”).
- **What to persist:** At least: short summary of last meeting, outstanding action items, key decisions. Stored in workspace or a dedicated meeting-memory store.
- **Injection:** When starting a meeting in a series, load the last meeting’s summary + open actions and add them to the **brief** (or a “Previous meeting” block in shared context). So “what they need to know prior” includes what happened last time.

### 5. Role-based agents (Devil’s Advocate, Synthesis Agent)

**Research:** Frameworks like CrewAI use explicit role-based agents; debate frameworks use structured roles (e.g. critic, synthesizer). “Synthesis Agent” that weaves the best ideas from the transcript fits the convergent phase; “Devil’s Advocate” fits the critical phase.

**For our tool:**

- **Personas in registry:** Add optional meeting-only roles (e.g. `devil_advocate`, `synthesis_agent`) with descriptions and, if needed, a different system prompt (e.g. “You only challenge assumptions; do not propose solutions”).
- **Injection points:** During discussion rounds, either (a) include these agents in the round-robin with phase-aware instructions, or (b) run them at phase boundaries (e.g. after convergent round, run Synthesis Agent once to propose a merged plan; after that, run Devil’s Advocate to stress-test it). Option (b) keeps the main flow simple and uses roles at clear moments.

### 6. Suggested implementation order (aligned with your meeting)

1. **Structured output templates** — Extend the plan phase with explicit schema (summary, decisions, action items, open questions). Optional: two-step synthesis (extract then summarize).
2. **Memory persistence** — Add meeting series id + storage of last summary and open actions; inject into brief/context for the next meeting in the series.
3. **“Answer this” targeting** — When building the user message for an agent’s turn, detect if they were directly asked something in the transcript and prepend a short “Address this: …” line.
4. **Role-based agents** — Add optional meeting roles and phase-boundary runs (e.g. Synthesis Agent after convergent, Devil’s Advocate after critical), once memory and structured output are in place.

This order matches your meeting’s priorities and keeps each step buildable on the previous one.
