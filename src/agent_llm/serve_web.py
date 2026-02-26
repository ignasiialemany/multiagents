import os
import json
import argparse
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

from agent_llm._runner import (
    _setup_infrastructure,
    _make_create_agent_tool,
    _INTERACTIVE_SYSTEM_PROMPT,
    run_interactive_turn,
)
from agent_llm.cli_output import RunDisplay
from agent_llm.agent import Agent
from agent_llm.tools import create_meeting_tool

app = Flask(__name__)

# Global state
infra = None
interactive_agent = None
memory = None
store = None
llm_client = None
agent_registry = None
agents = None
conversation = []
display = RunDisplay(verbose=False)


def get_or_init_state():
    global infra, interactive_agent, memory, store, llm_client, agent_registry, agents
    
    if infra is not None:
        return
        
    import dotenv
    dotenv.load_dotenv()
    
    # Set up arguments for infrastructure
    args = argparse.Namespace(
        work_dir=".",
        registry="agents/registry.json",
        agents_registry="agents/registry.json",
        tools_registry="tools/registry.json",
        sessions_dir="sessions",
        model=None,
        redis_host="localhost",
        redis_port=6379,
        meeting_dir=None,
    )
    
    infra = _setup_infrastructure(args, display)
    agent_registry = infra["agent_registry"]
    agents = infra["agents"]
    
    from agent_llm.llm import OpenRouterLLMClient
    
    # We are not in interactive mode, so some things may not exist in infra dict directly
    # Need to instantiate MemoryStream and SessionStore if missing, similar to _run_interactive
    memory = infra.get("memory")
    if memory is None:
        from agent_llm._runner import MemoryStream
        memory = MemoryStream(persist_path=Path(args.work_dir).resolve() / ".agent-llm" / "memory.json")
        
    store = infra.get("store")
    if store is None:
        from agent_llm.session import SessionStore
        sessions_dir = Path(args.work_dir).resolve() / args.sessions_dir
        store = SessionStore(sessions_dir)
        
    llm_clients = infra.get("llm_clients", {})
    llm_client = llm_clients.get("minimax") or (next(iter(llm_clients.values())) if llm_clients else None)
    if llm_client is None:
        try:
            from agent_llm.llm import OpenRouterLLMClient
            # Try to grab API key from environment
            api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY"))
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is not set")
            llm_client = OpenRouterLLMClient(api_key=api_key)
        except Exception as e:
            raise ValueError(f"No LLM client configured and could not create default client: {e}")
    
    # Reload previous conversation if it exists
    global conversation
    conversation = store.load("interactive_agent")
    
    # Build tools for interactive agent
    interactive_tools = {}
    
    meeting_tool = create_meeting_tool(
        agent_registry=agent_registry,
        llm_client=llm_client,
        display=display,
        run_meeting_fn=None,  # Not fully supported in minimal web UI yet
        meeting_dir=args.meeting_dir,
        workspace=infra.get("redis_workspace")
    )
    interactive_tools[meeting_tool["name"]] = meeting_tool
    interactive_tools[infra["sandbox_tool"]["name"]] = infra["sandbox_tool"]
    
    create_agent_tool = _make_create_agent_tool(
        agent_registry, agents, infra["agent_tools"], infra, args.registry
    )
    interactive_tools[create_agent_tool["name"]] = create_agent_tool
    
    interactive_agent = Agent(llm_client, interactive_tools, agent_id="interactive")


@app.route("/")
def index():
    # Simple inline HTML template
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent LLM</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #log { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; white-space: pre-wrap; }
            .user-msg { color: #0066cc; font-weight: bold; margin-bottom: 5px; }
            .agent-msg { color: #333; margin-bottom: 15px; }
            #input-area { display: flex; }
            #msg { flex-grow: 1; padding: 8px; }
            #send { padding: 8px 16px; margin-left: 10px; }
            .loading { color: #888; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Interactive Agent</h1>
        <div id="log"></div>
        <div id="input-area">
            <input type="text" id="msg" placeholder="Type a message..." autofocus>
            <button id="send">Send</button>
        </div>
        
        <script>
            const log = document.getElementById('log');
            const msgInput = document.getElementById('msg');
            const sendBtn = document.getElementById('send');
            
            function appendMsg(role, text) {
                const div = document.createElement('div');
                div.className = role === 'user' ? 'user-msg' : 'agent-msg';
                div.textContent = (role === 'user' ? 'You: ' : 'Agent: ') + text;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
            }
            
            async function sendMessage() {
                const text = msgInput.value.trim();
                if (!text) return;
                
                msgInput.value = '';
                msgInput.disabled = true;
                sendBtn.disabled = true;
                
                appendMsg('user', text);
                
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'agent-msg loading';
                loadingDiv.textContent = 'Agent is thinking...';
                log.appendChild(loadingDiv);
                log.scrollTop = log.scrollHeight;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });
                    
                    const data = await response.json();
                    
                    // Remove loading message
                    log.removeChild(loadingDiv);
                    
                    if (data.error) {
                        appendMsg('agent', 'Error: ' + data.error);
                    } else {
                        appendMsg('agent', data.response || '(no response)');
                    }
                } catch (err) {
                    log.removeChild(loadingDiv);
                    appendMsg('agent', 'Error: ' + err.message);
                } finally {
                    msgInput.disabled = false;
                    sendBtn.disabled = false;
                    msgInput.focus();
                }
            }
            
            sendBtn.addEventListener('click', sendMessage);
            msgInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
            
            // Load existing conversation
            fetch('/history').then(res => res.json()).then(data => {
                if (data.conversation) {
                    data.conversation.forEach(msg => {
                        appendMsg(msg.role, msg.content);
                    });
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/history", methods=["GET"])
def history():
    get_or_init_state()
    return jsonify({"conversation": conversation})


@app.route("/chat", methods=["POST"])
def chat():
    get_or_init_state()
    
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Missing message"}), 400
        
    user_input = data["message"]
    
    try:
        # Check for /meeting command rewrite (similar to CLI)
        if user_input.lower().startswith("/meeting"):
            parts = user_input.split(maxsplit=1)
            topic = parts[1].strip() if len(parts) > 1 else ""
            topic_clause = f" Topic: {topic!r}." if topic else ""
            known_ids = ", ".join(a["agent_id"] for a in agent_registry)
            user_input = (
                f"I'd like to run a meeting.{topic_clause} "
                f"Currently registered agents: {known_ids}. "
                "Before running the meeting, ask me which agents should attend, "
                "what the goal/desired outcome is, and whether we need any new "
                "agents that don't exist yet. Once you have that information, "
                "create any missing agents with create_agent, then start the "
                "meeting with create_meeting."
            )
            
        # Run one turn
        response_text, _ = run_interactive_turn(
            user_input=user_input,
            conversation=conversation,
            memory=memory,
            interactive_agent=interactive_agent,
            llm_client=llm_client,
            display=display,
            reflection_threshold=50
        )
        
        # Save state
        store.save("interactive_agent", conversation)
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
