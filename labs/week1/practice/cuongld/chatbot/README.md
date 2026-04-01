# Streamlit Chatbot Skeleton

A minimal project scaffold to practice concepts from agent-master-class.

## Structure

```
chatbot/
  app.py
  agent.py
  graph.py
  langgraph.json
  config.py
  memory.py
  requirements.txt
  .env.example
```

## Patterns implemented

- Prompt Chaining (constraint extraction -> response building)
- Prompt Chaining + Gate (quality check + retry once)
- Routing (simple vs advanced pipeline by difficulty)
- Parallelization (Send API fan-out to worker nodes)
- Orchestrator-Workers (dynamic task split + reduce)
- Human-in-the-loop (`interrupt` + `Command(resume=...)`)
- Production deploy structure (`graph.py`, `langgraph.json`)
- SQLite Checkpointer persistence (`seoul_agent_memory.db`)
- Time travel debugging (`get_state_history` + checkpoint fork)

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and update values if needed.
4. Run:

```bash
streamlit run app.py
```

## Practice roadmap

1. Replace the stub in `agent.py` with real model calls.
2. Add tools and function-calling.
3. Add multi-agent or graph orchestration.
4. Add guardrails, tracing, and evaluation.
