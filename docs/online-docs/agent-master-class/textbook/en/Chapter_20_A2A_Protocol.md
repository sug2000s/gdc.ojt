# Chapter 20: A2A (Agent-to-Agent) Protocol

---

## 1. Chapter Overview

In this chapter, we will learn about the **A2A (Agent-to-Agent) protocol**. The A2A protocol is an open standard protocol that enables different AI agents to **communicate over a network**. Through this protocol, agents built with different frameworks (Google ADK, LangGraph, etc.) can collaborate as if they were a single system.

### Learning Objectives

- Understand the concept and necessity of the A2A protocol
- Learn how to convert an agent into an A2A server using Google ADK's A2A utility (`to_a2a`)
- Learn how to connect remote agents as sub-agents using `RemoteA2aAgent`
- Understand the structure and role of Agent Cards
- Learn how to directly implement the A2A protocol to turn a LangGraph agent into an A2A server
- Confirm that inter-agent communication is possible regardless of the framework

### Project Structure

```
a2a/
├── .python-version          # Python 3.13
├── pyproject.toml           # Project dependency definitions
├── remote_adk_agent/        # Remote ADK agent running as A2A server
│   └── agent.py
├── user-facing-agent/       # Root agent that communicates directly with users
│   └── user_facing_agent/
│       ├── __init__.py
│       └── agent.py
└── langraph_agent/          # LangGraph-based A2A server agent
    ├── graph.py
    └── server.py
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `google-adk[a2a]` | 1.15.1 | Google ADK + A2A extensions |
| `google-genai` | 1.40.0 | Google Generative AI |
| `langchain[openai]` | 0.3.27 | LangChain + OpenAI integration |
| `langgraph` | 0.6.8 | LangGraph state graph |
| `litellm` | 1.77.7 | Unified LLM integration interface |
| `fastapi[standard]` | 0.118.0 | Web server framework |
| `uvicorn` | 0.37.0 | ASGI server |
| `python-dotenv` | 1.1.1 | Environment variable management |

---

## 2. Section-by-Section Detailed Explanation

---

### 20.0 Introduction - Project Initial Setup

#### Topic and Objectives

Set up the project environment for learning the A2A protocol. Create a new Python 3.13-based project and install all required dependencies.

#### Core Concept Explanation

**What is the A2A Protocol?**

A2A (Agent-to-Agent) is an open protocol led by Google that provides interoperability between different AI agent systems. Previously, inter-agent communication was only possible within a single framework, but with A2A, agents can exchange messages over the network **regardless of the framework**.

The core components of the A2A protocol are as follows:

1. **Agent Card**: A JSON document containing an agent's metadata. It defines the agent's name, description, capabilities, supported input/output formats, etc. It is accessible at the `/.well-known/agent-card.json` path.
2. **Message**: The standard format for messages exchanged between agents.
3. **Transport**: The message transmission method (e.g., JSON-RPC).

**Why a separate `a2a/` directory?**

Since the A2A protocol requires running multiple independent agent servers, a new project structure separate from the existing project is created. Each agent runs independently on a separate port.

#### Code Analysis

```toml
# a2a/pyproject.toml
[project]
name = "a2a"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi[standard]==0.118.0",
    "google-adk[a2a]==1.15.1",
    "google-genai==1.40.0",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.8",
    "litellm==1.77.7",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

Key points to note:
- `google-adk[a2a]`: When installing Google ADK, the `[a2a]` extras are included. This includes the A2A-related utilities (`to_a2a`, `RemoteA2aAgent`, etc.).
- `fastapi` and `uvicorn`: Web frameworks used when directly implementing the A2A protocol server.
- `litellm`: A library that allows using various LLM providers (OpenAI, Anthropic, Google, etc.) through a single interface.

#### Practice Points

1. Initialize the project and install dependencies using `uv`:
   ```bash
   cd a2a
   uv sync
   ```
2. Check the `.python-version` file to verify that Python 3.13 is specified.
3. Create a `.env` file and set up the required API keys (OpenAI, etc.).

---

### 20.1 A2A Using ADK - Creating A2A Agents with ADK

#### Topic and Objectives

Learn how to convert a regular ADK agent into an A2A protocol server using Google ADK's `to_a2a` utility. Also, create a "root agent" that communicates directly with users.

#### Core Concept Explanation

**Two Types of Agent Roles**

In this section, we create two types of agents:

1. **Remote Agent**: Operates as an A2A server and provides specialized knowledge in a specific domain (e.g., history). It runs independently on a separate port (8001).
2. **User-Facing Agent**: Communicates directly with users and delegates tasks to remote agents as needed.

**The `to_a2a` Function**

The `to_a2a` function from the `google.adk.a2a.utils.agent_to_a2a` module converts a regular ADK `Agent` object into a web application that supports the A2A protocol. What this function does:
- Automatically generates an Agent Card and exposes it at `/.well-known/agent-card.json`
- Creates a JSON-RPC-based message receiving endpoint
- Converts the agent's responses into the A2A protocol format

**LiteLlm Model**

`LiteLlm` is an adapter that allows using models from various LLM providers (OpenAI, Anthropic, etc.) within Google ADK. It is specified in the format `provider/model_name`, such as `"openai/gpt-4o"`.

#### Code Analysis

**Remote ADK Agent (A2A Server)**

```python
# a2a/remote_adk_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)

app = to_a2a(agent, port=8001)
```

Key points:
- `load_dotenv()` is called immediately after the import. This is because subsequent import statements may require environment variables.
- When creating the `Agent`, `name` and `description` are specified. This information is automatically included in the Agent Card.
- `to_a2a(agent, port=8001)` converts the agent into an A2A server app. `port=8001` is the port number this server will use.
- The returned `app` object is an ASGI application that can be run through `uvicorn`.

**User-Facing Agent (Root Agent)**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)
```

Key points:
- This agent does not use `to_a2a`. It is run directly through Google ADK's web UI (`adk web`).
- `sub_agents=[]` means no sub-agents are connected yet. Remote agents will be connected in later sections.
- Note that the variable name is `root_agent`. This is a convention recognized by the ADK web UI.

**`__init__.py` File**

```python
# a2a/user-facing-agent/user_facing_agent/__init__.py
from . import agent
```

This file allows ADK to automatically load the `agent` module within the package.

#### Practice Points

1. Start by running the remote agent:
   ```bash
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001
   ```
2. Access `http://localhost:8001/.well-known/agent-card.json` in your browser to view the automatically generated Agent Card.
3. Run the user-facing agent with the ADK web UI:
   ```bash
   cd a2a
   adk web user-facing-agent
   ```

---

### 20.2 A2A For Dummies - Adding Tools to the Agent

#### Topic and Objectives

Add a tool to the remote agent to verify that the agent's tool usage works correctly through A2A. Also, learn the pattern of using `tools` instead of `sub_agents`.

#### Core Concept Explanation

**Adding Tools to an Agent**

In the A2A protocol, remote agents can perform complex tasks using tools, not just simple text responses. An agent with tools can invoke them based on the user's request and generate responses based on the results.

**Changing from `sub_agents` to `tools`**

The previously empty list (`sub_agents=[]`) has been changed to `tools=[dummy_tool]`. This clarifies the agent's role:
- `sub_agents`: Delegates tasks to other agents
- `tools`: Function-type tools that the agent can use directly

#### Code Analysis

```python
# a2a/remote_adk_agent/agent.py (modified portion)
def dummy_tool(hello: str):
    """Dummy Tool. Helps the agent"""
    return "world"


agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    tools=[dummy_tool],
)

app = to_a2a(agent, port=8001)
```

Key points:
- `dummy_tool` is a simple Python function. ADK automatically parses the function's **name**, **parameter type hints**, and **docstring** to generate a tool definition that the LLM can understand.
- The type hint on the parameter `hello: str` is required. The LLM needs to know what values to pass.
- The docstring `"""Dummy Tool. Helps the agent"""` explains the purpose of this tool to the LLM.
- `tools=[dummy_tool]` registers the tool with the agent. The agent can automatically invoke this tool when needed.

**Tool Call Flow Through A2A**

```
User → Root Agent → [A2A Protocol] → Remote Agent → dummy_tool call → Response returned
```

The important point is that the tool call happens **inside the remote agent server**. The root agent does not need to know what tools the remote agent uses. The A2A protocol **encapsulates the internal implementation** of the agent.

#### Practice Points

1. Replace `dummy_tool` with a real useful tool (e.g., Wikipedia search, date calculation, etc.).
2. Modify the tool's docstring and observe how the LLM's tool call patterns change.
3. Register multiple tools like `tools=[tool1, tool2, tool3]`.

---

### 20.3 RemoteA2aAgent - Connecting Remote Agents

#### Topic and Objectives

Learn how to use `RemoteA2aAgent` so that the root agent can utilize agents on remote A2A servers as sub-agents. This is the **core usage pattern** of the A2A protocol.

#### Core Concept Explanation

**What is RemoteA2aAgent?**

`RemoteA2aAgent` is a class provided by Google ADK that allows agents running on remote A2A servers to be used as if they were local sub-agents. Internally, it exchanges A2A protocol messages via HTTP, but from the user's perspective, it can be treated the same as a regular sub-agent.

**AGENT_CARD_WELL_KNOWN_PATH**

According to the A2A protocol standard, an agent's metadata (Agent Card) should be located at the `/.well-known/agent-card.json` path. The `AGENT_CARD_WELL_KNOWN_PATH` constant is exactly this path string. Through this, `RemoteA2aAgent` can automatically retrieve the remote agent's information (name, description, capabilities, message receiving URL, etc.).

**Agent Delegation Pattern**

```
User: "Tell me about Napoleon"
    ↓
Root Agent (StudentHelperAgent)
    ↓ Determines it's a history-related question based on description
    ↓ Delegates to history_agent
    ↓
[A2A Protocol - HTTP Communication]
    ↓
Remote Agent (HistoryHelperAgent, port 8001)
    ↓ Generates response
    ↓
[A2A Protocol - HTTP Response]
    ↓
Root Agent → Delivers response to user
```

The root agent decides which agent to delegate tasks to based on each sub-agent's `description`. Therefore, writing clear and specific descriptions is extremely important.

#### Code Analysis

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

history_agent = RemoteA2aAgent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    agent_card=f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
    ],
)
```

Key points:

1. **Creating RemoteA2aAgent**:
   - `name`: The name of the remote agent. It must match the name registered on the remote server.
   - `description`: A description that the root agent references when making delegation decisions.
   - `agent_card`: The full URL of the Agent Card. `f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}"` becomes `http://127.0.0.1:8001/.well-known/agent-card.json`.

2. **Connecting to sub_agents**:
   - `sub_agents=[history_agent]` adds the remote agent to the sub-agent list.
   - The root agent can now delegate history-related questions to `history_agent` via the A2A protocol.

3. **Transparent Abstraction**:
   - `RemoteA2aAgent` behaves like a subclass of `Agent`. From the root agent's perspective, there is no difference between local and remote agents.
   - Complex tasks such as network communication and protocol conversion are handled internally by `RemoteA2aAgent`.

#### Practice Points

1. Open two terminals and run the remote agent and root agent respectively:
   ```bash
   # Terminal 1: Remote agent
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # Terminal 2: Root agent
   cd a2a
   adk web user-facing-agent
   ```
2. In the ADK web UI, try asking both history-related questions and general questions. Observe which questions are delegated to the remote agent.
3. Modify the `description` and experiment with how delegation decisions change.

---

### 20.5 SendMessageResponse - Direct A2A Server Implementation with LangGraph

#### Topic and Objectives

Learn how to make an agent built with **LangGraph** (not Google ADK) operate as an A2A server. By **directly implementing** the A2A protocol without the `to_a2a` utility, we gain a deep understanding of the protocol's internal workings. Through this, we confirm that **any agent, regardless of framework**, can support the A2A protocol.

#### Core Concept Explanation

**Building an Agent with LangGraph**

LangGraph is a state-based graph framework from the LangChain ecosystem. It uses `StateGraph` to define the agent's execution flow as a graph. Each node is a processing step, and edges represent the direction of flow.

**Direct Implementation of A2A Protocol**

To create an A2A server without `to_a2a`, you need to implement the following two endpoints:

1. **`GET /.well-known/agent-card.json`**: An endpoint that returns the Agent Card. It provides the agent's metadata in JSON format.
2. **`POST /messages`**: An endpoint that receives messages and returns responses. It accepts and processes requests in JSON-RPC format.

**Agent Card Structure**

The Agent Card serves as the agent's "business card" in the A2A protocol:

| Field | Description |
|-------|-------------|
| `name` | Agent name |
| `description` | Agent description |
| `url` | URL to send messages to |
| `protocolVersion` | A2A protocol version |
| `capabilities` | Agent capabilities |
| `defaultInputModes` | Supported input formats |
| `defaultOutputModes` | Supported output formats |
| `skills` | List of agent skills |
| `preferredTransport` | Preferred transport method |

**SendMessageResponse Structure**

In the A2A protocol, message responses follow the JSON-RPC format. The `result` object in the response contains fields such as `kind`, `message_id`, `role`, and `parts`.

#### Code Analysis

**LangGraph Graph Definition**

```python
# a2a/langraph_agent/graph.py
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph


llm = init_chat_model("openai:gpt-4o")


class ConversationState(MessagesState):
    pass


def call_model(state: ConversationState) -> ConversationState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(ConversationState)
graph_builder.add_node("llm", call_model)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()
```

Key points:
- `init_chat_model("openai:gpt-4o")`: LangChain's unified model initialization function. It uses the `provider:model_name` format (note the difference from LiteLlm's `provider/model_name`).
- `ConversationState(MessagesState)`: Inherits from `MessagesState` to manage the conversation message list as state.
- `call_model`: A node function that calls the LLM and adds the response to the message list.
- The graph structure is simple: `START → llm → END`. When a user message comes in, it calls the LLM and terminates.
- `graph_builder.compile()` creates an executable graph object.

**A2A Server Implementation (FastAPI)**

```python
# a2a/langraph_agent/server.py
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from graph import graph

app = FastAPI()


def run_graph(message: str):
    result = graph.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content


@app.get("/.well-known/agent-card.json")
def get_agent_card():
    return {
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "An agent that can help students with their philosophy homework",
        "name": "PhilosophyHelperAgent",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "skills": [
            {
                "description": "An agent that can help students with their philosophy homework",
                "id": "PhilosophyHelperAgent",
                "name": "model",
                "tags": ["llm"],
            },
        ],
        "supportsAuthenticatedExtendedCard": False,
        "url": "http://localhost:8002/messages",
        "version": "0.0.1",
    }


@app.post("/messages")
async def handle_message(req: Request):
    body = await req.json()
    messages = body.get("params").get("message").get("parts")
    messages.reverse()
    message_text = ""
    for message in messages:
        text = message.get("text")
        message_text += f"{text}\n"
    response = run_graph(message_text)
    return {
        "id": "message_1",
        "jsonrpc": "2.0",
        "result": {
            "kind": "message",
            "message_id": "239827493847289374",
            "role": "agent",
            "parts": [
                {"kind": "text", "text": response},
            ],
        },
    }
```

Key points:

1. **`run_graph` function**:
   - A wrapper function that calls the LangGraph graph.
   - Converts the user message into `{"role": "user", "content": message}` format and passes it to the graph.
   - Extracts and returns the `content` of the last message (the LLM's response) from the result.

2. **Agent Card endpoint** (`GET /.well-known/agent-card.json`):
   - `protocolVersion: "0.3.0"`: Uses A2A protocol version 0.3.0.
   - `url: "http://localhost:8002/messages"`: Specifies the URL to receive messages. `RemoteA2aAgent` sends messages to this URL.
   - `skills`: Defines the list of skills this agent provides.
   - `preferredTransport: "JSONRPC"`: Indicates communication via JSON-RPC.

3. **Message handling endpoint** (`POST /messages`):
   - Parses the JSON-RPC request of the A2A protocol.
   - Request structure: Extracts message parts from the `body.params.message.parts` path.
   - `messages.reverse()`: Reverses the order of message parts so the latest message comes first.
   - Extracts `text` from each part and combines them into a single string.
   - The response follows the JSON-RPC format: `jsonrpc: "2.0"`, with the agent's response contained in the `result` object.
   - `result.parts` is in A2A message part format, containing `kind: "text"` and the actual text.

**Adding Philosophy Agent to Root Agent**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py (added portion)
philosophy_agent = RemoteA2aAgent(
    name="PhilosophyHelperAgent",
    description="An agent that can help students with their philosophy homework",
    agent_card=f"http://127.0.0.1:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
        philosophy_agent,
    ],
)
```

Key points:
- The philosophy agent runs on port 8002.
- The root agent's `sub_agents` now contains two remote agents.
- The root agent automatically delegates to either the history agent (port 8001) or the philosophy agent (port 8002) based on the content of the question.
- **Key takeaway**: The history agent was built with ADK + `to_a2a`, and the philosophy agent was directly implemented with LangGraph + FastAPI. Despite using different frameworks, they communicate in the same way thanks to the A2A protocol.

#### Practice Points

1. Open three terminals and run all agents simultaneously:
   ```bash
   # Terminal 1: History agent (ADK-based)
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # Terminal 2: Philosophy agent (LangGraph-based)
   cd a2a/langraph_agent
   uvicorn server:app --port 8002

   # Terminal 3: Root agent
   cd a2a
   adk web user-facing-agent
   ```
2. Try various questions in the ADK web UI:
   - "What were the causes of World War II?" → Delegated to history agent
   - "What is Socrates' philosophy?" → Delegated to philosophy agent
   - "How's the weather today?" → Root agent responds directly
3. Access `http://localhost:8002/.well-known/agent-card.json` to view the directly implemented Agent Card.
4. Observe the logs of each server to trace the actual A2A message delivery process.

---

## 3. Chapter Key Summary

### Core Principles of the A2A Protocol

| Concept | Description |
|---------|-------------|
| **A2A Protocol** | An open standard that enables agents from different frameworks to communicate over a network |
| **Agent Card** | A JSON document containing the agent's metadata (`/.well-known/agent-card.json`) |
| **to_a2a** | A utility function that automatically converts an ADK agent into an A2A server |
| **RemoteA2aAgent** | An ADK class that allows using agents on remote A2A servers as if they were local sub-agents |
| **JSON-RPC** | The communication protocol used for A2A message exchange |

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│                      User                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Root Agent (StudentHelperAgent)               │
│            - Google ADK + LiteLlm                       │
│            - Runs with adk web                          │
└──────────┬──────────────────────────────┬───────────────┘
           │ A2A (port 8001)              │ A2A (port 8002)
           ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────┐
│ History Agent         │    │ Philosophy Agent          │
│ (HistoryHelperAgent) │    │ (PhilosophyHelperAgent)  │
│ - Google ADK         │    │ - LangGraph + FastAPI    │
│ - Uses to_a2a        │    │ - Direct A2A impl.       │
│ - port 8001          │    │ - port 8002              │
└──────────────────────┘    └──────────────────────────┘
```

### 5 Key Takeaways

1. **Framework Independence**: Using the A2A protocol, agents built with different frameworks such as Google ADK, LangGraph, and LangChain can collaborate as a single system.

2. **Agent Card is Essential**: Every A2A agent must expose its metadata at `/.well-known/agent-card.json`. Through this, other agents can automatically obtain connection information.

3. **Easy Conversion with `to_a2a`**: If using Google ADK, you can convert an agent into an A2A server with a single line of `to_a2a()`. The Agent Card and message handling endpoints are automatically generated.

4. **Direct Implementation is Also Possible**: Since the A2A protocol is based on standard HTTP + JSON-RPC, it can be directly implemented with any web framework. You only need to implement the Agent Card endpoint and the message handling endpoint.

5. **Description is the Key to Routing**: The root agent decides task delegation based on each sub-agent's `description`. Clear and specific descriptions ensure accurate routing.

---

## 4. Practice Exercises

### Exercise 1: Adding a New Specialist Agent (Basic)

Create a `MathHelperAgent` that helps with math homework using Google ADK + `to_a2a`, configured to run on port 8003. Add it to the root agent's `sub_agents` so that three specialist agents collaborate.

**Requirements:**
- Agent name: `MathHelperAgent`
- Port: 8003
- Add at least 1 math-related tool (e.g., calculator function)
- Verify that math questions are delegated to this agent from the root agent

### Exercise 2: Extending the LangGraph-based A2A Server (Intermediate)

Extend the LangGraph-based philosophy agent created in section 20.5 to add the following features:

**Requirements:**
- Conversation history management: Implement the ability to remember previous conversation content across multiple message exchanges
- Error handling: Return appropriate error responses for malformed request formats
- Utilize the `capabilities` field in the Agent Card to specify supported features

**Hints:**
- You can use a dictionary to store conversation history per session
- Refer to the JSON-RPC error response format

### Exercise 3: Building an A2A Agent with a Completely Different Framework (Advanced)

Implement an A2A agent using only FastAPI and direct HTTP calls (e.g., calling the OpenAI API directly with `httpx`), without using any AI framework. Prove that the A2A protocol is truly framework-independent.

**Requirements:**
- Do not use Google ADK, LangChain, LangGraph, etc.
- Call the OpenAI API directly with FastAPI + `httpx` (or `requests`)
- Directly implement the Agent Card and message handling endpoints
- Verify correct operation by connecting with the root agent's `RemoteA2aAgent`

**Hint:**
```python
import httpx

async def call_openai(message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]
```

### Exercise 4: Building an Agent Card Explorer (Advanced)

Create a CLI tool that fetches an Agent Card from a given URL, displays it in a readable format, and sends test messages to that agent.

**Requirements:**
- Run in the format `python explorer.py http://localhost:8001`
- Display all fields of the Agent Card in a readable format
- Accept user input and send A2A messages to the agent, then display the response
- Handle requests/responses in JSON-RPC format

---

> **Note**: All code in this chapter is located in the `a2a/` directory. You must set the required API keys in the `.env` file before running. Install dependencies with `uv sync`, then run each agent in a separate terminal.
