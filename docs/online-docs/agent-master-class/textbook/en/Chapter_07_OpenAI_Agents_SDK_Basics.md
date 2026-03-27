# Chapter 07: OpenAI Agents SDK Basics and Streamlit Integration

---

## 1. Chapter Overview

This chapter covers everything from the basics of building AI agents using the **OpenAI Agents SDK** (`openai-agents`) to practical UI integration. Starting from simple agent creation, we progressively learn streaming event handling, conversation persistence through session memory, agent-to-agent handoffs, structured output, graph visualization, and finally building a web UI with Streamlit.

### Learning Objectives

- Understand the core components of the OpenAI Agents SDK (`Agent`, `Runner`, `function_tool`)
- Learn two methods (high-level/low-level) for handling events in streaming responses
- Implement session-based memory management with `SQLiteSession`
- Design Handoff patterns in multi-agent systems
- Learn structured output using Pydantic `BaseModel` and agent graph visualization
- Understand the basic widgets and Data Flow model of the Streamlit framework

### Project Structure

```
chatgpt-clone/
├── .gitignore
├── .python-version          # Python 3.13.3
├── pyproject.toml           # Project dependency configuration
├── uv.lock                  # uv package manager lock file
├── dummy-agent.ipynb        # Jupyter notebook for agent experimentation
├── main.py                  # Streamlit web application
├── ai-memory.db             # SQLite session memory DB
└── README.md
```

### Key Dependencies

```toml
[project]
name = "chatgpt-clone"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    "python-dotenv>=1.1.1",
    "streamlit>=1.48.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

- **`openai-agents[viz]`**: OpenAI Agents SDK (including visualization extension)
- **`graphviz`**: Visualize agent graphs as SVG
- **`streamlit`**: Web UI framework
- **`python-dotenv`**: Environment variable (.env) management
- **`ipykernel`**: Jupyter notebook kernel (for development)

---

## 2. Detailed Section Explanations

---

### 2.1 Section 7.0 - Introduction (Initial Project Setup)

**Commit**: `fbc2f97`

#### Topic and Objectives

This is the stage where the basic skeleton of the project is established. Initialize a Python 3.13-based project using the `uv` package manager and install the necessary dependencies.

#### Core Concepts

**uv Package Manager**

This project uses `uv` instead of `pip`. `uv` is an ultra-fast Python package manager written in Rust that manages dependencies through `pyproject.toml` and `uv.lock` files. Since `3.13.3` is specified in the `.python-version` file, `uv` automatically uses that Python version.

**Project Initialization Commands (Reference)**

```bash
uv init chatgpt-clone
cd chatgpt-clone
uv add "openai-agents[viz]" python-dotenv streamlit
uv add --dev ipykernel
```

**What is the openai-agents SDK?**

An officially provided agent framework from OpenAI that offers the following core features:

| Component | Description |
|-----------|-------------|
| `Agent` | Class for defining agents. Configure name, instructions, tools, etc. |
| `Runner` | Class for executing agents. Supports synchronous/asynchronous/streaming execution |
| `function_tool` | Decorator that converts Python functions into tools usable by agents |
| `SQLiteSession` | SQLite-based session memory management |
| `ItemHelpers` | Utility for extracting messages from streaming events |

#### Practice Points

1. Install `uv` and initialize the project
2. Examine the dependency structure in `pyproject.toml`
3. Select the `.venv` kernel in Jupyter notebook to verify the development environment

---

### 2.2 Section 7.2 - Stream Events (Streaming Event Handling)

**Commit**: `996dae4`

#### Topic and Objectives

Learn how to process agent responses in real-time through streaming. Both high-level event handling and low-level (raw) event handling approaches are covered.

#### Core Concepts

**Agent and Tool Definition**

First, define a simple agent and tool:

```python
from agents import Agent, Runner, function_tool, ItemHelpers


@function_tool
def get_weather(city: str):
    """Get weather by city"""
    return "30 degrees"


agent = Agent(
    name="Assistant Agent",
    instructions="You are a helpful assistant. Use tools when needed to answer questions",
    tools=[get_weather],
)
```

Key points:
- The `@function_tool` decorator converts a regular Python function into an agent tool
- The function's **docstring** is used as the tool's description -- the agent reads this description to decide when to use the tool
- The function's **type hints** (e.g., `city: str`) are automatically converted into the tool's parameter schema
- Tools are passed to the `Agent` via the `tools` list

**Method 1: High-Level Event Handling (run_item_stream_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        continue
    elif event.type == "agent_updated_stream_event":
        print("Agent updated to", event.new_agent.name)
    elif event.type == "run_item_stream_event":
        if event.item.type == "tool_call_item":
            print(event.item.raw_item.to_dict())
        elif event.item.type == "tool_call_output_item":
            print(event.item.output)
        elif event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))
    print("=" * 20)
```

In this approach, streaming events are categorized into **three categories** for processing:

| Event Type | Description |
|-----------|-------------|
| `raw_response_event` | Unprocessed raw response (ignored in this approach) |
| `agent_updated_stream_event` | Occurs when the currently active agent changes |
| `run_item_stream_event` | Occurs when a run item (message, tool call, etc.) is generated |

Item types within `run_item_stream_event`:

| Item Type | Description |
|----------|-------------|
| `tool_call_item` | When the agent calls a tool |
| `tool_call_output_item` | When a tool execution result is returned |
| `message_output_item` | The agent's text response |

**Method 2: Low-Level Event Handling (raw_response_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

message = ""
args = ""

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        event_type = event.data.type
        if event_type == "response.output_text.delta":
            message += event.data.delta
            print(message)
        elif event_type == "response.function_call_arguments.delta":
            args += event.data.delta
            print(args)
        elif event_type == "response.completed":
            message = ""
            args = ""
```

This approach directly handles `raw_response_event` to implement **token-level real-time streaming**:

| Raw Event Type | Description |
|---------------|-------------|
| `response.output_text.delta` | A token fragment (delta) of the text response |
| `response.function_call_arguments.delta` | A fragment (delta) of tool call arguments |
| `response.completed` | One response has completed |

Looking at the execution results, you can see the tool call arguments being progressively constructed:

```
{"
{"city
{"city":"
{"city":"Madrid
{"city":"Madrid"}
```

Then the text response also accumulates token by token:

```
Hello
Hello!
Hello! I'm
Hello! I'm doing
Hello! I'm doing well
...
Hello! I'm doing well, thank you. The weather in Madrid, the capital of Spain, is currently 30 degrees Celsius. How can I assist you further?
```

#### Comparison of the Two Approaches

| Characteristic | High-Level (run_item) | Low-Level (raw_response) |
|---------------|----------------------|-------------------------|
| Granularity | Item level | Token (delta) level |
| Use Case | Logic processing, state management | Real-time UI updates |
| Complexity | Low | High (requires manual string accumulation) |
| ChatGPT-like UI | Not suitable | Suitable (typing effect) |

#### Practice Points

1. Run both streaming approaches and compare the output differences
2. Understand the logic of accumulating `delta` to reconstruct the full message
3. Think about why `message` and `args` are reset in the `response.completed` event -- because multiple responses can occur in a single execution

---

### 2.3 Section 7.3 - Session Memory

**Commit**: `35a1fe4`

#### Topic and Objectives

Implement session memory using `SQLiteSession` so the agent can remember previous conversations. This enables multi-turn conversations.

#### Core Concepts

**Why Session Memory is Needed**

By default, the agent has no memory of previous conversations each time `Runner.run()` is called. Each call is an independent new conversation. Even if a user says "My name is Nico" and then asks "What was my name?" in the next call, the agent cannot answer.

`SQLiteSession` automatically saves conversation history to a SQLite database and automatically retrieves it on the next call to provide to the agent.

**SQLiteSession Setup**

```python
from agents import Agent, Runner, function_tool, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")
```

- First argument `"user_1"`: **Session identifier**. Using the same identifier shares the same conversation history
- Second argument `"ai-memory.db"`: SQLite database file path

**Conversation Execution with Session**

```python
result = await Runner.run(
    agent,
    "What was my name again?",
    session=session,
)

print(result.final_output)
```

When the `session=session` parameter is passed to `Runner.run()`, the SDK automatically:
1. Loads the previous conversation history for that session from the DB
2. Sends it along with the new user message to the agent
3. Saves the agent's response to the DB

**Checking Session Data**

```python
await session.get_items()
```

This method returns the entire conversation history stored in the session as a list. Examining the structure of the returned data:

```python
[
    {'content': 'Hello how are you? My name is Nico', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'I live in Spain', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What is the weather in the third biggest city of the country i live on', 'role': 'user'},
    {'arguments': '{"city":"Valencia"}', 'name': 'get_weather', 'type': 'function_call', ...},
    {'call_id': '...', 'output': '30 degrees', 'type': 'function_call_output'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What was my name again?', 'role': 'user'},
    {'id': 'msg_...', 'content': [{'text': 'Your name is Nico.', ...}], 'role': 'assistant', ...}
]
```

Notable points:
- The conversation history stores not only user messages and agent responses but also **tool calls (`function_call`) and results (`function_call_output`)**
- For the context-dependent question "the third biggest city of the country I live in," the agent used information from the previous conversation "I live in Spain" to correctly infer **Valencia**
- For the question "What was my name again?", it remembered the information "My name is Nico" from the previous conversation and answered correctly

#### Practice Points

1. Change the session ID of `SQLiteSession` and verify that different conversations are maintained independently
2. Delete the `ai-memory.db` file and confirm that conversation history is reset
3. Use `await session.get_items()` to directly examine the structure of stored conversation history
4. Create context-dependent questions (e.g., "What is the capital of that country?") to test that session memory works properly

---

### 2.4 Section 7.4 - Handoffs (Agent-to-Agent Handoffs)

**Commit**: `b23f34f`

#### Topic and Objectives

Define multiple specialized agents and learn the multi-agent pattern where a main agent analyzes user questions and **hands off (delegates)** to the appropriate specialized agent.

#### Core Concepts

**What is a Handoff?**

A handoff is when one agent transfers control of a conversation to another agent. It is similar to a customer service center where a general representative transfers a call to a specialist. In the OpenAI Agents SDK, this is implemented through the `handoffs` parameter.

**Specialized Agent Definitions**

```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")


geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
)
economics_agent = Agent(
    name="Economics Expert Agent",
    instructions="You are a expert in economics, you answer questions related to them.",
    handoff_description="Use this to answer economics questions.",
)
```

Each specialized agent has two important settings:

| Parameter | Role |
|-----------|------|
| `instructions` | Directive that the agent itself follows. Used as the system prompt when that agent executes |
| `handoff_description` | Description that **other agents (the main agent)** reference when deciding whether to delegate to this agent |

**Main Agent (Orchestrator)**

```python
main_agent = Agent(
    name="Main Agent",
    instructions="You are a user facing agent. Transfer to the agent most capable of answering the user's question.",
    handoffs=[
        economics_agent,
        geaography_agent,
    ],
)
```

- Registers delegatable agents in the `handoffs` list
- The instruction "Transfer to the agent most capable..." explicitly makes the main agent serve as a router
- The main agent compares the user's question content with each agent's `handoff_description` to select the most appropriate agent

**Execution and Result Verification**

```python
result = await Runner.run(
    main_agent,
    "Why do countries sell bonds?",
    session=session,
)

print(result.last_agent.name)
print(result.final_output)
```

Output:
```
Economics Expert Agent
Countries sell bonds as a way to raise funds for various purposes...
```

- `result.last_agent.name` lets you check which agent ultimately generated the response
- For the question "Why do countries sell bonds?", the main agent determined it was an economics-related question and handed off to `Economics Expert Agent`

**Handoff Flow Summary**

```
User: "Why do countries sell bonds?"
    |
    v
[Main Agent] -- Analyzes question --> Determines it's economics-related
    |
    v  (handoff)
[Economics Expert Agent] -- Generates answer --> "Countries sell bonds..."
    |
    v
Response delivered to user
```

#### Practice Points

1. Send geography-related questions ("What is the longest river in the world?") and economics-related questions ("What is inflation?") respectively, and check which agent responds
2. Send an ambiguous question that spans both fields ("What impact does geography have on South Korea's GDP?") and observe which agent is selected
3. Add a new specialized agent (e.g., history expert) to expand handoff targets
4. Modify `handoff_description` and experiment with how routing results change

---

### 2.5 Section 7.5 - Viz and Structured Outputs (Visualization and Structured Output)

**Commit**: `45d261a`

#### Topic and Objectives

Learn how to visualize the structure of an agent system as a graph, and how to force agent output into a predefined structure using Pydantic `BaseModel`.

#### Core Concepts

**Structured Output**

By default, agents return free-form text. However, when responses need to be processed programmatically, a consistent structure is needed. By specifying a Pydantic model in the `output_type` parameter, the agent is forced to return JSON matching that structure.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    background_explanation: str
```

Applying this model to an agent:

```python
geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
    tools=[
        get_weather,
    ],
    output_type=Answer,
)
```

Execution result:
```
answer="The capital of Thailand's northern province, Chiang Mai, is Chiang Mai City."
background_explanation="Chiang Mai is both a city and a province in northern Thailand..."
```

- The agent's response is returned as a structured object with `answer` and `background_explanation` fields, not free text
- This enables programmatic access to specific fields like `result.final_output.answer`

**Agent Graph Visualization**

```python
from agents.extensions.visualization import draw_graph

draw_graph(main_agent)
```

The `draw_graph()` function visualizes the agent system's structure as an **SVG graph**. The generated graph includes the following elements:

| Node | Color | Meaning |
|------|-------|---------|
| `__start__` | Sky blue (ellipse) | Execution start point |
| `__end__` | Sky blue (ellipse) | Execution end point |
| `Main Agent` | Light yellow (rectangle) | Main agent |
| `Economics Expert Agent` | White (rounded rect) | Specialized agent |
| `Geo Expert Agent` | White (rounded rect) | Specialized agent |
| `get_weather` | Light green (ellipse) | Tool (function tool) |

The graph edges (arrows) represent handoff relationships and tool usage relationships:
- **Solid arrows**: Direction of handoffs between agents
- **Dashed arrows**: Call/return relationships between agents and tools

**Dependency Addition**

To use the visualization feature, the `graphviz` package is required:

```toml
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    ...
]
```

#### Practice Points

1. Run `draw_graph(main_agent)` to visualize the agent system structure
2. Add or modify fields in the `Answer` model to customize the output structure
3. Add new specialized agents and see how the graph changes
4. Compare the `result.final_output` type difference between agents with and without `output_type`

---

### 2.6 Section 7.8 - Welcome To Streamlit (Streamlit Basics)

**Commit**: `e763a74`

#### Topic and Objectives

Introduce the Streamlit framework and learn the basics of building web interfaces using various UI widgets. Also briefly covers agent execution tracing through the `trace` feature.

#### Core Concepts

**What is Streamlit?**

Streamlit is a framework that allows you to quickly build web applications using only Python. You can build data visualizations and AI demo apps without HTML, CSS, or JavaScript. Run command:

```bash
streamlit run main.py
```

**Basic Widget Usage**

```python
import streamlit as st
import time


st.header("Hello world!")

st.button("Click me please!")

st.text_input(
    "Write your API KEY",
    max_chars=20,
)

st.feedback("faces")
```

| Widget | Description |
|--------|-------------|
| `st.header()` | Display heading text |
| `st.button()` | Clickable button |
| `st.text_input()` | Text input field |
| `st.feedback()` | Feedback widget (face icons) |

**Sidebar**

```python
with st.sidebar:
    st.badge("Badge 1")
```

Using the `st.sidebar` context manager places widgets in the left sidebar.

**Tab Layout**

```python
tab1, tab2, tab3 = st.tabs(["Agent", "Chat", "Outpu"])

with tab1:
    st.header("Agent")
with tab2:
    st.header("Agent 2")
with tab3:
    st.header("Agent 3")
```

`st.tabs()` creates a tab interface, and content is placed within each tab's context.

**Chat Interface**

```python
with st.chat_message("ai"):
    st.text("Hello!")
    with st.status("Agent is using tool") as status:
        time.sleep(1)
        status.update(label="Agent is searching the web....")
        time.sleep(2)
        status.update(label="Agent is reading the page....")
        time.sleep(3)
        status.update(state="complete")

with st.chat_message("human"):
    st.text("Hi!")


st.chat_input(
    "Write a message for the assistant.",
    accept_file=True,
)
```

| Widget | Description |
|--------|-------------|
| `st.chat_message("ai")` | AI role chat message bubble |
| `st.chat_message("human")` | User role chat message bubble |
| `st.status()` | Progress status display widget (loading state, complete state, etc.) |
| `st.chat_input()` | Chat input field (with file attachment support) |

`st.status()` is a key widget for visually showing users the process of an agent using tools. Labels can be changed in real-time with `status.update()`, and completion status is indicated with `state="complete"`.

**trace Feature (Notebook Side)**

```python
from agents import trace

with trace("user_111111"):
    result = await Runner.run(
        main_agent,
        "What is the capital of Colombia's northen province.",
        session=session,
    )
    result = await Runner.run(
        main_agent,
        "What is the capital of Cambodia's northen province.",
        session=session,
    )
```

Using the `trace()` context manager groups multiple `Runner.run()` calls into a single tracing unit. This is useful for debugging and monitoring agent execution processes in OpenAI's dashboard.

#### Practice Points

1. Run the app with `streamlit run main.py` and check each widget's behavior
2. Change the labels and time intervals of `st.status()` to experiment with UX
3. Try setting `st.chat_message()` roles to custom names beyond "ai" and "human"
4. Add various widgets to the sidebar

---

### 2.7 Section 7.9 - Streamlit Data Flow

**Commit**: `8c438f5`

#### Topic and Objectives

Learn Streamlit's core operating principle, the **Data Flow** model, and **session state (`st.session_state`)** management.

#### Core Concepts

**Streamlit's Data Flow Model**

Streamlit's most important characteristic: **Every time a user interacts with a widget, the entire script (`main.py`) is re-executed from top to bottom.** This is Streamlit's "Data Flow" model.

For example:
1. When a user enters text -> the entire `main.py` is re-executed
2. When a button is clicked -> the entire `main.py` is re-executed
3. When a checkbox is toggled -> the entire `main.py` is re-executed

Because of this characteristic, regular variables within the script are reset every time. Therefore, `st.session_state` must be used to maintain state between interactions.

**State Management with session_state**

```python
import streamlit as st

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

st.header("Hello!")

name = st.text_input("What is your name?")

if name:
    st.write(f"Hello {name}")
    st.session_state["is_admin"] = True


print(st.session_state["is_admin"])
```

**Code Flow Analysis:**

1. **Initial State Check**: If the `"is_admin"` key is not in `st.session_state`, initialize it to `False`. This pattern initializes only on the first run and preserves existing values on subsequent re-runs.

2. **Widget Rendering**: `st.text_input()` displays a text input field and returns the value entered by the user to the `name` variable.

3. **Conditional Processing**: If `name` has a value (is not an empty string), display a greeting message and change `is_admin` to `True`.

4. **State Check**: `print()` outputs the current `is_admin` state to the server console.

**Key Points of session_state:**

```
[First Run]
- st.session_state["is_admin"] = False  (initialized)
- name = ""  (no input yet)
- print(False)

[User enters "Nico" -> entire script re-executes]
- "is_admin" in st.session_state == True  (already exists, skip initialization)
- name = "Nico"
- st.write("Hello Nico")
- st.session_state["is_admin"] = True
- print(True)

[User clears input -> entire script re-executes]
- "is_admin" in st.session_state == True  (still exists)
- st.session_state["is_admin"] was set to True in previous run (preserved!)
- name = ""
- if name: condition not met -> write not executed
- print(True)  <-- still True! session_state persists across re-runs
```

**Regular Variables vs session_state**

| Characteristic | Regular Variables | `st.session_state` |
|---------------|------------------|-------------------|
| On Re-run | Reset to initial value | Previous value preserved |
| Use Case | Temporary calculations | State preservation (conversation history, settings, etc.) |
| Scope | Current execution | Entire browser session |

This concept is very important when building a ChatGPT clone later:
- Conversation history must be stored in `st.session_state` so it persists across re-runs
- Agent instances and session objects are also stored in `st.session_state`

#### Practice Points

1. Run `main.py`, enter a name, and observe the `print()` output in the terminal
2. Enter a name then clear it, and check what happens to the `is_admin` value
3. Store a conversation history list in `st.session_state` and experiment with whether it persists when new messages are added
4. Refresh the browser tab and confirm that `session_state` is reset

---

## 3. Chapter Key Summary

### OpenAI Agents SDK Core Components

| Component | Role | Key Parameters |
|-----------|------|----------------|
| `Agent` | Agent definition | `name`, `instructions`, `tools`, `handoffs`, `output_type`, `handoff_description` |
| `Runner.run()` | Synchronous execution | `agent`, `input`, `session` |
| `Runner.run_streamed()` | Streaming execution | `agent`, `input` |
| `@function_tool` | Tool definition decorator | docstring as description, type hints as schema |
| `SQLiteSession` | Session memory | `session_id`, `db_path` |
| `draw_graph()` | Agent graph visualization | `agent` |
| `trace()` | Execution tracing | `trace_name` |

### Streaming Event Hierarchy

```
stream.stream_events()
├── raw_response_event          # Token-level raw events
│   ├── response.output_text.delta
│   ├── response.function_call_arguments.delta
│   └── response.completed
├── agent_updated_stream_event  # Agent switch events
└── run_item_stream_event       # Item-level events
    ├── tool_call_item
    ├── tool_call_output_item
    └── message_output_item
```

### Multi-Agent Pattern: Handoffs

```
[User Input]
     |
     v
[Main Agent (Orchestrator)]
     |
     ├──(economics question)--> [Economics Expert Agent]
     ├──(geography question)--> [Geo Expert Agent]
     └──(other)--------------> [Direct response or additional agents]
```

### Streamlit Core Principles

1. **Data Flow**: Entire script re-executes on every widget interaction
2. **session_state**: Dictionary for maintaining state across re-executions
3. **Chat UI**: `st.chat_message()`, `st.chat_input()`, `st.status()`

---

## 4. Practice Assignments

### Assignment 1: Build a Basic Agent (Difficulty: 1/3)

**Objective**: Build a basic agent and have it use tools.

- Create a `calculate` tool that performs four arithmetic operations on two numbers
- Use the `@function_tool` decorator to define parameters `operation: str`, `a: float`, `b: float`
- Ask the agent "What is 123 * 456 + 789?" and verify it uses the tool

### Assignment 2: Implement Typing Effect with Streaming (Difficulty: 2/3)

**Objective**: Implement a ChatGPT-like typing effect using low-level streaming events.

- Use `Runner.run_streamed()` and `raw_response_event`
- Each time a `response.output_text.delta` is received, print one character at a time to the terminal (`print(delta, end="", flush=True)`)
- Display tool names and arguments in real-time during tool calls

### Assignment 3: Multi-Turn Conversation Agent (Difficulty: 2/3)

**Objective**: Implement multi-turn conversations using `SQLiteSession`.

- Build an agent that remembers the user's name, favorite color, and favorite food
- In the first call, tell it your name; in the second call, tell it your color; in the third call, request "Summarize all my favorites"
- Verify that the third response includes all previous information

### Assignment 4: Handoff System with 3+ Specialized Agents (Difficulty: 3/3)

**Objective**: Build a routing system with specialized agents for various domains.

- Create at least 3 specialized agents (e.g., science, history, cooking)
- Set `output_type` with structured output for each specialized agent (e.g., `answer`, `confidence_level`, `sources`)
- Visualize the system structure with `draw_graph()`
- Send various questions and test that they route to the correct agent

### Assignment 5: Streamlit ChatGPT Clone UI (Difficulty: 3/3)

**Objective**: Build a ChatGPT-like conversational UI using Streamlit.

- Manage conversation history as a `messages` list in `st.session_state`
- Receive user input with `st.chat_input()` and display conversations with `st.chat_message()`
- Create a "New Chat" button in the sidebar to implement conversation reset functionality
- (Bonus) Use `st.status()` to display "Thinking..." status while the agent is working

---

## Appendix: References

- [OpenAI Agents SDK Official Documentation](https://openai.github.io/openai-agents-python/)
- [Streamlit Official Documentation](https://docs.streamlit.io/)
- [Pydantic Official Documentation](https://docs.pydantic.dev/)
- [uv Package Manager](https://docs.astral.sh/uv/)
