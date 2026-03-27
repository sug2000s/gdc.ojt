# Chapter 14: Building a LangGraph Chatbot

## Chapter Overview

In this chapter, we learn how to build a practical AI chatbot step by step using **LangGraph**. LangGraph is a core library in the LangChain ecosystem that enables the design of complex AI agent workflows through **Stateful Graph** patterns.

The key themes running throughout this chapter are:

- **Basic Chatbot Structure**: Building the chatbot skeleton using StateGraph and MessagesState
- **Tool Integration**: Patterns for LLM to call external tools and utilize their results
- **Memory**: Persisting conversation state through an SQLite-based checkpointer
- **Human-in-the-loop**: Interrupt patterns for integrating human feedback into the workflow
- **Time Travel**: Exploring state history and branching execution through forks
- **DevTools**: Transitioning to a production structure for LangGraph Studio

Each section builds upon the code from the previous section, ultimately completing a full agent system that includes tool calling, memory, human intervention, and state management.

---

## 14.0 LangGraph Chatbot (Basic Structure)

### Topic and Objectives

The goal of this section is to create the most basic chatbot structure in LangGraph. We strip away the complex routing graphs based on `StateGraph`, `Command`, and `TypedDict` from previous chapters and completely restructure it into a **simple message-based chatbot**.

### Core Concepts

#### What is MessagesState?

LangGraph provides a predefined state class called `MessagesState`. It is a state management tool optimized for chatbot development, internally containing a list field called `messages`. This list automatically accumulates various LangChain message types such as `HumanMessage`, `AIMessage`, and `ToolMessage`.

The biggest difference from the previous approach of defining state directly with `TypedDict` is that `MessagesState` supports **append** behavior by default. That is, when a node returns `{"messages": [new_message]}`, the new message is appended to the existing message list.

#### init_chat_model

The `init_chat_model` function from `langchain.chat_models` allows initializing various LLM providers through a unified interface. You simply pass a string in the format `provider:model_name`, such as `"openai:gpt-4o-mini"`.

### Code Analysis

#### Step 1: Imports and LLM Initialization

```python
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph.message import MessagesState

llm = init_chat_model("openai:gpt-4o-mini")
```

We remove `TypedDict` and `Command` used in previous code and instead import `MessagesState` and `init_chat_model`. `init_chat_model` is LangChain's universal chat model initialization function, where the provider and model name are separated by a colon.

#### Step 2: State Definition

```python
class State(MessagesState):
    custom_stuff: str

graph_builder = StateGraph(State)
```

We define our own `State` class by inheriting from `MessagesState`. Since `MessagesState` already includes the `messages` field, we only need to declare additional fields as needed (here, `custom_stuff`). This pattern is the standard state definition approach for LangGraph chatbots.

#### Step 3: Chatbot Node Definition

```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

This is the core chatbot node function. The operation is straightforward:
1. Extract the `messages` list from the current state.
2. Pass the entire message history to the LLM to generate a response.
3. Return the generated response appended to the `messages` list.

Thanks to `MessagesState`'s reducer logic, the `messages` list in the return value is **appended** to the existing state messages.

#### Step 4: Graph Construction and Execution

```python
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "how are you?"},
        ]
    }
)
```

The graph structure is very simple:
- `START` -> `chatbot` -> `END`

We pass the initial message in dictionary form to `graph.invoke()`. `{"role": "user", "content": "how are you?"}` is automatically converted to a `HumanMessage` object by LangChain.

The execution result returns a state dictionary containing `HumanMessage` and `AIMessage`.

### Practice Points

- Experiment with defining state directly using `TypedDict` instead of inheriting from `MessagesState` and see what differences arise.
- Try changing the `init_chat_model` provider to `"anthropic:claude-sonnet-4-20250514"` etc. and run the same graph with different LLMs.
- Devise a method to dynamically inject a system prompt using the `custom_stuff` field.

---

## 14.1 Tool Nodes

### Topic and Objectives

In this section, we connect **external tools** to the chatbot. We implement the **ReAct (Reasoning + Acting) pattern** where the LLM calls tools based on user requests and generates final responses using the tool results.

### Core Concepts

#### Tool Calling Mechanism

Modern LLMs like OpenAI's gpt-4o-mini support **function calling (tool calling)**. When you inform the LLM of available tools, it selects the appropriate tool based on the user's question, constructs the arguments, and generates a call request. The LLM does not execute the tool directly -- it expresses the **intent** of "please call this tool with these arguments."

#### ToolNode and tools_condition

The `langgraph.prebuilt` module provides two key utilities:

- **`ToolNode`**: A pre-built node responsible for tool execution. It detects `tool_calls` returned by the LLM, executes the corresponding tools, and returns results as `ToolMessage`.
- **`tools_condition`**: A conditional routing function. If the LLM response contains `tool_calls`, it routes to the `"tools"` node; otherwise, it routes to `END`.

#### Conditional Edges

`add_conditional_edges` adds edges that dynamically determine the next node based on the previous node's output. When used with `tools_condition`, it naturally implements a flow where the graph moves to the tool node only when the LLM requests a tool call, and otherwise ends the conversation.

### Code Analysis

#### Step 1: New Imports

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
```

`ToolNode` and `tools_condition` come from LangGraph's prebuilt module, and the `@tool` decorator comes from `langchain_core.tools`.

#### Step 2: Tool Definition

```python
@tool
def get_weather(city: str):
    """Gets weather in city"""
    return f"The weather in {city} is sunny."
```

The `@tool` decorator converts a regular Python function into a LangChain tool. The function's **docstring** becomes the tool's description, and the function's **parameter type hints** are automatically converted to the tool's input schema. The LLM uses this information to decide whether to call the tool and what arguments to pass.

#### Step 3: Binding Tools to LLM

```python
llm_with_tools = llm.bind_tools(tools=[get_weather])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

The `bind_tools()` method informs the LLM of the available tool list. Now the LLM can return tool call requests (`tool_calls`) instead of text. The chatbot node uses `llm_with_tools` instead of the plain `llm`.

#### Step 4: Graph Restructuring

```python
tool_node = ToolNode(
    tools=[get_weather],
)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()
```

The graph structure changes significantly:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

Key changes:
1. Add `ToolNode` as the `"tools"` node.
2. Instead of a direct connection from `chatbot` to `END`, set up **conditional routing** with `add_conditional_edges`.
3. Add an edge from `"tools"` to `"chatbot"` to form a **loop**.

Thanks to this structure, the LLM can repeatedly call tools as many times as needed. After receiving tool results, it returns to the `chatbot` node to generate the final response or call additional tools.

#### Step 5: Verifying the Execution Flow

```python
graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "what is the weather in machupichu"},
        ]
    }
)
```

Looking at the message flow in the execution result:

1. `HumanMessage`: "what is the weather in machupichu"
2. `AIMessage` (containing tool_calls): Request to call `get_weather(city="machupichu")`
3. `ToolMessage`: "The weather in machupichu is sunny."
4. `AIMessage`: "The weather in Machu Picchu is sunny." (final response)

You can see that the LLM organizes the tool call result into natural language and delivers it to the end user.

### Practice Points

- Register multiple tools simultaneously and verify whether the LLM selects the appropriate tool for each situation.
- Experiment with changing the tool's docstring and observe how the LLM's tool selection behavior changes.
- Replace `tools_condition` with a custom condition function to implement more complex routing logic.

---

## 14.2 Memory

### Topic and Objectives

In this section, we add **persistent memory** to the chatbot. A basic LangGraph graph loses its state after an `invoke()` call ends, but by using a **Checkpointer**, you can save conversation state to a database and maintain previous context in subsequent conversations.

### Core Concepts

#### Checkpointer

LangGraph's checkpointer is a mechanism that automatically saves state at **each step** of graph execution. This enables:

- **Conversation persistence**: When calling `invoke()` multiple times with the same `thread_id`, the previous conversation context is maintained.
- **State history**: All intermediate steps of graph execution can be queried later.
- **Error recovery**: Even if an error occurs during execution, you can resume from the last checkpoint.

#### SqliteSaver

`SqliteSaver` is a checkpointer implementation that uses an SQLite database as its backend. While PostgreSQL-based checkpointers are recommended for production environments, SQLite is convenient for development and learning purposes.

#### thread_id and config

When using a checkpointer, you must specify a `thread_id` in the `config`. The `thread_id` is an identifier that distinguishes conversation sessions -- using the same `thread_id` continues the same conversation, while using a different `thread_id` starts a new one.

#### Async Streaming

This section also introduces the async streaming pattern using `graph.astream()` instead of `graph.invoke()`. Setting `stream_mode="updates"` delivers events as each node's execution result occurs, allowing you to monitor the graph execution process in real time.

### Code Analysis

#### Step 1: SQLite Connection and Checkpointer Setup

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(
    "memory.db",
    check_same_thread=False,
)
```

This creates an SQLite database connection. `check_same_thread=False` is a setting for safely using SQLite in a multi-threaded environment. This option is necessary for operation in Jupyter notebook's async event loop.

Additionally, the `aiosqlite` dependency has been added to `pyproject.toml`:

```toml
dependencies = [
    "aiosqlite>=0.21.0",
    ...
]
```

This is a package for supporting async operations of `SqliteSaver`.

#### Step 2: Connecting the Checkpointer to the Graph

```python
graph = graph_builder.compile(
    checkpointer=SqliteSaver(conn),
)
```

Simply passing the `checkpointer` parameter to the `compile()` method activates the memory feature. All subsequent graph executions will automatically have their state saved to SQLite.

#### Step 3: Streaming Execution

```python
async for event in graph.astream(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the weather in berlin, budapest and bratislava.",
            },
        ]
    },
    stream_mode="updates",
):
    print(event)
```

`astream()` is an async generator that streams each node's execution result in real time. Using `stream_mode="updates"` efficiently delivers only the **changed parts** as events rather than the entire state.

Activating the commented-out `config` section allows you to specify a `thread_id` to continue a conversation:

```python
config={
    "configurable": {
        "thread_id": "2",
    },
},
```

#### Step 4: Querying State History

```python
for state in graph.get_state_history(
    {
        "configurable": {
            "thread_id": "2",
        },
    }
):
    print(state.next)
```

`get_state_history()` returns all state snapshots for a specific thread in reverse chronological order. Each snapshot contains the state values at that point and information about the next node to be executed (`next`). This feature forms the foundation for the time travel feature in the next section.

### Practice Points

- Verify that conversations actually continue by calling `invoke()` multiple times with the same `thread_id`.
- Confirm that calling with a different `thread_id` starts a completely new conversation.
- Analyze the results of `get_state_history()` to track how the state changes after each node execution.
- Compare how the output differs when changing `stream_mode` to `"values"`.

---

## 14.3 Human-in-the-loop

### Topic and Objectives

In this section, we implement one of LangGraph's most powerful features: the **Human-in-the-loop** pattern. Instead of the AI processing everything automatically, this pattern receives **human judgment or feedback** midway to continue the workflow.

We use LangGraph's `interrupt` function and `Command` class for this purpose.

### Core Concepts

#### interrupt Function

`interrupt()` is a function that **pauses** graph execution. When called inside a tool node, graph execution stops and the current state is saved to the checkpointer. The developer can send a question to the user at the point of interruption and, after receiving the user's response, resume execution with `Command(resume=...)`.

```python
feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
return feedback
```

The value passed to `interrupt()` becomes the message shown to the user at interruption, and the value passed via `Command(resume=...)` becomes the return value of `interrupt()`.

#### Command Class

`Command` is a command object for resuming a paused graph. When you pass the user's response in the `resume` parameter and provide it to `graph.invoke()`, the graph continues from the point of interruption with the response becoming the return value of `interrupt()`.

#### State Snapshot and next

You can query the current state snapshot of the graph through `graph.get_state(config)`. The snapshot's `next` attribute returns the name of the next node to be executed as a tuple.
- `('tools',)`: Interrupted at the tool node, waiting
- `()`: Graph execution is complete

### Code Analysis

#### Step 1: Importing interrupt and Command

```python
from langgraph.types import interrupt, Command
```

Import two key classes from LangGraph's `types` module.

#### Step 2: Defining the Human Feedback Tool

```python
@tool
def get_human_feedback(poem: str):
    """
    Asks the user for feedback on the poem.
    Use this before returning the final response.
    """
    feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
    return feedback
```

The operation of this tool is key:
1. The LLM generates a poem and calls this tool.
2. When `interrupt()` is called, graph execution pauses.
3. When the user provides feedback, `interrupt()` returns that feedback.
4. The feedback is delivered to the LLM as a `ToolMessage`.

#### Step 3: Chatbot Node with System Prompt

```python
def chatbot(state: State):
    response = llm_with_tools.invoke(
        f"""
        You are an expert in making poems.

        Use the `get_human_feedback` tool to get feedback on your poem.

        Only after you receive positive feedback you can return the final poem.

        ALWAYS ASK FOR FEEDBACK FIRST.

        Here is the conversation history:

        {state["messages"]}
    """
    )
    return {
        "messages": [response],
    }
```

The system prompt gives the LLM clear instructions:
- Role as a poetry expert
- Must use the `get_human_feedback` tool to receive feedback
- Only return the final poem after receiving positive feedback

This prompt design determines the success of the Human-in-the-loop pattern.

#### Step 4: First Execution (Until Interruption)

```python
config = {
    "configurable": {
        "thread_id": "3",
    },
}

result = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "Please make a poem about Python code."},
        ]
    },
    config=config,
)
```

When executed, the LLM generates a poem and calls the `get_human_feedback` tool. The graph is interrupted by `interrupt()`, and the conversation state up to this point is saved to the checkpointer.

#### Step 5: Checking State

```python
snapshot = graph.get_state(config)
snapshot.next  # ('tools',)
```

When `next` returns `('tools',)`, it means the graph is interrupted at the tool node and waiting for the user's response.

#### Step 6: Providing Feedback and Resuming

```python
response = Command(resume="It looks great!")

result = graph.invoke(
    response,
    config=config,
)
for message in result["messages"]:
    message.pretty_print()
```

When positive feedback is passed with `Command(resume="It looks great!")`:
1. `interrupt()` returns `"It looks great!"`.
2. This value is delivered to the LLM as a `ToolMessage`.
3. The LLM confirms the positive feedback and returns the final poem.

In the actual output, you can see the full conversation flow:
- First poem generated -> Feedback "It is too long! Make shorter." -> Shorter version generated -> Feedback "It looks great!" -> Final poem returned

#### Step 7: Completion Verification

```python
snapshot = graph.get_state(config)
snapshot.next  # ()
```

An empty tuple `()` indicates that graph execution is complete.

### Practice Points

- Observe how the LLM reacts when negative feedback is provided multiple times in succession.
- Compare with a version where the tool automatically returns feedback without `interrupt()` to feel the difference of Human-in-the-loop.
- Design a complex workflow with multiple interrupt points (e.g., review -> approval -> deployment pipeline).
- Experiment with passing structured data (dictionaries) to `Command(resume=...)`.

---

## 14.4 Time Travel

### Topic and Objectives

In this section, we learn the time travel feature that uses the **state history** saved by LangGraph's checkpointer to go back to a specific point in the past and **fork** graph execution.

This feature is used in various practical scenarios such as debugging, A/B testing, and user experience rollback.

### Core Concepts

#### State History

A graph with an active checkpointer saves a state snapshot **after every node execution**. Calling `get_state_history(config)` retrieves all snapshots for a specific thread in reverse chronological order. Each snapshot contains:

- `values`: The complete state at that point (message list, etc.)
- `next`: The next node to be executed
- `config`: Configuration identifying the snapshot (including checkpoint_id)

#### State Fork

Using `graph.update_state()`, you can create a **new branch** based on a specific past checkpoint. For example, going back to the point where the user said "I live in Valencia" and changing it to "I live in Zagreb" causes the LLM to generate responses based on Zagreb in the new branch.

#### checkpoint_id

Each state snapshot has a unique `checkpoint_id`. Including this ID in the `config` when calling `graph.invoke()` allows you to resume graph execution from that checkpoint.

### Code Analysis

In this section, we remove all previous tool calls and Human-in-the-loop and return to a simple chatbot structure. This is to focus on the time travel concept itself.

#### Step 1: Simplified Chatbot

```python
class State(MessagesState):
    pass

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
    }

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(
    checkpointer=SqliteSaver(conn),
)
```

A simple chatbot with no tool nodes, only memory activated. `State` directly inherits from `MessagesState` with no additional fields.

#### Step 2: Running the Conversation

```python
config = {
    "configurable": {
        "thread_id": "0_x",
    },
}

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I live in Europe now. And the city I live in is Valencia.",
            },
        ]
    },
    config=config,
)
```

The user says "I live in Europe, and my city is Valencia." The LLM generates a response to this.

#### Step 3: Exploring State History

```python
state_history = graph.get_state_history(config)

for state_snapshot in list(state_history):
    print(state_snapshot.next)
    print(state_snapshot.values["messages"])
    print("=========\n")
```

We iterate through all state snapshots, printing the `next` node and message content at each point. This allows you to understand the entire flow of graph execution in chronological order.

#### Step 4: Selecting the Fork Point

```python
state_history = graph.get_state_history(config)
to_fork = list(state_history)[-5]
to_fork.config
```

We select a snapshot at a specific index from the state history. This snapshot's `config` contains the `checkpoint_id` for that point in time.

#### Step 5: Modifying State (Creating a Fork)

```python
from langchain_core.messages import HumanMessage

graph.update_state(
    to_fork.config,
    {
        "messages": [
            HumanMessage(
                content="I live in Europe now. And the city I live in is Zagreb.",
                id="25169a3d-cc86-4a5f-9abd-03d575089a9f",
            )
        ]
    },
)
```

Key points of `update_state()`:
- **First argument**: The `config` of the point to fork from (including checkpoint_id)
- **Second argument**: The state values to modify
- **Message ID**: Specifying the same ID **replaces** the existing message; using a new ID **appends** the message.

Here we replace "Valencia" with "Zagreb" to create a new branch.

#### Step 6: Resuming Execution from the Forked State

```python
result = graph.invoke(
    None,
    {
        "configurable": {
            "thread_id": "0_x",
            "checkpoint_ns": "",
            "checkpoint_id": "1f08d808-b408-6ca2-8004-f964cbac5a14",
        }
    },
)

for message in result["messages"]:
    message.pretty_print()
```

In `graph.invoke(None, config)`:
- When the first argument is `None`, execution resumes from the existing state without new input.
- Specifying a specific `checkpoint_id` in the `config` starts from that checkpoint.

This causes the LLM to generate a new response based on the modified message "I live in Zagreb."

### Practice Points

- After having several conversations, experiment with going back to a midpoint and asking different questions.
- Verify the case where setting a different message ID in `update_state()` results in appending instead of replacing.
- Create multiple different forks from the same checkpoint to implement an A/B testing scenario.
- Analyze how forked states are recorded in the history using `get_state_history()`.

---

## 14.5 DevTools

### Topic and Objectives

In this section, we convert the Jupyter notebook-based prototype to a **production structure** and configure it for use with **LangGraph Studio** (LangGraph DevTools).

LangGraph Studio is a development tool that provides visual debugging, state tracking, time travel, and more through a GUI.

### Core Concepts

#### From Jupyter to Python Script

While Jupyter notebooks are useful for rapid prototyping during development, you need to convert to standard Python scripts (`.py`) for actual deployment and DevTools integration. In this process:

- Consolidate notebook cell-based code into a single script
- Recombine all features from previous sections: Human-in-the-loop, tool calling, etc.
- Export the `graph` object at the module level so it can be referenced externally

#### langgraph.json Configuration File

`langgraph.json` is the configuration file for LangGraph Studio to recognize the project. This file defines dependencies, environment variables, and graph entry points.

### Code Analysis

#### Step 1: langgraph.json Configuration

```json
{
    "dependencies": [
        "langchain_openai",
        "./main.py"
    ],
    "env": "./.env",
    "graphs": {
        "mr_poet": "./main.py:graph"
    }
}
```

Meaning of each field:
- **`dependencies`**: Packages and modules needed by the project. `langchain_openai` is the OpenAI integration package, and `./main.py` is the script file where the graph is defined.
- **`env`**: Path to the environment variables file. Loads secret information such as OpenAI API keys from the `.env` file.
- **`graphs`**: List of graphs to expose to DevTools. Registers the `graph` variable from `main.py` under the name `"mr_poet"`.

#### Step 2: main.py - Integrated Production Code

```python
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
```

All necessary modules are imported. All features learned in previous sections (tools, checkpointer, interrupt) are integrated into a single file.

#### Step 3: Tool Definition (Including Human-in-the-loop)

```python
@tool
def get_human_feedback(poem: str):
    """
    Get human feedback on a poem.
    Use this to get feedback on a poem.
    The user will tell you if the poem is ready or if it needs more work.
    """
    response = interrupt({"poem": poem})
    return response["feedback"]

tools = [get_human_feedback]
```

The Human-in-the-loop tool from Section 14.3 has been slightly improved. A dictionary is passed to `interrupt()`, and the response is also received in dictionary form (`response["feedback"]`), enabling structured data exchange.

#### Step 4: LLM and State Definition

```python
llm = init_chat_model("openai:gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class State(MessagesState):
    pass
```

#### Step 5: Chatbot Node (With System Prompt)

```python
def chatbot(state: State) -> State:
    response = llm_with_tools.invoke(
        f"""
    You are an expert at making poems.

    You are given a topic and need to write a poem about it.

    Use the `get_human_feedback` tool to get feedback on your poem.

    Only after the user says the poem is ready, you should return the poem.

    Here is the conversation history:
    {state['messages']}
    """
    )
    return {
        "messages": [response],
    }
```

A system prompt similar to Section 14.3 is used, with the addition of a type hint (`-> State`) for improved code clarity.

#### Step 6: Graph Construction

```python
tool_node = ToolNode(
    tools=tools,
)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

graph = graph_builder.compile(name="mr_poet")
```

Final graph structure:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

Notable points:
- `compile(name="mr_poet")` assigns a name to the graph. This name is linked to the `graphs` field in `langgraph.json`.
- A SQLite-based checkpointer is set up through the `memory.db` file.
- The `graph` variable is defined at the module level, making it referenceable externally via `./main.py:graph`.

### Practice Points

- Try running LangGraph Studio with the `langgraph dev` (or `langgraph up`) command.
- Visually inspect the graph structure in LangGraph Studio and trace each node's execution process.
- Use Studio's time travel feature to go back to past states, modify them, and create new branches.
- Experiment with registering multiple graphs in `langgraph.json` to manage them simultaneously.

---

## Chapter Key Summary

### 1. Basic Structure of LangGraph Chatbot
- Manage message-based conversations with a state class inheriting from `MessagesState`.
- Define nodes and edges with `StateGraph` and create an executable graph with `compile()`.
- Use various LLM providers through a unified interface with `init_chat_model()`.

### 2. Tool Integration Pattern
- Convert Python functions to LangChain tools with the `@tool` decorator.
- Bind tools to the LLM with `llm.bind_tools()` and automate tool execution flow with `ToolNode` and `tools_condition`.
- Dynamically route based on whether the LLM calls tools through conditional edges (`add_conditional_edges`).

### 3. Memory and Checkpointer
- Activate state persistence by passing `SqliteSaver` to `compile(checkpointer=...)`.
- Distinguish conversation sessions with `thread_id`; using the same ID continues the conversation.
- Real-time streaming is possible with `astream(stream_mode="updates")`.

### 4. Human-in-the-loop
- Pause graph execution and wait for user input with the `interrupt()` function.
- Deliver the user's response with `Command(resume=...)` to resume execution.
- Check the current pause state with `get_state(config).next`.

### 5. Time Travel
- Query all state snapshots with `get_state_history(config)`.
- Modify past states to create forks with `update_state(checkpoint_config, new_values)`.
- Resume execution from a specific checkpoint with `graph.invoke(None, checkpoint_config)`.
- Specifying the same message ID replaces it; specifying a different ID appends it.

### 6. DevTools Integration
- Define project settings with `langgraph.json`.
- Convert from Jupyter notebook to Python script to adopt a production structure.
- Perform visual debugging, state tracking, and time travel via GUI through LangGraph Studio.

---

## Practice Assignments

### Assignment 1: Multi-Tool Chatbot (Basic)

Implement a chatbot with 3 or more tools such as weather, exchange rate, and news search. The LLM should select the appropriate tool based on the user's question and, when necessary, call multiple tools sequentially to generate the final response.

**Requirements:**
- Define at least 3 `@tool` functions
- Graph construction using `ToolNode` and `tools_condition`
- Test scenarios requiring multi-tool calls

### Assignment 2: Approval Workflow (Intermediate)

Implement a 3-stage workflow: document creation -> review -> approval. Each stage requires human approval through Human-in-the-loop before proceeding to the next stage.

**Requirements:**
- Multiple approval points using `interrupt()`
- Logic to return to the previous stage upon rejection
- State persistence through checkpointer

### Assignment 3: Time Travel Debugger (Advanced)

Implement an interactive debugger that explores conversation history, goes back to specific points, and creates branches with different inputs.

**Requirements:**
- Visually display the full history using `get_state_history()`
- Interface for users to select specific checkpoints and modify state
- Fork creation and execution resumption using `update_state()`
- Functionality to compare results between original and forked branches

### Assignment 4: DevTools Deployment (Advanced)

Extend the `main.py` from Section 14.5 to create a structure managing multiple graphs within a single project.

**Requirements:**
- Register 2 or more different graphs in `langgraph.json`
- Each graph uses different tools and system prompts
- Verify and run both graphs in LangGraph Studio
- Environment variable management through `.env` file
