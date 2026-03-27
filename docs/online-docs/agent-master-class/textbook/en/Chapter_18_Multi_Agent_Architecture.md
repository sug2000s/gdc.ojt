# Chapter 18: Multi-Agent Architectures

---

## 1. Chapter Overview

In this chapter, we learn how to design and implement **multi-agent systems where multiple AI agents collaborate** using LangGraph. Going beyond the limitations of a single agent, we progressively build architectures where agents with specialized roles communicate with each other to handle complex tasks.

### Learning Objectives

- Understand the necessity and core concepts of multi-agent systems
- **Network Architecture**: Implement direct peer-to-peer (P2P) communication between agents
- **Supervisor Architecture**: Implement a system where a central coordinator manages agents
- **Supervisor-as-Tools Architecture**: Learn an advanced pattern of encapsulating agents as tools
- **Prebuilt Agents**: Learn how to concisely implement with the `langgraph-supervisor` library
- Learn graph visualization through LangGraph Studio

### Project Structure

```
multi-agent-architectures/
├── .python-version          # Python 3.13
├── pyproject.toml           # Project dependency definitions
├── main.ipynb               # Main practice notebook
├── graph.py                 # Graph definition for LangGraph Studio
├── langgraph.json           # LangGraph Studio configuration
└── uv.lock                  # Dependency lock file
```

### Key Dependencies

```toml
[project]
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "langgraph-supervisor==0.0.29",
    "langgraph-swarm==0.0.14",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]
```

---

## 2. Detailed Section Descriptions

---

### 18.0 Introduction - Project Initialization and Basic Imports

**Topic and Goal**: Establish the foundation for the multi-agent project and import core libraries.

#### Key Concepts

To build a multi-agent system, you need to understand several core modules of LangGraph. In this section, we initialize the project and import all necessary libraries.

#### Code Analysis

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

Let's look at the role of each import:

| Module | Role |
|--------|------|
| `StateGraph` | Core class for creating state-based graphs. Defines nodes and edges to configure the agent's execution flow. |
| `START`, `END` | Special node constants representing the graph's start and end points. |
| `Command` | A command object for controlling agent transitions (handoffs). Specifies the next node with `goto` and state changes with `update`. |
| `MessagesState` | A pre-defined state class that manages message lists. Provides the `messages` key by default. |
| `ToolNode` | A prebuilt node that handles tool calls. When the LLM decides to use a tool, this node handles the actual execution. |
| `tools_condition` | A conditional routing function that determines whether an LLM response contains tool calls. |
| `@tool` | A decorator that converts a regular Python function into a tool that LLMs can call. |
| `init_chat_model` | A utility that initializes various LLMs using a `"provider:model_name"` format string. |

#### Practice Points

- Initialize the project and install dependencies using `uv`: `uv sync`
- `OPENAI_API_KEY` must be set in the `.env` file
- Python 3.13 environment is required

---

### 18.1 Network Architecture

**Topic and Goal**: Implement a **network (P2P) architecture** where agents can directly pass conversations (handoff) to each other.

#### Key Concepts

In a network architecture, **without a central coordinator**, each agent autonomously decides to hand off conversations to other agents. In this example, we create Korean, Greek, and Spanish customer support agents that build a system automatically switching to the agent matching the customer's language.

```
┌──────────────┐     handoff     ┌──────────────┐
│ korean_agent │ ◄─────────────► │ greek_agent  │
└──────┬───────┘                 └──────┬───────┘
       │                                │
       │          handoff               │
       └────────►┌──────────────┐◄──────┘
                 │spanish_agent │
                 └──────────────┘
```

In this structure, each agent:
1. Attempts to respond in the language it understands
2. Detects an unfamiliar language and uses `handoff_tool` to switch to the appropriate agent

#### Code Analysis

**Step 1: State Definition**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str

llm = init_chat_model("openai:gpt-4o")
```

A custom state is defined by extending `MessagesState` to track the currently active agent (`current_agent`) and the agent that performed the transfer (`transfered_by`). This allows knowing which agent is handling the conversation and who transferred it.

**Step 2: Agent Factory Function**

```python
def make_agent(prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node(
        "tools",
        ToolNode(tools=tools),
    )

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    return agent_builder.compile()
```

`make_agent` is an **agent factory function** that allows creating agents with identical structures but different parameters. Each agent internally constitutes a complete `StateGraph` subgraph:

- `agent` node: Passes the prompt and conversation history to the LLM to generate a response
- `tools` node: `ToolNode` executes tool calls
- `tools_condition`: Routes to the `tools` node if the LLM calls a tool, otherwise to `END`
- `tools` -> `agent` edge: Passes tool execution results back to the agent for further reasoning

This structure gives each agent an **independent ReAct (Reasoning + Acting) loop**.

**Step 3: Handoff Tool Definition**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    """
    Handoff to another agent.

    Use this tool when the customer speaks a language that you don't understand.

    Possible values for `transfer_to`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Possible values for `transfered_by`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Args:
        transfer_to: The agent to transfer the conversation to
        transfered_by: The agent that transferred the conversation
    """
    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

`handoff_tool` is the **core mechanism** of the network architecture. Key points:

- Returns a `Command` object to directly control the graph's execution flow
- `goto=transfer_to`: Moves execution to the specified agent node
- `graph=Command.PARENT`: Performs the move in the **parent graph**. Since each agent is a subgraph, without this option, the move would only happen within the subgraph. `Command.PARENT` enables agent transitions at the top-level graph
- `update` refreshes the state to track current agent information
- The tool's docstring specifies possible values to guide the LLM to use correct values

**Step 4: Graph Assembly**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
)

graph_builder.add_edge(START, "korean_agent")

graph = graph_builder.compile()
```

Three agent nodes are registered in the top-level graph. Each node's value is the compiled subgraph returned by `make_agent`. Since `START -> korean_agent` is set, all conversations start at the Korean agent. If a user speaks Spanish, the Korean agent detects this and uses `handoff_tool` to switch to the Spanish agent.

#### Practice Points

- Change the starting agent to `spanish_agent` and send a message in Korean to see if automatic switching occurs
- Add a new language agent (e.g., Japanese). The `handoff_tool` docstring must also be modified
- Experiment with what error occurs when you remove `Command.PARENT`

---

### 18.2 Network Visualization

**Topic and Goal**: Use LangGraph Studio to visually inspect multi-agent graphs, debug execution flows, and defend against infinite loop bugs from self-transfers.

#### Key Concepts

When developing complex multi-agent systems, visually inspecting the graph structure is very important. LangGraph Studio is a tool that visualizes graph nodes and edges and tracks real-time execution flow.

However, for visualization, you must pre-declare which nodes the graph can move to from which nodes. In dynamic routing with `Command`, this is specified through the `destinations` parameter.

#### Code Analysis

**Step 1: Separating graph.py and Adding destinations**

While separating notebook code into `graph.py`, the `destinations` parameter is added to each agent node:

```python
graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "spanish_agent"),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
    destinations=("korean_agent", "spanish_agent"),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "korean_agent"),
)
```

The `destinations` parameter declares the target nodes reachable via `Command` from that node. This **does not affect execution logic** and is used by LangGraph Studio's visualization tools to display accurate graph structure.

**Step 2: Defending Against Self-Transfer Bugs**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    # ... (docstring omitted)
    if transfer_to == transfered_by:
        return {
            "error": "Stop trying to transfer to yourself and answer the question or i will fire you."
        }

    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

Defense statements are also added to the agent prompt:

```python
response = llm_with_tools.invoke(
    f"""
    {prompt}

    You have a tool called 'handoff_tool' use it to transfer to other agent,
    don't use it to transfer to yourself.

    Conversation History:
    {state["messages"]}
    """
)
```

LLMs may occasionally attempt to transfer to themselves, causing an **infinite loop**. This is defended with two layers:
1. **Prompt level**: Explicitly instruct "don't transfer to yourself"
2. **Code level**: `transfer_to == transfered_by` check rejects self-transfers and returns an error message

**Step 3: LangGraph Studio Configuration**

```json
{
    "dependencies": [
        "./graph.py"
    ],
    "graphs": {
        "agent": "./graph.py:graph"
    },
    "env": ".env"
}
```

The `langgraph.json` configuration file tells LangGraph Studio:
- `dependencies`: List of required Python files
- `graphs`: Location of graph objects to visualize (`filepath:variable_name` format)
- `env`: Environment variable file path

**Step 4: Streaming Execution**

```python
for event in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hola! Necesito ayuda con mi cuenta.",
            }
        ]
    },
    stream_mode="updates",
):
    print(event)
```

Using `stream_mode="updates"` allows you to receive state updates from each node in real-time. In the execution results, you can confirm the entire flow where the Korean agent detects Spanish, switches to the Spanish agent, and the Spanish agent responds in Spanish:

```
{'korean_agent': {'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
{'spanish_agent': {'messages': [...], 'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
```

#### Practice Points

- Try running the studio with LangGraph CLI: `langgraph dev`
- Open the studio with `destinations` removed and compare how the visualization differs
- Send a Greek message and confirm the transfer flow

---

### 18.3 Supervisor Architecture

**Topic and Goal**: Implement an architecture where a central **Supervisor** node analyzes conversations and routes them to the appropriate agent.

#### Key Concepts

In the network architecture, each agent autonomously decided on transfers. The supervisor architecture differs in that **one central coordinator (Supervisor)** handles all routing decisions.

```
                    ┌─────────────┐
          ┌────────►│  Supervisor │◄────────┐
          │         └──────┬──────┘         │
          │                │                │
          │         ┌──────┼──────┐         │
          │         ▼      ▼      ▼         │
    ┌─────┴──┐  ┌───┴───┐  ┌──┴──────┐     │
    │ korean │  │ greek │  │ spanish │─────┘
    │ _agent │  │_agent │  │ _agent  │
    └────────┘  └───────┘  └─────────┘
```

Advantages of this architecture:
- **Single entry point**: All requests go through the supervisor, centralizing routing logic
- **Consistency**: Agents don't need to worry about routing and can focus solely on their specialty
- **Easy control**: Modifying just the supervisor's prompt can change the entire routing strategy

#### Code Analysis

**Step 1: Structured Output Model Definition**

```python
from typing import Literal
from pydantic import BaseModel

class SupervisorOutput(BaseModel):
    next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
    reasoning: str
```

The supervisor's decision result is defined as a `Pydantic` model:
- `next_agent`: Uses `Literal` type to restrict possible values, forcing the LLM to return only valid agent names. `"__end__"` means terminating the conversation.
- `reasoning`: Explains why the supervisor chose that agent. Useful for debugging and transparency.

**Step 2: State Extension**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
    reasoning: str
```

A `reasoning` field is added to the state to track the supervisor's decision rationale.

**Step 3: Agent Simplification**

```python
def make_agent(prompt, tools):
    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}
    # ... (graph configuration is the same)
```

Unlike the network architecture, `handoff_tool`-related instructions are **removed** from the agent prompt. Since the supervisor handles routing, agents simply need to respond in their own language. An empty tool list is passed with `tools=[]`.

**Step 4: Supervisor Node Implementation**

```python
def supervisor(state: AgentState):
    structured_llm = llm.with_structured_output(SupervisorOutput)
    response = structured_llm.invoke(
        f"""
        You are a supervisor that routes conversations to the appropriate language agent.

        Analyse the customers request and the conversation history and decide which
        agent should handle the conversation.

        The options for the next agent are:
        - greek_agent
        - spanish_agent
        - korean_agent

        <CONVERSATION_HISTORY>
        {state.get("messages", [])}
        </CONVERSATION_HISTORY>

        IMPORTANT:

        Never transfer to the same agent twice in a row.

        If an agent has replied end the conversation by returning __end__
    """
    )
    return Command(
        goto=response.next_agent,
        update={"reasoning": response.reasoning},
    )
```

Key mechanisms of the supervisor:

- `llm.with_structured_output(SupervisorOutput)`: Forces the LLM to return JSON matching the `SupervisorOutput` schema. This enables stable routing without parsing errors.
- `<CONVERSATION_HISTORY>` XML tag: Clearly separates conversation history so the LLM accurately grasps context.
- `"Never transfer to the same agent twice in a row"`: Prompt constraint for infinite loop prevention.
- `"If an agent has replied end the conversation by returning __end__"`: Terminates the conversation if an agent has already responded, preventing unnecessary repetition.
- `Command(goto=response.next_agent)`: Since the supervisor is a top-level graph node (not a subgraph), `graph=Command.PARENT` is unnecessary.

**Step 5: Graph Assembly - Cyclic Structure**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "supervisor",
    supervisor,
    destinations=(
        "korean_agent",
        "spanish_agent",
        "greek_agent",
        END,
    ),
)

graph_builder.add_node("korean_agent", make_agent(
    prompt="You're a Korean customer support agent. You only speak and understand Korean.",
    tools=[],
))
graph_builder.add_node("greek_agent", make_agent(
    prompt="You're a Greek customer support agent. You only speak and understand Greek.",
    tools=[],
))
graph_builder.add_node("spanish_agent", make_agent(
    prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
    tools=[],
))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("korean_agent", "supervisor")
graph_builder.add_edge("spanish_agent", "supervisor")
graph_builder.add_edge("greek_agent", "supervisor")

graph = graph_builder.compile()
```

Summarizing the graph flow:

1. `START` -> `supervisor`: All conversations start at the supervisor
2. `supervisor` -> `{agent}` or `END`: The supervisor routes to the appropriate agent or termination via `Command`
3. `{agent}` -> `supervisor`: After an agent completes its response, it returns to the supervisor

This cyclic structure is the **core of the supervisor architecture**. The supervisor can decide whether to end the conversation (`__end__`) or route to another agent after an agent responds.

#### Network vs Supervisor Comparison

| Feature | Network | Supervisor |
|---------|---------|------------|
| Routing decision | Each agent autonomously | Central supervisor handles |
| Agent connections | P2P (direct connections) | Hub-spoke (via supervisor) |
| Agent complexity | High (includes routing logic) | Low (handles specialty only) |
| Scalability | All agents need modification when adding | Only supervisor needs modification |
| Debugging | Difficult | Easy (reasoning trackable) |

#### Practice Points

- Print the `reasoning` field to analyze what basis the supervisor uses for routing decisions
- Compare what code needs to be modified when adding agents in network architecture vs supervisor architecture
- Experiment with what happens when you remove the `__end__` option from `SupervisorOutput`

---

### 18.4 Supervisor As Tools - Encapsulating Agents as Tools

**Topic and Goal**: **Encapsulate agents as LLM tools** so that the supervisor naturally uses agents through the tool calling mechanism.

#### Key Concepts

In the previous section's supervisor architecture, `structured_output` was used for routing. In this section, we refactor by **converting each agent into a tool** and having the supervisor call agents through the `bind_tools` + `ToolNode` mechanism.

```
                ┌──────────────┐
    START ────► │  Supervisor  │ ────► END
                └──────┬───────┘
                       │
                 tools_condition
                       │
                ┌──────▼───────┐
                │   ToolNode   │
                │ ┌──────────┐ │
                │ │korean_ag.│ │
                │ │spanish_ag│ │
                │ │greek_ag. │ │
                │ └──────────┘ │
                └──────────────┘
```

Advantages of this approach:
- Leverages the LLM's **existing tool calling ability**, eliminating the need for separate routing logic
- The `@tool` decorator's `description` provides natural routing criteria
- `ToolNode` automatically executes the appropriate agent tool

#### Code Analysis

**Step 1: Agent-Tool Factory Function**

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

def make_agent_tool(tool_name, tool_description, system_prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {system_prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node("tools", ToolNode(tools=tools))

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    agent = agent_builder.compile()

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
    )
    def agent_tool(state: Annotated[dict, InjectedState]):
        result = agent.invoke(state)
        return result["messages"][-1].content

    return agent_tool
```

This function is structurally similar to the previous `make_agent`, but with a key difference:

- **The return value is a `@tool` function, not a compiled graph**
- `@tool(name_or_callable=tool_name, description=tool_description)`: Dynamically creates tools with name and description received as parameters
- `Annotated[dict, InjectedState]`: `InjectedState` is a special LangGraph annotation that **automatically injects the current graph state into the tool function**. The LLM doesn't recognize this parameter (it's not exposed in the tool schema); LangGraph automatically passes the state at execution time.
- `result["messages"][-1].content`: Extracts and returns only the agent's final response text

**Step 2: Tool List Creation**

```python
tools = [
    make_agent_tool(
        tool_name="korean_agent",
        tool_description="Use this when the user is speaking korean",
        system_prompt="You're a korean customer support agent you speak in korean",
        tools=[],
    ),
    make_agent_tool(
        tool_name="spanish_agent",
        tool_description="Use this when the user is speaking spanish",
        system_prompt="You're a spanish customer support agent you speak in spanish",
        tools=[],
    ),
    make_agent_tool(
        tool_name="greek_agent",
        tool_description="Use this when the user is speaking greek",
        system_prompt="You're a greek customer support agent you speak in greek",
        tools=[],
    ),
]
```

Each agent tool's `tool_description` serves as the LLM's routing criteria. The LLM detects the user's language and naturally calls the corresponding language's agent tool.

**Step 3: Supervisor Simplification**

```python
def supervisor(state: AgentState):
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [result],
    }
```

Compared to the previous version, this is **dramatically simplified**:
- Uses `bind_tools` instead of `structured_output`
- Simply passes messages instead of complex prompts
- The LLM reads tool descriptions on its own and selects the appropriate tool

**Step 4: Graph Assembly**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", tools_condition)
graph_builder.add_edge("tools", "supervisor")
graph_builder.add_edge("supervisor", END)

graph = graph_builder.compile()
```

In the previous supervisor architecture, each agent was a separate node, but now there are only **2 nodes**:
- `supervisor`: The node where the LLM decides on tool calls
- `tools`: The node where `ToolNode` executes agent tools

`tools_condition` routes to the `tools` node if the LLM response contains tool calls, otherwise to `END`. This is identical to the basic ReAct agent pattern, with the only difference being that the tools are agents.

#### Three Architecture Comparison

| Feature | Network | Supervisor | Supervisor+Tools |
|---------|---------|------------|-----------------|
| Graph node count | Same as agent count | Agent count + 1 | 2 (supervisor + tools) |
| Routing mechanism | Command + handoff_tool | structured_output | bind_tools + ToolNode |
| Agent implementation | Subgraph nodes | Subgraph nodes | Inside @tool functions |
| Code complexity | Medium | High | Low |

#### Practice Points

- Remove `InjectedState` and run it. Observe what arguments the LLM tries to pass to the agent
- Write more detailed agent tool `description`s and experiment whether routing accuracy improves
- Add a real tool (e.g., search tool) to one agent and check if nested tool calls work

---

### 18.5 Prebuilt Agents

**Topic and Goal**: Use `create_supervisor` and `create_react_agent` from the `langgraph-supervisor` library to implement a multi-agent supervisor system with **minimal code**.

#### Key Concepts

All patterns we manually implemented so far (agent factory, supervisor, tool-based routing) are provided by LangGraph as **prebuilt modules**. With the `langgraph-supervisor` and `langgraph-swarm` packages, you can build powerful multi-agent systems in just a few lines of code.

Additional dependencies:
```toml
"langgraph-supervisor==0.0.29",
"langgraph-swarm==0.0.14",
```

#### Code Analysis

**Step 1: Simplified Imports**

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
```

All the imports needed in previous sections -- `StateGraph`, `Command`, `ToolNode`, `tools_condition` -- are all gone. Just two functions, `create_react_agent` and `create_supervisor`, are sufficient.

**Step 2: Specialized Agent Creation**

```python
MODEL = "openai:gpt-5"

history_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="history_agent",
    prompt="You are a history expert. You only answer questions about history.",
)
geography_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="geography_agent",
    prompt="You are a geography expert. You only answer questions about geography.",
)
maths_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="maths_agent",
    prompt="You are a maths expert. You only answer questions about maths.",
)
philosophy_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="philosophy_agent",
    prompt="You are a philosophy expert. You only answer questions about philosophy.",
)
```

`create_react_agent` is a prebuilt function provided by LangGraph that creates a ReAct pattern agent in a single line:

- `model`: Model string or initialized model object
- `tools`: List of tools for the agent to use
- `name`: Unique name for the agent (used by the supervisor for routing)
- `prompt`: System prompt for the agent

All the manual implementation from previous sections -- `StateGraph` + `agent` node + `tools` node + edge connections -- is encapsulated in this single line.

**Step 3: Supervisor Creation**

```python
supervisor = create_supervisor(
    agents=[
        history_agent,
        maths_agent,
        geography_agent,
        philosophy_agent,
    ],
    model=init_chat_model(MODEL),
    prompt="""
    You are a supervisor that routes student questions to the appropriate subject expert.
    You manage a history agent, geography agent, maths agent, and philosophy agent.
    Analyze the student's question and assign it to the correct expert based on the subject matter:
        - history_agent: For historical events, dates, historical figures
        - geography_agent: For locations, rivers, mountains, countries
        - maths_agent: For mathematics, calculations, algebra, geometry
        - philosophy_agent: For philosophical concepts, ethics, logic
    """,
).compile()
```

What `create_supervisor` does internally:
1. Converts each agent into a tool (`transfer_to_{agent_name}` format)
2. Creates a supervisor node and binds tools
3. Sets up `ToolNode` and conditional edges
4. Automatically adds a `transfer_back_to_supervisor` tool that returns to the supervisor after agent execution

`.compile()` is called to compile into an executable graph.

**Step 4: Execution and Verification**

```python
questions = [
    "When was Madrid founded?",
    "What is the capital of France and what river runs through it?",
    "What is 15% of 240?",
    "Tell me about the Battle of Waterloo",
    "What are the highest mountains in Asia?",
    "If I have a rectangle with length 8 and width 5, what is its area and perimeter?",
    "Who was Alexander the Great?",
    "What countries border Switzerland?",
    "Solve for x: 2x + 10 = 30",
]

for question in questions:
    result = supervisor.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        }
    )
    if result["messages"]:
        for message in result["messages"]:
            message.pretty_print()
```

Looking at the execution flow:

1. The user's question is passed to the supervisor
2. The supervisor calls the `transfer_to_{agent_name}` tool
3. The corresponding agent answers the question
4. The agent automatically calls `transfer_back_to_supervisor` to return to the supervisor
5. The supervisor returns the final response

Execution result example:
```
Human Message: When was Madrid founded?
Ai Message (supervisor): transfer_to_history_agent call
Tool Message: Successfully transferred to history_agent
Ai Message (history_agent): Madrid originated in the mid-9th century...
Ai Message (history_agent): Transferring back to supervisor
Ai Message (supervisor): Final response delivered
```

#### Manual Implementation vs Prebuilt Comparison

```python
# Manual implementation (18.4): ~60 lines
def make_agent_tool(...): ...
def supervisor(...): ...
graph_builder = StateGraph(...)
graph_builder.add_node(...)
# ... multiple configuration lines

# Prebuilt (18.5): ~15 lines
agent = create_react_agent(model=MODEL, tools=[], name="agent", prompt="...")
supervisor = create_supervisor(agents=[...], model=..., prompt="...").compile()
```

#### Practice Points

- Add a real tool to `philosophy_agent` to create an agent that utilizes external data
- Change the supervisor's prompt to Korean and verify it works correctly
- Measure the rate at which the supervisor selects the correct agent for various questions
- The `langgraph-swarm` package was also installed. Research the Swarm pattern and compare it with the Supervisor pattern

---

## 3. Chapter Key Summary

### Three Multi-Agent Architecture Patterns

| Pattern | Description | Advantages | Disadvantages |
|---------|-------------|------------|---------------|
| **Network (P2P)** | Agents directly transfer via `Command` + `handoff_tool` | No central bottleneck, high autonomy | All agents need modification when adding new ones |
| **Supervisor** | Central node routes via `structured_output` | Easy control, easy debugging | Supervisor prompt complexity can increase |
| **Supervisor+Tools** | Agents encapsulated as `@tool`, routed via `bind_tools` | Concise code, leverages LLM's built-in ability | Limited access to agent internal state |

### Core LangGraph Concepts

1. **`Command`**: Object for programmatically controlling graph execution flow
   - `goto`: Specify next execution node
   - `update`: State update
   - `graph=Command.PARENT`: Move at parent graph level

2. **`InjectedState`**: Annotation for automatically injecting current graph state into tool functions. Not exposed in the LLM schema.

3. **`destinations`**: Parameter of `add_node` that declares possible target nodes in `Command`-based dynamic routing. Visualization only; does not affect execution logic.

4. **Subgraph pattern**: Uses compiled graphs returned by `make_agent` as nodes of other graphs to create hierarchical structures.

5. **`create_react_agent` / `create_supervisor`**: Prebuilt functions that encapsulate all the above patterns.

### Infinite Loop Defense Strategies

The most common problem in multi-agent systems is **infinite agent transitions**. Strategies to prevent this:

1. **Prompt constraint**: Explicitly state "don't transfer to yourself"
2. **Code-level validation**: `transfer_to == transfered_by` check
3. **Supervisor's termination condition**: Provide `__end__` option
4. **Structural constraint**: Allow only valid targets with `Literal` type

---

## 4. Practice Assignments

### Assignment 1: Extend Multilingual Customer Support System (Difficulty: Medium)

Based on the network architecture (18.1), implement the following:
- Add Japanese and Chinese agents
- Properly modify `handoff_tool`'s docstring and `destinations`
- Add simple FAQ tools to each agent to implement features like "check shipping status" and "request refund"

### Assignment 2: Supervisor Architecture Comparison Experiment (Difficulty: Medium)

For the same scenario (student question routing):
1. 18.3's `structured_output` supervisor
2. 18.4's tool-based supervisor
3. 18.5's `create_supervisor`

Implement each of the three approaches and measure for the same question set:
- Rate of routing to the correct agent
- Response time
- Token usage

Write a comparison report.

### Assignment 3: Hierarchical Multi-Supervisor (Difficulty: Hard)

Nest `create_supervisor` to implement the following 2-level hierarchical structure:

```
                    ┌─────────────────┐
                    │  Main Supervisor│
                    └───┬─────────┬───┘
                        │         │
              ┌─────────▼──┐  ┌──▼──────────┐
              │Science Sup.│  │Humanities S.│
              └──┬──────┬──┘  └──┬──────┬───┘
                 │      │        │      │
              ┌──▼┐  ┌──▼┐   ┌──▼┐  ┌──▼──────┐
              │Phy.│  │Chem│   │Hist│  │Philosophy│
              └───┘  └───┘   └───┘  └─────────┘
```

- Main Supervisor: Determines science/humanities field
- Science Supervisor: Manages physics/chemistry agents
- Humanities Supervisor: Manages history/philosophy agents

### Assignment 4: Swarm Architecture Exploration (Difficulty: Hard)

Using the `langgraph-swarm` package:
1. Research what Swarm architecture is
2. Analyze differences from network architecture
3. Implement the same customer support scenario with the Swarm pattern
4. Summarize the pros and cons of the network, supervisor, and swarm patterns

---

## Appendix: Key API Reference

### Command

```python
Command(
    goto="node_name",           # Node to move to
    update={"key": "value"},    # State update
    graph=Command.PARENT,       # Move at parent graph level (when used inside subgraph)
)
```

### create_react_agent

```python
agent = create_react_agent(
    model="openai:gpt-4o",      # Model string or initialized model
    tools=[tool1, tool2],        # List of tools to use
    name="agent_name",           # Unique agent name
    prompt="System prompt",      # Agent role definition
)
```

### create_supervisor

```python
supervisor = create_supervisor(
    agents=[agent1, agent2],     # List of agents to manage
    model=init_chat_model(...),  # Model for the supervisor
    prompt="Routing rules prompt",  # Supervisor instructions
).compile()                      # Must call compile()
```

### InjectedState

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def my_tool(state: Annotated[dict, InjectedState]):
    # state is automatically injected with the current graph state
    # The LLM does not recognize this parameter
    return state["messages"][-1].content
```
