# Chapter 13: LangGraph Basics (LangGraph Fundamentals)

## Chapter Overview

In this chapter, we learn the fundamentals of **LangGraph**, a core framework in the LangChain ecosystem, step by step from scratch. LangGraph is a framework that allows you to design and execute LLM-based applications as **Graph** structures, enabling intuitive construction of complex AI agent workflows.

Through this chapter, you will learn:

- Initial setup and environment configuration for LangGraph projects
- Basic graph structure: Nodes and Edges
- Graph State management and data transfer between nodes
- Input/output separation using Multiple Schemas
- State merging strategies through Reducer functions
- Performance optimization using Node Caching
- Dynamic flow control through Conditional Edges
- Dynamic parallel processing using the Send API
- Internal node routing using Command objects

### Project Environment

| Item | Version/Details |
|------|----------------|
| Python | >= 3.13 |
| LangGraph | >= 0.6.6 |
| LangChain | >= 0.3.27 (including OpenAI) |
| Development Tools | Jupyter Notebook (ipykernel) |
| Package Management | uv (pyproject.toml based) |

---

## 13.0 Introduction - Initial Project Setup

### Topic and Objectives
Create a new Python project for learning LangGraph and install the required dependencies.

### Key Concepts

To start a LangGraph project, we set up a new project directory called `hello-langgraph`. This project uses the **uv** package manager and exercises are conducted in a Jupyter Notebook environment.

#### Project Dependencies (`pyproject.toml`)

```toml
[project]
name = "hello-langgraph"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf>=0.8",
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.6",
    "langgraph-checkpoint-sqlite>=2.0.11",
    "langgraph-cli[inmem]>=0.4.0",
    "python-dotenv>=1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Role of each dependency:

| Package | Role |
|---------|------|
| `langgraph` | Graph-based workflow framework (core) |
| `langchain[openai]` | LangChain and OpenAI integration |
| `grandalf` | Graph visualization support |
| `langgraph-checkpoint-sqlite` | State checkpoint storage (SQLite) |
| `langgraph-cli[inmem]` | LangGraph CLI tools (in-memory mode) |
| `python-dotenv` | Environment variable management (.env files) |
| `ipykernel` | Jupyter Notebook kernel (development only) |

### Practice Points
- Initialize a project using `uv` and install dependencies.
- Use `.gitignore` to exclude virtual environments, cache files, etc. from version control.
- Verify that the Jupyter Notebook environment works correctly.

---

## 13.1 Your First Graph - Building Your First Graph

### Topic and Objectives
Understand the most basic building blocks of LangGraph -- **StateGraph**, **Node**, and **Edge** -- and construct your first graph.

### Key Concepts

A graph in LangGraph consists of three core elements:

1. **State**: A data structure shared across the entire graph. Defined using `TypedDict`.
2. **Node**: An individual function executed within the graph. Each node receives state as input.
3. **Edge**: A connection between nodes. Determines execution order.

LangGraph also provides two special nodes:
- **`START`**: The graph's entry point. Indicates which node executes first.
- **`END`**: The graph's exit point. The node connected here executes last.

### Code Analysis

#### Step 1: Imports and State Definition

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    hello: str

graph_builder = StateGraph(State)
```

- `StateGraph` is the core class for creating state-based graphs.
- `State` inherits from `TypedDict` to define the schema of the state used in the graph.
- `graph_builder` is an instance of `StateGraph`, through which we add nodes and edges.

#### Step 2: Node Function Definitions

```python
def node_one(state: State):
    print("node_one")

def node_two(state: State):
    print("node_two")

def node_three(state: State):
    print("node_three")
```

- Each node function must accept `state` as a parameter.
- At this stage, we're not modifying state yet, just printing for execution confirmation.

#### Step 3: Graph Construction

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

This code creates the following linear graph:

```
START -> node_one -> node_two -> node_three -> END
```

- `add_node(name, function)`: Registers a node in the graph.
- `add_edge(source, destination)`: Adds an edge connecting two nodes.

### Practice Points
- Check what error occurs when `START` and `END` are omitted.
- Change the order of nodes and observe how execution flow changes.
- Test whether the function name is automatically used when the first argument (string name) is omitted in `add_node`.

---

## 13.2 Graph State - Graph State Management

### Topic and Objectives
Understand how nodes **read and modify** state, and track how state changes as it passes through nodes.

### Key Concepts

State is central to graph execution in LangGraph. Each node:

1. Receives the current state as **input**.
2. **Returns** a dictionary to update the state.
3. The returned values **overwrite** the existing state (default behavior).

An important point is that values for keys not returned by a node remain unchanged.

### Code Analysis

#### Expanded State Definition

```python
class State(TypedDict):
    hello: str
    a: bool

graph_builder = StateGraph(State)
```

Now the state has two fields: `hello` (string) and `a` (boolean).

#### Reading and Modifying State in Nodes

```python
def node_one(state: State):
    print("node_one", state)
    return {
        "hello": "from node one.",
        "a": True,
    }

def node_two(state: State):
    print("node_two", state)
    return {"hello": "from node two."}

def node_three(state: State):
    print("node_three", state)
    return {"hello": "from node three."}
```

Key points:
- `node_one` updates both the `hello` and `a` fields.
- `node_two` only updates `hello`. The previous value of `a` (`True`) is preserved.
- `node_three` also only updates `hello`. `a` remains `True`.

#### Graph Compilation and Execution

```python
graph = graph_builder.compile()

result = graph.invoke(
    {
        "hello": "world",
    },
)
```

- `compile()`: Compiles the graph builder into an executable graph.
- `invoke()`: Passes the initial state to execute the graph.

#### Execution Result Tracking

```
node_one {'hello': 'world'}
node_two {'hello': 'from node one.', 'a': True}
node_three {'hello': 'from node two.', 'a': True}
```

| Timing | hello | a |
|--------|-------|---|
| Initial input | `"world"` | (none) |
| After node_one | `"from node one."` | `True` |
| After node_two | `"from node two."` | `True` |
| After node_three | `"from node three."` | `True` |

Final result: `{'hello': 'from node three.', 'a': True}`

**The default state update strategy is "overwrite."** The value returned by a node replaces the existing value for that key. Keys not returned remain unchanged.

### Practice Points
- Try including an `a` value in the initial input and see how it appears in nodes.
- Test what happens when a node returns a key that doesn't exist in the `state`.
- Print the `graph` object directly to see a visual diagram of the graph.

---

## 13.4 Multiple Schemas - Multiple Schemas

### Topic and Objectives
Learn how to separate **input schema**, **output schema**, and **internal (Private) schema** within a single graph.

### Key Concepts

In real applications, the following requirements frequently arise:

- The format of **input data** received from users differs from the data processed internally.
- The data ultimately **returned to users** should only be a portion of the internal state.
- Some nodes need **private state** that only they can access.

LangGraph solves this by specifying three schemas for `StateGraph`:

| Parameter | Role |
|-----------|------|
| First argument (State) | Internal full state (Private State) |
| `input_schema` | Input format passed from outside to the graph |
| `output_schema` | Output format returned from the graph to outside |

### Code Analysis

#### Multiple Schema Definitions

```python
class PrivateState(TypedDict):
    a: int
    b: int

class InputState(TypedDict):
    hello: str

class OutputState(TypedDict):
    bye: str

class MegaPrivate(TypedDict):
    secret: bool

graph_builder = StateGraph(
    PrivateState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

In this configuration:
- Externally, only `{"hello": "world"}` format can be provided as input.
- Internally, fields `a` and `b` are used for calculations.
- The final output is returned only in `{"bye": "world"}` format.
- `MegaPrivate` is an ultra-private state used only by specific nodes.

#### Nodes Using Various Schemas

```python
def node_one(state: InputState) -> InputState:
    print("node_one ->", state)
    return {"hello": "world"}

def node_two(state: PrivateState) -> PrivateState:
    print("node_two ->", state)
    return {"a": 1}

def node_three(state: PrivateState) -> PrivateState:
    print("node_three ->", state)
    return {"b": 1}

def node_four(state: PrivateState) -> OutputState:
    print("node_four ->", state)
    return {"bye": "world"}

def node_five(state: OutputState):
    return {"secret": True}

def node_six(state: MegaPrivate):
    print(state)
```

Note that each node uses a different schema as its type hint. This explicitly expresses **what data each node is interested in**.

#### Graph Construction and Execution

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)
graph_builder.add_node("node_five", node_five)
graph_builder.add_node("node_six", node_six)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", "node_four")
graph_builder.add_edge("node_four", "node_five")
graph_builder.add_edge("node_five", "node_six")
graph_builder.add_edge("node_six", END)
```

#### Execution Result Analysis

```
node_one -> {'hello': 'world'}
node_two -> {}
node_three -> {'a': 1}
node_four -> {'a': 1, 'b': 1}
{'secret': True}
```

Final return value: `{'bye': 'world'}`

Key observations:
- `node_one` can only see `InputState`, so it receives `{'hello': 'world'}`.
- `node_two` sees `PrivateState`, but since `a` and `b` haven't been set yet, it's `{}`.
- `node_three` can see `{'a': 1}` that was set by `node_two`.
- `node_four` can see the full PrivateState `{'a': 1, 'b': 1}`.
- **The final output only contains the `bye` field defined in `OutputState`**. Internal state (`a`, `b`, `secret`) is not exposed externally.

### Practice Points
- Check how the return value changes when `output_schema` is not specified.
- Test what happens when you pass a field to `invoke()` that doesn't exist in `input_schema`.
- Think about why schema separation is important in real production environments (security, API design, etc.).

---

## 13.5 Reducer Functions - Reducer Functions

### Topic and Objectives
Learn how to **accumulate** state using **Reducer functions** instead of the default "overwrite" strategy.

### Key Concepts

By default, LangGraph's state updates **completely replace** the previous value with the new one. However, in many cases (especially chat message histories), values need to be **accumulated**.

**Reducer functions** use the `Annotated` type hint to customize the update strategy for specific fields:

```
Annotated[type, reducer_function]
```

A reducer function takes two arguments:
- `old`: The current state value
- `new`: The new value returned by the node

And returns **the final value to be stored**.

### Code Analysis

#### Reducer Function Definition and State Application

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
import operator

def update_function(old, new):
    return old + new

class State(TypedDict):
    # messages: Annotated[list[str], update_function]
    messages: Annotated[list[str], operator.add]

graph_builder = StateGraph(State)
```

Key points:
- `Annotated[list[str], operator.add]` means "when the `messages` field is updated, **concatenate** the new list to the existing list."
- `operator.add` is a Python built-in function that performs the `+` operation (concatenation) on lists.
- The commented-out `update_function` is a custom reducer function with identical behavior. You can either create your own or use existing functions like `operator.add`.

#### Node Definitions

```python
def node_one(state: State):
    return {
        "messages": ["Hello, nice to meet you!"],
    }

def node_two(state: State):
    return {}

def node_three(state: State):
    return {}
```

- Only `node_one` adds a new item to `messages`.
- `node_two` and `node_three` return empty dictionaries, so they don't modify the state.

#### Execution and Results

```python
graph = graph_builder.compile()

graph.invoke(
    {"messages": ["Hello!"]},
)
```

Result: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`

| Timing | messages |
|--------|----------|
| Initial input | `["Hello!"]` |
| After node_one | `["Hello!"] + ["Hello, nice to meet you!"]` = `["Hello!", "Hello, nice to meet you!"]` |
| node_two, node_three | No change |

**Without a reducer**, `node_one`'s return value `["Hello, nice to meet you!"]` would have completely replaced the initial value `["Hello!"]`. Thanks to the reducer, the two lists were **combined**.

### Practice Points
- Write custom reducer functions (e.g., keep only the maximum value, remove duplicates, etc.).
- Add messages in `node_two` as well and verify that accumulation works correctly.
- Run the same code without a reducer and compare how results differ.
- Think about why reducers are essential in chat applications.

---

## 13.6 Node Caching - Node Caching

### Topic and Objectives
Learn how to cache specific node execution results using **CachePolicy** and use cached values without recomputation for a certain period.

### Key Concepts

Some nodes are expensive to execute (e.g., external API calls, LLM calls) or return the same result for the same input. In such cases, **caching** can optimize performance.

LangGraph provides per-node cache policies:

| Component | Role |
|-----------|------|
| `CachePolicy(ttl=seconds)` | Sets cache validity period (Time-To-Live) in seconds |
| `InMemoryCache()` | Memory-based cache store |
| `graph_builder.compile(cache=...)` | Connect cache store at compile time |

### Code Analysis

#### Imports and State Definition

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from datetime import datetime

class State(TypedDict):
    time: str

graph_builder = StateGraph(State)
```

#### Node Definition - Cache Target Node

```python
def node_one(state: State):
    return {}

def node_two(state: State):
    return {"time": f"{datetime.now()}"}

def node_three(state: State):
    return {}
```

`node_two` returns the current time. When caching is applied, the previously recorded time is returned as-is during the TTL period.

#### Applying Cache Policy

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node(
    "node_two",
    node_two,
    cache_policy=CachePolicy(ttl=20),  # Cache for 20 seconds
)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

Key: Only `node_two` has `cache_policy=CachePolicy(ttl=20)` specified. This node's result is **cached for 20 seconds**.

#### Compilation with Cache Enabled and Repeated Execution

```python
import time

graph = graph_builder.compile(cache=InMemoryCache())

print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
```

This code runs the graph 6 times at 5-second intervals. Since `node_two` has `ttl=20`:

- **First ~20 seconds**: The first execution result (time) is cached and the same time is returned.
- **After 20 seconds**: The cache expires, `node_two` runs again, and a new time is recorded.

### Practice Points
- Change the `ttl` value and observe cache expiration timing.
- Investigate whether other cache stores can be used instead of `InMemoryCache`.
- Think about real scenarios where caching is useful (e.g., external API rate limiting, cost reduction).
- Also think about cases where caching could be problematic (e.g., when real-time data is needed).

---

## 13.7 Conditional Edges - Conditional Edges

### Topic and Objectives
Implement branching logic that **dynamically selects the next node** based on state using **Conditional Edges**.

### Key Concepts

Up to now, all graph paths have been fixed (linear flow). However, in real applications, it's often necessary to choose different paths based on state.

The **`add_conditional_edges`** method allows:
1. Executing a **routing function** after a specific node.
2. Dynamically determining the next node based on the routing function's return value.

```
add_conditional_edges(
    source_node,
    routing_function,
    mapping_dictionary   # {return_value: destination_node}
)
```

### Code Analysis

#### State and Node Definitions

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal

class State(TypedDict):
    seed: int

graph_builder = StateGraph(State)

def node_one(state: State):
    print("node_one ->", state)
    return {}

def node_two(state: State):
    print("node_two ->", state)
    return {}

def node_three(state: State):
    print("node_three ->", state)
    return {}

def node_four(state: State):
    print("node_four ->", state)
    return {}
```

#### Routing Function Definition

The code shows two approaches for routing functions:

**Approach 1: Returning a string (commented out)**
```python
# def decide_path(state: State) -> Literal["node_three", "node_four"]:
#     if state["seed"] % 2 == 0:
#         return "node_three"
#     else:
#         return "node_four"
```
This approach returns node names directly. The `Literal` type hint specifies the possible return values.

**Approach 2: Returning an arbitrary value + mapping dictionary (actually used)**
```python
def decide_path(state: State):
    return state["seed"] % 2 == 0  # Returns True or False
```
The routing function returns arbitrary values like `True`/`False`, and the mapping dictionary converts them to actual nodes.

#### Conditional Edge Construction

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)

# Conditional branch from START
graph_builder.add_conditional_edges(
    START,
    decide_path,
    {
        True: "node_one",     # If seed is even, go to node_one
        False: "node_two",    # If seed is odd, go to node_two
        "hello": END,         # If "hello" is returned, terminate
    },
)

graph_builder.add_edge("node_one", "node_two")

# Conditional branch from node_two as well
graph_builder.add_conditional_edges(
    "node_two",
    decide_path,
    {
        True: "node_three",
        False: "node_four",
        "hello": END,
    },
)

graph_builder.add_edge("node_four", END)
graph_builder.add_edge("node_three", END)
```

The flow of this graph:

```
             ┌─ True ──> node_one ──> node_two ─┬─ True ──> node_three ──> END
START ───────┤                                  ├─ False ─> node_four ───> END
             ├─ False ─> node_two ──────────────┘
             └─ "hello" ──> END
```

### Practice Points
- Change the `seed` value in various ways and observe how execution paths differ.
- Switch to the approach that returns node names directly (Approach 1) without a mapping dictionary.
- Design conditional edges with 3 or more branches.
- Create complex workflows by mixing conditional edges and regular edges.

---

## 13.8 Send API - Dynamic Parallel Processing

### Topic and Objectives
Learn how to **dynamically create node instances** at runtime and **execute them in parallel** using the **Send API**.

### Key Concepts

Conditional edges decide "which node to go to," but the **Send API** goes one step further:

1. It can execute **the same node multiple times** in parallel.
2. Each instance can receive **different inputs**.
3. The **number of instances is determined at runtime**.

This is similar to the Map-Reduce pattern:
- **Map**: Split data and apply the same processing to each piece
- **Reduce**: Gather results together (used in combination with Reducer functions)

### Code Analysis

#### Imports and State Definition

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.types import Send
import operator
from typing import Union

class State(TypedDict):
    words: list[str]
    output: Annotated[list[dict[str, Union[str, int]]], operator.add]

graph_builder = StateGraph(State)
```

Key points:
- `words`: List of words to process
- `output`: List that **accumulates** processing results for each word. Uses `Annotated` with the `operator.add` reducer to combine results.
- `Send` is imported. This is the core of dynamic parallel processing.

#### Node Definitions

```python
def node_one(state: State):
    print(f"I want to count {len(state['words'])} words in my state.")

def node_two(word: str):
    return {
        "output": [
            {
                "word": word,
                "letters": len(word),
            }
        ]
    }
```

Important differences:
- `node_one` is a regular node that receives the full `State`.
- **`node_two` receives an individual `word` (string) instead of `State`.** This is custom input passed through the Send API.
- `node_two` adds results to the `output` list. Thanks to the reducer (`operator.add`), results from all parallel executions are automatically combined.

#### Dispatcher Function and Graph Construction

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)

def dispatcher(state: State):
    return [Send("node_two", word) for word in state["words"]]

graph_builder.add_edge(START, "node_one")
graph_builder.add_conditional_edges("node_one", dispatcher, ["node_two"])
graph_builder.add_edge("node_two", END)
```

Key analysis:

1. **`dispatcher` function**: Creates a `Send` object for each word in `state["words"]`.
   - `Send("node_two", word)`: "Execute `node_two` with `word` as input"
   - Since it returns a list, `node_two` is executed in parallel **as many times as there are words**.

2. **Passing a list to `add_conditional_edges`**: `["node_two"]` is the list of possible destination nodes.

#### Execution Results

```python
graph.invoke(
    {
        "words": ["hello", "world", "how", "are", "you", "doing"],
    }
)
```

Output:
```
I want to count 6 words in my state.
```

Result:
```python
{
    'words': ['hello', 'world', 'how', 'are', 'you', 'doing'],
    'output': [
        {'word': 'hello', 'letters': 5},
        {'word': 'world', 'letters': 5},
        {'word': 'how', 'letters': 3},
        {'word': 'are', 'letters': 3},
        {'word': 'you', 'letters': 3},
        {'word': 'doing', 'letters': 5}
    ]
}
```

6 instances of `node_two` each processed one word, and their results were combined into the `output` list by the `operator.add` reducer.

### Practice Points
- Increase the size of the word list and observe performance differences.
- Add `time.sleep` to `node_two` to feel the effect of parallel execution.
- Write code that produces the same result without using the Send API and compare.
- Think of practical use cases (e.g., summarizing multiple documents simultaneously, collecting data from multiple sources, etc.).

---

## 13.9 Command - Command Object

### Topic and Objectives
Learn how to perform **state updates and routing simultaneously** from within a node using the **Command** object.

### Key Concepts

In the methods learned so far:
- State updates: Node returns a dictionary
- Routing: `add_conditional_edges` + separate routing function

These two were **separated**. The **Command** object **unifies** them:

```python
Command(
    goto="destination_node",       # Next node to move to
    update={"key": "value"},  # State update
)
```

Advantages of this approach:
- Routing logic is inside the node, making it more intuitive.
- State updates and routing are handled atomically.
- No separate routing functions or conditional edges are needed.

### Code Analysis

#### Imports and State Definition

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import Command

class State(TypedDict):
    transfer_reason: str

graph_builder = StateGraph(State)
```

#### Node Definitions - Router Node Returning Command

```python
from typing import Literal

def triage_node(state: State) -> Command[Literal["account_support", "tech_support"]]:
    return Command(
        goto="account_support",
        update={
            "transfer_reason": "The user wants to change password.",
        },
    )

def tech_support(state: State):
    return {}

def account_support(state: State):
    print("account_support running")
    return {}
```

Key analysis:

1. **Return type of `triage_node`**: `Command[Literal["account_support", "tech_support"]]`
   - This specifies via type that the node returns a `Command` and that the possible destinations are `"account_support"` or `"tech_support"`.
   - Thanks to this type hint, LangGraph can know the possible paths **without `add_edge` or `add_conditional_edges`**.

2. **`Command` object**:
   - `goto="account_support"`: Move to the `account_support` node next
   - `update={"transfer_reason": "The user wants to change password."}`: Update the state's `transfer_reason`

#### Graph Construction

```python
graph_builder.add_node("triage_node", triage_node)
graph_builder.add_node("tech_support", tech_support)
graph_builder.add_node("account_support", account_support)

graph_builder.add_edge(START, "triage_node")
# No add_edge needed after triage_node! Command handles the routing.

graph_builder.add_edge("tech_support", END)
graph_builder.add_edge("account_support", END)
```

Note: No edge is defined after `triage_node`. The `Command` object's `goto` determines the next node at runtime.

Graph structure:
```
                         ┌──> tech_support ────> END
START ──> triage_node ───┤
                         └──> account_support ──> END
```

#### Execution Results

```python
graph = graph_builder.compile()
graph.invoke({})
```

Output:
```
account_support running
```

Result: `{'transfer_reason': 'The user wants to change password.'}`

`triage_node` used `Command` to:
1. Update `transfer_reason` and
2. Route to `account_support`.

### Practice Points
- Modify `triage_node` to route to `tech_support` based on a condition.
- Compare the pros and cons of `Command` vs `add_conditional_edges` approaches.
- Implement multi-stage routing like a real customer support system using `Command`.
- Investigate whether `goto` in `Command` can specify multiple nodes.

---

## Chapter Key Summary (Key Takeaways)

### 1. LangGraph Basic Structure
- **StateGraph**: The core class for state-based graphs
- **Node**: A function that receives state, processes it, and returns updates
- **Edge**: Connections between nodes (determines execution order)
- **START / END**: The graph's entry and exit points

### 2. State Management
- State schema is defined with `TypedDict`.
- The default update strategy is **overwrite**.
- Using `Annotated` with reducer functions allows applying an **accumulate** strategy.
- `operator.add` is the most commonly used reducer for list concatenation.

### 3. Multiple Schemas
- `input_schema`: Restricts external input format
- `output_schema`: Restricts external output format
- Internal state is not exposed externally, which is beneficial for security and API design.

### 4. Caching
- Set per-node cache policies with `CachePolicy(ttl=seconds)`.
- Activate with `InMemoryCache()` and `compile(cache=...)`.
- Can significantly improve performance for expensive operations (API calls, etc.).

### 5. Flow Control
| Method | Characteristics | When to Use |
|--------|----------------|-------------|
| `add_edge` | Fixed path | Always the same next node |
| `add_conditional_edges` | Dynamic routing based on routing function | Change path based on state |
| `Send` API | Dynamic parallel execution | Execute same node with different inputs multiple times |
| `Command` | Internal node routing + state update | Handle routing and state changes at once |

### 6. Core Design Principles
- Graphs are constructed **declaratively**: Define nodes and edges first, then compile and execute later.
- State is treated as if it were **immutable**: Nodes return new dictionaries to update state.
- **Separation of concerns**: Each node has only one responsibility.

---

## Practice Exercises

### Exercise 1: Basic Graph (Difficulty: low)

Create a linear graph with 4 nodes (`start_node`, `process_a`, `process_b`, `end_node`). Add a `counter: int` field to the state and have each node increment `counter` by 1. The final `counter` value should be 4.

**Hint**: Without a reducer, overwriting occurs. Read the current value in each node and return the value +1.

### Exercise 2: Chat Message Accumulation (Difficulty: medium)

Create a simple chat simulator using reducers:
- State: `messages: Annotated[list[str], operator.add]`
- `user_node`: Add `["User: Hello"]`
- `assistant_node`: Add `["Assistant: How can I help you?"]`
- `user_reply_node`: Add `["User: Tell me the weather"]`

The final `messages` should contain 3 messages in order.

### Exercise 3: Conditional Routing (Difficulty: medium)

Create a graph that branches to different paths based on user age:
- State: `age: int`, `message: str`
- After `check_age` node, conditional branch:
  - Under 18: `minor_node` -> "You are a minor."
  - 18 to under 65: `adult_node` -> "You are an adult."
  - 65 and over: `senior_node` -> "You are eligible for senior benefits."

### Exercise 4: Send API Usage (Difficulty: high)

Create a graph that takes a sentence as input and converts each word to uppercase simultaneously:
- State: `sentence: str`, `results: Annotated[list[str], operator.add]`
- `splitter_node`: Split the sentence into words
- `upper_node`: Convert individual words to uppercase (parallel execution via Send API)
- Input: `{"sentence": "hello world from langgraph"}`
- Expected output: `{"sentence": "...", "results": ["HELLO", "WORLD", "FROM", "LANGGRAPH"]}`

### Exercise 5: Command-Based Agent (Difficulty: high)

Implement a simple customer support router using Command objects:
- State: `query: str`, `department: str`, `response: str`
- `router_node`: Route with Command based on query content
  - Contains "refund" or "payment" -> `billing_node`
  - Contains "error" or "bug" -> `tech_node`
  - Otherwise -> `general_node`
- Each department node sets an appropriate guidance message in `response`

**Bonus**: Use `Command`'s type hint (`Command[Literal[...]]`) correctly so that all possible paths are displayed when visualizing the graph.
