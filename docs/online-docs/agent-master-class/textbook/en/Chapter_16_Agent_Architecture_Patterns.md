# Chapter 16: Workflow Architecture Patterns

---

## 1. Chapter Overview

In this chapter, we learn the core **workflow architecture patterns** that can be used when building AI agent systems. We practice various ways of systematically combining LLM calls using LangGraph, with the goal of understanding which pattern is appropriate for which situation.

### Architecture Patterns Covered

| Section | Pattern | Core Concept |
|---------|---------|--------------|
| 16.0 | Introduction | Project environment setup |
| 16.1 | Prompt Chaining | Sequential LLM call chain |
| 16.2 | Prompt Chaining Gate | Conditional branching (gate) |
| 16.3 | Routing | Dynamic routing based on input |
| 16.4 | Parallelization | Parallel execution and result aggregation |
| 16.5 | Orchestrator-Workers | Dynamic task distribution (Map-Reduce) |

### Technology Stack

- **Python 3.13**
- **LangGraph 0.6.6** -- Workflow graph composition framework
- **LangChain 0.3.27** -- LLM integration layer
- **OpenAI GPT-4o** -- Primary LLM model
- **Pydantic** -- Structured Output definition

---

## 2. Detailed Section Descriptions

---

### 16.0 Introduction -- Project Environment Setup

#### Topic and Objectives

Create a new project `workflow-architectures` and set up the development environment for LangGraph-based workflow experiments.

#### Core Concepts

In this section, we initialize a Python project using the `uv` package manager. All dependencies are declared in `pyproject.toml`, and Jupyter Notebook (`main.ipynb`) is used as the execution environment.

#### Code Analysis

**Project Dependencies (`pyproject.toml`):**

```toml
[project]
name = "workflow-architectures"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "python-dotenv==1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Key dependency descriptions:
- **`langgraph`**: The core library for composing state-based workflow graphs. Defines LLM call flows using nodes and edges.
- **`langchain[openai]`**: Provides integration with OpenAI models. Various models can be initialized with `init_chat_model()`.
- **`grandalf`**: A library for graph visualization.
- **`ipykernel`**: A development dependency for using the virtual environment's kernel in Jupyter Notebooks.

#### Practice Points

1. Learn how to create a project with `uv init` and add dependencies with `uv add`.
2. Understand the pattern of pinning the Python version with a `.python-version` file.
3. The `OPENAI_API_KEY` must be set in a `.env` file for LLM calls to work properly.

---

### 16.1 Prompt Chaining Architecture -- Sequential Prompt Chaining

#### Topic and Objectives

Implement the most basic workflow pattern, **Prompt Chaining**. Multiple LLM calls are **connected sequentially**, where the output of the previous step becomes the input of the next step, forming a pipeline.

#### Core Concepts

**Prompt Chaining** is a pattern that decomposes complex tasks into multiple smaller steps. Each step is handled by a single LLM call, and the result is stored in the State to be passed to the next step.

In this example, the cooking recipe generation process is decomposed into 3 steps:
1. **List ingredients** (list_ingredients) -- Generate a list of required ingredients for the dish
2. **Create recipe** (create_recipe) -- Generate cooking instructions based on the ingredients
3. **Describe plating** (describe_plating) -- Describe plating methods based on the recipe

```
START --> list_ingredients --> create_recipe --> describe_plating --> END
```

The key to this pattern is that **each step depends on the result of the previous step**. You cannot write a recipe without knowing the ingredients, and you cannot describe plating without knowing the recipe.

#### Code Analysis

**Step 1: State and Data Model Definition**

```python
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

llm = init_chat_model("openai:gpt-4o")
```

`init_chat_model()` is a universal model initialization function provided by LangChain. The provider and model name are specified in the format `"openai:gpt-4o"`.

```python
class State(TypedDict):
    dish: str
    ingredients: list[dict]
    recipe_steps: str
    plating_instructions: str
```

`State` is the **state object** shared across the entire workflow. `TypedDict` is used to explicitly specify the type of each field. In LangGraph, all nodes read and update this State.

```python
class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: str

class IngredientsOutput(BaseModel):
    ingredients: List[Ingredient]
```

Pydantic `BaseModel` is used to define **Structured Output**. This forces the LLM to return JSON conforming to a defined schema instead of free text.

**Step 2: Node Function Definition**

```python
def list_ingredients(state: State):
    structured_llm = llm.with_structured_output(IngredientsOutput)
    response = structured_llm.invoke(
        f"List 5-8 ingredients needed to make {state['dish']}"
    )
    return {
        "ingredients": response.ingredients,
    }
```

`with_structured_output(IngredientsOutput)` configures the LLM to automatically parse its response into the `IngredientsOutput` Pydantic model. This causes the LLM to return structured data in the form `{"ingredients": [{"name": "Chickpeas", "quantity": "1", "unit": "cup"}, ...]}`.

Each node function must return a **dictionary**. The key-value pairs in the returned dictionary are updated in the State.

```python
def create_recipe(state: State):
    response = llm.invoke(
        f"Write a step by step cooking instruction for {state['dish']}, "
        f"using these ingredients {state['ingredients']}",
    )
    return {
        "recipe_steps": response.content,
    }

def describe_plating(state: State):
    response = llm.invoke(
        f"Describe how to beautifully plate this dish {state['dish']} "
        f"based on this recipe {state['recipe_steps']}"
    )
    return {
        "plating_instructions": response.content,
    }
```

`create_recipe` references `state['ingredients']`, and `describe_plating` references `state['recipe_steps']`. This is the **essence of chaining** -- the output of the previous step is included in the next step's prompt.

**Step 3: Graph Construction and Execution**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_edge("list_ingredients", "create_recipe")
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

A graph builder is created with `StateGraph(State)`, nodes are registered with `add_node()`, and connections between nodes are defined with `add_edge()`. `START` and `END` are special nodes provided by LangGraph that represent the beginning and end of the graph.

```python
graph.invoke({"dish": "hummus"})
```

Passing the initial State to `graph.invoke()` runs the graph. Providing only the `dish` value allows each node to sequentially fill in the remaining fields.

#### Practice Points

1. Examine the return value of `graph.invoke()` to observe how each field gets populated.
2. Change the `dish` value and compare results for different dishes.
3. Understand the difference between nodes using `with_structured_output()` and nodes using plain `invoke()`.

---

### 16.2 Prompt Chaining Gate -- Conditional Gate

#### Topic and Objectives

Add a **conditional branch (Gate)** to Prompt Chaining that **re-executes** a previous step if certain conditions are not met.

#### Core Concepts

In real applications, LLM output may not always meet expectations. A **Gate** serves as a quality verification checkpoint that checks whether LLM output meets specific criteria. If the criteria are not met, the step is re-executed to obtain better results.

In this example, the process can only advance to the next step if the number of ingredients falls within the range of 3 to 8:

```
START --> list_ingredients --[gate]--> create_recipe --> describe_plating --> END
                ^                          |
                |    (if condition not met) |
                +----------<---------------+
```

#### Code Analysis

**Gate Function Definition:**

```python
def gate(state: State):
    ingredients = state["ingredients"]

    if len(ingredients) > 8 or len(ingredients) < 3:
        return False

    return True
```

The gate function receives the State and returns `True` or `False`. The next path in the graph is determined by this return value.

- **True**: Number of ingredients is within the 3-8 range -- proceed to `create_recipe`
- **False**: Outside the range -- re-execute `list_ingredients` (retry)

**Conditional Edge Setup:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_conditional_edges(
    "list_ingredients",
    gate,
    {
        True: "create_recipe",
        False: "list_ingredients",
    },
)
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

The key is the `add_conditional_edges()` method:
- **First argument**: Source node (`"list_ingredients"`)
- **Second argument**: Condition evaluation function (`gate`)
- **Third argument**: Mapping dictionary of return values to destination nodes

When `gate` returns `False`, it loops back to `"list_ingredients"`, forming a retry loop until the condition is satisfied.

#### Gate Pattern Use Cases

| Scenario | Gate Condition |
|----------|---------------|
| Code generation | Does the generated code pass syntax checking? |
| Translation | Is the translation result in the correct language? |
| Data extraction | Are all required fields filled? |
| Summarization | Is the summary length appropriate? |

#### Practice Points

1. Modify the `gate` function's conditions and observe how the number of retries changes.
2. Consider how to add a maximum retry count to prevent infinite loops (e.g., adding a `retry_count` field to State).
3. It is also possible to branch into multiple paths from the gate instead of just `True`/`False`.

---

### 16.3 Routing Architecture -- Dynamic Routing

#### Topic and Objectives

Implement a **Routing** pattern that branches to **different processing paths** based on input characteristics. The LLM classifies the input, and the appropriate model or processing logic is selected based on the classification result.

#### Core Concepts

The Routing pattern is based on the idea that "not every task needs the same approach." Using an expensive model for easy questions is wasteful, and using a weak model for difficult questions degrades quality.

In this example, the difficulty of a question is automatically assessed, and a model matching the difficulty is selected to generate the response:

```
                    +--> dumb_node (GPT-3.5) ---+
                    |                           |
START --> assess_difficulty --> average_node (GPT-4o) --> END
                    |                           |
                    +--> smart_node (GPT-5) ----+
```

#### Code Analysis

**Model Initialization:**

```python
llm = init_chat_model("openai:gpt-4o")

dumb_llm = init_chat_model("openai:gpt-3.5-turbo")
average_llm = init_chat_model("openai:gpt-4o")
smart_llm = init_chat_model("openai:gpt-5-2025-08-07")
```

Three LLMs of different capability levels are prepared. In production, this strategy is frequently used to balance cost and performance.

**State and Schema Definition:**

```python
class State(TypedDict):
    question: str
    difficulty: str
    answer: str
    model_used: str

class DifficultyResponse(BaseModel):
    difficulty_level: Literal["easy", "medium", "hard"]
```

`DifficultyResponse` uses the `Literal` type to force the LLM to select only one of `"easy"`, `"medium"`, or `"hard"`. This is a powerful advantage of Structured Output -- it constrains LLM responses to a programmatically controllable form.

**Difficulty Assessment and Routing Node:**

```python
def assess_difficulty(state: State):
    structured_llm = llm.with_structured_output(DifficultyResponse)

    response = structured_llm.invoke(
        f"""
        Assess the difficulty of this question
        Question: {state["question"]}

        - EASY: Simple facts, basic definitions, yes/no answers
        - MEDIUM: Requires explanation, comparison, analysis
        - HARD: Complex reasoning, multiple steps, deep expertise.
        """
    )

    difficulty_level = response.difficulty_level

    if difficulty_level == "easy":
        goto = "dumb_node"
    elif difficulty_level == "medium":
        goto = "average_node"
    elif difficulty_level == "hard":
        goto = "smart_node"

    return Command(
        goto=goto,
        update={
            "difficulty": difficulty_level,
        },
    )
```

There are two important concepts in this function:

1. **`Command` object**: LangGraph's `Command` performs **state update and routing simultaneously**. `goto` specifies the next node to execute, and `update` updates the State. Unlike the previous `add_conditional_edges()`, routing can be decided directly within the node function.

2. **Difficulty assessment prompt**: Clear criteria for each difficulty level are provided to guide the LLM toward consistent classification.

**Processing Nodes:**

```python
def dumb_node(state: State):
    response = dumb_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-3.5",
    }

def average_node(state: State):
    response = average_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-4o",
    }

def smart_node(state: State):
    response = smart_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-o3",
    }
```

Each node answers the question using its assigned LLM. The `model_used` field allows tracking which model was used.

**Graph Construction:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("dumb_node", dumb_node)
graph_builder.add_node("average_node", average_node)
graph_builder.add_node("smart_node", smart_node)
graph_builder.add_node(
    "assess_difficulty",
    assess_difficulty,
    destinations=(
        "dumb_node",
        "average_node",
        "smart_node",
    ),
)

graph_builder.add_edge(START, "assess_difficulty")
graph_builder.add_edge("dumb_node", END)
graph_builder.add_edge("average_node", END)
graph_builder.add_edge("smart_node", END)

graph = graph_builder.compile()
```

Nodes that return `Command` should have a `destinations` parameter added. This tells LangGraph which nodes this node can route to. It is used for graph visualization and validation.

**Execution:**

```python
graph.invoke({"question": "Investment potential of Uranium in 2026"})
```

This question requires complex analysis, so it would be classified as `"hard"` and routed to `smart_node` (GPT-5).

#### Practice Points

1. Input questions of various difficulties and verify that routing works correctly.
2. Check the `model_used` field to verify which model was actually selected.
3. Design scenarios that route to different prompt templates or tools, beyond just model selection.

---

### 16.4 Parallelization Architecture -- Parallel Execution

#### Topic and Objectives

Implement a **Parallelization** pattern that executes multiple LLM calls **simultaneously in parallel** and then performs **aggregation** once all results are collected.

#### Core Concepts

Sequential execution is simple, but it is inefficient when there are multiple independent tasks. For example, performing summary, sentiment analysis, key point extraction, and recommendation derivation sequentially on a document takes 4 times as long. Since these tasks do not depend on each other, they can be executed simultaneously.

In this example, a Fed chair's press conference transcript is analyzed from 4 perspectives simultaneously:

```
            +--> get_summary --------+
            |                        |
            +--> get_sentiment ------+
START ----->|                        +--> get_final_analysis --> END
            +--> get_key_points -----+
            |                        |
            +--> get_recommendation -+
```

The 4 analysis nodes run **simultaneously**, and once all nodes complete, `get_final_analysis` performs the comprehensive analysis.

#### Code Analysis

**State Definition:**

```python
class State(TypedDict):
    document: str
    summary: str
    sentiment: str
    key_points: str
    recommendation: str
    final_analysis: str
```

Each parallel node has its own dedicated field. These fields are independent of each other, so updating them simultaneously causes no conflicts.

**Parallel Node Functions:**

```python
def get_summary(state: State):
    response = llm.invoke(
        f"Write a 3-sentence summary of this document {state['document']}"
    )
    return {"summary": response.content}

def get_sentiment(state: State):
    response = llm.invoke(
        f"Analyse the sentiment and tone of this document {state['document']}"
    )
    return {"sentiment": response.content}

def get_key_points(state: State):
    response = llm.invoke(
        f"List the 5 most important points of this document {state['document']}"
    )
    return {"key_points": response.content}

def get_recommendation(state: State):
    response = llm.invoke(
        f"Based on the document, list 3 recommended next steps {state['document']}"
    )
    return {"recommendation": response.content}
```

All 4 functions read the same `state["document"]` but store results in different fields. This is why parallel execution is possible -- reads are shared, but writes are separated.

**Aggregation Node:**

```python
def get_final_analysis(state: State):
    response = llm.invoke(
        f"""
    Give me an analysis of the following report

    DOCUMENT ANALYSIS REPORT
    ========================

    EXECUTIVE SUMMARY:
    {state['summary']}

    SENTIMENT ANALYSIS:
    {state['sentiment']}

    KEY POINTS:
    {state.get("key_points", "")}

    RECOMMENDATIONS:
    {state.get('recommendation', "N/A")}
    """
    )
    return {"final_analysis": response.content}
```

The aggregation node synthesizes all fields populated by the parallel nodes to generate the final analysis. This node is only executed **after all parallel nodes have completed**.

**Graph Construction -- The Key to Parallel Edges:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("get_summary", get_summary)
graph_builder.add_node("get_sentiment", get_sentiment)
graph_builder.add_node("get_key_points", get_key_points)
graph_builder.add_node("get_recommendation", get_recommendation)
graph_builder.add_node("get_final_analysis", get_final_analysis)

# Connecting 4 nodes from START simultaneously = parallel execution!
graph_builder.add_edge(START, "get_summary")
graph_builder.add_edge(START, "get_sentiment")
graph_builder.add_edge(START, "get_key_points")
graph_builder.add_edge(START, "get_recommendation")

# All 4 nodes connect to get_final_analysis = execute after all complete
graph_builder.add_edge("get_summary", "get_final_analysis")
graph_builder.add_edge("get_sentiment", "get_final_analysis")
graph_builder.add_edge("get_key_points", "get_final_analysis")
graph_builder.add_edge("get_recommendation", "get_final_analysis")

graph_builder.add_edge("get_final_analysis", END)

graph = graph_builder.compile()
```

Implementing parallel execution in LangGraph is very intuitive:
- **Connecting edges from `START` to multiple nodes** causes those nodes to run simultaneously.
- **Connecting edges from multiple nodes to a single node** causes it to wait until all preceding nodes complete before executing (join/barrier).

**Streaming Execution:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

Using `graph.stream()`, results can be received each time a node completes execution. `stream_mode="updates"` streams only the State updates from each node. Since parallel node results are output in the order they complete, you can see in real time which task finishes first.

#### Practice Points

1. Compare total elapsed time between sequential and parallel execution.
2. Observe the order in which results arrive with `stream_mode="updates"` -- which analysis completes fastest?
3. Measure performance changes by increasing or decreasing the number of parallel nodes.
4. Evaluate the quality of analysis results using an actual document (Fed transcript).

---

### 16.5 Orchestrator-Workers Architecture

#### Topic and Objectives

Implement an **Orchestrator-Workers (Map-Reduce)** pattern that **dynamically creates workers based on input**, distributes tasks, and collects all worker results to produce the final output.

#### Core Concepts

Section 16.4's Parallelization had a **fixed number of nodes** (always 4 analysis nodes). However, in real scenarios, the number of parallel tasks often needs to vary based on input. For example:
- A document with 3 paragraphs needs 3 summary workers
- A document with 20 paragraphs needs 20 summary workers

In this pattern, the **orchestrator (dispatcher)** analyzes the input and dynamically creates as many workers as needed. This is implemented using LangGraph's `Send` API.

```
            +--> summarize_p (paragraph 0) --+
            |                                |
            +--> summarize_p (paragraph 1) --+
START ----->|                                +--> final_summary --> END
            +--> summarize_p (paragraph 2) --+
            |                                |
            +--> summarize_p (paragraph N) --+
```

The number of workers (N) is determined at runtime based on the number of paragraphs in the document.

#### Code Analysis

**New Imports:**

```python
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.types import Send
from operator import add
```

- **`Send`**: An object that directs execution by passing specific arguments to a specific node
- **`Annotated` and `add`**: Used for defining **reducers** on list fields

**State Definition -- Annotated Reducer:**

```python
class State(TypedDict):
    document: str
    final_summary: str
    summaries: Annotated[list[dict], add]
```

`Annotated[list[dict], add]` is a core LangGraph concept called a **reducer**. When multiple workers return values to the `summaries` field simultaneously, the default behavior would be to overwrite with the last value. However, by specifying the `add` reducer, **all worker results are accumulated in the list**.

For example, if worker A returns `{"summaries": [item_a]}` and worker B returns `{"summaries": [item_b]}`, the final `summaries` will be `[item_a, item_b]`. This is because `operator.add` performs list `+` (concatenation).

**Worker Node:**

```python
def summarize_p(args):
    paragraph = args["paragraph"]
    index = args["index"]
    response = llm.invoke(
        f"Write a 3-sentence summary for this paragraph: {paragraph}",
    )
    return {
        "summaries": [
            {
                "summary": response.content,
                "index": index,
            }
        ],
    }
```

Key points:
- The parameter for this function is `args`, not `state`. It receives custom arguments passed through `Send`.
- `index` is stored alongside the summary to enable restoring paragraph order later.
- The `summaries` return value is wrapped in a list -- this is required for use with the `add` reducer.

**Orchestrator (Dispatcher) Function:**

```python
def dispatch_summarizers(state: State):
    chunks = state["document"].split("\n\n")
    return [
        Send("summarize_p", {"paragraph": chunk, "index": index})
        for index, chunk in enumerate(chunks)
    ]
```

This is the core of the Orchestrator-Workers pattern:

1. The document is split by `"\n\n"` (blank lines) to create a list of paragraphs.
2. A `Send("summarize_p", {...})` object is created for each paragraph.
3. The first argument of `Send` is the node name to execute, and the second is the data to pass to that node.
4. Returning a list of `Send` objects causes LangGraph to **execute those nodes simultaneously in parallel**.

If the document has 15 paragraphs, 15 instances of `summarize_p` will execute simultaneously. This is the key difference from the static parallelization of section 16.4.

**Final Aggregation Node:**

```python
def final_summary(state: State):
    response = llm.invoke(
        f"Using the following summaries, give me a final one {state['summaries']}"
    )
    return {
        "final_summary": response.content,
    }
```

Once all worker summaries are collected in the `summaries` list, the `final_summary` node synthesizes them to generate the final summary.

**Graph Construction:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("summarize_p", summarize_p)
graph_builder.add_node("final_summary", final_summary)

graph_builder.add_conditional_edges(
    START,
    dispatch_summarizers,
    ["summarize_p"],
)

graph_builder.add_edge("summarize_p", "final_summary")
graph_builder.add_edge("final_summary", END)

graph = graph_builder.compile()
```

`add_conditional_edges()` is used here, but for a different purpose than section 16.2's gate:
- **Third argument `["summarize_p"]`**: The list of possible destination nodes. When `dispatch_summarizers` returns `Send` objects, those nodes are dynamically created.
- LangGraph waits until all `Send` operations complete before proceeding to the next edge (`"summarize_p"` --> `"final_summary"`).

**Execution:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

When run with streaming, results are output as each paragraph's summary completes. You can confirm that the number of workers automatically adjusts based on the number of paragraphs.

#### Practice Points

1. Test the number of `Send` objects `dispatch_summarizers` creates with various documents.
2. Verify what problems occur when the `Annotated[list[dict], add]` reducer is removed.
3. Add post-processing logic that sorts results in original order using the `index` field.
4. Apply the same pattern to other tasks (translation, keyword extraction, etc.) beyond summarization.

---

## 3. Chapter Key Summary

### Architecture Pattern Comparison Table

| Pattern | Execution Mode | Node Count | Suitable Situations | LangGraph API |
|---------|---------------|------------|---------------------|---------------|
| **Prompt Chaining** | Sequential | Fixed | Tasks with step-by-step dependencies | `add_edge()` |
| **Prompt Chaining + Gate** | Sequential + retry | Fixed | Tasks requiring quality verification | `add_conditional_edges()` |
| **Routing** | Branching | Fixed | Tasks requiring different processing based on input characteristics | `Command(goto=...)` |
| **Parallelization** | Parallel | Fixed | Performing multiple independent analyses simultaneously | Multiple `add_edge(START, ...)` |
| **Orchestrator-Workers** | Dynamic parallel | Variable | Cases where the number of tasks varies based on input size | `Send()` + `Annotated[..., add]` |

### Core LangGraph Concepts Summary

1. **StateGraph**: The basic building block of state-based graphs. Nodes share state defined with `TypedDict`.
2. **add_edge()**: Defines fixed paths between nodes.
3. **add_conditional_edges()**: Dynamically determines the next node based on a function's return value.
4. **Command**: Performs routing and state updates simultaneously from within a node function.
5. **Send**: Implements dynamic parallel execution by passing custom arguments to a specific node.
6. **Annotated + Reducer**: Defines the merge strategy when multiple nodes add values to the same field.
7. **with_structured_output()**: Structures LLM output into Pydantic models for programmatic use.

---

## 4. Practice Exercises

### Exercise 1: Multi-Language Translation Chain (Prompt Chaining)

Referencing the cooking recipe example, implement the following pipeline:
1. The user inputs Korean text.
2. The first node translates it to English.
3. The second node corrects the English translation's grammar.
4. The third node back-translates the corrected English into natural Korean.

**Bonus**: Add a gate that evaluates whether the back-translated result is similar to the original text.

### Exercise 2: Customer Inquiry Routing System (Routing)

Implement a system that classifies customer inquiries and routes them to the appropriate processing path:
- **Technical support**: Provide detailed technical solutions
- **Payment inquiry**: Provide payment-related information
- **General inquiry**: Provide simple answers
- **Complaint handling**: Empathetic response + solution proposal

Design each path to use a different prompt template.

### Exercise 3: News Article Comprehensive Analyzer (Parallelization)

Implement a system that takes a news article as input and performs the following analyses **simultaneously**:
- 3-line summary
- Sentiment/tone analysis
- Related keyword extraction (using Structured Output)
- Fact-check point listing
- Reader impact assessment

Generate a comprehensive report once all analyses are complete.

### Exercise 4: Large Document Processor (Orchestrator-Workers)

Implement a system that takes a PDF or long text document as input and performs the following:
1. Split the document into appropriately sized chunks.
2. Dynamically create workers for each chunk to perform keyword extraction and summarization simultaneously.
3. Collect all worker results to generate a final document summary and keyword cloud data.

**Hint**: Use the `Annotated[list[dict], add]` reducer to accumulate worker results.

### Exercise 5: Integrated Architecture Design (Comprehensive)

Combine all the above patterns to design an "AI Essay Writing Assistant":
1. **Routing**: Assess the difficulty of the essay topic and determine the appropriate research depth
2. **Orchestrator-Workers**: Decompose the topic into multiple subtopics and conduct research on each
3. **Parallelization**: Simultaneously draft the introduction, body, and conclusion based on research results
4. **Prompt Chaining + Gate**: Edit the draft and rewrite until quality criteria are met

---

> **Note**: All code in this chapter can be run in the `workflow-architectures/main.ipynb` notebook. Before running, set the OpenAI API key in the `.env` file and install dependencies with `uv sync`.
