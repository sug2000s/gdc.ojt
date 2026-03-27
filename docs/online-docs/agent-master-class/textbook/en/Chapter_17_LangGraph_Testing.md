# Chapter 17: LangGraph Workflow Testing

---

## 1. Chapter Overview

In this chapter, we learn **how to systematically test AI agent workflows built with LangGraph**. Starting from a simple rule-based graph, we transition to AI (LLM)-based nodes and then progressively cover how to reliably verify non-deterministic AI responses.

### Learning Objectives

- Building an email processing workflow using LangGraph's `StateGraph`
- Setting up a graph test framework with `pytest`
- Individual node unit testing and Partial Execution testing
- Understanding the process of converting rule-based nodes to AI (LLM) nodes
- Establishing test strategies suited to the non-deterministic nature of AI responses
- Evaluating AI response quality using the LLM-as-a-Judge pattern

### Project Structure

```
workflow-testing/
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py          # LangGraph workflow definition
├── tests.py         # pytest test code
└── README.md
```

---

## 2. Detailed Section Descriptions

---

### 17.0 Introduction -- Initial Project Setup

**Topic and Goal:** Set up the Python project environment for testing exercises.

#### Key Concepts

In this section, we create a new Python project using the `uv` package manager. Dependencies are defined in `pyproject.toml`, and the roles of the core libraries are as follows:

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain[openai]` | 0.3.27 | LLM integration framework |
| `langgraph` | 0.6.6 | Workflow graph builder |
| `langgraph-checkpoint-sqlite` | 2.0.11 | State checkpoint storage |
| `pytest` | 8.4.2 | Python test framework |
| `python-dotenv` | 1.1.1 | Environment variable (.env) loading |

#### Code Analysis

```toml
# pyproject.toml
[project]
name = "workflow-testing"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]
```

**Points to note:**
- Requires Python 3.13 or higher. The `.python-version` file specifies `3.13`, so `uv` automatically uses the correct Python version.
- `pytest` is included in the project dependencies. This shows that testing is a core component of the development process.
- `grandalf` is a library for graph visualization.

#### Practice Points

1. Create a project with the `uv init workflow-testing` command.
2. Add dependencies with commands like `uv add langgraph pytest langchain[openai]`.
3. Run `uv sync` to verify that all packages install correctly.

---

### 17.1 Email Graph -- Building an Email Processing Workflow

**Topic and Goal:** Build a 3-step workflow that classifies emails, assigns priorities, and generates responses using LangGraph's `StateGraph`.

#### Key Concepts

A LangGraph workflow consists of three core elements:

1. **State:** A data structure shared across the entire workflow. Defined with `TypedDict`, where each node reads and writes parts of the state.
2. **Node:** A function that receives the state as input, processes it, and returns state updates.
3. **Edge:** A connection that defines the execution order between nodes.

The workflow flow built in this section:

```
START --> categorize_email --> assing_priority --> draft_response --> END
```

#### Code Analysis

**Step 1: State Definition**

```python
from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class EmailState(TypedDict):
    email: str
    category: Literal["spam", "normal", "urgent"]
    priority_score: int
    response: str
```

`EmailState` defines the schema of the data the workflow will process. The `Literal` type constrains `category` to only take one of "spam", "normal", or "urgent". The state includes the original email (`email`), classification result (`category`), priority score (`priority_score`), and generated response (`response`).

**Step 2: Node Function Definitions**

```python
def categorize_email(state: EmailState):
    email = state["email"].lower()

    if "urgent" in email or "asap" in email:
        category = "urgent"
    elif "offer" in email or "discount" in email:
        category = "spam"
    else:
        category = "normal"

    return {
        "category": category,
    }
```

The `categorize_email` node classifies categories based on keywords contained in the email body. It uses simple rule-based logic: if "urgent" or "asap" is present, it's urgent; if "offer" or "discount" is present, it's spam; otherwise, it's normal.

**Key point:** Each node function receives the full state (`EmailState`) as a parameter but returns a dictionary containing only the fields it modifies. LangGraph **merges** this return value into the existing state.

```python
def assing_priority(state: EmailState):
    scores = {
        "urgent": 10,
        "normal": 5,
        "spam": 1,
    }
    return {
        "priority_score": scores[state["category"]],
    }


def draft_response(state: EmailState) -> EmailState:
    responses = {
        "urgent": "I will answer you as fast as i can",
        "normal": "I'll get back to you soon",
        "spam": "Go away!",
    }
    return {
        "response": responses[state["category"]],
    }
```

`assing_priority` assigns a fixed score per category, and `draft_response` generates a fixed response message per category. At this point, all logic is deterministic -- the same input always guarantees the same output.

**Step 3: Graph Assembly and Execution**

```python
graph_builder = StateGraph(EmailState)

graph_builder.add_node("categorize_email", categorize_email)
graph_builder.add_node("assing_priority", assing_priority)
graph_builder.add_node("draft_response", draft_response)

graph_builder.add_edge(START, "categorize_email")
graph_builder.add_edge("categorize_email", "assing_priority")
graph_builder.add_edge("assing_priority", "draft_response")
graph_builder.add_edge("draft_response", END)

graph = graph_builder.compile()

result = graph.invoke({"email": "i have an offer for you!"})
print(result)
```

Nodes are registered in `StateGraph`, edges specify the order, and `compile()` creates an executable graph. Passing the initial state (email body) to `invoke()` runs the entire workflow sequentially.

#### Practice Points

1. Call `graph.invoke()` with various email texts besides "i have an offer for you!" and check the results.
2. Think about which parts need to be modified to add a new category (e.g., "important").
3. Try using conditional branching (`add_conditional_edges`) to skip `draft_response` for spam emails.

---

### 17.2 Pytest -- Introducing the Test Framework

**Topic and Goal:** Write automated tests for LangGraph workflows using `pytest`. Learn parameterized testing with `@pytest.mark.parametrize`.

#### Key Concepts

**pytest** is Python's leading test framework. Key features:

- Functions starting with `test_` are automatically recognized as tests
- Concise verification with `assert` statements
- `@pytest.mark.parametrize` runs the same test logic repeatedly with different input values

**Parameterized Testing** is a key technique for eliminating test code duplication. A single test function can cover multiple scenarios.

#### Code Analysis

First, remove the direct execution code (invoke + print) from `main.py`:

```python
# Removed code (bottom of main.py)
# result = graph.invoke({"email": "i have an offer for you!"})
# print(result)
```

Separating production code from test code is a fundamental principle. `main.py` is responsible only for graph definition, while execution and verification are performed in `tests.py`.

```python
# tests.py
import pytest
from main import graph


@pytest.mark.parametrize(
    "email, expected_category, expected_score",
    [
        ("this is urgent!", "urgent", 10),
        ("i wanna talk to you", "normal", 5),
        ("i have an offer for you", "spam", 1),
    ],
)
def test_full_graph(email, expected_category, expected_score):

    result = graph.invoke({"email": email})

    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

**Code explanation:**

1. `from main import graph`: Imports the compiled graph from `main.py`.
2. `@pytest.mark.parametrize`: The first argument of the decorator is the parameter names (comma-separated string), and the second argument is the list of test cases.
3. Each tuple `("this is urgent!", "urgent", 10)` represents one test case.
4. `graph.invoke()` runs the full workflow, and `assert` verifies the expected results.

This test runs as 3 independent test cases:
- Urgent email -> category="urgent", priority_score=10
- Normal email -> category="normal", priority_score=5
- Spam email -> category="spam", priority_score=1

#### Practice Points

1. Run `pytest tests.py -v` in the terminal to see each test case executed individually. (`-v` is verbose mode)
2. Intentionally set incorrect expected values to see the failure messages. Understand what information pytest provides.
3. Add edge cases: What result do you get with "URGENT offer" where both keywords are present?

---

### 17.3 Testing Nodes -- Node Unit Testing and Partial Execution

**Topic and Goal:** Beyond full graph execution, learn advanced testing techniques: (1) testing individual nodes independently, and (2) injecting intermediate state into the graph to perform partial execution from a specific point.

#### Key Concepts

There are three levels of workflow testing:

| Test Level | Description | Use Case |
|------------|-------------|----------|
| **Full Graph Test** | Run from start to finish with `graph.invoke()` | Integration testing, E2E verification |
| **Individual Node Test** | Run a specific node with `graph.nodes["node_name"].invoke()` | Unit testing, node logic verification |
| **Partial Execution Test** | Inject intermediate state with `graph.update_state()` then continue execution | Reproducing specific scenarios, debugging |

**MemorySaver (Checkpointer):** For partial execution, the graph must be able to save state. `MemorySaver` is a memory-based checkpointer that saves the graph's execution state per `thread_id`.

#### Code Analysis

**Adding Checkpointer (main.py):**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# ... (node, edge definitions omitted) ...

graph = graph_builder.compile(checkpointer=checkpointer)
```

When `checkpointer` is passed to `compile()`, the graph automatically saves state after each node execution. You can then query or modify the state of a specific execution through the `thread_id`.

**Modified Full Graph Test:**

```python
def test_full_graph(email, expected_category, expected_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

Graphs using a checkpointer must provide a `thread_id` in the `config`.

**Individual Node Test:**

```python
def test_individual_nodes():

    # Run only the categorize_email node
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # Run only the assing_priority node
    result = graph.nodes["assing_priority"].invoke({"category": "spam"})
    assert result["priority_score"] == 1

    # Run only the draft_response node
    result = graph.nodes["draft_response"].invoke({"category": "spam"})
    assert "Go away" in result["response"]
```

`graph.nodes` is a dictionary of registered nodes. Each node has an `invoke()` method and can be executed independently by passing only the state needed by that node function. This means:

- `categorize_email` only needs the `email` field
- `assing_priority` only needs the `category` field
- `draft_response` only needs the `category` field

By isolating each node's inputs and outputs for testing, you can quickly identify which node has a bug when issues arise.

**Partial Execution Test:**

```python
def test_partial_execution():

    # Step 1: Directly inject intermediate state
    graph.update_state(
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        values={
            "email": "please check out this offer",
            "category": "spam",
        },
        as_node="categorize_email",  # Set state as if this node was executed
    )

    # Step 2: Continue execution from the injected state
    result = graph.invoke(
        None,  # Continue from existing state without new input
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        interrupt_after="draft_response",
    )

    assert result["priority_score"] == 1
```

Key behaviors of this test:

1. `update_state()` injects state as if the `categorize_email` node has already completed. `as_node="categorize_email"` means "this state is the output of the categorize_email node."
2. `graph.invoke(None, ...)` continues execution from the saved state without input. It starts from `assing_priority`, the next node after `categorize_email`.
3. `interrupt_after="draft_response"` stops execution after `draft_response` runs.

This technique is useful in the following situations:
- When preceding nodes have high execution costs (e.g., LLM calls)
- When you only want to verify behavior at a specific intermediate state
- When you need to artificially create edge cases

#### Practice Points

1. Print what keys exist in `graph.nodes`.
2. Change `as_node` in `update_state()` to `"assing_priority"` and try running from a later node.
3. Check what happens when you call `invoke(None, ...)` with a non-existent `thread_id`.

---

### 17.4 AI Nodes -- Transitioning from Rule-Based to LLM-Based

**Topic and Goal:** Replace hardcoded rule-based nodes with AI nodes powered by LLM (GPT-4o). Use Pydantic's `BaseModel` and LangChain's `with_structured_output` to structure LLM output.

#### Key Concepts

Limitations of rule-based systems:
- Cannot handle urgent emails that don't contain the word "urgent"
- Must manually add if/elif conditions for every new pattern
- Difficult to cover the diverse expressions of natural language

Using an LLM allows classification based on the **semantics** of natural language. However, since LLM output is free-form text, **Structured Output** is used to force it into a programmatically processable format.

**Structured Output Pattern:**
1. Define the desired output schema with a Pydantic `BaseModel`
2. Wrap the LLM with `llm.with_structured_output(Model)`
3. Force the LLM to always return JSON matching that schema

#### Code Analysis

**LLM Initialization and Output Schema Definition:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class EmailClassificationOuput(BaseModel):
    category: Literal["spam", "normal", "urgent"] = Field(
        description="Category of the email",
    )


class PriorityScoreOutput(BaseModel):
    priority_score: int = Field(
        description="Priority score from 1 to 10",
        ge=1,
        le=10,
    )
```

`EmailClassificationOuput` forces the LLM to return one of "spam", "normal", or "urgent". `PriorityScoreOutput` includes `ge` (greater than or equal) and `le` (less than or equal) validation to ensure an integer between 1 and 10 is returned.

**AI-Based categorize_email:**

```python
def categorize_email(state: EmailState):
    s_llm = llm.with_structured_output(EmailClassificationOuput)

    result = s_llm.invoke(
        f"""Classify this email into one of three categories:
        - urgent: time-sensitive, requires immediate attention
        - normal: regular business communication
        - spam: promotional, marketing, or unwanted content

        Email: {state['email']}"""
    )

    return {
        "category": result.category,
    }
```

Instead of keyword matching, the classification criteria are passed to the LLM via a prompt. The `s_llm` returned by `with_structured_output()` always returns an `EmailClassificationOuput` instance. You can access the value in a type-safe manner with `result.category`.

**AI-Based assing_priority:**

```python
def assing_priority(state: EmailState):
    s_llm = llm.with_structured_output(PriorityScoreOutput)

    result = s_llm.invoke(
        f"""Assign a priority score from 1-10 for this {state['category']} email.
        Consider:
        - Category: {state['category']}
        - Email content: {state['email']}

        Guidelines:
        - Urgent emails: usually 8-10
        - Normal emails: usually 4-7
        - Spam emails: usually 1-3"""
    )

    return {"priority_score": result.priority_score}
```

Instead of a fixed mapping (`urgent=10, normal=5, spam=1`), the LLM comprehensively considers the email content and category to assign a flexible score within the 1-10 range. The prompt includes guidelines to indicate score ranges per category.

**AI-Based draft_response:**

```python
def draft_response(state: EmailState) -> EmailState:
    result = llm.invoke(
        f"""Draft a brief, professional response for this {state['category']} email.

        Original email: {state['email']}
        Category: {state['category']}
        Priority: {state['priority_score']}/10

        Guidelines:
        - Urgent: Acknowledge urgency, promise immediate attention
        - Normal: Professional acknowledgment, standard timeline
        - Spam: Brief notice that message was filtered

        Keep response under 2 sentences."""
    )
    return {
        "response": result.content,
    }
```

This node uses a regular LLM without structured output, since the response is free-form text. The LLM's text response is accessed via `result.content`.

**Rule-Based vs AI-Based Comparison:**

| Item | Rule-Based (17.1) | AI-Based (17.4) |
|------|-------------------|-----------------|
| Classification method | Keyword matching | Semantic understanding |
| Score assignment | Fixed value per category | Context-based flexible value |
| Response generation | Fixed templates | Dynamic generation |
| Determinism | Deterministic (same input = same output) | Non-deterministic (same input may produce different output) |
| Test difficulty | Easy (exact value comparison) | Hard (range and semantic comparison needed) |

#### Practice Points

1. Test with emails like "Please help me, my server is down and clients are complaining!" that are urgent without containing the "urgent" keyword. Compare the difference between rule-based and AI-based results.
2. Add a new category (e.g., "inquiry") to `EmailClassificationOuput` and modify the prompt.
3. Change the `ge` and `le` ranges in `Field` to verify that Pydantic validation works.

---

### 17.5 Testing AI Nodes -- Test Strategies for AI Nodes

**Topic and Goal:** Learn strategies for effectively testing the non-deterministic output of AI (LLM)-based nodes. Transition from exact value comparison to range-based comparison.

#### Key Concepts

Introducing AI nodes breaks existing tests. The reasons:

1. **Category classification:** The LLM classifies correctly, but there may be subtle interpretation differences even for the same input.
2. **Priority score:** Returns a range (e.g., 8-10) instead of a fixed value (e.g., 10).
3. **Response text:** Different sentences are generated each time.

Therefore, the key principle for testing AI nodes is:

> **Verify acceptable ranges instead of exact values.**

#### Code Analysis

**Adding Environment Variable Loading:**

```python
import dotenv
dotenv.load_dotenv()
```

Since AI nodes call the OpenAI API, `OPENAI_API_KEY` must be loaded from the `.env` file. It is important that this code is placed **at the top of the file**. When `from main import graph` is executed, `init_chat_model("openai:gpt-4o")` in `main.py` is called, so the environment variable must be loaded before that.

**Full Graph Test -- Switching to Range-Based:**

```python
@pytest.mark.parametrize(
    "email, expected_category, min_score, max_score",
    [
        ("this is urgent!", "urgent", 8, 10),
        ("i wanna talk to you", "normal", 4, 7),
        ("i have an offer for you", "spam", 1, 3),
    ],
)
def test_full_graph(email, expected_category, min_score, max_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert min_score <= result["priority_score"] <= max_score
```

Changes:
- `min_score` and `max_score` range replaces the single `expected_score`
- `assert min_score <= result["priority_score"] <= max_score` replaces `assert result["priority_score"] == expected_score`
- Category can still be compared exactly since it is enforced by Structured Output

**Modified Individual Node Test:**

```python
def test_individual_nodes():

    # categorize_email -- still allows exact value comparison
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # assing_priority -- switched to range comparison, email field added
    result = graph.nodes["assing_priority"].invoke(
        {"category": "spam", "email": "buy this pot."}
    )
    assert 1 <= result["priority_score"] <= 3

    # draft_response -- commented out (no appropriate verification method yet)
    # result = graph.nodes["draft_response"].invoke({"category": "spam"})
    # assert "Go away" in result["response"]
```

Points to note:
- The `email` field was added to `assing_priority`. The AI version also references email content in the prompt.
- `draft_response` is commented out. Since the AI generates different responses each time, keyword verification like `"Go away" in result["response"]` is not possible. This problem is solved in 17.6.

**Modified Partial Execution Test:**

```python
def test_partial_execution():
    # ... (update_state part is the same) ...

    result = graph.invoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after="draft_response",
    )
    assert 1 <= result["priority_score"] <= 3  # Range instead of fixed value 1
```

#### Practice Points

1. Deliberately set a narrow range (e.g., `min_score=10, max_score=10`) and observe how often the AI test fails.
2. Run the same test 10 times consecutively and check the distribution of results: `pytest tests.py -v --count=10` (requires pytest-repeat plugin)
3. Check the `draft_response` node output multiple times and think about what verification method would be appropriate.

---

### 17.6 Testing AI Responses -- LLM-as-a-Judge Pattern

**Topic and Goal:** Implement the **LLM-as-a-Judge** pattern to verify the quality of free-form AI responses. Test AI-generated text through example-based similarity evaluation.

#### Key Concepts

The reason `draft_response` test was commented out in 17.5 is that the AI generates different text each time. Exact string matching like "Go away!" is impossible.

The **LLM-as-a-Judge** pattern is a representative technique for solving this problem:

1. Pre-define **golden examples** of ideal responses per category.
2. Pass the test subject AI response along with examples to a **judge LLM**.
3. The judge LLM returns a similarity score.
4. If the score is above the threshold, the test passes.

Advantages of this pattern:
- Free-form text can be evaluated semantically
- Evaluation criteria can be easily adjusted by adding/modifying examples
- More flexible and robust than keyword matching

#### Code Analysis

**Similarity Score Output Schema:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class SimilarityScoreOutput(BaseModel):
    similarity_score: int = Field(
        description="How similar is the response to the examples?",
        gt=0,
        lt=100,
    )
```

Defines a schema for the judge LLM to return a similarity score between 0 and 100. `gt=0, lt=100` enforces a range of 1-99, excluding 0 and 100.

**Response Examples (Golden Examples):**

```python
RESPONSE_EXAMPLES = {
    "urgent": [
        "Thank you for your urgent message. We are addressing this immediately and will respond as soon as possible.",
        "We've received your urgent request and are prioritizing it. Our team is on it right away.",
        "This urgent matter has our immediate attention. We'll respond promptly.",
    ],
    "normal": [
        "Thank you for your email. We'll review it and get back to you within 24-48 hours.",
        "We've received your message and will respond soon. Thank you for reaching out.",
        "Thank you for contacting us. We'll process your request and respond shortly.",
        "Thank you for the update. I will review the information and follow up as needed.",
        "Thank you for the update on the project status. I will review and follow up by the end of the week.",
        "Thanks for sharing this update. We'll review and respond accordingly.",
    ],
    "spam": [
        "This message has been flagged as spam and filtered.",
        "This email has been identified as promotional content.",
        "This message has been marked as spam.",
    ],
}
```

Multiple examples per category provide what "a desirable response looks like." The more diverse the examples, the higher the accuracy of the judgment. The `normal` category has the most examples because responses to regular emails can vary the most.

**Judge Function:**

```python
def judge_response(response: str, category: str):

    s_llm = llm.with_structured_output(SimilarityScoreOutput)

    examples = RESPONSE_EXAMPLES[category]
    result = s_llm.invoke(
        f"""
        Score how similar this response is to the examples.

        Category: {category}

        Examples:
        {"\n".join(examples)}

        Response to evaluate:
        {response}

        Scoring criteria:
        - 90-100: Very similar in tone, content, and intent
        - 70-89: Similar with minor differences
        - 50-69: Moderately similar, captures main idea
        - 30-49: Some similarity but missing key elements
        - 0-29: Very different or inappropriate
    """
    )

    return result.similarity_score
```

How the `judge_response` function works:

1. Retrieves examples matching the category.
2. Passes both examples and the response under evaluation to the judge LLM.
3. Includes clear evaluation criteria (rubric) in the prompt.
4. Receives an integer score via structured output.

**Usage in Test Code:**

```python
def test_individual_nodes():

    # ... (categorize_email, assing_priority tests unchanged) ...

    # draft_response -- verified with LLM-as-a-Judge
    result = graph.nodes["draft_response"].invoke(
        {
            "category": "spam",
            "email": "Get rich quick!!! I have a pyramid scheme for you!",
            "priority_score": 1,
        }
    )

    similarity_score = judge_response(result["response"], "spam")
    assert similarity_score >= 70
```

A complete state (including category, email, and priority_score) is passed to the `draft_response` node, then the generated response is passed to `judge_response` for similarity evaluation. The test passes if the threshold of 70 or above is met.

**Meaning of the 70 Threshold:**
- According to the rubric, 70-89 means "similar with minor differences"
- If too high (e.g., 90), the test becomes unstable due to subtle expression differences
- If too low (e.g., 40), low-quality responses would pass
- 70 is a balanced point that allows "intent and tone match but expressions may differ"

#### Practice Points

1. Set thresholds to 50, 70, and 90 respectively and observe changes in test stability.
2. Add/remove examples in `RESPONSE_EXAMPLES` and see how the judgment results change.
3. Add tests that apply `judge_response` to `urgent` and `normal` categories as well.
4. Replace the judge LLM with a different model (e.g., `gpt-4o-mini`) and experiment with the cost-accuracy tradeoff.

---

## 3. Chapter Key Summary

### Evolution of Test Strategies

```
17.1 Rule-based graph    -->  17.2 Exact value comparison tests
        |                              |
17.4 AI-based graph      -->  17.5 Range-based tests
        |                              |
                             17.6 LLM-as-a-Judge tests
```

### Key Principles Summary

| Principle | Description |
|-----------|-------------|
| **Separate test levels** | Test full graph, individual nodes, and partial execution separately. |
| **Compare deterministic output exactly** | Categories enforced by Structured Output can be compared with `==`. |
| **Compare non-deterministic output with ranges** | Numbers generated by LLMs are verified with `min <= value <= max`. |
| **Use LLM to judge free-form text** | Use another LLM as a judge to evaluate semantic similarity. |
| **Golden Examples** | Pre-define ideal response examples for use as evaluation criteria. |
| **Specify evaluation criteria in prompts** | Clearly communicate scoring ranges and their meanings to the judge LLM. |
| **Partial execution with checkpointer** | Use `MemorySaver` and `update_state()` to test from specific points. |

### Technology Stack Summary

| Technology | Purpose |
|------------|---------|
| `langgraph.StateGraph` | Workflow graph definition |
| `langgraph.checkpoint.memory.MemorySaver` | In-memory state checkpointing |
| `pytest` + `@pytest.mark.parametrize` | Parameterized testing |
| `pydantic.BaseModel` + `Field` | LLM output schema definition and validation |
| `langchain.chat_models.init_chat_model` | LLM initialization |
| `llm.with_structured_output()` | Enforce structured output |

---

## 4. Practice Assignments

### Assignment 1: Add a New Category (Difficulty: Medium)

Add an `"inquiry"` category to the email classification.

- Add `"inquiry"` to the `category` Literal in `EmailState`
- Reflect this in `EmailClassificationOuput` as well
- Add "inquiry: questions or information requests" guideline to the classification prompt
- Add "Inquiry emails: usually 5-7" to the `PriorityScoreOutput` prompt
- Add 3 or more examples for the `"inquiry"` category in `RESPONSE_EXAMPLES`
- Add test cases for the new category to `test_full_graph`'s `parametrize`

### Assignment 2: Conditional Branching Graph (Difficulty: Medium)

Modify the graph so that spam emails skip the `draft_response` node and go directly to END.

- Use `add_conditional_edges` to branch after `assing_priority` based on category
- If spam, go to END; otherwise, go to `draft_response`
- Update the test code to match this change. For spam emails, the `response` field should not exist.

### Assignment 3: Advanced Judge LLM (Difficulty: Hard)

Improve the current `judge_response`.

- Instead of a single similarity score, create a schema that evaluates multiple dimensions (tone, professionalism, appropriateness, length) separately.
- Calculate the final score by averaging each dimension's score.
- Allow different weights per dimension (e.g., appropriateness 40%, tone 30%, professionalism 20%, length 10%).
- Make it so that when a test fails, it outputs which dimension scored low.

### Assignment 4: Test Stability Analysis (Difficulty: Hard)

Run the same test 20 times to analyze AI test stability.

- Install the `pytest-repeat` plugin.
- Run with `pytest tests.py --count=20 -v`.
- Tally the pass/fail ratio for each test.
- If there are failed cases, analyze the cause (threshold issue, prompt issue, or model issue).
- Write a report on what adjustments are needed to raise test stability above 95%.
