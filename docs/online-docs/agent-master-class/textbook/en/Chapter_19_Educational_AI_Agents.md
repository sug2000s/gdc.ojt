# Chapter 19: Educational AI Agents (Tutor Agent)

---

## 1. Chapter Overview

In Chapter 19, we build an **educational AI tutor agent system**. This system leverages LangGraph's multi-agent architecture to assess learners' levels and guide them with optimal learning methods, creating an intelligent educational platform.

### System Components

| Agent | Role | Learning Methodology |
|-------|------|---------------------|
| **Classification Agent** | Learner level assessment and routing | Educational assessment specialist |
| **Teacher Agent** | Structured step-by-step education | Structured lecture-style teaching |
| **Feynman Agent** | Feynman technique-based comprehension verification | "If you can't explain it simply, you don't understand it" |
| **Quiz Agent** | Quiz-based active learning assessment | Research-based multiple-choice quiz generation |

### Key Learning Objectives

- Multi-agent system design using LangGraph's `create_react_agent`
- Implementing agent transfer patterns
- Agent routing within graphs using `Command` objects
- Dynamic workflows using conditional edges
- Utilizing Pydantic-based Structured Output
- Implementing web search tools with Firecrawl

### Project Architecture

```
tutor-agent/
├── main.py                          # Main graph definition
├── langgraph.json                   # LangGraph configuration
├── pyproject.toml                   # Project dependencies
├── agents/
│   ├── classification_agent.py      # Learner classification agent
│   ├── teacher_agent.py             # Teacher agent
│   ├── feynman_agent.py             # Feynman technique agent
│   └── quiz_agent.py                # Quiz agent
└── tools/
    ├── shared_tools.py              # Shared tools (transfer, web search)
    └── quiz_tools.py                # Quiz generation tools
```

### Agent Flow Diagram

```
[START] → [router_check] ─→ [classification_agent] → [END]
                │
                ├─→ [teacher_agent]
                ├─→ [feynman_agent]
                └─→ [quiz_agent]
```

After `classification_agent` evaluates the learner, it transfers to the appropriate agent via the `transfer_to_agent` tool. When the conversation resumes, `router_check` verifies the `current_agent` state to route to the correct agent.

---

## 2. Detailed Section Descriptions

---

### 2.1 Section 19.0 -- Introduction (Initial Project Setup)

**Commit**: `0516cd0` "19.0 Introduction"

#### Topic and Objectives

Build the foundation for the new `tutor-agent` project. Set up the Python project structure and define all required dependency packages.

#### Core Concepts

##### Project Structure Initialization

A new project is created using `uv` (Python package manager). `uv` is a modern Python package management tool that is much faster than `pip`.

##### Python Version Management

```
3.13
```

The `.python-version` file specifies the Python version for the project. Tools like `pyenv` and `uv` automatically recognize this file and use the correct Python version.

##### Dependency Definition (pyproject.toml)

```toml
[project]
name = "tutor-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "firecrawl-py==2.16",
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

[dependency-groups]
dev = [
    "ipykernel==6.30.1",
]
```

**Key Package Descriptions:**

| Package | Version | Role |
|---------|---------|------|
| `firecrawl-py` | 2.16 | Web search and scraping API client |
| `grandalf` | 0.8 | Graph visualization (for LangGraph graph rendering) |
| `langchain[openai]` | 0.3.27 | LangChain framework + OpenAI integration |
| `langgraph` | 0.6.6 | State-based agent graph framework |
| `langgraph-checkpoint-sqlite` | 2.0.11 | SQLite-based checkpoint store |
| `langgraph-cli[inmem]` | 0.4.0 | LangGraph CLI (with in-memory mode) |
| `langgraph-supervisor` | 0.0.29 | Supervisor agent pattern |
| `langgraph-swarm` | 0.0.14 | Swarm agent pattern |
| `pytest` | 8.4.2 | Testing framework |
| `python-dotenv` | 1.1.1 | Load environment variables from `.env` files |

##### langgraph-supervisor vs langgraph-swarm

This project installs both multi-agent pattern libraries:
- **Supervisor pattern**: A central manager directs agents
- **Swarm pattern**: Agents autonomously pass tasks to each other

This project uses a pattern **closer to Swarm**, as each agent directly transfers to other agents through the `transfer_to_agent` tool.

#### Practice Points

1. Create the project with `uv init tutor-agent`, then modify `pyproject.toml` to add dependencies.
2. Run `uv sync` to install all dependencies.
3. Create a `.env` file and set `OPENAI_API_KEY` and `FIRECRAWL_API_KEY`.

---

### 2.2 Section 19.1 -- Classification Agent (Learner Classification Agent)

**Commit**: `269599b` "19.1 Classification Agent"

#### Topic and Objectives

Implement a **classification agent** that identifies the learner's level, learning style, and learning goals to connect them with the optimal learning agent. This agent serves as the system's entry point.

#### Core Concepts

##### Understanding create_react_agent

LangGraph's `create_react_agent` is a function that conveniently creates agents following the **ReAct (Reasoning + Acting) pattern**.

```python
from langgraph.prebuilt import create_react_agent

classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="...",       # System prompt
    tools=[...],        # Available tool list
)
```

**What is the ReAct Pattern?**
- **Reasoning**: The LLM analyzes the current situation and decides the next action
- **Acting**: Calls tools or generates responses based on the decision
- These two steps repeat until the task is completed

##### Classification Agent's Assessment Process

```python
classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are an Educational Assessment Specialist. Your role is to understand
    each learner's knowledge level, learning style, and educational needs
    through conversation.

    ## Your Assessment Process:

    ### Phase 1: Topic & Current Knowledge
    - Ask what topic they want to learn about
    - Probe their current understanding with 2-3 targeted questions
    - Gauge their experience level: complete beginner, some knowledge, or intermediate

    ### Phase 2: Learning Preference Identification
    Ask strategic questions to identify their preferred learning approach:
    - **Examples vs Theory**: "Do you prefer learning through concrete examples
      or understanding the theory first?"
    - **Detail Level**: "Do you like simple, straightforward explanations
      or detailed technical depth?"
    - **Learning Pace**: "Do you prefer step-by-step breakdowns
      or big-picture overviews?"
    - **Interaction Style**: "Do you learn better by practicing with questions
      or by reading explanations?"

    ### Phase 3: Learning Goals & Preferences
    - What's their learning goal? (understand basics, pass test, apply in work, etc.)
    - How much time do they have?
    - Do they prefer structured lessons or flexible exploration?
    ...
    """,
    tools=[transfer_to_agent],
)
```

Key design principles of this prompt:

1. **3-phase assessment structure**: Topic identification -> Learning preference -> Learning goals, in a systematic evaluation sequence
2. **Overload prevention**: "Don't overwhelm - max 2 questions at a time" -- no more than 2 questions at once
3. **Utilizing implicit cues**: If the user correctly uses technical terms, they are assumed to have some foundation

##### Agent Recommendation Logic

```python
    ## Your Recommendations & Transfer:
    After completing your assessment, choose the best learning approach
    and USE the transfer_to_agent tool:

    - **"quiz_agent"**: If they want to test knowledge, prefer active recall,
      or learn through practice
    - **"teacher_agent"**: If they need structured, step-by-step explanations
      or are beginners
    - **"feynman_agent"**: If they claim to understand concepts
      but may need validation
```

Transfer criteria for each agent:
- **quiz_agent**: Learners who prefer active recall
- **teacher_agent**: Beginners or learners needing structured explanations
- **feynman_agent**: Learners who claim to understand but need verification

##### Developer Cheat Code

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to a random agent (quiz_agent, teacher_agent, or feynman_agent)
    for testing purposes using the transfer_to_agent tool.
```

For testing convenience, entering "GODMODE" skips the assessment process and immediately transfers to an agent. This is a practical pattern for quick testing during development.

##### transfer_to_agent Tool (Initial Version)

```python
from langgraph.types import Command
from langchain_core.tools import tool


@tool
def transfer_to_agent(agent_name: str):
    """
    Transfer to the given agent

    Args:
        agent_name: Name of the agent to transfer to, one of:
                    'quiz_agent', 'teacher_agent' or 'feynman_agent'
    """
    return f"Transfer to {agent_name} completed."
    # return Command(
    #     goto=agent_name,
    #     graph=Command.PARENT,
    # )
```

**Important Point:** In this initial version, the actual transfer logic based on `Command` is **commented out**. Since other agent nodes haven't been registered in the graph yet, it is implemented as a stub that simply returns a string. This is a good example of an incremental development strategy.

##### Main Graph Construction

```python
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import START, END, StateGraph, MessagesState
from agents.classification_agent import classification_agent


class TutorState(MessagesState):
    pass


graph_builder = StateGraph(TutorState)

graph_builder.add_node("classification_agent", classification_agent)

graph_builder.add_edge(START, "classification_agent")
graph_builder.add_edge("classification_agent", END)

graph = graph_builder.compile()
```

**Code Analysis:**

1. **`load_dotenv()`**: Loads environment variables such as API keys from the `.env` file. **Must be called before imports** -- other modules may need environment variables at import time.

2. **`TutorState(MessagesState)`**: Inherits from LangGraph's `MessagesState` to automatically manage conversation messages. At this point, it is used with `pass` and no additional state.

3. **Graph structure**: A simple linear structure of `START -> classification_agent -> END`.

##### LangGraph Configuration File

```json
{
    "dependencies": [
        "agents/classification_agent.py",
        "tools/shared_tools.py",
        "main.py"
    ],
    "graphs": {
        "tutor": "./main.py:graph"
    },
    "env": "./env"
}
```

`langgraph.json` is the configuration file referenced by the LangGraph CLI (`langgraph dev`):
- **dependencies**: Dependency file list (for change detection)
- **graphs**: Graphs to expose and their entry points
- **env**: Environment variable file path

#### Practice Points

1. Run the development server with `langgraph dev` and check the graph in LangGraph Studio.
2. Converse with the Classification Agent and test whether the assessment process proceeds naturally.
3. Modify the prompt to add different assessment criteria (e.g., "visual learner vs auditory learner").

---

### 2.3 Section 19.2 -- Feynman Agent & Teacher Agent

**Commit**: `5c2dfa9` "19.2 Feynman Agent"

#### Topic and Objectives

In this section, we implement two core learning agents and complete the agent transfer mechanism:
- **Teacher Agent**: Structured step-by-step education
- **Feynman Agent**: Comprehension verification via the Feynman technique
- **Web search tool**: Real-time information search based on Firecrawl
- **Actual agent transfer**: Routing using `Command` objects

#### Core Concepts

##### Feynman Agent -- The Feynman Learning Technique

Richard Feynman's learning philosophy is implemented as an AI agent. The core principle is **"If you can't explain it simply, you don't understand it well enough."**

```python
from langgraph.prebuilt import create_react_agent
from tools.shared_tools import transfer_to_agent, web_search_tool


feynman_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Feynman Technique Master. Your approach follows the systematic
    Feynman Method: Research → Request Simple Explanation → Evaluate Complexity
    → Ask Clarifying Questions → Complete or Repeat.

    ## The Feynman Philosophy:
    "If you can't explain it simply, you don't understand it well enough."
    Your job is to reveal gaps in understanding through the power of
    simple explanation.
    ...
    """,
    tools=[
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**The 6-Step Feynman Technique Process:**

| Step | Name | Description |
|------|------|-------------|
| Step 1 | Research Phase | Gather accurate information on the concept via web search |
| Step 2 | Request Simple Explanation | Ask to "explain as if to an 8-year-old child" |
| Step 3 | Get User Explanation | Listen to and analyze the user's explanation |
| Step 4 | Evaluate Complexity | Assess jargon usage, logical gaps, vague explanations |
| Step 5 | Ask Clarifying Questions | Ask specific questions about complex parts |
| Step 6 | Complete | If sufficiently simple, acknowledge mastery |

**Key Evaluation Criteria:**

```
    ## Your Evaluation Criteria:
    - No unexplained technical terms
    - Clear cause-and-effect relationships
    - Uses analogies or examples a child would understand
    - Logical flow without gaps
    - Their own words, not memorized definitions
```

These criteria are used to distinguish whether a learner has merely memorized definitions or truly understands the concept. The emphasis on "Their own words" is particularly important.

##### Teacher Agent -- Structured Education Agent

```python
from langgraph.prebuilt import create_react_agent
from tools.shared_tools import transfer_to_agent, web_search_tool


teacher_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Master Teacher who builds understanding through structured,
    step-by-step learning. Your approach follows a proven teaching methodology:
    Research → Break Down → Explain → Confirm → Progress.
    ...
    """,
    tools=[
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**Teacher Agent's Teaching Methodology:**

```
    ### Step 1: Research Phase
    - Use web_search_tool to get current, accurate information

    ### Step 2: Concept Breakdown
    - Divide complex topics into smaller, logical chunks
    - Arrange concepts from foundational to advanced

    ### Step 3: Explain One Concept at a Time
    - Use simple, clear language
    - Provide concrete examples and analogies
    - Present just ONE concept - don't overwhelm

    ### Step 4: Confirmation Check (Critical!)
    - Ask directly: "Does this make sense so far?"
    - Wait for their response and evaluate it carefully

    ### Step 5: Re-explain or Progress
    - If "No" or confused: Re-explain using different approach
    - If "Yes" and demonstrate understanding: Move to Step 6

    ### Step 6: Next Concept or Complete
    - More concepts: Move to next (back to Step 3)
    - Topic complete: Summarize connections
```

**Teacher Agent's Critical Teaching Rules:**

```
    ## Critical Teaching Rules:
    1. Always confirm understanding before moving to the next concept
    2. If they don't understand, explain differently (not just repeat)
    3. Break complex topics into the smallest possible pieces
    4. Use examples from their world and experience
    5. Be patient - true understanding takes time
```

Rule #2 is particularly important -- when understanding is not achieved, the explanation should not simply be repeated but delivered in a **different way**. This is also a core competency of truly excellent teachers.

##### Web Search Tool (web_search_tool)

```python
import re
import os
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain_core.tools import tool


@tool
def web_search_tool(query: str):
    """
    Web Search Tool.
    Args:
        query: str
            The query to search the web for.
    Returns
        A list of search results with the website content in Markdown format.
    """
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    response = app.search(
        query=query,
        limit=5,
        scrape_options=ScrapeOptions(
            formats=["markdown"],
        ),
    )

    if not response.success:
        return "Error using tool."

    cleaned_chunks = []

    for result in response.data:
        title = result["title"]
        url = result["url"]
        markdown = result["markdown"]

        cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
        cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)

        cleaned_result = {
            "title": title,
            "url": url,
            "markdown": cleaned,
        }

        cleaned_chunks.append(cleaned_result)

    return cleaned_chunks
```

**Code Analysis:**

1. **FirecrawlApp**: Performs web searches using the Firecrawl API. A service that combines Google search + page scraping.
2. **`limit=5`**: Limits search results to 5 to save token costs.
3. **`ScrapeOptions(formats=["markdown"])`**: Receives results in markdown format for easy LLM processing.
4. **Text Cleaning**:
   - `re.sub(r"\\+|\n+", "", markdown)`: Removes unnecessary escape characters and excessive line breaks
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: Removes markdown links and URLs

This cleaning process is very important for **token savings** and **noise reduction**. Raw markdown from web pages contains much unnecessary information like navigation links and ad links.

##### transfer_to_agent Tool Completion (Using Command)

```python
@tool
def transfer_to_agent(agent_name: str):
    """
    Transfer to the given agent

    Args:
        agent_name: Name of the agent to transfer to, one of:
                    'teacher_agent' or 'feynman_agent'
    """
    return Command(
        goto=agent_name,
        graph=Command.PARENT,
        update={
            "current_agent": agent_name,
        },
    )
```

The previous section's stub implementation has been changed to **actual `Command`-based transfer logic**.

**Each parameter of the `Command` object:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `goto` | `agent_name` | Target node name to move to |
| `graph` | `Command.PARENT` | Indicates finding the node in the parent graph |
| `update` | `{"current_agent": agent_name}` | Updates graph state |

**Why `Command.PARENT` is needed:**
`transfer_to_agent` is called inside a tool. Tools execute in the subordinate context of an agent, so to move to another node at the top level of the graph, `Command.PARENT` must be specified to indicate it is a parent graph-level transfer.

##### Main Graph Evolution -- Router Pattern

```python
from agents.classification_agent import classification_agent
from agents.teacher_agent import teacher_agent
from agents.feynman_agent import feynman_agent


class TutorState(MessagesState):
    current_agent: str


def router_check(state: TutorState):
    current_agent = state.get("current_agent", "classification_agent")
    return current_agent


graph_builder = StateGraph(TutorState)

graph_builder.add_node(
    "classification_agent",
    classification_agent,
    destinations=(
        "quiz_agent",
        "teacher_agent",
        "feynman_agent",
    ),
)
graph_builder.add_node("teacher_agent", teacher_agent)
graph_builder.add_node("feynman_agent", feynman_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
    ],
)
graph_builder.add_edge("classification_agent", END)

graph = graph_builder.compile()
```

**Code Analysis:**

1. **`current_agent` field added to `TutorState`**: Tracks which agent is currently active.

2. **`router_check` function**: When a conversation resumes (a new message comes in), it checks the `current_agent` value in the current state and routes to the appropriate node. The default value is `"classification_agent"`.

3. **`destinations` parameter**: Specifying `destinations` on the `classification_agent` node tells LangGraph which other nodes are reachable from this node. This is necessary for `Command`-based agent transfers to work properly.

4. **`add_conditional_edges`**: Adds conditional branching from `START`. Depending on the `router_check` function's return value, it enters one of the three agents.

**Routing Flow:**
```
New conversation starts:
  current_agent not set → router_check default → "classification_agent"

classification_agent calls transfer_to_agent("teacher_agent"):
  Command(goto="teacher_agent", update={"current_agent": "teacher_agent"})
  → Switches to teacher_agent

Next message:
  current_agent = "teacher_agent" → router_check → Goes directly to teacher_agent
```

#### Practice Points

1. Explain a concept you know well to the Feynman Agent and see what feedback it gives.
2. Request learning on a new topic from the Teacher Agent and experience the step-by-step education process.
3. Modify the regex in `web_search_tool` to experiment with different cleaning strategies.
4. Add logging to the `router_check` function to trace the routing process.

---

### 2.4 Section 19.3 -- Quiz Agent

**Commit**: `e188909` "19.3 Quiz Agent"

#### Topic and Objectives

Implement a **Quiz Agent** that dynamically generates structured multiple-choice quizzes based on web search results. This utilizes Pydantic's **Structured Output** to ensure the LLM generates quizzes in a defined format.

#### Core Concepts

##### Pydantic-Based Structured Output

The most important technical concept in this section is **Structured Output**. Instead of free-form text, the LLM is asked to generate data conforming to a predefined schema.

```python
from pydantic import BaseModel, Field
from typing import Literal, List


class Question(BaseModel):

    question: str = Field(description="The quiz question text")
    options: List[str] = Field(
        description="Exactly 4 multiple choice options, labeled A, B, C, D."
    )
    correct_answer: str = Field(
        description="The correct answer (MUST MATCH ONE OF 'options')"
    )
    explanation: str = Field(
        description="Exaplanation of why the answer is correct "
                    "and the other ones are wrong."
    )


class Quiz(BaseModel):
    topic: str = Field(description="The main topic being tested")
    questions: List[Question] = Field(
        description="List of the quiz questions"
    )
```

**Key Points in Schema Design:**

1. **`Question` model**: Strictly defines the structure of each question
   - `question`: Question text
   - `options`: Exactly 4 choices (A, B, C, D)
   - `correct_answer`: Correct answer (must match one of the options)
   - `explanation`: Explanation of why the answer is correct and others are wrong

2. **`Quiz` model**: Overall quiz structure
   - `topic`: Quiz topic
   - `questions`: List of Question objects

3. **`Field(description=...)`**: Passes field descriptions to the LLM for more accurate output. It is particularly important to specify constraints like `"MUST MATCH ONE OF 'options'"`.

##### generate_quiz Tool

```python
from langchain.chat_models import init_chat_model


@tool
def generate_quiz(
    research_text: str,
    topic: str,
    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ],
    num_questions: int,
):
    """
    Generate a structured quiz with multiple choice questions
    based on research information.

    Args:
        research_text: str - Research information about the topic.
        topic: str - The main topic/subject for the quiz
        difficulty: Literal["easy", "medium", "hard"] - The difficulty level
        num_questions: int - Number of questions to generate (between 1-30)

    Returns:
        Quiz object with structured questions
    """
    model = init_chat_model("openai:gpt-4o")
    structured_model = model.with_structured_output(Quiz)

    prompt = f"""
    Create a {difficulty} quiz, about {topic} with {num_questions}
    using the following research information.

    <RESEARCH_INFORMATION>
    {research_text}
    </RESEARCH_INFORMATION>

    Make sure to use the RESEARCH_INFORMATION to create
    the most accurate questions.
    """

    quiz = structured_model.invoke(prompt)

    return quiz
```

**Code Analysis:**

1. **`init_chat_model("openai:gpt-4o")`**: LangChain's universal model initialization function. Specifies the provider and model in string format.

2. **`model.with_structured_output(Quiz)`**: This is the key. Converts a regular chat model into a **structured output model**. It internally leverages OpenAI's JSON mode / function calling to guarantee output matching the `Quiz` Pydantic model.

3. **`Literal["easy", "medium", "hard"]`**: Python type hints restrict the difficulty parameter to only 3 values. The LLM recognizes this constraint when calling the tool.

4. **`<RESEARCH_INFORMATION>` XML tags**: XML tags are used to clearly separate research data within the prompt. This prevents the LLM from confusing prompt instructions with data.

**Pattern of calling a separate LLM inside a tool:**
`generate_quiz` is a tool, but internally calls an LLM again. This is a variation of the **"Agent as Tool"** pattern. When the outer agent (Quiz Agent) calls this tool, the LLM inside the tool generates a structured quiz and returns it. This cleanly encapsulates the quiz generation logic.

##### Quiz Agent Prompt -- Strict Workflow

```python
quiz_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Quiz Master and Learning Assessment Specialist.
    Your role is to create engaging, research-based quizzes
    and provide detailed educational feedback.

    ## Your Tools:
    - **web_search_tool**: Research current information on any topic
    - **generate_quiz**: Create structured multiple-choice quizzes
      based on research data
    - **transfer_to_agent**: Switch to other learning agents when appropriate

    ## Your Systematic Quiz Process:

    ### Step 1: Research the Topic
    - Use web_search_tool to gather current, accurate information

    ### Step 2: Ask About Quiz Length
    - **"short"**: 3-5 questions
    - **"medium"**: 6-10 questions
    - **"long"**: 11-15 questions

    ### Step 3: Generate Structured Quiz
    Use the generate_quiz tool with research_text, topic, difficulty,
    num_questions

    ### Step 4: Present Questions One by One
    - Wait for their answer before revealing the correct answer

    ### Step 5: Provide Detailed Feedback
    - If Correct: celebration + explanation
    - If Incorrect: correct answer + detailed explanation

    ### Step 6: Continue Through Quiz
    - Keep track of score, provide final summary
    ...
    """,
    tools=[
        generate_quiz,
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**Enforced Workflow Pattern in Prompt:**

```
    ## CRITICAL WORKFLOW - MUST FOLLOW IN ORDER:
    1. STEP 1: RESEARCH FIRST - You MUST use web_search_tool before anything else
    2. STEP 2: ASK LENGTH - Ask student how many questions they want
    3. STEP 3: CALL generate_quiz - Pass the research_text from step 1
    4. STEP 4: PRESENT ONE BY ONE - Show questions individually
    5. STEP 5: USE EXPLANATIONS - Use the explanations provided by the quiz tool

    NEVER call generate_quiz without research_text from web_search_tool first!
```

This section represents **prompt engineering to strictly control LLM behavior**. Emphasis expressions like "CRITICAL", "MUST", "NEVER" along with numbered steps are used to ensure the LLM follows the prescribed order. Since generating a quiz without web search could include inaccurate information, research is forcefully required first.

##### Integrating Quiz Agent into the Main Graph

```python
from agents.quiz_agent import quiz_agent

# ... added to existing code ...
graph_builder.add_node("quiz_agent", quiz_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
        "quiz_agent",       # Added
    ],
)
```

The quiz_agent node is registered in the graph and added to the conditional edge list of `router_check`.

##### Classification Agent Update

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to quiz_agent for testing purposes using the transfer_to_agent tool.
```

The GODMODE cheat code has been changed from "random agent" to **fixed `quiz_agent`**. This change is for focused testing of the newly added Quiz Agent.

##### shared_tools.py Update

```python
    agent_name: Name of the agent to transfer to, one of:
                'quiz_agent', 'teacher_agent' or 'feynman_agent'
```

`quiz_agent` has been added back to the `transfer_to_agent` docstring. Since the LLM reads the docstring to determine tool parameter values, it is important to accurately list all possible agent names here.

#### Practice Points

1. Request a quiz on a specific topic from the Quiz Agent and evaluate the quality of the generated quiz.
2. Change the `difficulty` parameter of `generate_quiz` to see how questions differ by difficulty level.
3. Add a `hint` field to the `Question` model to implement a hint feature.
4. Design a feature that saves quiz results to state to track learning history.

---

## 3. Chapter Key Summary

### Architecture Patterns

1. **Multi-Agent Swarm Pattern**: Agents autonomously pass tasks to each other using `Command` objects. Without a central manager, each agent decides and transfers to the appropriate agent.

2. **Conditional Routing Pattern**: Combines the `router_check` function with `add_conditional_edges` to automatically route to the appropriate agent based on conversation state.

3. **Incremental Development Strategy**: Develop in the order of stub -> actual implementation (create `transfer_to_agent` as a stub in 19.1 and complete it in 19.2).

### Prompt Engineering

4. **Step-by-Step Process Prompts**: All agents use systematic prompts with clear steps (Step 1, Step 2...). This makes LLM behavior predictable.

5. **Role-Based Personas**: Each agent is given a specific role such as "Educational Assessment Specialist", "Master Teacher", "Feynman Technique Master", "Quiz Master" to constrain their behavioral scope.

6. **Enforced Workflow Pattern**: Prompt design using emphasis expressions like "CRITICAL", "MUST", "NEVER" to ensure the LLM follows a prescribed order.

### Tool Design

7. **Structured Output**: Uses Pydantic models + `with_structured_output()` to transform LLM output into programmatically processable structured data.

8. **LLM Call Inside Tool Pattern**: The `generate_quiz` tool internally calls a separate LLM to generate structured quizzes. This is a good example of logic encapsulation.

9. **Web Search Result Cleaning**: Uses regex to remove unnecessary links, URLs, and escape characters, saving tokens and improving LLM input quality.

### Core LangGraph APIs

| API | Purpose |
|-----|---------|
| `create_react_agent()` | Create ReAct pattern agents |
| `StateGraph` | Define state-based graphs |
| `MessagesState` | Message-based state management |
| `Command(goto, graph, update)` | Agent transfer within graphs |
| `Command.PARENT` | Transfer at parent graph level |
| `add_conditional_edges()` | Add conditional branching edges |
| `add_node(destinations=...)` | Declare reachable targets from a node |

---

## 4. Practice Exercises

### Exercise 1: Add a New Agent (Basic)

Create a **Flashcard Agent** and add it to the system.

- Create `agents/flashcard_agent.py`
- Generate flashcards (front: question, back: answer) after web search on the learning topic
- Define `Flashcard` and `FlashcardDeck` schemas with Pydantic models
- Add the node to `main.py`'s graph and configure routing
- Add flashcard_agent transfer conditions to `classification_agent`'s prompt

### Exercise 2: Learning History Tracking (Intermediate)

Extend `TutorState` to implement learning history tracking.

- Add `quiz_scores: list[dict]`, `topics_learned: list[str]`, `current_topic: str` fields to `TutorState`
- Modify Quiz Agent to save quiz results (score, topic, date) to state
- Modify Teacher Agent to reference previously learned topics and suggest related concepts
- Implement a `progress_report` tool that summarizes learning history

### Exercise 3: Adaptive Difficulty Adjustment (Advanced)

Implement a system that automatically adjusts difficulty based on the learner's quiz performance.

- If consecutive accuracy exceeds 80%, raise the difficulty by one level
- If consecutive accuracy drops below 50%, automatically transfer to Teacher Agent
- Record difficulty change history in state
- Visualize the difficulty change curve as text in the final learning report

### Exercise 4: Inter-Agent Context Sharing (Advanced)

In the current system, the previous agent's assessment results are not explicitly passed to the new agent during transfers.

- Add a `learner_profile: dict` field to `TutorState` to store the Classification Agent's assessment results in a structured format
- Modify each agent's prompt to reference `learner_profile` and generate responses appropriate to the learner's level
- Add a transfer reason (`transfer_reason`) to the `transfer_to_agent` tool so the next agent can understand the context

### Exercise 5: Custom Tool Development (Advanced)

Referencing the `generate_quiz` pattern, implement the following tools.

- **`generate_summary`**: A tool that generates learning summaries based on web search results. Define `Section` and `Summary` schemas with Pydantic models.
- **`evaluate_explanation`**: An automatic evaluation tool for use by the Feynman Agent. Analyzes the learner's explanation to score technical term usage, logical consistency, and conciseness on a 0-10 scale.

---

> **Note**: Running the code in this chapter requires `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` environment variables. Set them in the `.env` file or specify them with `export` in the terminal. To visually inspect graphs in LangGraph Studio, use the `langgraph dev` command.
