# Chapter 6: AutoGen Advanced Multi-Agent Projects

---

## Chapter Overview

In this chapter, we will use Microsoft's **AutoGen** framework to build production-level multi-agent systems. Going beyond the basic agent concepts learned in previous chapters, we will learn advanced patterns where multiple agents collaborate as **Teams**.

This chapter proceeds through two core projects:

1. **Email Optimizer Team**: Using `RoundRobinGroupChat` to build a pipeline where agents take turns sequentially improving an email
2. **Deep Research Clone**: Using `SelectorGroupChat` to build an intelligent research system where AI automatically selects the appropriate agent to perform web research

Through these two projects, you can deeply understand **two core patterns of agent orchestration** -- sequential pipelines and dynamic selection.

### Learning Objectives

- Understand AutoGen framework's Team concept and group chat patterns
- Design sequential multi-agent pipelines using `RoundRobinGroupChat`
- Implement dynamic agent selection systems using `SelectorGroupChat`
- Learn how to connect external tools (Tools) to agents
- Control workflows using termination conditions
- Hands-on practice integrating real web search APIs (Firecrawl) into agents

### Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13+ | Runtime |
| AutoGen (autogen) | >= 0.9.7 | Agent framework core |
| autogen-agentchat | >= 0.7.2 | Team/group chat features |
| autogen-ext[openai] | >= 0.7.2 | OpenAI model integration |
| firecrawl-py | >= 2.16.5 | Web search and scraping |
| python-dotenv | >= 1.1.1 | Environment variable management |
| ipykernel | >= 6.30.1 | Jupyter notebook execution |
| gpt-4o-mini | - | LLM model |

---

## 6.0 Project Introduction and Environment Setup

### Topic and Objectives

In this section, we set up the basic structure of the "Deep Research Clone" project. We define project dependencies through the `pyproject.toml` file and learn how to initialize a Python project using the **uv** package manager.

### Core Concepts

#### What is pyproject.toml?

`pyproject.toml` is the standard configuration file for modern Python projects. It replaces the older `setup.py` and `requirements.txt` approaches, allowing you to manage project metadata and dependencies in a single file.

#### uv Package Manager

This project uses **uv**, a next-generation Python package manager. Written in Rust, uv provides speeds 10-100x faster than pip and handles virtual environment creation and dependency management in an integrated manner.

### Code Analysis

```toml
[project]
name = "deep-research-clone"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "autogen>=0.9.7",
    "autogen-agentchat>=0.7.2",
    "autogen-ext[openai]>=0.7.2",
    "firecrawl-py>=2.16.5",
    "python-dotenv>=1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

**Dependency Analysis:**

| Package | Role |
|---------|------|
| `autogen` | AutoGen framework core. Foundation for agent creation and management |
| `autogen-agentchat` | Provides group chat (team) features. `RoundRobinGroupChat`, `SelectorGroupChat`, etc. |
| `autogen-ext[openai]` | Extension for OpenAI model integration (GPT-4o-mini, etc.) |
| `firecrawl-py` | Web search and webpage content extraction API client |
| `python-dotenv` | Loads environment variables like API keys from `.env` files |
| `ipykernel` | Python kernel for Jupyter notebooks (development dependency) |

**Notable Points:**
- `requires-python = ">=3.13"` requires the latest Python version
- The `dev` group in `[dependency-groups]` separates packages only needed in the development environment
- The brackets `[openai]` in `autogen-ext[openai]` indicate optional dependencies called "extras"

### Practice Points

1. **Project Initialization**: You can start the project with the following terminal commands:
   ```bash
   mkdir deep-research-clone
   cd deep-research-clone
   uv init
   ```

2. **Dependency Installation**: Install dependencies using uv:
   ```bash
   uv add autogen autogen-agentchat "autogen-ext[openai]" firecrawl-py python-dotenv
   uv add --dev ipykernel
   ```

3. **Environment Variable Setup**: Create a `.env` file to manage API keys:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   echo "FIRECRAWL_API_KEY=fc-your-key-here" >> .env
   ```

---

## 6.1 Email Optimizer Team

### Topic and Objectives

In this section, we build a pipeline using **RoundRobinGroupChat** where 5 specialized agents sequentially improve an email. Each agent is responsible for a unique specialty area (clarity, tone, persuasion, synthesis, critique) and works sequentially in a round-robin fashion.

### Core Concepts

#### RoundRobinGroupChat

`RoundRobinGroupChat` is the simplest yet powerful team pattern provided by AutoGen. It operates by having participating agents **take turns in a fixed order**.

```
User Input → ClarityAgent → ToneAgent → PersuasionAgent → SynthesizerAgent → CriticAgent
                  ↑                                                                    |
                  └────────────────── (cycles again if standards not met) ──────────────┘
```

This pattern is suitable for the following situations:
- When each agent's role is clearly defined
- When iterative refinement is needed
- When the workflow order is fixed

#### Termination Conditions

AutoGen provides termination conditions to control team execution. In this project, two termination conditions are combined:

- **TextMentionTermination**: Terminates when specific text (e.g., "TERMINATE") appears in an agent's response
- **MaxMessageTermination**: Terminates when the maximum number of messages is reached

When these two conditions are combined with the `|` (OR) operator, the team stops when **either one is satisfied**.

#### Agent Specialization

The core of a multi-agent system is assigning **one clear role** to each agent. Multiple specialized agents collaborating produces better results than a single "do-everything" agent. This is similar to the **Single Responsibility Principle** in software engineering.

### Code Analysis

#### Step 1: Imports and Model Setup

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
```

Role of each import:
- `RoundRobinGroupChat`: Sequential group chat team class
- `AssistantAgent`: AI model-based agent class
- `OpenAIChatCompletionClient`: Communication client for OpenAI API
- `MaxMessageTermination`, `TextMentionTermination`: Termination condition classes
- `Console`: UI utility for real-time console output of team execution results

#### Step 2: Defining Specialized Agents

The core of this project is designing 5 specialized agents. Pay attention to how each agent's `system_message` restricts and focuses their role.

**ClarityAgent (Clarity Specialist)**

```python
clarity_agent = AssistantAgent(
    "ClarityAgent",
    model_client=model,
    system_message="""You are an expert editor focused on clarity and simplicity.
            Your job is to eliminate ambiguity, redundancy, and make every sentence
            crisp and clear. Don't worry about persuasion or tone — just make the
            message easy to read and understand.""",
)
```

> **Design Point**: Note the explicit scope restriction "Don't worry about persuasion or tone." Clearly limiting the role scope like this prevents role conflicts with other agents.

**ToneAgent (Tone Specialist)**

```python
tone_agent = AssistantAgent(
    "ToneAgent",
    model_client=model,
    system_message="""You are a communication coach focused on emotional tone and
            professionalism. Your job is to make the email sound warm, confident,
            and human — while staying professional and appropriate for the audience.
            Improve the emotional resonance, polish the phrasing, and adjust any
            words that may come off as stiff, cold, or overly casual.""",
)
```

> **Design Point**: Explicitly requires balance between emotional tone and professionalism. By simultaneously instructing potentially conflicting elements -- "warm, confident, and human" vs "professional and appropriate" -- it guides balanced results.

**PersuasionAgent (Persuasion Specialist)**

```python
persuasion_agent = AssistantAgent(
    "PersuasionAgent",
    model_client=model,
    system_message="""You are a persuasion expert trained in marketing, behavioral
            psychology, and copywriting. Your job is to enhance the email's persuasive
            power: improve call to action, structure arguments, and emphasize benefits.
            Remove weak or passive language.""",
)
```

> **Design Point**: By specifying concrete areas of expertise -- marketing, behavioral psychology, and copywriting -- the quality of the agent's responses is elevated. Specific action directives like "Remove weak or passive language" are included.

**SynthesizerAgent (Synthesis Specialist)**

```python
synthesizer_agent = AssistantAgent(
    "SynthesizerAgent",
    model_client=model,
    system_message="""You are an advanced email-writing specialist. Your role is to
            read all prior agent responses and revisions, and then **synthesize the
            best ideas** into a unified, polished draft of the email. Focus on:
            Integrating clarity, tone, and persuasion improvements; Ensuring coherence,
            fluency, and a natural voice; Creating a version that feels professional,
            effective, and readable.""",
)
```

> **Design Point**: This agent performs a meta role of **synthesizing** the results of the three preceding agents. The instruction "read all prior agent responses" encourages active use of previous conversation context. This is the core value of a Synthesizer agent in the RoundRobin pattern.

**CriticAgent (Critique Specialist)**

```python
critic_agent = AssistantAgent(
    "CriticAgent",
    model_client=model,
    system_message="""You are an email quality evaluator. Your job is to perform a
            final review of the synthesized email and determine if it meets professional
            standards. Review the email for: Clarity and flow, appropriate professional
            tone, effective call-to-action, and overall coherence. Be constructive but
            decisive. If the email has major flaws (unclear message, unprofessional tone,
            or missing key elements), provide ONE specific improvement suggestion.
            If the email meets professional standards and communicates effectively,
            respond with 'The email meets professional standards.' followed by
            `TERMINATE` on a new line. You should only approve emails that are perfect
            enough for professional use, dont settle.""",
)
```

> **Design Point**: The CriticAgent serves as a "gatekeeper." If quality standards are met, it outputs "TERMINATE" to end the team; if not, it provides improvement suggestions to trigger the next round. The instruction "dont settle" maintains high quality standards.

#### Step 3: Setting Termination Conditions

```python
text_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=30)

termination_condition = text_termination | max_messages_termination
```

What this configuration means:
- If CriticAgent outputs "TERMINATE" -> **immediate termination** (quality approved)
- If maximum 30 messages are reached -> **forced termination** (infinite loop prevention)
- The `|` operator means an OR condition. Termination occurs when either one is satisfied

> **Design Principle**: Always include `MaxMessageTermination` as a safety net. This prevents situations where agents fail to reach consensus and cycle infinitely.

#### Step 4: Team Creation and Execution

```python
team = RoundRobinGroupChat(
    participants=[
        clarity_agent,
        tone_agent,
        persuasion_agent,
        synthesizer_agent,
        critic_agent,
    ],
    termination_condition=termination_condition,
)

await Console(
    team.run_stream(
        task="Hi! Im hungry, buy me lunch and invest in my business. Thanks."
    )
)
```

**Key Analysis:**

- The **order of the `participants` list is the execution order**. ClarityAgent goes first, CriticAgent goes last.
- `run_stream()` runs the team in an asynchronous streaming manner. Wrapping it with `Console` lets you check each agent's response in real-time.
- The `await` keyword indicates this code runs in an asynchronous (async) environment. In Jupyter notebooks, top-level `await` is automatically supported.

#### Execution Result Analysis

Let's examine the actual execution results step by step:

**1) Input (User)**:
```
Hi! Im hungry, buy me lunch and invest in my business. Thanks.
```
An informal, direct, unprofessional email.

**2) ClarityAgent Output**:
```
Hi! I'm hungry. Please buy me lunch and invest in my business. Thank you.
```
Focused on grammar correction ("Im" -> "I'm"), sentence separation, and polite expression additions.

**3) ToneAgent Output**:
```
Subject: A Quick Favor

Hi there!
I hope you're doing well! I find myself feeling a bit peckish today...
Warm regards,
[Your Name]
```
Completely restructured with a warm and professional tone. Added subject line, greeting, and signature.

**4) PersuasionAgent Output**:
```
Subject: Let's Make Delicious Opportunities Happen!

Hi [Recipient's Name],
...I promise to make it worth your while...
Together, we can turn potential into profit!
```
Added persuasive language ("turn potential into profit"), calls to action ("I promise to make it worth your while"), and benefit emphasis.

**5) SynthesizerAgent Output**:
Wrote the final email integrating the best elements from the three preceding agents.

**6) CriticAgent Output**:
```
The email meets professional standards.
TERMINATE
```
Passed the quality standards and output "TERMINATE," ending the team execution.

### Tool File: tools.py

The Email Optimizer Team doesn't need tools, but the `tools.py` file was created alongside for the next section.

```python
import os, re
from firecrawl import FirecrawlApp, ScrapeOptions


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

This web search tool utilizes the Firecrawl API:

1. **FirecrawlApp Initialization**: Creates a client by loading the API key from environment variables
2. **Search Execution**: Fetches up to 5 search results in markdown format for the query
3. **Result Cleaning**: Two stages of regex processing to remove unnecessary elements
   - `re.sub(r"\\+|\n+", "", markdown)`: Removes excessive line breaks and backslashes
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: Removes markdown links and URLs

> **Why clean the text?** Web scraping results contain many unnecessary elements like navigation links and ad URLs. Removing these reduces the number of tokens sent to the LLM and allows it to focus on essential information.

### Practice Points

1. **Agent Order Experiment**: Change the order of the `participants` list and see how results differ. For example, what happens if PersuasionAgent is placed before ClarityAgent?

2. **Add/Remove Agents**: Add a new specialized agent (e.g., "BrevityAgent" -- brevity specialist) or remove an existing agent and observe the differences in results.

3. **Change Termination Conditions**: Reduce `MaxMessageTermination` to 10 or increase to 50. Make CriticAgent's standards stricter so multiple rounds execute.

4. **Various Input Tests**: Test with various types of input such as formal emails, apology emails, sales emails, etc.

---

## 6.2 Deep Research

### Topic and Objectives

In this section, we build an intelligent research system that clones OpenAI's "Deep Research" feature using **SelectorGroupChat**. Unlike the previous section's fixed-order (RoundRobin) approach, this implements dynamic orchestration where AI analyzes the conversation context and **automatically decides which agent should act next**.

### Core Concepts

#### SelectorGroupChat vs RoundRobinGroupChat

Comparing the key differences between the two team patterns:

| Characteristic | RoundRobinGroupChat | SelectorGroupChat |
|---------------|-------------------|-------------------|
| Agent Selection Method | Fixed order (sequential) | AI dynamically selects |
| Flexibility | Low | High |
| Predictability | High | Relatively low |
| Suitable Situations | Pipelines, review chains | Complex workflows, branching tasks |
| Additional Cost | None | Extra LLM call for selection |

#### How SelectorGroupChat Works

```
User question input
       ↓
  ┌─────────────────┐
  │  Selector LLM   │ ← references selector_prompt + conversation history
  │  (agent selection)│
  └────────┬────────┘
           ↓
  ┌────────┴────────────────────────────────────────────┐
  │                                                      │
  ▼              ▼              ▼            ▼           ▼
research    research     research     research     quality
_planner    _agent       _enhancer    _analyst     _reviewer
  │              │              │            │           │
  └──────────────┴──────────────┴────────────┴───────────┘
                          ↓
                  Selector LLM selects
                  next agent again
                          ↓
                      (repeats...)
```

`SelectorGroupChat` uses a **separate LLM call** each turn to select the next agent. It references the workflow rules defined in the `selector_prompt` and the current conversation history.

#### UserProxyAgent

`UserProxyAgent` is an agent that includes a **human** as a member of the team. It is used when the agent team needs to request human feedback or obtain approval. Human responses are received through standard input via `input_func=input`.

This is an implementation of the **Human-in-the-Loop (HITL)** pattern. As a balance point between full automation and full manual control, AI performs most of the work while key decisions are made by humans.

### Code Analysis

#### Step 1: Imports and Model Setup

```python
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from tools import web_search_tool, save_report_to_md
```

Key changes compared to 6.1:
- Uses `SelectorGroupChat` instead of `RoundRobinGroupChat`
- `UserProxyAgent` added (Human-in-the-Loop)
- Custom tools (`web_search_tool`, `save_report_to_md`) imported

#### Step 2: Defining 6 Specialized Agents

This project defines 6 agents, each responsible for a stage of the research process.

**research_planner (Research Planner)**

```python
research_planner = AssistantAgent(
    "research_planner",
    description="A strategic research coordinator that breaks down complex questions into research subtasks",
    model_client=model_client,
    system_message="""You are a research planning specialist. Your job is to create a focused research plan.

For each research question, create a FOCUSED research plan with:

1. **Core Topics**: 2-3 main areas to investigate
2. **Search Queries**: Create 3-5 specific search queries covering:
   - Latest developments and news
   - Key statistics or data
   - Expert analysis or studies
   - Future outlook

Keep the plan focused and achievable. Quality over quantity.""",
)
```

> **Design Point**: Note the `description` parameter. In `SelectorGroupChat`, this `description` is used to explain the agent's role to the Selector LLM. While unnecessary in the RoundRobin approach, it is **essential** in the dynamic selection approach.

**research_agent (Web Research Executor)**

```python
research_agent = AssistantAgent(
    "research_agent",
    description="A web research specialist that searches and extracts content",
    tools=[web_search_tool],
    model_client=model_client,
    system_message="""You are a web research specialist. Your job is to conduct focused searches based on the research plan.

RESEARCH STRATEGY:
1. **Execute 3-5 searches** from the research plan
2. **Extract key information** from the results:
   - Main facts and statistics
   - Recent developments
   - Expert opinions
   - Important context

3. **Quality focus**:
   - Prioritize authoritative sources
   - Look for recent information (within 2 years)
   - Note diverse perspectives

After completing the searches from the plan, summarize what you found. Your goal is to gather 5-10 quality sources.""",
)
```

> **Key**: `tools=[web_search_tool]` connects the web search tool. Only this agent can actually access the external web. By assigning tools only to specific agents, unnecessary search calls are prevented and roles are clearly defined.

**research_analyst (Research Analyst)**

```python
research_analyst = AssistantAgent(
    "research_analyst",
    description="An expert analyst that creates research reports",
    model_client=model_client,
    system_message="""You are a research analyst. Create a comprehensive report from the gathered research.

CREATE A RESEARCH REPORT with:

## Executive Summary
- Key findings and conclusions
- Main insights

## Background & Current State
- Current landscape
- Recent developments
- Key statistics and data

## Analysis & Insights
- Main trends
- Different perspectives
- Expert opinions

## Future Outlook
- Emerging trends
- Predictions
- Implications

## Sources
- List all sources used

Write a clear, well-structured report based on the research gathered. End with "REPORT_COMPLETE" when finished.""",
)
```

> **Design Point**: The system message provides the **exact structure** (section headers) of the report. Specifying such a structured output format yields consistent results. Using the signal word "REPORT_COMPLETE" to indicate task completion serves as a **protocol** with other agents (quality_reviewer).

**quality_reviewer (Quality Reviewer)**

```python
quality_reviewer = AssistantAgent(
    "quality_reviewer",
    description="A quality assurance specialist that evaluates research completeness and accuracy",
    tools=[save_report_to_md],
    model_client=model_client,
    system_message="""You are a quality reviewer. Your job is to check if the research analyst has produced a complete research report.

Look for:
- A comprehensive research report from the research analyst that ends with "REPORT_COMPLETE"
- The research question is fully answered
- Sources are cited and reliable
- The report includes summary, key information, analysis, and sources

When you see a complete research report that ends with "REPORT_COMPLETE":
1. First, use the save_report_to_md tool to save the report to report.md
2. Then say: "The research is complete. The report has been saved to report.md. Please review the report and let me know if you approve it or need additional research."

If the research analyst has NOT yet created a complete report, tell them to create one now.""",
)
```

> **Key**: This agent has the `save_report_to_md` tool connected. By giving only the quality review agent a tool with a **side effect** (file saving), the workflow is controlled so that reports are only saved after passing quality review.

**research_enhancer (Research Enhancement Specialist)**

```python
research_enhancer = AssistantAgent(
    "research_enhancer",
    description="A specialist that identifies critical gaps only",
    model_client=model_client,
    system_message="""You are a research enhancement specialist. Your job is to identify ONLY CRITICAL gaps.

Review the research and ONLY suggest additional searches if there are MAJOR gaps like:
- Completely missing recent developments (last 6 months)
- No statistics or data at all
- Missing a crucial perspective that was specifically asked for

If the research covers the basics reasonably well, say: "The research is sufficient to proceed with the report."

Only suggest 1-2 additional searches if absolutely necessary. We prioritize getting a good report done rather than perfect coverage.""",
)
```

> **Design Point**: The instructions "ONLY CRITICAL gaps" and "We prioritize getting a good report done rather than perfect coverage" are intended to **prevent excessive research loops**. Perfectionist agents endlessly requesting additional searches is a common problem in multi-agent systems.

**user_proxy (User Proxy)**

```python
user_proxy = UserProxyAgent(
    "user_proxy",
    description="Human reviewer who can request additional research or approve final results",
    input_func=input,
)
```

> This serves the role of obtaining human approval for the final report. When the user enters "APPROVED," the entire workflow terminates.

#### Step 3: Selector Prompt (Agent Selection Prompt)

This is the most important part of this project. The Selector Prompt is the directive used by SelectorGroupChat to determine which agent to select next.

```python
selector_prompt = """
Choose the best agent for the current task based on the conversation history:

{roles}

Current conversation:
{history}

Available agents:
- research_planner: Plan the research approach (ONLY at the start)
- research_agent: Search for and extract content from web sources (after planning)
- research_enhancer: Identify CRITICAL gaps only (use sparingly)
- research_analyst: Write the final research report
- quality_reviewer: Check if a complete report exists
- user_proxy: Ask the human for feedback

WORKFLOW:
1. If no planning done yet → select research_planner
2. If planning done but no research → select research_agent
3. After research_agent completes initial searches → select research_enhancer ONCE
4. If enhancer says "sufficient to proceed" → select research_analyst
5. If enhancer suggests critical searches → select research_agent ONCE more then research_analyst
6. If research_analyst said "REPORT_COMPLETE" → select quality_reviewer
7. If quality_reviewer asked for user feedback → select user_proxy

IMPORTANT: After research_agent has searched 2 times maximum, proceed to research_analyst regardless.

Pick the agent that should work next based on this workflow."""
```

**Detailed Analysis:**

1. **`{roles}` and `{history}` Template Variables**: AutoGen automatically injects agent descriptions and conversation history. This allows the Selector LLM to understand the current state.

2. **Explicit Workflow Rules**: A 7-step workflow is clearly defined in conditional form. This is a pattern similar to a **State Machine**:
   ```
   Start → [Plan] → [Search] → [Enhancement Review] → [Report Writing] → [Quality Review] → [User Approval] → End
                      ↑           |
                      └───────────┘ (if enhancement needed)
   ```

3. **Safety Guard**: "After research_agent has searched 2 times maximum, proceed to research_analyst regardless." -- This rule prevents infinite search loops.

4. **Modifiers like "use sparingly"**: Provides hints to the Selector LLM about the frequency of using specific agents.

#### Step 4: Team Creation and Execution

```python
text_termination = TextMentionTermination("APPROVED")
max_message_termination = MaxMessageTermination(max_messages=50)
termination_condition = text_termination | max_message_termination

team = SelectorGroupChat(
    participants=[
        research_agent,
        research_analyst,
        research_enhancer,
        research_planner,
        quality_reviewer,
        user_proxy,
    ],
    selector_prompt=selector_prompt,
    model_client=model_client,
    termination_condition=termination_condition,
)
```

**Key Comparison (6.1 vs 6.2):**

| Element | 6.1 Email Optimizer | 6.2 Deep Research |
|---------|-------------------|-------------------|
| Team Class | `RoundRobinGroupChat` | `SelectorGroupChat` |
| Termination Keyword | "TERMINATE" | "APPROVED" |
| Max Messages | 30 | 50 |
| `selector_prompt` | None | Detailed workflow definition |
| `model_client` (team level) | None | LLM needed for agent selection |
| `participants` order | Determines execution order | Order irrelevant |

> **Note**: In `SelectorGroupChat`, the order of the `participants` list does not affect execution order. Instead, the rules defined in the `selector_prompt` determine the order.

#### Step 5: Execution

```python
await Console(
    team.run_stream(task="Research about the new development in Nuclear Energy"),
)
```

This single line starts the entire research pipeline. The execution flow is as follows:

1. Selector LLM selects `research_planner`
2. research_planner generates core topics and search queries
3. Selector LLM selects `research_agent`
4. research_agent performs actual web searches using the web search tool
5. Selector LLM selects `research_enhancer`
6. research_enhancer evaluates research sufficiency
7. Selector LLM selects `research_analyst`
8. research_analyst writes a comprehensive report and outputs "REPORT_COMPLETE"
9. Selector LLM selects `quality_reviewer`
10. quality_reviewer saves the report to `report.md`
11. Selector LLM selects `user_proxy`
12. Terminates when user enters "APPROVED"

#### Changes in tools.py (6.1 -> 6.2)

```python
# Change 1: Reduced search results (5 → 2)
response = app.search(
    query=query,
    limit=2,  # Previously: limit=5
    scrape_options=ScrapeOptions(
        formats=["markdown"],
    ),
)

# Change 2: New tool function added
def save_report_to_md(content: str) -> str:
    """Save report content to report.md file."""
    with open("report.md", "w") as f:
        f.write(content)
    return "report.md"
```

**Analysis of Changes:**

1. **`limit=5` -> `limit=2`**: Reducing search results is for **token savings** and **cost optimization**. In deep research, multiple searches are performed, so fetching too many results per search quickly exhausts the context window.

2. **`save_report_to_md` added**: A tool for saving research results to a file. Connected to the `quality_reviewer` agent, ensuring only reports that pass quality review are saved.

#### Generated Report Example (report.md)

Structure of the report generated from execution:

```markdown
# Comprehensive Report on New Developments in Nuclear Energy

## Executive Summary
The nuclear energy sector is witnessing a renaissance as nations prioritize
decarbonization and seek reliable energy sources...

## Background & Current State
In 2023, global electricity production from nuclear energy increased by 2.6%...

## Analysis & Insights
1. **Small Modular Reactors (SMRs)**: These offer cheaper and quicker deployment...
2. **Nuclear Fusion**: Significant progress is being made in fusion research...
3. **Policy Evolution**: Governments are increasingly recognizing nuclear energy's potential...

## Future Outlook
Emerging trends in nuclear energy indicate a potential shift towards more
integrated energy systems...

## Sources
1. International Atomic Energy Agency (IAEA) Report on Nuclear Power for 2023.
2. U.S. Department of Energy Blog: "10 Big Wins for Nuclear Energy in 2023."
...
```

This report was automatically generated based on data actually searched from the web by agents. Thanks to structured system messages, a consistently formatted professional report was produced.

### Practice Points

1. **Test Different Research Topics**: Try running with various topics like "Research about the impact of AI on healthcare." Observe how agents adjust search queries to fit the topic.

2. **Modify Selector Prompt**: Change the workflow rules and see how behavior differs. For example, try creating a rule that skips research_enhancer and goes directly from research_agent to research_analyst.

3. **Add Agents**: Add a "fact_checker" agent to insert a step that verifies the factual accuracy of the report.

4. **Extend Tools**: Create additional tools beyond `web_search_tool` (e.g., academic paper search, news-only search) and provide them to research_agent.

5. **Human-in-the-Loop Variation**: Change the timing of `user_proxy` intervention. For example, modify it so that user approval is also required at the research planning stage.

---

## Chapter Key Summary

### 1. Two Team Orchestration Patterns

| Pattern | Class | Selection Method | When to Use |
|---------|-------|-----------------|-------------|
| **Sequential Pipeline** | `RoundRobinGroupChat` | Fixed order | Tasks with clear roles and fixed order |
| **Dynamic Selection** | `SelectorGroupChat` | AI-based selection | Complex workflows with branching |

### 2. Agent Design Principles

- **Single Responsibility**: Each agent should perform only one clear role
- **Scope Restriction**: Also specify "what not to do" in the system_message (e.g., "Don't worry about persuasion or tone")
- **Use description**: In SelectorGroupChat, the description is directly used for agent selection
- **Signal Word Protocol**: Build inter-agent communication protocols with keywords like "TERMINATE", "REPORT_COMPLETE", "APPROVED"

### 3. Termination Condition Design

- Always **combine** semantic termination (TextMentionTermination) with a **safety net** (MaxMessageTermination)
- Combine with the `|` (OR) operator for flexible termination conditions
- Using only semantic termination without a safety net risks infinite loops

### 4. Tool Assignment Strategy

- Assign tools **selectively only to agents that need them**
- Assign tools with side effects (like file saving) to **verification agents** to implement quality gates
- Appropriately limit the number of results for web search tools considering token costs

### 5. Selector Prompt Design Tips

- Define the workflow from a **state machine** perspective
- Describe which agent to select at each state (condition) with **explicit rules**
- Include **safety guard rules** (e.g., "2 times maximum, proceed regardless")
- Provide hints about agent usage frequency (e.g., "use sparingly", "ONLY at the start")

---

## Practice Assignments

### Assignment 1: Build a Code Review Team (Basic)

**Objective**: Build a code review team using `RoundRobinGroupChat`.

**Requirements**:
- SecurityAgent: Security vulnerability review
- PerformanceAgent: Performance issue analysis
- ReadabilityAgent: Code readability evaluation
- SummaryAgent: Synthesize all reviews into final feedback
- ApprovalAgent: Final approval or revision request (termination control)

**Hint**: Reference the email optimizer team structure, but modify system_messages for code review.

### Assignment 2: Selector Prompt Optimization (Intermediate)

**Objective**: Modify the `selector_prompt` in 6.2's deep research system to add the following features.

**Requirements**:
- Add a `fact_checker` agent and modify the workflow so that fact_checker runs after research_analyst writes the report but before quality_reviewer
- fact_checker cross-verifies the report's key claims through web searches
- Add a new step to the selector_prompt's workflow rules

### Assignment 3: Extend Your Own Deep Research System (Advanced)

**Objective**: Extend 6.2's deep research system to implement the following features.

**Requirements**:
1. Split the search tool into 2: `academic_search_tool` (academic papers only) and `news_search_tool` (news only)
2. Assign each tool to a separate agent
3. Allow the user to select the report format from `user_proxy`'s initial input
4. Save the final report in both markdown and PDF formats

**Hints**:
- You can add domain filters to Firecrawl's `search()` method to separate academic/news results
- Extend `save_report_to_md` to create a `save_report_to_pdf` tool

### Assignment 4: Agent Collaboration Debugging (Analysis)

**Objective**: Analyze the following situation and propose solutions.

**Scenario**: In the deep research system, research_agent repeats searches more than 5 times and does not transition to research_analyst.

**Questions**:
1. Suggest 3 possible causes for this problem
2. How should the selector_prompt be modified to resolve this issue?
3. What safety guards can be added at the code level?

---

## References

- [AutoGen Official Documentation](https://microsoft.github.io/autogen/)
- [Firecrawl API Documentation](https://docs.firecrawl.dev/)
- Project Source Code: `deep-research-clone/` directory
  - `email-optimizer-team.ipynb`: Email optimizer team notebook
  - `deep-research-team.ipynb`: Deep research team notebook
  - `tools.py`: Web search and file saving tools
  - `report.md`: Generated research report example
  - `pyproject.toml`: Project dependency definitions
