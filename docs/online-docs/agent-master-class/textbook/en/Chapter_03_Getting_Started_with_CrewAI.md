# Chapter 3: Getting Started with CrewAI

---

## Chapter Overview

In this chapter, we learn step by step how to build AI agents using the **CrewAI** framework from the ground up. CrewAI is a Python framework designed so that multiple AI agents can collaborate to perform complex tasks, organized around the concept of a "Crew" (team) with agents and tasks.

This chapter consists of 4 sections, starting from a simple translation agent and progressively developing into a production-level news reader agent system.

| Section | Topic | Key Learning Content |
|---------|-------|----------------|
| 3.1 | Your First CrewAI Agent | Project structure, Agent/Task/Crew basic concepts, YAML configuration |
| 3.2 | Custom Tools | Custom tool creation, `@tool` decorator, connecting tools to agents |
| 3.3 | News Reader Tasks and Agents | Production agent design, detailed prompt writing, multi-agent configuration |
| 3.4 | News Reader Crew | Real tool (search/scraping) integration, LLM model assignment, Crew execution and results |

### Learning Objectives

After completing this chapter, you will be able to:

1. Set up and structure a CrewAI project from scratch
2. Define agents and tasks using YAML configuration files
3. Create custom tools and connect them to agents
4. Configure and execute a Crew where multiple agents collaborate sequentially
5. Build a production-level news collection/summarization/curation pipeline

### Prerequisites

- Python 3.13 or higher
- `uv` package manager (for Python project management)
- OpenAI API key (configured in a `.env` file)
- Basic understanding of Python syntax

---

## 3.1 Your First CrewAI Agent

### Topic and Objective

In this first section, we understand the basic structure of CrewAI and create a simple **Translator Agent**. Through this process, we grasp the relationship between CrewAI's three core elements -- **Agent**, **Task**, and **Crew** -- and learn the role of YAML-based configuration files.

### Core Concepts

#### The Three Core Concepts of CrewAI

CrewAI is built on three core components:

1. **Agent**: An AI worker with a specific role, goal, and backstory. Think of it as a member of a team.
2. **Task**: A specific piece of work that an agent must perform. It includes a description and an expected_output.
3. **Crew**: A team unit that bundles agents and tasks together for execution. Agents process tasks in order.

```
+------------------------------------------+
|                  Crew                    |
|                                          |
|  +----------+    +------------------+    |
|  |  Agent   |--->|     Task 1       |    |
|  |(Translator)|  |(Eng->Italian)    |    |
|  +----------+    +------------------+    |
|       |                   |              |
|       |          (result flows to next)  |
|       |                   v              |
|       |          +------------------+    |
|       +--------->|     Task 2       |    |
|                  |(Italian->Greek)  |    |
|                  +------------------+    |
+------------------------------------------+
```

#### The `@CrewBase` Decorator and Project Structure

CrewAI uses a **decorator-based class pattern**. When the `@CrewBase` decorator is applied to a class, CrewAI automatically reads `config/agents.yaml` and `config/tasks.yaml` files and provides them as `self.agents_config` and `self.tasks_config` dictionaries.

#### Project Directory Structure

```
news-reader-agent/
├── .python-version          # Python version specification (3.13)
├── .gitignore               # Git exclusion file list
├── pyproject.toml           # Project configuration and dependencies
├── config/
│   ├── agents.yaml          # Agent definitions
│   └── tasks.yaml           # Task definitions
├── main.py                  # Main execution file
└── uv.lock                  # Dependency lock file
```

This structure follows CrewAI's convention. When YAML files are placed in the `config/` directory, `@CrewBase` automatically recognizes them.

### Code Analysis

#### Project Dependencies (`pyproject.toml`)

```toml
[project]
name = "news-reader-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "python-dotenv>=1.1.1",
]
```

- `crewai[tools]`: Installs the CrewAI framework along with tool extension packages. `[tools]` is an extras specifier that includes additional tool-related dependencies.
- `python-dotenv`: A library for loading environment variables from `.env` files. Used to securely manage API keys.

#### Agent Configuration (`config/agents.yaml`)

```yaml
translator_agent:
  role: >
    Translator to translate from English to Italian
  goal: >
    To be a good and useful translator to avoid misunderstandings.
  backstory: >
    You grew up between New York and Palermo, you can speak two languages
    fluently, and you can detect the cultural differences.
```

**Role of each field:**

| Field | Description | Role in the Example |
|-------|-------------|-----------------|
| `role` | Defines the agent's job/role | English-Italian translator |
| `goal` | The objective the agent must achieve | Accurate translation without misunderstandings |
| `backstory` | Background setting for the agent (grants personality and expertise) | Bilingual person who grew up in New York and Palermo |

> **Why is `backstory` important?**
> `backstory` is not mere decoration. The LLM uses this background information as context when generating responses, producing more consistent and professional results. For example, the statement "can detect cultural differences" encourages the translation to reflect cultural nuances.

#### Task Configuration (`config/tasks.yaml`)

```yaml
translate_task:
  description: >
    Translate {sentence} from English to Italian without making mistakes.
  expected_output: >
    A well formatted translation from English to Italian using proper
    capitalization of names and places.
  agent: translator_agent

retranslate_task:
  description: >
    Translate {sentence} from Italian to Greek without making mistakes.
  expected_output: >
    A well formatted translation from Italian to Greek using proper
    capitalization of names and places.
  agent: translator_agent
```

**Key Points:**

- `{sentence}`: A **variable placeholder** wrapped in curly braces. It is replaced at runtime with the value passed via `kickoff(inputs={"sentence": "..."})`.
- `expected_output`: Clearly tells the agent what form the result should take. This is necessary for the agent to understand exactly what to return.
- `agent`: The name of the agent that will perform this task. It must match a key defined in `agents.yaml`.
- Both tasks use the same `translator_agent`. A single agent can perform multiple tasks.

#### Main Execution File (`main.py`)

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, agent, task, crew


@CrewBase
class TranslatorCrew:

    @agent
    def translator_agent(self):
        return Agent(
            config=self.agents_config["translator_agent"],
        )

    @task
    def translate_task(self):
        return Task(
            config=self.tasks_config["translate_task"],
        )

    @task
    def retranslate_task(self):
        return Task(
            config=self.tasks_config["retranslate_task"],
        )

    @crew
    def assemble_crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )


TranslatorCrew().assemble_crew().kickoff(
    inputs={
        "sentence": "I'm Nico and I like to ride my bicicle in Napoli",
    }
)
```

**Detailed Code Operation Flow Analysis:**

1. **`dotenv.load_dotenv()`**: Loads environment variables such as `OPENAI_API_KEY` from the `.env` file. CrewAI internally uses this key to call the LLM API.

2. **`@CrewBase` decorator**: When applied to a class, it automatically reads `config/agents.yaml` and `config/tasks.yaml`. This makes `self.agents_config` and `self.tasks_config` accessible.

3. **`@agent` decorator**: Tells CrewAI that the method returns an Agent object. The method name (`translator_agent`) must match the key in YAML.

4. **`@task` decorator**: Tells CrewAI that the method returns a Task object. Tasks are executed **in the order they are defined**. `translate_task` -> `retranslate_task` is the execution order.

5. **`@crew` decorator**: Applied to the method that assembles the Crew object.
   - `self.agents`: A list of all agents marked with `@agent` (automatically collected)
   - `self.tasks`: A list of all tasks marked with `@task` (automatically collected)
   - `verbose=True`: Prints detailed execution progress to the console.

6. **`kickoff(inputs={...})`**: Executes the Crew. The values in the `inputs` dictionary replace the `{sentence}` placeholder in the YAML.

> **Execution Result Flow:**
> 1. `translate_task`: "I'm Nico and I like to ride my bicicle in Napoli" -> Translated to Italian
> 2. `retranslate_task`: Italian translation result -> Translated to Greek
>
> The second task **automatically receives the first task's result as context**. This is how task chaining works in CrewAI.

### Practice Points

- After setting `OPENAI_API_KEY=sk-...` in the `.env` file, try running with `uv run python main.py`.
- Use `verbose=True` to observe the agent's thought process (Chain of Thought).
- Try changing the `backstory` and compare how the quality of results differs.
- Try adding a third task (e.g., Greek -> Korean translation).

---

## 3.2 Custom Tools

### Topic and Objective

In this section, we learn how to connect **Custom Tools** to CrewAI agents. While LLMs can only generate text by default, tools enable them to perform external functions (calculations, API calls, file reading, etc.).

### Core Concepts

#### What Is a Tool?

A **Tool** for an AI agent is a function that the agent can use in addition to the LLM's text generation capabilities. When an agent determines "this task requires a tool" while performing a task, it automatically calls the appropriate tool.

```
+--------------------------------------+
|           Agent                      |
|                                      |
|  "I need to count the letters in    |
|   this sentence...                   |
|   Let me use the count_letters tool!"|
|                                      |
|   +-----------------------------+    |
|   |  Tool: count_letters        |    |
|   |  Input: "Hello World"       |    |
|   |  Output: 11                 |    |
|   +-----------------------------+    |
+--------------------------------------+
```

Tools are important because LLMs can produce **hallucinations** in mathematical calculations or precise data lookups. Even a simple task like counting letters can be wrong if the LLM does it directly, but providing `len()` as a tool guarantees accurate results.

#### The `@tool` Decorator

CrewAI provides the `@tool` decorator to convert a regular Python function into a tool that agents can use. The function's **docstring** serves as the explanation of the tool's purpose to the agent.

### Code Analysis

#### Custom Tool Definition (`tools.py`)

```python
from crewai.tools import tool


@tool
def count_letters(sentence: str):
    """
    This function is to count the amount of letters in a sentence.
    The input is a `sentence` string.
    The output is a number.
    """
    print("tool called with input:", sentence)
    return len(sentence)
```

**Key Analysis:**

- **`@tool` decorator**: This decorator converts the function into a CrewAI tool. Internally, it analyzes the function's signature and docstring to generate a tool schema that the LLM can understand.
- **Type hint `sentence: str`**: This is required. CrewAI uses this type hint to inform the LLM of the input parameter format.
- **docstring**: Used by the agent to determine "when should I use this tool?" It should be written clearly and in detail. It's good practice to describe the format of inputs and outputs.
- **`print()` statement**: For debugging. It lets you confirm whether the tool is actually being called and what input it receives.
- **`return len(sentence)`**: The actual logic. Instead of the LLM counting letters itself, Python's `len()` function returns an accurate result.

#### Adding a New Agent and Task (`config/agents.yaml`)

```yaml
counter_agent:
  role: >
    To count the lenght of things.
  goal: >
    To be a good counter that never lies or makes things up.
  backstory: >
    You are a genius counter.
```

Note the expression "never lies or makes things up" in the `goal`. This is a prompt technique that encourages the agent to always use the tool rather than guessing.

#### Adding a Task (`config/tasks.yaml`)

```yaml
count_task:
  description: >
    Count the amount of letters in a sentence.
  expected_output: >
    The number of letters in a sentence.
  agent: counter_agent
```

#### Connecting Tools in main.py

```python
from tools import count_letters

# ... inside the class ...

@agent
def counter_agent(self):
    return Agent(
        config=self.agents_config["counter_agent"],
        tools=[count_letters],  # Connect tool to agent
    )

@task
def count_task(self):
    return Task(
        config=self.tasks_config["count_task"],
    )
```

**Key Points:**

- `tools=[count_letters]`: Pass a list of tools via the `tools` parameter when creating an Agent. Multiple tools can be connected to a single agent.
- Tools are connected at the **agent level** (not the task level). Whatever task the agent performs, it can use the tools assigned to it.
- No tools are specified directly in `count_task`. This is because the agent responsible for the task (`counter_agent`) already has the tools.

### Practice Points

- Observe the `print()` output to confirm when the tool is actually called.
- Try making the docstring vague and test whether the agent still calls the tool correctly.
- Create new tools (e.g., word counter, uppercase converter) and add them to the agent.
- Connect multiple tools to a single agent and observe whether the agent selects the appropriate tool for each situation.

---

## 3.3 News Reader Tasks and Agents

### Topic and Objective

In this section, we completely restructure the simple translator/counter examples from earlier into a **production news reader system**. We design 3 specialized agents and 3 detailed tasks, learning production-level prompt engineering techniques.

### Core Concepts

#### Multi-Agent Architecture

The news reader system follows a **3-stage pipeline** structure:

```
+--------------+     +--------------+     +--------------+
| News Hunter  |---->|  Summarizer  |---->|   Curator    |
|   Agent      |     |    Agent     |     |    Agent     |
|              |     |              |     |              |
| News         |     | Article      |     | Final Report |
| Collection   |     | Summary      |     | Editing &    |
| & Filtering  |     | (3 levels)   |     | Curation     |
+--------------+     +--------------+     +--------------+
  Task 1:              Task 2:              Task 3:
  content_harvesting   summarization        final_report_assembly

  output:              output:              output:
  content_harvest.md   summary.md           final_report.md
```

Each agent has a **different specialty**. This is the key advantage of multi-agent systems. Having each agent focus on its own expertise produces better results than a single agent doing everything.

#### Prompt Engineering: Writing Detailed `backstory`

The most important change in this section is the **depth and detail** of agent configurations. The simple 2-line backstories from 3.1 evolve into detailed profiles of 10+ lines.

#### Task `output_file` Configuration

Each task is configured to **automatically save its results as a markdown file**. This allows you to inspect and debug intermediate results at each stage.

### Code Analysis

#### Agent Configuration (`config/agents.yaml`)

**1. News Hunter Agent - News Collection Specialist**

```yaml
news_hunter_agent:
  role: >
    Senior News Intelligence Specialist
  goal: >
    Discover and collect the most relevant, credible, and up-to-date news
    articles from diverse sources across specified topics, ensuring
    comprehensive coverage while filtering out misinformation and
    low-quality content
  backstory: >
    You are a seasoned digital journalist with 15 years of experience in
    news aggregation and fact-checking. You have an exceptional ability to
    identify credible sources, spot trending stories before they break
    mainstream, and navigate the complex landscape of digital media. Your
    network spans traditional media outlets, independent journalists, and
    expert sources across multiple industries. You pride yourself on your
    ability to separate signal from noise in the overwhelming flow of daily
    news, and you have a keen sense for detecting bias and misinformation.
    You understand the importance of source diversity and always
    cross-reference information from multiple outlets before considering
    it reliable.
  verbose: true
  inject_date: true
```

**New Configuration Options:**

| Option | Description |
|--------|-------------|
| `verbose: true` | Prints the agent's thought process in detail |
| `inject_date: true` | Automatically injects the current date into the agent's context. Essential for assessing news timeliness |

**`backstory` Analysis - Why Write It in Such Detail:**

- "15 years of experience": Sets the level of expertise to encourage the LLM to make high-quality judgments
- "separate signal from noise": Emphasizes filtering ability to encourage filtering out irrelevant articles
- "detecting bias and misinformation": Activates credibility assessment capability
- "source diversity": Encourages gathering information from diverse sources

**2. Summarizer Agent - Summarization Specialist**

```yaml
summarizer_agent:
  role: >
    Expert News Analyst and Content Synthesizer
  goal: >
    Transform raw news articles into clear, concise, and comprehensive
    summaries that capture essential information, context, and implications
    while maintaining objectivity and highlighting key insights for busy
    readers
  backstory: >
    You are a skilled news analyst with a background in journalism and
    information science. You've worked as an editor for major news
    publications and have a talent for distilling complex stories into
    digestible summaries without losing critical nuance. Your expertise
    spans multiple domains including politics, technology, economics, and
    international affairs. ...
  verbose: true
  inject_date: true
  llm: openai/o3
```

**New Setting: `llm: openai/o3`**

A **different LLM model can be specified** for specific agents. Since summarization requires a high level of comprehension and expression, a more powerful model (o3) is used. By configuring different models per agent, you can optimize cost and performance.

**3. Curator Agent - Editorial Specialist**

```yaml
curator_agent:
  role: >
    Senior News Editor and Editorial Curator
  goal: >
    Curate and editorialize summarized news content into a cohesive,
    engaging narrative that provides context, identifies the most important
    stories, and creates a meaningful reading experience that helps users
    understand not just what happened, but why it matters
  backstory: >
    You are a veteran news editor with 20+ years of experience at top-tier
    publications like The New York Times, The Economist, and Reuters. ...
  verbose: true
  inject_date: true
```

#### Task Configuration (`config/tasks.yaml`)

**1. Content Harvesting Task - News Collection Task**

This task includes the most detailed instructions. Let's examine the key parts:

```yaml
content_harvesting_task:
  description: >
    Collect recent news articles based on {topic}.

    Steps include:
    1. Use the search tool to search for recent news articles about {topic}
    2. From the search results, identify URLs from credible sources.

    3. **IMPORTANT: Only select actual article pages, not topic hubs or
       tag listings**
      You must filter out any URLs that are likely to be:
      - Topic/tag/section index pages (e.g., URLs containing "/tag/",
        "/topic/", "/hub/", "/section/", "/category/")
      - Pages with no unique headline or timestamp
      - Pages that only contain a list of other stories or links
```

**Prompt Engineering Technique Analysis:**

1. **Step-by-step instructions**: Tasks to perform are specified in numbered order.
2. **Explicit filtering rules**: "IMPORTANT" is emphasized in uppercase, with allow/deny URL patterns provided alongside concrete examples.
3. **Checklist format**: Visual distinction using allow and deny symbols.
4. **Numeric criteria**: Concrete numbers are provided such as "remove articles under 200 words," "remove articles older than 48 hours."
5. **Scoring system**: Credibility (1-10) and relevance (1-10) scores are required.

```yaml
  expected_output: >
    A well-structured markdown document containing the collected news
    articles with this exact format:

    # News Articles Collection: {topic}

    **Collection Summary**
    - Total articles found:
    - Articles after filtering:
    - Duplicates removed:
    ...
  agent: news_hunter_agent
  markdown: true
  output_file: output/content_harvest.md
  create_directory: true
```

**Task Output Settings:**

| Option | Description |
|--------|-------------|
| `markdown: true` | Process output in markdown format |
| `output_file: output/content_harvest.md` | Automatically save results to the specified file |
| `create_directory: true` | Automatically create the `output/` directory if it doesn't exist |

**2. Summarization Task**

```yaml
summarization_task:
  description: >
    Take each of the URLs from the previous task and generate a summary
    for each article.

    Use the scrape tool to extract the full article content from the URL.

    For each article found in the file, create:
    1. **Headline Summary** (≤280 characters, tweet-style)
    2. **Executive Summary** (150-200 words, concise briefing)
    3. **Comprehensive Summary** (500-700 words with full context)
```

This task requires a **3-level summary system**. This is a practical design pattern that satisfies the needs of different reader audiences:
- Tweet-level breaking news -> For social media sharing
- Executive summary -> For busy professionals
- Detailed summary -> For readers needing in-depth understanding

**3. Final Report Assembly Task**

```yaml
final_report_assembly_task:
  description: >
    Create the final, publication-ready markdown news briefing by combining
    all previous work into a professional, cohesive report suitable for
    daily publication.

    Assembly process:
    1. **Follow the editorial plan** from the curation task for structure
    2. **Apply appropriate summary levels** for each story
    3. **Include editorial transitions** and section introductions
    4. **Add professional opening** that summarizes the day's key
       developments
    5. **Create closing section** that ties together themes
    6. **Ensure consistent formatting** and professional presentation
    7. **Include proper attribution** and source references
```

This task synthesizes the results of the two preceding tasks to generate a **publication-ready** news briefing.

#### Changes in the Main File (`main.py`)

```python
@CrewBase
class NewsReaderAgent:

    @agent
    def news_hunter_agent(self):
        return Agent(
            config=self.agents_config["news_hunter_agent"],
        )

    @agent
    def summarizer_agent(self):
        return Agent(
            config=self.agents_config["summarizer_agent"],
        )

    @agent
    def curator_agent(self):
        return Agent(
            config=self.agents_config["curator_agent"],
        )

    @task
    def content_harvesting_task(self):
        return Task(
            config=self.tasks_config["content_harvesting_task"],
        )

    @task
    def summarization_task(self):
        return Task(
            config=self.tasks_config["summarization_task"],
        )

    @task
    def final_report_assembly_task(self):
        return Task(
            config=self.tasks_config["final_report_assembly_task"],
        )

    @crew
    def crew(self):
        return Crew(
            tasks=self.tasks,
            agents=self.agents,
            verbose=True,
        )


NewsReaderAgent().crew().kickoff()
```

**Key Changes:**

1. The class name changed from `TranslatorCrew` to `NewsReaderAgent`
2. The Crew method name was simplified from `assemble_crew` to `crew`
3. `kickoff()` does not have `inputs` yet (added in the next section)
4. At this point, agents do not yet have tools connected (design phase)

### Practice Points

- Change the agent's `backstory` to be more detailed or more brief and compare the quality differences in the results.
- Change the `expected_output` format to experiment with different output structures.
- Change the `output_file` path in `tasks.yaml` and verify that the file is created correctly.
- Add a fourth agent (e.g., translator) and design a pipeline that translates the final report into Korean.

---

## 3.4 News Reader Crew

### Topic and Objective

In this final section, we connect **real tools** to the designed news reader system to complete a fully operational Crew. We implement web search and web scraping tools, assign appropriate LLM models to each agent, and run the system with a real topic ("Cambodia Thailand War") to verify the results.

### Core Concepts

#### Production Tools

In 3.2, we created a simple tool wrapping `len()`, but in this section, we implement tools that interact with real web services:

1. **Search Tool**: Google search using the Serper API
2. **Scrape Tool**: Web page content extraction using Playwright + BeautifulSoup

#### Built-in Tools vs Custom Tools

| Category | Built-in Tools | Custom Tools |
|----------|---------------|-------------|
| Example | `SerperDevTool` | `scrape_tool` |
| Advantage | Simple setup, ready to use immediately | Full control, handles special requirements |
| Disadvantage | Limited customization | Must implement yourself |

#### Assigning LLM Models Per Agent

Using the same model for all agents is inefficient. By assigning different models based on task complexity, you can save costs while ensuring high quality where needed.

### Code Analysis

#### Tool Implementation (`tools.py`)

**1. Search Tool - SerperDevTool**

```python
import time
from crewai.tools import tool
from crewai_tools import SerperDevTool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

search_tool = SerperDevTool(
    n_results=30,
)
```

- `SerperDevTool`: A built-in search tool provided by CrewAI. It fetches Google search results via the Serper API.
- `n_results=30`: Fetches up to 30 search results. Set generously for comprehensive news collection.
- Using it requires setting `SERPER_API_KEY` in the `.env` file.

**2. Scrape Tool - Custom Implementation**

```python
@tool
def scrape_tool(url: str):
    """
    Use this when you need to read the content of a website.
    Returns the content of a website, in case the website is not
    available, it returns 'No content'.
    Input should be a `url` string. for example
    (https://www.reuters.com/world/asia-pacific/...)
    """

    print(f"Scrapping URL: {url}")

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(url)

        time.sleep(5)

        html = page.content()

        browser.close()

        soup = BeautifulSoup(html, "html.parser")

        unwanted_tags = [
            "header",
            "footer",
            "nav",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "button",
            "input",
            "select",
            "textarea",
            "img",
            "svg",
            "canvas",
            "audio",
            "video",
            "embed",
            "object",
        ]

        for tag in soup.find_all(unwanted_tags):
            tag.decompose()

        content = soup.get_text(separator=" ")

        return content if content != "" else "No content"
```

**Code Operation Flow Analysis:**

1. **Playwright browser launch**: Launches a Chromium browser in headless (no display) mode using `sync_playwright()`. This allows handling dynamic web pages that render with JavaScript.

2. **Page load and wait**: After `page.goto(url)`, waits 5 seconds with `time.sleep(5)`. This provides time for JavaScript rendering and dynamic content loading to complete.

3. **HTML parsing**: Parses the HTML with `BeautifulSoup`.

4. **Removing unwanted tags**: Removes all tags defined in the `unwanted_tags` list. This filters out noise like navigation, ads, and scripts, extracting only the pure article text.

5. **Text extraction**: Extracts all text connected by spaces using `soup.get_text(separator=" ")`.

6. **Safe return**: Returns "No content" if the content is empty, so the agent can recognize the failure.

> **Why use Playwright instead of `requests`?**
> Most modern news sites dynamically render content with JavaScript. The `requests` library only fetches static HTML, so article body text may be missing. Playwright runs a real browser, so it can obtain the complete DOM after JavaScript execution.

#### Assigning LLM Models to Agents (`config/agents.yaml`)

```yaml
news_hunter_agent:
  # ... (existing settings)
  llm: openai/o4-mini-2025-04-16

summarizer_agent:
  # ... (existing settings)
  llm: openai/o4-mini-2025-04-16   # Changed from o3 to o4-mini

curator_agent:
  # ... (existing settings)
  llm: openai/o4-mini-2025-04-16
```

The `openai/o3` specified for summarizer_agent in 3.3 has been changed to `openai/o4-mini-2025-04-16`. All agents now use the same model, reflecting the balance between cost efficiency and sufficient performance. The `llm` field format is `provider/model-name`.

#### Completing the Main File (`main.py`)

```python
from tools import search_tool, scrape_tool


@CrewBase
class NewsReaderAgent:

    @agent
    def news_hunter_agent(self):
        return Agent(
            config=self.agents_config["news_hunter_agent"],
            tools=[search_tool, scrape_tool],
        )

    @agent
    def summarizer_agent(self):
        return Agent(
            config=self.agents_config["summarizer_agent"],
            tools=[
                scrape_tool,
            ],
        )

    @agent
    def curator_agent(self):
        return Agent(
            config=self.agents_config["curator_agent"],
        )

    # ... task definitions are the same ...

    @crew
    def crew(self):
        return Crew(
            tasks=self.tasks,
            agents=self.agents,
            verbose=True,
        )


result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

**Tool Assignment Strategy Per Agent:**

| Agent | Tools | Reason |
|-------|-------|--------|
| `news_hunter_agent` | `search_tool`, `scrape_tool` | Needs to find articles via search and read content via scraping |
| `summarizer_agent` | `scrape_tool` | Re-reads articles from previous task URLs for detailed summarization |
| `curator_agent` | (none) | Only edits summary results from previous tasks, no external tools needed |

**`kickoff()` Execution and Result Processing:**

```python
result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

- `inputs={"topic": "Cambodia Thailand War."}`: Replaces the `{topic}` placeholder in the YAML.
- `result.tasks_output`: Returns the execution results of each task as a list. Since there are 3 tasks, 3 results are included.

#### Execution Output

After Crew execution, 3 markdown files are generated in the `output/` directory:

**1. `output/content_harvest.md` - Collected Article List**

```markdown
# News Articles Collection: Cambodia Thailand War.
**Collection Summary**
- Total articles found: 4
- Articles after filtering: 3
- Duplicates removed: 0
- Sources accessed: Reuters, AP News, BBC
- Search queries used: "Cambodia Thailand War recent news August 2025"...
- Search timestamp: 2025-08-05

---
## Article 1: Cambodia and Thailand begin talks in Malaysia...
**Source:** Reuters
**Date:** 2025-08-04 06:19 UTC
**URL:** https://www.reuters.com/world/asia-pacific/...
**Category:** International
**Credibility Score:** 9
**Relevance Score:** 10
```

You can see that news_hunter_agent collected 3 trustworthy articles (Reuters, AP News, BBC) and assigned credibility and relevance scores to each.

**2. `output/summary.md` - 3-Level Summary**

For each article, a tweet-level summary (under 280 characters), executive summary (150-200 words), and detailed summary (500-700 words) are generated. The format faithfully follows what was specified in `expected_output`.

**3. `output/final_report.md` - Final News Briefing**

The final report synthesizing all information into a publication-quality news briefing. A professional news briefing structured with sections such as Executive Summary, Lead Story, Breaking News, and Editor's Analysis.

### Practice Points

- Try changing the `topic` in `inputs` to a different subject (e.g., "AI regulation 2025", "climate change policy").
- Experiment with the speed-stability tradeoff by adjusting the `time.sleep(5)` value in `scrape_tool`.
- Try modifying the `unwanted_tags` list to improve extraction quality.
- Change `n_results=30` to a smaller or larger value to observe the impact of search scope.
- Try adding a new agent (e.g., fact checker) to extend the pipeline.

---

## Chapter Key Takeaways

### 1. CrewAI's Core Architecture

- **Agent**: An AI worker with a role, goal, and backstory. Configured in YAML and instantiated in Python.
- **Task**: A specific work instruction. The key elements are `description`, `expected_output`, and `agent` assignment.
- **Crew**: A team unit that bundles Agents and Tasks for execution. Launched with `kickoff()`.
- **Tool**: A function that provides external capabilities to an agent. Created with the `@tool` decorator.

### 2. Project Structure Conventions

```
project/
├── config/
│   ├── agents.yaml    # Agent definitions (role, goal, backstory)
│   └── tasks.yaml     # Task definitions (description, expected_output)
├── main.py            # @CrewBase class and execution code
├── tools.py           # Custom tool definitions
├── output/            # Task result file storage
└── pyproject.toml     # Dependency management
```

### 3. Prompt Engineering Principles

- **Detailed backstory**: Setting concrete expertise and personality for the agent improves output quality.
- **Step-by-step instructions**: Number tasks to perform in order within the `description`.
- **Concrete criteria**: Remove ambiguity with numbers, examples, and allow/deny patterns.
- **Output format templates**: Providing a markdown format template in `expected_output` yields consistent results.

### 4. Tool Design Principles

- The docstring determines when the tool is used. It must be written clearly and in detail.
- Type hints are required. They are used by the LLM to pass correct arguments.
- Assign only the tools an agent needs. Unnecessary tools cause confusion.

### 5. Multi-Agent Design Patterns

- **Specialization principle**: Each agent focuses on one area of expertise.
- **Pipeline pattern**: Tasks execute sequentially, with each task's result becoming the next task's input.
- **Model optimization**: Different LLM models can be assigned per agent based on task complexity.

---

## Practice Assignments

### Assignment 1: Basic - Create Your Own First Crew

**Objective**: Implement the basic structure of CrewAI yourself.

**Requirements**:
1. Create a Crew with 2 agents:
   - `writer_agent`: An agent that writes a short piece on a given topic
   - `reviewer_agent`: An agent that reviews the written piece and provides feedback
2. Write appropriate `role`, `goal`, and `backstory` for each agent
3. Define 2 tasks:
   - `writing_task`: Write a 300-word piece about `{topic}`
   - `review_task`: Review the written piece for grammar, logic, and readability
4. Run with `verbose=True` and observe the agents' thought processes

### Assignment 2: Intermediate - Custom Tool Usage

**Objective**: Create practical custom tools and connect them to agents.

**Requirements**:
1. Implement the following custom tools:
   - `get_weather(city: str)`: Calls a weather API to return current weather (use a free API)
   - `calculate(expression: str)`: Evaluates a mathematical expression and returns the result
2. Create a `travel_planner_agent` and connect both tools to it
3. Define a `plan_trip_task` that checks a specific city's weather and creates a travel plan
4. Change the docstrings and observe changes in tool calling patterns

### Assignment 3: Advanced - News Reader Extension

**Objective**: Extend the news reader system built in this chapter.

**Requirements**:
1. Add the following agents to the existing news reader:
   - `translator_agent`: Translates the final report into Korean
   - `fact_checker_agent`: Cross-verifies facts between articles
2. Assign a different LLM model to `translator_agent` (e.g., `openai/gpt-4o`)
3. Design and connect appropriate tools for `fact_checker_agent`
4. Configure tasks so the 5-stage pipeline operates sequentially
5. Run with various topics and compare/analyze the results in each `output_file`

### Assignment 4: Challenge - Design an Autonomous Agent Team

**Objective**: Design a multi-agent system applicable to real-world work from scratch.

**Requirements**:
1. Choose a domain you're interested in (finance, education, health, etc.)
2. Design at least 3 specialized agents
3. Create and connect at least 1 custom tool to each agent
4. Write detailed `backstory` and concrete `expected_output` formats
5. Run the entire system and establish criteria for evaluating the quality of results
6. Save results as markdown files and prepare presentation materials

---

> **Next Chapter Preview**: In Chapter 4, we learn CrewAI's advanced features such as inter-agent communication, conditional task execution, and memory systems to build more sophisticated agent systems.
