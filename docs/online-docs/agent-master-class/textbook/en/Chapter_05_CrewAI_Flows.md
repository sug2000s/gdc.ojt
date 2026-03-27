# Chapter 5: CrewAI Flows - Building a Content Pipeline Agent

---

## 1. Chapter Overview

In this chapter, we will learn how to build an AI-based content generation pipeline from start to finish using **CrewAI Flows**. Flow is a **workflow orchestration system** provided by CrewAI, and it is a core feature that allows you to execute multi-step tasks sequentially or conditionally.

### Learning Objectives

- Understand the basic structure of CrewAI Flow and its decorators (`@start`, `@listen`, `@router`)
- Manage Flow state using Pydantic models
- Implement conditional routing and Refinement Loops
- Call LLMs and Agents directly within a Flow
- Integrate Crews into a Flow to complete complex AI pipelines

### Project Structure

```
content-pipeline-agent/
├── main.py              # Flow main logic
├── tools.py             # Web search tool (Firecrawl)
├── seo_crew.py          # SEO analysis Crew
├── virality_crew.py     # Virality analysis Crew
├── pyproject.toml       # Project dependencies
├── crewai_flow.html     # Flow visualization file
└── .gitignore
```

### Key Dependencies

```toml
[project]
name = "content-pipeline-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "firecrawl-py>=2.16.3",
    "python-dotenv>=1.1.1",
]
```

- **crewai[tools]**: CrewAI framework and tool extension pack
- **firecrawl-py**: Web search and scraping API client
- **python-dotenv**: Environment variable (.env) management

---

## 2. Detailed Section Explanations

---

### 2.1 Your First Flow

**Commit:** `c47fd95`

#### Topic and Objectives

Understand the most basic structure of a CrewAI Flow. Learn what a Flow is, what decorators are available, and how state is managed.

#### Core Concepts

**What is a Flow?**

A Flow is a **workflow engine** provided by CrewAI. You define multiple functions (steps) and declare the execution order of each function using decorators. This allows you to intuitively compose complex AI pipelines.

**Key Decorators:**

| Decorator | Role | Description |
|-----------|------|-------------|
| `@start()` | Entry Point | The first function executed when the Flow starts |
| `@listen(fn)` | Listener | Executed when the specified function completes |
| `@router(fn)` | Router | Branches to different paths based on return value |
| `and_(a, b)` | AND Condition | Executes only when **both** functions are complete |
| `or_(a, b)` | OR Condition | Executes when **any one** of the functions completes |

**Flow State (State Management):**

Flow uses a Pydantic `BaseModel` as its state object. All steps can access and modify the shared state through `self.state`.

#### Code Analysis

```python
from crewai.flow.flow import Flow, listen, start, router, and_, or_
from pydantic import BaseModel


class MyFirstFlowState(BaseModel):
    user_id: int = 1
    is_admin: bool = False


class MyFirstFlow(Flow[MyFirstFlowState]):

    @start()
    def first(self):
        print(self.state.user_id)
        print("Hello")

    @listen(first)
    def second(self):
        self.state.user_id = 2
        print("world")

    @listen(first)
    def third(self):
        print("!")

    @listen(and_(second, third))
    def final(self):
        print(":)")

    @router(final)
    def route(self):
        if self.state.is_admin:
            return "even"
        else:
            return "odd"

    @listen("even")
    def handle_even(self):
        print("even")

    @listen("odd")
    def handle_odd(self):
        print("odd")


flow = MyFirstFlow()

flow.plot()
flow.kickoff()
```

**Detailed Code Flow Analysis:**

1. **`MyFirstFlowState`**: Defines the Flow's state as a Pydantic model. It has two fields: `user_id` and `is_admin`.

2. **`MyFirstFlow(Flow[MyFirstFlowState])`**: Specifies the state class as a generic type. This makes `self.state` of type `MyFirstFlowState`.

3. **`@start()` - `first()`**: The entry point of the Flow. It prints `self.state.user_id` and "Hello".

4. **`@listen(first)` - `second()` and `third()`**: These run **simultaneously (in parallel)** when `first()` completes. `second()` modifies the state (`user_id = 2`), and `third()` prints "!". An important point is that multiple functions listening to the same function execute in parallel.

5. **`@listen(and_(second, third))` - `final()`**: Executes only after **both** `second()` and `third()` complete. The `and_()` condition acts as a synchronization point.

6. **`@router(final)` - `route()`**: Branches based on conditions after `final()`. The function that listens to the returned string (`"even"` or `"odd"`) will execute.

7. **`@listen("even")` / `@listen("odd")`**: Listens for the string value returned by the router. The key point is that you can listen for **strings**, not just function references.

8. **`flow.plot()`**: Visualizes the Flow's execution path as an HTML file (`crewai_flow.html`).

9. **`flow.kickoff()`**: Runs the Flow.

#### Accompanying Tool File: `tools.py`

```python
import os, re
from crewai.tools import tool
from firecrawl import FirecrawlApp, ScrapeOptions


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

**Tool Analysis:**

- Defines a CrewAI tool using the `@tool` decorator.
- Performs web searches using the Firecrawl API.
- Removes unnecessary links and special characters from search results using regular expressions (cleaning).
- Returns clean markdown-formatted results.

#### Practice Points

- Change `is_admin` to `True` and verify that the routing path changes.
- Understand the difference between using a function reference (`first`) and a string (`"even"`) in the `@listen` decorator.
- Open the HTML file generated by `flow.plot()` in a browser to visually inspect the Flow structure.
- Experiment with the behavioral differences between `and_` and `or_`.

---

### 2.2 Content Pipeline Flow

**Commit:** `1e78354`

#### Topic and Objectives

Using the Flow concepts learned in the first example, design the skeleton of a practical **content generation pipeline**. Build a pipeline structure that handles various content types such as tweets, blog posts, and LinkedIn posts.

#### Core Concepts

**Practical Pipeline Design Principles:**

1. **Input Validation**: Block invalid inputs early at Flow start
2. **Conditional Routing**: Select different processing paths based on content type
3. **Quality Check Branching**: SEO check for blogs, virality check for social media
4. **Unified Completion**: All paths ultimately converge to a single completion step

#### Code Analysis

**State Model Design:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""    # One of "tweet", "blog", "linkedin"
    topic: str = ""           # Content topic

    # Internal
    max_length: int = 0       # Maximum length per content type
```

State is organized into **Inputs** (external input) and **Internal** (for internal processing). This is a good practice for clean code structure.

**Pipeline Initialization and Validation:**

```python
class ContentPipelineFlow(Flow[ContentPipelineState]):

    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The content type is wrong.")

        if self.topic == "":
            raise ValueError("The topic can't be blank.")

        if self.state.content_type == "tweet":
            self.state.max_length = 150
        elif self.state.content_type == "blog":
            self.state.max_length = 800
        elif self.state.content_type == "linkedin":
            self.state.max_length = 500
```

- Validates inputs at the start step (Fail Fast pattern).
- Sets `max_length` based on content type for use in subsequent steps.

**Research and Routing:**

```python
    @listen(init_content_pipeline)
    def conduct_research(self):
        print("Researching....")
        return True

    @router(conduct_research)
    def router(self):
        content_type = self.state.content_type
        if content_type == "blog":
            return "make_blog"
        elif content_type == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"
```

The string returned by the `@router` decorator determines the execution path. This is the core pattern for implementing **dynamic branching** in a Flow.

**Content Type-Specific Processing and Quality Checks:**

```python
    @listen("make_blog")
    def handle_make_blog(self):
        print("Making blog post...")

    @listen("make_tweet")
    def handle_make_tweet(self):
        print("Making tweet...")

    @listen("make_linkedin_post")
    def handle_make_linkedin_post(self):
        print("Making linkedin post...")

    @listen(handle_make_blog)
    def check_seo(self):
        print("Checking Blog SEO")

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        print("Checking virality...")

    @listen(or_(check_virality, check_seo))
    def finalize_content(self):
        print("Finalizing content")
```

**Execution Flow Diagram:**

```
init_content_pipeline
        |
  conduct_research
        |
      router -----> "make_blog" -----> handle_make_blog -----> check_seo ------\
        |                                                                        |
        +-------> "make_tweet" -----> handle_make_tweet ------\                  |
        |                                                      +--> check_virality --> finalize_content
        +-------> "make_linkedin_post" -> handle_make_linkedin_post --/
```

**Key Design Points:**
- Blogs branch to the **SEO check** path, while tweets and LinkedIn posts branch to the **virality check** path.
- Using `or_(check_virality, check_seo)` ensures that `finalize_content` executes regardless of which check completes.
- Applying different quality criteria based on content type is a very common pattern in practice.

**Flow Execution (passing inputs):**

```python
flow = ContentPipelineFlow()

flow.kickoff(
    inputs={
        "content_type": "tweet",
        "topic": "AI Dog Training",
    },
)
```

Passing an `inputs` dictionary to `kickoff()` automatically sets the corresponding state fields.

#### Practice Points

- Change `content_type` to `"blog"`, `"tweet"`, and `"linkedin"` respectively and observe how the execution path changes.
- Check the visual structure of the pipeline with `flow.plot()`.
- Think about the difference between `or_` and `and_` in this context: understand why `or_` is used in `finalize_content`.

---

### 2.3 Refinement Loop

**Commit:** `482e52c`

#### Topic and Objectives

Implement a **Refinement Loop** pattern that automatically regenerates content when the quality of AI-generated content falls below a threshold. This is a core pattern for quality assurance in AI agent systems.

#### Core Concepts

**What is a Refinement Loop?**

A Refinement Loop is a cyclical structure of "generate -> evaluate -> regenerate." It iteratively improves content until the evaluation score meets the threshold. This pattern is essential in the following situations:

- When the LLM cannot produce perfect results on the first try
- When quality standards are high and multiple attempts are needed
- When an automated quality assurance (QA) process is required

**Implementing Loops with Routers:**

Flow's `@router` can be used not only for simple branching but also for creating **loops that return to previous steps**. By adding string conditions to `@listen`, the previous step can re-execute based on the router's return value.

#### Code Analysis

**Adding Score and Content Fields to State:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: int = 0                # Quality score added

    # Content
    blog_post: str = ""           # Generated blog post
    tweet: str = ""               # Generated tweet
    linkedin_post: str = ""       # Generated LinkedIn post
```

Score (`score`) and fields to store results for each content type have been added. These state fields play a key role in the loop.

**Re-entry Points Using `or_` with Strings:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        # If blog already exists, show existing one to AI and request improvement,
        # otherwise generate new
        print("Making blog post...")

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        print("Making tweet...")

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        print("Making linkedin post...")
```

**Key Change**: `@listen("make_blog")` has been changed to `@listen(or_("make_blog", "remake_blog"))`. This means:
- On initial creation: The router returns `"make_blog"` to execute
- On regeneration: `score_router` returns `"remake_blog"` to re-execute the same function

This is the **loop re-entry point**.

**Score-Based Router (Loop Control):**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score >= 8:
            return "check_passed"       # Pass -> to finalize_content
        else:
            if content_type == "blog":
                return "remake_blog"     # Loop back -> to handle_make_blog
            elif content_type == "linkedin":
                return "remake_linkedin_post"  # Loop back
            else:
                return "remake_tweet"          # Loop back

    @listen("check_passed")
    def finalize_content(self):
        print("Finalizing content")
```

**Improved Execution Flow:**

```
init_content_pipeline
        |
  conduct_research
        |
  conduct_research_router
        |
   +---------+-----------+
   |         |           |
make_blog  make_tweet  make_linkedin_post
   |         |           |
check_seo  check_virality (or_)
   |         |
   +----+----+
        |
   score_router
    /        \
score >= 8   score < 8
    |            |
"check_passed"  "remake_blog" / "remake_tweet" / "remake_linkedin_post"
    |                    |
finalize_content    (Loop: returns to corresponding content regeneration step)
```

**Router Rename Note:**

```python
    @router(conduct_research)
    def conduct_research_router(self):   # Renamed from "router"
```

The method name was changed from `router` to `conduct_research_router`. Since method names are important for visualization and debugging in Flows, it is good practice to use names that clearly describe the role.

#### Practice Points

- Change the `score >= 8` threshold and observe how many times the loop repeats.
- Add maximum iteration count (max iteration) logic to prevent infinite loops.
- Design logic in the `remake_*` path that references existing content for improvement (implemented in the next section).

---

### 2.4 LLMs and Agents

**Commit:** `c341770`

#### Topic and Objectives

Connect **actual LLM calls** and **Agents** to the placeholders that were previously replaced with `print()` statements. Learn both how to call LLMs directly within a CrewAI Flow and how to use Agents independently.

#### Core Concepts

**Two Ways to Use AI within a Flow:**

1. **`LLM.call()`**: Calls the LLM directly. Fast and simple, useful when structured output is needed.
2. **`Agent.kickoff()`**: Creates and runs an Agent. Used when tool usage or more complex reasoning is needed.

**Structured Output Using Pydantic Models:**

LLM responses can be received in a **predefined structure** rather than free text. This enables stable data handling in subsequent processing steps.

#### Code Analysis

**Defining Pydantic Models for Structured Output:**

```python
from typing import List
from pydantic import BaseModel


class BlogPost(BaseModel):
    title: str
    subtitle: str
    sections: List[str]


class Tweet(BaseModel):
    content: str
    hashtags: str


class LinkedInPost(BaseModel):
    hook: str
    content: str
    call_to_action: str


class Score(BaseModel):
    score: int = 0
    reason: str = ""
```

Defines **output schemas** for each content type:
- **BlogPost**: Composed of title, subtitle, and multiple sections
- **Tweet**: Content and hashtags
- **LinkedInPost**: Hook (attention-grabbing first sentence), body, and CTA (call to action)
- **Score**: Score and its reasoning (for quality evaluation)

**State Model Update:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    research: str = ""              # Stores research results
    score: Score | None = None      # Changed to Score object

    # Content
    blog_post: BlogPost | None = None   # Changed to Pydantic model
    tweet: str = ""
    linkedin_post: str = ""
```

`blog_post`, which was a `str` type, has been changed to `BlogPost | None`. `None` represents a state where it hasn't been generated yet.

**Research Step Using an Agent:**

```python
from crewai.agent import Agent
from tools import web_search_tool

    @listen(init_content_pipeline)
    def conduct_research(self):

        researcher = Agent(
            role="Head Researcher",
            backstory="You're like a digital detective who loves digging up "
                      "fascinating facts and insights. You have a knack for "
                      "finding the good stuff that others miss.",
            goal=f"Find the most interesting and useful info about "
                 f"{self.state.topic}",
            tools=[web_search_tool],
        )

        self.state.research = researcher.kickoff(
            f"Find the most interesting and useful info about "
            f"{self.state.topic}"
        )
```

**Difference Between Agent and Direct LLM Calls:**

| Property | `Agent.kickoff()` | `LLM.call()` |
|----------|-------------------|---------------|
| Tool Usage | Possible (web search, etc.) | Not possible |
| Reasoning Steps | Multi-step reasoning | Single call |
| Speed | Relatively slow | Fast |
| Suitable Use Cases | Research, complex tasks | Content generation, transformation |

The research step uses an Agent because **web search tools** are needed, while content generation directly calls the LLM.

**Blog Generation Using LLM (Structured Output):**

```python
from crewai import LLM

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):

        blog_post = self.state.blog_post

        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            # Initial generation
            self.state.blog_post = llm.call(
                f"""
            Make a blog post on the topic {self.state.topic}
            using the following research:

            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
        else:
            # Regeneration (improvement based on existing content + score feedback)
            self.state.blog_post = llm.call(
                f"""
            You wrote this blog post on {self.state.topic},
            but it does not have a good SEO score because of
            {self.state.score.reason}

            Improve it.

            <blog post>
            {self.state.blog_post.model_dump_json()}
            </blog post>

            Use the following research.

            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
```

**Key Analysis:**

1. **`LLM(model="openai/o4-mini", response_format=BlogPost)`**: Passing a Pydantic model to `response_format` makes the LLM generate a response matching that model's JSON schema.

2. **Initial Generation vs Regeneration Branch**: Determined by `blog_post is None`.
   - Initial: Only research results are passed
   - Regeneration: Existing content + the reason for the low score (`self.state.score.reason`) are passed together to guide improvement

3. **`model_dump_json()`**: Converts the Pydantic model to a JSON string for passing to the LLM.

4. **Using XML Tags in Prompts**: XML tags like `<research>` and `<blog post>` are used to delineate sections of the prompt. This is an effective technique that helps the LLM better understand the prompt structure.

#### Practice Points

- Remove `response_format` and run the code to compare the difference between structured output and free text output.
- Switch to different LLM models (e.g., `"openai/gpt-4o"`) and compare result quality.
- Modify the prompt to change the tone or style of the blog post.
- Experiment with modifying the Agent's `backstory` to see how research results change.

---

### 2.5 Adding Crews To Flows

**Commit:** `8e039ec`

#### Topic and Objectives

Going beyond individual Agent or LLM calls, integrate **Crews** (agent teams) into a Flow. Create an SEO analysis Crew and a virality analysis Crew to perform content quality evaluation. This section is the highlight and most important part of this chapter.

#### Core Concepts

**Value of Flow + Crew Integration:**

Flow **controls the overall workflow**, and Crew **performs complex tasks at specific steps**. Combining these two:

- Flow handles orchestration of the entire pipeline
- Specialized Crews are called when needed at each step
- Crew output results are captured as Flow state for use in subsequent steps

**`@CrewBase` Decorator:**

A decorator used when defining a Crew as a class in CrewAI. Used together with `@agent`, `@task`, and `@crew` decorators to declaratively define the Crew's members and tasks.

#### Code Analysis

**SEO Analysis Crew (`seo_crew.py`):**

```python
from crewai.project import CrewBase, agent, task, crew
from crewai import Agent, Task, Crew
from pydantic import BaseModel


class Score(BaseModel):
    score: int
    reason: str


@CrewBase
class SeoCrew:

    @agent
    def seo_expert(self):
        return Agent(
            role="SEO Specialist",
            goal="Analyze blog posts for SEO optimization and provide a score "
                 "with detailed reasoning. Be very very very demanding, "
                 "don't give underserved good scores.",
            backstory="""You are an experienced SEO specialist with expertise
            in content optimization. You analyze blog posts for keyword usage,
            meta descriptions, content structure, readability, and search
            intent alignment to help content rank better in search engines.""",
            verbose=True,
        )

    @task
    def seo_audit(self):
        return Task(
            description="""Analyze the blog post for SEO effectiveness
            and provide:

            1. An SEO score from 0-10 based on:
               - Keyword optimization
               - Title effectiveness
               - Content structure (headers, paragraphs)
               - Content length and quality
               - Readability
               - Search intent alignment

            2. A clear reason explaining the score, focusing on:
               - Main strengths (if score is high)
               - Critical weaknesses that need improvement (if score is low)
               - The most important factor affecting the score

            Blog post to analyze: {blog_post}
            Target topic: {topic}
            """,
            expected_output="""A Score object with:
            - score: integer from 0-10 rating the SEO quality
            - reason: string explaining the main factors affecting the score""",
            agent=self.seo_expert(),
            output_pydantic=Score,
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
```

**Crew Structure Analysis:**

1. **`@CrewBase`**: Declares the class as a Crew base class. Automatically provides `self.agents` and `self.tasks` properties.

2. **`@agent`**: Defines an Agent. The SEO expert includes the instruction to "score very strictly." This is a design decision to ensure the Refinement Loop works properly (doesn't pass too easily).

3. **`@task`**: Defines a Task. `{blog_post}` and `{topic}` are variables passed from `kickoff(inputs={})`. `output_pydantic=Score` specifies structured output.

4. **`@crew`**: Returns the final Crew object. `self.agents` and `self.tasks` are lists automatically collected by `@CrewBase`.

**Virality Analysis Crew (`virality_crew.py`):**

```python
@CrewBase
class ViralityCrew:

    @agent
    def virality_expert(self):
        return Agent(
            role="Social Media Virality Expert",
            goal="Analyze social media content for viral potential and "
                 "provide a score with actionable feedback",
            backstory="""You are a social media strategist with deep
            expertise in viral content creation. You've analyzed thousands
            of viral posts across Twitter and LinkedIn, understanding the
            psychology of engagement, shareability, and what makes content
            spread. You know the specific mechanics that drive virality on
            each platform - from hook writing to emotional triggers.""",
            verbose=True,
        )

    @task
    def virality_audit(self):
        return Task(
            description="""Analyze the social media content for viral
            potential and provide:

            1. A virality score from 0-10 based on:
               - Hook strength and attention-grabbing potential
               - Emotional resonance and relatability
               - Shareability factor
               - Call-to-action effectiveness
               - Platform-specific best practices
               - Trending topic alignment
               - Content format optimization

            2. A clear reason explaining the score

            Content to analyze: {content}
            Content type: {content_type}
            Target topic: {topic}
            """,
            expected_output="""A Score object with:
            - score: integer from 0-10 rating the viral potential
            - reason: string explaining the main factors affecting virality""",
            agent=self.virality_expert(),
            output_pydantic=Score,
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
```

The Virality Crew has the same structure as the SEO Crew, but with different evaluation criteria:
- SEO Crew: Keyword optimization, title effectiveness, content structure, etc.
- Virality Crew: Hook strength, emotional resonance, shareability, platform-specific best practices, etc.

**Calling Crews from Flow:**

```python
from seo_crew import SeoCrew
from virality_crew import ViralityCrew

    @listen(handle_make_blog)
    def check_seo(self):

        result = (
            SeoCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "blog_post": self.state.blog_post.model_dump_json(),
                }
            )
        )
        self.state.score = result.pydantic

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        result = (
            ViralityCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "content_type": self.state.content_type,
                    "content": (
                        self.state.tweet
                        if self.state.contenty_type == "tweet"
                        else self.state.linkedin_post
                    ),
                }
            )
        )
        self.state.score = result.pydantic
```

**Crew Call Pattern Analysis:**

```python
SeoCrew()           # 1. Create Crew class instance
    .crew()         # 2. Get Crew object
    .kickoff(       # 3. Execute Crew
        inputs={    # 4. Pass input values mapped to Task {variables}
            "topic": self.state.topic,
            "blog_post": self.state.blog_post.model_dump_json(),
        }
    )
```

- `result.pydantic`: Extracts the Pydantic model from the Crew's execution result. Since `output_pydantic=Score` was specified in the Task, `result.pydantic` is a `Score` object.

**Completing Tweet and LinkedIn Content Generation Logic:**

In this commit, not only the blog but also tweet and LinkedIn post generation logic has been fully implemented using the LLM:

```python
    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        tweet = self.state.tweet
        llm = LLM(model="openai/o4-mini", response_format=Tweet)

        if tweet is None:
            result = llm.call(
                f"""
            Make a tweet that can go viral on the topic
            {self.state.topic} using the following research:
            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
        else:
            result = llm.call(
                f"""
            You wrote this tweet on {self.state.topic}, but it does
            not have a good virality score because of
            {self.state.score.reason}

            Improve it.
            <tweet>
            {self.state.tweet.model_dump_json()}
            </tweet>
            Use the following research.
            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )

        self.state.tweet = Tweet.model_validate_json(result)
```

**`model_validate_json(result)`**: Parses the JSON string returned by the LLM into a Pydantic model. Since `response_format=Tweet` was specified, the LLM returns JSON matching the `Tweet` schema, which is then converted back into a Pydantic object.

**Final State Model Update:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    research: str = ""
    score: Score | None = None

    # Content
    blog_post: BlogPost | None = None
    tweet: Tweet | None = None           # Changed from str to Tweet
    linkedin_post: LinkedInPost | None = None  # Changed from str to LinkedInPost
```

#### Practice Points

- Remove "Be very very very demanding" from `SeoCrew`'s `goal` and see how the scores change.
- Create new Crews (e.g., Grammar Check Crew, Fact Check Crew) and add them to the pipeline.
- Extend `virality_crew.py` by adding Agents to make it a multi-agent Crew.
- Observe the execution process through `verbose=True` on the Crew.

---

### 2.6 Conclusions

**Commit:** `be0cf85`

#### Topic and Objectives

Complete the final stage of the pipeline. Adjust quality thresholds, organize final content output, and finalize so the entire pipeline works end-to-end.

#### Core Concepts

**Final Adjustments:**

1. **Relaxed Score Threshold**: Changed from `score >= 8` to `score >= 7` for a practical level
2. **Regeneration Logging**: Added log messages for debugging
3. **Final Output Formatting**: Result output based on content type
4. **Return Value Implementation**: Return the final result of the Flow

#### Code Analysis

**Score Threshold Adjustment:**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score.score >= 7:        # Relaxed from 8 to 7
            return "check_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"
```

Lowering the score threshold to 7 is a practical decision. Too high a threshold can cause near-infinite loops, while too low reduces quality. In a production environment, it is recommended to manage this threshold through configuration files or environment variables.

**Final Content Output:**

```python
    @listen("check_passed")
    def finalize_content(self):
        """Finalize the content"""
        print("Finalizing content...")

        if self.state.content_type == "blog":
            print(f"Blog Post: {self.state.blog_post.title}")
            print(f"SEO Score: {self.state.score.score}/100")
        elif self.state.content_type == "tweet":
            print(f"Tweet: {self.state.tweet}")
            print(f"Virality Score: {self.state.score.score}/100")
        elif self.state.content_type == "linkedin":
            print(f"LinkedIn: {self.state.linkedin_post.title}")
            print(f"Virality Score: {self.state.score.score}/100")

        print("Content ready for publication!")
        return (
            self.state.linkedin_post
            if self.state.content_type == "linkedin"
            else (
                self.state.tweet
                if self.state.content_type == "tweet"
                else self.state.blog_post
            )
        )
```

**Return Value Pattern Analysis:**

`finalize_content()` returns the corresponding Pydantic model based on content type. The value returned by the last step of the Flow becomes the return value of `flow.kickoff()`. This allows external code calling the Flow to receive and utilize the result.

**Regeneration Log Addition:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        blog_post = self.state.blog_post
        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            result = llm.call(...)
        else:
            print("Remaking blog.")   # Debugging log added
            result = llm.call(...)
```

**Virality Check Fix:**

```python
    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        result = (
            ViralityCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "content_type": self.state.content_type,
                    "content": (
                        self.state.tweet.model_dump_json()       # Fixed
                        if self.state.contenty_type == "tweet"
                        else self.state.linkedin_post.model_dump_json()  # Fixed
                    ),
                }
            )
        )
        self.state.score = result.pydantic
```

`.model_dump_json()` has been added to properly serialize Pydantic models as JSON strings.

#### Practice Points

- Run the full pipeline for each `content_type` and compare the results.
- Write code to store the return value of `flow.kickoff()` in a variable and use it.
- Add maximum iteration count limiting logic to `score_router`.

---

## 3. Chapter Key Summary

### Flow Basic Structure

| Concept | Description | Decorator |
|---------|-------------|-----------|
| Entry Point | First function where the Flow starts | `@start()` |
| Listener | Executes after a specific function completes | `@listen(fn)` |
| Router | Branches based on conditions | `@router(fn)` |
| AND Condition | Requires all preceding functions to complete | `and_(a, b)` |
| OR Condition | Executes when any one completes | `or_(a, b)` |
| String Listening | Responds to router return values | `@listen("string")` |

### State Management Pattern

```python
class MyState(BaseModel):
    # Input values
    input_field: str = ""

    # For internal processing
    intermediate_data: str = ""

    # Results (using Pydantic models)
    output: MyOutputModel | None = None
```

### AI Call Method Comparison

| Method | Use Case | Tool Usage | Structured Output |
|--------|----------|------------|-------------------|
| `LLM.call()` | Simple generation/transformation | Not possible | `response_format` |
| `Agent.kickoff()` | Research, complex reasoning | Possible | Limited |
| `Crew.kickoff()` | Team-based complex tasks | Possible | `output_pydantic` |

### Refinement Loop Pattern

```
Generate --> Evaluate --> Check Score --[Pass]--> Complete
                    |
              [Fail] --> Regenerate (loop back)
```

Key: Handle both initial generation and regeneration in the same function with `@listen(or_("make_x", "remake_x"))`

### Pattern for Integrating Crews into Flow

```python
result = MyCrewClass().crew().kickoff(inputs={...})
self.state.score = result.pydantic
```

---

## 4. Practice Assignments

### Assignment 1: Build a Basic Flow (Difficulty: Beginner)

Write a Flow that meets the following requirements:
- Accept a user's name and language (Korean/English) as input
- Generate different greetings based on the language (using a router)
- Output the final greeting

**Hint:** Use `@start()`, `@router()`, and `@listen("string")` decorators.

### Assignment 2: LLM-Integrated Flow (Difficulty: Intermediate)

Create a recipe generation Flow:
- Input: Ingredient list and cooking style (Korean/Western/Chinese)
- Use an Agent in the `conduct_research` step to search for dishes that can be made with the ingredients
- Directly call the LLM in the `generate_recipe` step to generate a structured recipe
- Pydantic model: `Recipe(title: str, ingredients: List[str], steps: List[str], cooking_time: int)`

### Assignment 3: Pipeline with Refinement Loop (Difficulty: Advanced)

Build an email marketing content generation pipeline:
- Input: Product name, target audience, email type (promotion/newsletter/welcome email)
- Research Agent investigates the product and target audience
- LLM generates email content
- Quality evaluation Crew evaluates email effectiveness (subject line appeal, CTA effectiveness, tone appropriateness)
- Implement a Refinement Loop that regenerates if the score is below 7
- Limit to maximum 3 iterations, returning the best result after that

**Hints:**
- Add an `iteration_count: int = 0` field to the state
- In `score_router`, return `"check_passed"` regardless of score when `iteration_count >= 3`

### Assignment 4: Multi-Crew Pipeline (Difficulty: Advanced)

Extend the Content Pipeline from this chapter:
- Add a new Crew: `GrammarCrew` (grammar and readability check)
- For blog posts, run SEO check and grammar check **in parallel** (using `and_`)
- Calculate the final score as a weighted average of both checks
- Only proceed to `finalize_content` when all checks pass

---

## Appendix: Complete Final Code Structure Summary

### main.py Final Structure

```
ContentPipelineState (Pydantic BaseModel)
├── content_type, topic         # Input
├── max_length, research, score # Internal
└── blog_post, tweet, linkedin_post  # Output (Pydantic models)

ContentPipelineFlow (Flow)
├── init_content_pipeline()          @start       - Input validation and initialization
├── conduct_research()               @listen      - Web research with Agent
├── conduct_research_router()        @router      - Branch by content type
├── handle_make_blog()               @listen(or_) - Generate/regenerate blog with LLM
├── handle_make_tweet()              @listen(or_) - Generate/regenerate tweet with LLM
├── handle_make_linkedin_post()      @listen(or_) - Generate/regenerate LinkedIn with LLM
├── check_seo()                      @listen      - SEO evaluation with SeoCrew
├── check_virality()                 @listen(or_) - Virality evaluation with ViralityCrew
├── score_router()                   @router(or_) - Score-based pass/loop back
└── finalize_content()               @listen      - Final result output and return
```

### Key File Relationships

```
main.py ──imports──> tools.py (web_search_tool)
   |
   ├──imports──> seo_crew.py (SeoCrew) ──> Returns Score model
   └──imports──> virality_crew.py (ViralityCrew) ──> Returns Score model
```

Through this chapter, we learned all the core features of CrewAI Flow and completed a content generation pipeline that can be used in production. Remember that Flow is not just a simple workflow tool, but an **orchestration framework** that organically connects LLMs, Agents, and Crews.
