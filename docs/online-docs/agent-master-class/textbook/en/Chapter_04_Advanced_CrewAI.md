# Chapter 4: Advanced CrewAI - Building a Job Hunter Agent

---

## Chapter Overview

In this chapter, we use the CrewAI framework to build a **complex AI agent system that can be used in real-world practice**. Rather than a single agent, we design and implement a structure where multiple agents **divide roles** and **pass work results to each other** to collaborate, learning the process step by step.

The project topic is **"Job Hunter Agent"**, which automates the following workflow:

1. Search and extract job postings from the web
2. Match the user's resume against job postings and assign scores
3. Select the optimal job posting
4. Rewrite the resume tailored to the selected posting
5. Research the company
6. Generate interview preparation materials

Key skills learned in this chapter:

| Skill | Description |
|-------|-------------|
| **Agent Definition (YAML)** | Declarative management of role, goal, and backstory in YAML |
| **Task Definition (YAML)** | Configuring task description, expected output, and assigned agent in YAML |
| **Structured Output (Pydantic)** | Forcing agent output to conform to Pydantic models for structuring |
| **Context Passing** | Automatically passing preceding Task results to subsequent Tasks |
| **Custom Tool (Firecrawl)** | Creating web search/scraping tools using external APIs |
| **Knowledge Source** | Injecting text files (resume) as agent knowledge |

---

## 4.1 Defining Agents and Tasks

### Topic and Objective

Learn how to define CrewAI's core components -- **Agent** and **Task** -- using YAML configuration files. In this stage, before writing any code, we first design **"who (Agent) will do what (Task)."**

### Core Concepts

#### Project Structure

```
job-hunter-agent/
├── config/
│   ├── agents.yaml      # Agent definitions
│   └── tasks.yaml       # Task definitions
├── knowledge/
│   └── resume.txt       # User's resume (knowledge source)
├── main.py              # Main execution file (empty at this stage)
├── pyproject.toml       # Project dependencies
└── output/              # Result output directory
```

CrewAI adopts a **declarative approach**. By defining agents and tasks in YAML files rather than code:
- Non-developers can understand and modify agent behavior
- Separating code from configuration improves maintainability
- Agent roles and goals are clearly documented

#### Core Elements of Agent Definition

In CrewAI, an Agent is defined by three core properties:

| Property | Role | Example |
|----------|------|---------|
| `role` | The agent's job title/role | "Senior Job Market Research Specialist" |
| `goal` | The objective the agent must achieve | "Discover and analyze relevant job opportunities..." |
| `backstory` | The agent's expertise and background | "You are an experienced talent acquisition specialist with 12+ years..." |

`backstory` is not mere description -- it is the core of **prompt engineering** that guides the LLM to make **expert-level judgments** appropriate to the role.

### Code Analysis

#### agents.yaml - Defining 5 Agents

This project defines 5 specialized agents:

**1) Job Search Agent (job_search_agent)**

```yaml
job_search_agent:
  role: >
    Senior Job Market Research Specialist
  goal: >
    Discover and analyze relevant job opportunities from major job platforms that
    match the user's skills, experience level, and career preferences, providing
    detailed job information and market insights for optimal application strategy
  backstory: >
    You are an experienced talent acquisition specialist with 12+ years in recruitment
    and job market analysis. You have deep expertise in navigating job boards,
    understanding hiring trends, and identifying the best opportunities for candidates
    across various industries. You excel at reading between the lines of job descriptions
    to understand what employers really want, and you have a keen eye for spotting red
    flags in job postings. Your background includes working with both startups and
    Fortune 500 companies, giving you insight into different hiring cultures and
    expectations.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

> **Key Point:** By specifying concrete experience like "12+ years in recruitment" and detailed abilities like "reading between the lines of job descriptions" in the `backstory`, the LLM is guided to perform **expert-level analysis** rather than simple searching.

**2) Job Matching Agent (job_matching_agent)**

```yaml
job_matching_agent:
  role: >
    Job Matching Expert
  goal: >
    Evaluate a list of extracted jobs and the user's resume to determine how well
    each opportunity aligns with the candidate's skills, preferences, and career goals.
    Provide match scores and rationales to guide the user toward the best-fit roles.
  backstory: >
    You are an intelligent job match evaluator trained on thousands of hiring decisions
    and successful placements. You analyze roles based on hard skills, soft skills, work
    preferences, and red flags mentioned in resumes. You understand that not all job
    titles are created equal, and that the fit depends on nuanced alignment between a
    candidate's profile and the opportunity's true requirements. Your job is to score
    each opportunity from 1 to 5 and justify that score clearly.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

**3) Resume Optimization Agent (resume_optimization_agent)**

```yaml
resume_optimization_agent:
  role: >
    Resume Optimization Specialist
  goal: >
    Rewrite and tailor the user's resume to closely match the selected job opportunity,
    increasing their chances of landing an interview.
  backstory: >
    You are a seasoned resume expert and former recruiter who has reviewed thousands of
    applications across tech, finance, and creative industries. You know exactly how to
    align a candidate's background to what employers are looking for. You understand how
    to optimize resumes with ATS-friendly keywords, clear summaries, and industry-relevant
    framing.
  verbose: true
  respect_context_window: true
  llm: openai/o4-mini-2025-04-16
```

> **Key Point:** The `respect_context_window: true` option ensures the agent respects the LLM's context window size, preventing errors from overly long inputs. This is especially useful for agents that handle a lot of text, like resume rewriting.

**4) Company Research Agent (company_research_agent)**

```yaml
company_research_agent:
  role: >
    Company Research and Interview Strategist
  goal: >
    Help candidates deeply understand the company they are applying to and anticipate
    key interview themes.
  backstory: >
    You are a hybrid of a recruiter, career coach, and market analyst. You've advised
    thousands of job seekers on how to position themselves based on company signals,
    mission alignment, and role structure.
  verbose: true
  respect_context_window: true
  llm: openai/o4-mini-2025-04-16
```

**5) Interview Prep Agent (interview_prep_agent)**

```yaml
interview_prep_agent:
  role: >
    Interview Strategist and Preparation Coach
  goal: >
    Generate a sharp, confident, and well-informed briefing for a job interview using
    all available assets.
  backstory: >
    You are a former head of talent, now an elite interview coach for engineers and
    product teams. You specialize in converting raw candidate and company data into
    clear interview strategies.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

#### tasks.yaml - Defining 6 Tasks

Each Task is composed of three core fields: `description`, `expected_output`, and `agent`.

**1) Job Extraction Task (job_extraction_task)**

```yaml
job_extraction_task:
  description: >
    Find and extract {level} level {position} jobs in {location}.

    Steps include:
    1. Use Web Search Tool to search for {level} level {position} jobs in {location}.
    2. Extract the job listings from the search results.
    3. Filter out job listings that are not {level} level {position} jobs in {location}.
  expected_output: >
    A JSON object matching the `JobList` schema.
  agent: job_search_agent
```

> **Key Point:** `{level}`, `{position}`, and `{location}` are **template variables**. They are replaced with actual values via `kickoff(inputs={...})` at runtime. This allows a single Task definition to handle various search conditions.

**2) Job Matching Task (job_matching_task)**

```yaml
job_matching_task:
  description: >
    Given a list of extracted jobs (JobList) and the user's resume, evaluate how well
    each job aligns with the user's:
    - Tech stack
    - Role level
    - Industry and company size preferences
    - Remote/work flexibility
    - Contract type
    - Salary expectations
    - Keywords and disqualifiers in the resume

    For each job, assign a `match_score` from 1 (poor fit) to 5 (perfect fit), and
    explain your reasoning.
  expected_output: >
    A JSON object matching the original `Job` schema, with two additional fields per job:
    - match_score: integer from 1 to 5
    - reason: a short explanation for the score
  agent: job_matching_agent
```

**3) Job Selection Task (job_selection_task)**

```yaml
job_selection_task:
  description: >
    Given a list of jobs that each contain a `match_score` and a `reason` field
    (RankedJobList), your task is to:
    1. Analyze the `match_score` and reasons to determine the best-fit job.
    2. Select the single best job.
    3. Justify your choice.
    4. Set the `selected` field to `true` for the top job and `false` for all others.
  expected_output: >
    A JSON object matching the `ChosenJob` schema of the selected job.
  agent: job_matching_agent
```

**4) Resume Rewriting Task (resume_rewriting_task)**

```yaml
resume_rewriting_task:
  description: >
    Given the user's real resume and the selected job (ChosenJob), rewrite the existing
    resume to emphasize alignment with the job, without fabricating or inflating any facts.
  expected_output: >
    A Markdown-formatted version of the real user's resume, rewritten and optimized for
    the selected job.
  agent: resume_optimization_agent
  output_file: output/rewritten_resume.md
  create_directory: true
  markdown: true
```

> **Key Point:** When `output_file` is specified, the Task's result is automatically saved to a file. `create_directory: true` automatically creates the output directory if it doesn't exist.

**5) Company Research Task (company_research_task)**

```yaml
company_research_task:
  description: >
    Given the selected job (ChosenJob), research the hiring company using public web
    resources.
  expected_output: >
    A Markdown file with the following sections:
    - ## Company Overview
    - ## Mission and Values
    - ## Recent News or Changes
    - ## Role Context and Product Involvement
    - ## Likely Interview Topics
    - ## Suggested Questions to Ask
  agent: company_research_agent
  markdown: true
  output_file: output/company_research.md
```

**6) Interview Prep Task (interview_prep_task)**

```yaml
interview_prep_task:
  description: >
    Combine the following information:
    1. The selected job (ChosenJob)
    2. The tailored resume (RewrittenResume)
    3. The company research summary (CompanyResearch)

    Create a detailed interview preparation document.
  expected_output: >
    A Markdown document titled "Interview Prep: $CompanyName - $JobTitle" with sections:
    - ## Job Overview
    - ## Why This Job Is a Fit
    - ## Resume Highlights for This Role
    - ## Company Summary
    - ## Predicted Interview Questions
    - ## Questions to Ask Them
    - ## Concepts To Know/Review
    - ## Strategic Advice
  agent: interview_prep_agent
  output_file: output/interview_prep.md
```

#### Dependency Configuration (pyproject.toml)

```toml
[project]
name = "job-hunter-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "firecrawl-py>=2.16.3",
    "python-dotenv>=1.1.1",
]
```

- `crewai[tools]`: CrewAI framework and built-in tool collection
- `firecrawl-py`: Web search and scraping API client
- `python-dotenv`: Loads environment variables (API keys, etc.) from `.env` files

#### Knowledge Source (resume.txt)

The user's resume is saved as a text file and used as a **knowledge source** for the agents:

```text
# Juan Srisaiwong
**Full Stack Developer**

## Professional Summary
Passionate Full Stack Developer with 3 years of experience building scalable web
applications and modern user interfaces. Proficient in JavaScript ecosystem with
expertise in React, Node.js, and cloud technologies.

## Technical Skills
Frontend: React, Vue.js, HTML5, CSS3, Sass, TypeScript, JavaScript (ES6+)
Backend: Node.js, Express.js, Python, Django, RESTful APIs, GraphQL
Databases: PostgreSQL, MySQL, MongoDB, Redis
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, CI/CD, Git, GitHub Actions
...
```

### Practice Points

1. **Agent design principles**: Each agent should have one clear area of expertise. "Multiple agents that each do one thing well" produce better results than "one agent that does everything."
2. **Writing backstory**: Including specific years of experience, particular areas of expertise, and work style significantly improves LLM output quality.
3. **Task description specificity**: The Task's `description` is the **actual prompt** given to the agent. Step-by-step instructions, evaluation criteria, and constraints should be clearly stated.
4. **Template variable usage**: Using variables like `{level}` and `{position}` allows the same configuration to handle various scenarios.

---

## 4.2 Context and Structured Outputs

### Topic and Objective

In this section, we implement two key features:
1. **Structured Output**: Using Pydantic models to force agent output into **structured data**
2. **Context Passing**: Configuring a **dependency chain** that automatically passes preceding Task results to subsequent Tasks

### Core Concepts

#### What Is Structured Output?

LLMs fundamentally generate free-form text. However, programming workflows require data with **predictable structure**. CrewAI can force agent output to conform to specific schemas through Pydantic models.

```
Free LLM output:    "I found 3 jobs. The first one is..."  (unparseable)
Structured output:   {"jobs": [{"job_title": "...", "company_name": "..."}]}  (programmable)
```

#### What Is Context?

In CrewAI, `context` defines the **data flow between Tasks**. It is used when passing the result of Task A as input to Task B.

```
job_extraction_task -> job_matching_task -> job_selection_task -+-> resume_rewriting_task --+
                                                                +-> company_research_task --+
                                                                +---------------------------+-> interview_prep_task
```

In the diagram above:
- `resume_rewriting_task` receives the result of `job_selection_task` (the selected job) as context
- `interview_prep_task` receives the results of all three Tasks (job_selection, resume_rewriting, company_research) as context

### Code Analysis

#### models.py - Pydantic Model Definitions

```python
from typing import List
from pydantic import BaseModel
from datetime import date


class Job(BaseModel):

    job_title: str
    company_name: str
    job_location: str
    is_remote_friendly: bool | None = None
    employment_type: str | None = None
    compensation: str | None = None
    job_posting_url: str
    job_summary: str

    key_qualifications: List[str] | None = None
    job_responsibilities: List[str] | None = None
    date_listed: date | None = None
    required_technologies: List[str] | None = None
    core_keywords: List[str] | None = None

    role_seniority_level: str | None = None
    years_of_experience_required: str | None = None
    minimum_education: str | None = None
    job_benefits: List[str] | None = None
    includes_equity: bool | None = None
    offers_visa_sponsorship: bool | None = None
    hiring_company_size: str | None = None
    hiring_industry: str | None = None
    source_listing_url: str | None = None
    full_raw_job_description: str | None = None


class JobList(BaseModel):
    jobs: List[Job]


class RankedJob(BaseModel):
    job: Job
    match_score: int
    reason: str


class RankedJobList(BaseModel):
    ranked_jobs: List[RankedJob]


class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
```

**Core Design Principles of the Model Structure:**

1. **Distinguishing required and optional fields**: `job_title` and `company_name` are always needed, but information like `includes_equity` and `offers_visa_sponsorship` is not present in every job posting. Defining optional fields with `| None = None` ensures flexibility.

2. **Progressive data enrichment pattern**: Data becomes richer as it passes through the pipeline:
   - `Job`: Basic job information
   - `RankedJob`: Job + match score + reason (evaluation result added)
   - `ChosenJob`: Job + selected status + reason (decision result added)

3. **Composition pattern**: `RankedJob` does not inherit from `Job` but rather **contains (composition)** it. This allows appending additional information without modifying the original Job data.

#### main.py - Crew Configuration and Execution

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, task, agent, crew
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool


@CrewBase
class JobHunterCrew:

    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config["job_search_agent"],
            tools=[web_search_tool],
        )

    @agent
    def job_matching_agent(self):
        return Agent(config=self.agents_config["job_matching_agent"])

    @agent
    def resume_optimization_agent(self):
        return Agent(config=self.agents_config["resume_optimization_agent"])

    @agent
    def company_research_agent(self):
        return Agent(config=self.agents_config["company_research_agent"])

    @agent
    def interview_prep_agent(self):
        return Agent(config=self.agents_config["interview_prep_agent"])
```

**Role of the `@CrewBase` Decorator:**

`@CrewBase` automatically provides the following capabilities to the class:
- `self.agents_config`: Automatically loads the `config/agents.yaml` file
- `self.tasks_config`: Automatically loads the `config/tasks.yaml` file
- `self.agents`: Collects return values of all methods with the `@agent` decorator into a list
- `self.tasks`: Collects return values of all methods with the `@task` decorator into a list

**Task Definitions and Structured Output Connection:**

```python
    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,  # Force output to JobList schema
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,  # Force output to RankedJobList schema
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,  # Force output to ChosenJob schema
        )
```

> **Key Point:** When `output_pydantic=JobList` is specified, CrewAI automatically parses the LLM's output to conform to the `JobList` schema. If output that doesn't match the schema is generated, it automatically retries.

**Data Passing Between Tasks via Context:**

```python
    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),  # Pass selected job information
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),  # Pass selected job information
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),      # Selected job
                self.resume_rewriting_task(),   # Rewritten resume
                self.company_research_task(),   # Company research results
            ],
        )
```

> **Key Point:** The `context` parameter accepts a **list** of multiple Tasks. `interview_prep_task` references the results of all 3 Tasks to generate comprehensive interview preparation materials. CrewAI waits until all Tasks specified in the context are completed before executing that Task.

**Crew Assembly and Execution:**

```python
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,  # @agent methods auto-collected
            tasks=self.tasks,    # @task methods auto-collected
            verbose=True,        # Print detailed execution progress
        )


JobHunterCrew().crew().kickoff()
```

`self.agents` and `self.tasks` are properties automatically generated by `@CrewBase`, which collect the return values of all methods with `@agent` and `@task` decorators respectively, in order, into lists.

### Practice Points

1. **Pydantic model design**: Keep required fields minimal, and set optional fields with `| None = None` to accommodate various data sources.
2. **Context dependency graph**: Drawing a graph of inter-Task dependencies helps identify Tasks that can run in parallel. For example, `resume_rewriting_task` and `company_research_task` only need the same context, so they can run in parallel.
3. **`output_pydantic` vs `output_file`**: Use `output_pydantic` when you need structured data, and `output_file` when you need human-readable documents. Both can be used simultaneously.

---

## 4.3 Firecrawl Tool - Custom Web Search Tool

### Topic and Objective

Create a **custom Tool** that enables CrewAI agents to interact with the outside world. Implement a tool that performs web searches using the Firecrawl API, cleans the results, and delivers them to the agent.

### Core Concepts

#### What Is a Tool?

In CrewAI, a Tool is a function that allows an agent to **access external resources** or **perform specific operations**. The agent calls a Tool at the appropriate moment based on LLM judgment.

```
Agent's thought process:
1. "I need to find Senior level Golang Developer job postings"
2. "Let me use web_search_tool"
3. web_search_tool("Senior Golang Developer jobs Netherlands") called
4. Analyze results and organize according to JobList schema
```

#### What Is Firecrawl?

Firecrawl is an API service that crawls web pages and converts content into **markdown format**. Unlike typical web scraping:
- Supports JavaScript rendering (can crawl SPA pages)
- Automatically removes unnecessary elements (ads, navigation, etc.)
- Converts to clean markdown text

### Code Analysis

#### tools.py - Web Search Tool Implementation

```python
import os, re

from crewai.tools import tool
from firecrawl import FirecrawlApp, ScrapeOptions


@tool
def web_search_tool(query: str):
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

**Detailed Code Analysis:**

1. **`@tool` decorator**: CrewAI's tool decorator that converts a regular function into a Tool usable by agents. The function's name and docstring are passed to the agent as the tool description.

2. **FirecrawlApp initialization**: Authenticates by reading `FIRECRAWL_API_KEY` from the `.env` file.

3. **Search execution**:
   ```python
   response = app.search(
       query=query,       # Search query
       limit=5,           # Maximum 5 results
       scrape_options=ScrapeOptions(
           formats=["markdown"],  # Return in markdown format
       ),
   )
   ```
   The reason for limiting with `limit=5` is to conserve the LLM's context window and prevent confusion from too much information.

4. **Data Cleaning**:
   ```python
   # Remove unnecessary backslashes and consecutive line breaks
   cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
   # Remove markdown links and URLs (to save tokens)
   cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)
   ```
   Web crawling results contain many unnecessary links and special characters. Removing them:
   - **Saves tokens** sent to the LLM
   - Helps the agent **focus on the core content**
   - Reduces API costs

5. **Return value**: Returns a list of dictionaries containing the title, URL, and cleaned markdown content.

### Practice Points

1. **Using the `@tool` decorator**: Any Python function can be used as an agent tool by adding `@tool`. It can be used for various purposes such as database queries, external API calls, and file processing.
2. **Importance of data cleaning**: The cleaner the data passed to the LLM, the better. Removing unnecessary HTML tags, URLs, and special characters improves output quality.
3. **Error handling**: `if not response.success` handles API call failures. In production environments, more sophisticated error handling and retry logic would be needed.
4. **Result limiting**: Limiting search results with `limit=5` is an important design decision balancing cost and quality.

---

## 4.5 Conclusions - Knowledge Source and Final Execution

### Topic and Objective

In the final section, we complete the following:
1. **Knowledge Source** connection: Inject the resume text file as agent knowledge
2. **Tool docstring addition**: Add descriptions so agents can use tools correctly
3. **Execution input passing**: Pass actual values to template variables via `kickoff(inputs={...})`
4. **Result verification**: Execute the entire pipeline and verify the output

### Core Concepts

#### What Is a Knowledge Source?

CrewAI's Knowledge Source is a mechanism for providing **pre-existing knowledge** to agents. It automatically includes data from various formats such as text files, PDFs, CSVs, etc. in the agent's context.

Unlike inserting text directly into a prompt, Knowledge Source:
- Uses a **vector database (ChromaDB)** to search for relevant information
- Can efficiently process large documents
- Allows sharing the same knowledge across multiple agents

#### The Importance of Tool Docstrings

Agents **read the Tool's docstring** to determine when and how to use that tool. Without a clear docstring, agents may misuse a tool or not use it at all.

### Code Analysis

#### main.py - Adding Knowledge Source and Final Configuration

**Knowledge Source Setup:**

```python
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt",
    ]
)
```

`TextFileKnowledgeSource` reads text files and internally stores them as embeddings in a ChromaDB vector database. When an agent later asks a relevant question, appropriate information is automatically provided through similarity search.

**Connecting Knowledge Source to Agents:**

```python
@agent
def job_matching_agent(self):
    return Agent(
        config=self.agents_config["job_matching_agent"],
        knowledge_sources=[resume_knowledge],  # Add resume knowledge
    )

@agent
def resume_optimization_agent(self):
    return Agent(
        config=self.agents_config["resume_optimization_agent"],
        knowledge_sources=[resume_knowledge],  # Add resume knowledge
    )

@agent
def company_research_agent(self):
    return Agent(
        config=self.agents_config["company_research_agent"],
        knowledge_sources=[resume_knowledge],  # Add resume knowledge
        tools=[web_search_tool],               # Add web search tool
    )

@agent
def interview_prep_agent(self):
    return Agent(
        config=self.agents_config["interview_prep_agent"],
        knowledge_sources=[resume_knowledge],  # Add resume knowledge
    )
```

> **Key Point:** All agents share the same `resume_knowledge` instance. Through this:
> - `job_matching_agent`: Calculates match scores between job postings and the resume
> - `resume_optimization_agent`: References the original resume for rewriting
> - `company_research_agent`: Considers the resume's tech stack in company research + web search
> - `interview_prep_agent`: Generates interview preparation materials reflecting resume content

**Note that only `company_research_agent` has `tools=[web_search_tool]` added.** This agent needs search tools to research company information from the web. In contrast, `resume_optimization_agent` and `interview_prep_agent` have sufficient information from context already passed to them.

#### tools.py - Adding Docstrings

```python
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
    # ... (rest is the same)
```

The docstring serves as the agent's **tool usage guide**. The agent reads this description and understands:
- What arguments to pass (`query: str`)
- What results to expect (a list of search results in markdown format)

and calls the tool appropriately.

#### Execution Input Passing and Result Output

```python
result = (
    JobHunterCrew()
    .crew()
    .kickoff(
        inputs={
            "level": "Senior",
            "position": "Golang Developer",
            "location": "Netherlands",
        }
    )
)

for task_output in result.tasks_output:
    print(task_output.pydantic)
```

The template variables defined in YAML are replaced via `kickoff(inputs={...})`:

```yaml
# Original in tasks.yaml
description: >
  Find and extract {level} level {position} jobs in {location}.

# Result after substitution at execution
description: >
  Find and extract Senior level Golang Developer jobs in Netherlands.
```

`result.tasks_output` contains each Task's execution results in order, and you can directly access the Pydantic model instance via `task_output.pydantic`.

#### Execution Result Examples

When the entire pipeline runs, three markdown files are generated in the `output/` directory:

**1) output/rewritten_resume.md** - Resume rewritten to match the selected job (Senior Golang Developer):
- Title changed from "Full Stack Developer" to "Senior Backend Developer (API Design | Microservices | Cloud-Native)"
- Tech stack restructured for Go-based FinTech environment
- Experience descriptions rewritten focusing on APIs, microservices, and performance optimization

**2) output/company_research.md** - Research report on the selected company (FinTech Innovators):
- Company overview, mission and values, recent news
- Analysis of the role's tech stack (Go, Kafka, Kubernetes, Terraform, etc.)
- Expected interview topics and questions for the applicant to ask

**3) output/interview_prep.md** - Comprehensive interview preparation document:
- Job overview and fit analysis
- Resume highlights
- Predicted interview questions (Golang, API design, event-driven architecture, etc.)
- Strategic advice (confident learner attitude, solution-oriented approach, etc.)

### Practice Points

1. **Knowledge Source usage**: You can provide background knowledge needed by agents as files -- resumes, company information, product documentation, etc. Large documents are efficiently processed through the vector database.
2. **Tool docstrings**: Clear docstrings are essential for agents to use tools correctly. Include argument descriptions, return value formats, and usage scenarios.
3. **Input templates**: Through `kickoff(inputs={...})`, you can reuse the same agent system under various conditions.
4. **Result access**: Use `result.tasks_output` to programmatically access each Task's results for post-processing.

---

## Chapter Key Takeaways

### 1. CrewAI's Declarative Structure

| Component | Definition Location | Role |
|-----------|-------------------|------|
| Agent | `config/agents.yaml` | Define role, goal, backstory |
| Task | `config/tasks.yaml` | Define task description, expected output, assigned agent |
| Crew | `main.py` (`@CrewBase`) | Combine agents and tasks for execution |

### 2. Data Flow Control

- **Structured Output (`output_pydantic`)**: Structure agent output with Pydantic models
- **Context**: Define inter-Task data dependencies with `context=[task_a(), task_b()]`
- **Knowledge Source**: Store text files etc. in a vector DB and use as prior knowledge for agents

### 3. Custom Tool Creation

- Convert Python functions to agent tools with the `@tool` decorator
- Guide correct tool usage with clear docstrings
- Always clean external API results to ensure token efficiency

### 4. Multi-Agent Collaboration Pattern

```
Search Agent --> Matching Agent --> Selection Agent --+-> Resume Agent (context: selection result)
                                                      +-> Company Research Agent (context: selection result)
                                                      +-> Interview Prep Agent (context: selection+resume+research)
```

- Each agent follows the **single responsibility principle**
- Distinguish between Tasks requiring sequential execution and those that can run in parallel
- Design information to flow naturally through context

### 5. Practical Design Tips

- **The longer the backstory, the better**: Detailing specific experience, expertise areas, and work style leads the LLM to generate more professional output
- **Be generous with fields, strict with output**: Include sufficient optional fields (`| None = None`) in Pydantic models while making core fields required
- **Save tokens**: Remove unnecessary links and special characters from web crawling results to reduce API costs
- **`respect_context_window: true`**: Set this for agents handling lots of text to prevent context window overflow errors

---

## Practice Assignments

### Assignment 1: Customizing Agent Roles (Difficulty: Easy)

Modify the agent settings in `agents.yaml` to create a Job Hunter Agent suitable for **a different profession (e.g., designer, marketer)**.

**Requirements:**
- Modify `job_search_agent`'s backstory to match the job market for that profession
- Modify `resume_optimization_agent`'s backstory to match resume writing conventions for that profession
- Replace `knowledge/resume.txt` with a new resume
- Change the input values in `kickoff(inputs={...})` and run

### Assignment 2: Adding New Pydantic Models (Difficulty: Medium)

Add the following fields to the current `ChosenJob` model and modify the related Task's description and expected_output:

```python
class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
    # New fields to add
    salary_competitiveness: str       # "above_market", "at_market", "below_market"
    career_growth_potential: int      # Score from 1-5
    work_life_balance_score: int      # Score from 1-5
    recommended_negotiation_points: List[str]  # List of negotiation points
```

### Assignment 3: Creating a New Tool (Difficulty: Medium)

Create a custom Tool using a different API instead of Firecrawl. Example:

```python
@tool
def glassdoor_review_tool(company_name: str):
    """
    Glassdoor Review Tool.
    Args:
        company_name: str
            The company name to search reviews for.
    Returns:
        A summary of employee reviews for the company.
    """
    # Try implementing this
    pass
```

**Hint:** Use SerpAPI, Google Custom Search API, etc. to implement logic that searches for and cleans company reviews.

### Assignment 4: Parallel Execution Optimization (Difficulty: Hard)

In the current pipeline, `resume_rewriting_task` and `company_research_task` only need the same context (`job_selection_task`), so they could theoretically run in parallel. Research CrewAI's `Process.hierarchical` or async execution features and modify these two Tasks to run in parallel.

```python
@crew
def crew(self):
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        verbose=True,
        process=Process.hierarchical,  # Hierarchical execution mode
        manager_llm="openai/gpt-4o",   # Manager LLM
    )
```

### Assignment 5: Extending the Full Pipeline (Difficulty: Hard)

Add new agents and Tasks to extend the pipeline:

1. **salary_negotiation_agent**: An agent that analyzes the salary range for the selected job and suggests negotiation strategies
2. **cover_letter_agent**: An agent that automatically generates a cover letter based on the resume and company research results

For each agent:
- Define role, goal, and backstory in `agents.yaml`
- Define description and expected_output in `tasks.yaml`
- Add necessary Pydantic models in `models.py`
- Add `@agent` and `@task` methods in `main.py` with appropriate context settings

---

## Reference: Complete Final Code

Below is the final state of all core files completed in this chapter.

### main.py (Final)

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, task, agent, crew
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool

resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt",
    ]
)


@CrewBase
class JobHunterCrew:

    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config["job_search_agent"],
            tools=[web_search_tool],
        )

    @agent
    def job_matching_agent(self):
        return Agent(
            config=self.agents_config["job_matching_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @agent
    def resume_optimization_agent(self):
        return Agent(
            config=self.agents_config["resume_optimization_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @agent
    def company_research_agent(self):
        return Agent(
            config=self.agents_config["company_research_agent"],
            knowledge_sources=[resume_knowledge],
            tools=[web_search_tool],
        )

    @agent
    def interview_prep_agent(self):
        return Agent(
            config=self.agents_config["interview_prep_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,
        )

    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),
                self.resume_rewriting_task(),
                self.company_research_task(),
            ],
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )


result = (
    JobHunterCrew()
    .crew()
    .kickoff(
        inputs={
            "level": "Senior",
            "position": "Golang Developer",
            "location": "Netherlands",
        }
    )
)

for task_output in result.tasks_output:
    print(task_output.pydantic)
```

### models.py (Final)

```python
from typing import List
from pydantic import BaseModel
from datetime import date


class Job(BaseModel):
    job_title: str
    company_name: str
    job_location: str
    is_remote_friendly: bool | None = None
    employment_type: str | None = None
    compensation: str | None = None
    job_posting_url: str
    job_summary: str
    key_qualifications: List[str] | None = None
    job_responsibilities: List[str] | None = None
    date_listed: date | None = None
    required_technologies: List[str] | None = None
    core_keywords: List[str] | None = None
    role_seniority_level: str | None = None
    years_of_experience_required: str | None = None
    minimum_education: str | None = None
    job_benefits: List[str] | None = None
    includes_equity: bool | None = None
    offers_visa_sponsorship: bool | None = None
    hiring_company_size: str | None = None
    hiring_industry: str | None = None
    source_listing_url: str | None = None
    full_raw_job_description: str | None = None


class JobList(BaseModel):
    jobs: List[Job]


class RankedJob(BaseModel):
    job: Job
    match_score: int
    reason: str


class RankedJobList(BaseModel):
    ranked_jobs: List[RankedJob]


class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
```

### tools.py (Final)

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
