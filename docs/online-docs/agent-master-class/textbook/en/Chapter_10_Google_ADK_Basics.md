# Chapter 10: Google ADK Basics - Building a Financial Advisor Agent

---

## Chapter Overview

In this chapter, we learn how to build a real financial advisor AI agent from start to finish using **Google ADK (Agent Development Kit)**. Google ADK is Google's agent development framework that systematically provides core capabilities needed for agent development, including agent creation, tool integration, sub-agent composition, state management, and artifact storage.

Through this chapter, you will learn:

- Initial setup of a Google ADK project and how to launch the ADK Web UI
- How to connect Tools and Sub-agents to an agent
- Designing hierarchical architectures between root agents and sub-agents
- Managing data flow through State sharing between agents
- File creation and storage using Artifacts

### Project Structure

The final project directory structure looks like this:

```
financial-analyst/
├── .python-version
├── pyproject.toml
├── tools.py
├── uv.lock
└── financial_advisor/
    ├── __init__.py
    ├── agent.py                    # Root agent definition
    ├── prompt.py                   # Root agent prompt
    └── sub_agents/
        ├── __init__.py
        ├── data_analyst.py         # Data analysis sub-agent
        ├── financial_analyst.py    # Financial analysis sub-agent
        └── news_analyst.py         # News analysis sub-agent
```

---

## 10.1 ADK Web - Initial Project Setup

### Topic and Objective

Set up a Google ADK project from scratch and build an environment where agents can be tested through the ADK Web UI. In this section, we learn ADK's project structure conventions and basic agent creation.

### Core Concepts

#### What is Google ADK?

Google ADK (Agent Development Kit) is Google's official framework for building AI agents. ADK has the following characteristics:

- **Standardized agent definition**: Agents are defined declaratively through the `Agent` class.
- **ADK Web UI provided**: Agents can be tested immediately through a web interface without separate frontend development.
- **Multiple LLM support**: In addition to Google's Gemini models, other models such as OpenAI can be used through `LiteLlm`.

#### ADK Project Structure Rules

ADK requires a specific project structure. For the ADK Web UI to automatically recognize agents, the following rules must be followed:

1. **Package directory**: Agent code must be inside a Python package (directory + `__init__.py`).
2. **Import agent module in `__init__.py`**: The package's `__init__.py` must import the `agent` module.
3. **`root_agent` variable**: A variable named `root_agent` must exist in the `agent.py` file. The ADK Web UI uses this variable as the entry point.

#### LiteLlm Integration

Google ADK uses Google's Gemini models by default, but through the `LiteLlm` wrapper, models from various LLM providers such as OpenAI and Anthropic can be used. This allows you to use your existing models directly within the ADK framework.

### Code Analysis

#### Project Dependency Configuration (`pyproject.toml`)

```toml
[project]
name = "financial-analyst"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "firecrawl-py==2.16.3",
    "google-adk>=1.11.0",
    "google-genai>=1.31.0",
    "litellm>=1.75.8",
    "python-dotenv>=1.1.1",
    "yfinance>=0.2.65",
]

[dependency-groups]
dev = [
    "watchdog>=6.0.0",
]
```

Let's examine the key dependencies:

| Package | Role |
|---------|------|
| `google-adk` | Google Agent Development Kit core library |
| `google-genai` | Google Generative AI client |
| `litellm` | Wrapper library integrating various LLM APIs |
| `yfinance` | Library for fetching stock data from Yahoo Finance |
| `firecrawl-py` | Web search and scraping API client |
| `python-dotenv` | Loads environment variables from `.env` files |
| `watchdog` | File change detection (for auto-reload during development) |

#### Package Initialization (`__init__.py`)

```python
from . import agent
```

This single line is very important. When the ADK Web UI loads the package, it first executes `__init__.py`, and by importing the `agent` module here, it gains access to the `root_agent` variable.

#### Basic Agent Definition (`agent.py`)

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm("openai/gpt-4o")


weather_agent = Agent(
    name="WeatherAgent",
    instruction="You help the user with weather related questions",
    model=MODEL,
)

root_agent = weather_agent
```

**Code Analysis:**

- `Agent`: The base class for agents provided by ADK. It declaratively defines the agent's name, instruction, and model to use.
- `LiteLlm("openai/gpt-4o")`: Uses LiteLlm to specify OpenAI's GPT-4o model. The provider and model name are specified together in the `"openai/gpt-4o"` format.
- `root_agent = weather_agent`: Assigns the agent to the `root_agent` variable so that the ADK Web UI can recognize it. This variable name **must be `root_agent`**.

#### Web Search Tool (`tools.py`)

```python
import dotenv
dotenv.load_dotenv()
import re
import os
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

This tool performs web searches using the Firecrawl API. The core operations are:

1. **Load environment variables**: Loads `FIRECRAWL_API_KEY` from the `.env` file using `dotenv.load_dotenv()`.
2. **Execute search**: Searches the query with `app.search()` and retrieves up to 5 results in markdown format.
3. **Clean results**: Uses regular expressions to remove unnecessary backslashes, line breaks, URLs, and markdown links to produce clean text.
4. **Structured return**: Organizes each result into a dictionary with `title`, `url`, and `markdown` keys, returning them as a list.

> **Note**: The **docstring** of tool functions is very important. ADK (and most agent frameworks) passes the docstring to the LLM as the tool's usage instructions. Therefore, Args, Returns, etc. should be written clearly.

### Practice Points

1. Initialize the project and install dependencies using `uv`:
   ```bash
   cd financial-analyst
   uv sync
   ```
2. Launch the ADK Web UI and verify that the basic agent works:
   ```bash
   adk web
   ```
3. Try changing the `root_agent` variable name to something else and observe how the ADK Web UI responds (it will produce an error).
4. Set up the required API keys in the `.env` file.

---

## 10.2 Tools and Subagents - Tools and Sub-agents

### Topic and Objective

Learn how to add **Tools** and **Sub-agents** to an agent. Tools allow the agent to invoke external functionality, while sub-agents allow delegating specific tasks to other agents.

### Core Concepts

#### Tools

In ADK, tools are defined as **regular Python functions**. The agent uses the LLM's Function Calling capability to invoke these tools at appropriate moments. Functions used as tools must meet the following conditions:

- **Type hints**: Parameters must have type hints so the LLM can pass the correct arguments.
- **Docstring**: The function's docstring tells the LLM the tool's purpose and usage.
- **Return value**: Must return a serializable value such as a string or dictionary.

#### Sub-agents

Sub-agents are independent agents placed below the main agent. When the main agent receives a certain type of question, it **transfers** the conversation to the sub-agent capable of handling it. The `description` attribute of sub-agents is crucial, as the root agent uses this description to determine which sub-agent to transfer work to.

### Code Analysis

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm("openai/gpt-4o")


def get_weather(city: str):
    return f"The weather in {city} is 30 degrees."


def convert_units(degrees: int):
    return f"That is 40 farenheit"


geo_agent = Agent(
    name="GeoAgent",
    instruction="You help with geo questions",
    model=MODEL,
    description="Transfer to this agent when you have a geo related question.",
)

weather_agent = Agent(
    name="WeatherAgent",
    instruction="You help the user with weather related questions",
    model=MODEL,
    tools=[
        get_weather,
        convert_units,
    ],
    sub_agents=[
        geo_agent,
    ],
)

root_agent = weather_agent
```

**Code Analysis:**

1. **Tool function definitions**:
   - `get_weather(city: str)`: A tool that takes a city name and returns weather information. The `city: str` type hint lets the LLM know it should pass a string argument.
   - `convert_units(degrees: int)`: A tool for converting temperature units.

2. **Sub-agent definition**:
   ```python
   geo_agent = Agent(
       name="GeoAgent",
       instruction="You help with geo questions",
       model=MODEL,
       description="Transfer to this agent when you have a geo related question.",
   )
   ```
   - The `description` attribute is key. The root agent (WeatherAgent) reads this description and decides to transfer the conversation to GeoAgent when the user's question is geography-related.

3. **Connecting tools and sub-agents to the agent**:
   ```python
   weather_agent = Agent(
       ...
       tools=[get_weather, convert_units],
       sub_agents=[geo_agent],
   )
   ```
   - Python functions are passed directly to the `tools` list. ADK automatically analyzes the function's signature and docstring to convert them into a tool schema that the LLM can understand.
   - Sub-agent instances are passed to the `sub_agents` list.

#### Tools vs Sub-agents Comparison

| Property | Tool | Sub-agent |
|----------|------|-----------|
| Definition | Python function | Agent instance |
| Execution | Current agent calls it directly | Conversation is transferred to the sub-agent |
| Complexity | Suitable for simple tasks | Suitable for complex multi-step tasks |
| LLM usage | Not used (pure code execution) | Performs reasoning with its own LLM |
| Best for | API calls, data retrieval | Specialized analysis, independent conversations |

### Practice Points

1. Try removing type hints from the tool functions and running them. Observe cases where the LLM fails to pass arguments correctly.
2. Try removing `geo_agent`'s `description` and running it. Check how it affects the root agent's ability to transfer work to the sub-agent.
3. Try adding a new tool function (e.g., `get_humidity(city: str)`).

---

## 10.3 Agent Architecture - Designing the Agent Architecture

### Topic and Objective

Design and implement the complete architecture of a real financial advisor agent. Build a structure where the root agent leverages multiple specialized sub-agents as tools, and learn about the agent-as-tool pattern using `AgentTool` and detailed system prompt writing.

### Core Concepts

#### AgentTool - Using Agents as Tools

The `sub_agents` approach learned in 10.2 transfers the conversation itself to the sub-agent. In contrast, `AgentTool` uses sub-agents **like tools**. The key differences between the two approaches are:

| Approach | `sub_agents=[]` | `AgentTool(agent=)` |
|----------|-----------------|---------------------|
| Behavior | Conversation control transfers to sub-agent | Root agent retains control while calling the sub-agent |
| Analogy | "Direct this customer to another department" | "Call another department and bring back the information" |
| Result | Sub-agent directly converses with the user | Sub-agent's results are returned to the root agent |
| Best for | Tasks in a completely different domain | Cases requiring information gathering followed by synthesis |

Since the financial advisor needs to **synthesize** multiple analysis results, the `AgentTool` pattern is more suitable. The root agent can collect data analysis, financial analysis, and news analysis results before providing the final investment advice.

#### Hierarchical Agent Architecture

The architecture designed in this project is as follows:

```
FinancialAdvisor (Root Agent)
├── DataAnalyst (Sub-agent/Tool)
│   ├── get_company_info()
│   ├── get_stock_price()
│   └── get_financial_metrics()
├── FinancialAnalyst (Sub-agent/Tool)
│   ├── get_income_statement()
│   ├── get_balance_sheet()
│   └── get_cash_flow()
├── NewsAnalyst (Sub-agent/Tool)
│   └── web_search_tool()
└── save_advice_report() (Direct Tool)
```

### Code Analysis

#### Root Agent (`agent.py`)

```python
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.data_analyst import data_analyst
from .sub_agents.financial_analyst import financial_analyst
from .sub_agents.news_analyst import news_analyst
from .prompt import PROMPT

MODEL = LiteLlm("openai/gpt-4o")


def save_advice_report():
    pass


financial_advisor = Agent(
    name="FinancialAdvisor",
    instruction=PROMPT,
    model=MODEL,
    tools=[
        AgentTool(agent=financial_analyst),
        AgentTool(agent=news_analyst),
        AgentTool(agent=data_analyst),
        save_advice_report,
    ],
)

root_agent = financial_advisor
```

**Key Analysis:**

- `AgentTool(agent=financial_analyst)`: Wraps the sub-agent with `AgentTool` and places it in the `tools` list. This allows the root agent to call the sub-agent as if it were a tool.
- `save_advice_report`: Regular Python functions can also be included in the `tools` list alongside `AgentTool`. At this point it is still an empty function (`pass`), and will be implemented in a later section.
- Note that the sub-agents are passed to `tools` wrapped in `AgentTool`, not to `sub_agents`.

#### Root Agent Prompt (`prompt.py`)

```python
PROMPT = """
You are a Professional Financial Advisor specializing in equity analysis
and investment recommendations. You ONLY provide advice on stocks, trading,
and investment decisions.

**STRICT SCOPE LIMITATIONS:**
- ONLY answer questions about stocks, trading, investments, financial markets,
  and company analysis
- REFUSE to answer questions about: general knowledge, technology help,
  personal advice unrelated to finance, or any non-financial topics
- If asked about non-financial topics, politely redirect: "I'm a specialized
  financial advisor. I can only help with stock analysis and investment decisions."

**INVESTMENT RECOMMENDATION PROCESS:**
Before providing any BUY/SELL/HOLD recommendation, you MUST:
1. Ask about the user's investment goals (growth, income, speculation, etc.)
2. Ask about their risk tolerance (conservative, moderate, aggressive)
3. Ask about their investment timeline (short-term, medium-term, long-term)

**Available Specialized Tools:**
- **data_analyst**: Gathers market data, company info, pricing, and financial metrics
- **news_analyst**: Searches current news and industry information using web tools
- **financial_analyst**: Analyzes detailed financial statements including income,
  balance sheet, and cash flow

**Direct Tools:**
- **save_company_report()**: Save comprehensive reports as artifacts

**ANALYSIS METHODOLOGY:**
For thorough analysis, you should:
1. Gather quantitative data (financial metrics, performance, valuation)
2. Research current news and market sentiment
3. Analyze financial statements for fundamental strength
4. Consider user's specific goals and risk profile
5. Provide confident recommendation with clear reasoning

**Communication Style:**
- Be confident and decisive in recommendations
- Use specific data points and metrics
- Explain reasoning clearly with supporting evidence
- Provide actionable investment guidance
- Show conviction in your analysis

You are an expert who makes money for clients through sound investment decisions.
Be opinionated and confident.
"""
```

**Prompt Design Analysis:**

This prompt is a good example of agent prompt engineering:

1. **Role definition**: Assigns a clear role as "Professional Financial Advisor."
2. **Scope limitation**: Explicitly restricts the agent from responding to non-financial questions. This prevents the agent from drifting into unintended areas.
3. **Process definition**: Specifies mandatory steps before making any BUY/SELL/HOLD recommendation.
4. **Tool guidance**: Lists available tools and their purposes within the prompt, enabling the LLM to select the appropriate tool.
5. **Analysis methodology**: Provides a systematic analysis procedure.

#### Data Analysis Sub-agent (`sub_agents/data_analyst.py`)

```python
import yfinance as yf
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o")


def get_company_info(ticker: str) -> str:
    """
    Retrieves basic company information for a given stock ticker.
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    Returns:
        dict: A dictionary containing ticker, success, company_name, industry, sector
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "success": True,
        "company_name": info.get("longName", "NA"),
        "industry": info.get("industry", "NA"),
        "sector": info.get("sector", "NA"),
    }


def get_stock_price(ticker: str, period: str) -> str:
    """
    Fetches historical stock price data and current trading price.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo',
                       '1y', '2y', '5y', '10y', 'ytd', 'max')
    Returns:
        dict: A dictionary containing ticker, success, history, current_price
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period=period)
    return {
        "ticker": ticker,
        "success": True,
        "history": history.to_json(),
        "current_price": info.get("currentPrice"),
    }


def get_financial_metrics(ticker: str) -> str:
    """
    Retrieves key financial metrics and valuation ratios.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
    Returns:
        dict: A dictionary containing ticker, success, market_cap,
              pe_ratio, dividend_yield, beta
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "success": True,
        "market_cap": info.get("marketCap", "NA"),
        "pe_ratio": info.get("trailingPE", "NA"),
        "dividend_yield": info.get("dividendYield", "NA"),
        "beta": info.get("beta", "NA"),
    }


data_analyst = LlmAgent(
    name="DataAnalyst",
    model=MODEL,
    description="Gathers and analyzes basic stock market data using multiple focused tools",
    instruction="""
    You are a Data Analyst who gathers stock information using specialized tools:

    1. **get_company_info(ticker)** - Learn about the company (name, sector, industry)
    2. **get_stock_price(ticker, period)** - Get current pricing and trading ranges
    3. **get_financial_metrics(ticker)** - Check key financial ratios

    Use multiple focused tools to gather different types of data.
    Explain what each tool provides and present the information clearly.
    """,
    tools=[
        get_company_info,
        get_stock_price,
        get_financial_metrics,
    ],
)
```

**Key Analysis:**

- **`LlmAgent` vs `Agent`**: `LlmAgent` is used here. `LlmAgent` is an alias for `Agent` and is functionally identical. It explicitly indicates that this is an agent powered by an LLM.
- **yfinance usage**: Each tool function uses the `yfinance` library to fetch real-time stock data from Yahoo Finance.
- **Structured return values**: All tools return consistent dictionaries containing `ticker` and `success` keys. This consistency helps the LLM parse the results.
- **`description` attribute**: Used by the root agent to understand this sub-agent's purpose when it is used as an `AgentTool`.

#### Financial Analysis Sub-agent (`sub_agents/financial_analyst.py`)

```python
import yfinance as yf
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o")


def get_income_statement(ticker: str):
    """Retrieves the income statement for revenue and profitability analysis."""
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "income_statement": stock.income_stmt.to_json(),
    }


def get_balance_sheet(ticker: str):
    """Retrieves the balance sheet for financial position analysis."""
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "balance_sheet": stock.balance_sheet.to_json(),
    }


def get_cash_flow(ticker: str):
    """Retrieves the cash flow statement for cash generation analysis."""
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "cash_flow": stock.cash_flow.to_json(),
    }


financial_analyst = Agent(
    name="FinancialAnalyst",
    model=MODEL,
    description="Analyzes detailed financial statements including income, balance sheet, and cash flow",
    instruction="""
    You are a Financial Analyst who performs deep financial statement analysis.

    1. **Income Analysis**: Use get_income_statement() to analyze revenue and margins
    2. **Balance Sheet Analysis**: Use get_balance_sheet() to examine assets and liabilities
    3. **Cash Flow Analysis**: Use get_cash_flow() to assess cash generation

    Analyze the financial health and performance of companies using comprehensive
    financial statement data.
    """,
    tools=[
        get_income_statement,
        get_balance_sheet,
        get_cash_flow,
    ],
)
```

#### News Analysis Sub-agent (`sub_agents/news_analyst.py`)

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from tools import web_search_tool

MODEL = LiteLlm(model="openai/gpt-4o")


news_analyst = Agent(
    name="NewsAnalyst",
    model=MODEL,
    description="Uses Web Search tools to search and scrape real web content from the web.",
    instruction="""
    You are a News Analyst Specialist who uses web tools to find current information.

    1. **Web Search**: Use web_search_tool() to find recent news about a company.
    3. **Summarize Findings**: Explain what you found and its relevance

    Use external APIs to search and scrape web content for current information.
    """,
    tools=[
        web_search_tool,
    ],
)
```

**Key Analysis:**

- The news analysis sub-agent leverages `web_search_tool` from the `tools.py` created in 10.1.
- It imports directly from the project root's `tools.py` via `from tools import web_search_tool`.
- Although it has only one tool, the LLM plays the role of generating search queries and analyzing the results.

### Practice Points

1. Try replacing `AgentTool` with `sub_agents` and compare the behavioral differences.
2. Try adding a new sub-agent (e.g., a `TechnicalAnalyst` that performs technical analysis).
3. Remove the "STRICT SCOPE LIMITATIONS" from the prompt and compare how the agent responds to non-financial questions.

---

## 10.4 Agent State - Agent State Management

### Topic and Objective

Learn the **State** sharing mechanism between agents. Use `output_key` to automatically save sub-agent outputs to state, and learn how to access state from tool functions through `ToolContext`.

### Core Concepts

#### ADK's State System

In ADK, State is a **key-value store** for saving and sharing data within an agent session. State has the following characteristics:

- **Session scope**: All agents within the same session share the state.
- **Dictionary format**: Accessed like a Python dictionary with `state["key"]`.
- **Automatic saving**: When `output_key` is used, the agent's final output is automatically saved to the state.

#### output_key

`output_key` is an attribute that automatically saves the agent's final response to the state:

```python
data_analyst = LlmAgent(
    ...
    output_key="data_analyst_result",
)
```

With this configuration, when the `DataAnalyst` agent completes execution, its final response is automatically saved to `state["data_analyst_result"]`.

#### ToolContext

`ToolContext` is an object that allows tool functions to access the agent's state, artifacts, and other execution context. Adding `tool_context: ToolContext` as a parameter to a tool function causes ADK to automatically inject the current context.

> **Important**: The `ToolContext` parameter is not exposed to the LLM. The LLM only sees regular parameters like `summary`, while `tool_context` is internally injected by the ADK framework.

### Code Analysis

#### Adding output_key to Sub-agents

```python
# sub_agents/data_analyst.py
data_analyst = LlmAgent(
    name="DataAnalyst",
    ...
    tools=[get_company_info, get_stock_price, get_financial_metrics],
    output_key="data_analyst_result",   # Added
)
```

```python
# sub_agents/financial_analyst.py
financial_analyst = Agent(
    name="FinancialAnalyst",
    ...
    tools=[get_income_statement, get_balance_sheet, get_cash_flow],
    output_key="financial_analyst_result",   # Added
)
```

```python
# sub_agents/news_analyst.py
news_analyst = Agent(
    name="NewsAnalyst",
    ...
    output_key="news_analyst_result",   # Added
    tools=[web_search_tool],
)
```

Each sub-agent now has an `output_key`. This means that when each sub-agent finishes execution, its result is automatically saved to the shared state.

#### Tool Function Using State (`agent.py`)

```python
from google.adk.tools import ToolContext
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.data_analyst import data_analyst
from .sub_agents.financial_analyst import financial_analyst
from .sub_agents.news_analyst import news_analyst
from .prompt import PROMPT

MODEL = LiteLlm("openai/gpt-4o")


def save_advice_report(tool_context: ToolContext, summary: str):
    state = tool_context.state
    data_analyst_result = state.get("data_analyst_result")
    financial_analyst_result = state.get("financial_analyst_result")
    news_analyst_analyst_result = state.get("news_analyst_analyst_result")
    report = f"""
        # Executive Summary and Advice:
        {summary}

        ## Data Analyst Report:
        {data_analyst_result}

        ## Financial Analyst Report:
        {financial_analyst_result}

        ## News Analyst Report:
        {news_analyst_analyst_result}
    """
    state["report"] = report
    return {
        "success": True,
    }
```

**Code Analysis:**

1. **`ToolContext` parameter**: `tool_context: ToolContext` is declared as the first parameter. ADK automatically injects the current execution context. The LLM is unaware of this parameter and only passes `summary: str` as an argument.

2. **Reading data from state**:
   ```python
   state = tool_context.state
   data_analyst_result = state.get("data_analyst_result")
   ```
   Access the state dictionary via `tool_context.state` and retrieve the results that sub-agents stored using `output_key`.

3. **Writing data to state**:
   ```python
   state["report"] = report
   ```
   Save the generated report to state. Other tools or agents can later access it via `state["report"]`.

#### Data Flow Summary

```
User: "Analyze AAPL"
    |
    v
FinancialAdvisor (Root)
    |
    +-- AgentTool(DataAnalyst) called
    |   +-- Result -> automatically saved to state["data_analyst_result"]
    |
    +-- AgentTool(FinancialAnalyst) called
    |   +-- Result -> automatically saved to state["financial_analyst_result"]
    |
    +-- AgentTool(NewsAnalyst) called
    |   +-- Result -> automatically saved to state["news_analyst_result"]
    |
    +-- save_advice_report(summary=...) called
        +-- Read all sub-agent results from state
        +-- Generate comprehensive report
        +-- Save report to state["report"]
```

### Practice Points

1. Remove `output_key` and confirm that `state.get()` returns `None` in `save_advice_report`.
2. Explore how to view data stored in state through the ADK Web UI.
3. Try saving custom data (e.g., analysis start time) to `tool_context.state` and using it.

---

## 10.5 Artifacts - File Creation with Artifacts

### Topic and Objective

Learn how to use ADK's **Artifacts** system to save data generated by agents as files. Artifacts are a mechanism for managing outputs produced by agents (reports, images, data files, etc.).

### Core Concepts

#### What are Artifacts?

Artifacts are **file-format outputs** generated during agent execution. Unlike text responses, artifacts are saved as independent files that users can download or use separately.

In the ADK Web UI, artifacts are automatically displayed in the UI, allowing users to view and download them immediately.

#### google.genai.types.Part and Blob

Artifacts are represented using `types.Part` and `types.Blob` objects from Google's `genai` library:

- **`types.Blob`**: A container that holds binary data and a MIME type.
- **`types.Part`**: A higher-level object wrapping a `Blob`, compatible with the Gemini API's multimodal message format.

#### save_artifact Method

The `save_artifact()` method of `ToolContext` is an **asynchronous** method. Therefore, the tool function calling it must also be declared as `async`.

### Code Analysis

```python
from google.genai import types
from google.adk.tools import ToolContext
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.data_analyst import data_analyst
from .sub_agents.financial_analyst import financial_analyst
from .sub_agents.news_analyst import news_analyst
from .prompt import PROMPT

MODEL = LiteLlm("openai/gpt-4o")


async def save_advice_report(tool_context: ToolContext, summary: str, ticker: str):
    state = tool_context.state
    data_analyst_result = state.get("data_analyst_result")
    financial_analyst_result = state.get("financial_analyst_result")
    news_analyst_analyst_result = state.get("news_analyst_analyst_result")
    report = f"""
        # Executive Summary and Advice:
        {summary}

        ## Data Analyst Report:
        {data_analyst_result}

        ## Financial Analyst Report:
        {financial_analyst_result}

        ## News Analyst Report:
        {news_analyst_analyst_result}
    """
    state["report"] = report

    filename = f"{ticker}_investment_advice.md"

    artifact = types.Part(
        inline_data=types.Blob(
            mime_type="text/markdown",
            data=report.encode("utf-8"),
        )
    )

    await tool_context.save_artifact(filename, artifact)

    return {
        "success": True,
    }
```

**Detailed Analysis of Changes from 10.4:**

1. **New import added**:
   ```python
   from google.genai import types
   ```
   Imports the `types` module needed for artifact creation.

2. **Function changed to async**:
   ```python
   async def save_advice_report(tool_context: ToolContext, summary: str, ticker: str):
   ```
   - Changed from `def` to `async def`. This is because `save_artifact()` is an asynchronous method requiring `await`.
   - A `ticker: str` parameter was added. The LLM passes the ticker symbol of the company being analyzed, which is used for filename generation.

3. **Filename generation**:
   ```python
   filename = f"{ticker}_investment_advice.md"
   ```
   Dynamically generates a meaningful filename using the ticker symbol. For example: `AAPL_investment_advice.md`

4. **Artifact object creation**:
   ```python
   artifact = types.Part(
       inline_data=types.Blob(
           mime_type="text/markdown",
           data=report.encode("utf-8"),
       )
   )
   ```
   - `types.Blob`: Holds the MIME type (`text/markdown`) and actual data (the report string encoded in UTF-8).
   - `types.Part`: Wraps the `Blob` as `inline_data`. This format is the standard multimodal format for the Google Generative AI API.
   - `report.encode("utf-8")`: Converts the string to bytes. The `Blob`'s `data` requires byte data.

5. **Artifact saving**:
   ```python
   await tool_context.save_artifact(filename, artifact)
   ```
   - Calls `tool_context.save_artifact()` to save the artifact to the ADK system.
   - Uses the `await` keyword to wait for the asynchronous save.
   - In the ADK Web UI, this artifact is automatically displayed, allowing the user to download the file.

#### MIME Type Usage

Besides `text/markdown`, various MIME types can be used:

| MIME Type | Use Case |
|-----------|----------|
| `text/markdown` | Markdown documents |
| `text/plain` | Plain text |
| `text/csv` | CSV data |
| `application/json` | JSON data |
| `image/png` | PNG images |
| `application/pdf` | PDF documents |

### Practice Points

1. Change the MIME type to `text/csv` and generate a CSV-format artifact.
2. Try saving multiple artifacts in a single tool call (e.g., a summary report + detailed data).
3. Check how artifacts are displayed in the ADK Web UI and try downloading them.
4. Think about the problems that could arise from using a fixed filename instead of the `ticker` parameter.

---

## Chapter Key Takeaways

### 1. ADK Project Structure

- ADK projects follow the Python package structure, and `__init__.py` must import the `agent` module.
- The `agent.py` file must contain a `root_agent` variable for the ADK Web UI to recognize the agent.
- Various LLM providers' models can be used through `LiteLlm`.

### 2. Tools and Sub-agents

- **Tools** are defined as regular Python functions, and type hints and docstrings are essential.
- **Sub-agents** can be connected via the `sub_agents` parameter or `AgentTool`.
- `sub_agents` transfers conversation control, while `AgentTool` calls and receives results like a tool.

### 3. Agent Architecture

- Using the `AgentTool` pattern allows the root agent to retain control while leveraging sub-agents.
- Specifying available tools and when to use them in the prompt improves the agent's tool utilization.
- Scope Limitation controls the agent to operate only within its intended domain.

### 4. State Management

- Setting `output_key` causes sub-agent results to be automatically saved to the shared state.
- `ToolContext` allows tool functions to access state (read/write).
- The `ToolContext` parameter is automatically injected by ADK and is not exposed to the LLM.

### 5. Artifacts

- `types.Part` and `types.Blob` are used to create artifact objects.
- `tool_context.save_artifact()` is an asynchronous method, so `async/await` must be used.
- Artifacts enable agents to generate files (reports, data, etc.) and deliver them to users.

---

## Practice Exercises

### Exercise 1: Extending the Basic Agent (Difficulty: Low)

Add a new tool `get_stock_recommendations(ticker: str)` to the current `DataAnalyst` sub-agent. This tool should use `yfinance`'s `recommendations` attribute to return analyst recommendation data.

**Hint**:
```python
def get_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "recommendations": stock.recommendations.to_json(),
    }
```

### Exercise 2: Adding a New Sub-agent (Difficulty: Medium)

Create a new sub-agent called `TechnicalAnalyst`. This agent should have the following tools:

- `get_moving_averages(ticker: str, period: str)`: Calculate moving average data
- `get_volume_analysis(ticker: str)`: Volume analysis data

Connect the created sub-agent to the root agent using `AgentTool`, and add the tool description to the prompt as well.

### Exercise 3: Diversifying Artifacts (Difficulty: Medium)

Extend the `save_advice_report` function to generate an additional JSON-format summary data artifact alongside the markdown report. Both artifacts must be saved in a single function call.

**Hint**: Simply call `save_artifact()` twice.

```python
# JSON artifact
import json
summary_data = {
    "ticker": ticker,
    "recommendation": "BUY",
    "confidence": 0.85,
}
json_artifact = types.Part(
    inline_data=types.Blob(
        mime_type="application/json",
        data=json.dumps(summary_data).encode("utf-8"),
    )
)
await tool_context.save_artifact(f"{ticker}_summary.json", json_artifact)
```

### Exercise 4: State-based Conversation Flow Control (Difficulty: High)

Build a system that tracks analysis progress using state in the root agent's `instruction` prompt. Record a completion flag in the state each time a sub-agent runs, and only call `save_advice_report` when all analyses are complete.

**Hint**: Create a new tool function `check_analysis_status(tool_context: ToolContext)` that checks the currently completed analysis stages.

### Exercise 5: Multi-company Comparative Analysis (Difficulty: High)

Add the ability to analyze and compare multiple companies (e.g., AAPL, GOOGL, MSFT) simultaneously. Store each company's analysis results separately by company in the state, and generate a final comparison report as an artifact.

**Considerations**:
- Since `output_key` only supports a single key, you may need to manage state directly within tool functions.
- Organizing each company's key metrics in a table format makes the comparison report more effective.

---

## References

- [Google ADK Official Documentation](https://google.github.io/adk-docs/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [yfinance Library](https://github.com/ranaroussi/yfinance)
- [Firecrawl Documentation](https://docs.firecrawl.dev/)
