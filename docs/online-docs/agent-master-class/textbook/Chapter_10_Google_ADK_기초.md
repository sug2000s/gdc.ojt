# Chapter 10: Google ADK 기초 - 금융 어드바이저 에이전트 구축

---

## 챕터 개요

이번 챕터에서는 **Google ADK(Agent Development Kit)**를 사용하여 실제 금융 어드바이저 AI 에이전트를 처음부터 끝까지 구축하는 과정을 학습한다. Google ADK는 Google이 제공하는 에이전트 개발 프레임워크로, 에이전트의 생성, 도구 연결, 서브 에이전트 구성, 상태 관리, 아티팩트 저장 등 에이전트 개발에 필요한 핵심 기능을 체계적으로 제공한다.

이 챕터를 통해 다음을 배우게 된다:

- Google ADK 프로젝트의 초기 설정과 ADK Web UI 실행 방법
- 에이전트에 도구(Tools)와 서브 에이전트(Sub-agents)를 연결하는 방법
- 루트 에이전트 - 서브 에이전트 간의 계층적 아키텍처 설계
- 에이전트 간 상태(State) 공유를 통한 데이터 흐름 관리
- 아티팩트(Artifacts)를 활용한 파일 생성 및 저장

### 프로젝트 구조

최종적으로 완성되는 프로젝트의 디렉토리 구조는 다음과 같다:

```
financial-analyst/
├── .python-version
├── pyproject.toml
├── tools.py
├── uv.lock
└── financial_advisor/
    ├── __init__.py
    ├── agent.py                    # 루트 에이전트 정의
    ├── prompt.py                   # 루트 에이전트 프롬프트
    └── sub_agents/
        ├── __init__.py
        ├── data_analyst.py         # 데이터 분석 서브 에이전트
        ├── financial_analyst.py    # 재무 분석 서브 에이전트
        └── news_analyst.py         # 뉴스 분석 서브 에이전트
```

---

## 10.1 ADK Web - 프로젝트 초기 설정

### 주제 및 목표

Google ADK 프로젝트를 처음부터 설정하고, ADK Web UI를 통해 에이전트를 테스트할 수 있는 환경을 구축한다. 이 섹션에서는 ADK의 프로젝트 구조 규칙과 기본 에이전트 생성 방법을 익힌다.

### 핵심 개념 설명

#### Google ADK란?

Google ADK(Agent Development Kit)는 AI 에이전트를 구축하기 위한 Google의 공식 프레임워크이다. ADK는 다음과 같은 특징을 가진다:

- **에이전트 정의의 표준화**: `Agent` 클래스를 통해 에이전트를 선언적으로 정의한다.
- **ADK Web UI 제공**: 별도의 프론트엔드 개발 없이 웹 인터페이스에서 에이전트를 즉시 테스트할 수 있다.
- **다양한 LLM 지원**: Google의 Gemini 모델뿐만 아니라 `LiteLlm`을 통해 OpenAI 등 다른 모델도 사용할 수 있다.

#### ADK 프로젝트 구조 규칙

ADK는 특정한 프로젝트 구조를 요구한다. ADK Web UI가 에이전트를 자동으로 인식하려면 다음 규칙을 따라야 한다:

1. **패키지 디렉토리**: 에이전트 코드는 Python 패키지(디렉토리 + `__init__.py`) 안에 있어야 한다.
2. **`__init__.py`에서 agent 모듈 임포트**: 패키지의 `__init__.py`에서 `agent` 모듈을 임포트해야 한다.
3. **`root_agent` 변수**: `agent.py` 파일에 `root_agent`라는 이름의 변수가 반드시 존재해야 한다. ADK Web UI는 이 변수를 진입점으로 사용한다.

#### LiteLlm 통합

Google ADK는 기본적으로 Google의 Gemini 모델을 사용하지만, `LiteLlm` 래퍼를 통해 OpenAI, Anthropic 등 다양한 LLM 제공자의 모델을 사용할 수 있다. 이를 통해 기존에 사용하던 모델을 ADK 프레임워크 안에서 그대로 활용할 수 있다.

### 코드 분석

#### 프로젝트 의존성 설정 (`pyproject.toml`)

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

주요 의존성을 살펴보면:

| 패키지 | 역할 |
|--------|------|
| `google-adk` | Google Agent Development Kit 핵심 라이브러리 |
| `google-genai` | Google Generative AI 클라이언트 |
| `litellm` | 다양한 LLM API를 통합하는 래퍼 라이브러리 |
| `yfinance` | Yahoo Finance에서 주식 데이터를 가져오는 라이브러리 |
| `firecrawl-py` | 웹 검색 및 스크래핑 API 클라이언트 |
| `python-dotenv` | `.env` 파일에서 환경 변수를 로드 |
| `watchdog` | 파일 변경 감지 (개발 시 자동 리로드용) |

#### 패키지 초기화 (`__init__.py`)

```python
from . import agent
```

이 한 줄이 매우 중요하다. ADK Web UI는 패키지를 로드할 때 `__init__.py`를 먼저 실행하고, 여기서 `agent` 모듈을 임포트함으로써 `root_agent` 변수에 접근할 수 있게 된다.

#### 기본 에이전트 정의 (`agent.py`)

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

**코드 분석:**

- `Agent`: ADK에서 제공하는 에이전트 기본 클래스이다. 에이전트의 이름, 지시문(instruction), 사용할 모델을 선언적으로 정의한다.
- `LiteLlm("openai/gpt-4o")`: LiteLlm을 사용하여 OpenAI의 GPT-4o 모델을 지정한다. `"openai/gpt-4o"` 형식으로 제공자와 모델명을 함께 지정한다.
- `root_agent = weather_agent`: ADK Web UI가 인식할 수 있도록 `root_agent` 변수에 에이전트를 할당한다. 이 변수명은 **반드시 `root_agent`**여야 한다.

#### 웹 검색 도구 (`tools.py`)

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

**코드 분석:**

이 도구는 Firecrawl API를 사용하여 웹 검색을 수행한다. 핵심 동작은 다음과 같다:

1. **환경 변수 로드**: `dotenv.load_dotenv()`로 `.env` 파일에서 `FIRECRAWL_API_KEY`를 로드한다.
2. **검색 실행**: `app.search()`로 쿼리를 검색하고, 최대 5개의 결과를 마크다운 형식으로 가져온다.
3. **결과 정제**: 정규 표현식을 사용하여 불필요한 백슬래시, 줄바꿈, URL, 마크다운 링크를 제거하여 깔끔한 텍스트를 생성한다.
4. **구조화된 반환**: 각 결과를 `title`, `url`, `markdown` 키를 가진 딕셔너리로 정리하여 리스트로 반환한다.

> **주의**: 도구 함수의 **docstring**은 매우 중요하다. ADK(및 대부분의 에이전트 프레임워크)는 docstring을 LLM에게 도구의 사용법으로 전달한다. 따라서 Args, Returns 등을 명확하게 작성해야 한다.

### 실습 포인트

1. `uv`를 사용하여 프로젝트를 초기화하고 의존성을 설치해 본다:
   ```bash
   cd financial-analyst
   uv sync
   ```
2. ADK Web UI를 실행하여 기본 에이전트가 동작하는지 확인한다:
   ```bash
   adk web
   ```
3. `root_agent` 변수명을 다른 이름으로 바꿔보고 ADK Web UI가 어떻게 반응하는지 확인한다 (에러가 발생할 것이다).
4. `.env` 파일에 필요한 API 키들을 설정한다.

---

## 10.2 Tools and Subagents - 도구와 서브 에이전트

### 주제 및 목표

에이전트에 **도구(Tools)**와 **서브 에이전트(Sub-agents)**를 추가하는 방법을 학습한다. 도구는 에이전트가 외부 기능을 호출할 수 있게 해주고, 서브 에이전트는 특정 작업을 다른 에이전트에 위임할 수 있게 해준다.

### 핵심 개념 설명

#### 도구(Tools)

ADK에서 도구는 **일반 Python 함수**로 정의된다. 에이전트는 LLM의 Function Calling 기능을 사용하여 적절한 시점에 이 도구들을 호출한다. 도구로 사용할 함수는 다음 조건을 만족해야 한다:

- **타입 힌트**: 매개변수에 타입 힌트가 있어야 LLM이 올바른 인자를 전달할 수 있다.
- **Docstring**: 함수의 docstring이 LLM에게 도구의 용도와 사용법을 알려준다.
- **반환값**: 문자열이나 딕셔너리 등 직렬화 가능한 값을 반환해야 한다.

#### 서브 에이전트(Sub-agents)

서브 에이전트는 메인 에이전트의 하위에 배치되는 독립적인 에이전트이다. 메인 에이전트가 특정 유형의 질문을 받으면, 해당 질문을 처리할 수 있는 서브 에이전트에게 대화를 **전달(transfer)**한다. 서브 에이전트에는 `description` 속성이 중요한데, 이 설명을 기반으로 루트 에이전트가 어떤 서브 에이전트에게 작업을 전달할지 결정하기 때문이다.

### 코드 분석

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

**코드 분석:**

1. **도구 함수 정의**:
   - `get_weather(city: str)`: 도시명을 받아 날씨 정보를 반환하는 도구이다. `city: str` 타입 힌트가 있어 LLM이 문자열 인자를 전달해야 함을 알 수 있다.
   - `convert_units(degrees: int)`: 온도 단위를 변환하는 도구이다.

2. **서브 에이전트 정의**:
   ```python
   geo_agent = Agent(
       name="GeoAgent",
       instruction="You help with geo questions",
       model=MODEL,
       description="Transfer to this agent when you have a geo related question.",
   )
   ```
   - `description` 속성이 핵심이다. 루트 에이전트(WeatherAgent)는 이 description을 읽고, 사용자의 질문이 지리 관련일 때 GeoAgent에게 대화를 전달하기로 결정한다.

3. **에이전트에 도구와 서브 에이전트 연결**:
   ```python
   weather_agent = Agent(
       ...
       tools=[get_weather, convert_units],
       sub_agents=[geo_agent],
   )
   ```
   - `tools` 리스트에 Python 함수를 직접 전달한다. ADK가 자동으로 함수의 시그니처와 docstring을 분석하여 LLM이 이해할 수 있는 도구 스키마로 변환한다.
   - `sub_agents` 리스트에 서브 에이전트 인스턴스를 전달한다.

#### 도구 vs 서브 에이전트 비교

| 특성 | 도구 (Tool) | 서브 에이전트 (Sub-agent) |
|------|------------|------------------------|
| 정의 방식 | Python 함수 | Agent 인스턴스 |
| 실행 주체 | 현재 에이전트가 직접 호출 | 대화가 서브 에이전트로 전달됨 |
| 복잡도 | 단순 작업에 적합 | 복잡한 멀티스텝 작업에 적합 |
| LLM 사용 | 사용하지 않음 (순수 코드 실행) | 자체 LLM으로 추론 수행 |
| 적합한 경우 | API 호출, 데이터 조회 | 전문 분석, 독립적 대화 |

### 실습 포인트

1. 도구 함수에서 타입 힌트를 제거하고 실행해 본다. LLM이 인자를 올바르게 전달하지 못하는 경우를 관찰한다.
2. `geo_agent`의 `description`을 제거하고 실행해 본다. 루트 에이전트가 서브 에이전트에게 작업을 전달하는 데 어떤 영향이 있는지 확인한다.
3. 새로운 도구 함수(예: `get_humidity(city: str)`)를 추가해 본다.

---

## 10.3 Agent Architecture - 에이전트 아키텍처 설계

### 주제 및 목표

실제 금융 어드바이저 에이전트의 전체 아키텍처를 설계하고 구현한다. 루트 에이전트가 여러 전문 서브 에이전트를 도구로 활용하는 구조를 구축하며, `AgentTool`을 사용한 에이전트-도구 패턴과 상세한 시스템 프롬프트 작성법을 학습한다.

### 핵심 개념 설명

#### AgentTool - 에이전트를 도구로 사용하기

10.2에서 배운 `sub_agents` 방식은 대화 자체를 서브 에이전트에게 전달(transfer)하는 방식이다. 반면 `AgentTool`은 서브 에이전트를 **도구처럼** 사용하는 방식이다. 두 방식의 핵심 차이점은 다음과 같다:

| 방식 | `sub_agents=[]` | `AgentTool(agent=)` |
|------|-----------------|---------------------|
| 동작 | 대화 제어권이 서브 에이전트로 이동 | 루트 에이전트가 제어권을 유지하면서 서브 에이전트를 호출 |
| 비유 | "이 고객을 다른 부서로 안내" | "다른 부서에 전화해서 정보를 가져와" |
| 결과 | 서브 에이전트가 직접 사용자와 대화 | 서브 에이전트의 결과가 루트 에이전트에게 반환됨 |
| 적합한 경우 | 완전히 다른 도메인의 작업 | 정보 수집 후 종합 판단이 필요한 경우 |

금융 어드바이저는 여러 분석 결과를 **종합**해야 하므로, `AgentTool` 패턴이 더 적합하다. 루트 에이전트가 데이터 분석, 재무 분석, 뉴스 분석 결과를 모두 수집한 후 최종 투자 조언을 제공할 수 있기 때문이다.

#### 계층적 에이전트 아키텍처

이 프로젝트에서 설계하는 아키텍처는 다음과 같다:

```
FinancialAdvisor (루트 에이전트)
├── DataAnalyst (서브 에이전트/도구)
│   ├── get_company_info()
│   ├── get_stock_price()
│   └── get_financial_metrics()
├── FinancialAnalyst (서브 에이전트/도구)
│   ├── get_income_statement()
│   ├── get_balance_sheet()
│   └── get_cash_flow()
├── NewsAnalyst (서브 에이전트/도구)
│   └── web_search_tool()
└── save_advice_report() (직접 도구)
```

### 코드 분석

#### 루트 에이전트 (`agent.py`)

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

**핵심 분석:**

- `AgentTool(agent=financial_analyst)`: 서브 에이전트를 `AgentTool`로 감싸서 `tools` 리스트에 넣는다. 이렇게 하면 루트 에이전트가 서브 에이전트를 하나의 도구처럼 호출할 수 있다.
- `save_advice_report`: 일반 Python 함수도 `AgentTool`과 함께 `tools` 리스트에 넣을 수 있다. 이 시점에서는 아직 빈 함수(`pass`)이며, 이후 섹션에서 구현된다.
- 서브 에이전트들은 `sub_agents`가 아닌 `tools`에 `AgentTool`로 감싸서 전달된다는 점에 주목한다.

#### 루트 에이전트 프롬프트 (`prompt.py`)

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

**프롬프트 설계 분석:**

이 프롬프트는 에이전트 프롬프트 엔지니어링의 좋은 예시이다:

1. **역할 정의**: "Professional Financial Advisor"로 명확한 역할을 부여한다.
2. **범위 제한**: 금융 이외의 질문에 대응하지 않도록 명시적으로 제한한다. 이는 에이전트가 의도치 않은 영역으로 벗어나는 것을 방지한다.
3. **프로세스 정의**: BUY/SELL/HOLD 추천 전 반드시 거쳐야 할 단계를 명시한다.
4. **도구 안내**: 사용 가능한 도구들과 각각의 용도를 프롬프트 안에 명시하여, LLM이 적절한 도구를 선택할 수 있게 한다.
5. **분석 방법론**: 체계적인 분석 절차를 제공한다.

#### 데이터 분석 서브 에이전트 (`sub_agents/data_analyst.py`)

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

**핵심 분석:**

- **`LlmAgent` vs `Agent`**: 여기서는 `LlmAgent`를 사용하고 있다. `LlmAgent`는 `Agent`의 별칭으로, 기능적으로 동일하다. LLM을 기반으로 동작하는 에이전트임을 명시적으로 나타낸다.
- **yfinance 활용**: 각 도구 함수는 `yfinance` 라이브러리를 사용하여 Yahoo Finance에서 실시간 주식 데이터를 가져온다.
- **구조화된 반환값**: 모든 도구가 `ticker`, `success` 키를 포함하는 일관된 딕셔너리를 반환한다. 이러한 일관성은 LLM이 결과를 파싱하는 데 도움이 된다.
- **`description` 속성**: `AgentTool`로 사용될 때 루트 에이전트가 이 서브 에이전트의 용도를 이해하는 데 활용된다.

#### 재무 분석 서브 에이전트 (`sub_agents/financial_analyst.py`)

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

#### 뉴스 분석 서브 에이전트 (`sub_agents/news_analyst.py`)

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

**핵심 분석:**

- 뉴스 분석 서브 에이전트는 10.1에서 만든 `tools.py`의 `web_search_tool`을 활용한다.
- `from tools import web_search_tool`로 프로젝트 루트의 `tools.py`에서 직접 임포트한다.
- 하나의 도구만 가지고 있지만, LLM이 검색 쿼리를 생성하고 결과를 분석하는 역할을 한다.

### 실습 포인트

1. `AgentTool`을 `sub_agents`로 바꿔보고 동작 차이를 비교한다.
2. 새로운 서브 에이전트(예: 기술적 분석을 수행하는 `TechnicalAnalyst`)를 추가해 본다.
3. 프롬프트에서 "STRICT SCOPE LIMITATIONS"를 제거하고 금융 외 질문을 했을 때의 반응을 비교한다.

---

## 10.4 Agent State - 에이전트 상태 관리

### 주제 및 목표

에이전트 간 **상태(State)** 공유 메커니즘을 학습한다. `output_key`를 사용하여 서브 에이전트의 출력을 상태에 자동 저장하고, `ToolContext`를 통해 도구 함수에서 상태에 접근하는 방법을 배운다.

### 핵심 개념 설명

#### ADK의 상태(State) 시스템

ADK에서 상태(State)는 에이전트 세션 내에서 데이터를 저장하고 공유하는 **키-값 저장소**이다. 상태의 특징은 다음과 같다:

- **세션 범위**: 같은 세션 내의 모든 에이전트가 상태를 공유한다.
- **딕셔너리 형태**: Python 딕셔너리처럼 `state["key"]`로 접근한다.
- **자동 저장 가능**: `output_key`를 사용하면 에이전트의 최종 출력이 자동으로 상태에 저장된다.

#### output_key

`output_key`는 에이전트의 최종 응답을 자동으로 상태에 저장하는 속성이다:

```python
data_analyst = LlmAgent(
    ...
    output_key="data_analyst_result",
)
```

위와 같이 설정하면, `DataAnalyst` 에이전트가 실행을 완료했을 때 그 최종 응답이 자동으로 `state["data_analyst_result"]`에 저장된다.

#### ToolContext

`ToolContext`는 도구 함수에서 에이전트의 상태, 아티팩트 등 실행 컨텍스트에 접근할 수 있게 해주는 객체이다. 도구 함수의 매개변수에 `tool_context: ToolContext`를 추가하면, ADK가 자동으로 현재 컨텍스트를 주입한다.

> **중요**: `ToolContext` 매개변수는 LLM에게 노출되지 않는다. LLM은 `summary`와 같은 일반 매개변수만 보게 되며, `tool_context`는 ADK 프레임워크가 내부적으로 주입한다.

### 코드 분석

#### 서브 에이전트에 output_key 추가

```python
# sub_agents/data_analyst.py
data_analyst = LlmAgent(
    name="DataAnalyst",
    ...
    tools=[get_company_info, get_stock_price, get_financial_metrics],
    output_key="data_analyst_result",   # 추가됨
)
```

```python
# sub_agents/financial_analyst.py
financial_analyst = Agent(
    name="FinancialAnalyst",
    ...
    tools=[get_income_statement, get_balance_sheet, get_cash_flow],
    output_key="financial_analyst_result",   # 추가됨
)
```

```python
# sub_agents/news_analyst.py
news_analyst = Agent(
    name="NewsAnalyst",
    ...
    output_key="news_analyst_result",   # 추가됨
    tools=[web_search_tool],
)
```

각 서브 에이전트에 `output_key`가 추가되었다. 이로써 각 서브 에이전트가 실행을 마치면 결과가 자동으로 공유 상태에 저장된다.

#### 상태를 활용하는 도구 함수 (`agent.py`)

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

**코드 분석:**

1. **`ToolContext` 매개변수**: `tool_context: ToolContext`를 첫 번째 매개변수로 선언한다. ADK가 자동으로 현재 실행 컨텍스트를 주입한다. LLM은 이 매개변수를 인식하지 못하며, `summary: str`만 인자로 전달한다.

2. **상태에서 데이터 읽기**:
   ```python
   state = tool_context.state
   data_analyst_result = state.get("data_analyst_result")
   ```
   `tool_context.state`로 상태 딕셔너리에 접근하고, 서브 에이전트들이 `output_key`로 저장한 결과를 가져온다.

3. **상태에 데이터 쓰기**:
   ```python
   state["report"] = report
   ```
   생성된 리포트를 상태에 저장한다. 이후 다른 도구나 에이전트에서 `state["report"]`로 접근할 수 있다.

#### 데이터 흐름 요약

```
사용자: "AAPL 분석해줘"
    │
    ▼
FinancialAdvisor (루트)
    │
    ├── AgentTool(DataAnalyst) 호출
    │   └── 결과 → state["data_analyst_result"]에 자동 저장
    │
    ├── AgentTool(FinancialAnalyst) 호출
    │   └── 결과 → state["financial_analyst_result"]에 자동 저장
    │
    ├── AgentTool(NewsAnalyst) 호출
    │   └── 결과 → state["news_analyst_result"]에 자동 저장
    │
    └── save_advice_report(summary=...) 호출
        ├── state에서 모든 서브 에이전트 결과 읽기
        ├── 종합 리포트 생성
        └── state["report"]에 리포트 저장
```

### 실습 포인트

1. `output_key`를 제거하고 `save_advice_report`에서 `state.get()`이 `None`을 반환하는 것을 확인한다.
2. 상태에 저장된 데이터를 ADK Web UI에서 확인하는 방법을 탐색한다.
3. `tool_context.state`에 커스텀 데이터(예: 분석 시작 시간)를 저장하고 활용해 본다.

---

## 10.5 Artifacts - 아티팩트를 활용한 파일 생성

### 주제 및 목표

ADK의 **아티팩트(Artifacts)** 시스템을 활용하여 에이전트가 생성한 데이터를 파일로 저장하는 방법을 학습한다. 아티팩트는 에이전트가 생성한 결과물(리포트, 이미지, 데이터 파일 등)을 관리하는 메커니즘이다.

### 핵심 개념 설명

#### 아티팩트(Artifacts)란?

아티팩트는 에이전트 실행 중에 생성되는 **파일 형태의 출력물**이다. 텍스트 응답과 달리, 아티팩트는 독립적인 파일로 저장되어 사용자가 다운로드하거나 별도로 활용할 수 있다.

ADK Web UI에서는 아티팩트가 자동으로 UI에 표시되어 사용자가 바로 확인하고 다운로드할 수 있다.

#### google.genai.types.Part와 Blob

아티팩트는 Google의 `genai` 라이브러리에서 제공하는 `types.Part`와 `types.Blob` 객체로 표현된다:

- **`types.Blob`**: 바이너리 데이터와 MIME 타입을 담는 컨테이너이다.
- **`types.Part`**: `Blob`을 감싸는 상위 객체로, Gemini API의 멀티모달 메시지 형식과 호환된다.

#### save_artifact 메서드

`ToolContext`의 `save_artifact()` 메서드는 **비동기(async)** 메서드이다. 따라서 이를 호출하는 도구 함수도 `async`로 선언해야 한다.

### 코드 분석

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

**10.4에서 변경된 부분 상세 분석:**

1. **새로운 import 추가**:
   ```python
   from google.genai import types
   ```
   아티팩트 생성에 필요한 `types` 모듈을 임포트한다.

2. **함수를 async로 변경**:
   ```python
   async def save_advice_report(tool_context: ToolContext, summary: str, ticker: str):
   ```
   - `def` → `async def`로 변경되었다. `save_artifact()`가 비동기 메서드이므로 `await`를 사용해야 하기 때문이다.
   - `ticker: str` 매개변수가 추가되었다. LLM이 분석 대상 기업의 티커 심볼을 전달하며, 이를 파일명 생성에 사용한다.

3. **파일명 생성**:
   ```python
   filename = f"{ticker}_investment_advice.md"
   ```
   티커 심볼을 사용하여 의미 있는 파일명을 동적으로 생성한다. 예: `AAPL_investment_advice.md`

4. **아티팩트 객체 생성**:
   ```python
   artifact = types.Part(
       inline_data=types.Blob(
           mime_type="text/markdown",
           data=report.encode("utf-8"),
       )
   )
   ```
   - `types.Blob`: MIME 타입(`text/markdown`)과 실제 데이터(UTF-8로 인코딩된 리포트 문자열)를 담는다.
   - `types.Part`: `Blob`을 `inline_data`로 감싼다. 이 형식은 Google Generative AI API의 표준 멀티모달 형식이다.
   - `report.encode("utf-8")`: 문자열을 바이트로 변환한다. `Blob`의 `data`는 바이트 데이터를 요구한다.

5. **아티팩트 저장**:
   ```python
   await tool_context.save_artifact(filename, artifact)
   ```
   - `tool_context.save_artifact()`를 호출하여 아티팩트를 ADK 시스템에 저장한다.
   - `await` 키워드로 비동기 저장을 기다린다.
   - ADK Web UI에서는 이 아티팩트가 자동으로 표시되어 사용자가 파일을 다운로드할 수 있다.

#### MIME 타입 활용

`text/markdown` 외에도 다양한 MIME 타입을 사용할 수 있다:

| MIME 타입 | 용도 |
|-----------|------|
| `text/markdown` | 마크다운 문서 |
| `text/plain` | 일반 텍스트 |
| `text/csv` | CSV 데이터 |
| `application/json` | JSON 데이터 |
| `image/png` | PNG 이미지 |
| `application/pdf` | PDF 문서 |

### 실습 포인트

1. MIME 타입을 `text/csv`로 변경하고 CSV 형식의 아티팩트를 생성해 본다.
2. 여러 개의 아티팩트를 하나의 도구 호출에서 저장해 본다 (예: 요약 리포트 + 상세 데이터).
3. ADK Web UI에서 아티팩트가 어떻게 표시되는지 확인하고 다운로드해 본다.
4. `ticker` 매개변수 없이 고정 파일명을 사용했을 때의 문제점을 생각해 본다.

---

## 챕터 핵심 정리

### 1. ADK 프로젝트 구조

- ADK 프로젝트는 Python 패키지 구조를 따르며, `__init__.py`에서 `agent` 모듈을 임포트해야 한다.
- `agent.py`에는 반드시 `root_agent` 변수가 존재해야 ADK Web UI가 에이전트를 인식한다.
- `LiteLlm`을 통해 OpenAI 등 다양한 LLM 제공자의 모델을 사용할 수 있다.

### 2. 도구(Tools)와 서브 에이전트(Sub-agents)

- **도구**는 일반 Python 함수로 정의하며, 타입 힌트와 docstring이 필수이다.
- **서브 에이전트**는 `sub_agents` 매개변수 또는 `AgentTool`로 연결할 수 있다.
- `sub_agents`는 대화 제어권을 이전하고, `AgentTool`은 도구처럼 호출 후 결과를 받아온다.

### 3. 에이전트 아키텍처

- `AgentTool` 패턴을 사용하면 루트 에이전트가 제어권을 유지하면서 서브 에이전트를 활용할 수 있다.
- 프롬프트에 사용 가능한 도구 목록과 사용 시점을 명시하면 에이전트의 도구 활용도가 높아진다.
- 범위 제한(Scope Limitation)을 통해 에이전트가 의도한 영역 내에서만 동작하도록 제어한다.

### 4. 상태(State) 관리

- `output_key`를 설정하면 서브 에이전트의 결과가 공유 상태에 자동 저장된다.
- `ToolContext`를 통해 도구 함수에서 상태에 접근(읽기/쓰기)할 수 있다.
- `ToolContext` 매개변수는 ADK가 자동 주입하며, LLM에게는 노출되지 않는다.

### 5. 아티팩트(Artifacts)

- `types.Part`와 `types.Blob`을 사용하여 아티팩트 객체를 생성한다.
- `tool_context.save_artifact()`는 비동기 메서드이므로 `async/await`를 사용해야 한다.
- 아티팩트를 통해 에이전트가 파일(리포트, 데이터 등)을 생성하여 사용자에게 전달할 수 있다.

---

## 실습 과제

### 과제 1: 기본 에이전트 확장 (난이도: 하)

현재 `DataAnalyst` 서브 에이전트에 새로운 도구 `get_stock_recommendations(ticker: str)`를 추가하라. 이 도구는 `yfinance`의 `recommendations` 속성을 활용하여 애널리스트들의 추천 데이터를 반환해야 한다.

**힌트**:
```python
def get_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "recommendations": stock.recommendations.to_json(),
    }
```

### 과제 2: 새로운 서브 에이전트 추가 (난이도: 중)

`TechnicalAnalyst`라는 새로운 서브 에이전트를 생성하라. 이 에이전트는 다음 도구들을 가져야 한다:

- `get_moving_averages(ticker: str, period: str)`: 이동평균선 데이터 계산
- `get_volume_analysis(ticker: str)`: 거래량 분석 데이터

생성한 서브 에이전트를 `AgentTool`로 루트 에이전트에 연결하고, 프롬프트에도 해당 도구 설명을 추가하라.

### 과제 3: 아티팩트 다양화 (난이도: 중)

`save_advice_report` 함수를 확장하여 마크다운 리포트 외에 추가로 JSON 형식의 요약 데이터 아티팩트를 생성하라. 두 개의 아티팩트를 하나의 함수 호출에서 모두 저장해야 한다.

**힌트**: `save_artifact()`를 두 번 호출하면 된다.

```python
# JSON 아티팩트
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

### 과제 4: 상태 기반 대화 흐름 제어 (난이도: 상)

루트 에이전트의 `instruction` 프롬프트에서 상태를 활용하여 분석 진행 상황을 추적하는 시스템을 구축하라. 각 서브 에이전트가 실행될 때마다 상태에 완료 플래그를 기록하고, 모든 분석이 완료되었을 때만 `save_advice_report`를 호출하도록 하라.

**힌트**: 새로운 도구 함수 `check_analysis_status(tool_context: ToolContext)`를 만들어 현재 완료된 분석 단계를 확인할 수 있게 한다.

### 과제 5: 멀티 기업 비교 분석 (난이도: 상)

여러 기업(예: AAPL, GOOGL, MSFT)을 동시에 분석하고 비교하는 기능을 추가하라. 각 기업의 분석 결과를 상태에 기업별로 분류하여 저장하고, 최종 비교 리포트를 아티팩트로 생성하라.

**고려사항**:
- `output_key`는 하나의 키만 지원하므로, 도구 함수 내에서 직접 상태를 관리해야 할 수 있다.
- 비교 리포트에는 각 기업의 핵심 지표를 표 형태로 정리하면 효과적이다.

---

## 참고 자료

- [Google ADK 공식 문서](https://google.github.io/adk-docs/)
- [LiteLLM 문서](https://docs.litellm.ai/)
- [yfinance 라이브러리](https://github.com/ranaroussi/yfinance)
- [Firecrawl 문서](https://docs.firecrawl.dev/)
