# Chương 10: Google ADK Cơ bản - Xây dựng Agent Cố vấn Tài chính

---

## Tổng quan chương

Trong chương này, chúng ta học quy trình xây dựng agent AI cố vấn tài chính thực tế từ đầu đến cuối bằng **Google ADK (Agent Development Kit)**. Google ADK là framework phát triển agent của Google, cung cấp một cách có hệ thống các tính năng cốt lõi cần thiết cho phát triển agent, bao gồm tạo agent, kết nối công cụ, tổ chức sub-agent, quản lý trạng thái và lưu trữ artifact.

Qua chương này, bạn sẽ học:

- Cách thiết lập dự án Google ADK và chạy ADK Web UI
- Cách kết nối Tools và Sub-agents với agent
- Thiết kế kiến trúc phân cấp giữa root agent và sub-agents
- Quản lý luồng dữ liệu thông qua chia sẻ State giữa các agent
- Tạo và lưu trữ tệp bằng Artifacts

### Cấu trúc dự án

Cấu trúc thư mục dự án hoàn chỉnh cuối cùng như sau:

```
financial-analyst/
├── .python-version
├── pyproject.toml
├── tools.py
├── uv.lock
└── financial_advisor/
    ├── __init__.py
    ├── agent.py                    # Định nghĩa root agent
    ├── prompt.py                   # Prompt root agent
    └── sub_agents/
        ├── __init__.py
        ├── data_analyst.py         # Sub-agent phân tích dữ liệu
        ├── financial_analyst.py    # Sub-agent phân tích tài chính
        └── news_analyst.py         # Sub-agent phân tích tin tức
```

---

## 10.1 ADK Web - Thiết lập dự án ban đầu

### Chủ đề và mục tiêu

Thiết lập dự án Google ADK từ đầu và xây dựng môi trường có thể kiểm thử agent thông qua ADK Web UI. Trong phần này, học các quy tắc cấu trúc dự án ADK và phương pháp tạo agent cơ bản.

### Khái niệm chính

#### Google ADK là gì?

Google ADK (Agent Development Kit) là framework chính thức của Google để xây dựng agent AI. ADK có các đặc điểm sau:

- **Định nghĩa agent chuẩn hóa**: Định nghĩa agent theo cách khai báo thông qua lớp `Agent`.
- **Cung cấp ADK Web UI**: Kiểm thử agent ngay lập tức trong giao diện web mà không cần phát triển frontend riêng.
- **Hỗ trợ nhiều LLM**: Không chỉ mô hình Gemini của Google mà còn các mô hình khác như OpenAI thông qua `LiteLlm`.

#### Quy tắc cấu trúc dự án ADK

ADK yêu cầu cấu trúc dự án cụ thể. Để ADK Web UI tự động nhận diện agent, phải tuân theo các quy tắc sau:

1. **Thư mục package**: Mã agent phải nằm trong Python package (thư mục + `__init__.py`).
2. **Import module agent trong `__init__.py`**: `__init__.py` của package phải import module `agent`.
3. **Biến `root_agent`**: Biến có tên `root_agent` phải tồn tại trong tệp `agent.py`. ADK Web UI sử dụng biến này làm điểm vào.

#### Tích hợp LiteLlm

Google ADK mặc định sử dụng mô hình Gemini của Google, nhưng thông qua wrapper `LiteLlm`, có thể sử dụng mô hình từ nhiều nhà cung cấp LLM như OpenAI, Anthropic. Điều này cho phép bạn sử dụng các mô hình đã quen thuộc trong framework ADK.

### Phân tích mã

#### Thiết lập phụ thuộc dự án (`pyproject.toml`)

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

Các phụ thuộc chính:

| Gói | Vai trò |
|--------|------|
| `google-adk` | Thư viện lõi Google Agent Development Kit |
| `google-genai` | Client Google Generative AI |
| `litellm` | Thư viện wrapper tích hợp các API LLM đa dạng |
| `yfinance` | Thư viện lấy dữ liệu cổ phiếu từ Yahoo Finance |
| `firecrawl-py` | Client API tìm kiếm và thu thập web |
| `python-dotenv` | Tải biến môi trường từ tệp `.env` |
| `watchdog` | Phát hiện thay đổi tệp (để tự động tải lại trong phát triển) |

#### Khởi tạo Package (`__init__.py`)

```python
from . import agent
```

Dòng duy nhất này rất quan trọng. ADK Web UI thực thi `__init__.py` trước khi tải package, và bằng cách import module `agent` ở đây, nó có thể truy cập biến `root_agent`.

#### Định nghĩa Agent cơ bản (`agent.py`)

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

**Phân tích mã:**

- `Agent`: Lớp agent cơ bản do ADK cung cấp. Định nghĩa khai báo tên, instruction và model của agent.
- `LiteLlm("openai/gpt-4o")`: Sử dụng LiteLlm để chỉ định model GPT-4o của OpenAI. Nhà cung cấp và tên model được chỉ định cùng nhau ở dạng `"openai/gpt-4o"`.
- `root_agent = weather_agent`: Gán agent cho biến `root_agent` để ADK Web UI nhận diện. Tên biến này **bắt buộc phải là `root_agent`**.

#### Công cụ tìm kiếm Web (`tools.py`)

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

**Phân tích mã:**

Công cụ này sử dụng Firecrawl API để thực hiện tìm kiếm web. Các thao tác cốt lõi:

1. **Tải biến môi trường**: Tải `FIRECRAWL_API_KEY` từ tệp `.env` bằng `dotenv.load_dotenv()`.
2. **Thực thi tìm kiếm**: Tìm kiếm truy vấn bằng `app.search()` và lấy tối đa 5 kết quả ở định dạng markdown.
3. **Làm sạch kết quả**: Sử dụng biểu thức chính quy để loại bỏ backslash, xuống dòng, URL và liên kết markdown không cần thiết.
4. **Trả về có cấu trúc**: Tổ chức mỗi kết quả thành dictionary với các khóa `title`, `url`, `markdown` và trả về dạng danh sách.

> **Lưu ý**: **Docstring** của hàm công cụ rất quan trọng. ADK (và hầu hết framework agent) truyền docstring cho LLM làm hướng dẫn sử dụng công cụ. Do đó, Args, Returns, v.v. phải được viết rõ ràng.

### Điểm thực hành

1. Khởi tạo dự án bằng `uv` và cài đặt phụ thuộc:
   ```bash
   cd financial-analyst
   uv sync
   ```
2. Chạy ADK Web UI và xác minh agent cơ bản hoạt động:
   ```bash
   adk web
   ```
3. Đổi tên biến `root_agent` thành tên khác và xem ADK Web UI phản ứng thế nào (sẽ xảy ra lỗi).
4. Thiết lập các API key cần thiết trong tệp `.env`.

---

## 10.2 Tools and Subagents - Công cụ và Sub-agents

### Chủ đề và mục tiêu

Học cách thêm **Công cụ (Tools)** và **Sub-agents** vào agent. Công cụ cho phép agent gọi các hàm bên ngoài, và sub-agents cho phép ủy thác các tác vụ cụ thể cho agent khác.

### Khái niệm chính

#### Công cụ (Tools)

Trong ADK, công cụ được định nghĩa là **hàm Python thông thường**. Agent sử dụng tính năng Function Calling của LLM để gọi các công cụ này vào thời điểm thích hợp. Hàm được dùng làm công cụ phải đáp ứng các điều kiện sau:

- **Type hints**: Tham số phải có type hints để LLM truyền đúng đối số.
- **Docstring**: Docstring của hàm cho LLM biết mục đích và cách sử dụng công cụ.
- **Giá trị trả về**: Phải trả về giá trị tuần tự hóa được như chuỗi hoặc dictionary.

#### Sub-agents

Sub-agents là các agent độc lập được đặt dưới agent chính. Khi agent chính nhận một loại câu hỏi nhất định, nó **chuyển** hội thoại cho sub-agent có thể xử lý. Thuộc tính `description` của sub-agent quan trọng vì root agent quyết định chuyển công việc cho sub-agent nào dựa trên mô tả này.

### Phân tích mã

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

**Phân tích mã:**

1. **Định nghĩa hàm công cụ**:
   - `get_weather(city: str)`: Công cụ nhận tên thành phố và trả về thông tin thời tiết. Type hint `city: str` cho LLM biết phải truyền đối số chuỗi.
   - `convert_units(degrees: int)`: Công cụ chuyển đổi đơn vị nhiệt độ.

2. **Định nghĩa sub-agent**:
   - Thuộc tính `description` là chìa khóa. Root agent (WeatherAgent) đọc mô tả này và quyết định chuyển hội thoại cho GeoAgent khi câu hỏi liên quan đến địa lý.

3. **Kết nối công cụ và sub-agents**:
   - Hàm Python được truyền trực tiếp vào danh sách `tools`. ADK tự động phân tích chữ ký hàm và docstring để chuyển đổi thành schema công cụ mà LLM hiểu được.
   - Instance sub-agent được truyền vào danh sách `sub_agents`.

#### So sánh Tools vs Sub-agents

| Thuộc tính | Công cụ (Tool) | Sub-agent |
|------|------------|------------------------|
| Định nghĩa | Hàm Python | Instance Agent |
| Người thực thi | Agent hiện tại gọi trực tiếp | Hội thoại chuyển cho sub-agent |
| Độ phức tạp | Phù hợp tác vụ đơn giản | Phù hợp tác vụ đa bước phức tạp |
| Sử dụng LLM | Không sử dụng (thực thi mã thuần túy) | Suy luận bằng LLM riêng |
| Phù hợp cho | Gọi API, truy xuất dữ liệu | Phân tích chuyên biệt, hội thoại độc lập |

### Điểm thực hành

1. Xóa type hints từ hàm công cụ và chạy. Quan sát trường hợp LLM không truyền đúng đối số.
2. Xóa `description` của `geo_agent` và chạy. Kiểm tra ảnh hưởng đến khả năng chuyển công việc của root agent.
3. Thêm hàm công cụ mới (ví dụ: `get_humidity(city: str)`).

---

## 10.3 Agent Architecture - Thiết kế kiến trúc Agent

### Chủ đề và mục tiêu

Thiết kế và triển khai kiến trúc hoàn chỉnh của agent cố vấn tài chính thực tế. Xây dựng cấu trúc trong đó root agent sử dụng nhiều sub-agent chuyên biệt làm công cụ, và học mẫu agent-tool bằng `AgentTool` cùng cách viết system prompt chi tiết.

### Khái niệm chính

#### AgentTool - Sử dụng Agent như Công cụ

Cách tiếp cận `sub_agents` học ở 10.2 chuyển hội thoại cho sub-agent. Ngược lại, `AgentTool` sử dụng sub-agent **như một công cụ**. Sự khác biệt chính:

| Cách tiếp cận | `sub_agents=[]` | `AgentTool(agent=)` |
|------|-----------------|---------------------|
| Hành vi | Quyền kiểm soát hội thoại chuyển cho sub-agent | Root agent duy trì kiểm soát trong khi gọi sub-agent |
| Ví von | "Hướng dẫn khách hàng này sang bộ phận khác" | "Gọi điện cho bộ phận khác để lấy thông tin" |
| Kết quả | Sub-agent trực tiếp trò chuyện với người dùng | Kết quả sub-agent được trả về cho root agent |
| Phù hợp cho | Tác vụ hoàn toàn khác lĩnh vực | Khi cần tổng hợp thông tin để đưa ra phán đoán cuối |

Cố vấn tài chính cần **tổng hợp** nhiều kết quả phân tích, nên mẫu `AgentTool` phù hợp hơn. Root agent có thể thu thập kết quả phân tích dữ liệu, phân tích tài chính và phân tích tin tức, sau đó đưa ra lời khuyên đầu tư cuối cùng.

#### Kiến trúc Agent phân cấp

Kiến trúc được thiết kế trong dự án này:

```
FinancialAdvisor (Root Agent)
├── DataAnalyst (Sub-agent/Công cụ)
│   ├── get_company_info()
│   ├── get_stock_price()
│   └── get_financial_metrics()
├── FinancialAnalyst (Sub-agent/Công cụ)
│   ├── get_income_statement()
│   ├── get_balance_sheet()
│   └── get_cash_flow()
├── NewsAnalyst (Sub-agent/Công cụ)
│   └── web_search_tool()
└── save_advice_report() (Công cụ trực tiếp)
```

### Phân tích mã

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

**Phân tích chính:**

- `AgentTool(agent=financial_analyst)`: Bọc sub-agent trong `AgentTool` và đặt vào danh sách `tools`. Điều này cho phép root agent gọi sub-agent như một công cụ.
- `save_advice_report`: Hàm Python thông thường cũng có thể đưa vào danh sách `tools` cùng với `AgentTool`. Thời điểm này vẫn là hàm rỗng (`pass`), sẽ được triển khai ở các phần sau.
- Lưu ý sub-agents được truyền vào `tools` bọc trong `AgentTool`, không phải `sub_agents`.

#### Prompt Root Agent (`prompt.py`)

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

**Phân tích thiết kế Prompt:**

Prompt này là ví dụ tốt về prompt engineering cho agent:

1. **Định nghĩa vai trò**: Gán vai trò rõ ràng "Professional Financial Advisor."
2. **Giới hạn phạm vi**: Hạn chế rõ ràng agent không phản hồi câu hỏi ngoài tài chính. Ngăn agent đi lệch khỏi lĩnh vực dự định.
3. **Định nghĩa quy trình**: Chỉ định các bước bắt buộc trước khi đưa ra khuyến nghị BUY/SELL/HOLD.
4. **Hướng dẫn công cụ**: Chỉ định công cụ có sẵn và mục đích trong prompt, cho phép LLM chọn công cụ phù hợp.
5. **Phương pháp phân tích**: Cung cấp quy trình phân tích có hệ thống.

#### Sub-agent phân tích dữ liệu (`sub_agents/data_analyst.py`)

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

**Phân tích chính:**

- **`LlmAgent` vs `Agent`**: Ở đây sử dụng `LlmAgent`. `LlmAgent` là bí danh của `Agent`, chức năng giống hệt nhau. Nó chỉ ra rõ ràng đây là agent hoạt động dựa trên LLM.
- **Sử dụng yfinance**: Mỗi hàm công cụ sử dụng thư viện `yfinance` để lấy dữ liệu cổ phiếu thời gian thực từ Yahoo Finance.
- **Giá trị trả về có cấu trúc**: Tất cả công cụ trả về dictionary nhất quán chứa khóa `ticker` và `success`. Tính nhất quán này giúp LLM phân tích kết quả.
- **Thuộc tính `description`**: Khi được sử dụng như `AgentTool`, root agent dùng thuộc tính này để hiểu mục đích của sub-agent.

#### Sub-agent phân tích tài chính (`sub_agents/financial_analyst.py`)

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

#### Sub-agent phân tích tin tức (`sub_agents/news_analyst.py`)

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

**Phân tích chính:**

- Sub-agent phân tích tin tức sử dụng `web_search_tool` từ `tools.py` tạo ở 10.1.
- Import trực tiếp từ `tools.py` gốc của dự án bằng `from tools import web_search_tool`.
- Mặc dù chỉ có một công cụ, LLM tạo truy vấn tìm kiếm và phân tích kết quả.

### Điểm thực hành

1. Thay thế `AgentTool` bằng `sub_agents` và so sánh sự khác biệt hành vi.
2. Thêm sub-agent mới (ví dụ: `TechnicalAnalyst` để phân tích kỹ thuật).
3. Xóa "STRICT SCOPE LIMITATIONS" khỏi prompt và so sánh phản hồi với câu hỏi ngoài tài chính.

---

## 10.4 Agent State - Quản lý trạng thái Agent

### Chủ đề và mục tiêu

Học cơ chế chia sẻ **State** giữa các agent. Sử dụng `output_key` để tự động lưu đầu ra sub-agent vào state, và học cách truy cập state từ hàm công cụ thông qua `ToolContext`.

### Khái niệm chính

#### Hệ thống State của ADK

Trong ADK, State là **kho lưu trữ key-value** để lưu trữ và chia sẻ dữ liệu trong phiên agent. Đặc điểm của State:

- **Phạm vi phiên**: Tất cả agent trong cùng phiên chia sẻ state.
- **Dạng dictionary**: Truy cập như dictionary Python bằng `state["key"]`.
- **Khả năng tự động lưu**: Sử dụng `output_key` tự động lưu đầu ra cuối cùng của agent vào state.

#### output_key

`output_key` là thuộc tính tự động lưu phản hồi cuối cùng của agent vào state:

```python
data_analyst = LlmAgent(
    ...
    output_key="data_analyst_result",
)
```

Với cài đặt này, khi agent `DataAnalyst` hoàn thành thực thi, phản hồi cuối cùng được tự động lưu vào `state["data_analyst_result"]`.

#### ToolContext

`ToolContext` là đối tượng cho phép hàm công cụ truy cập state, artifact và ngữ cảnh thực thi khác của agent. Khi `tool_context: ToolContext` được thêm làm tham số của hàm công cụ, ADK tự động tiêm ngữ cảnh hiện tại.

> **Quan trọng**: Tham số `ToolContext` không được hiển thị cho LLM. LLM chỉ thấy các tham số thông thường như `summary`, trong khi `tool_context` được framework ADK tiêm nội bộ.

### Phân tích mã

#### Thêm output_key cho Sub-agents

```python
# sub_agents/data_analyst.py
data_analyst = LlmAgent(
    name="DataAnalyst",
    ...
    tools=[get_company_info, get_stock_price, get_financial_metrics],
    output_key="data_analyst_result",   # Thêm mới
)
```

```python
# sub_agents/financial_analyst.py
financial_analyst = Agent(
    name="FinancialAnalyst",
    ...
    tools=[get_income_statement, get_balance_sheet, get_cash_flow],
    output_key="financial_analyst_result",   # Thêm mới
)
```

```python
# sub_agents/news_analyst.py
news_analyst = Agent(
    name="NewsAnalyst",
    ...
    output_key="news_analyst_result",   # Thêm mới
    tools=[web_search_tool],
)
```

#### Hàm công cụ sử dụng State (`agent.py`)

```python
from google.adk.tools import ToolContext

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

#### Tóm tắt luồng dữ liệu

```
Người dùng: "Phân tích AAPL"
    |
    v
FinancialAdvisor (Root)
    |
    ├── AgentTool(DataAnalyst) được gọi
    │   └── Kết quả → tự động lưu vào state["data_analyst_result"]
    |
    ├── AgentTool(FinancialAnalyst) được gọi
    │   └── Kết quả → tự động lưu vào state["financial_analyst_result"]
    |
    ├── AgentTool(NewsAnalyst) được gọi
    │   └── Kết quả → tự động lưu vào state["news_analyst_result"]
    |
    └── save_advice_report(summary=...) được gọi
        ├── Đọc tất cả kết quả sub-agent từ state
        ├── Tạo báo cáo tổng hợp
        └── Lưu báo cáo vào state["report"]
```

### Điểm thực hành

1. Xóa `output_key` và xác nhận `state.get()` trả về `None` trong `save_advice_report`.
2. Khám phá cách kiểm tra dữ liệu state đã lưu trong ADK Web UI.
3. Lưu dữ liệu tùy chỉnh (ví dụ: thời gian bắt đầu phân tích) vào `tool_context.state` và sử dụng.

---

## 10.5 Artifacts - Tạo tệp bằng Artifacts

### Chủ đề và mục tiêu

Học cách lưu dữ liệu do agent tạo ra dưới dạng tệp bằng hệ thống **Artifacts** của ADK. Artifacts là cơ chế quản lý đầu ra (báo cáo, hình ảnh, tệp dữ liệu, v.v.) do agent tạo ra.

### Khái niệm chính

#### Artifacts là gì?

Artifacts là **đầu ra dạng tệp** được tạo trong quá trình thực thi agent. Khác với phản hồi văn bản, artifacts được lưu dưới dạng tệp độc lập mà người dùng có thể tải về hoặc sử dụng riêng.

Trong ADK Web UI, artifacts được tự động hiển thị để người dùng xem và tải về ngay lập tức.

#### google.genai.types.Part và Blob

Artifacts được biểu diễn bằng đối tượng `types.Part` và `types.Blob` từ thư viện `genai` của Google:

- **`types.Blob`**: Container chứa dữ liệu nhị phân và MIME type.
- **`types.Part`**: Đối tượng cấp cao hơn bọc `Blob`, tương thích với định dạng tin nhắn đa phương tiện của Gemini API.

#### Phương thức save_artifact

Phương thức `save_artifact()` của `ToolContext` là phương thức **async**. Do đó, hàm công cụ gọi nó cũng phải được khai báo `async`.

### Phân tích mã

```python
from google.genai import types
from google.adk.tools import ToolContext

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

**Phân tích chi tiết thay đổi từ 10.4:**

1. **Thêm import mới**: `from google.genai import types` - Import module `types` cần cho tạo artifact.

2. **Hàm chuyển sang async**: `async def` thay vì `def`. Vì `save_artifact()` là phương thức async cần `await`. Tham số `ticker: str` được thêm để LLM truyền mã cổ phiếu, dùng cho tạo tên tệp.

3. **Tạo tên tệp**: `filename = f"{ticker}_investment_advice.md"` - Tạo động tên tệp có ý nghĩa bằng mã cổ phiếu. Ví dụ: `AAPL_investment_advice.md`

4. **Tạo đối tượng artifact**:
   - `types.Blob`: Chứa MIME type (`text/markdown`) và dữ liệu thực tế (chuỗi báo cáo mã hóa UTF-8).
   - `types.Part`: Bọc `Blob` dưới dạng `inline_data`. Định dạng chuẩn đa phương tiện của Google Generative AI API.
   - `report.encode("utf-8")`: Chuyển đổi chuỗi thành bytes. `data` của `Blob` yêu cầu dữ liệu byte.

5. **Lưu trữ artifact**: `await tool_context.save_artifact(filename, artifact)` - Lưu artifact vào hệ thống ADK. Trong ADK Web UI, artifact được tự động hiển thị để người dùng tải về.

#### Sử dụng MIME Type

Có thể sử dụng nhiều MIME type ngoài `text/markdown`:

| MIME Type | Mục đích |
|-----------|------|
| `text/markdown` | Tài liệu Markdown |
| `text/plain` | Văn bản thuần |
| `text/csv` | Dữ liệu CSV |
| `application/json` | Dữ liệu JSON |
| `image/png` | Hình ảnh PNG |
| `application/pdf` | Tài liệu PDF |

### Điểm thực hành

1. Thay đổi MIME type thành `text/csv` và tạo artifact định dạng CSV.
2. Lưu nhiều artifact trong một lần gọi công cụ (ví dụ: báo cáo tóm tắt + dữ liệu chi tiết).
3. Kiểm tra cách artifact được hiển thị trong ADK Web UI và tải về.
4. Cân nhắc vấn đề khi sử dụng tên tệp cố định mà không có tham số `ticker`.

---

## Tóm tắt trọng điểm chương

### 1. Cấu trúc dự án ADK

- Dự án ADK tuân theo cấu trúc Python package và phải import module `agent` trong `__init__.py`.
- `agent.py` phải chứa biến `root_agent` để ADK Web UI nhận diện agent.
- Thông qua `LiteLlm`, có thể sử dụng model từ nhiều nhà cung cấp LLM bao gồm OpenAI.

### 2. Công cụ (Tools) và Sub-agents

- **Công cụ** được định nghĩa là hàm Python thông thường, type hints và docstring là bắt buộc.
- **Sub-agents** có thể kết nối qua tham số `sub_agents` hoặc `AgentTool`.
- `sub_agents` chuyển quyền kiểm soát hội thoại, trong khi `AgentTool` gọi và nhận kết quả như công cụ.

### 3. Kiến trúc Agent

- Sử dụng mẫu `AgentTool`, root agent có thể sử dụng sub-agents trong khi duy trì quyền kiểm soát.
- Chỉ định danh sách công cụ có sẵn và thời điểm sử dụng trong prompt cải thiện mức độ sử dụng công cụ.
- Giới hạn phạm vi (Scope Limitation) kiểm soát agent chỉ hoạt động trong lĩnh vực dự định.

### 4. Quản lý State

- Đặt `output_key` tự động lưu kết quả sub-agent vào state chia sẻ.
- `ToolContext` cho phép hàm công cụ truy cập (đọc/ghi) state.
- Tham số `ToolContext` được ADK tự động tiêm và không hiển thị cho LLM.

### 5. Artifacts

- Sử dụng `types.Part` và `types.Blob` để tạo đối tượng artifact.
- `tool_context.save_artifact()` là phương thức async cần `async/await`.
- Thông qua artifacts, agent có thể tạo tệp (báo cáo, dữ liệu, v.v.) và gửi cho người dùng.

---

## Bài tập thực hành

### Bài tập 1: Mở rộng Agent cơ bản (Độ khó: Dễ)

Thêm công cụ mới `get_stock_recommendations(ticker: str)` vào sub-agent `DataAnalyst` hiện tại. Công cụ này sử dụng thuộc tính `recommendations` của `yfinance` để trả về dữ liệu khuyến nghị của nhà phân tích.

**Gợi ý**:
```python
def get_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker)
    return {
        "ticker": ticker,
        "success": True,
        "recommendations": stock.recommendations.to_json(),
    }
```

### Bài tập 2: Thêm Sub-agent mới (Độ khó: Trung bình)

Tạo sub-agent mới tên `TechnicalAnalyst`. Agent này cần có các công cụ:

- `get_moving_averages(ticker: str, period: str)`: Tính dữ liệu đường trung bình động
- `get_volume_analysis(ticker: str)`: Dữ liệu phân tích khối lượng

Kết nối sub-agent đã tạo với root agent dưới dạng `AgentTool` và thêm mô tả công cụ vào prompt.

### Bài tập 3: Đa dạng hóa Artifacts (Độ khó: Trung bình)

Mở rộng hàm `save_advice_report` để tạo thêm artifact dữ liệu tóm tắt định dạng JSON bên cạnh báo cáo markdown. Cả hai artifact phải được lưu trong một lần gọi hàm.

**Gợi ý**: Gọi `save_artifact()` hai lần.

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

### Bài tập 4: Kiểm soát luồng hội thoại dựa trên State (Độ khó: Cao)

Xây dựng hệ thống theo dõi tiến trình phân tích sử dụng state trong prompt `instruction` của root agent. Ghi cờ hoàn thành vào state mỗi khi sub-agent chạy, và chỉ gọi `save_advice_report` khi tất cả phân tích hoàn tất.

**Gợi ý**: Tạo hàm công cụ mới `check_analysis_status(tool_context: ToolContext)` để kiểm tra các giai đoạn phân tích đã hoàn thành.

### Bài tập 5: Phân tích so sánh nhiều công ty (Độ khó: Cao)

Thêm chức năng phân tích đồng thời nhiều công ty (ví dụ: AAPL, GOOGL, MSFT) và so sánh. Lưu kết quả phân tích phân loại theo công ty trong state và tạo báo cáo so sánh cuối cùng dưới dạng artifact.

**Cân nhắc**:
- `output_key` chỉ hỗ trợ một khóa duy nhất, nên có thể cần quản lý state trực tiếp trong hàm công cụ.
- Tổ chức chỉ số chính của mỗi công ty dạng bảng trong báo cáo so sánh sẽ hiệu quả.

---

## Tài liệu tham khảo

- [Tài liệu chính thức Google ADK](https://google.github.io/adk-docs/)
- [Tài liệu LiteLLM](https://docs.litellm.ai/)
- [Thư viện yfinance](https://github.com/ranaroussi/yfinance)
- [Tài liệu Firecrawl](https://docs.firecrawl.dev/)
