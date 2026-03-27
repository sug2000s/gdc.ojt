# Chapter 10: Google ADK Cơ bản - Xây dựng Agent Cố vấn Tài chính

---

## Tổng quan chương

Trong chương này, chúng ta học quy trình xây dựng AI agent cố vấn tài chính thực tế từ đầu đến cuối bằng **Google ADK (Agent Development Kit)**. Google ADK là framework phát triển agent do Google cung cấp, cung cấp một cách có hệ thống các tính năng cốt lõi cần thiết cho phát triển agent như tạo agent, kết nối công cụ, cấu hình sub-agent, quản lý trạng thái, lưu trữ artifact.

Qua chương này, bạn sẽ học được:

- Thiết lập ban đầu dự án Google ADK và cách chạy ADK Web UI
- Cách kết nối công cụ (Tools) và sub-agent với agent
- Thiết kế kiến trúc phân cấp giữa root agent và sub-agent
- Quản lý luồng dữ liệu thông qua chia sẻ trạng thái (State) giữa các agent
- Tạo và lưu trữ file bằng artifact

### Cấu trúc dự án

Cấu trúc thư mục của dự án hoàn thành cuối cùng như sau:

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

## 10.1 ADK Web - Thiết lập ban đầu dự án

### Chủ đề và mục tiêu

Thiết lập dự án Google ADK từ đầu và xây dựng môi trường để kiểm thử agent thông qua ADK Web UI. Trong phần này, nắm vững quy tắc cấu trúc dự án ADK và cách tạo agent cơ bản.

### Giải thích khái niệm cốt lõi

#### Google ADK là gì?

Google ADK (Agent Development Kit) là framework chính thức của Google để xây dựng AI agent. ADK có các đặc điểm sau:

- **Chuẩn hóa định nghĩa agent**: Định nghĩa agent theo cách khai báo thông qua lớp `Agent`.
- **Cung cấp ADK Web UI**: Có thể kiểm thử agent ngay trên giao diện web mà không cần phát triển frontend riêng.
- **Hỗ trợ đa LLM**: Không chỉ mô hình Gemini của Google, mà còn có thể sử dụng các mô hình khác như OpenAI thông qua `LiteLlm`.

#### Quy tắc cấu trúc dự án ADK

ADK yêu cầu cấu trúc dự án cụ thể. Để ADK Web UI tự động nhận diện agent, cần tuân theo các quy tắc sau:

1. **Thư mục package**: Code agent phải nằm trong Python package (thư mục + `__init__.py`).
2. **Import module agent trong `__init__.py`**: Phải import module `agent` trong `__init__.py` của package.
3. **Biến `root_agent`**: File `agent.py` phải có biến tên là `root_agent`. ADK Web UI sử dụng biến này làm điểm khởi đầu.

#### Tích hợp LiteLlm

Google ADK mặc định sử dụng mô hình Gemini của Google, nhưng thông qua wrapper `LiteLlm`, có thể sử dụng mô hình từ các nhà cung cấp LLM đa dạng như OpenAI, Anthropic. Điều này cho phép sử dụng mô hình quen thuộc trong framework ADK.

### Phân tích code

#### Thiết lập dependency dự án (`pyproject.toml`)

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

Xem xét các dependency chính:

| Gói | Vai trò |
|-----|---------|
| `google-adk` | Thư viện cốt lõi Google Agent Development Kit |
| `google-genai` | Client Google Generative AI |
| `litellm` | Thư viện wrapper tích hợp đa API LLM |
| `yfinance` | Thư viện lấy dữ liệu chứng khoán từ Yahoo Finance |
| `firecrawl-py` | Client API tìm kiếm và scraping web |
| `python-dotenv` | Tải biến môi trường từ file `.env` |
| `watchdog` | Phát hiện thay đổi file (dùng cho auto-reload khi phát triển) |

#### Khởi tạo package (`__init__.py`)

```python
from . import agent
```

Dòng code này rất quan trọng. ADK Web UI khi tải package sẽ thực thi `__init__.py` trước, và bằng việc import module `agent` ở đây, nó có thể truy cập biến `root_agent`.

#### Định nghĩa agent cơ bản (`agent.py`)

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

**Phân tích code:**

- `Agent`: Lớp agent cơ bản do ADK cung cấp. Định nghĩa khai báo tên, chỉ dẫn (instruction), mô hình sử dụng của agent.
- `LiteLlm("openai/gpt-4o")`: Chỉ định mô hình GPT-4o của OpenAI bằng LiteLlm. Chỉ định cả nhà cung cấp và tên mô hình theo định dạng `"openai/gpt-4o"`.
- `root_agent = weather_agent`: Gán agent cho biến `root_agent` để ADK Web UI có thể nhận diện. Tên biến **bắt buộc phải là `root_agent`**.

#### Công cụ tìm kiếm web (`tools.py`)

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

**Phân tích code:**

Công cụ này sử dụng Firecrawl API để thực hiện tìm kiếm web. Hoạt động cốt lõi như sau:

1. **Tải biến môi trường**: Tải `FIRECRAWL_API_KEY` từ file `.env` bằng `dotenv.load_dotenv()`.
2. **Thực thi tìm kiếm**: Tìm kiếm query bằng `app.search()` và lấy tối đa 5 kết quả ở định dạng markdown.
3. **Tinh chỉnh kết quả**: Sử dụng regex để loại bỏ backslash thừa, xuống dòng, URL, link markdown để tạo văn bản sạch.
4. **Trả về có cấu trúc**: Tổ chức mỗi kết quả thành dictionary với key `title`, `url`, `markdown` và trả về dưới dạng danh sách.

> **Lưu ý**: **Docstring** của hàm công cụ rất quan trọng. ADK (và hầu hết framework agent) truyền docstring cho LLM làm hướng dẫn sử dụng công cụ. Do đó, cần viết rõ ràng Args, Returns, v.v.

### Điểm thực hành

1. Khởi tạo dự án và cài đặt dependency bằng `uv`:
   ```bash
   cd financial-analyst
   uv sync
   ```
2. Chạy ADK Web UI và xác nhận agent cơ bản hoạt động:
   ```bash
   adk web
   ```
3. Đổi tên biến `root_agent` sang tên khác và xem ADK Web UI phản ứng thế nào (sẽ gặp lỗi).
4. Thiết lập các API key cần thiết trong file `.env`.

---

## 10.2 Tools and Subagents - Công cụ và Sub-agent

### Chủ đề và mục tiêu

Học cách thêm **công cụ (Tools)** và **sub-agent** vào agent. Công cụ cho phép agent gọi chức năng bên ngoài, còn sub-agent cho phép ủy quyền tác vụ cụ thể cho agent khác.

### Giải thích khái niệm cốt lõi

#### Công cụ (Tools)

Trong ADK, công cụ được định nghĩa bằng **hàm Python thông thường**. Agent sử dụng tính năng Function Calling của LLM để gọi các công cụ này tại thời điểm thích hợp. Hàm dùng làm công cụ cần đáp ứng các điều kiện sau:

- **Type hint**: Phải có type hint cho tham số để LLM có thể truyền đối số đúng.
- **Docstring**: Docstring của hàm cho LLM biết mục đích và cách sử dụng công cụ.
- **Giá trị trả về**: Phải trả về giá trị có thể serialize như chuỗi hoặc dictionary.

#### Sub-agent

Sub-agent là agent độc lập được đặt dưới agent chính. Khi agent chính nhận câu hỏi thuộc loại cụ thể, nó **chuyển (transfer)** hội thoại cho sub-agent có thể xử lý câu hỏi đó. Thuộc tính `description` của sub-agent rất quan trọng, vì root agent dựa vào mô tả này để quyết định chuyển tác vụ cho sub-agent nào.

### Phân tích code

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

**Phân tích code:**

1. **Định nghĩa hàm công cụ**:
   - `get_weather(city: str)`: Công cụ nhận tên thành phố và trả về thông tin thời tiết. Type hint `city: str` cho LLM biết cần truyền đối số kiểu chuỗi.
   - `convert_units(degrees: int)`: Công cụ chuyển đổi đơn vị nhiệt độ.

2. **Định nghĩa sub-agent**:
   ```python
   geo_agent = Agent(
       name="GeoAgent",
       instruction="You help with geo questions",
       model=MODEL,
       description="Transfer to this agent when you have a geo related question.",
   )
   ```
   - Thuộc tính `description` là cốt lõi. Root agent (WeatherAgent) đọc description này và quyết định chuyển hội thoại cho GeoAgent khi câu hỏi của người dùng liên quan đến địa lý.

3. **Kết nối công cụ và sub-agent với agent**:
   ```python
   weather_agent = Agent(
       ...
       tools=[get_weather, convert_units],
       sub_agents=[geo_agent],
   )
   ```
   - Truyền trực tiếp hàm Python vào danh sách `tools`. ADK tự động phân tích chữ ký và docstring của hàm để chuyển đổi thành schema công cụ mà LLM hiểu được.
   - Truyền instance sub-agent vào danh sách `sub_agents`.

#### So sánh Công cụ vs Sub-agent

| Đặc tính | Công cụ (Tool) | Sub-agent |
|----------|---------------|-----------|
| Cách định nghĩa | Hàm Python | Instance Agent |
| Chủ thể thực thi | Agent hiện tại gọi trực tiếp | Hội thoại được chuyển cho sub-agent |
| Độ phức tạp | Phù hợp tác vụ đơn giản | Phù hợp tác vụ multi-step phức tạp |
| Sử dụng LLM | Không (thực thi code thuần) | Thực hiện suy luận bằng LLM riêng |
| Trường hợp phù hợp | Gọi API, truy vấn dữ liệu | Phân tích chuyên sâu, hội thoại độc lập |

### Điểm thực hành

1. Xóa type hint trong hàm công cụ và chạy thử. Quan sát trường hợp LLM không truyền đối số đúng.
2. Xóa `description` của `geo_agent` và chạy thử. Kiểm tra ảnh hưởng đến việc root agent chuyển tác vụ cho sub-agent.
3. Thêm hàm công cụ mới (ví dụ: `get_humidity(city: str)`).

---

## 10.3 Agent Architecture - Thiết kế Kiến trúc Agent

### Chủ đề và mục tiêu

Thiết kế và triển khai kiến trúc tổng thể của agent cố vấn tài chính thực tế. Xây dựng cấu trúc trong đó root agent sử dụng nhiều sub-agent chuyên biệt làm công cụ, và học mẫu agent-tool sử dụng `AgentTool` cùng cách viết system prompt chi tiết.

### Giải thích khái niệm cốt lõi

#### AgentTool - Sử dụng Agent như Công cụ

Phương thức `sub_agents` học ở 10.2 là phương thức chuyển (transfer) hội thoại cho sub-agent. Ngược lại, `AgentTool` là phương thức sử dụng sub-agent **như công cụ**. Sự khác biệt cốt lõi giữa hai phương thức:

| Phương thức | `sub_agents=[]` | `AgentTool(agent=)` |
|------------|-----------------|---------------------|
| Hoạt động | Quyền kiểm soát hội thoại chuyển sang sub-agent | Root agent duy trì quyền kiểm soát và gọi sub-agent |
| Ví dụ | "Hướng dẫn khách hàng này sang bộ phận khác" | "Gọi điện cho bộ phận khác để lấy thông tin" |
| Kết quả | Sub-agent trực tiếp trò chuyện với người dùng | Kết quả sub-agent được trả về cho root agent |
| Trường hợp phù hợp | Tác vụ thuộc domain hoàn toàn khác | Khi cần thu thập thông tin rồi tổng hợp đánh giá |

Cố vấn tài chính cần **tổng hợp** nhiều kết quả phân tích, nên mẫu `AgentTool` phù hợp hơn. Vì root agent có thể thu thập kết quả phân tích dữ liệu, phân tích tài chính, phân tích tin tức rồi cung cấp lời khuyên đầu tư cuối cùng.

#### Kiến trúc Agent phân cấp

Kiến trúc được thiết kế trong dự án này như sau:

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

### Phân tích code

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

**Phân tích cốt lõi:**

- `AgentTool(agent=financial_analyst)`: Wrap sub-agent bằng `AgentTool` và đặt vào danh sách `tools`. Như vậy root agent có thể gọi sub-agent như một công cụ.
- `save_advice_report`: Hàm Python thông thường cũng có thể đặt cùng `AgentTool` trong danh sách `tools`. Tại thời điểm này vẫn là hàm rỗng (`pass`), sẽ được triển khai ở phần sau.
- Lưu ý rằng sub-agent được truyền vào `tools` wrap bằng `AgentTool`, không phải `sub_agents`.

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

**Phân tích thiết kế prompt:**

Prompt này là ví dụ tốt về prompt engineering cho agent:

1. **Định nghĩa vai trò**: Giao vai trò rõ ràng "Professional Financial Advisor".
2. **Giới hạn phạm vi**: Giới hạn rõ ràng không phản hồi câu hỏi ngoài lĩnh vực tài chính. Ngăn agent đi lạc sang lĩnh vực không mong muốn.
3. **Định nghĩa quy trình**: Chỉ rõ các bước bắt buộc trước khi đưa ra khuyến nghị BUY/SELL/HOLD.
4. **Hướng dẫn công cụ**: Chỉ rõ các công cụ có sẵn và mục đích sử dụng trong prompt để LLM có thể chọn công cụ phù hợp.
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

**Phân tích cốt lõi:**

- **`LlmAgent` vs `Agent`**: Ở đây sử dụng `LlmAgent`. `LlmAgent` là alias của `Agent`, chức năng tương đương nhau. Chỉ rõ ràng hơn rằng đây là agent hoạt động dựa trên LLM.
- **Sử dụng yfinance**: Mỗi hàm công cụ sử dụng thư viện `yfinance` để lấy dữ liệu chứng khoán thời gian thực từ Yahoo Finance.
- **Giá trị trả về có cấu trúc**: Tất cả công cụ trả về dictionary nhất quán với key `ticker`, `success`. Tính nhất quán này giúp LLM parse kết quả.
- **Thuộc tính `description`**: Được sử dụng khi dùng làm `AgentTool` để root agent hiểu mục đích của sub-agent này.

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

**Phân tích cốt lõi:**

- Sub-agent phân tích tin tức sử dụng `web_search_tool` từ `tools.py` đã tạo ở 10.1.
- Import trực tiếp từ `tools.py` ở thư mục gốc dự án bằng `from tools import web_search_tool`.
- Chỉ có một công cụ, nhưng LLM đảm nhận vai trò tạo truy vấn tìm kiếm và phân tích kết quả.

### Điểm thực hành

1. Đổi `AgentTool` thành `sub_agents` và so sánh sự khác biệt trong hoạt động.
2. Thêm sub-agent mới (ví dụ: `TechnicalAnalyst` thực hiện phân tích kỹ thuật).
3. Xóa "STRICT SCOPE LIMITATIONS" từ prompt và so sánh phản ứng khi hỏi câu hỏi ngoài lĩnh vực tài chính.

---

## 10.4 Agent State - Quản lý Trạng thái Agent

### Chủ đề và mục tiêu

Học cơ chế chia sẻ **trạng thái (State)** giữa các agent. Sử dụng `output_key` để tự động lưu đầu ra của sub-agent vào trạng thái, và học cách truy cập trạng thái từ hàm công cụ thông qua `ToolContext`.

### Giải thích khái niệm cốt lõi

#### Hệ thống State của ADK

Trong ADK, State là **kho lưu trữ key-value** để lưu và chia sẻ dữ liệu trong session agent. Đặc điểm của state:

- **Phạm vi session**: Tất cả agent trong cùng session chia sẻ state.
- **Dạng dictionary**: Truy cập bằng `state["key"]` giống Python dictionary.
- **Có thể lưu tự động**: Sử dụng `output_key` thì đầu ra cuối cùng của agent tự động được lưu vào state.

#### output_key

`output_key` là thuộc tính tự động lưu response cuối cùng của agent vào state:

```python
data_analyst = LlmAgent(
    ...
    output_key="data_analyst_result",
)
```

Thiết lập như trên, khi `DataAnalyst` agent hoàn thành thực thi, response cuối cùng tự động được lưu vào `state["data_analyst_result"]`.

#### ToolContext

`ToolContext` là đối tượng cho phép truy cập context thực thi của agent (state, artifact, v.v.) từ hàm công cụ. Khi thêm `tool_context: ToolContext` vào tham số hàm công cụ, ADK tự động inject context hiện tại.

> **Quan trọng**: Tham số `ToolContext` không bị lộ ra cho LLM. LLM chỉ thấy các tham số thông thường như `summary`, còn `tool_context` được ADK framework inject nội bộ.

### Phân tích code

#### Thêm output_key cho sub-agent

```python
# sub_agents/data_analyst.py
data_analyst = LlmAgent(
    name="DataAnalyst",
    ...
    tools=[get_company_info, get_stock_price, get_financial_metrics],
    output_key="data_analyst_result",   # Đã thêm
)
```

```python
# sub_agents/financial_analyst.py
financial_analyst = Agent(
    name="FinancialAnalyst",
    ...
    tools=[get_income_statement, get_balance_sheet, get_cash_flow],
    output_key="financial_analyst_result",   # Đã thêm
)
```

```python
# sub_agents/news_analyst.py
news_analyst = Agent(
    name="NewsAnalyst",
    ...
    output_key="news_analyst_result",   # Đã thêm
    tools=[web_search_tool],
)
```

Mỗi sub-agent đã được thêm `output_key`. Nhờ vậy, khi mỗi sub-agent hoàn thành thực thi, kết quả tự động được lưu vào state chia sẻ.

#### Hàm công cụ sử dụng state (`agent.py`)

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

**Phân tích code:**

1. **Tham số `ToolContext`**: Khai báo `tool_context: ToolContext` làm tham số đầu tiên. ADK tự động inject context thực thi hiện tại. LLM không nhận biết tham số này, chỉ truyền `summary: str` làm đối số.

2. **Đọc dữ liệu từ state**:
   ```python
   state = tool_context.state
   data_analyst_result = state.get("data_analyst_result")
   ```
   Truy cập dictionary state bằng `tool_context.state` và lấy kết quả mà sub-agent đã lưu bằng `output_key`.

3. **Ghi dữ liệu vào state**:
   ```python
   state["report"] = report
   ```
   Lưu report đã tạo vào state. Sau đó có thể truy cập bằng `state["report"]` từ công cụ hoặc agent khác.

#### Tóm tắt luồng dữ liệu

```
Người dùng: "Phân tích AAPL cho tôi"
    │
    ▼
FinancialAdvisor (Root)
    │
    ├── Gọi AgentTool(DataAnalyst)
    │   └── Kết quả → Tự động lưu vào state["data_analyst_result"]
    │
    ├── Gọi AgentTool(FinancialAnalyst)
    │   └── Kết quả → Tự động lưu vào state["financial_analyst_result"]
    │
    ├── Gọi AgentTool(NewsAnalyst)
    │   └── Kết quả → Tự động lưu vào state["news_analyst_result"]
    │
    └── Gọi save_advice_report(summary=...)
        ├── Đọc tất cả kết quả sub-agent từ state
        ├── Tạo report tổng hợp
        └── Lưu report vào state["report"]
```

### Điểm thực hành

1. Xóa `output_key` và xác nhận `state.get()` trong `save_advice_report` trả về `None`.
2. Tìm cách kiểm tra dữ liệu lưu trong state trên ADK Web UI.
3. Lưu dữ liệu tùy chỉnh (ví dụ: thời gian bắt đầu phân tích) vào `tool_context.state` và sử dụng.

---

## 10.5 Artifacts - Tạo File bằng Artifact

### Chủ đề và mục tiêu

Học cách sử dụng hệ thống **Artifact** của ADK để lưu dữ liệu agent tạo ra thành file. Artifact là cơ chế quản lý sản phẩm (report, hình ảnh, file dữ liệu, v.v.) được agent tạo ra.

### Giải thích khái niệm cốt lõi

#### Artifact là gì?

Artifact là **sản phẩm dạng file** được tạo ra trong quá trình thực thi agent. Khác với response văn bản, artifact được lưu dưới dạng file độc lập để người dùng có thể tải xuống hoặc sử dụng riêng.

Trên ADK Web UI, artifact tự động hiển thị trên UI để người dùng có thể kiểm tra và tải xuống ngay.

#### google.genai.types.Part và Blob

Artifact được biểu diễn bằng đối tượng `types.Part` và `types.Blob` do thư viện `genai` của Google cung cấp:

- **`types.Blob`**: Container chứa dữ liệu nhị phân và MIME type.
- **`types.Part`**: Đối tượng cấp cao bao bọc `Blob`, tương thích với định dạng tin nhắn multimodal của Gemini API.

#### Phương thức save_artifact

Phương thức `save_artifact()` của `ToolContext` là phương thức **bất đồng bộ (async)**. Do đó, hàm công cụ gọi nó cũng phải được khai báo `async`.

### Phân tích code

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

**Phân tích chi tiết phần thay đổi so với 10.4:**

1. **Thêm import mới**:
   ```python
   from google.genai import types
   ```
   Import module `types` cần thiết cho tạo artifact.

2. **Đổi hàm thành async**:
   ```python
   async def save_advice_report(tool_context: ToolContext, summary: str, ticker: str):
   ```
   - Đổi từ `def` sang `async def`. Vì `save_artifact()` là phương thức bất đồng bộ nên cần sử dụng `await`.
   - Thêm tham số `ticker: str`. LLM truyền ticker symbol của công ty đang phân tích, dùng để tạo tên file.

3. **Tạo tên file**:
   ```python
   filename = f"{ticker}_investment_advice.md"
   ```
   Tạo tên file có ý nghĩa một cách động bằng ticker symbol. Ví dụ: `AAPL_investment_advice.md`

4. **Tạo đối tượng artifact**:
   ```python
   artifact = types.Part(
       inline_data=types.Blob(
           mime_type="text/markdown",
           data=report.encode("utf-8"),
       )
   )
   ```
   - `types.Blob`: Chứa MIME type (`text/markdown`) và dữ liệu thực tế (chuỗi report đã encode UTF-8).
   - `types.Part`: Bao bọc `Blob` dưới dạng `inline_data`. Định dạng này là định dạng multimodal tiêu chuẩn của Google Generative AI API.
   - `report.encode("utf-8")`: Chuyển đổi chuỗi thành bytes. `data` của `Blob` yêu cầu dữ liệu bytes.

5. **Lưu artifact**:
   ```python
   await tool_context.save_artifact(filename, artifact)
   ```
   - Gọi `tool_context.save_artifact()` để lưu artifact vào hệ thống ADK.
   - Chờ lưu bất đồng bộ bằng từ khóa `await`.
   - Trên ADK Web UI, artifact tự động hiển thị để người dùng có thể tải xuống file.

#### Sử dụng MIME type

Ngoài `text/markdown`, có thể sử dụng nhiều MIME type khác:

| MIME type | Mục đích |
|-----------|----------|
| `text/markdown` | Tài liệu markdown |
| `text/plain` | Văn bản thuần |
| `text/csv` | Dữ liệu CSV |
| `application/json` | Dữ liệu JSON |
| `image/png` | Hình ảnh PNG |
| `application/pdf` | Tài liệu PDF |

### Điểm thực hành

1. Đổi MIME type thành `text/csv` và tạo artifact định dạng CSV.
2. Lưu nhiều artifact trong một lệnh gọi công cụ (ví dụ: report tóm tắt + dữ liệu chi tiết).
3. Kiểm tra cách artifact hiển thị trên ADK Web UI và tải xuống.
4. Suy nghĩ về vấn đề khi sử dụng tên file cố định mà không có tham số `ticker`.

---

## Tổng kết cốt lõi chương

### 1. Cấu trúc dự án ADK

- Dự án ADK tuân theo cấu trúc Python package, phải import module `agent` trong `__init__.py`.
- File `agent.py` phải có biến `root_agent` để ADK Web UI nhận diện agent.
- Có thể sử dụng mô hình từ nhiều nhà cung cấp LLM như OpenAI thông qua `LiteLlm`.

### 2. Công cụ (Tools) và Sub-agent

- **Công cụ** được định nghĩa bằng hàm Python thông thường, type hint và docstring là bắt buộc.
- **Sub-agent** có thể kết nối qua tham số `sub_agents` hoặc `AgentTool`.
- `sub_agents` chuyển quyền kiểm soát hội thoại, `AgentTool` gọi như công cụ và nhận kết quả.

### 3. Kiến trúc Agent

- Sử dụng mẫu `AgentTool` cho phép root agent duy trì quyền kiểm soát và sử dụng sub-agent.
- Chỉ rõ danh sách công cụ có sẵn và thời điểm sử dụng trong prompt giúp tăng mức độ sử dụng công cụ.
- Kiểm soát agent hoạt động trong phạm vi mong muốn thông qua giới hạn phạm vi (Scope Limitation).

### 4. Quản lý State

- Thiết lập `output_key` giúp kết quả sub-agent tự động được lưu vào state chia sẻ.
- Truy cập (đọc/ghi) state từ hàm công cụ thông qua `ToolContext`.
- Tham số `ToolContext` được ADK inject tự động và không bị lộ cho LLM.

### 5. Artifact

- Tạo đối tượng artifact bằng `types.Part` và `types.Blob`.
- `tool_context.save_artifact()` là phương thức bất đồng bộ nên phải sử dụng `async/await`.
- Thông qua artifact, agent có thể tạo file (report, dữ liệu, v.v.) và truyền cho người dùng.

---

## Bài tập thực hành

### Bài 1: Mở rộng Agent cơ bản (Độ khó: Thấp)

Thêm công cụ mới `get_stock_recommendations(ticker: str)` vào sub-agent `DataAnalyst` hiện tại. Công cụ này phải sử dụng thuộc tính `recommendations` của `yfinance` để trả về dữ liệu khuyến nghị của analyst.

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

### Bài 2: Thêm Sub-agent mới (Độ khó: Trung bình)

Tạo sub-agent mới tên `TechnicalAnalyst`. Agent này cần có các công cụ sau:

- `get_moving_averages(ticker: str, period: str)`: Tính toán dữ liệu đường trung bình di động
- `get_volume_analysis(ticker: str)`: Dữ liệu phân tích khối lượng giao dịch

Kết nối sub-agent đã tạo với root agent bằng `AgentTool` và thêm mô tả công cụ vào prompt.

### Bài 3: Đa dạng hóa Artifact (Độ khó: Trung bình)

Mở rộng hàm `save_advice_report` để tạo thêm artifact dữ liệu tóm tắt định dạng JSON ngoài report markdown. Cả hai artifact phải được lưu trong một lệnh gọi hàm.

**Gợi ý**: Gọi `save_artifact()` hai lần.

```python
# Artifact JSON
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

### Bài 4: Kiểm soát luồng hội thoại dựa trên State (Độ khó: Cao)

Xây dựng hệ thống theo dõi tiến trình phân tích bằng cách sử dụng state trong prompt `instruction` của root agent. Ghi cờ hoàn thành vào state mỗi khi sub-agent được thực thi, và chỉ gọi `save_advice_report` khi tất cả phân tích hoàn thành.

**Gợi ý**: Tạo hàm công cụ mới `check_analysis_status(tool_context: ToolContext)` để kiểm tra các bước phân tích đã hoàn thành.

### Bài 5: Phân tích so sánh đa doanh nghiệp (Độ khó: Cao)

Thêm tính năng phân tích và so sánh nhiều doanh nghiệp (ví dụ: AAPL, GOOGL, MSFT) đồng thời. Lưu kết quả phân tích vào state phân loại theo từng doanh nghiệp và tạo report so sánh cuối cùng dưới dạng artifact.

**Cân nhắc**:
- `output_key` chỉ hỗ trợ một key duy nhất, nên có thể cần quản lý state trực tiếp trong hàm công cụ.
- Tổ chức các chỉ số cốt lõi của mỗi doanh nghiệp dưới dạng bảng trong report so sánh sẽ hiệu quả hơn.

---

## Tài liệu tham khảo

- [Tài liệu chính thức Google ADK](https://google.github.io/adk-docs/)
- [Tài liệu LiteLLM](https://docs.litellm.ai/)
- [Thư viện yfinance](https://github.com/ranaroussi/yfinance)
- [Tài liệu Firecrawl](https://docs.firecrawl.dev/)
