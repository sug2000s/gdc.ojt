# Chapter 19: Agent AI Giáo dục (Tutor Agent)

---

## 1. Tổng quan chương

Trong Chapter 19, chúng ta xây dựng **hệ thống agent AI gia sư giáo dục**. Hệ thống này tận dụng kiến trúc đa agent của LangGraph để đánh giá trình độ người học và hướng dẫn họ đến phương pháp học tập tối ưu -- một nền tảng giáo dục thông minh.

### Các thành phần hệ thống

| Agent | Vai trò | Phương pháp học tập |
|-------|---------|---------------------|
| **Classification Agent** | Đánh giá trình độ người học và định tuyến | Chuyên gia đánh giá giáo dục |
| **Teacher Agent** | Giáo dục có hệ thống từng bước | Giảng dạy kiểu bài giảng có cấu trúc |
| **Feynman Agent** | Xác minh mức độ hiểu bằng kỹ thuật Feynman | "Nếu bạn không thể giải thích đơn giản, bạn chưa hiểu" |
| **Quiz Agent** | Đánh giá học tập chủ động qua câu hỏi trắc nghiệm | Tạo bài trắc nghiệm dựa trên nghiên cứu |

### Mục tiêu học tập cốt lõi

- Thiết kế hệ thống đa agent sử dụng `create_react_agent` của LangGraph
- Triển khai mẫu chuyển giao giữa các agent
- Định tuyến agent trong đồ thị sử dụng đối tượng `Command`
- Workflow động sử dụng edge có điều kiện (Conditional Edges)
- Sử dụng Structured Output dựa trên Pydantic
- Triển khai công cụ tìm kiếm web với Firecrawl

### Kiến trúc dự án

```
tutor-agent/
├── main.py                          # Định nghĩa đồ thị chính
├── langgraph.json                   # Cấu hình LangGraph
├── pyproject.toml                   # Dependency dự án
├── agents/
│   ├── classification_agent.py      # Agent phân loại người học
│   ├── teacher_agent.py             # Agent giáo viên
│   ├── feynman_agent.py             # Agent kỹ thuật Feynman
│   └── quiz_agent.py                # Agent trắc nghiệm
└── tools/
    ├── shared_tools.py              # Công cụ chia sẻ (chuyển giao, tìm kiếm web)
    └── quiz_tools.py                # Công cụ tạo trắc nghiệm
```

### Sơ đồ luồng Agent

```
[START] → [router_check] ─→ [classification_agent] → [END]
                │
                ├─→ [teacher_agent]
                ├─→ [feynman_agent]
                └─→ [quiz_agent]
```

Sau khi `classification_agent` đánh giá người học, nó chuyển giao đến agent phù hợp thông qua công cụ `transfer_to_agent`. Khi cuộc hội thoại tiếp tục, `router_check` kiểm tra trạng thái `current_agent` và định tuyến đến đúng agent.

---

## 2. Mô tả chi tiết từng phần

---

### 2.1 Phần 19.0 -- Introduction (Thiết lập dự án ban đầu)

**Commit**: `0516cd0` "19.0 Introduction"

#### Chủ đề và Mục tiêu

Xây dựng nền tảng cho dự án `tutor-agent` mới. Thiết lập cấu trúc dự án Python và định nghĩa tất cả gói dependency cần thiết.

#### Các khái niệm chính

##### Khởi tạo cấu trúc dự án

Dự án mới được tạo bằng `uv` (trình quản lý gói Python). `uv` là công cụ quản lý gói Python hiện đại nhanh hơn nhiều so với `pip`.

##### Quản lý phiên bản Python

```
3.13
```

File `.python-version` chỉ định phiên bản Python sử dụng cho dự án. Các công cụ như `pyenv`, `uv` tự động nhận diện file này và sử dụng đúng phiên bản Python.

##### Định nghĩa Dependency (pyproject.toml)

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

**Mô tả các gói chính:**

| Gói | Phiên bản | Vai trò |
|-----|-----------|---------|
| `firecrawl-py` | 2.16 | Client API tìm kiếm và thu thập dữ liệu web |
| `grandalf` | 0.8 | Trực quan hóa đồ thị (để render đồ thị LangGraph) |
| `langchain[openai]` | 0.3.27 | Framework LangChain + tích hợp OpenAI |
| `langgraph` | 0.6.6 | Framework đồ thị agent dựa trên trạng thái |
| `langgraph-checkpoint-sqlite` | 2.0.11 | Kho lưu trữ checkpoint dựa trên SQLite |
| `langgraph-cli[inmem]` | 0.4.0 | CLI LangGraph (bao gồm chế độ trong bộ nhớ) |
| `langgraph-supervisor` | 0.0.29 | Mẫu agent Supervisor |
| `langgraph-swarm` | 0.0.14 | Mẫu agent Swarm |
| `pytest` | 8.4.2 | Framework kiểm thử |
| `python-dotenv` | 1.1.1 | Tải biến môi trường từ file `.env` |

##### langgraph-supervisor vs langgraph-swarm

Dự án này cài đặt cả hai thư viện mẫu đa agent:
- **Mẫu Supervisor**: Một quản lý trung tâm điều phối các agent
- **Mẫu Swarm**: Các agent tự chủ truyền tác vụ cho nhau

Dự án này sử dụng **mẫu gần với Swarm hơn**, vì mỗi agent trực tiếp chuyển giao đến agent khác thông qua công cụ `transfer_to_agent`.

#### Điểm thực hành

1. Tạo dự án bằng `uv init tutor-agent` và sửa đổi `pyproject.toml` để thêm dependency.
2. Chạy `uv sync` để cài đặt tất cả dependency.
3. Tạo file `.env` và thiết lập `OPENAI_API_KEY` và `FIRECRAWL_API_KEY`.

---

### 2.2 Phần 19.1 -- Classification Agent (Agent phân loại người học)

**Commit**: `269599b` "19.1 Classification Agent"

#### Chủ đề và Mục tiêu

Triển khai **agent phân loại** xác định trình độ, phong cách học tập và mục tiêu học tập của người học để kết nối họ với agent học tập tối ưu. Agent này đóng vai trò là Điểm vào (Entry Point) cho toàn bộ hệ thống.

#### Các khái niệm chính

##### Hiểu về create_react_agent

`create_react_agent` của LangGraph là hàm tạo agent theo **mẫu ReAct (Reasoning + Acting)** một cách tiện lợi.

```python
from langgraph.prebuilt import create_react_agent

classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="...",       # System prompt
    tools=[...],        # Danh sách công cụ có sẵn
)
```

**Mẫu ReAct là gì?**
- **Reasoning**: LLM phân tích tình huống hiện tại và quyết định hành động tiếp theo
- **Acting**: Gọi công cụ hoặc tạo phản hồi dựa trên quyết định
- Hai bước này lặp lại để hoàn thành tác vụ

##### Quy trình đánh giá của Classification Agent

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

Các nguyên tắc thiết kế cốt lõi của prompt này:

1. **Cấu trúc đánh giá 3 giai đoạn**: Xác định chủ đề -> Sở thích học tập -> Mục tiêu học tập, theo thứ tự có hệ thống
2. **Ngăn ngừa quá tải**: "Don't overwhelm - max 2 questions at a time" -- tối đa 2 câu hỏi mỗi lần
3. **Sử dụng gợi ý ngầm**: Nếu người dùng sử dụng thuật ngữ kỹ thuật đúng cách, giả định họ có một số kiến thức nền tảng

##### Logic đề xuất Agent

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

Tiêu chí chuyển giao cho từng agent:
- **quiz_agent**: Người học ưa thích hồi tưởng chủ động (Active Recall)
- **teacher_agent**: Người mới bắt đầu hoặc người học cần giải thích có cấu trúc
- **feynman_agent**: Người học tuyên bố đã hiểu nhưng cần xác minh

##### Mã bí mật dành cho nhà phát triển

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to a random agent (quiz_agent, teacher_agent, or feynman_agent)
    for testing purposes using the transfer_to_agent tool.
```

Để thuận tiện cho kiểm thử, nhập "GODMODE" sẽ bỏ qua quy trình đánh giá và chuyển ngay đến một agent. Đây là mẫu thực tế để kiểm thử nhanh trong quá trình phát triển.

##### Công cụ transfer_to_agent (Phiên bản ban đầu)

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

**Điểm quan trọng:** Trong phiên bản ban đầu này, logic chuyển giao thực tế dựa trên `Command` được **comment**. Vì các node agent khác chưa được đăng ký trong đồ thị, nó được triển khai dưới dạng stub đơn giản trả về chuỗi. Đây là ví dụ tốt về chiến lược phát triển gia tăng (Incremental Development).

##### Cấu hình đồ thị chính

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

**Phân tích mã nguồn:**

1. **`load_dotenv()`**: Tải biến môi trường như API key từ file `.env`. **Phải được gọi trước các import** -- các module khác có thể cần biến môi trường tại thời điểm import.

2. **`TutorState(MessagesState)`**: Kế thừa từ `MessagesState` của LangGraph để tự động quản lý tin nhắn hội thoại. Tại thời điểm này, sử dụng `pass` không có trạng thái bổ sung.

3. **Cấu trúc đồ thị**: Cấu trúc tuyến tính đơn giản `START -> classification_agent -> END`.

##### File cấu hình LangGraph

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

`langgraph.json` là file cấu hình được LangGraph CLI (`langgraph dev`) tham chiếu:
- **dependencies**: Danh sách file phụ thuộc (để phát hiện thay đổi)
- **graphs**: Đồ thị cần hiển thị và điểm vào của chúng
- **env**: Đường dẫn file biến môi trường

#### Điểm thực hành

1. Chạy server phát triển bằng `langgraph dev` và kiểm tra đồ thị trong LangGraph Studio.
2. Trò chuyện với Classification Agent và kiểm tra xem quy trình đánh giá có diễn ra tự nhiên không.
3. Sửa đổi prompt để thêm tiêu chí đánh giá khác (ví dụ: "người học thị giác vs người học thính giác").

---

### 2.3 Phần 19.2 -- Feynman Agent & Teacher Agent

**Commit**: `5c2dfa9` "19.2 Feynman Agent"

#### Chủ đề và Mục tiêu

Trong phần này, chúng ta triển khai hai agent học tập cốt lõi và hoàn thiện cơ chế chuyển giao giữa agent:
- **Teacher Agent**: Giáo dục có hệ thống từng bước
- **Feynman Agent**: Xác minh mức độ hiểu bằng kỹ thuật Feynman
- **Công cụ tìm kiếm web**: Tìm kiếm thông tin thời gian thực dựa trên Firecrawl
- **Chuyển giao agent thực tế**: Định tuyến sử dụng đối tượng `Command`

#### Các khái niệm chính

##### Feynman Agent -- Kỹ thuật học tập Feynman

Triết lý học tập của Richard Feynman được triển khai thành AI agent. Nguyên tắc cốt lõi là **"Nếu bạn không thể giải thích đơn giản, bạn chưa hiểu đủ rõ."**

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

**Quy trình 6 bước của kỹ thuật Feynman:**

| Bước | Tên | Mô tả |
|------|-----|-------|
| Bước 1 | Giai đoạn nghiên cứu | Thu thập thông tin chính xác về khái niệm qua tìm kiếm web |
| Bước 2 | Yêu cầu giải thích đơn giản | Yêu cầu "giải thích như cho trẻ 8 tuổi" |
| Bước 3 | Nhận giải thích từ người dùng | Lắng nghe và phân tích lời giải thích của người dùng |
| Bước 4 | Đánh giá độ phức tạp | Đánh giá thuật ngữ chuyên môn, lỗ hổng logic, giải thích mơ hồ |
| Bước 5 | Đặt câu hỏi làm rõ | Đặt câu hỏi cụ thể về phần phức tạp |
| Bước 6 | Hoàn thành | Nếu đủ đơn giản, công nhận sự thành thạo |

**Tiêu chí đánh giá cốt lõi:**

```
    ## Your Evaluation Criteria:
    - No unexplained technical terms
    - Clear cause-and-effect relationships
    - Uses analogies or examples a child would understand
    - Logical flow without gaps
    - Their own words, not memorized definitions
```

Các tiêu chí này được sử dụng để phân biệt liệu người học chỉ thuộc lòng định nghĩa hay thực sự hiểu. Nhấn mạnh "Their own words" (lời của chính họ) đặc biệt quan trọng.

##### Teacher Agent -- Agent giáo dục có hệ thống

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

**Phương pháp giảng dạy của Teacher Agent:**

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

**Quy tắc giảng dạy quan trọng của Teacher Agent:**

```
    ## Critical Teaching Rules:
    1. Always confirm understanding before moving to the next concept
    2. If they don't understand, explain differently (not just repeat)
    3. Break complex topics into the smallest possible pieces
    4. Use examples from their world and experience
    5. Be patient - true understanding takes time
```

Quy tắc số 2 đặc biệt quan trọng -- khi người học không hiểu, bạn nên **giải thích theo cách khác** thay vì lặp lại cùng lời giải thích. Đây cũng là năng lực cốt lõi của những giáo viên xuất sắc.

##### Công cụ tìm kiếm web (web_search_tool)

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

**Phân tích mã nguồn:**

1. **FirecrawlApp**: Sử dụng API Firecrawl để thực hiện tìm kiếm web. Đây là dịch vụ kết hợp tìm kiếm Google + thu thập trang web.
2. **`limit=5`**: Giới hạn kết quả tìm kiếm còn 5 để tiết kiệm chi phí token.
3. **`ScrapeOptions(formats=["markdown"])`**: Nhận kết quả ở định dạng markdown để LLM xử lý dễ dàng.
4. **Làm sạch văn bản (Text Cleaning)**:
   - `re.sub(r"\\+|\n+", "", markdown)`: Xóa ký tự escape không cần thiết và xuống dòng quá mức
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: Xóa liên kết markdown và URL

Quá trình làm sạch này rất quan trọng cho **tiết kiệm token** và **giảm nhiễu**. Markdown gốc từ trang web chứa nhiều thông tin không cần thiết như liên kết điều hướng và liên kết quảng cáo.

##### Hoàn thiện công cụ transfer_to_agent (Sử dụng Command)

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

Triển khai stub từ phần trước đã được thay đổi thành **logic chuyển giao thực tế dựa trên `Command`**.

**Các tham số của đối tượng `Command`:**

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `goto` | `agent_name` | Tên node đích cần di chuyển đến |
| `graph` | `Command.PARENT` | Chỉ ra tìm node ở đồ thị cha |
| `update` | `{"current_agent": agent_name}` | Cập nhật trạng thái đồ thị |

**Tại sao cần `Command.PARENT`:**
`transfer_to_agent` được gọi từ bên trong công cụ. Công cụ thực thi trong ngữ cảnh phụ của agent, vì vậy để di chuyển đến node khác ở cấp cao nhất của đồ thị, phải chỉ định `Command.PARENT` để chỉ ra đó là chuyển đổi ở cấp đồ thị cha.

##### Tiến hóa đồ thị chính -- Mẫu Router

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

**Phân tích mã nguồn:**

1. **Thêm trường `current_agent` vào `TutorState`**: Theo dõi agent nào đang hoạt động hiện tại.

2. **Hàm `router_check`**: Khi cuộc hội thoại tiếp tục (tin nhắn mới đến), kiểm tra giá trị `current_agent` trong trạng thái hiện tại và định tuyến đến node phù hợp. Giá trị mặc định là `"classification_agent"`.

3. **Tham số `destinations`**: Chỉ định `destinations` trên node `classification_agent` cho LangGraph biết những node nào có thể đến được từ node này. Điều này cần thiết để chuyển giao agent sử dụng `Command` hoạt động đúng.

4. **`add_conditional_edges`**: Thêm phân nhánh có điều kiện từ `START`. Tùy thuộc vào giá trị trả về của hàm `router_check`, hệ thống đi vào một trong ba agent.

**Luồng định tuyến:**
```
Cuộc hội thoại mới bắt đầu:
  current_agent chưa thiết lập → router_check mặc định → "classification_agent"

classification_agent gọi transfer_to_agent("teacher_agent"):
  Command(goto="teacher_agent", update={"current_agent": "teacher_agent"})
  → Chuyển sang teacher_agent

Tin nhắn tiếp theo:
  current_agent = "teacher_agent" → router_check → Đi thẳng đến teacher_agent
```

#### Điểm thực hành

1. Giải thích một khái niệm bạn biết rõ cho Feynman Agent và xem phản hồi nó đưa ra.
2. Yêu cầu Teacher Agent dạy một chủ đề mới và trải nghiệm quá trình giáo dục từng bước.
3. Sửa đổi regex trong `web_search_tool` để thử nghiệm các chiến lược làm sạch khác nhau.
4. Thêm logging vào hàm `router_check` để theo dõi quá trình định tuyến.

---

### 2.4 Phần 19.3 -- Quiz Agent (Agent trắc nghiệm)

**Commit**: `e188909` "19.3 Quiz Agent"

#### Chủ đề và Mục tiêu

Triển khai **Quiz Agent** tạo động các bài trắc nghiệm nhiều lựa chọn có cấu trúc dựa trên kết quả tìm kiếm web. Sử dụng **Structured Output** của Pydantic để LLM tạo trắc nghiệm theo định dạng được xác định.

#### Các khái niệm chính

##### Structured Output dựa trên Pydantic

Khái niệm kỹ thuật quan trọng nhất trong phần này là **Structured Output**. Thay vì văn bản tự do, LLM được yêu cầu tạo dữ liệu khớp với schema được định nghĩa trước.

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

**Các điểm chính trong thiết kế Schema:**

1. **Mô hình `Question`**: Định nghĩa nghiêm ngặt cấu trúc mỗi câu hỏi
   - `question`: Văn bản câu hỏi
   - `options`: Chính xác 4 lựa chọn (A, B, C, D)
   - `correct_answer`: Đáp án đúng (phải khớp với một trong các options)
   - `explanation`: Giải thích tại sao đáp án đúng và tại sao các đáp án khác sai

2. **Mô hình `Quiz`**: Cấu trúc tổng thể bài trắc nghiệm
   - `topic`: Chủ đề trắc nghiệm
   - `questions`: Danh sách các đối tượng Question

3. **`Field(description=...)`**: Truyền mô tả trường cho LLM để hướng dẫn đầu ra chính xác hơn. Đặc biệt quan trọng khi chỉ định ràng buộc như `"MUST MATCH ONE OF 'options'"`.

##### Công cụ generate_quiz

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

**Phân tích mã nguồn:**

1. **`init_chat_model("openai:gpt-4o")`**: Hàm khởi tạo mô hình phổ quát của LangChain. Provider và mô hình được chỉ định ở định dạng chuỗi.

2. **`model.with_structured_output(Quiz)`**: Đây là điểm then chốt. Chuyển đổi mô hình chat thông thường thành **mô hình đầu ra có cấu trúc**. Nội bộ sử dụng JSON mode / function calling của OpenAI để đảm bảo đầu ra khớp mô hình Pydantic `Quiz`.

3. **`Literal["easy", "medium", "hard"]`**: Sử dụng type hint của Python để giới hạn tham số difficulty chỉ có 3 giá trị. LLM nhận ra ràng buộc này khi gọi công cụ.

4. **Thẻ XML `<RESEARCH_INFORMATION>`**: Sử dụng thẻ XML để phân tách rõ ràng dữ liệu nghiên cứu trong prompt. Điều này ngăn LLM nhầm lẫn giữa chỉ thị prompt và dữ liệu.

**Mẫu gọi LLM riêng biệt bên trong công cụ:**
`generate_quiz` là công cụ, nhưng nội bộ gọi lại LLM. Đây là biến thể của mẫu **"Agent as Tool" (Agent như Công cụ)**. Khi agent bên ngoài (Quiz Agent) gọi công cụ này, LLM bên trong công cụ tạo bài trắc nghiệm có cấu trúc và trả về. Điều này cho phép đóng gói sạch sẽ logic tạo trắc nghiệm.

##### Prompt Quiz Agent -- Workflow nghiêm ngặt

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

**Mẫu workflow bắt buộc trong Prompt:**

```
    ## CRITICAL WORKFLOW - MUST FOLLOW IN ORDER:
    1. STEP 1: RESEARCH FIRST - You MUST use web_search_tool before anything else
    2. STEP 2: ASK LENGTH - Ask student how many questions they want
    3. STEP 3: CALL generate_quiz - Pass the research_text from step 1
    4. STEP 4: PRESENT ONE BY ONE - Show questions individually
    5. STEP 5: USE EXPLANATIONS - Use the explanations provided by the quiz tool

    NEVER call generate_quiz without research_text from web_search_tool first!
```

Phần này là **kỹ thuật prompt engineering để kiểm soát nghiêm ngặt hành vi LLM**. Các biểu thức nhấn mạnh như "CRITICAL", "MUST", "NEVER" cùng số bước được sử dụng để đảm bảo LLM tuân theo thứ tự định sẵn. Đặc biệt, tạo trắc nghiệm không qua tìm kiếm web có thể bao gồm thông tin không chính xác, nên nghiên cứu phải được thực hiện trước.

##### Tích hợp Quiz Agent vào đồ thị chính

```python
from agents.quiz_agent import quiz_agent

# ... thêm vào mã hiện có ...
graph_builder.add_node("quiz_agent", quiz_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
        "quiz_agent",       # Đã thêm
    ],
)
```

Node quiz_agent được đăng ký trong đồ thị và thêm vào danh sách edge có điều kiện của `router_check`.

##### Cập nhật Classification Agent

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to quiz_agent for testing purposes using the transfer_to_agent tool.
```

Mã bí mật GODMODE đã thay đổi từ "agent ngẫu nhiên" sang **`quiz_agent` cố định**. Thay đổi này nhằm kiểm thử tập trung Quiz Agent mới được thêm.

##### Cập nhật shared_tools.py

```python
    agent_name: Name of the agent to transfer to, one of:
                'quiz_agent', 'teacher_agent' or 'feynman_agent'
```

`quiz_agent` đã được thêm lại vào docstring của `transfer_to_agent`. Vì LLM đọc docstring để xác định giá trị tham số công cụ, việc liệt kê chính xác tên agent có thể ở đây rất quan trọng.

#### Điểm thực hành

1. Yêu cầu Quiz Agent tạo trắc nghiệm về một chủ đề cụ thể và đánh giá chất lượng trắc nghiệm được tạo.
2. Thay đổi tham số `difficulty` của `generate_quiz` và kiểm tra sự khác biệt câu hỏi theo mức độ khó.
3. Thêm trường `hint` vào mô hình `Question` và triển khai tính năng gợi ý.
4. Thiết kế tính năng lưu kết quả trắc nghiệm vào trạng thái để theo dõi lịch sử học tập.

---

## 3. Tổng kết trọng tâm chương

### Các mẫu kiến trúc

1. **Mẫu Swarm đa Agent**: Các agent tự chủ truyền tác vụ cho nhau sử dụng đối tượng `Command`. Không có quản lý trung tâm, mỗi agent tự đánh giá và chuyển giao đến agent phù hợp.

2. **Mẫu định tuyến có điều kiện**: Kết hợp hàm `router_check` với `add_conditional_edges` để tự động định tuyến đến agent phù hợp dựa trên trạng thái hội thoại.

3. **Chiến lược phát triển gia tăng**: Phát triển theo thứ tự stub -> triển khai thực tế (tạo `transfer_to_agent` dưới dạng stub ở 19.1 và hoàn thành ở 19.2).

### Kỹ thuật Prompt Engineering

4. **Prompt quy trình từng bước**: Tất cả agent sử dụng prompt có hệ thống với các bước rõ ràng (Step 1, Step 2...). Điều này làm cho hành vi LLM có thể dự đoán được.

5. **Persona dựa trên vai trò**: Mỗi agent được gán vai trò cụ thể như "Educational Assessment Specialist", "Master Teacher", "Feynman Technique Master", "Quiz Master" để giới hạn phạm vi hành vi.

6. **Mẫu workflow bắt buộc**: Thiết kế prompt với biểu thức nhấn mạnh "CRITICAL", "MUST", "NEVER" để đảm bảo LLM tuân theo thứ tự định sẵn.

### Thiết kế công cụ

7. **Structured Output**: Sử dụng mô hình Pydantic + `with_structured_output()` để chuyển đổi đầu ra LLM thành dữ liệu có cấu trúc có thể xử lý bằng lập trình.

8. **Mẫu gọi LLM bên trong công cụ**: Công cụ `generate_quiz` gọi LLM riêng biệt nội bộ để tạo trắc nghiệm có cấu trúc. Đây là ví dụ tốt về đóng gói logic.

9. **Làm sạch kết quả tìm kiếm web**: Sử dụng regex để xóa liên kết, URL và ký tự escape không cần thiết, tiết kiệm token và cải thiện chất lượng đầu vào LLM.

### API LangGraph cốt lõi

| API | Mục đích |
|-----|----------|
| `create_react_agent()` | Tạo agent mẫu ReAct |
| `StateGraph` | Định nghĩa đồ thị dựa trên trạng thái |
| `MessagesState` | Quản lý trạng thái dựa trên tin nhắn |
| `Command(goto, graph, update)` | Chuyển giao agent trong đồ thị |
| `Command.PARENT` | Chuyển giao ở cấp đồ thị cha |
| `add_conditional_edges()` | Thêm edge phân nhánh có điều kiện |
| `add_node(destinations=...)` | Khai báo đích có thể đến từ một node |

---

## 4. Bài tập thực hành

### Bài tập 1: Thêm Agent mới (Cơ bản)

Tạo **Flashcard Agent** và thêm vào hệ thống.

- Tạo `agents/flashcard_agent.py`
- Sau khi tìm kiếm web về chủ đề học tập, tạo flashcard (mặt trước: câu hỏi, mặt sau: câu trả lời)
- Định nghĩa schema `Flashcard` và `FlashcardDeck` bằng mô hình Pydantic
- Thêm node vào đồ thị trong `main.py` và cấu hình định tuyến
- Thêm điều kiện chuyển giao flashcard_agent vào prompt của `classification_agent`

### Bài tập 2: Theo dõi lịch sử học tập (Trung cấp)

Mở rộng `TutorState` để triển khai theo dõi lịch sử học tập.

- Thêm trường `quiz_scores: list[dict]`, `topics_learned: list[str]`, `current_topic: str` vào `TutorState`
- Sửa Quiz Agent để lưu kết quả trắc nghiệm (điểm, chủ đề, ngày) vào trạng thái
- Sửa Teacher Agent để tham chiếu các chủ đề đã học trước và đề xuất khái niệm liên quan
- Triển khai công cụ `progress_report` tóm tắt lịch sử học tập

### Bài tập 3: Điều chỉnh độ khó thích ứng (Nâng cao)

Triển khai hệ thống tự động điều chỉnh độ khó dựa trên kết quả trắc nghiệm của người học.

- Nếu tỷ lệ đúng liên tiếp từ 80% trở lên, tăng độ khó một cấp
- Nếu tỷ lệ đúng liên tiếp dưới 50%, tự động chuyển sang Teacher Agent
- Ghi lại lịch sử thay đổi độ khó vào trạng thái
- Trực quan hóa đường cong thay đổi độ khó dưới dạng văn bản trong báo cáo học tập cuối cùng

### Bài tập 4: Chia sẻ ngữ cảnh giữa các Agent (Nâng cao)

Trong hệ thống hiện tại, kết quả đánh giá từ agent trước không được truyền rõ ràng cho agent mới khi chuyển giao.

- Thêm trường `learner_profile: dict` vào `TutorState` để lưu kết quả đánh giá của Classification Agent ở dạng có cấu trúc
- Sửa prompt mỗi agent để tham chiếu `learner_profile` và tạo phản hồi phù hợp với trình độ người học
- Thêm lý do chuyển giao (`transfer_reason`) vào công cụ `transfer_to_agent` để agent tiếp theo hiểu ngữ cảnh

### Bài tập 5: Phát triển công cụ tùy chỉnh (Nâng cao)

Tham khảo mẫu `generate_quiz` và triển khai các công cụ sau.

- **`generate_summary`**: Công cụ tạo bản tóm tắt học tập dựa trên kết quả tìm kiếm web. Định nghĩa schema `Section` và `Summary` bằng mô hình Pydantic.
- **`evaluate_explanation`**: Công cụ đánh giá tự động cho Feynman Agent. Phân tích lời giải thích của người học và đánh giá việc sử dụng thuật ngữ chuyên môn, tính nhất quán logic và tính ngắn gọn trên thang điểm 0-10.

---

> **Lưu ý**: Để chạy mã trong chương này, bạn cần biến môi trường `OPENAI_API_KEY` và `FIRECRAWL_API_KEY`. Thiết lập chúng trong file `.env` hoặc chỉ định bằng `export` trong terminal. Để kiểm tra trực quan đồ thị trong LangGraph Studio, sử dụng lệnh `langgraph dev`.
