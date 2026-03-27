# Chapter 19: AI Agent Giáo dục (Tutor Agent)

---

## 1. Tổng quan chương

Trong Chapter 19, chúng ta sẽ xây dựng **hệ thống AI tutor agent giáo dục**. Hệ thống này tận dụng kiến trúc multi-agent của LangGraph để đánh giá trình độ người học và hướng dẫn bằng phương pháp học tập tối ưu -- một nền tảng giáo dục thông minh.

### Thành phần hệ thống

| Agent | Vai trò | Phương pháp học tập |
|---------|------|------------|
| **Classification Agent** | Đánh giá trình độ người học và định tuyến | Chuyên gia đánh giá giáo dục |
| **Teacher Agent** | Giáo dục từng bước có hệ thống | Giáo dục theo phương pháp giảng dạy có cấu trúc |
| **Feynman Agent** | Xác minh mức độ hiểu biết dựa trên kỹ thuật Feynman | "Nếu không giải thích đơn giản được thì chưa hiểu" |
| **Quiz Agent** | Đánh giá học tập chủ động qua quiz | Tạo quiz trắc nghiệm dựa trên nghiên cứu |

### Mục tiêu học tập cốt lõi

- Thiết kế hệ thống multi-agent sử dụng `create_react_agent` của LangGraph
- Triển khai mẫu chuyển đổi (Transfer) giữa các agent
- Định tuyến agent trong đồ thị bằng đối tượng `Command`
- Workflow động sử dụng edge có điều kiện (Conditional Edges)
- Sử dụng đầu ra có cấu trúc (Structured Output) dựa trên Pydantic
- Triển khai công cụ tìm kiếm web sử dụng Firecrawl

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
│   └── quiz_agent.py                # Agent quiz
└── tools/
    ├── shared_tools.py              # Tool dùng chung (chuyển đổi, tìm kiếm web)
    └── quiz_tools.py                # Tool tạo quiz
```

### Sơ đồ luồng agent

```
[START] → [router_check] ─→ [classification_agent] → [END]
                │
                ├─→ [teacher_agent]
                ├─→ [feynman_agent]
                └─→ [quiz_agent]
```

`classification_agent` đánh giá người học, sau đó chuyển đổi sang agent phù hợp thông qua tool `transfer_to_agent`. Khi hội thoại tiếp tục, `router_check` kiểm tra trạng thái `current_agent` để định tuyến đến đúng agent.

---

## 2. Giải thích chi tiết từng phần

---

### 2.1 Phần 19.0 -- Introduction (Thiết lập ban đầu dự án)

**Commit**: `0516cd0` "19.0 Introduction"

#### Chủ đề và mục tiêu

Xây dựng nền tảng cho dự án `tutor-agent` mới. Thiết lập cấu trúc dự án Python và định nghĩa tất cả dependency cần thiết.

#### Giải thích khái niệm cốt lõi

##### Khởi tạo cấu trúc dự án

Sử dụng `uv` (trình quản lý gói Python) để tạo dự án mới. `uv` là công cụ quản lý gói Python hiện đại, nhanh hơn nhiều so với `pip`.

##### Quản lý phiên bản Python

```
3.13
```

File `.python-version` chỉ định phiên bản Python sử dụng trong dự án. File này được các công cụ như `pyenv`, `uv` tự động nhận biết để sử dụng đúng phiên bản Python.

##### Định nghĩa dependency (pyproject.toml)

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

**Giải thích các gói chính:**

| Gói | Phiên bản | Vai trò |
|--------|------|------|
| `firecrawl-py` | 2.16 | Client API tìm kiếm và scraping web |
| `grandalf` | 0.8 | Trực quan hóa đồ thị (render đồ thị LangGraph) |
| `langchain[openai]` | 0.3.27 | Framework LangChain + tích hợp OpenAI |
| `langgraph` | 0.6.6 | Framework đồ thị agent dựa trên trạng thái |
| `langgraph-checkpoint-sqlite` | 2.0.11 | Kho lưu trữ checkpoint dựa trên SQLite |
| `langgraph-cli[inmem]` | 0.4.0 | LangGraph CLI (bao gồm chế độ in-memory) |
| `langgraph-supervisor` | 0.0.29 | Mẫu agent supervisor |
| `langgraph-swarm` | 0.0.14 | Mẫu agent Swarm |
| `pytest` | 8.4.2 | Framework test |
| `python-dotenv` | 1.1.1 | Tải biến môi trường từ file `.env` |

##### langgraph-supervisor vs langgraph-swarm

Dự án này cài đặt cả hai thư viện mẫu multi-agent:
- **Mẫu Supervisor**: Quản lý trung tâm chỉ huy các agent
- **Mẫu Swarm**: Các agent tự chủ chuyển giao tác vụ cho nhau

Dự án này sử dụng **mẫu gần với Swarm**, vì mỗi agent trực tiếp chuyển đổi sang agent khác thông qua tool `transfer_to_agent`.

#### Điểm thực hành

1. Tạo dự án bằng `uv init tutor-agent` rồi sửa `pyproject.toml` để thêm dependency.
2. Chạy `uv sync` để cài đặt tất cả dependency.
3. Tạo file `.env` và thiết lập `OPENAI_API_KEY` và `FIRECRAWL_API_KEY`.

---

### 2.2 Phần 19.1 -- Classification Agent (Agent phân loại người học)

**Commit**: `269599b` "19.1 Classification Agent"

#### Chủ đề và mục tiêu

Triển khai **agent phân loại** nắm bắt trình độ, phong cách học tập và mục tiêu học tập của người học để kết nối với agent học tập tối ưu. Agent này đóng vai trò **điểm vào (Entry Point)** của toàn bộ hệ thống.

#### Giải thích khái niệm cốt lõi

##### Hiểu create_react_agent

`create_react_agent` của LangGraph là hàm tạo agent theo **mẫu ReAct (Reasoning + Acting)** một cách thuận tiện.

```python
from langgraph.prebuilt import create_react_agent

classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="...",       # Prompt hệ thống
    tools=[...],        # Danh sách tool có thể sử dụng
)
```

**Mẫu ReAct là gì?**
- **Reasoning**: LLM phân tích tình huống hiện tại và quyết định hành động tiếp theo
- **Acting**: Gọi tool hoặc tạo phản hồi theo quyết định
- Lặp lại hai bước này để hoàn thành tác vụ

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

Nguyên tắc thiết kế cốt lõi của prompt:

1. **Cấu trúc đánh giá 3 giai đoạn**: Nắm bắt chủ đề -> Sở thích học tập -> Mục tiêu học tập theo thứ tự có hệ thống
2. **Tránh quá tải**: "Don't overwhelm - max 2 questions at a time" -- tối đa 2 câu hỏi mỗi lần
3. **Tận dụng dấu hiệu ngầm**: Nếu người dùng sử dụng đúng thuật ngữ kỹ thuật, phán đoán có nền tảng nhất định

##### Logic đề xuất agent

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

Tiêu chí chuyển đổi sang mỗi agent:
- **quiz_agent**: Người học ưa thích Active Recall (gợi nhớ chủ động)
- **teacher_agent**: Người mới bắt đầu hoặc cần giải thích có hệ thống
- **feynman_agent**: Người học tuyên bố đã hiểu nhưng cần xác minh

##### Cheat code dành cho nhà phát triển

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to a random agent (quiz_agent, teacher_agent, or feynman_agent)
    for testing purposes using the transfer_to_agent tool.
```

Thêm tính năng nhập "GODMODE" để bỏ qua quy trình đánh giá và chuyển đổi ngay sang agent để tiện test. Đây là mẫu thực dụng cho việc test nhanh trong quá trình phát triển.

##### Tool transfer_to_agent (phiên bản ban đầu)

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

**Điểm quan trọng:** Trong phiên bản ban đầu này, logic chuyển đổi thực tế dựa trên `Command` đã bị **comment out**. Vì các node agent khác chưa được đăng ký trong đồ thị, nên triển khai dưới dạng stub chỉ trả về chuỗi. Đây là ví dụ tốt về chiến lược **phát triển tăng dần (Incremental Development)**.

##### Xây dựng đồ thị chính

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

**Phân tích code:**

1. **`load_dotenv()`**: Tải biến môi trường như API key từ file `.env`. **Bắt buộc gọi trước khi import** -- vì các module khác có thể tham chiếu biến môi trường tại thời điểm import.

2. **`TutorState(MessagesState)`**: Kế thừa `MessagesState` của LangGraph để tự động quản lý tin nhắn hội thoại. Tại thời điểm này, sử dụng `pass` không có trạng thái bổ sung.

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
- **graphs**: Đồ thị cần expose và điểm vào
- **env**: Đường dẫn file biến môi trường

#### Điểm thực hành

1. Chạy dev server bằng `langgraph dev` và kiểm tra đồ thị trong LangGraph Studio.
2. Trò chuyện với Classification Agent và test xem quy trình đánh giá diễn ra tự nhiên không.
3. Sửa prompt để thêm tiêu chí đánh giá khác (ví dụ: "người học trực quan vs người học thính giác").

---

### 2.3 Phần 19.2 -- Feynman Agent & Teacher Agent

**Commit**: `5c2dfa9` "19.2 Feynman Agent"

#### Chủ đề và mục tiêu

Trong phần này, triển khai hai agent học tập cốt lõi và hoàn thiện cơ chế chuyển đổi giữa các agent:
- **Teacher Agent**: Giáo dục từng bước có hệ thống
- **Feynman Agent**: Xác minh mức độ hiểu biết bằng kỹ thuật Feynman
- **Tool tìm kiếm web**: Tìm kiếm thông tin thời gian thực dựa trên Firecrawl
- **Chuyển đổi agent thực tế**: Định tuyến sử dụng đối tượng `Command`

#### Giải thích khái niệm cốt lõi

##### Feynman Agent -- Kỹ thuật học tập Feynman

Triển khai triết lý học tập của Richard Feynman thành AI agent. Nguyên tắc cốt lõi là **"Nếu không giải thích đơn giản được thì chưa hiểu"**.

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
|------|------|------|
| Step 1 | Research Phase | Tìm kiếm web để có thông tin chính xác về khái niệm |
| Step 2 | Request Simple Explanation | Yêu cầu "giải thích như cho trẻ 8 tuổi" |
| Step 3 | Get User Explanation | Lắng nghe và phân tích giải thích của người dùng |
| Step 4 | Evaluate Complexity | Đánh giá thuật ngữ chuyên môn, lỗ hổng logic, giải thích mơ hồ |
| Step 5 | Ask Clarifying Questions | Đặt câu hỏi cụ thể về phần phức tạp |
| Step 6 | Complete | Nếu đủ ngắn gọn, công nhận mastery |

**Tiêu chí đánh giá cốt lõi:**

```
    ## Your Evaluation Criteria:
    - No unexplained technical terms
    - Clear cause-and-effect relationships
    - Uses analogies or examples a child would understand
    - Logical flow without gaps
    - Their own words, not memorized definitions
```

Tiêu chí này được sử dụng để phân biệt người học chỉ thuộc lòng định nghĩa hay thực sự hiểu. Đặc biệt, việc nhấn mạnh "lời nói của chính mình (Their own words)" rất quan trọng.

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

**Phương pháp giáo dục của Teacher Agent:**

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

**Quy tắc giáo dục cốt lõi của Teacher Agent:**

```
    ## Critical Teaching Rules:
    1. Always confirm understanding before moving to the next concept
    2. If they don't understand, explain differently (not just repeat)
    3. Break complex topics into the smallest possible pieces
    4. Use examples from their world and experience
    5. Be patient - true understanding takes time
```

Quy tắc số 2 đặc biệt quan trọng -- khi không hiểu, không phải lặp lại cùng cách giải thích mà phải **giải thích theo cách khác**. Đây cũng là năng lực cốt lõi của giáo viên giỏi thực sự.

##### Tool tìm kiếm web (web_search_tool)

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

**Phân tích code:**

1. **FirecrawlApp**: Sử dụng API Firecrawl để tìm kiếm web. Dịch vụ kết hợp Google Search + scraping trang.
2. **`limit=5`**: Giới hạn kết quả tìm kiếm còn 5 để tiết kiệm chi phí token.
3. **`ScrapeOptions(formats=["markdown"])`**: Nhận kết quả dạng markdown để LLM dễ xử lý.
4. **Làm sạch văn bản (Cleaning)**:
   - `re.sub(r"\\+|\n+", "", markdown)`: Xóa ký tự escape không cần thiết và xuống dòng quá mức
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: Xóa link markdown và URL

Quá trình làm sạch này rất quan trọng để **tiết kiệm token** và **giảm nhiễu**. Markdown gốc từ trang web chứa nhiều thông tin không cần thiết như link điều hướng, link quảng cáo.

##### Hoàn thiện tool transfer_to_agent (sử dụng Command)

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

Triển khai stub ở phần trước đã được **thay đổi thành logic chuyển đổi thực tế dựa trên `Command`**.

**Các tham số của đối tượng `Command`:**

| Tham số | Giá trị | Mô tả |
|---------|---|------|
| `goto` | `agent_name` | Tên node đích cần di chuyển |
| `graph` | `Command.PARENT` | Có nghĩa là tìm node ở đồ thị cha |
| `update` | `{"current_agent": agent_name}` | Cập nhật trạng thái đồ thị |

**Tại sao cần `Command.PARENT`:**
`transfer_to_agent` được gọi bên trong tool. Tool được thực thi trong context con của agent, nên để di chuyển đến node khác ở cấp cao nhất của đồ thị, cần chỉ định `Command.PARENT` để chỉ rõ đây là chuyển đổi ở cấp đồ thị cha.

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

**Phân tích code:**

1. **Thêm trường `current_agent` vào `TutorState`**: Theo dõi agent nào đang hoạt động.

2. **Hàm `router_check`**: Khi hội thoại được tiếp tục (tin nhắn mới đến), kiểm tra giá trị `current_agent` của trạng thái hiện tại để định tuyến đến node phù hợp. Giá trị mặc định là `"classification_agent"`.

3. **Tham số `destinations`**: Chỉ định `destinations` cho node `classification_agent` để cho LangGraph biết các node khác có thể đến từ node này. Cần thiết để chuyển đổi agent sử dụng `Command` hoạt động bình thường.

4. **`add_conditional_edges`**: Thêm phân nhánh có điều kiện từ `START`. Vào một trong ba agent tùy theo giá trị trả về của hàm `router_check`.

**Luồng định tuyến:**
```
Khi bắt đầu hội thoại mới:
  current_agent chưa thiết lập → giá trị mặc định router_check → "classification_agent"

Khi classification_agent gọi transfer_to_agent("teacher_agent"):
  Command(goto="teacher_agent", update={"current_agent": "teacher_agent"})
  → chuyển sang teacher_agent

Tin nhắn tiếp theo:
  current_agent = "teacher_agent" → router_check → đi thẳng đến teacher_agent
```

#### Điểm thực hành

1. Thử giải thích khái niệm mình biết rõ cho Feynman Agent và xem nhận được phản hồi gì.
2. Yêu cầu Teacher Agent dạy chủ đề mới để trải nghiệm quy trình giáo dục từng bước.
3. Sửa regex của `web_search_tool` để thử nghiệm chiến lược làm sạch khác.
4. Thêm logging vào hàm `router_check` để theo dõi quá trình định tuyến.

---

### 2.4 Phần 19.3 -- Quiz Agent (Agent quiz)

**Commit**: `e188909` "19.3 Quiz Agent"

#### Chủ đề và mục tiêu

Triển khai **Quiz Agent** tạo động quiz trắc nghiệm có cấu trúc dựa trên kết quả tìm kiếm web. Sử dụng **Structured Output** của Pydantic để LLM tạo quiz theo định dạng cố định.

#### Giải thích khái niệm cốt lõi

##### Đầu ra có cấu trúc dựa trên Pydantic (Structured Output)

Khái niệm kỹ thuật quan trọng nhất trong phần này là **Structured Output**. Yêu cầu LLM tạo dữ liệu theo schema đã định nghĩa trước thay vì văn bản tự do.

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

**Cốt lõi thiết kế schema:**

1. **Model `Question`**: Định nghĩa nghiêm ngặt cấu trúc mỗi câu hỏi
   - `question`: Văn bản câu hỏi
   - `options`: Chính xác 4 lựa chọn (A, B, C, D)
   - `correct_answer`: Đáp án đúng (bắt buộc khớp với một trong các options)
   - `explanation`: Giải thích lý do đáp án đúng và lý do đáp án sai

2. **Model `Quiz`**: Cấu trúc tổng thể quiz
   - `topic`: Chủ đề quiz
   - `questions`: Danh sách đối tượng Question

3. **`Field(description=...)`**: Truyền mô tả mỗi trường cho LLM để tạo đầu ra chính xác hơn. Đặc biệt, việc chỉ rõ ràng buộc như `"MUST MATCH ONE OF 'options'"` rất quan trọng.

##### Tool generate_quiz

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

**Phân tích code:**

1. **`init_chat_model("openai:gpt-4o")`**: Hàm khởi tạo model phổ quát của LangChain. Chỉ định provider và model bằng chuỗi.

2. **`model.with_structured_output(Quiz)`**: Đây là cốt lõi. Chuyển đổi model chat thông thường thành **model đầu ra có cấu trúc**. Sử dụng JSON mode / function calling của OpenAI bên trong để đảm bảo đầu ra khớp model Pydantic `Quiz`.

3. **`Literal["easy", "medium", "hard"]`**: Sử dụng type hint Python để giới hạn tham số difficulty thành 3 giá trị. LLM nhận biết ràng buộc này khi gọi tool.

4. **Tag XML `<RESEARCH_INFORMATION>`**: Sử dụng tag XML trong prompt để phân tách rõ ràng dữ liệu nghiên cứu. Giúp LLM không nhầm lẫn giữa chỉ thị prompt và dữ liệu.

**Mẫu gọi LLM bên trong tool:**
`generate_quiz` là tool nhưng bên trong lại gọi LLM. Đây là biến thể của mẫu **"Agent as Tool"**. Khi agent bên ngoài (Quiz Agent) gọi tool này, LLM bên trong tool tạo quiz có cấu trúc và trả về. Nhờ đó, logic tạo quiz được đóng gói gọn gàng.

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

**Mẫu ép buộc workflow trong prompt:**

```
    ## CRITICAL WORKFLOW - MUST FOLLOW IN ORDER:
    1. STEP 1: RESEARCH FIRST - You MUST use web_search_tool before anything else
    2. STEP 2: ASK LENGTH - Ask student how many questions they want
    3. STEP 3: CALL generate_quiz - Pass the research_text from step 1
    4. STEP 4: PRESENT ONE BY ONE - Show questions individually
    5. STEP 5: USE EXPLANATIONS - Use the explanations provided by the quiz tool

    NEVER call generate_quiz without research_text from web_search_tool first!
```

Phần này là **prompt engineering để kiểm soát nghiêm ngặt hành vi LLM**. Sử dụng các biểu thức nhấn mạnh như "CRITICAL", "MUST", "NEVER" và số thứ tự để LLM tuân theo thứ tự đã định. Đặc biệt, nếu tạo quiz không qua tìm kiếm web thì có thể chứa thông tin sai, nên bắt buộc phải nghiên cứu trước.

##### Tích hợp Quiz Agent vào đồ thị chính

```python
from agents.quiz_agent import quiz_agent

# ... thêm vào code hiện có ...
graph_builder.add_node("quiz_agent", quiz_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
        "quiz_agent",       # được thêm
    ],
)
```

Đăng ký node quiz_agent vào đồ thị và thêm vào danh sách edge có điều kiện của `router_check`.

##### Cập nhật Classification Agent

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to quiz_agent for testing purposes using the transfer_to_agent tool.
```

Cheat code GODMODE đã được thay đổi từ "agent ngẫu nhiên" sang **cố định `quiz_agent`**. Thay đổi này để tập trung test Quiz Agent mới được thêm.

##### Cập nhật shared_tools.py

```python
    agent_name: Name of the agent to transfer to, one of:
                'quiz_agent', 'teacher_agent' or 'feynman_agent'
```

`quiz_agent` đã được thêm lại vào docstring của `transfer_to_agent`. Vì LLM đọc docstring để quyết định giá trị tham số tool, nên việc liệt kê chính xác tên agent có thể ở đây rất quan trọng.

#### Điểm thực hành

1. Yêu cầu Quiz Agent tạo quiz về chủ đề cụ thể và đánh giá chất lượng quiz được tạo.
2. Thay đổi tham số `difficulty` của `generate_quiz` để xác nhận sự khác biệt câu hỏi theo độ khó.
3. Thêm trường `hint` vào model `Question` để triển khai tính năng gợi ý.
4. Thiết kế tính năng lưu kết quả quiz vào trạng thái để theo dõi lịch sử học tập.

---

## 3. Tổng kết chương

### Mẫu kiến trúc

1. **Mẫu Multi-Agent Swarm**: Các agent sử dụng đối tượng `Command` để tự chủ chuyển giao tác vụ cho nhau. Mỗi agent tự phán đoán và chuyển đổi sang agent phù hợp mà không cần quản lý trung tâm.

2. **Mẫu định tuyến có điều kiện**: Kết hợp hàm `router_check` và `add_conditional_edges` để tự động định tuyến đến agent phù hợp theo trạng thái hội thoại.

3. **Chiến lược phát triển tăng dần**: Phát triển theo thứ tự stub -> triển khai thực tế (tạo `transfer_to_agent` dạng stub ở 19.1 và hoàn thiện ở 19.2).

### Prompt Engineering

4. **Prompt quy trình từng bước**: Tất cả agent sử dụng prompt có hệ thống với các bước rõ ràng (Step 1, Step 2...). Điều này giúp hành vi LLM trở nên dự đoán được.

5. **Persona dựa trên vai trò**: Gán vai trò cụ thể cho mỗi agent như "Educational Assessment Specialist", "Master Teacher", "Feynman Technique Master", "Quiz Master" để giới hạn phạm vi hành vi.

6. **Mẫu workflow ép buộc**: Thiết kế prompt sử dụng các biểu thức nhấn mạnh "CRITICAL", "MUST", "NEVER" để LLM tuân theo thứ tự đã định.

### Thiết kế tool

7. **Structured Output**: Sử dụng model Pydantic + `with_structured_output()` để chuyển đổi đầu ra LLM thành dữ liệu có cấu trúc xử lý được bằng lập trình.

8. **Mẫu gọi LLM bên trong tool**: Tool `generate_quiz` gọi LLM riêng bên trong để tạo quiz có cấu trúc. Đây là ví dụ tốt về đóng gói logic.

9. **Làm sạch kết quả tìm kiếm web**: Sử dụng regex để xóa link, URL, ký tự escape không cần thiết, tiết kiệm token và nâng cao chất lượng đầu vào LLM.

### API cốt lõi LangGraph

| API | Mục đích |
|-----|------|
| `create_react_agent()` | Tạo agent mẫu ReAct |
| `StateGraph` | Định nghĩa đồ thị dựa trên trạng thái |
| `MessagesState` | Quản lý trạng thái dựa trên tin nhắn |
| `Command(goto, graph, update)` | Chuyển đổi agent trong đồ thị |
| `Command.PARENT` | Chuyển đổi ở cấp đồ thị cha |
| `add_conditional_edges()` | Thêm edge phân nhánh có điều kiện |
| `add_node(destinations=...)` | Chỉ rõ đích có thể đến từ node |

---

## 4. Bài tập thực hành

### Bài 1: Thêm agent mới (Cơ bản)

Tạo **Flashcard Agent** và thêm vào hệ thống.

- Tạo `agents/flashcard_agent.py`
- Tìm kiếm web về chủ đề học tập rồi tạo flashcard (mặt trước: câu hỏi, mặt sau: câu trả lời)
- Định nghĩa schema `Flashcard` và `FlashcardDeck` bằng model Pydantic
- Thêm node vào đồ thị trong `main.py` và cấu hình định tuyến
- Thêm điều kiện chuyển đổi sang flashcard_agent vào prompt `classification_agent`

### Bài 2: Theo dõi lịch sử học tập (Trung cấp)

Mở rộng `TutorState` để triển khai tính năng theo dõi lịch sử học tập.

- Thêm trường `quiz_scores: list[dict]`, `topics_learned: list[str]`, `current_topic: str` vào `TutorState`
- Sửa Quiz Agent để lưu kết quả quiz (điểm, chủ đề, ngày) vào trạng thái
- Sửa Teacher Agent để tham chiếu chủ đề đã học trước đó và đề xuất khái niệm liên quan
- Triển khai tool `progress_report` tóm tắt lịch sử học tập

### Bài 3: Điều chỉnh độ khó thích ứng (Nâng cao)

Triển khai hệ thống tự động điều chỉnh độ khó theo thành tích quiz của người học.

- Nếu tỷ lệ đúng liên tục trên 80% thì tăng một bậc độ khó
- Nếu tỷ lệ đúng liên tục dưới 50% thì tự động chuyển sang Teacher Agent
- Ghi lại lịch sử thay đổi độ khó vào trạng thái
- Trực quan hóa đường cong thay đổi độ khó bằng văn bản trong báo cáo học tập cuối cùng

### Bài 4: Chia sẻ context giữa các agent (Nâng cao)

Trong hệ thống hiện tại, kết quả đánh giá của agent trước không được truyền rõ ràng cho agent mới khi chuyển đổi.

- Thêm trường `learner_profile: dict` vào `TutorState` để lưu trữ có cấu trúc kết quả đánh giá của Classification Agent
- Sửa prompt để mỗi agent tham chiếu `learner_profile` và tạo phản hồi phù hợp trình độ người học
- Thêm lý do chuyển đổi (`transfer_reason`) vào tool `transfer_to_agent` để agent tiếp theo hiểu ngữ cảnh

### Bài 5: Phát triển tool tùy chỉnh (Chuyên sâu)

Tham khảo mẫu `generate_quiz` để triển khai các tool sau.

- **`generate_summary`**: Tool tạo bản tóm tắt học tập dựa trên kết quả tìm kiếm web. Định nghĩa schema `Section`, `Summary` bằng model Pydantic.
- **`evaluate_explanation`**: Tool đánh giá tự động có thể sử dụng trong Feynman Agent. Phân tích giải thích của người học để chấm điểm 0-10 về việc sử dụng thuật ngữ chuyên môn, tính nhất quán logic, sự ngắn gọn.

---

> **Lưu ý**: Để chạy code của chương này, cần biến môi trường `OPENAI_API_KEY` và `FIRECRAWL_API_KEY`. Thiết lập trong file `.env` hoặc chỉ định bằng `export` trong terminal. Để xem đồ thị trực quan trong LangGraph Studio, sử dụng lệnh `langgraph dev`.
