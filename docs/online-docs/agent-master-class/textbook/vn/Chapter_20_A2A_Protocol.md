# Chapter 20: Giao thức A2A (Agent-to-Agent)

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ tìm hiểu về **giao thức A2A (Agent-to-Agent)**. Giao thức A2A là một giao thức tiêu chuẩn mở cho phép các AI agent khác nhau **giao tiếp qua mạng**. Thông qua giao thức này, các agent được xây dựng bằng các framework khác nhau (Google ADK, LangGraph, v.v.) có thể phối hợp như một hệ thống duy nhất.

### Mục tiêu học tập

- Hiểu khái niệm và sự cần thiết của giao thức A2A
- Học cách chuyển đổi agent thành A2A server bằng tiện ích A2A của Google ADK (`to_a2a`)
- Học cách kết nối agent từ xa làm sub-agent bằng `RemoteA2aAgent`
- Hiểu cấu trúc và vai trò của Agent Card
- Học cách triển khai trực tiếp giao thức A2A để biến agent LangGraph thành A2A server
- Xác nhận rằng giao tiếp giữa các agent là khả thi bất kể framework nào

### Cấu trúc dự án

```
a2a/
├── .python-version          # Python 3.13
├── pyproject.toml           # Định nghĩa dependency của dự án
├── remote_adk_agent/        # Agent ADK từ xa hoạt động như A2A server
│   └── agent.py
├── user-facing-agent/       # Root agent giao tiếp trực tiếp với người dùng
│   └── user_facing_agent/
│       ├── __init__.py
│       └── agent.py
└── langraph_agent/          # Agent A2A server dựa trên LangGraph
    ├── graph.py
    └── server.py
```

### Dependency chính

| Gói | Phiên bản | Mục đích |
|-----|-----------|----------|
| `google-adk[a2a]` | 1.15.1 | Google ADK + phần mở rộng A2A |
| `google-genai` | 1.40.0 | Google Generative AI |
| `langchain[openai]` | 0.3.27 | LangChain + tích hợp OpenAI |
| `langgraph` | 0.6.8 | State graph của LangGraph |
| `litellm` | 1.77.7 | Giao diện tích hợp LLM đa dạng |
| `fastapi[standard]` | 0.118.0 | Framework web server |
| `uvicorn` | 0.37.0 | ASGI server |
| `python-dotenv` | 1.1.1 | Quản lý biến môi trường |

---

## 2. Giải thích chi tiết từng phần

---

### 20.0 Introduction - Thiết lập ban đầu dự án

#### Chủ đề và mục tiêu

Cấu hình môi trường dự án để học giao thức A2A. Tạo dự án mới dựa trên Python 3.13 và cài đặt tất cả dependency cần thiết.

#### Giải thích khái niệm cốt lõi

**Giao thức A2A là gì?**

A2A (Agent-to-Agent) là một giao thức mở do Google dẫn đầu phát triển, cung cấp khả năng tương tác (interoperability) giữa các hệ thống AI agent khác nhau. Trước đây, giao tiếp giữa các agent chỉ có thể thực hiện trong một framework duy nhất, nhưng với A2A, các agent có thể trao đổi tin nhắn qua mạng **bất kể framework nào**.

Các thành phần cốt lõi của giao thức A2A bao gồm:

1. **Agent Card**: Tài liệu JSON chứa metadata của agent. Nó định nghĩa tên, mô tả, khả năng (capabilities), định dạng đầu vào/đầu ra được hỗ trợ, v.v. Có thể truy cập tại đường dẫn `/.well-known/agent-card.json`.
2. **Message**: Định dạng tiêu chuẩn cho tin nhắn trao đổi giữa các agent.
3. **Transport**: Phương thức truyền tin nhắn (ví dụ: JSON-RPC).

**Tại sao lại cần thư mục `a2a/` riêng biệt?**

Vì giao thức A2A yêu cầu chạy nhiều agent server độc lập, nên cần tạo cấu trúc dự án mới tách biệt khỏi dự án hiện có. Mỗi agent chạy độc lập trên một port riêng.

#### Phân tích code

```toml
# a2a/pyproject.toml
[project]
name = "a2a"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi[standard]==0.118.0",
    "google-adk[a2a]==1.15.1",
    "google-genai==1.40.0",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.8",
    "litellm==1.77.7",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

Những điểm cần lưu ý:
- `google-adk[a2a]`: Khi cài đặt Google ADK, bao gồm extras `[a2a]`. Điều này bao gồm các tiện ích liên quan đến A2A (`to_a2a`, `RemoteA2aAgent`, v.v.).
- `fastapi` và `uvicorn`: Framework web được sử dụng khi triển khai trực tiếp A2A protocol server.
- `litellm`: Thư viện cho phép sử dụng nhiều nhà cung cấp LLM khác nhau (OpenAI, Anthropic, Google, v.v.) thông qua một giao diện duy nhất.

#### Điểm thực hành

1. Khởi tạo dự án và cài đặt dependency bằng `uv`:
   ```bash
   cd a2a
   uv sync
   ```
2. Kiểm tra file `.python-version` để xác nhận Python 3.13 đã được chỉ định.
3. Tạo file `.env` và thiết lập các API key cần thiết (OpenAI, v.v.).

---

### 20.1 A2A Using ADK - Tạo A2A Agent với ADK

#### Chủ đề và mục tiêu

Học cách chuyển đổi agent ADK thông thường thành A2A protocol server bằng tiện ích `to_a2a` của Google ADK. Đồng thời, tạo "root agent" giao tiếp trực tiếp với người dùng.

#### Giải thích khái niệm cốt lõi

**Hai loại vai trò Agent**

Trong phần này, chúng ta tạo hai loại agent:

1. **Agent từ xa (Remote Agent)**: Hoạt động như A2A server và cung cấp kiến thức chuyên môn trong một lĩnh vực cụ thể (ví dụ: lịch sử). Chạy độc lập trên một port riêng (8001).
2. **Agent hướng người dùng (User-Facing Agent)**: Giao tiếp trực tiếp với người dùng và ủy quyền tác vụ cho agent từ xa khi cần thiết.

**Hàm `to_a2a`**

Hàm `to_a2a` từ module `google.adk.a2a.utils.agent_to_a2a` chuyển đổi đối tượng ADK `Agent` thông thường thành ứng dụng web hỗ trợ giao thức A2A. Những gì hàm này thực hiện:
- Tự động tạo Agent Card và hiển thị tại `/.well-known/agent-card.json`
- Tạo endpoint nhận tin nhắn dựa trên JSON-RPC
- Chuyển đổi phản hồi của agent sang định dạng giao thức A2A

**LiteLlm Model**

`LiteLlm` là adapter cho phép sử dụng model từ nhiều nhà cung cấp LLM khác nhau (OpenAI, Anthropic, v.v.) trong Google ADK. Nó được chỉ định theo định dạng `nhà_cung_cấp/tên_model`, ví dụ `"openai/gpt-4o"`.

#### Phân tích code

**Agent ADK từ xa (A2A Server)**

```python
# a2a/remote_adk_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)

app = to_a2a(agent, port=8001)
```

Những điểm chính:
- `load_dotenv()` được gọi ngay sau import. Điều này là do các câu lệnh import sau đó có thể cần biến môi trường.
- Khi tạo `Agent`, `name` và `description` được chỉ định. Thông tin này tự động được bao gồm trong Agent Card.
- `to_a2a(agent, port=8001)` chuyển đổi agent thành ứng dụng A2A server. `port=8001` là số port mà server này sẽ sử dụng.
- Đối tượng `app` được trả về là ứng dụng ASGI, có thể chạy thông qua `uvicorn`.

**Agent hướng người dùng (Root Agent)**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)
```

Những điểm chính:
- Agent này không sử dụng `to_a2a`. Nó được chạy trực tiếp thông qua giao diện web của Google ADK (`adk web`).
- `sub_agents=[]` nghĩa là chưa có sub-agent nào được kết nối. Agent từ xa sẽ được kết nối trong các phần sau.
- Lưu ý tên biến là `root_agent`. Đây là quy ước được giao diện web ADK nhận diện.

**File `__init__.py`**

```python
# a2a/user-facing-agent/user_facing_agent/__init__.py
from . import agent
```

File này cho phép ADK tự động tải module `agent` trong package.

#### Điểm thực hành

1. Bắt đầu bằng cách chạy agent từ xa:
   ```bash
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001
   ```
2. Truy cập `http://localhost:8001/.well-known/agent-card.json` trong trình duyệt để xem Agent Card được tạo tự động.
3. Chạy agent hướng người dùng với giao diện web ADK:
   ```bash
   cd a2a
   adk web user-facing-agent
   ```

---

### 20.2 A2A For Dummies - Thêm công cụ cho Agent

#### Chủ đề và mục tiêu

Thêm công cụ (tool) vào agent từ xa để xác nhận rằng việc sử dụng công cụ qua A2A hoạt động chính xác. Đồng thời, học pattern sử dụng `tools` thay vì `sub_agents`.

#### Giải thích khái niệm cốt lõi

**Thêm công cụ vào Agent**

Trong giao thức A2A, agent từ xa có thể thực hiện các tác vụ phức tạp bằng công cụ, không chỉ là phản hồi văn bản đơn giản. Agent có công cụ có thể gọi chúng theo yêu cầu của người dùng và tạo phản hồi dựa trên kết quả.

**Thay đổi từ `sub_agents` sang `tools`**

Danh sách trống trước đó (`sub_agents=[]`) đã được thay đổi thành `tools=[dummy_tool]`. Điều này làm rõ vai trò của agent:
- `sub_agents`: Ủy quyền tác vụ cho các agent khác
- `tools`: Công cụ dạng hàm mà agent có thể sử dụng trực tiếp

#### Phân tích code

```python
# a2a/remote_adk_agent/agent.py (phần đã sửa đổi)
def dummy_tool(hello: str):
    """Dummy Tool. Helps the agent"""
    return "world"


agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    tools=[dummy_tool],
)

app = to_a2a(agent, port=8001)
```

Những điểm chính:
- `dummy_tool` là một hàm Python đơn giản. ADK tự động phân tích **tên hàm**, **type hint của tham số**, và **docstring** để tạo định nghĩa công cụ mà LLM có thể hiểu.
- Type hint trên tham số `hello: str` là bắt buộc. LLM cần biết loại giá trị nào cần truyền vào.
- Docstring `"""Dummy Tool. Helps the agent"""` giải thích mục đích của công cụ này cho LLM.
- `tools=[dummy_tool]` đăng ký công cụ với agent. Agent có thể tự động gọi công cụ này khi cần.

**Luồng gọi công cụ qua A2A**

```
Người dùng → Root Agent → [Giao thức A2A] → Agent từ xa → Gọi dummy_tool → Trả về phản hồi
```

Điểm quan trọng là việc gọi công cụ xảy ra **bên trong agent server từ xa**. Root agent không cần biết agent từ xa sử dụng công cụ gì. Giao thức A2A **đóng gói việc triển khai nội bộ** của agent.

#### Điểm thực hành

1. Thay thế `dummy_tool` bằng công cụ hữu ích thực sự (ví dụ: tìm kiếm Wikipedia, tính ngày, v.v.).
2. Sửa đổi docstring của công cụ và quan sát cách pattern gọi công cụ của LLM thay đổi.
3. Đăng ký nhiều công cụ như `tools=[tool1, tool2, tool3]`.

---

### 20.3 RemoteA2aAgent - Kết nối Agent từ xa

#### Chủ đề và mục tiêu

Học cách sử dụng `RemoteA2aAgent` để root agent có thể sử dụng các agent trên A2A server từ xa làm sub-agent. Đây là **pattern sử dụng cốt lõi** của giao thức A2A.

#### Giải thích khái niệm cốt lõi

**RemoteA2aAgent là gì?**

`RemoteA2aAgent` là một class do Google ADK cung cấp, cho phép sử dụng các agent đang chạy trên A2A server từ xa như thể chúng là sub-agent cục bộ. Bên trong, nó trao đổi tin nhắn giao thức A2A qua HTTP, nhưng từ góc nhìn người sử dụng, nó có thể được xử lý giống như sub-agent thông thường.

**AGENT_CARD_WELL_KNOWN_PATH**

Theo tiêu chuẩn giao thức A2A, metadata của agent (Agent Card) phải nằm tại đường dẫn `/.well-known/agent-card.json`. Hằng số `AGENT_CARD_WELL_KNOWN_PATH` chính là chuỗi đường dẫn này. Thông qua đó, `RemoteA2aAgent` có thể tự động lấy thông tin của agent từ xa (tên, mô tả, khả năng, URL nhận tin nhắn, v.v.).

**Pattern ủy quyền Agent (Delegation)**

```
Người dùng: "Hãy kể cho tôi về Napoleon"
    ↓
Root Agent (StudentHelperAgent)
    ↓ Xác định đây là câu hỏi liên quan đến lịch sử dựa trên description
    ↓ Ủy quyền cho history_agent
    ↓
[Giao thức A2A - Giao tiếp HTTP]
    ↓
Agent từ xa (HistoryHelperAgent, port 8001)
    ↓ Tạo phản hồi
    ↓
[Giao thức A2A - Phản hồi HTTP]
    ↓
Root Agent → Chuyển phản hồi đến người dùng
```

Root agent quyết định ủy quyền tác vụ cho agent nào dựa trên `description` của mỗi sub-agent. Do đó, việc viết mô tả rõ ràng và cụ thể là cực kỳ quan trọng.

#### Phân tích code

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

history_agent = RemoteA2aAgent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    agent_card=f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
    ],
)
```

Những điểm chính:

1. **Tạo RemoteA2aAgent**:
   - `name`: Tên của agent từ xa. Phải khớp với tên đã đăng ký trên server từ xa.
   - `description`: Mô tả mà root agent tham chiếu khi đưa ra quyết định ủy quyền.
   - `agent_card`: URL đầy đủ của Agent Card. `f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}"` trở thành `http://127.0.0.1:8001/.well-known/agent-card.json`.

2. **Kết nối vào sub_agents**:
   - `sub_agents=[history_agent]` thêm agent từ xa vào danh sách sub-agent.
   - Root agent giờ có thể ủy quyền câu hỏi liên quan đến lịch sử cho `history_agent` thông qua giao thức A2A.

3. **Trừu tượng hóa trong suốt**:
   - `RemoteA2aAgent` hoạt động như subclass của `Agent`. Từ góc nhìn root agent, không có sự khác biệt giữa agent cục bộ và agent từ xa.
   - Các tác vụ phức tạp như giao tiếp mạng và chuyển đổi giao thức được `RemoteA2aAgent` xử lý nội bộ.

#### Điểm thực hành

1. Mở hai terminal và chạy agent từ xa và root agent tương ứng:
   ```bash
   # Terminal 1: Agent từ xa
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # Terminal 2: Root agent
   cd a2a
   adk web user-facing-agent
   ```
2. Trong giao diện web ADK, thử đặt câu hỏi liên quan đến lịch sử và câu hỏi chung. Quan sát câu hỏi nào được ủy quyền cho agent từ xa.
3. Sửa đổi `description` và thử nghiệm cách quyết định ủy quyền thay đổi.

---

### 20.5 SendMessageResponse - Triển khai trực tiếp A2A Server với LangGraph

#### Chủ đề và mục tiêu

Học cách làm cho agent được xây dựng bằng **LangGraph** (không phải Google ADK) hoạt động như A2A server. Bằng cách **triển khai trực tiếp** giao thức A2A mà không cần tiện ích `to_a2a`, chúng ta hiểu sâu về hoạt động nội bộ của giao thức. Qua đó, xác nhận rằng **bất kỳ agent nào, bất kể framework**, đều có thể hỗ trợ giao thức A2A.

#### Giải thích khái niệm cốt lõi

**Xây dựng Agent với LangGraph**

LangGraph là framework đồ thị dựa trên trạng thái thuộc hệ sinh thái LangChain. Nó sử dụng `StateGraph` để định nghĩa luồng thực thi của agent dưới dạng đồ thị. Mỗi node là một bước xử lý, và các cạnh biểu thị hướng luồng.

**Triển khai trực tiếp giao thức A2A**

Để tạo A2A server mà không cần `to_a2a`, bạn cần triển khai hai endpoint sau:

1. **`GET /.well-known/agent-card.json`**: Endpoint trả về Agent Card. Cung cấp metadata của agent ở định dạng JSON.
2. **`POST /messages`**: Endpoint nhận tin nhắn và trả về phản hồi. Chấp nhận và xử lý yêu cầu ở định dạng JSON-RPC.

**Cấu trúc Agent Card**

Agent Card đóng vai trò "danh thiếp" của agent trong giao thức A2A:

| Trường | Mô tả |
|--------|-------|
| `name` | Tên agent |
| `description` | Mô tả agent |
| `url` | URL để gửi tin nhắn |
| `protocolVersion` | Phiên bản giao thức A2A |
| `capabilities` | Khả năng của agent |
| `defaultInputModes` | Định dạng đầu vào được hỗ trợ |
| `defaultOutputModes` | Định dạng đầu ra được hỗ trợ |
| `skills` | Danh sách kỹ năng của agent |
| `preferredTransport` | Phương thức truyền ưu tiên |

**Cấu trúc SendMessageResponse**

Trong giao thức A2A, phản hồi tin nhắn tuân theo định dạng JSON-RPC. Đối tượng `result` trong phản hồi chứa các trường như `kind`, `message_id`, `role`, và `parts`.

#### Phân tích code

**Định nghĩa đồ thị LangGraph**

```python
# a2a/langraph_agent/graph.py
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph


llm = init_chat_model("openai:gpt-4o")


class ConversationState(MessagesState):
    pass


def call_model(state: ConversationState) -> ConversationState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(ConversationState)
graph_builder.add_node("llm", call_model)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()
```

Những điểm chính:
- `init_chat_model("openai:gpt-4o")`: Hàm khởi tạo model thống nhất của LangChain. Sử dụng định dạng `nhà_cung_cấp:tên_model` (lưu ý sự khác biệt với `nhà_cung_cấp/tên_model` của LiteLlm).
- `ConversationState(MessagesState)`: Kế thừa từ `MessagesState` để quản lý danh sách tin nhắn hội thoại như trạng thái.
- `call_model`: Hàm node gọi LLM và thêm phản hồi vào danh sách tin nhắn.
- Cấu trúc đồ thị rất đơn giản: `START → llm → END`. Khi tin nhắn người dùng đến, nó gọi LLM và kết thúc.
- `graph_builder.compile()` tạo đối tượng đồ thị có thể thực thi.

**Triển khai A2A Server (FastAPI)**

```python
# a2a/langraph_agent/server.py
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from graph import graph

app = FastAPI()


def run_graph(message: str):
    result = graph.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content


@app.get("/.well-known/agent-card.json")
def get_agent_card():
    return {
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "An agent that can help students with their philosophy homework",
        "name": "PhilosophyHelperAgent",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "skills": [
            {
                "description": "An agent that can help students with their philosophy homework",
                "id": "PhilosophyHelperAgent",
                "name": "model",
                "tags": ["llm"],
            },
        ],
        "supportsAuthenticatedExtendedCard": False,
        "url": "http://localhost:8002/messages",
        "version": "0.0.1",
    }


@app.post("/messages")
async def handle_message(req: Request):
    body = await req.json()
    messages = body.get("params").get("message").get("parts")
    messages.reverse()
    message_text = ""
    for message in messages:
        text = message.get("text")
        message_text += f"{text}\n"
    response = run_graph(message_text)
    return {
        "id": "message_1",
        "jsonrpc": "2.0",
        "result": {
            "kind": "message",
            "message_id": "239827493847289374",
            "role": "agent",
            "parts": [
                {"kind": "text", "text": response},
            ],
        },
    }
```

Những điểm chính:

1. **Hàm `run_graph`**:
   - Hàm wrapper gọi đồ thị LangGraph.
   - Chuyển đổi tin nhắn người dùng sang định dạng `{"role": "user", "content": message}` và truyền vào đồ thị.
   - Trích xuất và trả về `content` của tin nhắn cuối cùng (phản hồi của LLM) từ kết quả.

2. **Endpoint Agent Card** (`GET /.well-known/agent-card.json`):
   - `protocolVersion: "0.3.0"`: Sử dụng giao thức A2A phiên bản 0.3.0.
   - `url: "http://localhost:8002/messages"`: Chỉ định URL để nhận tin nhắn. `RemoteA2aAgent` gửi tin nhắn đến URL này.
   - `skills`: Định nghĩa danh sách kỹ năng mà agent này cung cấp.
   - `preferredTransport: "JSONRPC"`: Chỉ ra giao tiếp qua JSON-RPC.

3. **Endpoint xử lý tin nhắn** (`POST /messages`):
   - Phân tích yêu cầu JSON-RPC của giao thức A2A.
   - Cấu trúc yêu cầu: Trích xuất các phần tin nhắn từ đường dẫn `body.params.message.parts`.
   - `messages.reverse()`: Đảo ngược thứ tự các phần tin nhắn để tin nhắn mới nhất đến trước.
   - Trích xuất `text` từ mỗi phần và kết hợp thành một chuỗi duy nhất.
   - Phản hồi tuân theo định dạng JSON-RPC: `jsonrpc: "2.0"`, với phản hồi của agent nằm trong đối tượng `result`.
   - `result.parts` ở định dạng phần tin nhắn A2A, chứa `kind: "text"` và văn bản thực tế.

**Thêm Agent triết học vào Root Agent**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py (phần bổ sung)
philosophy_agent = RemoteA2aAgent(
    name="PhilosophyHelperAgent",
    description="An agent that can help students with their philosophy homework",
    agent_card=f"http://127.0.0.1:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
        philosophy_agent,
    ],
)
```

Những điểm chính:
- Agent triết học chạy trên port 8002.
- `sub_agents` của root agent giờ chứa hai agent từ xa.
- Root agent tự động ủy quyền cho agent lịch sử (port 8001) hoặc agent triết học (port 8002) dựa trên nội dung câu hỏi.
- **Điểm mấu chốt**: Agent lịch sử được xây dựng bằng ADK + `to_a2a`, và agent triết học được triển khai trực tiếp bằng LangGraph + FastAPI. Mặc dù sử dụng framework khác nhau, chúng giao tiếp theo cùng một cách nhờ giao thức A2A.

#### Điểm thực hành

1. Mở ba terminal và chạy tất cả agent đồng thời:
   ```bash
   # Terminal 1: Agent lịch sử (dựa trên ADK)
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # Terminal 2: Agent triết học (dựa trên LangGraph)
   cd a2a/langraph_agent
   uvicorn server:app --port 8002

   # Terminal 3: Root agent
   cd a2a
   adk web user-facing-agent
   ```
2. Thử nhiều câu hỏi khác nhau trong giao diện web ADK:
   - "Nguyên nhân của Thế chiến thứ hai là gì?" → Ủy quyền cho agent lịch sử
   - "Triết học của Socrates là gì?" → Ủy quyền cho agent triết học
   - "Hôm nay thời tiết thế nào?" → Root agent trả lời trực tiếp
3. Truy cập `http://localhost:8002/.well-known/agent-card.json` để xem Agent Card được triển khai trực tiếp.
4. Quan sát log của mỗi server để theo dõi quá trình truyền tin nhắn A2A thực tế.

---

## 3. Tóm tắt nội dung chính của chương

### Nguyên lý cốt lõi của giao thức A2A

| Khái niệm | Mô tả |
|------------|-------|
| **Giao thức A2A** | Tiêu chuẩn mở cho phép các agent từ framework khác nhau giao tiếp qua mạng |
| **Agent Card** | Tài liệu JSON chứa metadata của agent (`/.well-known/agent-card.json`) |
| **to_a2a** | Hàm tiện ích tự động chuyển đổi agent ADK thành A2A server |
| **RemoteA2aAgent** | Class ADK cho phép sử dụng agent trên A2A server từ xa như sub-agent cục bộ |
| **JSON-RPC** | Giao thức truyền thông được sử dụng cho trao đổi tin nhắn A2A |

### Pattern kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                   Người dùng (User)                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            Root Agent (StudentHelperAgent)               │
│            - Google ADK + LiteLlm                       │
│            - Chạy bằng adk web                          │
└──────────┬──────────────────────────────┬───────────────┘
           │ A2A (port 8001)              │ A2A (port 8002)
           ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────┐
│ Agent lịch sử         │    │ Agent triết học           │
│ (HistoryHelperAgent) │    │ (PhilosophyHelperAgent)  │
│ - Google ADK         │    │ - LangGraph + FastAPI    │
│ - Sử dụng to_a2a    │    │ - Triển khai A2A trực tiếp│
│ - port 8001          │    │ - port 8002              │
└──────────────────────┘    └──────────────────────────┘
```

### 5 Điểm mấu chốt

1. **Độc lập với Framework**: Sử dụng giao thức A2A, các agent được xây dựng bằng framework khác nhau như Google ADK, LangGraph, LangChain có thể phối hợp như một hệ thống duy nhất.

2. **Agent Card là bắt buộc**: Mọi agent A2A phải hiển thị metadata tại `/.well-known/agent-card.json`. Thông qua đó, các agent khác có thể tự động lấy thông tin kết nối.

3. **Chuyển đổi dễ dàng với `to_a2a`**: Nếu sử dụng Google ADK, bạn có thể chuyển đổi agent thành A2A server chỉ với một dòng `to_a2a()`. Agent Card và endpoint xử lý tin nhắn được tạo tự động.

4. **Triển khai trực tiếp cũng khả thi**: Vì giao thức A2A dựa trên HTTP + JSON-RPC tiêu chuẩn, nó có thể được triển khai trực tiếp với bất kỳ web framework nào. Bạn chỉ cần triển khai endpoint Agent Card và endpoint xử lý tin nhắn.

5. **Description là chìa khóa định tuyến**: Root agent quyết định ủy quyền tác vụ dựa trên `description` của mỗi sub-agent. Mô tả rõ ràng và cụ thể đảm bảo định tuyến chính xác.

---

## 4. Bài tập thực hành

### Bài tập 1: Thêm Agent chuyên gia mới (Cơ bản)

Tạo `MathHelperAgent` giúp đỡ bài tập toán bằng Google ADK + `to_a2a`, được cấu hình để chạy trên port 8003. Thêm vào `sub_agents` của root agent để ba agent chuyên gia phối hợp cùng nhau.

**Yêu cầu:**
- Tên agent: `MathHelperAgent`
- Port: 8003
- Thêm ít nhất 1 công cụ liên quan đến toán (ví dụ: hàm máy tính)
- Xác nhận câu hỏi toán được ủy quyền cho agent này từ root agent

### Bài tập 2: Mở rộng A2A Server dựa trên LangGraph (Trung bình)

Mở rộng agent triết học dựa trên LangGraph được tạo trong phần 20.5 để thêm các tính năng sau:

**Yêu cầu:**
- Quản lý lịch sử hội thoại: Triển khai khả năng ghi nhớ nội dung hội thoại trước đó qua nhiều lần trao đổi tin nhắn
- Xử lý lỗi: Trả về phản hồi lỗi phù hợp cho định dạng yêu cầu không hợp lệ
- Sử dụng trường `capabilities` trong Agent Card để chỉ định các tính năng được hỗ trợ

**Gợi ý:**
- Bạn có thể sử dụng dictionary để lưu trữ lịch sử hội thoại theo phiên
- Tham khảo định dạng phản hồi lỗi JSON-RPC

### Bài tập 3: Xây dựng A2A Agent với Framework hoàn toàn khác (Nâng cao)

Triển khai A2A agent chỉ sử dụng FastAPI và gọi HTTP trực tiếp (ví dụ: gọi OpenAI API trực tiếp bằng `httpx`), không sử dụng bất kỳ AI framework nào. Chứng minh rằng giao thức A2A thực sự độc lập với framework.

**Yêu cầu:**
- Không sử dụng Google ADK, LangChain, LangGraph, v.v.
- Gọi OpenAI API trực tiếp bằng FastAPI + `httpx` (hoặc `requests`)
- Triển khai trực tiếp endpoint Agent Card và xử lý tin nhắn
- Xác nhận hoạt động chính xác bằng cách kết nối với `RemoteA2aAgent` của root agent

**Gợi ý:**
```python
import httpx

async def call_openai(message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]
```

### Bài tập 4: Xây dựng Agent Card Explorer (Nâng cao)

Tạo công cụ CLI lấy Agent Card từ URL cho trước, hiển thị dễ đọc, và gửi tin nhắn thử nghiệm đến agent đó.

**Yêu cầu:**
- Chạy theo định dạng `python explorer.py http://localhost:8001`
- Hiển thị tất cả trường của Agent Card ở định dạng dễ đọc
- Nhận đầu vào từ người dùng và gửi tin nhắn A2A đến agent, sau đó hiển thị phản hồi
- Xử lý yêu cầu/phản hồi theo định dạng JSON-RPC

---

> **Lưu ý**: Tất cả code trong chương này nằm trong thư mục `a2a/`. Bạn phải thiết lập các API key cần thiết trong file `.env` trước khi chạy. Cài đặt dependency bằng `uv sync`, sau đó chạy mỗi agent trong terminal riêng.
