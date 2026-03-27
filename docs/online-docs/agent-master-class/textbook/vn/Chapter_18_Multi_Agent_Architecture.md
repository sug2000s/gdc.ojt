# Chapter 18: Kiến trúc Đa Agent (Multi-Agent Architectures)

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ học cách thiết kế và triển khai **hệ thống đa agent nơi nhiều AI agent phối hợp với nhau** sử dụng LangGraph. Vượt qua giới hạn của agent đơn lẻ, chúng ta từng bước xây dựng các kiến trúc nơi các agent có vai trò chuyên biệt giao tiếp với nhau để xử lý các tác vụ phức tạp.

### Mục tiêu học tập

- Hiểu sự cần thiết và các khái niệm cốt lõi của hệ thống đa agent
- **Kiến trúc mạng**: Triển khai giao tiếp trực tiếp ngang hàng (P2P) giữa các agent
- **Kiến trúc Supervisor**: Triển khai hệ thống nơi một điều phối viên trung tâm quản lý các agent
- **Kiến trúc Supervisor-as-Tools**: Học mẫu nâng cao đóng gói agent thành công cụ
- **Prebuilt Agents**: Học cách triển khai ngắn gọn với thư viện `langgraph-supervisor`
- Học trực quan hóa đồ thị thông qua LangGraph Studio

### Cấu trúc dự án

```
multi-agent-architectures/
├── .python-version          # Python 3.13
├── pyproject.toml           # Định nghĩa dependency dự án
├── main.ipynb               # Notebook thực hành chính
├── graph.py                 # Định nghĩa đồ thị cho LangGraph Studio
├── langgraph.json           # Cấu hình LangGraph Studio
└── uv.lock                  # File khóa dependency
```

### Dependency chính

```toml
[project]
dependencies = [
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
```

---

## 2. Mô tả chi tiết từng phần

---

### 18.0 Introduction - Khởi tạo dự án và Import cơ bản

**Chủ đề và Mục tiêu**: Thiết lập nền tảng cho dự án đa agent và import các thư viện cốt lõi.

#### Các khái niệm chính

Để xây dựng hệ thống đa agent, bạn cần hiểu một số module cốt lõi của LangGraph. Trong phần này, chúng ta khởi tạo dự án và import tất cả các thư viện cần thiết.

#### Phân tích mã nguồn

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

Hãy xem vai trò của từng import:

| Module | Vai trò |
|--------|---------|
| `StateGraph` | Lớp cốt lõi để tạo đồ thị dựa trên trạng thái. Định nghĩa node và edge để cấu hình luồng thực thi của agent. |
| `START`, `END` | Hằng số node đặc biệt đại diện cho điểm bắt đầu và kết thúc của đồ thị. |
| `Command` | Đối tượng lệnh để điều khiển chuyển đổi (handoff) giữa các agent. Chỉ định node tiếp theo bằng `goto` và thay đổi trạng thái bằng `update`. |
| `MessagesState` | Lớp trạng thái được định nghĩa sẵn quản lý danh sách tin nhắn. Cung cấp key `messages` mặc định. |
| `ToolNode` | Node được xây dựng sẵn xử lý các lệnh gọi công cụ. Khi LLM quyết định sử dụng công cụ, node này xử lý việc thực thi thực tế. |
| `tools_condition` | Hàm định tuyến có điều kiện xác định phản hồi LLM có chứa lệnh gọi công cụ hay không. |
| `@tool` | Decorator chuyển đổi hàm Python thông thường thành công cụ mà LLM có thể gọi. |
| `init_chat_model` | Tiện ích khởi tạo các LLM khác nhau sử dụng chuỗi định dạng `"provider:model_name"`. |

#### Điểm thực hành

- Khởi tạo dự án và cài đặt dependency bằng `uv`: `uv sync`
- `OPENAI_API_KEY` phải được thiết lập trong file `.env`
- Cần môi trường Python 3.13

---

### 18.1 Network Architecture - Kiến trúc mạng

**Chủ đề và Mục tiêu**: Triển khai **kiến trúc mạng (P2P)** nơi các agent có thể trực tiếp chuyển giao (handoff) cuộc hội thoại cho nhau.

#### Các khái niệm chính

Trong kiến trúc mạng, **không có điều phối viên trung tâm**, mỗi agent tự quyết định chuyển giao cuộc hội thoại cho agent khác. Trong ví dụ này, chúng ta tạo các agent hỗ trợ khách hàng tiếng Hàn, tiếng Hy Lạp và tiếng Tây Ban Nha, xây dựng hệ thống tự động chuyển đến agent phù hợp với ngôn ngữ khách hàng.

```
┌──────────────┐     handoff     ┌──────────────┐
│ korean_agent │ ◄─────────────► │ greek_agent  │
└──────┬───────┘                 └──────┬───────┘
       │                                │
       │          handoff               │
       └────────►┌──────────────┐◄──────┘
                 │spanish_agent │
                 └──────────────┘
```

Trong cấu trúc này, mỗi agent:
1. Cố gắng phản hồi bằng ngôn ngữ mà nó hiểu
2. Phát hiện ngôn ngữ không quen thuộc và sử dụng `handoff_tool` để chuyển đến agent phù hợp

#### Phân tích mã nguồn

**Bước 1: Định nghĩa trạng thái**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str

llm = init_chat_model("openai:gpt-4o")
```

Trạng thái tùy chỉnh được định nghĩa bằng cách mở rộng `MessagesState` để theo dõi agent đang hoạt động (`current_agent`) và agent đã thực hiện chuyển giao (`transfered_by`). Điều này cho phép biết agent nào đang xử lý cuộc hội thoại và ai đã chuyển giao nó.

**Bước 2: Hàm tạo Agent (Factory Function)**

```python
def make_agent(prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node(
        "tools",
        ToolNode(tools=tools),
    )

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    return agent_builder.compile()
```

`make_agent` là **hàm tạo agent (factory function)** cho phép tạo các agent có cấu trúc giống nhau nhưng tham số khác nhau. Mỗi agent nội bộ tạo thành một đồ thị con `StateGraph` hoàn chỉnh:

- Node `agent`: Truyền prompt và lịch sử hội thoại cho LLM để tạo phản hồi
- Node `tools`: `ToolNode` thực thi các lệnh gọi công cụ
- `tools_condition`: Định tuyến đến node `tools` nếu LLM gọi công cụ, ngược lại đến `END`
- Edge `tools` -> `agent`: Truyền kết quả thực thi công cụ trở lại agent để suy luận thêm

Cấu trúc này cho mỗi agent một **vòng lặp ReAct (Reasoning + Acting) độc lập**.

**Bước 3: Định nghĩa công cụ Handoff**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    """
    Handoff to another agent.

    Use this tool when the customer speaks a language that you don't understand.

    Possible values for `transfer_to`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Possible values for `transfered_by`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Args:
        transfer_to: The agent to transfer the conversation to
        transfered_by: The agent that transferred the conversation
    """
    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

`handoff_tool` là **cơ chế cốt lõi** của kiến trúc mạng. Các điểm chính:

- Trả về đối tượng `Command` để trực tiếp điều khiển luồng thực thi của đồ thị
- `goto=transfer_to`: Di chuyển thực thi đến node agent được chỉ định
- `graph=Command.PARENT`: Thực hiện di chuyển ở **đồ thị cha**. Vì mỗi agent là đồ thị con, nếu không có tùy chọn này, di chuyển chỉ xảy ra trong đồ thị con. `Command.PARENT` cho phép chuyển đổi agent ở cấp đồ thị cao nhất
- `update` cập nhật trạng thái để theo dõi thông tin agent hiện tại
- Docstring của công cụ chỉ định các giá trị có thể để hướng dẫn LLM sử dụng giá trị đúng

**Bước 4: Lắp ráp đồ thị**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
)

graph_builder.add_edge(START, "korean_agent")

graph = graph_builder.compile()
```

Ba node agent được đăng ký trong đồ thị cấp cao nhất. Giá trị của mỗi node là đồ thị con đã biên dịch được trả về bởi `make_agent`. Vì `START -> korean_agent` được thiết lập, tất cả cuộc hội thoại bắt đầu tại agent tiếng Hàn. Nếu người dùng nói tiếng Tây Ban Nha, agent tiếng Hàn phát hiện điều này và sử dụng `handoff_tool` để chuyển đến agent tiếng Tây Ban Nha.

#### Điểm thực hành

- Thay đổi agent khởi đầu thành `spanish_agent` và gửi tin nhắn bằng tiếng Hàn để xem chuyển đổi tự động có xảy ra không
- Thêm agent ngôn ngữ mới (ví dụ: tiếng Nhật). Docstring của `handoff_tool` cũng phải được sửa đổi
- Thử nghiệm lỗi gì xảy ra khi bạn xóa `Command.PARENT`

---

### 18.2 Network Visualization - Trực quan hóa mạng

**Chủ đề và Mục tiêu**: Sử dụng LangGraph Studio để kiểm tra trực quan đồ thị đa agent, gỡ lỗi luồng thực thi và phòng chống lỗi vòng lặp vô hạn từ tự chuyển giao.

#### Các khái niệm chính

Khi phát triển hệ thống đa agent phức tạp, kiểm tra trực quan cấu trúc đồ thị rất quan trọng. LangGraph Studio là công cụ trực quan hóa node và edge của đồ thị, đồng thời theo dõi luồng thực thi thời gian thực.

Tuy nhiên, để trực quan hóa, bạn phải khai báo trước node nào đồ thị có thể di chuyển đến từ node nào. Trong định tuyến động với `Command`, điều này được chỉ định thông qua tham số `destinations`.

#### Phân tích mã nguồn

**Bước 1: Tách graph.py và thêm destinations**

Khi tách mã notebook thành `graph.py`, tham số `destinations` được thêm vào mỗi node agent:

```python
graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "spanish_agent"),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
    destinations=("korean_agent", "spanish_agent"),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "korean_agent"),
)
```

Tham số `destinations` khai báo các node đích có thể đến được qua `Command` từ node đó. Điều này **không ảnh hưởng đến logic thực thi** và được công cụ trực quan hóa của LangGraph Studio sử dụng để hiển thị cấu trúc đồ thị chính xác.

**Bước 2: Phòng chống lỗi tự chuyển giao**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    # ... (docstring bỏ qua)
    if transfer_to == transfered_by:
        return {
            "error": "Stop trying to transfer to yourself and answer the question or i will fire you."
        }

    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

Câu phòng thủ cũng được thêm vào prompt của agent:

```python
response = llm_with_tools.invoke(
    f"""
    {prompt}

    You have a tool called 'handoff_tool' use it to transfer to other agent,
    don't use it to transfer to yourself.

    Conversation History:
    {state["messages"]}
    """
)
```

LLM đôi khi cố gắng chuyển giao cho chính mình, gây ra **vòng lặp vô hạn**. Điều này được phòng thủ bằng hai lớp:
1. **Cấp prompt**: Chỉ thị rõ ràng "đừng chuyển giao cho chính mình"
2. **Cấp mã**: Kiểm tra `transfer_to == transfered_by` từ chối tự chuyển giao và trả về thông báo lỗi

**Bước 3: Cấu hình LangGraph Studio**

```json
{
    "dependencies": [
        "./graph.py"
    ],
    "graphs": {
        "agent": "./graph.py:graph"
    },
    "env": ".env"
}
```

File cấu hình `langgraph.json` cho LangGraph Studio biết:
- `dependencies`: Danh sách file Python cần thiết
- `graphs`: Vị trí đối tượng đồ thị cần trực quan hóa (định dạng `đường_dẫn_file:tên_biến`)
- `env`: Đường dẫn file biến môi trường

**Bước 4: Thực thi streaming**

```python
for event in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hola! Necesito ayuda con mi cuenta.",
            }
        ]
    },
    stream_mode="updates",
):
    print(event)
```

Sử dụng `stream_mode="updates"` cho phép nhận cập nhật trạng thái từ mỗi node theo thời gian thực. Trong kết quả thực thi, bạn có thể xác nhận toàn bộ luồng nơi agent tiếng Hàn phát hiện tiếng Tây Ban Nha, chuyển đến agent tiếng Tây Ban Nha, và agent tiếng Tây Ban Nha phản hồi bằng tiếng Tây Ban Nha:

```
{'korean_agent': {'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
{'spanish_agent': {'messages': [...], 'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
```

#### Điểm thực hành

- Thử chạy studio bằng LangGraph CLI: `langgraph dev`
- Mở studio với `destinations` đã xóa và so sánh trực quan hóa khác nhau như thế nào
- Gửi tin nhắn tiếng Hy Lạp và xác nhận luồng chuyển giao

---

### 18.3 Supervisor Architecture - Kiến trúc Supervisor

**Chủ đề và Mục tiêu**: Triển khai kiến trúc nơi node **Supervisor** trung tâm phân tích cuộc hội thoại và định tuyến đến agent phù hợp.

#### Các khái niệm chính

Trong kiến trúc mạng, mỗi agent tự quyết định chuyển giao. Kiến trúc supervisor khác ở chỗ **một điều phối viên trung tâm (Supervisor)** xử lý tất cả quyết định định tuyến.

```
                    ┌─────────────┐
          ┌────────►│  Supervisor │◄────────┐
          │         └──────┬──────┘         │
          │                │                │
          │         ┌──────┼──────┐         │
          │         ▼      ▼      ▼         │
    ┌─────┴──┐  ┌───┴───┐  ┌──┴──────┐     │
    │ korean │  │ greek │  │ spanish │─────┘
    │ _agent │  │_agent │  │ _agent  │
    └────────┘  └───────┘  └─────────┘
```

Ưu điểm của kiến trúc này:
- **Điểm vào duy nhất**: Tất cả yêu cầu đều đi qua supervisor, tập trung logic định tuyến
- **Tính nhất quán**: Các agent không cần lo về định tuyến và chỉ tập trung vào chuyên môn của mình
- **Dễ kiểm soát**: Chỉ cần sửa đổi prompt của supervisor có thể thay đổi toàn bộ chiến lược định tuyến

#### Phân tích mã nguồn

**Bước 1: Định nghĩa mô hình đầu ra có cấu trúc**

```python
from typing import Literal
from pydantic import BaseModel

class SupervisorOutput(BaseModel):
    next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
    reasoning: str
```

Kết quả quyết định của supervisor được định nghĩa dưới dạng mô hình `Pydantic`:
- `next_agent`: Sử dụng kiểu `Literal` để hạn chế các giá trị có thể, ép buộc LLM chỉ trả về tên agent hợp lệ. `"__end__"` nghĩa là kết thúc cuộc hội thoại.
- `reasoning`: Giải thích lý do supervisor chọn agent đó. Hữu ích cho gỡ lỗi và tính minh bạch.

**Bước 2: Mở rộng trạng thái**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
    reasoning: str
```

Trường `reasoning` được thêm vào trạng thái để theo dõi lý do quyết định của supervisor.

**Bước 3: Đơn giản hóa Agent**

```python
def make_agent(prompt, tools):
    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}
    # ... (cấu hình đồ thị giống nhau)
```

Khác với kiến trúc mạng, các chỉ thị liên quan đến `handoff_tool` được **loại bỏ** khỏi prompt agent. Vì supervisor xử lý định tuyến, agent chỉ cần phản hồi bằng ngôn ngữ của mình. Danh sách công cụ rỗng được truyền với `tools=[]`.

**Bước 4: Triển khai node Supervisor**

```python
def supervisor(state: AgentState):
    structured_llm = llm.with_structured_output(SupervisorOutput)
    response = structured_llm.invoke(
        f"""
        You are a supervisor that routes conversations to the appropriate language agent.

        Analyse the customers request and the conversation history and decide which
        agent should handle the conversation.

        The options for the next agent are:
        - greek_agent
        - spanish_agent
        - korean_agent

        <CONVERSATION_HISTORY>
        {state.get("messages", [])}
        </CONVERSATION_HISTORY>

        IMPORTANT:

        Never transfer to the same agent twice in a row.

        If an agent has replied end the conversation by returning __end__
    """
    )
    return Command(
        goto=response.next_agent,
        update={"reasoning": response.reasoning},
    )
```

Các cơ chế chính của supervisor:

- `llm.with_structured_output(SupervisorOutput)`: Ép buộc LLM trả về JSON khớp schema `SupervisorOutput`. Điều này cho phép định tuyến ổn định không có lỗi phân tích.
- Thẻ XML `<CONVERSATION_HISTORY>`: Phân tách rõ ràng lịch sử hội thoại để LLM nắm bắt chính xác ngữ cảnh.
- `"Never transfer to the same agent twice in a row"`: Ràng buộc prompt để ngăn vòng lặp vô hạn.
- `"If an agent has replied end the conversation by returning __end__"`: Kết thúc cuộc hội thoại nếu agent đã phản hồi, ngăn lặp lại không cần thiết.
- `Command(goto=response.next_agent)`: Vì supervisor là node đồ thị cấp cao nhất (không phải đồ thị con), `graph=Command.PARENT` không cần thiết.

**Bước 5: Lắp ráp đồ thị - Cấu trúc vòng**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "supervisor",
    supervisor,
    destinations=(
        "korean_agent",
        "spanish_agent",
        "greek_agent",
        END,
    ),
)

graph_builder.add_node("korean_agent", make_agent(
    prompt="You're a Korean customer support agent. You only speak and understand Korean.",
    tools=[],
))
graph_builder.add_node("greek_agent", make_agent(
    prompt="You're a Greek customer support agent. You only speak and understand Greek.",
    tools=[],
))
graph_builder.add_node("spanish_agent", make_agent(
    prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
    tools=[],
))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("korean_agent", "supervisor")
graph_builder.add_edge("spanish_agent", "supervisor")
graph_builder.add_edge("greek_agent", "supervisor")

graph = graph_builder.compile()
```

Tóm tắt luồng đồ thị:

1. `START` -> `supervisor`: Tất cả cuộc hội thoại bắt đầu tại supervisor
2. `supervisor` -> `{agent}` hoặc `END`: Supervisor định tuyến đến agent phù hợp hoặc kết thúc qua `Command`
3. `{agent}` -> `supervisor`: Sau khi agent hoàn thành phản hồi, quay lại supervisor

Cấu trúc vòng này là **cốt lõi của kiến trúc supervisor**. Supervisor có thể quyết định kết thúc cuộc hội thoại (`__end__`) hay định tuyến đến agent khác sau khi agent phản hồi.

#### So sánh Mạng vs Supervisor

| Đặc điểm | Mạng | Supervisor |
|-----------|------|------------|
| Quyết định định tuyến | Mỗi agent tự quyết | Supervisor trung tâm xử lý |
| Kết nối giữa agent | P2P (kết nối trực tiếp) | Hub-spoke (qua supervisor) |
| Độ phức tạp agent | Cao (bao gồm logic định tuyến) | Thấp (chỉ xử lý chuyên môn) |
| Khả năng mở rộng | Tất cả agent cần sửa đổi khi thêm mới | Chỉ supervisor cần sửa đổi |
| Gỡ lỗi | Khó | Dễ (có thể theo dõi reasoning) |

#### Điểm thực hành

- In trường `reasoning` để phân tích cơ sở supervisor sử dụng cho quyết định định tuyến
- So sánh mã nào cần sửa đổi khi thêm agent trong kiến trúc mạng vs kiến trúc supervisor
- Thử nghiệm điều gì xảy ra khi xóa tùy chọn `__end__` khỏi `SupervisorOutput`

---

### 18.4 Supervisor As Tools - Đóng gói Agent thành công cụ

**Chủ đề và Mục tiêu**: **Đóng gói agent thành công cụ LLM** để supervisor sử dụng agent một cách tự nhiên thông qua cơ chế gọi công cụ.

#### Các khái niệm chính

Trong kiến trúc supervisor ở phần trước, `structured_output` được sử dụng cho định tuyến. Trong phần này, chúng ta tái cấu trúc bằng cách **chuyển đổi mỗi agent thành công cụ** và supervisor gọi agent thông qua cơ chế `bind_tools` + `ToolNode`.

```
                ┌──────────────┐
    START ────► │  Supervisor  │ ────► END
                └──────┬───────┘
                       │
                 tools_condition
                       │
                ┌──────▼───────┐
                │   ToolNode   │
                │ ┌──────────┐ │
                │ │korean_ag.│ │
                │ │spanish_ag│ │
                │ │greek_ag. │ │
                │ └──────────┘ │
                └──────────────┘
```

Ưu điểm của cách tiếp cận này:
- Tận dụng **khả năng gọi công cụ sẵn có** của LLM, loại bỏ nhu cầu logic định tuyến riêng biệt
- `description` của decorator `@tool` cung cấp tiêu chí định tuyến tự nhiên
- `ToolNode` tự động thực thi công cụ agent phù hợp

#### Phân tích mã nguồn

**Bước 1: Hàm tạo Agent-Tool**

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

def make_agent_tool(tool_name, tool_description, system_prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {system_prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node("tools", ToolNode(tools=tools))

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    agent = agent_builder.compile()

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
    )
    def agent_tool(state: Annotated[dict, InjectedState]):
        result = agent.invoke(state)
        return result["messages"][-1].content

    return agent_tool
```

Hàm này có cấu trúc tương tự `make_agent` trước đó, nhưng với sự khác biệt chính:

- **Giá trị trả về là hàm `@tool`, không phải đồ thị đã biên dịch**
- `@tool(name_or_callable=tool_name, description=tool_description)`: Tạo động công cụ với tên và mô tả nhận được qua tham số
- `Annotated[dict, InjectedState]`: `InjectedState` là annotation đặc biệt của LangGraph **tự động tiêm trạng thái đồ thị hiện tại vào hàm công cụ**. LLM không nhận ra tham số này (không hiển thị trong schema công cụ); LangGraph tự động truyền trạng thái khi thực thi.
- `result["messages"][-1].content`: Trích xuất và trả về chỉ văn bản phản hồi cuối cùng của agent

**Bước 2: Tạo danh sách công cụ**

```python
tools = [
    make_agent_tool(
        tool_name="korean_agent",
        tool_description="Use this when the user is speaking korean",
        system_prompt="You're a korean customer support agent you speak in korean",
        tools=[],
    ),
    make_agent_tool(
        tool_name="spanish_agent",
        tool_description="Use this when the user is speaking spanish",
        system_prompt="You're a spanish customer support agent you speak in spanish",
        tools=[],
    ),
    make_agent_tool(
        tool_name="greek_agent",
        tool_description="Use this when the user is speaking greek",
        system_prompt="You're a greek customer support agent you speak in greek",
        tools=[],
    ),
]
```

`tool_description` của mỗi công cụ agent đóng vai trò tiêu chí định tuyến của LLM. LLM phát hiện ngôn ngữ của người dùng và tự nhiên gọi công cụ agent ngôn ngữ tương ứng.

**Bước 3: Đơn giản hóa Supervisor**

```python
def supervisor(state: AgentState):
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [result],
    }
```

So với phiên bản trước, điều này được **đơn giản hóa đáng kể**:
- Sử dụng `bind_tools` thay vì `structured_output`
- Đơn giản truyền tin nhắn thay vì prompt phức tạp
- LLM tự đọc mô tả công cụ và chọn công cụ phù hợp

**Bước 4: Lắp ráp đồ thị**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", tools_condition)
graph_builder.add_edge("tools", "supervisor")
graph_builder.add_edge("supervisor", END)

graph = graph_builder.compile()
```

Trong kiến trúc supervisor trước, mỗi agent là node riêng biệt, nhưng bây giờ chỉ có **2 node**:
- `supervisor`: Node nơi LLM quyết định gọi công cụ
- `tools`: Node nơi `ToolNode` thực thi công cụ agent

`tools_condition` định tuyến đến node `tools` nếu phản hồi LLM chứa lệnh gọi công cụ, ngược lại đến `END`. Điều này giống hệt mẫu ReAct agent cơ bản, chỉ khác ở chỗ các công cụ là agent.

#### So sánh ba kiến trúc

| Đặc điểm | Mạng | Supervisor | Supervisor+Công cụ |
|-----------|------|------------|-------------------|
| Số node đồ thị | Bằng số agent | Số agent + 1 | 2 (supervisor + tools) |
| Cơ chế định tuyến | Command + handoff_tool | structured_output | bind_tools + ToolNode |
| Triển khai agent | Node đồ thị con | Node đồ thị con | Bên trong hàm @tool |
| Độ phức tạp mã | Trung bình | Cao | Thấp |

#### Điểm thực hành

- Xóa `InjectedState` và chạy. Quan sát LLM cố gắng truyền đối số gì cho agent
- Viết `description` công cụ agent chi tiết hơn và thử nghiệm xem độ chính xác định tuyến có cải thiện không
- Thêm công cụ thực sự (ví dụ: công cụ tìm kiếm) vào một agent và kiểm tra xem lệnh gọi công cụ lồng nhau có hoạt động không

---

### 18.5 Prebuilt Agents - Agent được xây dựng sẵn

**Chủ đề và Mục tiêu**: Sử dụng `create_supervisor` và `create_react_agent` từ thư viện `langgraph-supervisor` để triển khai hệ thống supervisor đa agent với **mã tối thiểu**.

#### Các khái niệm chính

Tất cả các mẫu chúng ta đã triển khai thủ công (factory agent, supervisor, định tuyến dựa trên công cụ) đều được LangGraph cung cấp dưới dạng **module dựng sẵn**. Với các gói `langgraph-supervisor` và `langgraph-swarm`, bạn có thể xây dựng hệ thống đa agent mạnh mẽ chỉ trong vài dòng mã.

Dependency bổ sung:
```toml
"langgraph-supervisor==0.0.29",
"langgraph-swarm==0.0.14",
```

#### Phân tích mã nguồn

**Bước 1: Import đơn giản hóa**

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
```

Tất cả import cần thiết trong các phần trước -- `StateGraph`, `Command`, `ToolNode`, `tools_condition` -- đều biến mất. Chỉ hai hàm `create_react_agent` và `create_supervisor` là đủ.

**Bước 2: Tạo Agent chuyên biệt**

```python
MODEL = "openai:gpt-5"

history_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="history_agent",
    prompt="You are a history expert. You only answer questions about history.",
)
geography_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="geography_agent",
    prompt="You are a geography expert. You only answer questions about geography.",
)
maths_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="maths_agent",
    prompt="You are a maths expert. You only answer questions about maths.",
)
philosophy_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="philosophy_agent",
    prompt="You are a philosophy expert. You only answer questions about philosophy.",
)
```

`create_react_agent` là hàm dựng sẵn của LangGraph tạo agent mẫu ReAct chỉ trong một dòng:

- `model`: Chuỗi mô hình hoặc đối tượng mô hình đã khởi tạo
- `tools`: Danh sách công cụ agent sử dụng
- `name`: Tên duy nhất của agent (supervisor sử dụng khi định tuyến)
- `prompt`: System prompt của agent

Toàn bộ triển khai thủ công từ các phần trước -- `StateGraph` + node `agent` + node `tools` + kết nối edge -- đều được đóng gói trong dòng này.

**Bước 3: Tạo Supervisor**

```python
supervisor = create_supervisor(
    agents=[
        history_agent,
        maths_agent,
        geography_agent,
        philosophy_agent,
    ],
    model=init_chat_model(MODEL),
    prompt="""
    You are a supervisor that routes student questions to the appropriate subject expert.
    You manage a history agent, geography agent, maths agent, and philosophy agent.
    Analyze the student's question and assign it to the correct expert based on the subject matter:
        - history_agent: For historical events, dates, historical figures
        - geography_agent: For locations, rivers, mountains, countries
        - maths_agent: For mathematics, calculations, algebra, geometry
        - philosophy_agent: For philosophical concepts, ethics, logic
    """,
).compile()
```

Những gì `create_supervisor` thực hiện nội bộ:
1. Chuyển đổi mỗi agent thành công cụ (định dạng `transfer_to_{agent_name}`)
2. Tạo node supervisor và bind công cụ
3. Thiết lập `ToolNode` và edge có điều kiện
4. Tự động thêm công cụ `transfer_back_to_supervisor` quay lại supervisor sau khi agent thực thi

`.compile()` được gọi để biên dịch thành đồ thị có thể thực thi.

**Bước 4: Thực thi và xác minh**

```python
questions = [
    "When was Madrid founded?",
    "What is the capital of France and what river runs through it?",
    "What is 15% of 240?",
    "Tell me about the Battle of Waterloo",
    "What are the highest mountains in Asia?",
    "If I have a rectangle with length 8 and width 5, what is its area and perimeter?",
    "Who was Alexander the Great?",
    "What countries border Switzerland?",
    "Solve for x: 2x + 10 = 30",
]

for question in questions:
    result = supervisor.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        }
    )
    if result["messages"]:
        for message in result["messages"]:
            message.pretty_print()
```

Xem luồng thực thi:

1. Câu hỏi của người dùng được truyền cho supervisor
2. Supervisor gọi công cụ `transfer_to_{agent_name}`
3. Agent tương ứng trả lời câu hỏi
4. Agent tự động gọi `transfer_back_to_supervisor` để quay lại supervisor
5. Supervisor trả về phản hồi cuối cùng

Ví dụ kết quả thực thi:
```
Human Message: When was Madrid founded?
Ai Message (supervisor): gọi transfer_to_history_agent
Tool Message: Successfully transferred to history_agent
Ai Message (history_agent): Madrid originated in the mid-9th century...
Ai Message (history_agent): Transferring back to supervisor
Ai Message (supervisor): Truyền đạt phản hồi cuối cùng
```

#### So sánh triển khai thủ công vs Prebuilt

```python
# Triển khai thủ công (18.4): ~60 dòng
def make_agent_tool(...): ...
def supervisor(...): ...
graph_builder = StateGraph(...)
graph_builder.add_node(...)
# ... nhiều dòng cấu hình

# Prebuilt (18.5): ~15 dòng
agent = create_react_agent(model=MODEL, tools=[], name="agent", prompt="...")
supervisor = create_supervisor(agents=[...], model=..., prompt="...").compile()
```

#### Điểm thực hành

- Thêm công cụ thực sự cho `philosophy_agent` để tạo agent sử dụng dữ liệu bên ngoài
- Thay đổi prompt của supervisor sang tiếng Hàn và xác minh hoạt động bình thường
- Đo tỷ lệ supervisor chọn đúng agent cho nhiều câu hỏi
- Gói `langgraph-swarm` cũng đã được cài đặt. Nghiên cứu mẫu Swarm và so sánh với mẫu Supervisor

---

## 3. Tổng kết trọng tâm chương

### Ba mẫu kiến trúc đa Agent

| Mẫu | Mô tả | Ưu điểm | Nhược điểm |
|-----|-------|---------|-----------|
| **Mạng (P2P)** | Agent chuyển giao trực tiếp qua `Command` + `handoff_tool` | Không có nút thắt trung tâm, tự chủ cao | Tất cả agent cần sửa đổi khi thêm mới |
| **Supervisor** | Node trung tâm định tuyến qua `structured_output` | Dễ kiểm soát, dễ gỡ lỗi | Độ phức tạp prompt supervisor có thể tăng |
| **Supervisor+Công cụ** | Agent đóng gói thành `@tool`, định tuyến qua `bind_tools` | Mã ngắn gọn, tận dụng khả năng sẵn có của LLM | Hạn chế truy cập trạng thái nội bộ agent |

### Các khái niệm LangGraph cốt lõi

1. **`Command`**: Đối tượng điều khiển luồng thực thi đồ thị bằng lập trình
   - `goto`: Chỉ định node thực thi tiếp theo
   - `update`: Cập nhật trạng thái
   - `graph=Command.PARENT`: Di chuyển ở cấp đồ thị cha

2. **`InjectedState`**: Annotation tự động tiêm trạng thái đồ thị hiện tại vào hàm công cụ. Không hiển thị trong schema LLM.

3. **`destinations`**: Tham số của `add_node` khai báo các node đích có thể trong định tuyến động dựa trên `Command`. Chỉ dùng cho trực quan hóa; không ảnh hưởng logic thực thi.

4. **Mẫu đồ thị con**: Sử dụng đồ thị đã biên dịch trả về bởi `make_agent` làm node của đồ thị khác để tạo cấu trúc phân cấp.

5. **`create_react_agent` / `create_supervisor`**: Hàm dựng sẵn đóng gói tất cả các mẫu trên.

### Chiến lược phòng chống vòng lặp vô hạn

Vấn đề phổ biến nhất trong hệ thống đa agent là **chuyển đổi vô hạn giữa agent**. Các chiến lược ngăn chặn:

1. **Ràng buộc prompt**: Nêu rõ "đừng chuyển giao cho chính mình"
2. **Xác thực cấp mã**: Kiểm tra `transfer_to == transfered_by`
3. **Điều kiện kết thúc của supervisor**: Cung cấp tùy chọn `__end__`
4. **Ràng buộc cấu trúc**: Chỉ cho phép đích hợp lệ với kiểu `Literal`

---

## 4. Bài tập thực hành

### Bài tập 1: Mở rộng hệ thống hỗ trợ khách hàng đa ngôn ngữ (Độ khó: Trung bình)

Dựa trên kiến trúc mạng (18.1), triển khai:
- Thêm agent tiếng Nhật và tiếng Trung
- Sửa đổi phù hợp docstring và `destinations` của `handoff_tool`
- Thêm công cụ FAQ đơn giản cho mỗi agent để triển khai các tính năng như "kiểm tra trạng thái giao hàng" và "yêu cầu hoàn tiền"

### Bài tập 2: Thí nghiệm so sánh kiến trúc Supervisor (Độ khó: Trung bình)

Cho cùng kịch bản (định tuyến câu hỏi học sinh):
1. Supervisor `structured_output` của 18.3
2. Supervisor dựa trên công cụ của 18.4
3. `create_supervisor` của 18.5

Triển khai từng cách trong ba cách và đo cho cùng bộ câu hỏi:
- Tỷ lệ định tuyến đến đúng agent
- Thời gian phản hồi
- Lượng token sử dụng

Viết báo cáo so sánh.

### Bài tập 3: Đa Supervisor phân cấp (Độ khó: Cao)

Lồng `create_supervisor` để triển khai cấu trúc phân cấp 2 cấp sau:

```
                    ┌─────────────────┐
                    │  Main Supervisor│
                    └───┬─────────┬───┘
                        │         │
              ┌─────────▼──┐  ┌──▼──────────┐
              │Science Sup.│  │Humanities S.│
              └──┬──────┬──┘  └──┬──────┬───┘
                 │      │        │      │
              ┌──▼┐  ┌──▼┐   ┌──▼┐  ┌──▼──────┐
              │Vật│  │Hóa│   │Sử │  │Triết học│
              │lý │  │học│   │   │  │         │
              └───┘  └───┘   └───┘  └─────────┘
```

- Main Supervisor: Phân biệt lĩnh vực khoa học/nhân văn
- Science Supervisor: Quản lý agent vật lý/hóa học
- Humanities Supervisor: Quản lý agent lịch sử/triết học

### Bài tập 4: Khám phá kiến trúc Swarm (Độ khó: Cao)

Sử dụng gói `langgraph-swarm`:
1. Nghiên cứu kiến trúc Swarm là gì
2. Phân tích sự khác biệt với kiến trúc mạng
3. Triển khai cùng kịch bản hỗ trợ khách hàng với mẫu Swarm
4. Tóm tắt ưu nhược điểm của ba mẫu: mạng, supervisor và swarm

---

## Phụ lục: Tham chiếu API chính

### Command

```python
Command(
    goto="node_name",           # Node cần di chuyển đến
    update={"key": "value"},    # Cập nhật trạng thái
    graph=Command.PARENT,       # Di chuyển ở cấp đồ thị cha (khi sử dụng bên trong đồ thị con)
)
```

### create_react_agent

```python
agent = create_react_agent(
    model="openai:gpt-4o",      # Chuỗi mô hình hoặc mô hình đã khởi tạo
    tools=[tool1, tool2],        # Danh sách công cụ sử dụng
    name="agent_name",           # Tên agent duy nhất
    prompt="System prompt",      # Định nghĩa vai trò agent
)
```

### create_supervisor

```python
supervisor = create_supervisor(
    agents=[agent1, agent2],     # Danh sách agent cần quản lý
    model=init_chat_model(...),  # Mô hình cho supervisor
    prompt="Prompt quy tắc định tuyến",  # Chỉ thị supervisor
).compile()                      # Bắt buộc gọi compile()
```

### InjectedState

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def my_tool(state: Annotated[dict, InjectedState]):
    # state được tự động tiêm trạng thái đồ thị hiện tại
    # LLM không nhận ra tham số này
    return state["messages"][-1].content
```
