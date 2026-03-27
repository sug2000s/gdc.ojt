# Chapter 18: Kiến trúc Multi-Agent (Multi-Agent Architectures)

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ học cách thiết kế và triển khai **hệ thống multi-agent -- nơi nhiều AI agent hợp tác với nhau** bằng LangGraph. Vượt qua giới hạn của agent đơn lẻ, chúng ta sẽ xây dựng từng bước kiến trúc trong đó các agent có vai trò chuyên biệt giao tiếp với nhau để xử lý tác vụ phức tạp.

### Mục tiêu học tập

- Hiểu sự cần thiết và khái niệm cốt lõi của hệ thống multi-agent
- **Kiến trúc Network**: Triển khai giao tiếp trực tiếp giữa các agent (P2P)
- **Kiến trúc Supervisor**: Triển khai phương thức quản lý agent bởi bộ điều phối trung tâm
- **Kiến trúc Supervisor-Tool**: Nắm vững mẫu nâng cao đóng gói agent thành tool
- **Agent dựng sẵn (Prebuilt)**: Học cách triển khai ngắn gọn bằng thư viện `langgraph-supervisor`
- Học cách trực quan hóa đồ thị thông qua LangGraph Studio

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

## 2. Giải thích chi tiết từng phần

---

### 18.0 Introduction - Khởi tạo dự án và import cơ bản

**Chủ đề và mục tiêu**: Xây dựng nền tảng cho dự án multi-agent và import các thư viện cốt lõi.

#### Giải thích khái niệm cốt lõi

Để xây dựng hệ thống multi-agent, cần hiểu nhiều module cốt lõi của LangGraph. Phần này khởi tạo dự án và import tất cả thư viện cần thiết.

#### Phân tích code

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

Vai trò của mỗi import:

| Module | Vai trò |
|------|------|
| `StateGraph` | Class cốt lõi tạo đồ thị dựa trên trạng thái. Định nghĩa luồng thực thi agent bằng node và edge. |
| `START`, `END` | Hằng số node đặc biệt biểu thị điểm bắt đầu và kết thúc đồ thị. |
| `Command` | Đối tượng lệnh điều khiển chuyển đổi (handoff) giữa các agent. Chỉ định node tiếp theo bằng `goto` và thay đổi trạng thái bằng `update`. |
| `MessagesState` | Class trạng thái định nghĩa sẵn quản lý danh sách tin nhắn. Cung cấp key `messages` mặc định. |
| `ToolNode` | Node dựng sẵn xử lý gọi tool. Khi LLM quyết định sử dụng tool, node này thực thi thực tế. |
| `tools_condition` | Hàm định tuyến có điều kiện phán đoán phản hồi LLM có chứa lời gọi tool hay không. |
| `@tool` | Decorator chuyển đổi hàm Python thông thường thành tool mà LLM có thể gọi. |
| `init_chat_model` | Tiện ích khởi tạo nhiều LLM khác nhau bằng chuỗi định dạng `"provider:model_name"`. |

#### Điểm thực hành

- Khởi tạo dự án và cài đặt dependency bằng `uv`: `uv sync`
- Cần thiết lập `OPENAI_API_KEY` trong file `.env`
- Yêu cầu môi trường Python 3.13

---

### 18.1 Network Architecture - Kiến trúc Network

**Chủ đề và mục tiêu**: Triển khai **kiến trúc network (P2P)** trong đó các agent có thể trực tiếp chuyển giao (handoff) cuộc hội thoại cho nhau.

#### Giải thích khái niệm cốt lõi

Trong kiến trúc network, **không có bộ điều phối trung tâm** -- mỗi agent tự phán đoán và chuyển giao cuộc hội thoại cho agent khác. Trong ví dụ này, chúng ta tạo agent hỗ trợ khách hàng tiếng Hàn, tiếng Hy Lạp và tiếng Tây Ban Nha, xây dựng hệ thống tự động chuyển đổi sang agent phù hợp với ngôn ngữ khách hàng sử dụng.

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
2. Khi phát hiện ngôn ngữ mà nó không hiểu, sử dụng `handoff_tool` để chuyển đổi sang agent phù hợp

#### Phân tích code

**Bước 1: Định nghĩa state**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str

llm = init_chat_model("openai:gpt-4o")
```

Mở rộng `MessagesState` để định nghĩa state tùy chỉnh theo dõi agent đang hoạt động (`current_agent`) và agent đã thực hiện chuyển đổi (`transfered_by`). Nhờ đó có thể biết agent nào đang xử lý cuộc hội thoại và ai đã chuyển đổi.

**Bước 2: Hàm factory agent**

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

`make_agent` là **hàm factory agent**, cho phép tạo agent có cùng cấu trúc chỉ bằng cách thay đổi tham số. Mỗi agent bên trong tạo thành một sub-graph `StateGraph` hoàn chỉnh:

- Node `agent`: Truyền prompt và lịch sử hội thoại cho LLM để tạo phản hồi
- Node `tools`: `ToolNode` thực thi lời gọi tool
- `tools_condition`: Nếu LLM gọi tool thì định tuyến đến node `tools`, ngược lại đến `END`
- Edge `tools` -> `agent`: Truyền kết quả thực thi tool lại cho agent để cho phép phán đoán thêm

Cấu trúc này cho phép mỗi agent có **vòng lặp ReAct (Reasoning + Acting) độc lập**.

**Bước 3: Định nghĩa handoff tool**

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

`handoff_tool` là **cơ chế cốt lõi** của kiến trúc network. Các điểm chính:

- Trả về đối tượng `Command` để điều khiển trực tiếp luồng thực thi đồ thị
- `goto=transfer_to`: Di chuyển thực thi đến node agent được chỉ định
- `graph=Command.PARENT`: Thực hiện di chuyển ở **đồ thị cha**. Vì mỗi agent là sub-graph, nếu không có tùy chọn này, di chuyển chỉ xảy ra bên trong sub-graph. Thông qua `Command.PARENT`, chuyển đổi giữa các agent ở cấp đồ thị cao nhất trở nên khả thi
- Cập nhật trạng thái bằng `update` để theo dõi thông tin agent hiện tại
- Liệt kê các giá trị có thể trong docstring của tool để hướng dẫn LLM sử dụng giá trị đúng

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

Đăng ký ba node agent trong đồ thị cao nhất. Giá trị của mỗi node là sub-graph đã biên dịch mà `make_agent` trả về. Vì thiết lập `START -> korean_agent`, tất cả cuộc hội thoại bắt đầu từ agent tiếng Hàn. Nếu người dùng nói tiếng Tây Ban Nha, agent tiếng Hàn phát hiện điều này và sử dụng `handoff_tool` để chuyển đổi sang agent tiếng Tây Ban Nha.

#### Điểm thực hành

- Thử đổi agent bắt đầu sang `spanish_agent` và gửi tin nhắn bằng tiếng Hàn để kiểm tra chuyển đổi tự động
- Thử thêm agent ngôn ngữ mới (ví dụ: tiếng Nhật). Cần sửa cả docstring của `handoff_tool`
- Thử nghiệm lỗi gì xảy ra khi loại bỏ `Command.PARENT`

---

### 18.2 Network Visualization - Trực quan hóa Network

**Chủ đề và mục tiêu**: Sử dụng LangGraph Studio để xác nhận trực quan đồ thị multi-agent và debug luồng thực thi. Đồng thời phòng chống bug vòng lặp vô hạn khi chuyển đổi sang chính mình.

#### Giải thích khái niệm cốt lõi

Khi phát triển hệ thống multi-agent phức tạp, việc xác nhận trực quan cấu trúc đồ thị rất quan trọng. LangGraph Studio là công cụ có thể trực quan hóa node và edge của đồ thị, đồng thời theo dõi luồng thực thi thời gian thực.

Tuy nhiên, để trực quan hóa, cần khai báo trước node nào có thể di chuyển đến node nào. Trong định tuyến động sử dụng `Command`, điều này được chỉ định thông qua tham số `destinations`.

#### Phân tích code

**Bước 1: Tách graph.py và thêm destinations**

Tách code notebook thành `graph.py`, đồng thời thêm tham số `destinations` cho mỗi node agent:

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

Tham số `destinations` khai báo các node đích có thể di chuyển đến từ node đó thông qua `Command`. Điều này **không ảnh hưởng đến logic thực thi**, chỉ được sử dụng bởi công cụ trực quan hóa của LangGraph Studio để hiển thị cấu trúc đồ thị chính xác.

**Bước 2: Phòng chống bug chuyển đổi sang chính mình**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    # ... (bỏ qua docstring)
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

Thêm câu phòng chống vào prompt agent:

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

LLM đôi khi cố gắng chuyển đổi sang chính mình, gây ra **vòng lặp vô hạn**. Phòng chống bằng hai lớp:
1. **Cấp prompt**: Chỉ thị rõ ràng "không chuyển đổi sang chính mình"
2. **Cấp code**: Kiểm tra `transfer_to == transfered_by` để từ chối chuyển đổi sang chính mình và trả về thông báo lỗi

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

Sử dụng `stream_mode="updates"` để nhận cập nhật trạng thái của mỗi node thời gian thực. Trong kết quả thực thi, có thể xác nhận toàn bộ luồng: agent tiếng Hàn phát hiện tiếng Tây Ban Nha, chuyển đổi sang agent tiếng Tây Ban Nha, và agent tiếng Tây Ban Nha phản hồi bằng tiếng Tây Ban Nha:

```
{'korean_agent': {'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
{'spanish_agent': {'messages': [...], 'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
```

#### Điểm thực hành

- Chạy Studio bằng LangGraph CLI: `langgraph dev`
- Mở Studio khi loại bỏ `destinations` và so sánh trực quan hóa thay đổi như thế nào
- Gửi tin nhắn tiếng Hy Lạp để kiểm tra luồng chuyển đổi

---

### 18.3 Supervisor Architecture - Kiến trúc Supervisor

**Chủ đề và mục tiêu**: Triển khai kiến trúc trong đó node **supervisor** trung tâm phân tích cuộc hội thoại và định tuyến đến agent phù hợp.

#### Giải thích khái niệm cốt lõi

Trong kiến trúc network, mỗi agent tự quyết định chuyển đổi. Kiến trúc supervisor khác biệt ở chỗ **một bộ điều phối trung tâm (Supervisor)** chịu trách nhiệm tất cả quyết định định tuyến.

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
- **Điểm vào duy nhất**: Tất cả yêu cầu đều đi qua supervisor nên logic định tuyến được tập trung hóa
- **Nhất quán**: Agent không cần lo lắng về định tuyến, chỉ tập trung vào lĩnh vực chuyên môn
- **Dễ kiểm soát**: Chỉ cần sửa prompt của supervisor là có thể thay đổi toàn bộ chiến lược định tuyến

#### Phân tích code

**Bước 1: Định nghĩa model đầu ra có cấu trúc**

```python
from typing import Literal
from pydantic import BaseModel

class SupervisorOutput(BaseModel):
    next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
    reasoning: str
```

Định nghĩa kết quả phán đoán của supervisor bằng model `Pydantic`:
- `next_agent`: Giới hạn giá trị có thể bằng kiểu `Literal` để buộc LLM chỉ trả về tên agent hợp lệ. `"__end__"` có nghĩa là kết thúc cuộc hội thoại.
- `reasoning`: Giải thích lý do supervisor chọn agent đó. Hữu ích cho debug và tính minh bạch.

**Bước 2: Mở rộng state**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
    reasoning: str
```

Thêm trường `reasoning` vào state để theo dõi lý do phán đoán của supervisor.

**Bước 3: Đơn giản hóa agent**

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
    # ... (xây dựng đồ thị giống nhau)
```

Khác với kiến trúc network, chỉ thị liên quan đến `handoff_tool` đã bị **loại bỏ** khỏi prompt agent. Vì supervisor chịu trách nhiệm định tuyến, agent chỉ đơn giản phản hồi bằng ngôn ngữ của mình. Truyền danh sách tool rỗng `tools=[]`.

**Bước 4: Triển khai node supervisor**

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

Cơ chế cốt lõi của supervisor:

- `llm.with_structured_output(SupervisorOutput)`: Buộc LLM trả về JSON đúng schema `SupervisorOutput`. Nhờ đó, định tuyến ổn định không có lỗi phân tích.
- Tag XML `<CONVERSATION_HISTORY>`: Phân tách rõ ràng lịch sử hội thoại để LLM nắm bắt chính xác ngữ cảnh.
- `"Never transfer to the same agent twice in a row"`: Ràng buộc prompt để ngăn vòng lặp vô hạn.
- `"If an agent has replied end the conversation by returning __end__"`: Kết thúc cuộc hội thoại nếu agent đã phản hồi, ngăn lặp lại không cần thiết.
- `Command(goto=response.next_agent)`: Supervisor là node ở đồ thị cao nhất chứ không phải sub-graph, nên không cần `graph=Command.PARENT`.

**Bước 5: Lắp ráp đồ thị - Cấu trúc vòng lặp**

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

1. `START` -> `supervisor`: Tất cả cuộc hội thoại bắt đầu từ supervisor
2. `supervisor` -> `{agent}` hoặc `END`: Supervisor định tuyến đến agent phù hợp hoặc kết thúc bằng `Command`
3. `{agent}` -> `supervisor`: Sau khi agent hoàn thành phản hồi, quay lại supervisor

Cấu trúc vòng lặp này là **cốt lõi của kiến trúc supervisor**. Supervisor có thể phán đoán kết thúc cuộc hội thoại (`__end__`) hay định tuyến thêm đến agent khác sau khi agent phản hồi.

#### So sánh Network vs Supervisor

| Đặc tính | Network | Supervisor |
|------|----------|------------|
| Quyết định định tuyến | Mỗi agent tự quyết | Supervisor trung tâm chịu trách nhiệm |
| Kết nối giữa agent | P2P (kết nối trực tiếp) | Hub-Spoke (qua supervisor) |
| Độ phức tạp agent | Cao (bao gồm logic định tuyến) | Thấp (chỉ phụ trách lĩnh vực chuyên môn) |
| Khả năng mở rộng | Cần sửa tất cả agent khi thêm agent | Chỉ cần sửa supervisor |
| Debug | Khó | Dễ (có thể theo dõi reasoning) |

#### Điểm thực hành

- Xuất trường `reasoning` để phân tích supervisor quyết định định tuyến dựa trên căn cứ nào
- So sánh code cần sửa khi thêm agent trong kiến trúc network và supervisor
- Thử nghiệm xem chuyện gì xảy ra khi loại bỏ tùy chọn `__end__` khỏi `SupervisorOutput`

---

### 18.4 Supervisor As Tools - Đóng gói agent thành tool

**Chủ đề và mục tiêu**: **Đóng gói agent thành tool LLM** để supervisor sử dụng agent một cách tự nhiên thông qua cơ chế gọi tool.

#### Giải thích khái niệm cốt lõi

Trong phần supervisor trước, chúng ta sử dụng `structured_output` để định tuyến. Phần này **chuyển đổi mỗi agent thành tool** và refactor để supervisor gọi agent thông qua cơ chế `bind_tools` + `ToolNode`.

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
- Tận dụng **khả năng gọi tool sẵn có** của LLM nên không cần logic định tuyến riêng
- Cung cấp tiêu chí định tuyến tự nhiên thông qua `description` của decorator `@tool`
- `ToolNode` tự động thực thi tool agent phù hợp

#### Phân tích code

**Bước 1: Hàm factory agent-tool**

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

Hàm này có cấu trúc tương tự `make_agent` trước đó, nhưng có điểm khác biệt cốt lõi:

- **Giá trị trả về là hàm `@tool` chứ không phải đồ thị đã biên dịch**
- `@tool(name_or_callable=tool_name, description=tool_description)`: Tạo tool động bằng cách nhận tên và mô tả tool làm tham số
- `Annotated[dict, InjectedState]`: `InjectedState` là annotation đặc biệt của LangGraph, **tự động inject trạng thái đồ thị hiện tại vào hàm tool**. LLM không nhận biết tham số này (không xuất hiện trong schema tool), LangGraph tự động truyền trạng thái khi thực thi.
- `result["messages"][-1].content`: Chỉ trích xuất và trả về văn bản phản hồi cuối cùng của agent

**Bước 2: Tạo danh sách tool**

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

`tool_description` của mỗi tool agent trở thành tiêu chí định tuyến của LLM. LLM phát hiện ngôn ngữ của người dùng và tự nhiên gọi tool agent ngôn ngữ tương ứng.

**Bước 3: Đơn giản hóa supervisor**

```python
def supervisor(state: AgentState):
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [result],
    }
```

So với phiên bản trước, **đã được đơn giản hóa đáng kể**:
- Sử dụng `bind_tools` thay vì `structured_output`
- Truyền tin nhắn đơn giản thay vì prompt phức tạp
- LLM tự đọc mô tả tool và chọn tool phù hợp

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
- `supervisor`: Node LLM quyết định gọi tool
- `tools`: Node `ToolNode` thực thi tool agent

`tools_condition` định tuyến đến node `tools` nếu phản hồi LLM có chứa lời gọi tool, ngược lại đến `END`. Đây là mẫu agent ReAct cơ bản, chỉ khác ở chỗ tool chính là agent.

#### So sánh ba kiến trúc

| Đặc tính | Network | Supervisor | Supervisor+Tool |
|------|----------|------------|-----------------|
| Số node đồ thị | Bằng số agent | Số agent + 1 | 2 (supervisor + tools) |
| Cơ chế định tuyến | Command + handoff_tool | structured_output | bind_tools + ToolNode |
| Triển khai agent | Node sub-graph | Node sub-graph | Bên trong hàm @tool |
| Độ phức tạp code | Trung bình | Cao | Thấp |

#### Điểm thực hành

- Thử loại bỏ `InjectedState` và quan sát LLM cố truyền tham số gì cho agent
- Viết `description` chi tiết hơn cho tool agent và thử nghiệm liệu độ chính xác định tuyến có cải thiện
- Thêm tool thực tế (ví dụ: tool tìm kiếm) vào một agent để kiểm tra gọi tool lồng nhau có hoạt động không

---

### 18.5 Prebuilt Agents - Agent dựng sẵn

**Chủ đề và mục tiêu**: Sử dụng `create_supervisor` và `create_react_agent` từ thư viện `langgraph-supervisor` để triển khai hệ thống multi-agent supervisor với **code tối thiểu**.

#### Giải thích khái niệm cốt lõi

Tất cả mẫu mà chúng ta đã triển khai thủ công (factory agent, supervisor, định tuyến dựa trên tool) đều được LangGraph cung cấp dưới dạng **module dựng sẵn**. Sử dụng gói `langgraph-supervisor` và `langgraph-swarm`, có thể xây dựng hệ thống multi-agent mạnh mẽ chỉ với vài dòng code.

Dependency bổ sung:
```toml
"langgraph-supervisor==0.0.29",
"langgraph-swarm==0.0.14",
```

#### Phân tích code

**Bước 1: Đơn giản hóa import**

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
```

Tất cả `StateGraph`, `Command`, `ToolNode`, `tools_condition` cần thiết trong phần trước đều đã biến mất. Chỉ cần hai hàm `create_react_agent` và `create_supervisor` là đủ.

**Bước 2: Tạo agent chuyên biệt**

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

`create_react_agent` là hàm dựng sẵn của LangGraph, tạo agent mẫu ReAct chỉ với một dòng:

- `model`: Chuỗi model hoặc đối tượng model đã khởi tạo
- `tools`: Danh sách tool mà agent sử dụng
- `name`: Tên duy nhất của agent (supervisor sử dụng khi định tuyến)
- `prompt`: Prompt hệ thống của agent

Tất cả `StateGraph` + node `agent` + node `tools` + kết nối edge mà phần trước triển khai thủ công đều được đóng gói trong một dòng này.

**Bước 3: Tạo supervisor**

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

Những gì `create_supervisor` thực hiện bên trong:
1. Chuyển đổi mỗi agent thành tool (định dạng `transfer_to_{agent_name}`)
2. Tạo node supervisor và bind tool
3. Thiết lập `ToolNode` và edge có điều kiện
4. Tự động thêm tool `transfer_back_to_supervisor` để quay lại supervisor sau khi agent thực thi

Gọi `.compile()` để biên dịch thành đồ thị thực thi được.

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

Luồng thực thi:

1. Câu hỏi người dùng được truyền đến supervisor
2. Supervisor gọi tool `transfer_to_{agent_name}`
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
Ai Message (supervisor): Truyền phản hồi cuối cùng
```

#### So sánh triển khai thủ công vs Prebuilt

```python
# Triển khai thủ công (18.4): ~60 dòng
def make_agent_tool(...): ...
def supervisor(...): ...
graph_builder = StateGraph(...)
graph_builder.add_node(...)
# ... nhiều code cấu hình

# Prebuilt (18.5): ~15 dòng
agent = create_react_agent(model=MODEL, tools=[], name="agent", prompt="...")
supervisor = create_supervisor(agents=[...], model=..., prompt="...").compile()
```

#### Điểm thực hành

- Thử thêm tool thực tế vào `philosophy_agent` để tạo agent sử dụng dữ liệu bên ngoài
- Thay đổi prompt supervisor sang tiếng Việt và kiểm tra hoạt động bình thường
- Đo lường tỷ lệ supervisor chọn đúng agent cho nhiều câu hỏi
- Gói `langgraph-swarm` cũng đã được cài đặt. Nghiên cứu mẫu Swarm và so sánh với mẫu Supervisor

---

## 3. Tổng kết chương

### 3 mẫu kiến trúc Multi-Agent

| Mẫu | Mô tả | Ưu điểm | Nhược điểm |
|------|------|------|------|
| **Network (P2P)** | Agent chuyển đổi trực tiếp bằng `Command` + `handoff_tool` | Không có nút thắt trung tâm, tính tự chủ cao | Cần sửa tất cả agent khi thêm agent |
| **Supervisor** | Node trung tâm định tuyến bằng `structured_output` | Dễ kiểm soát, dễ debug | Độ phức tạp prompt supervisor có thể tăng |
| **Supervisor+Tool** | Đóng gói agent thành `@tool`, định tuyến bằng `bind_tools` | Code ngắn gọn, tận dụng khả năng cơ bản LLM | Hạn chế truy cập trạng thái nội bộ agent |

### Khái niệm cốt lõi LangGraph

1. **`Command`**: Đối tượng điều khiển lập trình luồng thực thi đồ thị
   - `goto`: Chỉ định node thực thi tiếp theo
   - `update`: Cập nhật trạng thái
   - `graph=Command.PARENT`: Di chuyển ở cấp đồ thị cha

2. **`InjectedState`**: Annotation tự động inject trạng thái đồ thị hiện tại vào hàm tool. Không xuất hiện trong schema LLM.

3. **`destinations`**: Tham số của `add_node`, khai báo node đích có thể đến trong định tuyến động dựa trên `Command`. Chỉ dùng cho trực quan hóa, không ảnh hưởng logic thực thi.

4. **Mẫu sub-graph**: Sử dụng đồ thị đã biên dịch mà `make_agent` trả về làm node của đồ thị khác để tạo cấu trúc phân cấp.

5. **`create_react_agent` / `create_supervisor`**: Hàm dựng sẵn đóng gói tất cả mẫu trên.

### Chiến lược phòng chống vòng lặp vô hạn

Vấn đề phổ biến nhất trong hệ thống multi-agent là **chuyển đổi vô hạn giữa các agent**. Chiến lược phòng chống:

1. **Ràng buộc prompt**: Chỉ thị rõ ràng "không chuyển đổi sang chính mình"
2. **Xác minh cấp code**: Kiểm tra `transfer_to == transfered_by`
3. **Điều kiện kết thúc supervisor**: Cung cấp tùy chọn `__end__`
4. **Ràng buộc cấu trúc**: Chỉ cho phép đối tượng hợp lệ bằng kiểu `Literal`

---

## 4. Bài tập thực hành

### Bài 1: Mở rộng hệ thống hỗ trợ khách hàng đa ngôn ngữ (Độ khó: Trung bình)

Dựa trên kiến trúc network (18.1), triển khai:
- Thêm agent tiếng Nhật và tiếng Trung
- Sửa docstring và `destinations` của `handoff_tool` phù hợp
- Thêm tool FAQ đơn giản cho mỗi agent để triển khai chức năng "kiểm tra trạng thái giao hàng", "yêu cầu hoàn tiền", v.v.

### Bài 2: Thí nghiệm so sánh kiến trúc Supervisor (Độ khó: Trung bình)

Cho cùng kịch bản (định tuyến câu hỏi học sinh):
1. Supervisor `structured_output` của 18.3
2. Supervisor dựa trên tool của 18.4
3. `create_supervisor` của 18.5

Triển khai theo ba cách, đo lường cho cùng bộ câu hỏi:
- Tỷ lệ định tuyến đến đúng agent
- Thời gian phản hồi
- Lượng token sử dụng

và viết báo cáo so sánh.

### Bài 3: Multi-Supervisor phân cấp (Độ khó: Cao)

Lồng `create_supervisor` để triển khai cấu trúc phân cấp 2 tầng:

```
                    ┌─────────────────┐
                    │  Main Supervisor│
                    └───┬─────────┬───┘
                        │         │
              ┌─────────▼──┐  ┌──▼──────────┐
              │Science Sup.│  │Humanities S.│
              └──┬──────┬──┘  └──┬──────┬───┘
                 │      │        │      │
              ┌──▼┐  ┌──▼┐   ┌──▼┐  ┌──▼──┐
              │Vật│  │Hóa│   │Lịch│  │Triết│
              │lý │  │học│   │sử │  │học  │
              └───┘  └───┘   └───┘  └─────┘
```

- Main Supervisor: Phân biệt lĩnh vực khoa học/nhân văn
- Science Supervisor: Quản lý agent vật lý/hóa học
- Humanities Supervisor: Quản lý agent lịch sử/triết học

### Bài 4: Khám phá kiến trúc Swarm (Độ khó: Cao)

Sử dụng gói `langgraph-swarm`:
1. Nghiên cứu kiến trúc Swarm là gì
2. Phân tích sự khác biệt với kiến trúc network
3. Triển khai cùng kịch bản hỗ trợ khách hàng bằng mẫu Swarm
4. Tổng hợp ưu nhược điểm của ba mẫu: Network, Supervisor, Swarm

---

## Phụ lục: Tham chiếu API chính

### Command

```python
Command(
    goto="node_name",           # Node đích
    update={"key": "value"},    # Cập nhật trạng thái
    graph=Command.PARENT,       # Di chuyển ở đồ thị cha (khi sử dụng bên trong sub-graph)
)
```

### create_react_agent

```python
agent = create_react_agent(
    model="openai:gpt-4o",      # Chuỗi model hoặc model đã khởi tạo
    tools=[tool1, tool2],        # Danh sách tool sử dụng
    name="agent_name",           # Tên duy nhất agent
    prompt="Prompt hệ thống",    # Định nghĩa vai trò agent
)
```

### create_supervisor

```python
supervisor = create_supervisor(
    agents=[agent1, agent2],     # Danh sách agent quản lý
    model=init_chat_model(...),  # Model cho supervisor
    prompt="Prompt quy tắc định tuyến",  # Chỉ thị supervisor
).compile()                      # Bắt buộc gọi compile()
```

### InjectedState

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def my_tool(state: Annotated[dict, InjectedState]):
    # state được tự động inject trạng thái đồ thị hiện tại
    # LLM không nhận biết tham số này
    return state["messages"][-1].content
```
