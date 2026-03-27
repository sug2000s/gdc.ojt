# Chapter 07: OpenAI Agents SDK Cơ bản và Tích hợp Streamlit

---

## 1. Tổng quan chương

Chương này đề cập từ cơ bản đến tích hợp UI thực tế trong việc xây dựng AI agent bằng **OpenAI Agents SDK** (`openai-agents`). Bắt đầu từ việc tạo agent đơn giản, chúng ta dần dần học xử lý sự kiện streaming, duy trì hội thoại thông qua session memory, handoff giữa các agent, structured output, graph visualization, và cuối cùng là xây dựng web UI bằng Streamlit.

### Mục tiêu học tập

- Hiểu các thành phần cốt lõi của OpenAI Agents SDK (`Agent`, `Runner`, `function_tool`)
- Nắm vững hai phương pháp xử lý sự kiện trong streaming response (high-level/low-level)
- Triển khai quản lý memory dựa trên session thông qua `SQLiteSession`
- Thiết kế mẫu Handoff trong hệ thống multi-agent
- Học structured output sử dụng Pydantic `BaseModel` và visualization graph agent
- Hiểu các widget cơ bản và mô hình Data Flow của framework Streamlit

### Cấu trúc dự án

```
chatgpt-clone/
├── .gitignore
├── .python-version          # Python 3.13.3
├── pyproject.toml           # Cấu hình dependency dự án
├── uv.lock                  # File lock trình quản lý gói uv
├── dummy-agent.ipynb        # Jupyter notebook thử nghiệm agent
├── main.py                  # Ứng dụng web Streamlit
├── ai-memory.db             # DB session memory SQLite
└── README.md
```

### Dependency cốt lõi

```toml
[project]
name = "chatgpt-clone"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    "python-dotenv>=1.1.1",
    "streamlit>=1.48.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

- **`openai-agents[viz]`**: OpenAI Agents SDK (bao gồm extension visualization)
- **`graphviz`**: Visualization graph agent thành SVG
- **`streamlit`**: Framework web UI
- **`python-dotenv`**: Quản lý biến môi trường (.env)
- **`ipykernel`**: Kernel Jupyter notebook (dành cho phát triển)

---

## 2. Giải thích chi tiết từng phần

---

### 2.1 Phần 7.0 - Introduction (Thiết lập ban đầu dự án)

**Commit**: `fbc2f97`

#### Chủ đề và mục tiêu

Đây là giai đoạn thiết lập khung cơ bản cho dự án. Khởi tạo dự án dựa trên Python 3.13 bằng trình quản lý gói `uv` và cài đặt các dependency cần thiết.

#### Giải thích khái niệm cốt lõi

**Trình quản lý gói uv**

Dự án này sử dụng `uv` thay vì `pip`. `uv` là trình quản lý gói Python siêu tốc được viết bằng Rust, quản lý dependency thông qua file `pyproject.toml` và `uv.lock`. Phiên bản `3.13.3` được chỉ định trong file `.python-version`, và `uv` tự động sử dụng phiên bản Python đó.

**Lệnh khởi tạo dự án (tham khảo)**

```bash
uv init chatgpt-clone
cd chatgpt-clone
uv add "openai-agents[viz]" python-dotenv streamlit
uv add --dev ipykernel
```

**openai-agents SDK là gì?**

Đây là framework agent chính thức do OpenAI cung cấp, cung cấp các tính năng cốt lõi sau:

| Thành phần | Mô tả |
|------------|-------|
| `Agent` | Lớp định nghĩa agent. Thiết lập tên, chỉ dẫn (instructions), công cụ (tools), v.v. |
| `Runner` | Lớp thực thi agent. Hỗ trợ thực thi đồng bộ/bất đồng bộ/streaming |
| `function_tool` | Decorator chuyển đổi hàm Python thành công cụ mà agent có thể sử dụng |
| `SQLiteSession` | Quản lý session memory dựa trên SQLite |
| `ItemHelpers` | Tiện ích trích xuất tin nhắn từ sự kiện streaming |

#### Điểm thực hành

1. Cài đặt `uv` và khởi tạo dự án
2. Xem cấu trúc dependency trong `pyproject.toml`
3. Chọn kernel `.venv` trong Jupyter notebook để xác nhận môi trường phát triển

---

### 2.2 Phần 7.2 - Stream Events (Xử lý sự kiện Streaming)

**Commit**: `996dae4`

#### Chủ đề và mục tiêu

Học cách xử lý streaming response của agent theo thời gian thực. Đề cập đến cả hai phương pháp: xử lý sự kiện high-level và xử lý sự kiện low-level (raw).

#### Giải thích khái niệm cốt lõi

**Định nghĩa Agent và Tool**

Trước tiên, định nghĩa một agent và tool đơn giản:

```python
from agents import Agent, Runner, function_tool, ItemHelpers


@function_tool
def get_weather(city: str):
    """Get weather by city"""
    return "30 degrees"


agent = Agent(
    name="Assistant Agent",
    instructions="You are a helpful assistant. Use tools when needed to answer questions",
    tools=[get_weather],
)
```

Điểm cốt lõi:
- Decorator `@function_tool` chuyển đổi hàm Python thông thường thành công cụ của agent
- **Docstring** của hàm được sử dụng làm mô tả (description) của công cụ -- agent nhìn mô tả này để quyết định khi nào sử dụng công cụ
- **Type hint** của hàm (ví dụ: `city: str`) được tự động chuyển đổi thành schema tham số của công cụ
- Truyền công cụ vào `Agent` qua danh sách `tools`

**Phương pháp 1: Xử lý sự kiện High-level (run_item_stream_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        continue
    elif event.type == "agent_updated_stream_event":
        print("Agent updated to", event.new_agent.name)
    elif event.type == "run_item_stream_event":
        if event.item.type == "tool_call_item":
            print(event.item.raw_item.to_dict())
        elif event.item.type == "tool_call_output_item":
            print(event.item.output)
        elif event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))
    print("=" * 20)
```

Phương pháp này phân loại sự kiện streaming thành **ba loại** để xử lý:

| Loại sự kiện | Mô tả |
|------------|-------|
| `raw_response_event` | Response thô chưa xử lý (bỏ qua trong phương pháp này) |
| `agent_updated_stream_event` | Phát sinh khi agent đang hoạt động thay đổi |
| `run_item_stream_event` | Phát sinh khi item thực thi (tin nhắn, tool call, v.v.) được tạo |

Các loại item bên trong `run_item_stream_event`:

| Loại item | Mô tả |
|-----------|-------|
| `tool_call_item` | Khi agent gọi công cụ |
| `tool_call_output_item` | Khi kết quả thực thi công cụ được trả về |
| `message_output_item` | Response văn bản của agent |

**Phương pháp 2: Xử lý sự kiện Low-level (raw_response_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

message = ""
args = ""

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        event_type = event.data.type
        if event_type == "response.output_text.delta":
            message += event.data.delta
            print(message)
        elif event_type == "response.function_call_arguments.delta":
            args += event.data.delta
            print(args)
        elif event_type == "response.completed":
            message = ""
            args = ""
```

Phương pháp này xử lý trực tiếp `raw_response_event` để triển khai **streaming thời gian thực ở mức token**:

| Loại sự kiện raw | Mô tả |
|-----------------|-------|
| `response.output_text.delta` | Mảnh token (delta) của response văn bản |
| `response.function_call_arguments.delta` | Mảnh (delta) của đối số tool call |
| `response.completed` | Một response đã hoàn thành |

Khi xem kết quả thực thi, bạn có thể thấy quá trình đối số tool call được xây dựng dần dần:

```
{"
{"city
{"city":"
{"city":"Madrid
{"city":"Madrid"}
```

Sau đó, response văn bản cũng được tích lũy từng token:

```
Hello
Hello!
Hello! I'm
Hello! I'm doing
Hello! I'm doing well
...
Hello! I'm doing well, thank you. The weather in Madrid, the capital of Spain, is currently 30 degrees Celsius. How can I assist you further?
```

#### So sánh hai phương pháp

| Đặc tính | High-level (run_item) | Low-level (raw_response) |
|----------|----------------------|-------------------------|
| Độ chi tiết | Mức item | Mức token (delta) |
| Mục đích | Xử lý logic, quản lý trạng thái | Cập nhật UI thời gian thực |
| Độ phức tạp | Thấp | Cao (cần tự tích lũy chuỗi) |
| UI kiểu ChatGPT | Không phù hợp | Phù hợp (hiệu ứng gõ phím) |

#### Điểm thực hành

1. Chạy hai phương pháp streaming và so sánh sự khác biệt đầu ra
2. Hiểu logic tích lũy `delta` để khôi phục tin nhắn hoàn chỉnh
3. Suy nghĩ về lý do khởi tạo `message` và `args` tại sự kiện `response.completed` -- vì có thể phát sinh nhiều response trong một lần thực thi

---

### 2.3 Phần 7.3 - Session Memory (Bộ nhớ Session)

**Commit**: `35a1fe4`

#### Chủ đề và mục tiêu

Triển khai session memory sử dụng `SQLiteSession` để agent nhớ nội dung hội thoại trước đó. Thông qua đây, hội thoại đa lượt (multi-turn) trở nên khả thi.

#### Giải thích khái niệm cốt lõi

**Tại sao cần Session Memory**

Mặc định, mỗi khi gọi `Runner.run()`, agent không nhớ gì về hội thoại trước đó. Mỗi lệnh gọi là một hội thoại mới độc lập. Nếu người dùng nói "Tên tôi là Nico", rồi ở lệnh gọi tiếp theo hỏi "Tên tôi là gì?", agent không thể trả lời.

`SQLiteSession` tự động lưu lịch sử hội thoại vào database SQLite và tự động tải lại khi gọi lần sau để cung cấp cho agent.

**Thiết lập SQLiteSession**

```python
from agents import Agent, Runner, function_tool, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")
```

- Đối số thứ nhất `"user_1"`: **Định danh session**. Sử dụng cùng định danh sẽ chia sẻ cùng lịch sử hội thoại
- Đối số thứ hai `"ai-memory.db"`: Đường dẫn file database SQLite

**Thực thi hội thoại sử dụng Session**

```python
result = await Runner.run(
    agent,
    "What was my name again?",
    session=session,
)

print(result.final_output)
```

Khi truyền tham số `session=session` cho `Runner.run()`, SDK tự động:
1. Tải lịch sử hội thoại trước đó của session từ DB
2. Truyền cùng tin nhắn người dùng mới cho agent
3. Lưu response của agent vào DB

**Xác nhận dữ liệu Session**

```python
await session.get_items()
```

Phương thức này trả về toàn bộ lịch sử hội thoại lưu trong session dưới dạng danh sách. Cấu trúc dữ liệu trả về:

```python
[
    {'content': 'Hello how are you? My name is Nico', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'I live in Spain', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What is the weather in the third biggest city of the country i live on', 'role': 'user'},
    {'arguments': '{"city":"Valencia"}', 'name': 'get_weather', 'type': 'function_call', ...},
    {'call_id': '...', 'output': '30 degrees', 'type': 'function_call_output'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What was my name again?', 'role': 'user'},
    {'id': 'msg_...', 'content': [{'text': 'Your name is Nico.', ...}], 'role': 'assistant', ...}
]
```

Điểm đáng chú ý:
- Lịch sử hội thoại lưu cả tin nhắn người dùng, response agent, lẫn **tool call (`function_call`) và kết quả (`function_call_output`)**
- Agent sử dụng thông tin "Tôi sống ở Tây Ban Nha" từ hội thoại trước để suy luận chính xác **Valencia** cho câu hỏi phụ thuộc ngữ cảnh "thành phố lớn thứ ba của Tây Ban Nha"
- Câu hỏi "Tên tôi là gì?" cũng được trả lời chính xác bằng cách nhớ thông tin "Tên tôi là Nico" từ hội thoại trước

#### Điểm thực hành

1. Thay đổi session ID của `SQLiteSession` và xác nhận rằng các hội thoại khác nhau được duy trì độc lập
2. Xóa file `ai-memory.db` và xác nhận lịch sử hội thoại được khởi tạo lại
3. Trực tiếp xem cấu trúc lịch sử hội thoại đã lưu bằng `await session.get_items()`
4. Tạo câu hỏi phụ thuộc ngữ cảnh (ví dụ: "Thủ đô của nước đó là gì?") và kiểm tra session memory hoạt động đúng

---

### 2.4 Phần 7.4 - Handoffs (Chuyển giao giữa các Agent)

**Commit**: `b23f34f`

#### Chủ đề và mục tiêu

Định nghĩa nhiều agent chuyên biệt và học mẫu multi-agent trong đó agent chính phân tích câu hỏi của người dùng và **chuyển giao (handoff/ủy quyền)** cho agent chuyên biệt phù hợp.

#### Giải thích khái niệm cốt lõi

**Handoff là gì?**

Handoff là việc một agent chuyển quyền kiểm soát hội thoại cho agent khác. Giống như khi nhân viên tổng đài chuyển cuộc gọi cho nhân viên chuyên môn. Trong OpenAI Agents SDK, điều này được triển khai thông qua tham số `handoffs`.

**Định nghĩa Agent chuyên biệt**

```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")


geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
)
economics_agent = Agent(
    name="Economics Expert Agent",
    instructions="You are a expert in economics, you answer questions related to them.",
    handoff_description="Use this to answer economics questions.",
)
```

Mỗi agent chuyên biệt có hai thiết lập quan trọng:

| Tham số | Vai trò |
|---------|---------|
| `instructions` | Chỉ dẫn mà agent tự tuân theo. Được sử dụng làm system prompt khi agent đó được thực thi |
| `handoff_description` | Mô tả mà **agent khác (agent chính)** tham khảo khi quyết định có nên ủy quyền cho agent này hay không |

**Agent chính (Orchestrator)**

```python
main_agent = Agent(
    name="Main Agent",
    instructions="You are a user facing agent. Transfer to the agent most capable of answering the user's question.",
    handoffs=[
        economics_agent,
        geaography_agent,
    ],
)
```

- Đăng ký các agent có thể ủy quyền trong danh sách `handoffs`
- Chỉ định rõ trong `instructions` "Transfer to the agent most capable..." để agent chính đóng vai trò router
- Agent chính so sánh nội dung câu hỏi người dùng với `handoff_description` của mỗi agent để chọn agent phù hợp nhất

**Thực thi và xác nhận kết quả**

```python
result = await Runner.run(
    main_agent,
    "Why do countries sell bonds?",
    session=session,
)

print(result.last_agent.name)
print(result.final_output)
```

Đầu ra:
```
Economics Expert Agent
Countries sell bonds as a way to raise funds for various purposes...
```

- Có thể xác nhận agent nào đã tạo response cuối cùng thông qua `result.last_agent.name`
- Cho câu hỏi "Tại sao các quốc gia bán trái phiếu?", agent chính đã đánh giá đây là câu hỏi liên quan đến kinh tế và chuyển giao cho `Economics Expert Agent`

**Tóm tắt luồng Handoff**

```
Người dùng: "Why do countries sell bonds?"
    |
    v
[Main Agent] -- Phân tích câu hỏi --> Đánh giá là câu hỏi kinh tế
    |
    v  (handoff)
[Economics Expert Agent] -- Tạo câu trả lời --> "Countries sell bonds..."
    |
    v
Truyền response cho người dùng
```

#### Điểm thực hành

1. Gửi câu hỏi về địa lý ("Sông dài nhất thế giới là gì?") và câu hỏi kinh tế ("Lạm phát là gì?") và xác nhận agent nào response
2. Gửi câu hỏi mơ hồ liên quan cả hai lĩnh vực ("Tác động của địa lý đến GDP Hàn Quốc?") và quan sát agent nào được chọn
3. Thêm agent chuyên biệt mới (ví dụ: chuyên gia lịch sử) để mở rộng đối tượng handoff
4. Sửa đổi `handoff_description` và thử nghiệm xem kết quả routing thay đổi như thế nào

---

### 2.5 Phần 7.5 - Viz and Structured Outputs (Visualization và Structured Output)

**Commit**: `45d261a`

#### Chủ đề và mục tiêu

Học cách visualization cấu trúc hệ thống agent thành graph và phương pháp ép buộc đầu ra agent theo cấu trúc định trước bằng Pydantic `BaseModel`.

#### Giải thích khái niệm cốt lõi

**Structured Output**

Mặc định, agent trả về văn bản tự do. Tuy nhiên, khi cần xử lý response theo chương trình, cần một cấu trúc nhất định. Khi chỉ định mô hình Pydantic cho tham số `output_type`, agent bắt buộc phải trả về JSON theo cấu trúc đó.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    background_explanation: str
```

Áp dụng mô hình này cho agent:

```python
geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
    tools=[
        get_weather,
    ],
    output_type=Answer,
)
```

Kết quả thực thi:
```
answer="The capital of Thailand's northern province, Chiang Mai, is Chiang Mai City."
background_explanation="Chiang Mai is both a city and a province in northern Thailand..."
```

- Response của agent được trả về dưới dạng đối tượng có cấu trúc với các trường `answer` và `background_explanation`, thay vì văn bản tự do
- Thông qua đây, có thể truy cập trường cụ thể theo chương trình như `result.final_output.answer`

**Visualization Graph Agent**

```python
from agents.extensions.visualization import draw_graph

draw_graph(main_agent)
```

Hàm `draw_graph()` visualization cấu trúc hệ thống agent thành **SVG graph**. Graph được tạo bao gồm các yếu tố sau:

| Node | Màu sắc | Ý nghĩa |
|------|---------|---------|
| `__start__` | Xanh da trời (ellipse) | Điểm bắt đầu thực thi |
| `__end__` | Xanh da trời (ellipse) | Điểm kết thúc thực thi |
| `Main Agent` | Vàng nhạt (rectangle) | Agent chính |
| `Economics Expert Agent` | Trắng (rounded rect) | Agent chuyên biệt |
| `Geo Expert Agent` | Trắng (rounded rect) | Agent chuyên biệt |
| `get_weather` | Xanh lá nhạt (ellipse) | Công cụ (function tool) |

Các edge (mũi tên) trong graph biểu thị quan hệ handoff và sử dụng công cụ:
- **Mũi tên liền nét**: Hướng handoff giữa các agent
- **Mũi tên nét đứt**: Quan hệ gọi/trả về giữa agent và công cụ

**Thêm dependency**

Để sử dụng tính năng visualization cần gói `graphviz`:

```toml
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    ...
]
```

#### Điểm thực hành

1. Chạy `draw_graph(main_agent)` để visualization cấu trúc hệ thống agent
2. Thêm hoặc sửa đổi trường trong mô hình `Answer` để tùy chỉnh cấu trúc đầu ra
3. Thêm agent chuyên biệt mới và xem graph thay đổi như thế nào
4. So sánh sự khác biệt kiểu `result.final_output` giữa agent có và không có `output_type`

---

### 2.6 Phần 7.8 - Welcome To Streamlit (Cơ bản Streamlit)

**Commit**: `e763a74`

#### Chủ đề và mục tiêu

Giới thiệu framework Streamlit và học cơ bản xây dựng giao diện web sử dụng các widget UI đa dạng. Cũng đề cập ngắn gọn đến tính năng `trace` để theo dõi thực thi agent.

#### Giải thích khái niệm cốt lõi

**Streamlit là gì?**

Streamlit là framework cho phép nhanh chóng tạo ứng dụng web chỉ bằng Python. Có thể xây dựng ứng dụng data visualization hay AI demo mà không cần HTML, CSS, JavaScript. Lệnh thực thi:

```bash
streamlit run main.py
```

**Sử dụng Widget cơ bản**

```python
import streamlit as st
import time


st.header("Hello world!")

st.button("Click me please!")

st.text_input(
    "Write your API KEY",
    max_chars=20,
)

st.feedback("faces")
```

| Widget | Mô tả |
|--------|-------|
| `st.header()` | Hiển thị văn bản tiêu đề |
| `st.button()` | Nút có thể nhấp |
| `st.text_input()` | Trường nhập văn bản |
| `st.feedback()` | Widget phản hồi (biểu tượng biểu cảm) |

**Sidebar**

```python
with st.sidebar:
    st.badge("Badge 1")
```

Sử dụng context manager `st.sidebar` cho phép đặt widget vào sidebar bên trái.

**Bố cục Tab**

```python
tab1, tab2, tab3 = st.tabs(["Agent", "Chat", "Outpu"])

with tab1:
    st.header("Agent")
with tab2:
    st.header("Agent 2")
with tab3:
    st.header("Agent 3")
```

Tạo giao diện tab bằng `st.tabs()` và đặt nội dung bên trong context của mỗi tab.

**Giao diện Chat**

```python
with st.chat_message("ai"):
    st.text("Hello!")
    with st.status("Agent is using tool") as status:
        time.sleep(1)
        status.update(label="Agent is searching the web....")
        time.sleep(2)
        status.update(label="Agent is reading the page....")
        time.sleep(3)
        status.update(state="complete")

with st.chat_message("human"):
    st.text("Hi!")


st.chat_input(
    "Write a message for the assistant.",
    accept_file=True,
)
```

| Widget | Mô tả |
|--------|-------|
| `st.chat_message("ai")` | Bong bóng tin nhắn chat vai trò AI |
| `st.chat_message("human")` | Bong bóng tin nhắn chat vai trò người dùng |
| `st.status()` | Widget hiển thị trạng thái tiến trình (loading, hoàn thành, v.v.) |
| `st.chat_input()` | Trường nhập chat (có thể hỗ trợ đính kèm file) |

`st.status()` là widget cốt lõi để hiển thị trực quan cho người dùng quá trình agent sử dụng công cụ. Thay đổi label theo thời gian thực bằng `status.update()` và hiển thị trạng thái hoàn thành bằng `state="complete"`.

**Tính năng trace (phía notebook)**

```python
from agents import trace

with trace("user_111111"):
    result = await Runner.run(
        main_agent,
        "What is the capital of Colombia's northen province.",
        session=session,
    )
    result = await Runner.run(
        main_agent,
        "What is the capital of Cambodia's northen province.",
        session=session,
    )
```

Sử dụng context manager `trace()` cho phép nhóm nhiều lệnh gọi `Runner.run()` thành một đơn vị theo dõi. Hữu ích khi debug và monitoring quá trình thực thi agent trên dashboard của OpenAI.

#### Điểm thực hành

1. Chạy ứng dụng bằng `streamlit run main.py` và kiểm tra hoạt động của mỗi widget
2. Thay đổi label và khoảng thời gian của `st.status()` để thử nghiệm UX
3. Thử đặt vai trò `st.chat_message()` bằng tên tùy chỉnh ngoài "ai", "human"
4. Thêm các widget đa dạng vào sidebar

---

### 2.7 Phần 7.9 - Streamlit Data Flow (Luồng dữ liệu Streamlit)

**Commit**: `8c438f5`

#### Chủ đề và mục tiêu

Học mô hình **Data Flow** -- nguyên lý hoạt động cốt lõi của Streamlit -- và quản lý **trạng thái session (`st.session_state`)**.

#### Giải thích khái niệm cốt lõi

**Mô hình Data Flow của Streamlit**

Đặc tính quan trọng nhất của Streamlit: **Mỗi khi người dùng tương tác với widget, toàn bộ script (`main.py`) được chạy lại từ trên xuống dưới.** Đây là mô hình "Data Flow" của Streamlit.

Ví dụ:
1. Người dùng nhập văn bản -> Toàn bộ `main.py` chạy lại
2. Nhấn nút -> Toàn bộ `main.py` chạy lại
3. Toggle checkbox -> Toàn bộ `main.py` chạy lại

Do đặc tính này, biến thông thường trong script bị khởi tạo lại mỗi lần. Vì vậy, để duy trì trạng thái giữa các tương tác, cần sử dụng `st.session_state`.

**Quản lý trạng thái bằng session_state**

```python
import streamlit as st

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

st.header("Hello!")

name = st.text_input("What is your name?")

if name:
    st.write(f"Hello {name}")
    st.session_state["is_admin"] = True


print(st.session_state["is_admin"])
```

**Phân tích luồng code:**

1. **Kiểm tra trạng thái ban đầu**: Nếu key `"is_admin"` không có trong `st.session_state`, khởi tạo thành `False`. Mẫu này chỉ khởi tạo ở lần chạy đầu tiên và duy trì giá trị hiện có ở các lần chạy lại sau.

2. **Render widget**: `st.text_input()` hiển thị trường nhập văn bản và trả về giá trị người dùng nhập vào biến `name`.

3. **Xử lý có điều kiện**: Nếu `name` có giá trị (không phải chuỗi rỗng), hiển thị tin nhắn chào và thay đổi `is_admin` thành `True`.

4. **Xác nhận trạng thái**: `print()` xuất trạng thái `is_admin` hiện tại ra console server.

**Điểm cốt lõi của session_state:**

```
[Lần chạy đầu tiên]
- st.session_state["is_admin"] = False  (khởi tạo)
- name = ""  (chưa nhập)
- print(False)

[Người dùng nhập "Nico" -> Toàn bộ script chạy lại]
- "is_admin" in st.session_state == True  (đã tồn tại nên bỏ qua khởi tạo)
- name = "Nico"
- st.write("Hello Nico")
- st.session_state["is_admin"] = True
- print(True)

[Người dùng xóa input -> Toàn bộ script chạy lại]
- "is_admin" in st.session_state == True  (vẫn tồn tại)
- st.session_state["is_admin"] đã được set True ở lần chạy trước (duy trì!)
- name = ""
- if name: điều kiện không thỏa mãn -> write không thực thi
- print(True)  <-- Vẫn True! session_state được duy trì giữa các lần chạy lại
```

**Biến thông thường vs session_state**

| Đặc tính | Biến thông thường | `st.session_state` |
|----------|-------------------|-------------------|
| Khi chạy lại | Reset về giá trị ban đầu | Duy trì giá trị trước đó |
| Mục đích | Tính toán tạm thời | Bảo toàn trạng thái (lịch sử hội thoại, cài đặt, v.v.) |
| Phạm vi | Lần thực thi hiện tại | Toàn bộ browser session |

Khái niệm này rất quan trọng khi tạo ChatGPT clone sau này:
- Cần lưu lịch sử hội thoại trong `st.session_state` để duy trì hội thoại trước đó khi chạy lại
- Instance agent hoặc đối tượng session cũng được lưu trong `st.session_state`

#### Điểm thực hành

1. Chạy `main.py`, nhập tên và quan sát output `print()` trong terminal
2. Nhập tên rồi xóa và kiểm tra giá trị `is_admin` trở thành gì
3. Lưu danh sách lịch sử hội thoại trong `st.session_state` và thử nghiệm xem nó có được duy trì khi thêm tin nhắn mới
4. Refresh tab trình duyệt và xác nhận `session_state` bị khởi tạo lại

---

## 3. Tổng kết cốt lõi chương

### Thành phần cốt lõi OpenAI Agents SDK

| Thành phần | Vai trò | Tham số chính |
|------------|---------|---------------|
| `Agent` | Định nghĩa agent | `name`, `instructions`, `tools`, `handoffs`, `output_type`, `handoff_description` |
| `Runner.run()` | Thực thi đồng bộ | `agent`, `input`, `session` |
| `Runner.run_streamed()` | Thực thi streaming | `agent`, `input` |
| `@function_tool` | Decorator định nghĩa công cụ | docstring là mô tả, type hint là schema |
| `SQLiteSession` | Session memory | `session_id`, `db_path` |
| `draw_graph()` | Visualization graph agent | `agent` |
| `trace()` | Theo dõi thực thi | `trace_name` |

### Cấu trúc phân cấp sự kiện Streaming

```
stream.stream_events()
├── raw_response_event          # Sự kiện thô mức token
│   ├── response.output_text.delta
│   ├── response.function_call_arguments.delta
│   └── response.completed
├── agent_updated_stream_event  # Sự kiện chuyển đổi agent
└── run_item_stream_event       # Sự kiện mức item
    ├── tool_call_item
    ├── tool_call_output_item
    └── message_output_item
```

### Mẫu Multi-Agent: Handoff

```
[Đầu vào người dùng]
     |
     v
[Main Agent (Orchestrator)]
     |
     ├──(Câu hỏi kinh tế)--> [Economics Expert Agent]
     ├──(Câu hỏi địa lý)--> [Geo Expert Agent]
     └──(Khác)-------> [Trả lời trực tiếp hoặc agent bổ sung]
```

### Nguyên lý cốt lõi Streamlit

1. **Data Flow**: Toàn bộ script chạy lại mỗi khi tương tác với widget
2. **session_state**: Dictionary duy trì trạng thái giữa các lần chạy lại
3. **UI Chat**: `st.chat_message()`, `st.chat_input()`, `st.status()`

---

## 4. Bài tập thực hành

### Bài 1: Tạo Agent cơ bản (Độ khó: 1/3)

**Mục tiêu**: Tạo agent cơ bản và cho sử dụng công cụ.

- Tạo công cụ `calculate` thực hiện bốn phép tính cơ bản với hai số
- Sử dụng decorator `@function_tool` để định nghĩa tham số `operation: str`, `a: float`, `b: float`
- Hỏi agent "What is 123 * 456 + 789?" và xác nhận agent có sử dụng công cụ không

### Bài 2: Triển khai hiệu ứng gõ phím bằng Streaming (Độ khó: 2/3)

**Mục tiêu**: Sử dụng sự kiện streaming low-level để triển khai hiệu ứng gõ phím tương tự ChatGPT.

- Sử dụng `Runner.run_streamed()` và `raw_response_event`
- Mỗi khi nhận `response.output_text.delta`, xuất từng ký tự ra terminal (`print(delta, end="", flush=True)`)
- Hiển thị tên công cụ và đối số theo thời gian thực khi tool call

### Bài 3: Agent hội thoại đa lượt (Độ khó: 2/3)

**Mục tiêu**: Triển khai hội thoại đa lượt sử dụng `SQLiteSession`.

- Tạo agent nhớ tên, màu yêu thích, món ăn yêu thích của người dùng
- Ở lệnh gọi đầu tiên cho biết tên, lệnh gọi thứ hai cho biết màu, lệnh gọi thứ ba yêu cầu "Hãy tổng hợp những thứ tôi thích"
- Xác nhận response thứ ba có bao gồm tất cả thông tin trước đó

### Bài 4: Hệ thống Handoff với 3+ Agent chuyên biệt (Độ khó: 3/3)

**Mục tiêu**: Xây dựng hệ thống routing với agent chuyên biệt đa lĩnh vực.

- Tạo tối thiểu 3 agent chuyên biệt (ví dụ: khoa học, lịch sử, nấu ăn)
- Thiết lập structured output bằng `output_type` cho mỗi agent (ví dụ: `answer`, `confidence_level`, `sources`)
- Visualization cấu trúc hệ thống bằng `draw_graph()`
- Gửi câu hỏi đa dạng và kiểm tra routing đến agent đúng

### Bài 5: UI ChatGPT Clone bằng Streamlit (Độ khó: 3/3)

**Mục tiêu**: Xây dựng UI hội thoại tương tự ChatGPT bằng Streamlit.

- Quản lý lịch sử hội thoại bằng danh sách `messages` trong `st.session_state`
- Nhận input người dùng bằng `st.chat_input()` và hiển thị hội thoại bằng `st.chat_message()`
- Tạo nút "New Chat" trong sidebar để khởi tạo lại hội thoại
- (Bonus) Sử dụng `st.status()` để hiển thị trạng thái agent đang "suy nghĩ..."

---

## Phụ lục: Tài liệu tham khảo

- [Tài liệu chính thức OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Tài liệu chính thức Streamlit](https://docs.streamlit.io/)
- [Tài liệu chính thức Pydantic](https://docs.pydantic.dev/)
- [Trình quản lý gói uv](https://docs.astral.sh/uv/)
