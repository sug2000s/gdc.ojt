# Chương 07: Cơ bản OpenAI Agents SDK và Tích hợp Streamlit

---

## 1. Tổng quan chương

Chương này bao gồm mọi thứ từ cơ bản xây dựng AI agent sử dụng **OpenAI Agents SDK** (`openai-agents`) đến tích hợp UI thực tế. Bắt đầu từ tạo agent đơn giản, chúng ta dần dần học xử lý sự kiện streaming, duy trì hội thoại thông qua session memory, handoff giữa các agent, đầu ra có cấu trúc (Structured Output), trực quan hóa đồ thị, và cuối cùng xây dựng web UI với Streamlit.

### Mục tiêu học tập

- Hiểu các thành phần cốt lõi của OpenAI Agents SDK (`Agent`, `Runner`, `function_tool`)
- Học hai phương pháp (cao cấp/thấp cấp) xử lý sự kiện trong phản hồi streaming
- Triển khai quản lý bộ nhớ dựa trên session với `SQLiteSession`
- Thiết kế pattern Handoff trong hệ thống multi-agent
- Học đầu ra có cấu trúc sử dụng Pydantic `BaseModel` và trực quan hóa đồ thị agent
- Hiểu các widget cơ bản và mô hình Data Flow của framework Streamlit

### Cấu trúc dự án

```
chatgpt-clone/
├── .gitignore
├── .python-version          # Python 3.13.3
├── pyproject.toml           # Cấu hình phụ thuộc dự án
├── uv.lock                  # File lock trình quản lý gói uv
├── dummy-agent.ipynb        # Jupyter notebook thí nghiệm agent
├── main.py                  # Ứng dụng web Streamlit
├── ai-memory.db             # DB bộ nhớ session SQLite
└── README.md
```

### Phụ thuộc chính

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

- **`openai-agents[viz]`**: OpenAI Agents SDK (bao gồm extension trực quan hóa)
- **`graphviz`**: Trực quan hóa đồ thị agent dưới dạng SVG
- **`streamlit`**: Framework web UI
- **`python-dotenv`**: Quản lý biến môi trường (.env)
- **`ipykernel`**: Jupyter notebook kernel (cho phát triển)

---

## 2. Giải thích chi tiết từng phần

---

### 2.1 Phần 7.0 - Introduction (Thiết lập dự án ban đầu)

**Commit**: `fbc2f97`

#### Chủ đề và mục tiêu

Đây là giai đoạn thiết lập khung sườn cơ bản của dự án. Khởi tạo dự án dựa trên Python 3.13 sử dụng trình quản lý gói `uv` và cài đặt các phụ thuộc cần thiết.

#### Khái niệm cốt lõi

**Trình quản lý gói uv**

Dự án này sử dụng `uv` thay vì `pip`. `uv` là trình quản lý gói Python siêu nhanh viết bằng Rust, quản lý phụ thuộc thông qua file `pyproject.toml` và `uv.lock`. Vì `3.13.3` được chỉ định trong file `.python-version`, `uv` tự động sử dụng phiên bản Python đó.

**Lệnh khởi tạo dự án (Tham khảo)**

```bash
uv init chatgpt-clone
cd chatgpt-clone
uv add "openai-agents[viz]" python-dotenv streamlit
uv add --dev ipykernel
```

**openai-agents SDK là gì?**

Framework agent được cung cấp chính thức bởi OpenAI, cung cấp các tính năng cốt lõi sau:

| Thành phần | Mô tả |
|-----------|-------|
| `Agent` | Class để định nghĩa agent. Cấu hình tên, hướng dẫn (instructions), công cụ (tools), v.v. |
| `Runner` | Class để thực thi agent. Hỗ trợ thực thi đồng bộ/bất đồng bộ/streaming |
| `function_tool` | Decorator chuyển đổi hàm Python thành công cụ agent có thể sử dụng |
| `SQLiteSession` | Quản lý bộ nhớ session dựa trên SQLite |
| `ItemHelpers` | Tiện ích trích xuất tin nhắn từ sự kiện streaming |

#### Điểm thực hành

1. Cài đặt `uv` và khởi tạo dự án
2. Xem xét cấu trúc phụ thuộc trong `pyproject.toml`
3. Chọn kernel `.venv` trong Jupyter notebook để xác nhận môi trường phát triển

---

### 2.2 Phần 7.2 - Stream Events (Xử lý sự kiện Streaming)

**Commit**: `996dae4`

#### Chủ đề và mục tiêu

Học cách xử lý phản hồi agent theo thời gian thực thông qua streaming. Cả hai cách tiếp cận xử lý sự kiện cao cấp và thấp cấp (raw) đều được đề cập.

#### Khái niệm cốt lõi

**Định nghĩa Agent và Tool**

Đầu tiên, định nghĩa agent và tool đơn giản:

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

Điểm chính:
- Decorator `@function_tool` chuyển đổi hàm Python thông thường thành công cụ agent
- **Docstring** của hàm được sử dụng làm mô tả công cụ -- agent đọc mô tả này để quyết định khi nào sử dụng công cụ
- **Type hint** của hàm (ví dụ: `city: str`) được tự động chuyển đổi thành schema tham số của công cụ
- Công cụ được truyền cho `Agent` qua danh sách `tools`

**Phương pháp 1: Xử lý sự kiện cao cấp (run_item_stream_event)**

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

Trong cách tiếp cận này, sự kiện streaming được phân loại thành **ba nhóm** để xử lý:

| Loại sự kiện | Mô tả |
|-------------|-------|
| `raw_response_event` | Phản hồi raw chưa xử lý (bỏ qua trong cách này) |
| `agent_updated_stream_event` | Xảy ra khi agent đang hoạt động hiện tại thay đổi |
| `run_item_stream_event` | Xảy ra khi một mục thực thi (tin nhắn, lệnh gọi tool, v.v.) được tạo |

Các loại mục trong `run_item_stream_event`:

| Loại mục | Mô tả |
|---------|-------|
| `tool_call_item` | Khi agent gọi tool |
| `tool_call_output_item` | Khi kết quả thực thi tool được trả về |
| `message_output_item` | Phản hồi văn bản của agent |

**Phương pháp 2: Xử lý sự kiện thấp cấp (raw_response_event)**

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

Cách tiếp cận này xử lý trực tiếp `raw_response_event` để triển khai **streaming thời gian thực ở cấp token**:

| Loại sự kiện raw | Mô tả |
|-----------------|-------|
| `response.output_text.delta` | Mảnh token (delta) của phản hồi văn bản |
| `response.function_call_arguments.delta` | Mảnh (delta) của tham số gọi tool |
| `response.completed` | Một phản hồi đã hoàn thành |

Nhìn vào kết quả thực thi, bạn có thể thấy tham số gọi tool được xây dựng dần dần:

```
{"
{"city
{"city":"
{"city":"Madrid
{"city":"Madrid"}
```

Sau đó phản hồi văn bản cũng tích lũy từng token:

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

| Đặc điểm | Cao cấp (run_item) | Thấp cấp (raw_response) |
|----------|---------------------|------------------------|
| Độ chi tiết | Cấp mục (item) | Cấp token (delta) |
| Trường hợp sử dụng | Xử lý logic, quản lý trạng thái | Cập nhật UI thời gian thực |
| Độ phức tạp | Thấp | Cao (cần tích lũy chuỗi thủ công) |
| UI giống ChatGPT | Không phù hợp | Phù hợp (hiệu ứng gõ chữ) |

#### Điểm thực hành

1. Chạy cả hai cách streaming và so sánh sự khác biệt đầu ra
2. Hiểu logic tích lũy `delta` để khôi phục toàn bộ tin nhắn
3. Suy nghĩ tại sao `message` và `args` được reset trong sự kiện `response.completed` -- vì nhiều phản hồi có thể xảy ra trong một lần thực thi

---

### 2.3 Phần 7.3 - Session Memory (Bộ nhớ Session)

**Commit**: `35a1fe4`

#### Chủ đề và mục tiêu

Triển khai bộ nhớ session sử dụng `SQLiteSession` để agent có thể nhớ các cuộc hội thoại trước. Điều này cho phép hội thoại nhiều lượt (multi-turn).

#### Khái niệm cốt lõi

**Tại sao cần bộ nhớ Session**

Mặc định, agent không có ký ức gì về cuộc hội thoại trước mỗi khi `Runner.run()` được gọi. Mỗi lần gọi là cuộc hội thoại mới độc lập. Ngay cả khi người dùng nói "Tên tôi là Nico" rồi hỏi "Tên tôi là gì?" trong lần gọi tiếp theo, agent không thể trả lời.

`SQLiteSession` tự động lưu lịch sử hội thoại vào cơ sở dữ liệu SQLite và tự động truy xuất trong lần gọi tiếp theo để cung cấp cho agent.

**Thiết lập SQLiteSession**

```python
from agents import Agent, Runner, function_tool, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")
```

- Tham số đầu tiên `"user_1"`: **Định danh session**. Sử dụng cùng định danh sẽ chia sẻ cùng lịch sử hội thoại
- Tham số thứ hai `"ai-memory.db"`: Đường dẫn file cơ sở dữ liệu SQLite

**Thực thi hội thoại với Session**

```python
result = await Runner.run(
    agent,
    "What was my name again?",
    session=session,
)

print(result.final_output)
```

Khi tham số `session=session` được truyền cho `Runner.run()`, SDK tự động:
1. Tải lịch sử hội thoại trước của session đó từ DB
2. Gửi cùng với tin nhắn người dùng mới cho agent
3. Lưu phản hồi của agent vào DB

**Kiểm tra dữ liệu Session**

```python
await session.get_items()
```

Phương thức này trả về toàn bộ lịch sử hội thoại được lưu trong session dưới dạng danh sách. Xem xét cấu trúc dữ liệu trả về:

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
- Lịch sử hội thoại lưu không chỉ tin nhắn người dùng và phản hồi agent mà cả **lệnh gọi tool (`function_call`) và kết quả (`function_call_output`)**
- Với câu hỏi phụ thuộc ngữ cảnh "thành phố lớn thứ ba của nước tôi sống," agent đã sử dụng thông tin "Tôi sống ở Tây Ban Nha" từ cuộc hội thoại trước để suy luận chính xác **Valencia**
- Với câu hỏi "Tên tôi là gì?", nó nhớ thông tin "Tên tôi là Nico" từ cuộc hội thoại trước và trả lời chính xác

#### Điểm thực hành

1. Thay đổi session ID của `SQLiteSession` và xác nhận rằng các cuộc hội thoại khác nhau được duy trì độc lập
2. Xóa file `ai-memory.db` và xác nhận lịch sử hội thoại được reset
3. Sử dụng `await session.get_items()` để trực tiếp xem xét cấu trúc lịch sử hội thoại được lưu
4. Tạo câu hỏi phụ thuộc ngữ cảnh (ví dụ: "Thủ đô của nước đó là gì?") để kiểm thử bộ nhớ session hoạt động đúng

---

### 2.4 Phần 7.4 - Handoffs (Chuyển giao giữa các Agent)

**Commit**: `b23f34f`

#### Chủ đề và mục tiêu

Định nghĩa nhiều agent chuyên biệt và học pattern multi-agent nơi agent chính phân tích câu hỏi người dùng và **chuyển giao (ủy quyền)** cho agent chuyên biệt phù hợp.

#### Khái niệm cốt lõi

**Handoff là gì?**

Handoff là khi một agent chuyển quyền điều khiển cuộc hội thoại cho agent khác. Tương tự như trung tâm dịch vụ khách hàng nơi nhân viên tổng đài chuyển cuộc gọi cho chuyên gia. Trong OpenAI Agents SDK, điều này được triển khai thông qua tham số `handoffs`.

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
| `instructions` | Chỉ dẫn mà agent tự tuân theo. Được sử dụng làm system prompt khi agent đó thực thi |
| `handoff_description` | Mô tả mà **agent khác (agent chính)** tham chiếu khi quyết định có ủy quyền cho agent này hay không |

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
- Hướng dẫn "Transfer to the agent most capable..." rõ ràng làm agent chính đóng vai trò router
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

- `result.last_agent.name` cho phép kiểm tra agent nào cuối cùng tạo phản hồi
- Với câu hỏi "Tại sao các nước bán trái phiếu?", agent chính xác định đây là câu hỏi liên quan kinh tế và chuyển giao cho `Economics Expert Agent`

**Tóm tắt luồng Handoff**

```
Người dùng: "Why do countries sell bonds?"
    |
    v
[Main Agent] -- Phân tích câu hỏi --> Xác định là câu hỏi kinh tế
    |
    v  (handoff)
[Economics Expert Agent] -- Tạo câu trả lời --> "Countries sell bonds..."
    |
    v
Phản hồi được gửi đến người dùng
```

#### Điểm thực hành

1. Gửi câu hỏi liên quan địa lý ("Sông dài nhất thế giới là gì?") và câu hỏi kinh tế ("Lạm phát là gì?") lần lượt, kiểm tra agent nào phản hồi
2. Gửi câu hỏi mơ hồ thuộc cả hai lĩnh vực ("Địa lý ảnh hưởng đến GDP của Hàn Quốc như thế nào?") và quan sát agent nào được chọn
3. Thêm agent chuyên biệt mới (ví dụ: chuyên gia lịch sử) để mở rộng mục tiêu handoff
4. Sửa đổi `handoff_description` và thử nghiệm kết quả định tuyến thay đổi như thế nào

---

### 2.5 Phần 7.5 - Viz and Structured Outputs (Trực quan hóa và Đầu ra có cấu trúc)

**Commit**: `45d261a`

#### Chủ đề và mục tiêu

Học cách trực quan hóa cấu trúc hệ thống agent dưới dạng đồ thị, và cách ép đầu ra agent vào cấu trúc được định trước sử dụng Pydantic `BaseModel`.

#### Khái niệm cốt lõi

**Đầu ra có cấu trúc (Structured Output)**

Mặc định, agent trả về văn bản dạng tự do. Tuy nhiên, khi cần xử lý phản hồi theo chương trình, cần cấu trúc nhất quán. Bằng cách chỉ định Pydantic model trong tham số `output_type`, agent bị buộc phải trả về JSON khớp cấu trúc đó.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    background_explanation: str
```

Áp dụng model này cho agent:

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

- Phản hồi agent được trả về dưới dạng đối tượng có cấu trúc với các trường `answer` và `background_explanation`, không phải văn bản tự do
- Điều này cho phép truy cập lập trình vào các trường cụ thể như `result.final_output.answer`

**Trực quan hóa đồ thị Agent**

```python
from agents.extensions.visualization import draw_graph

draw_graph(main_agent)
```

Hàm `draw_graph()` trực quan hóa cấu trúc hệ thống agent dưới dạng **đồ thị SVG**. Đồ thị được tạo bao gồm các yếu tố sau:

| Node | Màu sắc | Ý nghĩa |
|------|---------|---------|
| `__start__` | Xanh da trời (ellipse) | Điểm bắt đầu thực thi |
| `__end__` | Xanh da trời (ellipse) | Điểm kết thúc thực thi |
| `Main Agent` | Vàng nhạt (rectangle) | Agent chính |
| `Economics Expert Agent` | Trắng (rounded rect) | Agent chuyên biệt |
| `Geo Expert Agent` | Trắng (rounded rect) | Agent chuyên biệt |
| `get_weather` | Xanh lá nhạt (ellipse) | Công cụ (function tool) |

Các cạnh (mũi tên) của đồ thị biểu diễn mối quan hệ handoff và sử dụng công cụ:
- **Mũi tên nét liền**: Hướng handoff giữa các agent
- **Mũi tên nét đứt**: Mối quan hệ gọi/trả về giữa agent và công cụ

**Thêm phụ thuộc**

Để sử dụng tính năng trực quan hóa, cần gói `graphviz`:

```toml
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    ...
]
```

#### Điểm thực hành

1. Chạy `draw_graph(main_agent)` để trực quan hóa cấu trúc hệ thống agent
2. Thêm hoặc sửa đổi trường trong model `Answer` để tùy chỉnh cấu trúc đầu ra
3. Thêm agent chuyên biệt mới và xem đồ thị thay đổi như thế nào
4. So sánh sự khác biệt kiểu `result.final_output` giữa agent có và không có `output_type`

---

### 2.6 Phần 7.8 - Welcome To Streamlit (Cơ bản Streamlit)

**Commit**: `e763a74`

#### Chủ đề và mục tiêu

Giới thiệu framework Streamlit và học cơ bản xây dựng giao diện web sử dụng các widget UI đa dạng. Cũng đề cập ngắn gọn việc theo dõi thực thi agent thông qua tính năng `trace`.

#### Khái niệm cốt lõi

**Streamlit là gì?**

Streamlit là framework cho phép bạn nhanh chóng xây dựng ứng dụng web chỉ bằng Python. Bạn có thể xây dựng trực quan hóa dữ liệu và ứng dụng demo AI mà không cần HTML, CSS, hay JavaScript. Lệnh chạy:

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
| `st.button()` | Nút có thể click |
| `st.text_input()` | Trường nhập văn bản |
| `st.feedback()` | Widget phản hồi (biểu tượng khuôn mặt) |

**Sidebar**

```python
with st.sidebar:
    st.badge("Badge 1")
```

Sử dụng context manager `st.sidebar` đặt widget vào sidebar bên trái.

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

`st.tabs()` tạo giao diện tab, và nội dung được đặt trong ngữ cảnh của mỗi tab.

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
| `st.chat_message("ai")` | Bong bóng tin nhắn chat vai AI |
| `st.chat_message("human")` | Bong bóng tin nhắn chat vai người dùng |
| `st.status()` | Widget hiển thị trạng thái tiến trình (trạng thái loading, hoàn thành, v.v.) |
| `st.chat_input()` | Trường nhập chat (hỗ trợ đính kèm file) |

`st.status()` là widget quan trọng để hiển thị trực quan cho người dùng quá trình agent sử dụng công cụ. Nhãn có thể thay đổi theo thời gian thực bằng `status.update()`, và trạng thái hoàn thành được chỉ bằng `state="complete"`.

**Tính năng trace (Phía Notebook)**

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

Sử dụng context manager `trace()` nhóm nhiều lệnh gọi `Runner.run()` thành một đơn vị theo dõi. Điều này hữu ích cho việc gỡ lỗi và giám sát quá trình thực thi agent trong dashboard của OpenAI.

#### Điểm thực hành

1. Chạy ứng dụng với `streamlit run main.py` và kiểm tra hành vi của mỗi widget
2. Thay đổi nhãn và khoảng thời gian của `st.status()` để thử nghiệm UX
3. Thử đặt vai trò `st.chat_message()` thành tên tùy chỉnh ngoài "ai" và "human"
4. Thêm các widget đa dạng vào sidebar

---

### 2.7 Phần 7.9 - Streamlit Data Flow (Luồng dữ liệu Streamlit)

**Commit**: `8c438f5`

#### Chủ đề và mục tiêu

Học nguyên lý hoạt động cốt lõi của Streamlit, mô hình **Data Flow**, và quản lý **session state (`st.session_state`)**.

#### Khái niệm cốt lõi

**Mô hình Data Flow của Streamlit**

Đặc tính quan trọng nhất của Streamlit: **Mỗi khi người dùng tương tác với widget, toàn bộ script (`main.py`) được thực thi lại từ trên xuống dưới.** Đây là mô hình "Data Flow" của Streamlit.

Ví dụ:
1. Khi người dùng nhập văn bản -> toàn bộ `main.py` được thực thi lại
2. Khi click nút -> toàn bộ `main.py` được thực thi lại
3. Khi toggle checkbox -> toàn bộ `main.py` được thực thi lại

Do đặc tính này, biến thông thường trong script bị reset mỗi lần. Do đó, `st.session_state` phải được sử dụng để duy trì trạng thái giữa các tương tác.

**Quản lý trạng thái với session_state**

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

**Phân tích luồng mã:**

1. **Kiểm tra trạng thái ban đầu**: Nếu key `"is_admin"` không có trong `st.session_state`, khởi tạo thành `False`. Pattern này chỉ khởi tạo ở lần chạy đầu tiên và giữ giá trị hiện có ở các lần chạy lại tiếp theo.

2. **Render Widget**: `st.text_input()` hiển thị trường nhập văn bản và trả về giá trị người dùng nhập vào biến `name`.

3. **Xử lý có điều kiện**: Nếu `name` có giá trị (không phải chuỗi rỗng), hiển thị tin nhắn chào và thay đổi `is_admin` thành `True`.

4. **Kiểm tra trạng thái**: `print()` xuất trạng thái `is_admin` hiện tại ra console server.

**Điểm chính của session_state:**

```
[Lần chạy đầu tiên]
- st.session_state["is_admin"] = False  (khởi tạo)
- name = ""  (chưa nhập)
- print(False)

[Người dùng nhập "Nico" -> toàn bộ script chạy lại]
- "is_admin" in st.session_state == True  (đã tồn tại, bỏ qua khởi tạo)
- name = "Nico"
- st.write("Hello Nico")
- st.session_state["is_admin"] = True
- print(True)

[Người dùng xóa input -> toàn bộ script chạy lại]
- "is_admin" in st.session_state == True  (vẫn tồn tại)
- st.session_state["is_admin"] đã được set True ở lần chạy trước (được giữ!)
- name = ""
- if name: điều kiện không thỏa -> write không thực thi
- print(True)  <-- vẫn True! session_state được giữ qua các lần chạy lại
```

**Biến thông thường vs session_state**

| Đặc điểm | Biến thông thường | `st.session_state` |
|----------|-------------------|-------------------|
| Khi chạy lại | Reset về giá trị ban đầu | Giá trị trước được giữ |
| Trường hợp sử dụng | Tính toán tạm thời | Bảo tồn trạng thái (lịch sử hội thoại, cài đặt, v.v.) |
| Phạm vi | Lần thực thi hiện tại | Toàn bộ session trình duyệt |

Khái niệm này rất quan trọng khi xây dựng clone ChatGPT sau này:
- Lịch sử hội thoại phải được lưu trong `st.session_state` để giữ lại qua các lần chạy lại
- Instance agent và đối tượng session cũng được lưu trong `st.session_state`

#### Điểm thực hành

1. Chạy `main.py`, nhập tên, và quan sát đầu ra `print()` trong terminal
2. Nhập tên rồi xóa, kiểm tra giá trị `is_admin` như thế nào
3. Lưu danh sách lịch sử hội thoại trong `st.session_state` và thử nghiệm liệu nó có được giữ khi thêm tin nhắn mới
4. Refresh tab trình duyệt và xác nhận `session_state` được reset

---

## 3. Tóm tắt trọng tâm chương

### Thành phần cốt lõi OpenAI Agents SDK

| Thành phần | Vai trò | Tham số chính |
|-----------|---------|--------------|
| `Agent` | Định nghĩa agent | `name`, `instructions`, `tools`, `handoffs`, `output_type`, `handoff_description` |
| `Runner.run()` | Thực thi đồng bộ | `agent`, `input`, `session` |
| `Runner.run_streamed()` | Thực thi streaming | `agent`, `input` |
| `@function_tool` | Decorator định nghĩa tool | docstring là mô tả, type hint là schema |
| `SQLiteSession` | Bộ nhớ session | `session_id`, `db_path` |
| `draw_graph()` | Trực quan hóa đồ thị agent | `agent` |
| `trace()` | Theo dõi thực thi | `trace_name` |

### Phân cấp sự kiện Streaming

```
stream.stream_events()
├── raw_response_event          # Sự kiện raw cấp token
│   ├── response.output_text.delta
│   ├── response.function_call_arguments.delta
│   └── response.completed
├── agent_updated_stream_event  # Sự kiện chuyển agent
└── run_item_stream_event       # Sự kiện cấp mục
    ├── tool_call_item
    ├── tool_call_output_item
    └── message_output_item
```

### Pattern Multi-Agent: Handoff

```
[Đầu vào người dùng]
     |
     v
[Main Agent (Orchestrator)]
     |
     ├──(câu hỏi kinh tế)--> [Economics Expert Agent]
     ├──(câu hỏi địa lý)--> [Geo Expert Agent]
     └──(khác)--------------> [Phản hồi trực tiếp hoặc agent bổ sung]
```

### Nguyên lý cốt lõi Streamlit

1. **Data Flow**: Toàn bộ script chạy lại mỗi khi tương tác widget
2. **session_state**: Dictionary để duy trì trạng thái qua các lần chạy lại
3. **Chat UI**: `st.chat_message()`, `st.chat_input()`, `st.status()`

---

## 4. Bài tập thực hành

### Bài tập 1: Xây dựng Agent cơ bản (Độ khó: 1/3)

**Mục tiêu**: Xây dựng agent cơ bản và cho nó sử dụng công cụ.

- Tạo tool `calculate` thực hiện bốn phép tính số học trên hai số
- Sử dụng decorator `@function_tool` để định nghĩa tham số `operation: str`, `a: float`, `b: float`
- Hỏi agent "What is 123 * 456 + 789?" và xác nhận nó sử dụng tool

### Bài tập 2: Triển khai hiệu ứng gõ chữ với Streaming (Độ khó: 2/3)

**Mục tiêu**: Triển khai hiệu ứng gõ chữ giống ChatGPT sử dụng sự kiện streaming thấp cấp.

- Sử dụng `Runner.run_streamed()` và `raw_response_event`
- Mỗi khi nhận `response.output_text.delta`, in từng ký tự ra terminal (`print(delta, end="", flush=True)`)
- Hiển thị tên tool và tham số theo thời gian thực khi gọi tool

### Bài tập 3: Agent hội thoại nhiều lượt (Độ khó: 2/3)

**Mục tiêu**: Triển khai hội thoại nhiều lượt sử dụng `SQLiteSession`.

- Xây dựng agent nhớ tên, màu yêu thích, và thức ăn yêu thích của người dùng
- Trong lần gọi đầu, cho biết tên; lần gọi thứ hai, cho biết màu; lần gọi thứ ba, yêu cầu "Tổng hợp tất cả những gì tôi thích"
- Xác nhận phản hồi thứ ba bao gồm tất cả thông tin trước đó

### Bài tập 4: Hệ thống Handoff với 3+ Agent chuyên biệt (Độ khó: 3/3)

**Mục tiêu**: Xây dựng hệ thống định tuyến với agent chuyên biệt cho nhiều lĩnh vực.

- Tạo ít nhất 3 agent chuyên biệt (ví dụ: khoa học, lịch sử, nấu ăn)
- Đặt `output_type` với đầu ra có cấu trúc cho mỗi agent chuyên biệt (ví dụ: `answer`, `confidence_level`, `sources`)
- Trực quan hóa cấu trúc hệ thống với `draw_graph()`
- Gửi nhiều câu hỏi và kiểm thử chúng được định tuyến đến agent đúng

### Bài tập 5: UI Clone ChatGPT với Streamlit (Độ khó: 3/3)

**Mục tiêu**: Xây dựng UI hội thoại giống ChatGPT sử dụng Streamlit.

- Quản lý lịch sử hội thoại dưới dạng danh sách `messages` trong `st.session_state`
- Nhận đầu vào người dùng bằng `st.chat_input()` và hiển thị hội thoại bằng `st.chat_message()`
- Tạo nút "New Chat" trong sidebar để triển khai chức năng reset hội thoại
- (Bonus) Sử dụng `st.status()` để hiển thị trạng thái "Đang suy nghĩ..." khi agent đang làm việc

---

## Phụ lục: Tài liệu tham khảo

- [Tài liệu chính thức OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Tài liệu chính thức Streamlit](https://docs.streamlit.io/)
- [Tài liệu chính thức Pydantic](https://docs.pydantic.dev/)
- [Trình quản lý gói uv](https://docs.astral.sh/uv/)
