# Chương 8: Xây dựng bản sao ChatGPT với OpenAI Tools và MCP

---

## 1. Tổng quan chương

Trong chương này, chúng ta xây dựng một ứng dụng bản sao ChatGPT hoàn chỉnh bằng cách sử dụng các **Công cụ tích hợp sẵn (Built-in Tools)** và **MCP (Model Context Protocol)** do OpenAI Agents SDK cung cấp. Sử dụng Streamlit làm framework giao diện, chúng ta bắt đầu từ cuộc hội thoại văn bản đơn giản và từng bước mở rộng chức năng bao gồm tìm kiếm web, tìm kiếm tệp, nhập/tạo hình ảnh, thực thi mã và tích hợp máy chủ MCP bên ngoài.

### Mục tiêu học tập

- Triển khai giao diện chat bằng cách tích hợp Streamlit với OpenAI Agents SDK
- Hiểu cách lưu trữ lịch sử hội thoại bền vững bằng `SQLiteSession`
- Nắm vững cách sử dụng các công cụ tích hợp của OpenAI như `WebSearchTool`, `FileSearchTool`, `ImageGenerationTool`, và `CodeInterpreterTool`
- Học cách xử lý đầu vào đa phương tiện (hình ảnh)
- Hiểu các mẫu tích hợp công cụ bên ngoài thông qua `HostedMCPTool` và `MCPServerStdio`
- Thành thạo kỹ thuật cập nhật giao diện thời gian thực bằng sự kiện streaming

### Công nghệ sử dụng

| Công nghệ | Vai trò |
|------|------|
| **Streamlit** | Framework giao diện chat dựa trên web |
| **OpenAI Agents SDK** | Quản lý Agent, Runner và công cụ |
| **SQLiteSession** | Lưu trữ cục bộ lịch sử hội thoại |
| **OpenAI API** | Tải lên tệp và quản lý Vector Store |
| **MCP (Model Context Protocol)** | Giao thức tích hợp máy chủ công cụ bên ngoài |

### Cấu trúc dự án

```
chatgpt-clone/
├── main.py                      # Ứng dụng chính
├── chat-gpt-clone-memory.db     # DB lịch sử hội thoại SQLite
├── facts.txt                    # Dữ liệu mẫu cho File Search
└── international.png            # Hình ảnh để kiểm thử đa phương tiện
```

---

## 2. Mô tả chi tiết từng phần

---

### 8.0 Chat UI - Xây dựng giao diện chat Streamlit

#### Chủ đề và mục tiêu

Xây dựng giao diện chat cơ bản bằng Streamlit và tạo cấu trúc nền tảng tích hợp `Agent` và `Runner` của OpenAI Agents SDK để hiển thị phản hồi streaming theo thời gian thực.

#### Khái niệm chính

**`session_state` của Streamlit** là cơ chế cốt lõi để duy trì trạng thái trong ứng dụng web. Vì Streamlit chạy lại toàn bộ script mỗi khi người dùng tương tác, các đối tượng như Agent và Session phải được lưu trong `session_state` để tránh tạo lại mỗi lần.

**`SQLiteSession`** là công cụ lưu trữ bền vững lịch sử hội thoại do OpenAI Agents SDK cung cấp, tự động lưu và tải hội thoại trong cơ sở dữ liệu SQLite. Điều này đảm bảo các cuộc hội thoại trước đó được giữ lại ngay cả sau khi làm mới trang.

**`Runner.run_streamed()`** chạy agent ở chế độ streaming, cho phép nhận sự kiện thời gian thực khi phản hồi đang được tạo.

#### Phân tích mã

```python
import dotenv
dotenv.load_dotenv()

import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession

# Lưu Agent vào session_state để duy trì cùng một instance qua các lần chạy lại
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.
        """,
    )
agent = st.session_state["agent"]

# Lưu trữ bền vững lịch sử hội thoại vào DB cục bộ bằng SQLiteSession
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",               # Định danh phiên
        "chat-gpt-clone-memory.db",   # Đường dẫn tệp DB SQLite
    )
session = st.session_state["session"]
```

Mẫu quan trọng trong đoạn mã trên là bộ bảo vệ `if "key" not in st.session_state`. Vì Streamlit chạy lại toàn bộ `main.py` mỗi khi người dùng tương tác, nếu không có bộ bảo vệ này, Agent và Session sẽ bị tạo lại mỗi lần và toàn bộ trạng thái trước đó sẽ bị mất.

```python
async def run_agent(message):
    stream = Runner.run_streamed(
        agent,
        message,
        session=session,  # Truyền session để quản lý lịch sử hội thoại tự động
    )

    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                with st.chat_message("ai"):
                    st.write_stream(event.data.delta)
```

`stream.stream_events()` là một async iterator, truyền từng sự kiện một khi chúng xảy ra trong quá trình tạo phản hồi. Trong các loại `raw_response_event`, sự kiện có loại `response.output_text.delta` chứa các đoạn văn bản thực tế (delta).

```python
# Giao diện nhập chat
prompt = st.chat_input("Write a message for your assistant")

if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))

# Thanh bên: Đặt lại bộ nhớ và gỡ lỗi
with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```

`st.chat_input()` là widget nhập liệu chuyên dụng cho chat của Streamlit, và `st.chat_message()` là container phân biệt trực quan tin nhắn người dùng/AI. Thanh bên hiển thị nút đặt lại phiên và các mục hội thoại đã lưu để gỡ lỗi.

#### Điểm thực hành

- Chạy ứng dụng bằng `streamlit run main.py` và thử trò chuyện
- Làm mới trình duyệt và xác minh lịch sử hội thoại được giữ lại trong thanh bên
- Xóa hội thoại bằng `session.clear_session()` và xác minh hành vi

---

### 8.1 Conversation History - Hiển thị lịch sử hội thoại

#### Chủ đề và mục tiêu

Khôi phục lịch sử hội thoại trước đó khi làm mới trang và cải thiện cách hiển thị phản hồi streaming để văn bản xuất hiện dần dần.

#### Khái niệm chính

Ở phần trước, lịch sử hội thoại đã được lưu vào DB nhưng không hiển thị trên màn hình sau khi làm mới trang. Bằng cách thêm hàm **`paint_history()`**, chúng ta triển khai chức năng đọc tin nhắn đã lưu từ SQLiteSession và vẽ lại chúng trên màn hình mỗi khi ứng dụng tải.

Ngoài ra, trước đó mỗi delta kích hoạt một lệnh `st.write()` mới, khiến tin nhắn bị trùng lặp trên nhiều dòng. Điều này được cải thiện bằng cách sử dụng placeholder **`st.empty()`** để tích lũy và cập nhật văn bản trong một vùng duy nhất.

#### Phân tích mã

```python
async def paint_history():
    messages = await session.get_items()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                if message["type"] == "message":
                    st.write(message["content"][0]["text"])

# Thực thi ngay khi ứng dụng tải
asyncio.run(paint_history())
```

`session.get_items()` trả về toàn bộ lịch sử hội thoại đã lưu dưới dạng danh sách. Mỗi tin nhắn là một dictionary, có cấu trúc khác nhau tùy thuộc vào trường `role` là `"user"` hay `"assistant"`. Tin nhắn người dùng có `content` là chuỗi đơn giản, trong khi phản hồi AI có `content` dạng danh sách (`[{"text": "..."}]`).

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()  # Tạo placeholder trống
        response = ""
        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta        # Tích lũy văn bản
                    text_placeholder.write(response)    # Cập nhật tại cùng vị trí
```

**Vai trò của `st.empty()`**: Trong Streamlit, `st.empty()` tạo một container trống có thể được điền nội dung sau. Khi gọi `.write()` trên container này, nó **thay thế** nội dung trước đó, tạo hiệu ứng tự nhiên khi văn bản streaming dần dần dài ra tại một vị trí.

#### Điểm thực hành

- Trò chuyện nhiều lần, sau đó làm mới trang để xác minh lịch sử được khôi phục
- So sánh sự khác biệt khi sử dụng `st.write()` trực tiếp thay vì `st.empty()`
- Phân tích cấu trúc dictionary tin nhắn qua đầu ra `get_items()` trong thanh bên

---

### 8.2 Web Search Tool - Thêm khả năng tìm kiếm web

#### Chủ đề và mục tiêu

Thêm `WebSearchTool` vào agent để cung cấp chức năng tìm kiếm web thời gian thực, và xây dựng hệ thống quản lý trạng thái hiển thị tiến trình tìm kiếm trên giao diện theo thời gian thực.

#### Khái niệm chính

**`WebSearchTool`** là công cụ tích hợp do OpenAI Agents SDK cung cấp, cho phép agent tìm kiếm trên web những thông tin mới nhất không có trong dữ liệu huấn luyện. Điều quan trọng là chỉ định hướng dẫn sử dụng công cụ trong `instructions` của agent để chỉ ra khi nào nên thực hiện tìm kiếm web.

**Container trạng thái (`st.status`)** là widget hiển thị tiến trình do Streamlit cung cấp, thông báo trực quan cho người dùng về quá trình thực thi công cụ.

#### Phân tích mã

```python
from agents import Agent, Runner, SQLiteSession, WebSearchTool

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data. Use this tool when the users asks about current
              or future events, when you think you don't know the answer, try
              searching for it in the web first.
        """,
        tools=[
            WebSearchTool(),  # Đăng ký công cụ tìm kiếm web
        ],
    )
```

Các điều kiện sử dụng công cụ được chỉ định trong `instructions` của agent. Hướng dẫn nói rằng hãy thử tìm kiếm web trước "khi được hỏi về các sự kiện hiện tại hoặc tương lai" hoặc "khi không biết câu trả lời." Đây là kỹ thuật prompt engineering giúp cải thiện độ chính xác trong việc chọn công cụ.

```python
def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": (
            "🔍 Starting web search...",
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 Web search in progress...",
            "running",
        ),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)
```

Hàm `update_status()` đóng vai trò là **bộ điều phối sự kiện** cập nhật hiển thị trạng thái giao diện dựa trên loại sự kiện streaming. Các sự kiện liên quan đến tìm kiếm web được chia thành ba giai đoạn:

1. `in_progress` - Bắt đầu tìm kiếm
2. `searching` - Đang tìm kiếm
3. `completed` - Tìm kiếm hoàn tất

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)  # Container trạng thái
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(agent, message, session=session)

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)  # Cập nhật trạng thái

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response)
```

Khi khôi phục lịch sử hội thoại, bản ghi cuộc gọi tìm kiếm web cũng được hiển thị:

```python
if "type" in message and message["type"] == "web_search_call":
    with st.chat_message("ai"):
        st.write("🔍 Searched the web...")
```

#### Điểm thực hành

- Đặt câu hỏi thời gian thực như "Thời tiết hôm nay thế nào?" và xác minh tìm kiếm web được kích hoạt
- Quan sát quá trình chuyển đổi của container trạng thái (bắt đầu -> đang tiến hành -> hoàn tất)
- Xác minh tìm kiếm web không được kích hoạt khi hỏi các câu hỏi kiến thức chung có trong dữ liệu huấn luyện

---

### 8.3 File Search Tool - Công cụ tìm kiếm tệp và Vector Store

#### Chủ đề và mục tiêu

Thêm khả năng tìm kiếm nội dung tệp đã tải lên bằng `FileSearchTool` và Vector Store của OpenAI. Đồng thời cho phép người dùng tải lên tệp văn bản trực tiếp thông qua tính năng tải lên tệp của Streamlit.

#### Khái niệm chính

**Vector Store** là cơ sở dữ liệu vector được OpenAI lưu trữ, tự động nhúng (embedding) các tệp văn bản đã tải lên, cho phép tìm kiếm dựa trên ngữ nghĩa. `FileSearchTool` là công cụ cho phép agent tìm kiếm thông tin liên quan trong Vector Store.

**Quy trình tải lên tệp** bao gồm hai bước:
1. Tải tệp lên OpenAI bằng `client.files.create()`
2. Liên kết tệp với Vector Store bằng `client.vector_stores.files.create()`

#### Phân tích mã

```python
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool

client = OpenAI()

# Vector Store ID (tạo trước qua bảng điều khiển OpenAI hoặc API)
VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: ...
            - File Search Tool: Use this tool when the user asks a question
              about facts related to themselves. Or when they ask questions
              about specific files.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],  # Vector Store mục tiêu tìm kiếm
                max_num_results=3,                    # Số kết quả tìm kiếm tối đa
            ),
        ],
    )
```

`FileSearchTool` chỉ định Vector Store mục tiêu qua `vector_store_ids` và giới hạn số kết quả tối đa qua `max_num_results`.

Mã tải lên tệp và liên kết Vector Store:

```python
# Bật tính năng đính kèm tệp trong nhập chat
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt"],
)

if prompt:
    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    # Bước 1: Tải tệp lên OpenAI
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching file...")

                    # Bước 2: Liên kết tệp với Vector Store
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded", state="complete")
```

Tệp dữ liệu mẫu (`facts.txt`) được sử dụng trong dự án này chứa danh mục đầu tư giả định và hồ sơ chi tiêu. Sau khi tải lên, agent có thể trả lời các câu hỏi thông tin cá nhân như "Tôi có bao nhiêu cổ phiếu Apple?"

Lưu ý việc áp dụng `replace("$", "\$")` cho các phản hồi chứa ký hiệu `$` để ngăn vấn đề hiển thị LaTeX của Streamlit:

```python
st.write(message["content"][0]["text"].replace("$", "\$"))
```

#### Điểm thực hành

- Tải lên `facts.txt` và đặt câu hỏi như "Tổng giá trị danh mục đầu tư của tôi là bao nhiêu?"
- Tạo Vector Store trực tiếp từ bảng điều khiển OpenAI và thay thế ID
- So sánh và quan sát khi nào tìm kiếm tệp vs. tìm kiếm web được kích hoạt

---

### 8.4 Multi Modal Agent - Đầu vào hình ảnh đa phương tiện

#### Chủ đề và mục tiêu

Thêm khả năng đa phương tiện để agent có thể nhận và phân tích hình ảnh làm đầu vào. Khi người dùng tải lên hình ảnh, mã hóa nó bằng Base64, lưu vào phiên và cho phép agent hiểu nó.

#### Khái niệm chính

**Đa phương tiện (Multi-Modal)** đề cập đến khả năng xử lý nhiều loại đầu vào ngoài văn bản, như hình ảnh và âm thanh. Các mô hình dòng GPT-4 của OpenAI có thể nhận hình ảnh làm đầu vào và phân tích, mô tả nội dung của chúng.

**Mã hóa Base64** được sử dụng để truyền hình ảnh đến API. Dữ liệu byte hình ảnh được chuyển đổi thành chuỗi Base64, sau đó được truyền dưới dạng Data URI có định dạng `data:image/png;base64,...`.

#### Phân tích mã

```python
import base64

# Cho phép tệp hình ảnh trong nhập chat
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=[
        "txt",
        "jpg",
        "jpeg",
        "png",
    ],
)
```

Xử lý tải lên hình ảnh:

```python
elif file.type.startswith("image/"):
    with st.status("⏳ Uploading image...") as status:
        file_bytes = file.getvalue()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{file.type};base64,{base64_data}"

        # Lưu hình ảnh dưới dạng tin nhắn người dùng trong phiên
        asyncio.run(
            session.add_items(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": data_uri,
                            }
                        ],
                    }
                ]
            )
        )
        status.update(label="✅ Image uploaded", state="complete")
    with st.chat_message("human"):
        st.image(data_uri)
```

Điểm mấu chốt là thêm hình ảnh trực tiếp vào lịch sử hội thoại qua `session.add_items()`. Nó sử dụng loại `input_image` và trường `image_url` theo yêu cầu định dạng API OpenAI. `detail: "auto"` cho phép mô hình tự động xác định độ phân giải hình ảnh.

Sửa đổi `paint_history()` để hiển thị hình ảnh khi khôi phục lịch sử hội thoại:

```python
if message["role"] == "user":
    content = message["content"]
    if isinstance(content, str):
        st.write(content)           # Tin nhắn văn bản
    elif isinstance(content, list):
        for part in content:
            if "image_url" in part:
                st.image(part["image_url"])  # Tin nhắn hình ảnh
```

`content` của tin nhắn người dùng có thể là chuỗi (văn bản thuần) hoặc danh sách (đa phương tiện). Kiểm tra `isinstance()` xử lý cả hai trường hợp.

#### Điểm thực hành

- Tải lên hình ảnh biểu đồ hoặc đồ thị và hỏi "Bạn thấy gì trong hình ảnh này?"
- Tải lên hình ảnh, sau đó đặt câu hỏi văn bản tiếp theo để kiểm tra agent có nhớ ngữ cảnh hình ảnh không
- Phân tích cấu trúc của Data URI mã hóa Base64

---

### 8.5 Image Generation Tool - Công cụ tạo hình ảnh

#### Chủ đề và mục tiêu

Thêm `ImageGenerationTool` để cho phép agent tạo hình ảnh theo yêu cầu của người dùng. Đồng thời triển khai kỹ thuật hiển thị kết quả trung gian (hình ảnh một phần) theo thời gian thực trong quá trình tạo.

#### Khái niệm chính

**`ImageGenerationTool`** bọc API tạo hình ảnh của OpenAI (DALL-E) để agent có thể gọi nó như một công cụ. Thông qua cài đặt `partial_images`, có thể nhận hình ảnh xem trước độ phân giải thấp trong quá trình tạo, cung cấp phản hồi tiến trình trực quan cho người dùng.

#### Phân tích mã

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
)

# Thêm ImageGenerationTool vào danh sách công cụ của agent
ImageGenerationTool(
    tool_config={
        "type": "image_generation",
        "quality": "high",           # Tạo hình ảnh chất lượng cao
        "output_format": "jpeg",     # Định dạng đầu ra
        "partial_images": 1,         # Số hình ảnh xem trước trung gian
    }
),
```

Các tùy chọn chính của `tool_config`:
- `quality`: `"high"` hoặc `"standard"`. Chất lượng cao tinh tế hơn nhưng mất nhiều thời gian hơn
- `output_format`: `"jpeg"` hoặc `"png"`
- `partial_images`: Số hình ảnh trung gian nhận được trong quá trình tạo. Đặt từ 1 trở lên cho phép hiệu ứng hiển thị tiến trình

Xử lý sự kiện hình ảnh trong streaming:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        image_placeholder = st.empty()  # Thêm placeholder hình ảnh
        response = ""

        # ... vòng lặp streaming ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # Hiển thị kết quả trung gian tạo hình ảnh
                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)

                elif event.data.type == "response.completed":
                    image_placeholder.empty()
                    text_placeholder.empty()
```

Hình ảnh trung gian (`partial_image`) được truyền dưới dạng mã hóa Base64, nên chúng được giải mã bằng `base64.b64decode()` và hiển thị bằng `st.image()`.

Hiển thị hình ảnh đã tạo khi khôi phục lịch sử hội thoại:

```python
elif message_type == "image_generation_call":
    image = base64.b64decode(message["result"])
    with st.chat_message("ai"):
        st.image(image)
```

Các thông báo trạng thái liên quan đến tạo hình ảnh cũng được thêm:

```python
"response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
"response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
```

#### Điểm thực hành

- Kiểm thử tạo hình ảnh với yêu cầu như "Vẽ bức tranh con mèo ăn pizza trong vũ trụ"
- Chuyển đổi `partial_images` giữa 0 và 1 và so sánh hiệu ứng xem trước
- Xác minh hình ảnh đã tạo được lưu trong lịch sử hội thoại và hiển thị sau khi làm mới

---

### 8.6 Code Interpreter Tool - Công cụ thực thi mã

#### Chủ đề và mục tiêu

Thêm `CodeInterpreterTool` để cho phép agent viết và thực thi mã Python cho các phép tính, phân tích dữ liệu, tạo biểu đồ và nhiều hơn nữa.

#### Khái niệm chính

**`CodeInterpreterTool`** là công cụ cho phép thực thi mã Python trong môi trường sandbox do OpenAI lưu trữ. Khi agent viết mã, nó được thực thi trong container an toàn và kết quả được trả về. Hữu ích cho các phép tính toán học, phân tích dữ liệu, trực quan hóa, v.v.

Cài đặt `container` với `"type": "auto"` cho phép OpenAI tự động chọn môi trường thực thi phù hợp.

#### Phân tích mã

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool, CodeInterpreterTool,
)

CodeInterpreterTool(
    tool_config={
        "type": "code_interpreter",
        "container": {
            "type": "auto",      # Tự động chọn container
        },
    }
),
```

Xử lý streaming của quá trình thực thi mã:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        code_placeholder = st.empty()    # Placeholder để hiển thị mã
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""
        code_response = ""

        # Lưu placeholder vào session_state (để dọn dẹp trong lần chạy tiếp theo)
        st.session_state["code_placeholder"] = code_placeholder
        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        # ... vòng lặp streaming ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # Hiển thị quá trình viết mã theo thời gian thực
                if event.data.type == "response.code_interpreter_call_code.delta":
                    code_response += event.data.delta
                    code_placeholder.code(code_response)  # Hiển thị dưới dạng khối mã
```

`st.code()` hiển thị khối mã với tô sáng cú pháp. Khả năng theo dõi quá trình viết mã theo thời gian thực nâng cao trải nghiệm người dùng.

Mã dọn dẹp placeholder trước đó khi chạy tin nhắn tiếp theo:

```python
if prompt:
    # Dọn dẹp placeholder trước đó
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()
```

Nếu không có mã dọn dẹp này, các placeholder streaming từ phản hồi trước sẽ vẫn còn trên màn hình và gây hiển thị trùng lặp với tin nhắn được khôi phục bởi `paint_history()`.

Các thông báo trạng thái liên quan đến thực thi mã:

```python
"response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
"response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
```

#### Điểm thực hành

- Thử yêu cầu thực thi mã như "Tính 20 số hạng đầu tiên của dãy Fibonacci"
- Cũng thử yêu cầu trực quan hóa như "Vẽ đồ thị hàm sin"
- Quan sát quá trình viết mã theo thời gian thực

---

### 8.7 Hosted MCP Tool - Tích hợp công cụ MCP được lưu trữ

#### Chủ đề và mục tiêu

Sử dụng **HostedMCPTool** để kết nối với máy chủ MCP được lưu trữ bên ngoài (Context7) và thêm khả năng tìm kiếm tài liệu dự án phần mềm.

#### Khái niệm chính

**MCP (Model Context Protocol)** là giao thức mở cho phép mô hình AI tương tác với các công cụ và nguồn dữ liệu bên ngoài. Thông qua MCP, agent có thể tận dụng các khả năng đa dạng do bên thứ ba cung cấp ngoài các công cụ tích hợp sẵn.

**HostedMCPTool** kết nối với máy chủ MCP được công khai trên internet qua HTTP. Thiết lập đơn giản vì chỉ cần biết URL máy chủ.

**Context7** là máy chủ MCP cung cấp tài liệu cập nhật cho các dự án phần mềm, cho phép agent tìm kiếm tài liệu chính thức của các thư viện hoặc framework cụ thể.

#### Phân tích mã

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
    CodeInterpreterTool, HostedMCPTool,
)

HostedMCPTool(
    tool_config={
        "server_url": "https://mcp.context7.com/mcp",  # URL máy chủ MCP
        "type": "mcp",
        "server_label": "Context7",                      # Nhãn hiển thị
        "server_description": "Use this to get the docs from software projects.",
        "require_approval": "never",                     # Tự động phê duyệt (không cần xác nhận người dùng)
    }
),
```

Các trường chính của `tool_config`:
- `server_url`: URL endpoint của máy chủ MCP
- `server_label`: Tên máy chủ hiển thị trên giao diện
- `server_description`: Mô tả được agent sử dụng để xác định khi nào dùng công cụ này
- `require_approval`: Đặt thành `"never"` để tự động gọi công cụ mà không cần phê duyệt

Khôi phục lịch sử hội thoại liên quan đến MCP:

```python
elif message_type == "mcp_list_tools":
    with st.chat_message("ai"):
        st.write(f"Listed {message['server_label']}'s tools")
elif message_type == "mcp_call":
    with st.chat_message("ai"):
        st.write(
            f"Called {message['server_label']}'s {message['name']} "
            f"with args {message['arguments']}"
        )
```

Cuộc gọi MCP diễn ra theo hai giai đoạn:
1. **`mcp_list_tools`**: Truy vấn danh sách công cụ có sẵn từ máy chủ
2. **`mcp_call`**: Thực sự gọi một công cụ cụ thể với các tham số

Các thông báo trạng thái liên quan đến MCP:

```python
"response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
"response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
"response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
"response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
"response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
"response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
```

#### Điểm thực hành

- Kiểm thử MCP Context7 với câu hỏi như "Hướng dẫn cách sử dụng st.chat_input của Streamlit"
- Quan sát xem `mcp_list_tools` và `mcp_call` có xảy ra tuần tự trong cuộc gọi MCP không
- Thay đổi `require_approval` thành `"always"` và quan sát sự khác biệt

---

### 8.8 Local MCP Server - Tích hợp máy chủ MCP cục bộ

#### Chủ đề và mục tiêu

Sử dụng **`MCPServerStdio`** để kết nối với máy chủ MCP chạy cục bộ (Yahoo Finance). Qua đó, hiểu sự khác biệt giữa MCP được lưu trữ và MCP cục bộ, và tái cấu trúc việc tạo agent sang mẫu async context manager.

#### Khái niệm chính

**`MCPServerStdio`** chạy máy chủ MCP như một tiến trình cục bộ và giao tiếp qua đầu vào/đầu ra tiêu chuẩn (stdin/stdout). Nó sử dụng `uvx` (trình chạy gói Python) để thực thi trực tiếp các gói máy chủ MCP.

**Sự khác biệt giữa Hosted MCP và Local MCP**:
| Thuộc tính | Hosted MCP | Local MCP |
|------|-----------|-----------|
| Vị trí thực thi | Máy chủ từ xa | Máy cục bộ |
| Phương thức kết nối | HTTP | stdin/stdout |
| Cách cấu hình | Chỉ định URL | Chỉ định lệnh thực thi |
| Vòng đời | Luôn sẵn sàng | Cần khởi động/dừng tiến trình |

Máy chủ MCP cục bộ phải được quản lý vòng đời bằng câu lệnh `async with` (async context manager). Vì điều này, vị trí tạo agent chuyển từ khởi tạo `session_state` sang bên trong hàm `run_agent()`.

#### Phân tích mã

```python
from agents.mcp.server import MCPServerStdio

async def run_agent(message):
    # Định nghĩa máy chủ MCP cục bộ
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",                    # Lệnh thực thi
            "args": ["mcp-yahoo-finance"],       # Tên gói
        },
        cache_tools_list=True,  # Cache danh sách công cụ để tối ưu hiệu suất
    )

    # Quản lý vòng đời máy chủ bằng async context manager
    async with yfinance_server:

        # Tạo Agent bên trong context (máy chủ MCP phải đang hoạt động)
        agent = Agent(
            mcp_servers=[
                yfinance_server,       # Kết nối máy chủ MCP cục bộ
            ],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.
        ...
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID],
                    max_num_results=3,
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        # Bây giờ sử dụng agent để thực thi streaming
        with st.chat_message("ai"):
            # ... logic streaming tương tự như trước ...
```

**Thay đổi cấu trúc quan trọng**: Cho đến phần trước, Agent được tạo một lần trong `st.session_state` và tái sử dụng. Tuy nhiên, vì máy chủ MCP cục bộ chỉ hợp lệ bên trong khối `async with`, Agent cũng phải được tạo lại bên trong khối đó mỗi lần. Điều này có một chút overhead hiệu suất nhưng là sự đánh đổi cần thiết để quản lý vòng đời ổn định của máy chủ MCP cục bộ.

`cache_tools_list=True` cache danh sách công cụ của máy chủ MCP để không cần truy vấn danh sách công cụ mỗi lần. Hữu ích cho các máy chủ có danh sách công cụ không thay đổi thường xuyên.

#### Điểm thực hành

- Kiểm thử Yahoo Finance MCP với câu hỏi tài chính như "Cho tôi biết giá cổ phiếu Apple hiện tại"
- Chạy `uvx mcp-yahoo-finance` trực tiếp trong terminal và quan sát hành vi máy chủ MCP
- Kiểm tra lỗi gì xảy ra khi tạo Agent bên ngoài khối `async with`

---

### 8.9 Conclusions - Thêm máy chủ MCP cục bộ thứ hai

#### Chủ đề và mục tiêu

Thêm máy chủ MCP cục bộ thứ hai (máy chủ múi giờ) để học mẫu sử dụng nhiều máy chủ MCP đồng thời.

#### Khái niệm chính

Câu lệnh `async with` của Python có thể quản lý **nhiều context manager đồng thời** bằng cách phân tách chúng bằng dấu phẩy (`,`). Điều này cho phép chạy nhiều máy chủ MCP cục bộ cùng lúc và kết nối tất cả với cùng một Agent.

#### Phân tích mã

```python
async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-yahoo-finance"],
        },
        cache_tools_list=True,
    )

    # Máy chủ MCP cục bộ thứ hai: cung cấp thông tin múi giờ
    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    # Quản lý cả hai máy chủ đồng thời như context manager
    async with yfinance_server, timezone_server:

        agent = Agent(
            mcp_servers=[
                yfinance_server,
                timezone_server,      # Thêm máy chủ MCP thứ hai
            ],
            name="ChatGPT Clone",
            # ... phần còn lại của cấu hình giống nhau ...
        )
```

`mcp-server-time` có thể chỉ định múi giờ mặc định bằng tham số `--local-timezone`. Agent có thể sử dụng máy chủ này để thực hiện các tác vụ như lấy thời gian hiện tại ở múi giờ cụ thể hoặc chuyển đổi giữa các múi giờ.

Cú pháp `async with yfinance_server, timezone_server:` khởi động cả hai máy chủ đồng thời và tắt sạch cả hai khi kết thúc khối. Ngay cả khi một lỗi xảy ra, các máy chủ còn lại vẫn được dọn dẹp đúng cách.

#### Điểm thực hành

- Kiểm thử MCP múi giờ với câu hỏi như "Bây giờ mấy giờ ở New York?"
- Thử câu hỏi kết hợp sử dụng nhiều máy chủ MCP trong một cuộc hội thoại (ví dụ: "Cho tôi biết giá cổ phiếu Apple và giờ hiện tại ở New York")
- Tìm và thêm các gói máy chủ MCP mới

---

## 3. Tóm tắt trọng điểm chương

### Mẫu kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │chat_input│  │chat_msg  │  │  sidebar (debug)   │   │
│  └────┬─────┘  └──────────┘  └────────────────────┘   │
│       │                                                 │
│       v                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              run_agent()                         │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  async with MCP_Server_1, MCP_Server_2  │    │   │
│  │  │  ┌───────────────────────────────────┐  │    │   │
│  │  │  │           Agent                   │  │    │   │
│  │  │  │  ┌──────────┐ ┌──────────────┐   │  │    │   │
│  │  │  │  │WebSearch │ │ FileSearch   │   │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │ImageGen  │ │CodeInterpreter│  │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │HostedMCP │ │ Local MCP x2 │  │  │    │   │
│  │  │  │  └──────────┘ └──────────────┘   │  │    │   │
│  │  │  └───────────────────────────────────┘  │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│              ┌─────────────────┐                       │
│              │  SQLiteSession   │                       │
│              │  (Lưu lịch sử)  │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### Tóm tắt công cụ

| Công cụ | Mục đích | Loại sự kiện chính |
|------|------|-----------------|
| **WebSearchTool** | Tìm kiếm thông tin web thời gian thực | `response.web_search_call.*` |
| **FileSearchTool** | Tìm kiếm nội dung tệp dựa trên Vector Store | `response.file_search_call.*` |
| **ImageGenerationTool** | Tạo hình ảnh DALL-E | `response.image_generation_call.*` |
| **CodeInterpreterTool** | Thực thi mã Python | `response.code_interpreter_call*` |
| **HostedMCPTool** | Tích hợp máy chủ MCP từ xa | `response.mcp_call.*`, `response.mcp_list_tools.*` |
| **MCPServerStdio** | Tích hợp máy chủ MCP cục bộ | `response.mcp_call.*`, `response.mcp_list_tools.*` |

### Tóm tắt khái niệm chính

1. **`st.session_state`**: Cơ chế quản lý trạng thái của Streamlit. Duy trì dữ liệu qua các lần chạy lại script.

2. **`st.empty()`**: Placeholder có thể thay thế nội dung sau. Thiết yếu cho hiển thị văn bản streaming.

3. **`st.status()`**: Widget hiển thị tiến trình tác vụ trực quan. Hỗ trợ trạng thái `running`/`complete`.

4. **`SQLiteSession`**: Tự động lưu/khôi phục lịch sử hội thoại vào DB SQLite. Truyền cho Runner qua tham số `session`.

5. **Mẫu sự kiện streaming**: Xác định loại sự kiện qua `raw_response_event` > `event.data.type` và thực hiện cập nhật giao diện phù hợp cho sự kiện của mỗi công cụ.

6. **`async with` context manager**: Quản lý an toàn vòng đời (khởi động/dừng) của máy chủ MCP cục bộ. Nhiều máy chủ có thể được quản lý đồng thời bằng cách nối chúng bằng dấu phẩy.

7. **Vector Store**: DB vector do OpenAI lưu trữ, tự động nhúng tệp đã tải lên, cho phép tìm kiếm dựa trên ngữ nghĩa.

---

## 4. Bài tập thực hành

### Bài tập 1: Cơ bản - Hệ thống ghi nhật ký sử dụng công cụ

Mở rộng hàm `update_status()` để tạo hệ thống ghi nhật ký tất cả cuộc gọi công cụ kèm timestamp vào tệp nhật ký. Triển khai để có thể xem nhật ký trong thanh bên.

**Gợi ý**:
- Sử dụng module `logging` của Python
- Hiển thị nội dung nhật ký trong `st.sidebar`
- Ghi lại thời gian bắt đầu, thời gian kết thúc và thời lượng của mỗi cuộc gọi công cụ

### Bài tập 2: Trung cấp - Thêm máy chủ MCP mới

Tìm và thêm `mcp-server-fetch` (để lấy nội dung trang web) hoặc gói máy chủ MCP khác vào dự án. Kết nối như máy chủ MCP cục bộ và sửa đổi `instructions` để agent có thể sử dụng công cụ phù hợp.

**Gợi ý**:
- Tìm trên PyPI các gói máy chủ MCP có thể chạy bằng `uvx`
- Đặt lệnh và tham số phù hợp trong `params` của `MCPServerStdio`
- Thêm máy chủ mới vào câu lệnh `async with` và đưa vào danh sách `mcp_servers`

### Bài tập 3: Trung cấp - Quản lý nhiều phiên hội thoại

Hiện tại chỉ hỗ trợ một phiên hội thoại. Triển khai chức năng tạo và chuyển đổi giữa nhiều phiên hội thoại từ thanh bên.

**Gợi ý**:
- Thay đổi động tham số đầu tiên (session ID) của `SQLiteSession`
- Hiển thị danh sách phiên bằng `st.sidebar.selectbox()`
- Gán ID duy nhất khi tạo phiên mới

### Bài tập 4: Nâng cao - Tích hợp công cụ tùy chỉnh

Sử dụng decorator `@function_tool` để tạo hàm Python tùy chỉnh thành công cụ và đăng ký chúng cùng với các công cụ tích hợp sẵn trong agent. Ví dụ, bạn có thể tạo công cụ khám phá hệ thống tệp cục bộ hoặc thực hiện các phép tính đơn giản.

**Gợi ý**:
- Áp dụng mẫu `@function_tool` đã học trong các chương trước
- Docstring của công cụ ảnh hưởng đến việc chọn công cụ của agent
- Chỉ định điều kiện sử dụng cho công cụ mới trong `instructions`

### Bài tập 5: Nâng cao - Mở rộng đa phương tiện đầu vào âm thanh

Mở rộng chức năng đa phương tiện hiện tại chỉ hỗ trợ hình ảnh sang hỗ trợ thêm tệp âm thanh. Triển khai bằng cách sử dụng Whisper API của OpenAI để chuyển đổi giọng nói thành văn bản trước khi truyền cho agent.

**Gợi ý**:
- Thêm định dạng âm thanh vào `file_type` của `st.chat_input`
- Chuyển đổi giọng nói thành văn bản bằng `client.audio.transcriptions.create()`
- Truyền văn bản đã chuyển đổi cho agent như tin nhắn thông thường

---

## Phụ lục: Mã hoàn chỉnh cuối cùng (main.py)

Dưới đây là `main.py` cuối cùng với tất cả tính năng Chương 8 được tích hợp:

```python
import dotenv

dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    HostedMCPTool,
)
from agents.mcp.server import MCPServerStdio

client = OpenAI()

VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"


if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )
session = st.session_state["session"]


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)
            elif message_type == "code_interpreter_call":
                with st.chat_message("ai"):
                    st.code(message["code"])
            elif message_type == "mcp_list_tools":
                with st.chat_message("ai"):
                    st.write(f"Listed {message['server_label']}'s tools")
            elif message_type == "mcp_call":
                with st.chat_message("ai"):
                    st.write(
                        f"Called {message['server_label']}'s {message['name']} "
                        f"with args {message['arguments']}"
                    )


asyncio.run(paint_history())


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
        "response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
        "response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
        "response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
        "response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
        "response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
        "response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
        "response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
        "response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
        "response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={"command": "uvx", "args": ["mcp-yahoo-finance"]},
        cache_tools_list=True,
    )

    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    async with yfinance_server, timezone_server:
        agent = Agent(
            mcp_servers=[yfinance_server, timezone_server],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data.
            - File Search Tool: Use this tool when the user asks a question about
              facts related to themselves.
            - Code Interpreter Tool: Use this tool when you need to write and run
              code to answer the user's question.
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID], max_num_results=3
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        with st.chat_message("ai"):
            status_container = st.status("⏳", expanded=False)
            code_placeholder = st.empty()
            image_placeholder = st.empty()
            text_placeholder = st.empty()
            response = ""
            code_response = ""

            st.session_state["code_placeholder"] = code_placeholder
            st.session_state["image_placeholder"] = image_placeholder
            st.session_state["text_placeholder"] = text_placeholder

            stream = Runner.run_streamed(agent, message, session=session)

            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    update_status(status_container, event.data.type)

                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                    if event.data.type == "response.code_interpreter_call_code.delta":
                        code_response += event.data.delta
                        code_placeholder.code(code_response)

                    elif event.data.type == "response.image_generation_call.partial_image":
                        image = base64.b64decode(event.data.partial_image_b64)
                        image_placeholder.image(image)

prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
)

if prompt:
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()), purpose="user_data"
                    )
                    status.update(label="⏳ Attaching file...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID, file_id=uploaded_file.id
                    )
                    status.update(label="✅ File uploaded", state="complete")
        elif file.type.startswith("image/"):
            with st.status("⏳ Uploading image...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")
            with st.chat_message("human"):
                st.image(data_uri)

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```
