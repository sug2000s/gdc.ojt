# Chapter 21: Triển khai AI Agent (Deployment)

## Tổng quan chương

Trong chương này, chúng ta sẽ học toàn bộ quy trình **triển khai AI agent được xây dựng bằng OpenAI Agents SDK lên môi trường production thực tế**. Vượt qua việc chỉ chạy agent trên local, chúng ta sẽ đề cập đến: bọc thành REST API bằng framework web FastAPI, quản lý trạng thái hội thoại sử dụng Conversations API của OpenAI, xử lý phản hồi đồng bộ/streaming, và cuối cùng triển khai lên nền tảng cloud Railway.

### Mục tiêu học tập

- Hiểu cách bọc AI agent thành REST API bằng FastAPI
- Nắm vững cách quản lý trạng thái hội thoại (context) bằng OpenAI Conversations API
- Học sự khác biệt và cách triển khai phản hồi đồng bộ (Sync) và streaming
- Thực hành triển khai cloud bằng nền tảng Railway

### Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|------|------|------|
| Python | 3.13 | Ngôn ngữ lập trình |
| FastAPI | 0.118.3 | Framework web |
| OpenAI Agents SDK | 0.3.3 | Framework AI agent |
| Uvicorn | 0.37.0 | ASGI server |
| python-dotenv | 1.1.1 | Quản lý biến môi trường |
| Railway | - | Nền tảng triển khai cloud |

---

## 21.0 Introduction - Thiết lập ban đầu dự án

### Chủ đề và mục tiêu

Tạo khung cơ bản cho dự án triển khai. Sử dụng `uv` (trình quản lý gói Python) để khởi tạo dự án Python mới và thiết lập dependency cần thiết.

### Giải thích khái niệm cốt lõi

#### Cấu trúc dự án

Trong chương này, chúng ta tạo thư mục độc lập `deployment/` tách biệt khỏi dự án masterclass hiện có. Điều này nhằm thiết kế như ứng dụng độc lập có thể triển khai.

```
deployment/
├── .python-version    # Chỉ định phiên bản Python (3.13)
├── README.md          # Mô tả dự án
├── main.py            # File ứng dụng chính
└── pyproject.toml     # Metadata và dependency dự án
```

#### pyproject.toml - Quản lý dependency

`pyproject.toml` là file cấu hình tiêu chuẩn của dự án Python hiện đại. File này khai báo metadata và dependency của dự án.

```toml
[project]
name = "deployment"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi==0.118.3",
    "openai-agents==0.3.3",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

Vai trò của mỗi dependency:

- **`fastapi`**: Framework web Python hiệu suất cao. Cung cấp tự động tạo tài liệu API, kiểm tra kiểu, v.v.
- **`openai-agents`**: SDK agent chính thức của OpenAI. Cung cấp các class cốt lõi cần thiết để thực thi agent như Agent, Runner.
- **`python-dotenv`**: Tải biến môi trường từ file `.env`. Sử dụng để tách thông tin nhạy cảm như API key khỏi code.
- **`uvicorn`**: ASGI server. Server cho phép ứng dụng FastAPI thực sự nhận HTTP request.

#### main.py ban đầu

```python
def main():
    print("Hello from deployment!")


if __name__ == "__main__":
    main()
```

Tại thời điểm này, `main.py` vẫn chỉ là code skeleton đơn giản. Từ phần tiếp theo, chúng ta sẽ bắt đầu chuyển đổi thành ứng dụng FastAPI.

### Điểm thực hành

1. Tạo dự án mới bằng lệnh `uv init deployment`.
2. Thêm dependency bằng `uv add fastapi openai-agents python-dotenv uvicorn`.
3. Kiểm tra file `.python-version` có được thiết lập là `3.13` không.

---

## 21.1 Conversations API - Xây dựng API quản lý hội thoại

### Chủ đề và mục tiêu

Xây dựng REST API để quản lý hội thoại (conversation) với AI agent bằng FastAPI. Sử dụng **Conversations API** của OpenAI để tạo phiên hội thoại và tạo endpoint cho phép thêm tin nhắn vào mỗi hội thoại.

### Giải thích khái niệm cốt lõi

#### OpenAI Conversations API là gì?

Conversations API là tính năng quản lý trạng thái hội thoại do OpenAI cung cấp. Trước đây phải tự quản lý lịch sử hội thoại (history), nhưng sử dụng Conversations API thì phía server OpenAI sẽ duy trì trạng thái hội thoại.

Luồng cốt lõi:
1. Tạo phiên hội thoại mới bằng `client.conversations.create()` sẽ nhận `conversation_id` duy nhất.
2. Sau đó khi thực thi agent, truyền `conversation_id` này thì OpenAI tự động duy trì ngữ cảnh hội thoại trước đó.

Ưu điểm của phương pháp này:
- **Tương thích serverless**: Không cần lưu trạng thái trên server, nên hội thoại được duy trì ngay cả khi server khởi động lại.
- **Triển khai đơn giản**: Không cần logic phức tạp quản lý mảng lịch sử hội thoại trực tiếp.
- **Khả năng mở rộng**: Có thể tiếp tục cùng hội thoại trên nhiều instance server.

#### Cấu trúc ứng dụng FastAPI

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner
```

**Lưu ý**: `load_dotenv()` được gọi **trước** `from agents import ...`. Vì module `agents` tham chiếu biến môi trường (đặc biệt `OPENAI_API_KEY`) khi import. Nếu đảo ngược thứ tự, sẽ phát sinh lỗi không tìm thấy API key.

#### Định nghĩa Agent

```python
agent = Agent(
    name="Assistant",
    instructions="You help users with their questions."
)
```

Agent được tạo chỉ một lần ở cấp module. Không cần tạo mới cho mỗi request. `instructions` đóng vai trò system prompt của agent.

#### Khởi tạo FastAPI app và OpenAI client

```python
app = FastAPI()
client = AsyncOpenAI()
```

`AsyncOpenAI()` là client OpenAI bất đồng bộ (async). Vì FastAPI là framework bất đồng bộ, sử dụng client bất đồng bộ phù hợp hơn về mặt hiệu suất.

#### Endpoint tạo hội thoại

```python
class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }
```

Điểm cốt lõi của code:

1. **Pydantic BaseModel**: `CreateConversationResponse` định nghĩa schema phản hồi. FastAPI tự động serialize thành JSON và phản ánh vào tài liệu Swagger.
2. **`client.conversations.create()`**: Gọi OpenAI API để tạo phiên hội thoại mới. Trường `.id` của giá trị trả về chứa ID duy nhất bắt đầu bằng `conv_`.
3. **Xử lý bất đồng bộ**: Sử dụng keyword `await` để chờ bất đồng bộ đến khi lời gọi API hoàn thành.

#### Endpoint tin nhắn (skeleton)

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(conversation_id: str):
    pass
```

Tại thời điểm này, endpoint tin nhắn chưa được triển khai. Đường dẫn URL chứa `{conversation_id}`, tạo cấu trúc cho phép gửi tin nhắn đến hội thoại cụ thể.

### Phân tích code - Toàn bộ luồng

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │                            │── conversations.create()─>│
   │                            │<── đối tượng conversation──│
   │<── { conversation_id } ────│                          │
```

### Điểm thực hành

1. Khởi động dev server bằng lệnh `uvicorn main:app --reload`.
2. Truy cập `http://127.0.0.1:8000/docs` trên trình duyệt để xem tài liệu Swagger tự động tạo của FastAPI.
3. Gọi endpoint `POST /conversations` để kiểm tra `conversation_id` được trả về bình thường.

---

## 21.2 Sync Responses - Triển khai phản hồi đồng bộ

### Chủ đề và mục tiêu

Hoàn thiện endpoint gửi tin nhắn đến hội thoại và nhận phản hồi agent **một cách đồng bộ**. Sử dụng `Runner.run()` để thực thi agent và truyền toàn bộ phản hồi đã hoàn thành cho client một lần.

### Giải thích khái niệm cốt lõi

#### Phản hồi đồng bộ vs Phản hồi streaming

| Đặc tính | Phản hồi đồng bộ (Sync) | Phản hồi streaming |
|------|-----------------|------------------------|
| Cách phản hồi | Gửi một lần sau khi hoàn thành toàn bộ | Gửi thời gian thực theo đơn vị token |
| Trải nghiệm người dùng | Có thời gian chờ đến phản hồi | Văn bản xuất hiện ngay lập tức |
| Độ khó triển khai | Tương đối đơn giản | Cần xử lý event stream |
| Tình huống phù hợp | Giao tiếp giữa backend, phản hồi ngắn | UI hướng người dùng, phản hồi dài |

Trong phần này, chúng ta triển khai phản hồi đồng bộ trước.

#### Định nghĩa model request/response

```python
class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str
```

Sử dụng Pydantic `BaseModel` để định nghĩa schema đầu vào/đầu ra nghiêm ngặt.

- **`CreateMessageInput`**: Body request mà client gửi. Câu hỏi người dùng nằm trong trường `question`.
- **`CreateMessageOutput`**: Phản hồi server trả về. Câu trả lời agent nằm trong trường `answer`.

FastAPI dựa trên model này:
- Tự động parse JSON body request và kiểm tra kiểu.
- Tự động trả về 422 Validation Error khi nhận request sai định dạng.

#### Endpoint xử lý tin nhắn

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }
```

Điểm cốt lõi của code:

1. **Tham số đường dẫn `conversation_id`**: Được trích xuất từ URL và truyền làm tham số hàm. Xác định hội thoại nào sẽ nhận tin nhắn.
2. **Tham số body `message_input`**: FastAPI tự động chuyển đổi JSON body request thành đối tượng `CreateMessageInput`.
3. **`Runner.run()`**: Phương thức cốt lõi thực thi agent.
   - `starting_agent`: Đối tượng agent cần thực thi
   - `input`: Văn bản câu hỏi người dùng
   - `conversation_id`: ID hội thoại của OpenAI Conversations API. Đây là **cốt lõi duy trì ngữ cảnh hội thoại**.
4. **`answer.final_output`**: Trích xuất đầu ra văn bản cuối cùng của agent từ giá trị trả về của `Runner.run()`.

#### Test API (api.http)

```http
POST http://127.0.0.1:8000/conversations

###

POST http://127.0.0.1:8000/conversations/conv_68ecdf11ff6081969cc4e8e9d126c015082054e6371dc260/message
Content-Type: application/json

{
    "question": "What is the first question i asked you?"
}
```

File `api.http` là file test HTTP request được sử dụng bởi extension REST Client của VS Code, v.v. Phân tách request bằng `###`, có thể thực thi từng request riêng lẻ.

Câu hỏi "What is the first question i asked you?" trong test trên nhằm **xác minh duy trì ngữ cảnh hội thoại**. Vì nội dung hội thoại trước đó được duy trì thông qua `conversation_id`, agent có thể nhớ và trả lời câu hỏi đã nhận trước đó.

### Phân tích code - Toàn bộ luồng

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │<── { conversation_id } ────│                          │
   │                            │                          │
   │── POST /conversations/     │                          │
   │   {id}/message ───────────>│                          │
   │   { question: "..." }      │── Runner.run() ─────────>│
   │                            │   (bao gồm conversation_id)│
   │                            │<── phản hồi hoàn thành ───│
   │<── { answer: "..." } ──────│                          │
```

### Điểm thực hành

1. Sau khi tạo hội thoại, sử dụng `conversation_id` trả về để gửi nhiều tin nhắn.
2. Nói "My name is [tên]" rồi hỏi "What is my name?" để xác nhận ngữ cảnh hội thoại được duy trì.
3. Xác minh hội thoại được duy trì độc lập khi sử dụng `conversation_id` khác nhau.

---

## 21.3 StreamingResponse - Triển khai phản hồi streaming

### Chủ đề và mục tiêu

Triển khai endpoint truyền phản hồi agent dưới dạng **streaming thời gian thực**. Nhờ đó, người dùng có thể xem quá trình agent tạo câu trả lời thời gian thực. Triển khai cả hai phương thức streaming (chỉ văn bản / toàn bộ event).

### Giải thích khái niệm cốt lõi

#### Sự cần thiết của phản hồi streaming

Phản hồi đồng bộ (`Runner.run()`) yêu cầu client phải chờ đến khi toàn bộ câu trả lời được tạo. Với câu trả lời dài, có thể mất vài giây đến vài chục giây, trải nghiệm người dùng không tốt.

Phản hồi streaming (`Runner.run_streamed()`) gửi đến client theo đơn vị token ngay khi câu trả lời được tạo. Cùng nguyên lý với việc văn bản xuất hiện từng ký tự trên giao diện web ChatGPT.

#### FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse
```

`StreamingResponse` của FastAPI nhận hàm generator và gửi dữ liệu đến client theo đơn vị chunk. Phương thức gửi dữ liệu dần dần trong khi duy trì kết nối HTTP.

#### Phương pháp 1: Chỉ streaming text delta

```python
@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")
```

Phân tích code từng bước:

**Bước 1 - Gọi `Runner.run_streamed()`**:
```python
events = Runner.run_streamed(
    starting_agent=agent,
    input=message_input.question,
    conversation_id=conversation_id,
)
```
Sử dụng `Runner.run_streamed()` thay vì `Runner.run()`. Phương thức này không trả về kết quả một lần mà trả về đối tượng event stream.

**Bước 2 - Lọc event**:
```python
async for event in events.stream_events():
    if (
        event.type == "raw_response_event"
        and event.data.type == "response.output_text.delta"
    ):
        yield event.data.delta
```

`stream_events()` tạo ra nhiều loại event khác nhau. Ở đây lọc bằng hai điều kiện:
- `event.type == "raw_response_event"`: Event raw truyền trực tiếp từ OpenAI API
- `event.data.type == "response.output_text.delta"`: Event tương ứng với **phần thay đổi (delta)** của đầu ra văn bản

`yield` là cú pháp generator bất đồng bộ Python. Tạo dữ liệu từng cái một và truyền cho `StreamingResponse`.

**Bước 3 - Trả về StreamingResponse**:
```python
return StreamingResponse(event_generator(), media_type="text/plain")
```
Thiết lập `media_type="text/plain"` để streaming dưới dạng văn bản thuần. Client duy trì kết nối và nhận chunk văn bản tuần tự.

#### Phương pháp 2: Streaming toàn bộ event

```python
@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

Endpoint này streaming không chỉ text delta mà **tất cả raw_response_event** dưới dạng JSON.

Điểm khác biệt chính:
- Điều kiện lọc chỉ còn `event.type == "raw_response_event"` (loại bỏ điều kiện text delta).
- Chuyển đổi toàn bộ event thành chuỗi JSON bằng `event.data.to_json()`.
- Thêm `\n` (xuống dòng) sau mỗi event để client có thể phân biệt event.

Phương thức này hữu ích khi frontend cần kiểm soát chi tiết hơn. Ví dụ, có thể nhận tất cả event gọi tool, chuyển đổi agent, v.v. và phản ánh lên UI.

#### So sánh hai phương thức

| Đặc tính | `/message-stream` | `/message-stream-all` |
|------|-------------------|----------------------|
| Nội dung gửi | Chỉ mảnh văn bản | Tất cả event (JSON) |
| Định dạng dữ liệu | Văn bản thuần | JSON (phân tách xuống dòng) |
| Lượng dữ liệu | Ít | Nhiều |
| Mục đích phù hợp | UI chat đơn giản | UI nâng cao (hiển thị thực thi tool, v.v.) |

#### Test streaming bằng curl

```bash
curl -N -X POST http://127.0.0.1:8000/conversations/{conv_id}/message-stream \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the size of the great wall of china?"}'
```

Flag `-N` của `curl` vô hiệu hóa buffering đầu ra. Không có tùy chọn này, curl sẽ gom dữ liệu trong buffer và xuất ra một lần, không thể xác nhận hiệu ứng streaming.

### Điểm thực hành

1. Gọi endpoint `/message-stream` bằng `curl -N` để xác nhận văn bản được xuất thời gian thực.
2. Gọi endpoint `/message-stream-all` để quan sát những loại event nào được truyền.
3. So sánh tốc độ cảm nhận giữa phản hồi đồng bộ (`/message`) và phản hồi streaming (`/message-stream`).
4. Phân tích xem có những loại event nào ngoài `response.output_text.delta` trong event JSON được streaming.

---

## 21.4 Deployment - Triển khai cloud Railway

### Chủ đề và mục tiêu

Triển khai API AI agent đã hoàn thành lên nền tảng cloud **Railway** để biến thành dịch vụ thực tế có thể truy cập từ internet. Đề cập đến viết file cấu hình triển khai, quản lý biến môi trường, cài đặt bảo mật.

### Giải thích khái niệm cốt lõi

#### Railway là gì?

Railway là nền tảng triển khai cloud thân thiện với nhà phát triển. Khi kết nối repository Git, nó tự động build và triển khai, đồng thời dễ dàng quản lý biến môi trường, xem log, cấu hình domain.

Ưu điểm của Railway:
- Triển khai tự động dựa trên Git push (CI/CD)
- Build tự động bằng NIXPACKS (không cần Dockerfile)
- Cung cấp tier miễn phí
- Quản lý biến môi trường đơn giản

#### Thêm endpoint health check

```python
@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }
```

Thêm endpoint GET đơn giản ở đường dẫn gốc (`/`). Phục vụ nhiều mục đích:
- **Health Check**: Dùng để xác nhận server hoạt động bình thường. Nền tảng cloud như Railway gọi định kỳ endpoint này để kiểm tra trạng thái dịch vụ.
- **Kiểm tra nhanh**: Có thể xác nhận ngay dịch vụ đang hoạt động khi truy cập URL đã triển khai trên trình duyệt.
- **Hàm đồng bộ**: Định nghĩa bằng `def` thay vì `async def`. Vì không có lời gọi API bên ngoài nên không cần bất đồng bộ.

#### railway.json - Cấu hình triển khai

```json
{
    "$schema": "https://railway.app/railway.schema.json",
    "build": {
        "builder": "NIXPACKS"
    },
    "deploy": {
        "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
    }
}
```

Phân tích từng mục cấu hình:

- **`$schema`**: URL JSON schema. Hỗ trợ tự động hoàn thành và kiểm tra tính hợp lệ trong IDE.
- **`build.builder: "NIXPACKS"`**: NIXPACKS là công cụ phân tích mã nguồn và tự động cấu hình môi trường build. Khi phát hiện `pyproject.toml`, tự động thiết lập môi trường Python và cài đặt dependency. Không cần viết Dockerfile trực tiếp.
- **`deploy.startCommand`**: Lệnh thực thi ứng dụng sau khi triển khai.
  - `uvicorn main:app`: Chạy đối tượng `app` trong file `main.py` dưới dạng ASGI server
  - `--host 0.0.0.0`: Cho phép truy cập từ tất cả giao diện mạng (bắt buộc trong môi trường container)
  - `--port $PORT`: Sử dụng port Railway cấp phát động (biến môi trường `$PORT`)

#### .gitignore - Bảo mật và dọn dẹp

```
.env
.venv
__pycache__
```

Các file không nên đưa vào Git repository khi triển khai:
- **`.env`**: File chứa biến môi trường nhạy cảm như API key. **Tuyệt đối không** commit vào Git.
- **`.venv`**: Thư mục môi trường ảo Python. Được tạo riêng trong môi trường triển khai.
- **`__pycache__`**: Cache bytecode Python. File không cần thiết.

#### Thay đổi URL sau triển khai

```http
POST https://my-agent-deployment-production.up.railway.app/conversations
```

URL `http://127.0.0.1:8000` khi phát triển local thay đổi thành `https://my-agent-deployment-production.up.railway.app` sau khi triển khai lên Railway. Railway tự động cung cấp HTTPS và gán subdomain dựa trên tên dự án.

### Tóm tắt quy trình triển khai

```
1. Tạo tài khoản Railway và tạo dự án
2. Kết nối repository GitHub
3. Cài đặt biến môi trường (OPENAI_API_KEY)
4. Tự động build và triển khai theo cấu hình railway.json
5. Test API bằng URL được gán
```

### Cấu trúc dự án cuối cùng

```
deployment/
├── .gitignore         # Danh sách file Git loại trừ
├── .python-version    # Python 3.13
├── .env               # Biến môi trường (loại trừ Git)
├── README.md          # Mô tả dự án
├── api.http           # File test API
├── main.py            # Ứng dụng chính (FastAPI + Agent)
├── pyproject.toml     # Quản lý dependency
├── railway.json       # Cấu hình triển khai Railway
└── uv.lock            # File khóa dependency
```

### Code đầy đủ main.py cuối cùng

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner


agent = Agent(
    name="Assistant",
    instructions="You help users with their questions.",
)

app = FastAPI()
client = AsyncOpenAI()


class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }


class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str


@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }


@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### Điểm thực hành

1. Đăng ký Railway (https://railway.app) và tạo dự án mới.
2. Kết nối repository GitHub, thiết lập biến môi trường `OPENAI_API_KEY` trên dashboard Railway.
3. Gọi `POST /conversations` bằng URL đã triển khai để tạo hội thoại và gửi tin nhắn.
4. Kiểm tra log dịch vụ đã triển khai trên dashboard Railway và quan sát quá trình xử lý request.

---

## Tổng kết chương

### 1. Mẫu kiến trúc
Khi triển khai AI agent lên production, **bọc thành REST API** là cách tiếp cận tiêu chuẩn. FastAPI cung cấp hỗ trợ bất đồng bộ, tự động tạo tài liệu, kiểm tra kiểu, v.v., tối ưu cho tác vụ này.

### 2. Quản lý trạng thái hội thoại
Sử dụng **Conversations API** của OpenAI thì không cần quản lý lịch sử hội thoại phía server. Chỉ cần truyền `conversation_id` thì OpenAI tự động duy trì ngữ cảnh trước đó. Đây là lợi thế lớn trong môi trường serverless hoặc mở rộng ngang (horizontal scaling).

### 3. Đồng bộ vs Streaming
- **`Runner.run()`**: Phải chờ toàn bộ phản hồi nhưng triển khai đơn giản. Phù hợp giao tiếp giữa backend.
- **`Runner.run_streamed()`**: Hỗ trợ streaming token thời gian thực. Cần thiết cho UI hướng người dùng.

### 4. Lọc event
Khi streaming, quan trọng là lọc chỉ những event phù hợp mục đích trong số nhiều event mà `stream_events()` tạo ra:
- Chỉ cần văn bản: `raw_response_event` + `response.output_text.delta`
- Cần toàn bộ event: Toàn bộ `raw_response_event`

### 5. Triển khai cloud
Sử dụng tổ hợp Railway + NIXPACKS cho phép build và triển khai tự động chỉ với `pyproject.toml` mà không cần Dockerfile. Cấu hình cốt lõi là `startCommand` trong `railway.json` và biến môi trường (`OPENAI_API_KEY`).

### 6. Bảo mật
File `.env` bắt buộc phải được đưa vào `.gitignore` để không commit vào Git repository. Trong môi trường triển khai, sử dụng tính năng quản lý biến môi trường của nền tảng.

---

## Bài tập thực hành

### Bài 1: Triển khai cơ bản (Độ khó: 2/5)
Nhập code theo chương để xây dựng API AI agent hoạt động trên local.

**Yêu cầu:**
- `POST /conversations` - Tạo hội thoại
- `POST /conversations/{id}/message` - Gửi tin nhắn đồng bộ
- `POST /conversations/{id}/message-stream` - Gửi tin nhắn streaming
- Xác minh ngữ cảnh hội thoại được duy trì đúng

### Bài 2: Tùy chỉnh agent (Độ khó: 3/5)
Thay đổi agent Assistant cơ bản thành agent chuyên biệt cho domain cụ thể.

**Ví dụ:**
```python
agent = Agent(
    name="Korean Teacher",
    instructions="""Bạn là giáo viên tiếng Hàn.
    Hãy giúp người dùng học tiếng Hàn.
    Nếu có lỗi ngữ pháp, hãy sửa và đề xuất cách diễn đạt tự nhiên.""",
)
```

### Bài 3: Thêm Tool (Độ khó: 4/5)
Mở rộng agent bằng cách thêm tool (function tool) để tận dụng dữ liệu bên ngoài.

**Gợi ý:**
```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Trả về thời tiết của thành phố được chỉ định."""
    # Kết nối API thời tiết thực tế hoặc trả về dữ liệu giả
    return f"Thời tiết hiện tại ở {city}: Nắng, 22 độ"

agent = Agent(
    name="Weather Assistant",
    instructions="You help users check the weather.",
    tools=[get_weather],
)
```

Quan sát cách event gọi tool được truyền trong endpoint streaming (`/message-stream-all`).

### Bài 4: Triển khai Railway (Độ khó: 4/5)
Thực sự triển khai lên Railway và gọi API bằng URL đã triển khai.

**Checklist:**
- [ ] Tạo dự án Railway
- [ ] Kết nối repository GitHub
- [ ] Cài đặt biến môi trường `OPENAI_API_KEY`
- [ ] Health check endpoint `/` sau khi triển khai hoàn tất
- [ ] Test tạo hội thoại và gửi tin nhắn
- [ ] Test phản hồi streaming (sử dụng `curl -N`)

### Bài 5: Tích hợp Frontend (Độ khó: 5/5)
Triển khai frontend chat đơn giản tích hợp với API đã triển khai.

**Gợi ý:**
- Đọc streaming bằng `fetch()` API:
```javascript
const response = await fetch(url, { method: 'POST', body: JSON.stringify({ question }), headers: { 'Content-Type': 'application/json' } });
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Thêm văn bản vào UI
}
```

---

> **Giới thiệu chương tiếp theo**: Trong chương tiếp theo, chúng ta sẽ học cách test phản hồi AI agent và xác minh chất lượng.
