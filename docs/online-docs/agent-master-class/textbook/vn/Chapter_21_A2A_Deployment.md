# Chapter 21: Triển khai AI Agent (Deployment)

## Tổng quan chương

Trong chương này, chúng ta sẽ học toàn bộ quá trình **triển khai AI agent được xây dựng bằng OpenAI Agents SDK vào môi trường production thực tế**. Vượt ra ngoài việc đơn giản chạy agent trên máy cục bộ, chúng ta sẽ đóng gói nó thành REST API bằng framework web FastAPI, quản lý trạng thái hội thoại với Conversations API của OpenAI, xử lý phản hồi đồng bộ/streaming, và cuối cùng triển khai lên nền tảng đám mây Railway.

### Mục tiêu học tập

- Hiểu cách đóng gói AI agent thành REST API bằng FastAPI
- Nắm vững phương pháp quản lý trạng thái hội thoại (context) sử dụng OpenAI Conversations API
- Học sự khác biệt giữa phản hồi đồng bộ (Sync) và streaming, cùng cách triển khai chúng
- Thực hành triển khai đám mây bằng nền tảng Railway

### Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.13 | Ngôn ngữ lập trình |
| FastAPI | 0.118.3 | Framework web |
| OpenAI Agents SDK | 0.3.3 | Framework AI agent |
| Uvicorn | 0.37.0 | ASGI server |
| python-dotenv | 1.1.1 | Quản lý biến môi trường |
| Railway | - | Nền tảng triển khai đám mây |

---

## 21.0 Introduction - Thiết lập ban đầu dự án

### Chủ đề và mục tiêu

Tạo bộ khung cơ bản cho dự án triển khai. Khởi tạo dự án Python mới bằng `uv` (trình quản lý package Python) và thiết lập các dependency cần thiết.

### Giải thích khái niệm cốt lõi

#### Cấu trúc dự án

Trong chương này, chúng ta tạo một thư mục độc lập có tên `deployment/`, tách biệt khỏi dự án masterclass hiện có. Điều này nhằm thiết kế nó như một ứng dụng độc lập có thể triển khai được.

```
deployment/
├── .python-version    # Chỉ định phiên bản Python (3.13)
├── README.md          # Mô tả dự án
├── main.py            # File ứng dụng chính
└── pyproject.toml     # Metadata và dependency của dự án
```

#### pyproject.toml - Quản lý dependency

`pyproject.toml` là file cấu hình tiêu chuẩn cho dự án Python hiện đại. File này khai báo metadata và dependency của dự án.

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

- **`fastapi`**: Framework web Python hiệu suất cao. Cung cấp tạo tài liệu API tự động, kiểm tra kiểu dữ liệu, v.v.
- **`openai-agents`**: SDK agent chính thức của OpenAI. Cung cấp các class cốt lõi cần thiết cho việc thực thi agent, như Agent và Runner.
- **`python-dotenv`**: Tải biến môi trường từ file `.env`. Được sử dụng để tách thông tin nhạy cảm như API key khỏi code.
- **`uvicorn`**: ASGI server. Server cho phép ứng dụng FastAPI thực sự nhận các yêu cầu HTTP.

#### main.py ban đầu

```python
def main():
    print("Hello from deployment!")


if __name__ == "__main__":
    main()
```

Tại thời điểm này, `main.py` vẫn chỉ là code khung (skeleton). Từ phần tiếp theo, chúng ta sẽ bắt đầu chuyển đổi nó thành ứng dụng FastAPI đầy đủ.

### Điểm thực hành

1. Tạo dự án mới bằng lệnh `uv init deployment`.
2. Thêm dependency bằng `uv add fastapi openai-agents python-dotenv uvicorn`.
3. Kiểm tra xem `3.13` đã được thiết lập trong file `.python-version` chưa.

---

## 21.1 Conversations API - Xây dựng API quản lý hội thoại

### Chủ đề và mục tiêu

Xây dựng REST API bằng FastAPI để quản lý các cuộc hội thoại (conversation) với AI agent. Tạo các endpoint để thiết lập phiên hội thoại và thêm tin nhắn vào mỗi cuộc hội thoại bằng **Conversations API** của OpenAI.

### Giải thích khái niệm cốt lõi

#### OpenAI Conversations API là gì?

Conversations API là tính năng quản lý trạng thái hội thoại do OpenAI cung cấp. Trước đây, bạn phải tự quản lý lịch sử hội thoại (history), nhưng với Conversations API, OpenAI duy trì trạng thái hội thoại phía server.

Luồng cốt lõi:
1. Tạo phiên hội thoại mới bằng `client.conversations.create()` trả về `conversation_id` duy nhất.
2. Sau đó khi truyền `conversation_id` này trong quá trình thực thi agent, OpenAI tự động duy trì ngữ cảnh hội thoại trước đó.

Ưu điểm của phương pháp này:
- **Tương thích serverless**: Không cần lưu trạng thái trên server, nên hội thoại được duy trì ngay cả khi server khởi động lại.
- **Triển khai đơn giản**: Loại bỏ logic phức tạp để quản lý trực tiếp mảng lịch sử hội thoại.
- **Khả năng mở rộng**: Có thể tiếp tục cùng một cuộc hội thoại trên nhiều instance server.

#### Cấu trúc ứng dụng FastAPI

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner
```

**Lưu ý quan trọng**: `load_dotenv()` được gọi **trước** `from agents import ...`. Điều này là vì module `agents` tham chiếu biến môi trường (đặc biệt là `OPENAI_API_KEY`) khi được import. Nếu đảo ngược thứ tự, API key không thể được tìm thấy và sẽ xảy ra lỗi.

#### Định nghĩa Agent

```python
agent = Agent(
    name="Assistant",
    instructions="You help users with their questions."
)
```

Agent được tạo chỉ một lần ở cấp module. Không cần tạo mới cho mỗi yêu cầu. `instructions` đóng vai trò là system prompt của agent.

#### Khởi tạo FastAPI App và OpenAI Client

```python
app = FastAPI()
client = AsyncOpenAI()
```

`AsyncOpenAI()` là client OpenAI bất đồng bộ (async). Vì FastAPI là framework bất đồng bộ, sử dụng client bất đồng bộ là phù hợp về mặt hiệu suất.

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

Những điểm chính của code này:

1. **Pydantic BaseModel**: `CreateConversationResponse` định nghĩa schema phản hồi. FastAPI tự động serialize thành JSON và phản ánh trong tài liệu Swagger.
2. **`client.conversations.create()`**: Gọi OpenAI API để tạo phiên hội thoại mới. Trường `.id` của giá trị trả về chứa ID duy nhất bắt đầu bằng `conv_`.
3. **Xử lý bất đồng bộ**: Sử dụng từ khóa `await` để chờ bất đồng bộ cho đến khi lệnh gọi API hoàn thành.

#### Endpoint tin nhắn (Khung)

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(conversation_id: str):
    pass
```

Tại thời điểm này, endpoint tin nhắn chưa được triển khai. Đường dẫn URL bao gồm `{conversation_id}`, tạo cấu trúc cho phép gửi tin nhắn đến một cuộc hội thoại cụ thể.

### Phân tích code - Luồng tổng thể

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │                            │── conversations.create()─>│
   │                            │<── đối tượng conversation ─│
   │<── { conversation_id } ────│                          │
```

### Điểm thực hành

1. Khởi động server phát triển bằng lệnh `uvicorn main:app --reload`.
2. Truy cập `http://127.0.0.1:8000/docs` trong trình duyệt để xem tài liệu Swagger tự động tạo của FastAPI.
3. Gọi endpoint `POST /conversations` và xác nhận `conversation_id` được trả về chính xác.

---

## 21.2 Sync Responses - Triển khai phản hồi đồng bộ

### Chủ đề và mục tiêu

Hoàn thành endpoint gửi tin nhắn vào hội thoại và nhận phản hồi từ agent một cách **đồng bộ**. Sử dụng `Runner.run()` để thực thi agent và chuyển toàn bộ phản hồi cho client sau khi phản hồi hoàn chỉnh được tạo.

### Giải thích khái niệm cốt lõi

#### Phản hồi đồng bộ vs Phản hồi streaming

| Đặc điểm | Phản hồi đồng bộ (Sync) | Phản hồi streaming |
|-----------|-------------------------|-------------------|
| Phương thức phản hồi | Gửi tất cả cùng lúc sau khi hoàn thành | Gửi real-time theo từng token |
| Trải nghiệm người dùng | Thời gian chờ đợi phản hồi | Văn bản xuất hiện ngay lập tức |
| Độ phức tạp triển khai | Tương đối đơn giản | Cần xử lý event stream |
| Phù hợp cho | Giao tiếp backend-to-backend, phản hồi ngắn | UI hướng người dùng, phản hồi dài |

Trong phần này, chúng ta triển khai phản hồi đồng bộ trước.

#### Định nghĩa model yêu cầu/phản hồi

```python
class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str
```

Định nghĩa schema đầu vào/đầu ra một cách nghiêm ngặt bằng Pydantic `BaseModel`.

- **`CreateMessageInput`**: Body yêu cầu mà client gửi đến. Câu hỏi của người dùng nằm trong trường `question`.
- **`CreateMessageOutput`**: Phản hồi mà server trả về. Câu trả lời của agent nằm trong trường `answer`.

Dựa trên các model này, FastAPI:
- Tự động phân tích JSON của body yêu cầu và kiểm tra kiểu dữ liệu.
- Tự động trả về 422 Validation Error cho các yêu cầu không đúng định dạng.

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

Những điểm chính của code này:

1. **Tham số đường dẫn `conversation_id`**: Được trích xuất từ URL và truyền làm đối số hàm. Điều này xác định cuộc hội thoại nào sẽ nhận tin nhắn.
2. **Tham số body `message_input`**: FastAPI tự động chuyển đổi JSON body yêu cầu thành đối tượng `CreateMessageInput`.
3. **`Runner.run()`**: Phương thức cốt lõi để thực thi agent.
   - `starting_agent`: Đối tượng agent cần thực thi
   - `input`: Văn bản câu hỏi của người dùng
   - `conversation_id`: ID hội thoại từ OpenAI Conversations API. Đây là **chìa khóa để duy trì ngữ cảnh hội thoại**.
4. **`answer.final_output`**: Trích xuất đầu ra văn bản cuối cùng của agent từ giá trị trả về của `Runner.run()`.

#### Kiểm thử API (api.http)

```http
POST http://127.0.0.1:8000/conversations

###

POST http://127.0.0.1:8000/conversations/conv_68ecdf11ff6081969cc4e8e9d126c015082054e6371dc260/message
Content-Type: application/json

{
    "question": "What is the first question i asked you?"
}
```

File `api.http` là file kiểm thử yêu cầu HTTP được sử dụng với extension REST Client của VS Code, v.v. Các yêu cầu được phân tách bằng `###`, và mỗi yêu cầu có thể được thực thi riêng lẻ.

Trong bài kiểm thử trên, câu hỏi "What is the first question i asked you?" được thiết kế để **xác minh tính bền vững của ngữ cảnh hội thoại**. Vì nội dung hội thoại trước đó được duy trì thông qua `conversation_id`, agent có thể nhớ và phản hồi về các câu hỏi đã nhận trước đó.

### Phân tích code - Luồng tổng thể

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │<── { conversation_id } ────│                          │
   │                            │                          │
   │── POST /conversations/     │                          │
   │   {id}/message ───────────>│                          │
   │   { question: "..." }      │── Runner.run() ─────────>│
   │                            │   (bao gồm conversation_id) │
   │                            │<── phản hồi hoàn chỉnh ───│
   │<── { answer: "..." } ──────│                          │
```

### Điểm thực hành

1. Sau khi tạo hội thoại, sử dụng `conversation_id` được trả về để gửi nhiều tin nhắn.
2. Nói "My name is [tên]", sau đó hỏi "What is my name?" để xác minh ngữ cảnh hội thoại được duy trì.
3. Xác minh rằng các hội thoại được duy trì độc lập khi sử dụng các giá trị `conversation_id` khác nhau.

---

## 21.3 StreamingResponse - Triển khai phản hồi streaming

### Chủ đề và mục tiêu

Triển khai endpoint chuyển phản hồi của agent qua **streaming real-time**. Điều này cho phép người dùng xem câu trả lời của agent được tạo theo thời gian thực. Chúng ta triển khai hai phương thức streaming (chỉ văn bản / tất cả event).

### Giải thích khái niệm cốt lõi

#### Sự cần thiết của phản hồi streaming

Phản hồi đồng bộ (`Runner.run()`) yêu cầu client chờ đến khi toàn bộ câu trả lời được tạo. Với câu trả lời dài, điều này có thể mất vài giây đến hàng chục giây, dẫn đến trải nghiệm người dùng kém.

Phản hồi streaming (`Runner.run_streamed()`) gửi token đến client ngay khi chúng được tạo. Điều này hoạt động theo cùng nguyên lý với văn bản xuất hiện từng ký tự một trong giao diện web ChatGPT.

#### FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse
```

`StreamingResponse` của FastAPI nhận hàm generator và gửi dữ liệu đến client theo từng chunk. Nó gửi dữ liệu dần dần trong khi duy trì kết nối HTTP.

#### Phương pháp 1: Streaming chỉ text delta

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

Hãy phân tích code này từng bước:

**Bước 1 - Gọi `Runner.run_streamed()`**:
```python
events = Runner.run_streamed(
    starting_agent=agent,
    input=message_input.question,
    conversation_id=conversation_id,
)
```
Thay vì `Runner.run()`, chúng ta sử dụng `Runner.run_streamed()`. Phương thức này không trả về kết quả cùng lúc mà trả về đối tượng event stream.

**Bước 2 - Lọc event**:
```python
async for event in events.stream_events():
    if (
        event.type == "raw_response_event"
        and event.data.type == "response.output_text.delta"
    ):
        yield event.data.delta
```

`stream_events()` tạo ra nhiều loại event khác nhau. Ở đây chúng ta lọc bằng hai điều kiện:
- `event.type == "raw_response_event"`: Event thô (raw) được chuyển trực tiếp từ OpenAI API
- `event.data.type == "response.output_text.delta"`: Event tương ứng với **phần thay đổi (delta)** của đầu ra văn bản

`yield` là cú pháp async generator của Python. Nó tạo dữ liệu từng phần một và truyền cho `StreamingResponse`.

**Bước 3 - Trả về StreamingResponse**:
```python
return StreamingResponse(event_generator(), media_type="text/plain")
```
Thiết lập `media_type="text/plain"` để stream dưới dạng văn bản thuần. Client duy trì kết nối và nhận các chunk văn bản tuần tự.

#### Phương pháp 2: Streaming tất cả event

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

Endpoint này stream không chỉ text delta mà **tất cả raw_response_event** ở định dạng JSON.

Sự khác biệt chính:
- Điều kiện lọc chỉ còn `event.type == "raw_response_event"` (đã loại bỏ điều kiện text delta).
- `event.data.to_json()` chuyển đổi toàn bộ event thành chuỗi JSON.
- Thêm `\n` (xuống dòng) sau mỗi event để client có thể phân biệt giữa các event.

Phương pháp này hữu ích khi cần kiểm soát chi tiết hơn ở frontend. Ví dụ, bạn có thể nhận event gọi công cụ, event chuyển đổi agent, v.v., và phản ánh chúng trong UI.

#### So sánh hai phương pháp

| Đặc điểm | `/message-stream` | `/message-stream-all` |
|-----------|-------------------|----------------------|
| Nội dung gửi | Chỉ đoạn văn bản | Tất cả event (JSON) |
| Định dạng dữ liệu | Văn bản thuần | JSON (phân tách bằng xuống dòng) |
| Lượng dữ liệu | Ít | Nhiều |
| Phù hợp cho | UI chat đơn giản | UI nâng cao (hiển thị thực thi công cụ, v.v.) |

#### Kiểm thử streaming bằng curl

```bash
curl -N -X POST http://127.0.0.1:8000/conversations/{conv_id}/message-stream \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the size of the great wall of china?"}'
```

Flag `-N` trong `curl` tắt bộ đệm đầu ra. Không có tùy chọn này, curl sẽ tích lũy dữ liệu trong bộ đệm và xuất tất cả cùng lúc, khiến không thể quan sát hiệu ứng streaming.

### Điểm thực hành

1. Gọi endpoint `/message-stream` bằng `curl -N` và xác nhận văn bản được xuất ra theo thời gian thực.
2. Gọi endpoint `/message-stream-all` và quan sát những loại event nào được chuyển đến.
3. So sánh sự khác biệt tốc độ cảm nhận giữa phản hồi đồng bộ (`/message`) và phản hồi streaming (`/message-stream`).
4. Phân tích những loại event nào tồn tại trong các event JSON được stream ngoài `response.output_text.delta`.

---

## 21.4 Deployment - Triển khai đám mây Railway

### Chủ đề và mục tiêu

Triển khai API AI agent hoàn chỉnh lên nền tảng đám mây **Railway** để biến nó thành dịch vụ thực tế có thể truy cập từ internet. Chúng ta sẽ đề cập đến việc viết file cấu hình triển khai, quản lý biến môi trường và cài đặt bảo mật.

### Giải thích khái niệm cốt lõi

#### Railway là gì?

Railway là nền tảng triển khai đám mây thân thiện với nhà phát triển. Kết nối Git repository và nó tự động build và deploy, giúp dễ dàng quản lý biến môi trường, kiểm tra log, cấu hình tên miền, v.v.

Ưu điểm của Railway:
- Triển khai tự động dựa trên Git push (CI/CD)
- Build tự động sử dụng NIXPACKS (không cần Dockerfile)
- Có tier miễn phí
- Quản lý biến môi trường đơn giản

#### Thêm endpoint Health Check

```python
@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }
```

Một endpoint GET đơn giản được thêm vào đường dẫn root (`/`). Điều này phục vụ nhiều mục đích:
- **Health Check**: Dùng để xác nhận server hoạt động bình thường. Các nền tảng đám mây như Railway định kỳ gọi endpoint này để kiểm tra trạng thái dịch vụ.
- **Xác nhận nhanh**: Cho phép bạn kiểm tra ngay lập tức dịch vụ có hoạt động không khi truy cập URL đã triển khai trong trình duyệt.
- **Hàm đồng bộ**: Được định nghĩa bằng `def` thay vì `async def`. Vì không có lệnh gọi API bên ngoài nên async không cần thiết.

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
- **`build.builder: "NIXPACKS"`**: NIXPACKS là công cụ phân tích mã nguồn và tự động cấu hình môi trường build. Khi phát hiện `pyproject.toml`, nó tự động thiết lập môi trường Python và cài đặt dependency. Không cần viết Dockerfile thủ công.
- **`deploy.startCommand`**: Lệnh chạy ứng dụng sau khi triển khai.
  - `uvicorn main:app`: Chạy đối tượng `app` từ `main.py` như ASGI server
  - `--host 0.0.0.0`: Cho phép kết nối từ tất cả giao diện mạng (bắt buộc trong môi trường container)
  - `--port $PORT`: Sử dụng port được Railway cấp phát động (biến môi trường `$PORT`)

#### .gitignore - Bảo mật và dọn dẹp

```
.env
.venv
__pycache__
```

Các file không nên được bao gồm trong Git repository khi triển khai:
- **`.env`**: File chứa biến môi trường nhạy cảm như API key. **Tuyệt đối không** được commit vào Git.
- **`.venv`**: Thư mục virtual environment Python. Được tạo riêng trong môi trường triển khai.
- **`__pycache__`**: Cache bytecode Python. Là file không cần thiết.

#### Thay đổi URL sau triển khai

```http
POST https://my-agent-deployment-production.up.railway.app/conversations
```

URL thay đổi từ `http://127.0.0.1:8000` trong phát triển cục bộ thành `https://my-agent-deployment-production.up.railway.app` sau khi triển khai lên Railway. Railway tự động cung cấp HTTPS và gán subdomain dựa trên tên dự án.

### Tóm tắt quy trình triển khai

```
1. Tạo tài khoản và dự án Railway
2. Kết nối GitHub repository
3. Thiết lập biến môi trường (OPENAI_API_KEY)
4. Build và triển khai tự động theo cấu hình railway.json
5. Kiểm thử API với URL được gán
```

### Cấu trúc dự án cuối cùng

```
deployment/
├── .gitignore         # Danh sách file loại trừ khỏi Git
├── .python-version    # Python 3.13
├── .env               # Biến môi trường (loại trừ khỏi Git)
├── README.md          # Mô tả dự án
├── api.http           # File kiểm thử API
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

1. Đăng ký tại Railway (https://railway.app) và tạo dự án mới.
2. Kết nối GitHub repository và thiết lập biến môi trường `OPENAI_API_KEY` trong bảng điều khiển Railway.
3. Gọi `POST /conversations` với URL đã triển khai để tạo hội thoại và gửi tin nhắn.
4. Kiểm tra log của dịch vụ đã triển khai trong bảng điều khiển Railway và quan sát luồng xử lý yêu cầu.

---

## Tóm tắt nội dung chính của chương

### 1. Pattern kiến trúc
Khi triển khai AI agent vào production, **đóng gói thành REST API** là cách tiếp cận tiêu chuẩn. FastAPI được tối ưu hóa cho công việc này, cung cấp hỗ trợ async, tạo tài liệu tự động, kiểm tra kiểu dữ liệu, v.v.

### 2. Quản lý trạng thái hội thoại
Sử dụng **Conversations API** của OpenAI loại bỏ nhu cầu quản lý lịch sử hội thoại phía server. Chỉ cần truyền `conversation_id` để OpenAI tự động duy trì ngữ cảnh trước đó. Đây là lợi thế lớn trong môi trường serverless hoặc khi mở rộng ngang (horizontal scaling).

### 3. Đồng bộ vs Streaming
- **`Runner.run()`**: Yêu cầu chờ phản hồi hoàn chỉnh nhưng triển khai đơn giản. Phù hợp cho giao tiếp backend-to-backend.
- **`Runner.run_streamed()`**: Hỗ trợ streaming token real-time. Thiết yếu cho UI hướng người dùng.

### 4. Lọc event
Khi streaming, việc lọc chỉ những event phù hợp với mục đích từ các event đa dạng được tạo bởi `stream_events()` là rất quan trọng:
- Chỉ cần văn bản: `raw_response_event` + `response.output_text.delta`
- Cần tất cả event: Toàn bộ `raw_response_event`

### 5. Triển khai đám mây
Sử dụng kết hợp Railway + NIXPACKS cho phép build và triển khai tự động chỉ với `pyproject.toml`, không cần Dockerfile. Cấu hình quan trọng là `startCommand` trong `railway.json` và biến môi trường (`OPENAI_API_KEY`).

### 6. Bảo mật
File `.env` phải được bao gồm trong `.gitignore` để ngăn không cho commit vào Git repository. Trong môi trường triển khai, sử dụng tính năng quản lý biến môi trường của nền tảng.

---

## Bài tập thực hành

### Bài tập 1: Triển khai cơ bản (Độ khó: 2/5)
Theo code của chương để xây dựng API AI agent hoạt động trên máy cục bộ.

**Yêu cầu:**
- `POST /conversations` - Tạo hội thoại
- `POST /conversations/{id}/message` - Gửi tin nhắn đồng bộ
- `POST /conversations/{id}/message-stream` - Gửi tin nhắn streaming
- Xác minh ngữ cảnh hội thoại được duy trì chính xác

### Bài tập 2: Tùy chỉnh Agent (Độ khó: 3/5)
Thay đổi agent Assistant cơ bản thành agent chuyên biệt cho một lĩnh vực cụ thể.

**Ví dụ:**
```python
agent = Agent(
    name="Korean Teacher",
    instructions="""You are a Korean language teacher.
    Help users learn Korean.
    Correct grammatical errors and suggest natural expressions.""",
)
```

### Bài tập 3: Thêm công cụ (Tool) (Độ khó: 4/5)
Thêm function tool vào agent để mở rộng khả năng sử dụng dữ liệu bên ngoài.

**Gợi ý:**
```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Returns the weather for a given city."""
    # Kết nối API thời tiết thực hoặc trả về dữ liệu giả
    return f"Current weather in {city}: Clear, 22 degrees"

agent = Agent(
    name="Weather Assistant",
    instructions="You help users check the weather.",
    tools=[get_weather],
)
```

Quan sát cách event gọi công cụ được chuyển trong endpoint streaming (`/message-stream-all`).

### Bài tập 4: Triển khai lên Railway (Độ khó: 4/5)
Thực sự triển khai lên Railway và gọi API với URL đã triển khai.

**Danh sách kiểm tra:**
- [ ] Tạo dự án Railway
- [ ] Kết nối GitHub repository
- [ ] Thiết lập biến môi trường `OPENAI_API_KEY`
- [ ] Health check qua endpoint `/` sau khi triển khai hoàn tất
- [ ] Kiểm thử tạo hội thoại và gửi tin nhắn
- [ ] Kiểm thử phản hồi streaming (sử dụng `curl -N`)

### Bài tập 5: Tích hợp Frontend (Độ khó: 5/5)
Triển khai frontend chat đơn giản kết nối với API đã triển khai.

**Gợi ý:**
- Đọc streaming với `fetch()` API:
```javascript
const response = await fetch(url, { method: 'POST', body: JSON.stringify({ question }), headers: { 'Content-Type': 'application/json' } });
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Thêm text vào UI
}
```

---

> **Giới thiệu chương tiếp theo**: Trong chương tiếp theo, chúng ta sẽ học cách kiểm thử và xác minh chất lượng phản hồi của AI agent.
