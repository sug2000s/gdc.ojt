# Chapter 12: ADK Server và Triển khai

---

## 12.0 Tổng quan chương

Trong chương này, chúng ta học toàn bộ quy trình xây dựng tác tử AI sử dụng Google ADK (Agent Development Kit), vận hành chúng như server, và cuối cùng triển khai lên Google Cloud Vertex AI.

Xuyên suốt chương, chúng ta đề cập hai dự án tác tử:

1. **Email Refiner Agent** - Hệ thống dựa trên LoopAgent nơi nhiều tác tử chuyên biệt hợp tác để cải thiện email lặp đi lặp lại
2. **Travel Advisor Agent** - Tác tử tư vấn du lịch dựa trên công cụ cung cấp thông tin thời tiết, tỷ giá hối đoái và điểm du lịch

Qua hai dự án này, chúng ta học các chủ đề chính sau:

| Phần | Chủ đề | Khái niệm chính |
|------|--------|-----------------|
| 12.0 | Introduction | Cấu trúc dự án ADK, định nghĩa Agent, thiết kế prompt |
| 12.1 | LoopAgent | Tác tử lặp, output_key, escalate, ToolContext |
| 12.3 | API Server | Server API tích hợp ADK, REST endpoint, quản lý phiên |
| 12.4 | Server Sent Events | Streaming SSE, xử lý phản hồi thời gian thực |
| 12.6 | Runner | Lớp Runner, DatabaseSessionService, thực thi chế độ code |
| 12.7 | Deployment to VertexAI | Triển khai Vertex AI, reasoning_engines, thực thi từ xa |

---

## 12.0 Introduction - Cấu trúc dự án ADK và định nghĩa tác tử

### Chủ đề và mục tiêu

Hiểu cấu trúc cơ bản của dự án tác tử dựa trên ADK và thiết kế từng thành phần của hệ thống đa tác tử có tên Email Refiner.

### Khái niệm chính

#### 1) Cấu trúc thư mục dự án ADK

ADK tuân theo các quy tắc cấu trúc thư mục cụ thể. Tệp `agent.py` và `__init__.py` phải tồn tại bên trong gói tác tử, và `__init__.py` phải import module `agent` để ADK tự động nhận dạng tác tử.

```
email-refiner-agent/
├── .python-version          # Phiên bản Python (3.13)
├── pyproject.toml           # Định nghĩa phụ thuộc dự án
├── uv.lock                  # Tệp khóa phụ thuộc
├── README.md
└── email_refiner/           # Gói tác tử
    ├── __init__.py          # Đăng ký module tác tử
    ├── agent.py             # Định nghĩa tác tử
    └── prompt.py            # Prompt và mô tả
```

**Vai trò của `__init__.py`:**

```python
from . import agent
```

Dòng duy nhất này cực kỳ quan trọng. Framework ADK tự động tìm kiếm module `agent` trong gói, và việc import rõ ràng này trong `__init__.py` là cần thiết để cơ chế khám phá tác tử (agent discovery) của ADK hoạt động.

#### 2) Cấu hình phụ thuộc (pyproject.toml)

```toml
[project]
name = "email-refiner-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "google-adk>=1.12.0",
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Phụ thuộc chính:
- **`google-adk`**: Thư viện cốt lõi Google Agent Development Kit
- **`google-genai`**: Client Google Generative AI
- **`litellm`**: Thư viện cho phép sử dụng các nhà cung cấp LLM khác nhau (OpenAI, Anthropic, Google, v.v.) qua giao diện thống nhất

#### 3) Thiết kế đa tác tử chuyên biệt

Email Refiner gồm 5 tác tử chuyên biệt. Mỗi tác tử phụ trách một khía cạnh khác nhau của việc cải thiện email:

```python
from google.adk.agents import Agent, LoopAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o-mini")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
)

literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
)
```

**Vai trò của từng tác tử:**

| Tác tử | Vai trò | Nhiệm vụ chính |
|--------|---------|----------------|
| ClarityEditorAgent | Biên tập viên rõ ràng | Loại bỏ mơ hồ, xóa cụm từ thừa, làm câu ngắn gọn |
| ToneStylistAgent | Nhà tạo phong cách giọng điệu | Duy trì giọng ấm áp và tự tin, giữ tính chuyên nghiệp |
| PersuationAgent | Chiến lược gia thuyết phục | Tăng cường CTA, cấu trúc lập luận, loại bỏ biểu đạt bị động |
| EmailSynthesizerAgent | Tổng hợp email | Tích hợp tất cả cải thiện vào một email |
| LiteraryCriticAgent | Nhà phê bình văn học | Đánh giá chất lượng cuối cùng và quyết định phê duyệt/làm lại |

#### 4) Mẫu thiết kế prompt

Prompt được tách thành `description` (giải thích vai trò tác tử) và `instruction` (chỉ thị chi tiết) để quản lý. Điều này tuân theo nguyên tắc Tách biệt mối quan tâm (Separation of Concerns).

```python
# Mô tả - định nghĩa ngắn gọn tác tử là gì
CLARITY_EDITOR_DESCRIPTION = "Expert editor focused on clarity and simplicity."

# Chỉ thị - mô tả chi tiết tác tử nên hoạt động như thế nào
CLARITY_EDITOR_INSTRUCTION = """
You are an expert editor focused on clarity and simplicity. Your job is to
eliminate ambiguity, redundancy, and make every sentence crisp and clear.

Take the email draft and improve it for clarity:
- Remove redundant phrases
- Simplify complex sentences
- Eliminate ambiguity
- Make every sentence clear and direct

Provide your improved version with focus on clarity.
"""
```

Điểm đáng chú ý đặc biệt là **mẫu pipeline**. Instruction của mỗi tác tử sử dụng biến mẫu tham chiếu đầu ra của tác tử trước:

```python
TONE_STYLIST_INSTRUCTION = """
...
Here's the clarity-improved version:
{clarity_output}
"""

PERSUASION_STRATEGIST_INSTRUCTION = """
...
Here's the tone-improved version:
{tone_output}
"""

EMAIL_SYNTHESIZER_INSTRUCTION = """
...
Clarity version: {clarity_output}
Tone version: {tone_output}
Persuasion version: {persuasion_output}

Synthesize the best elements from all versions into one polished final email.
"""
```

Các biến `{clarity_output}`, `{tone_output}` này kết nối với `output_key` mà chúng ta sẽ học ở phần tiếp theo.

### Điểm thực hành

1. Tự tạo thư mục dự án ADK và xây dựng cấu trúc import module agent từ `__init__.py`.
2. Đọc prompt của từng tác tử và vẽ sơ đồ luồng pipeline cải thiện email.
3. Sử dụng `LiteLlm` để chuyển từ mô hình OpenAI sang mô hình khác (ví dụ: `anthropic/claude-3-haiku`).

---

## 12.1 LoopAgent - Tác tử lặp và Escalation

### Chủ đề và mục tiêu

Xây dựng hệ thống nơi nhiều tác tử hợp tác lặp đi lặp lại sử dụng `LoopAgent` của ADK. Học về chia sẻ dữ liệu giữa các tác tử qua `output_key` và cơ chế kết thúc vòng lặp qua `escalate`.

### Khái niệm chính

#### 1) output_key - Truyền dữ liệu giữa các tác tử

`output_key` chỉ định tên khóa để lưu đầu ra của tác tử vào trạng thái phiên (state). Các biến mẫu `{clarity_output}`, `{tone_output}`, v.v. trong prompt ở phần trước được điền qua `output_key` này.

```python
MODEL = LiteLlm(model="openai/gpt-4o")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
    output_key="clarity_output",    # Đầu ra lưu vào state["clarity_output"]
    model=MODEL,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
    output_key="tone_output",       # Đầu ra lưu vào state["tone_output"]
    model=MODEL,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
    output_key="persuasion_output", # Đầu ra lưu vào state["persuasion_output"]
    model=MODEL,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
    output_key="synthesized_output", # Đầu ra lưu vào state["synthesized_output"]
    model=MODEL,
)
```

**Luồng dữ liệu:**

```
Đầu vào email người dùng
    │
    ▼
ClarityEditorAgent ──── output_key="clarity_output" ──────► Lưu vào state
    │
    ▼
ToneStylistAgent ────── output_key="tone_output" ──────────► Lưu vào state
    │                    (tham chiếu {clarity_output} trong instruction)
    ▼
PersuationAgent ─────── output_key="persuasion_output" ────► Lưu vào state
    │                    (tham chiếu {tone_output} trong instruction)
    ▼
EmailSynthesizerAgent ─ output_key="synthesized_output" ───► Lưu vào state
    │                    (tham chiếu cả 3 output)
    ▼
LiteraryCriticAgent ─── Đánh giá chất lượng
    │                    (tham chiếu {synthesized_output} trong instruction)
    ├── Không đạt → Khởi động lại vòng lặp
    └── Đạt → Thoát vòng lặp qua escalate
```

#### 2) ToolContext và escalate - Cơ chế kết thúc vòng lặp

`LoopAgent` mặc định lặp vô hạn (hoặc đến `max_iterations`). Để thoát vòng lặp theo điều kiện cụ thể, sử dụng cơ chế `escalate`.

```python
from google.adk.tools.tool_context import ToolContext

async def escalate_email_complete(tool_context: ToolContext):
    """Use this tool only when the email is good to go."""
    tool_context.actions.escalate = True
    return "Email optimization complete."
```

**Điểm chính:**
- `ToolContext` là đối tượng ngữ cảnh mà ADK tự động tiêm khi thực thi công cụ.
- Đặt `tool_context.actions.escalate = True` sẽ kết thúc vòng lặp hiện tại ngay lập tức.
- Công cụ này chỉ được cung cấp cho `LiteraryCriticAgent`, nên vòng lặp chỉ kết thúc khi nhà phê bình hài lòng với chất lượng email.

```python
literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
    tools=[
        escalate_email_complete,   # Cung cấp công cụ escalate
    ],
    model=MODEL,
)
```

#### 3) Cấu hình LoopAgent

Tất cả tác tử con được bọc trong `LoopAgent` để hoàn thành cấu trúc thực thi lặp:

```python
email_refiner_agent = LoopAgent(
    name="EmailRefinerAgent",
    max_iterations=50,                    # Tối đa 50 lần lặp (biện pháp an toàn)
    description=EMAIL_OPTIMIZER_DESCRIPTION,
    sub_agents=[
        clarity_agent,                     # 1. Cải thiện rõ ràng
        tone_stylist_agent,                # 2. Điều chỉnh giọng điệu
        persuation_agent,                  # 3. Tăng cường thuyết phục
        email_synthesizer_agent,           # 4. Tổng hợp
        literary_critic_agent,             # 5. Đánh giá cuối (có thể escalate)
    ],
)

root_agent = email_refiner_agent
```

**Tầm quan trọng của biến `root_agent`:** Framework ADK tự động tìm kiếm biến có tên `root_agent` và sử dụng nó làm tác tử điểm vào. Phải sử dụng tên này.

#### 4) Tăng cường prompt - Đảm bảo LLM thực sự gọi công cụ

Trong thực tế, LLM có thể "nói" sẽ gọi công cụ nhưng không thực sự làm. Để ngăn điều này, prompt đã được tăng cường:

```python
LITERARY_CRITIC_INSTRUCTION = """
...
2. If the email meets professional standards and communicates effectively:
   - Call the `escalate_email_complete` tool, CALL IT DONT JUST SAY YOU ARE
     GOING TO CALL IT. CALL THE THING!
   - Provide your final positive assessment of the email
...
## Tool Usage:
When the email is ready, CALL the tool: `escalate_email_complete()`
...
"""
```

Sử dụng chữ hoa và biểu đạt nhấn mạnh để hướng dẫn rõ ràng LLM thực thi lệnh gọi công cụ là kỹ thuật prompt engineering rất hữu ích trong thực tế.

### Điểm thực hành

1. Giảm `max_iterations` xuống 3 và chạy để quan sát hành vi khi vòng lặp đạt số lần lặp tối đa.
2. Kiểm tra điều gì xảy ra nếu chỉ trả về giá trị mà không đặt `escalate = True` trong hàm `escalate_email_complete`.
3. Xóa `output_key` và chạy để xác nhận tác tử tiếp theo không thể tham chiếu kết quả trước.

---

## 12.3 API Server - Server API tích hợp ADK

### Chủ đề và mục tiêu

Học cách phục vụ tác tử dưới dạng REST API sử dụng server web tích hợp của ADK. Tạo Travel Advisor Agent mới và tương tác với nó qua server API.

### Khái niệm chính

#### 1) Travel Advisor Agent - Tác tử dựa trên công cụ

Xây dựng tác tử tư vấn du lịch mới sử dụng công cụ để demo server API:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

MODEL = LiteLlm(model="openai/gpt-4o")


async def get_weather(tool_context: ToolContext, location: str):
    """Get current weather information for a location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind": "12 km/h",
        "forecast": "Mild weather with occasional clouds expected throughout the day",
    }


async def get_exchange_rate(
    tool_context: ToolContext, from_currency: str, to_currency: str, amount: float
):
    """Get exchange rate between two currencies.
    Args should always be from_currency str, to_currency str, amount flot
    """
    mock_rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("USD", "KRW"): 1325.00,
        ("EUR", "USD"): 1.09,
        ("EUR", "GBP"): 0.86,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
        ("KRW", "USD"): 0.00075,
    }

    rate = mock_rates.get((from_currency, to_currency), 1.0)
    converted_amount = amount * rate

    return {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "amount": amount,
        "exchange_rate": rate,
        "converted_amount": converted_amount,
        "timestamp": "2024-03-15 10:30:00 UTC",
    }


async def get_local_attractions(
    tool_context: ToolContext, location: str, category: str = "all"
):
    """Get popular attractions and points of interest for a location."""
    attractions = {
        "Paris": [
            {"name": "Eiffel Tower", "type": "landmark", "rating": 4.8,
             "description": "Iconic iron lattice tower"},
            {"name": "Louvre Museum", "type": "museum", "rating": 4.7,
             "description": "World's largest art museum"},
            # ... thêm dữ liệu điểm du lịch
        ],
        "Tokyo": [
            {"name": "Tokyo Tower", "type": "landmark", "rating": 4.5,
             "description": "Communications and observation tower"},
            {"name": "Senso-ji", "type": "temple", "rating": 4.6,
             "description": "Ancient Buddhist temple"},
            # ... thêm dữ liệu điểm du lịch
        ],
        "default": [
            {"name": "City Center", "type": "area", "rating": 4.2,
             "description": "Main downtown area"},
            # ... dữ liệu điểm du lịch mặc định
        ],
    }

    location_attractions = attractions.get(location, attractions["default"])

    if category != "all":
        location_attractions = [
            a for a in location_attractions if a["type"] == category
        ]

    return {
        "location": location,
        "category": category,
        "attractions": location_attractions,
        "total_count": len(location_attractions),
    }
```

**Mẫu thiết kế hàm công cụ:**
- Tất cả hàm công cụ được định nghĩa là hàm `async` bất đồng bộ.
- Tham số đầu tiên phải là `tool_context: ToolContext` (ADK tự động tiêm).
- Docstring đóng vai trò giải thích mục đích công cụ cho LLM.
- Giá trị trả về ở dạng dictionary, LLM diễn giải để phản hồi người dùng.

Đăng ký tác tử:

```python
travel_advisor_agent = Agent(
    name="TravelAdvisorAgent",
    description=TRAVEL_ADVISOR_DESCRIPTION,
    instruction=TRAVEL_ADVISOR_INSTRUCTION,
    tools=[
        get_weather,
        get_exchange_rate,
        get_local_attractions,
    ],
    model=MODEL,
)

root_agent = travel_advisor_agent
```

#### 2) Chạy server API tích hợp ADK

ADK có thể khởi chạy server web tích hợp ngay lập tức với lệnh `adk api_server`. Server này dựa trên FastAPI và tự động cung cấp REST endpoint để tương tác với tác tử.

```bash
# Chạy từ thư mục cha chứa dự án tác tử
adk api_server email-refiner-agent/
```

Khi server khởi động, có thể truy cập tại `http://127.0.0.1:8000`.

#### 3) Tương tác tác tử qua REST API

**Tạo phiên:**

```python
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"

# Tạo phiên mới
response = requests.post(
    f"{BASE_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions"
)
print(response.json())
# Nhận phản hồi chứa ID phiên
```

Mẫu endpoint tạo phiên của server API ADK:
```
POST /apps/{tên_app}/users/{ID_người_dùng}/sessions
```

**Gửi tin nhắn (Chế độ đồng bộ):**

```python
SESSION_ID = "ce085ce3-9637-4eca-b7a1-b0be58fa39f1"  # ID nhận được khi tạo phiên

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "Yes, I want to know the currency exchange rate"}],
        "role": "user",
    },
}
response = requests.post(f"{BASE_URL}/run", json=message)
print(response.json())
```

**Phân tích phản hồi:**

```python
data = response.json()

for event in data:
    content = event.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

Phản hồi ở dạng mảng sự kiện, trong `content.parts` của mỗi sự kiện:
- `functionCall`: Thông tin công cụ mà tác tử đã gọi
- `text`: Phản hồi văn bản của tác tử

#### 4) Cập nhật phụ thuộc

Phụ thuộc được thêm cho tính năng server API và đánh giá (eval):

```toml
dependencies = [
    "google-adk[eval]>=1.12.0",   # Thêm [eval] extra
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",            # Client HTTP (cho gọi API)
    "sseclient-py>=1.8.0",        # Client SSE (phần tiếp theo)
]
```

### Điểm thực hành

1. Chạy server với `adk api_server` và truy cập `http://127.0.0.1:8000/docs` trên trình duyệt để xem Swagger UI tự động tạo.
2. Tạo phiên và gửi nhiều tin nhắn liên tiếp để xác nhận ngữ cảnh hội thoại được duy trì.
3. Thử gửi yêu cầu API đến tác tử `email_refiner` sử dụng `APP_NAME` khác.

---

## 12.4 Server Sent Events (SSE) - Phản hồi streaming thời gian thực

### Chủ đề và mục tiêu

Học cách xử lý phản hồi streaming thời gian thực dựa trên Server-Sent Events sử dụng endpoint `/run_sse` thay vì endpoint đồng bộ `/run`.

### Khái niệm chính

#### 1) SSE (Server-Sent Events) là gì?

SSE là giao thức dựa trên HTTP để streaming dữ liệu thời gian thực một chiều từ server đến client. Khác với WebSocket, nó sử dụng kết nối HTTP thông thường nên triển khai đơn giản hơn.

**So sánh chế độ đồng bộ và chế độ SSE:**

| Đặc tính | `/run` (Đồng bộ) | `/run_sse` (Streaming) |
|----------|-------------------|----------------------|
| Cách phản hồi | Trả về toàn bộ phản hồi một lần | Gửi theo đơn vị sự kiện thời gian thực |
| Trải nghiệm người dùng | Chờ đến khi phản hồi hoàn tất | Xem tiến trình thời gian thực |
| Quan sát gọi công cụ | Bao gồm trong kết quả | Quan sát quy trình gọi thời gian thực |
| Phù hợp cho | Phản hồi ngắn, xử lý backend | Phản hồi dài, giao diện frontend |

#### 2) Triển khai client SSE

```python
import sseclient
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"
SESSION_ID = "3f673a5a-04ab-4edb-af23-6f42449a970b"

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "What is the weather there?"}],
        "role": "user",
    },
    "streaming": True,              # Cờ kích hoạt streaming
}

response = requests.post(
    f"{BASE_URL}/run_sse",           # Endpoint dành riêng cho SSE
    json=message,
    stream=True,                     # Kích hoạt chế độ streaming của requests
)

client = sseclient.SSEClient(response)

for event in client.events():
    data = json.loads(event.data)
    content = data.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

**Sự khác biệt mã so với chế độ đồng bộ:**

1. **`"streaming": True` thêm vào tin nhắn yêu cầu** - Thông báo chế độ streaming cho server.
2. **Thay đổi endpoint**: `/run_sse` thay vì `/run`
3. **Tùy chọn `stream=True`**: Kích hoạt chế độ streaming trên `requests.post()`
4. **Bọc bằng `sseclient.SSEClient`**: Phân tích phản hồi thành luồng sự kiện SSE
5. **Vòng lặp sự kiện**: Xử lý sự kiện từng cái với `client.events()`

#### 3) Cấu trúc sự kiện SSE

Mỗi sự kiện SSE chứa trường `data` ở định dạng JSON:

```json
{
    "content": {
        "parts": [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"}
                }
            }
        ]
    }
}
```

Hoặc phản hồi văn bản:

```json
{
    "content": {
        "parts": [
            {
                "text": "Thời tiết hiện tại ở Paris là 22 độ..."
            }
        ]
    }
}
```

### Điểm thực hành

1. Gửi cùng câu hỏi ở chế độ đồng bộ (`/run`) và chế độ SSE (`/run_sse`), so sánh sự khác biệt về thời gian phản hồi và trải nghiệm người dùng.
2. Quan sát rằng các lệnh gọi công cụ (functionCall) đến trước phản hồi văn bản khi nhận sự kiện SSE.
3. Viết mã phân tích luồng HTTP trực tiếp thay vì sử dụng `sseclient-py` (sử dụng `response.iter_lines()`).

---

## 12.6 Runner - Chạy tác tử trực tiếp từ mã

### Chủ đề và mục tiêu

Học cách chạy tác tử trực tiếp trong mã Python thuần sử dụng lớp `Runner` mà không cần ADK CLI hoặc server API. Cũng bao gồm quản lý phiên bền vững qua `DatabaseSessionService` và `InMemoryArtifactService`.

### Khái niệm chính

#### 1) Vai trò của Runner

`Runner` là trình điều phối cốt lõi cho thực thi tác tử. Sử dụng khi bạn muốn chạy tác tử trực tiếp trong mã mà không cần server API. Runner quản lý:
- Luồng thực thi tác tử
- Quản lý trạng thái phiên
- Quản lý artifact (tệp, v.v.)
- Streaming sự kiện

#### 2) Dịch vụ phiên và dịch vụ artifact

```python
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService

# Dịch vụ artifact: Dựa trên bộ nhớ (lưu trữ tạm thời tệp, v.v.)
in_memory_service_py = InMemoryArtifactService()

# Dịch vụ phiên: Dựa trên SQLite DB (lưu trữ phiên bền vững)
session_service = DatabaseSessionService(db_url="sqlite:///./session.db")
```

**Ưu điểm của `DatabaseSessionService`:**
- Dữ liệu phiên được lưu bền vững trong tệp SQLite (`session.db`).
- Có thể tiếp tục cuộc trò chuyện trước sau khi khởi động lại server.
- Thay đổi `db_url` sang PostgreSQL, v.v. cho phép sử dụng trong môi trường production.

#### 3) Tạo phiên và khởi tạo trạng thái

```python
session = await session_service.create_session(
    app_name="weather_agent",
    user_id="u_123",
    state={
        "user_name": "nico",    # Lưu tên người dùng vào trạng thái ban đầu
    },
)
```

Có thể đặt giá trị ban đầu trong dictionary `state`. Các giá trị này được tham chiếu làm biến mẫu trong instruction của tác tử:

```python
# prompt.py
TRAVEL_ADVISOR_INSTRUCTION = """
You are a helpful travel advisor agent...

You call the user by their name:

Their name is {user_name}
...
"""
```

`{user_name}` tự động được thay thế bằng giá trị `"user_name"` từ state phiên. Đây là tính năng **mẫu prompt dựa trên trạng thái** của ADK.

#### 4) Thực thi tác tử qua Runner

```python
from google.genai import types
from google.adk.runners import Runner

# Tạo Runner
runner = Runner(
    agent=travel_advisor_agent,           # Tác tử cần thực thi
    session_service=session_service,      # Dịch vụ quản lý phiên
    app_name="weather_agent",             # Tên app (phải khớp với dịch vụ phiên)
    artifact_service=in_memory_service_py, # Dịch vụ quản lý artifact
)

# Tạo tin nhắn người dùng
message = types.Content(
    role="user",
    parts=[
        types.Part(text="Im going to Vietnam, tell me all about it."),
    ],
)

# Thực thi streaming bất đồng bộ
async for event in runner.run_async(
    user_id="u_123",
    session_id=session.id,
    new_message=message
):
    if event.is_final_response():
        print(event.content.parts[0].text)
    else:
        print(event.get_function_calls())
        print(event.get_function_responses())
```

**Mẫu xử lý sự kiện:**
- `event.is_final_response()`: Kiểm tra xem có phải phản hồi văn bản cuối cùng không
- `event.get_function_calls()`: Kiểm tra sự kiện gọi công cụ
- `event.get_function_responses()`: Kiểm tra sự kiện phản hồi công cụ

#### 5) Phân tích kết quả thực thi

Xem kết quả thực thi thực tế, bạn có thể quan sát rõ ràng quá trình hoạt động của tác tử:

```
# Bước 1: Tác tử gọi 3 công cụ đồng thời (gọi công cụ song song)
[FunctionCall(name='get_weather', args={'location': 'Vietnam'}),
 FunctionCall(name='get_exchange_rate', args={'from_currency': 'USD', 'to_currency': 'VND', 'amount': 1}),
 FunctionCall(name='get_local_attractions', args={'location': 'Vietnam'})]

# Bước 2: Nhận phản hồi công cụ
[FunctionResponse(name='get_weather', response=<dict len=6>),
 FunctionResponse(name='get_exchange_rate', response=<dict len=6>),
 FunctionResponse(name='get_local_attractions', response={
     'error': "Invoking `get_local_attractions()` failed as the following
     mandatory input parameters are not present: category..."
 })]

# Bước 3: Phản hồi cuối (tổng hợp kết quả công cụ thành câu trả lời tự nhiên)
Hello Nico! Here's some information to help you prepare for your trip to Vietnam:

### Weather in Vietnam
- **Current Temperature:** 22°C
- **Condition:** Partly cloudy
...
```

Điểm đáng chú ý:
1. Tác tử đọc `{user_name}` từ state phiên và chào "Hello Nico!".
2. Gọi 3 công cụ **song song** để thu thập thông tin hiệu quả.
3. Dù `get_local_attractions` gặp lỗi thiếu tham số `category`, tác tử tự xử lý bằng cách trực tiếp tạo thông tin du lịch Việt Nam chung.

### Điểm thực hành

1. Thay `DatabaseSessionService` bằng `InMemorySessionService` và xác nhận phiên không được bảo tồn sau khi khởi động lại server.
2. Thêm `"preferred_language": "Korean"` vào `state` và sử dụng trong prompt để tác tử phản hồi bằng tiếng Hàn.
3. Tìm cách sử dụng phương thức đồng bộ `run` thay vì `run_async`.
4. Sử dụng `output_key` để lưu phản hồi tác tử vào state phiên và tham chiếu trong cuộc trò chuyện tiếp theo.

---

## 12.7 Deployment to Vertex AI - Triển khai đám mây

### Chủ đề và mục tiêu

Học cách triển khai tác tử ADK lên Vertex AI Agent Engine của Google Cloud để vận hành trong môi trường production.

### Khái niệm chính

#### 1) Vertex AI Agent Engine là gì?

Vertex AI Agent Engine (trước đây là Reasoning Engine) là dịch vụ lưu trữ và quản lý tác tử AI trên Google Cloud. Triển khai tác tử ADK lên đám mây mang lại:
- Không cần quản lý hạ tầng server
- Tự động mở rộng quy mô
- Tận dụng tính năng bảo mật và giám sát của Google Cloud
- Quản lý phiên từ xa và thực thi

#### 2) Script triển khai (deploy.py)

```python
import dotenv

dotenv.load_dotenv()

import os
import vertexai
import vertexai.agent_engines
from vertexai.preview import reasoning_engines
from travel_advisor_agent.agent import travel_advisor_agent

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"
BUCKET = "gs://nico-awesome-weather_agent"

# Khởi tạo Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,         # Bucket GCS để staging tệp triển khai
)

# Bọc tác tử ADK thành AdkApp
app = reasoning_engines.AdkApp(
    agent=travel_advisor_agent,
    enable_tracing=True,            # Kích hoạt theo dõi thực thi
)

# Triển khai lên Vertex AI
remote_app = vertexai.agent_engines.create(
    display_name="Travel Advisor Agent",
    agent_engine=app,
    requirements=[                  # Gói Python cần thiết
        "google-cloud-aiplatform[adk,agent_engines]",
        "litellm",
    ],
    extra_packages=["travel_advisor_agent"],  # Bao gồm gói tác tử
    env_vars={                      # Truyền biến môi trường
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    },
)
```

**Phân tích chi tiết quy trình triển khai:**

| Bước | Mã | Mô tả |
|------|-----|-------|
| 1. Thiết lập môi trường | `dotenv.load_dotenv()` | Tải biến môi trường như API key từ tệp `.env` |
| 2. Khởi tạo Vertex AI | `vertexai.init(...)` | Đặt dự án, vùng, bucket staging |
| 3. Bọc ứng dụng | `reasoning_engines.AdkApp(...)` | Bọc tác tử ADK sang định dạng tương thích Vertex AI |
| 4. Triển khai | `agent_engines.create(...)` | Thực thi triển khai thực tế lên đám mây |

**Vai trò tham số `extra_packages`:**
Bao gồm thư mục gói cục bộ (`travel_advisor_agent`) trong gói triển khai. Điều này cần thiết để mã tác tử có thể import trong môi trường đám mây.

**Quản lý bí mật qua `env_vars`:**
Thông tin nhạy cảm như API key được truyền dưới dạng biến môi trường. Không hard-code trực tiếp trong mã là điều quan trọng cho bảo mật.

#### 3) Phụ thuộc bổ sung

Gói thêm cho triển khai:

```toml
dependencies = [
    "cloudpickle>=3.1.1",                                    # Tuần tự hóa đối tượng
    "google-adk[eval]>=1.12.0",
    "google-cloud-aiplatform[adk,agent-engines]>=1.111.0",   # Vertex AI SDK
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",
    "sseclient-py>=1.8.0",
]
```

- **`cloudpickle`**: Dùng để tuần tự hóa đối tượng Python để truyền lên đám mây
- **`google-cloud-aiplatform[adk,agent-engines]`**: Bao gồm tính năng ADK và Agent Engine của Vertex AI

#### 4) Quản lý và thực thi tác tử từ xa (remote.py)

```python
import vertexai
from vertexai import agent_engines

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# Liệt kê các triển khai
# deployments = agent_engines.list()
# for deployment in deployments:
#     print(deployment)

# Lấy app từ xa theo ID triển khai cụ thể
DEPLOYMENT_ID = "projects/23382131925/locations/europe-southwest1/reasoningEngines/2153529862441140224"
remote_app = agent_engines.get(DEPLOYMENT_ID)

# Xóa triển khai (xóa cưỡng bức với force=True)
remote_app.delete(force=True)
```

**Tạo phiên từ xa và truy vấn streaming:**

```python
# Tạo phiên từ xa
# remote_session = remote_app.create_session(user_id="u_123")
# print(remote_session["id"])

SESSION_ID = "5724511082748313600"

# Gửi truy vấn streaming đến tác tử từ xa
# for event in remote_app.stream_query(
#     user_id="u_123",
#     session_id=SESSION_ID,
#     message="I'm going to Laos, any tips?",
# ):
#     print(event, "\n", "=" * 50)
```

**Tóm tắt API thực thi từ xa:**

| Phương thức | Mục đích |
|-------------|----------|
| `agent_engines.list()` | Liệt kê tất cả triển khai |
| `agent_engines.get(id)` | Lấy triển khai cụ thể |
| `remote_app.create_session(user_id=...)` | Tạo phiên từ xa |
| `remote_app.stream_query(...)` | Truy vấn chế độ streaming |
| `remote_app.delete(force=True)` | Xóa triển khai |

### Điểm thực hành

1. Tạo dự án GCP và bucket GCS, rồi thực sự triển khai tác tử.
2. Sau khi triển khai với `enable_tracing=True`, kiểm tra log tracing trong Google Cloud Console.
3. So sánh thời gian phản hồi của `remote_app.stream_query()` với thực thi Runner cục bộ.
4. Tạo phiên với nhiều user ID và xác nhận cách ly phiên hoạt động đúng.

---

## Tóm tắt chương

### 1. Kiến trúc tác tử ADK

```
                    ┌─────────────────────┐
                    │      ADK Agent      │
                    │                     │
                    │  - name             │
                    │  - description      │
                    │  - instruction      │
                    │  - model            │
                    │  - tools            │
                    │  - output_key       │
                    │  - sub_agents       │
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
         │  Agent  │   │ LoopAgent │   │  Runner   │
         │(Đơn lẻ)│   │ (Lặp)    │   │(Thực thi) │
         └─────────┘   └───────────┘   └───────────┘
```

### 2. So sánh chế độ thực thi

| Cách thực thi | Mô tả | Khi nào sử dụng |
|--------------|-------|-----------------|
| `adk web` | Test tác tử với giao diện web | Test nhanh trong phát triển |
| `adk api_server` | Chạy server REST API | Tích hợp frontend, dịch vụ cục bộ |
| `Runner` (Chế độ code) | Thực thi trực tiếp từ mã Python | Tích hợp ứng dụng tùy chỉnh |
| Triển khai Vertex AI | Môi trường production đám mây | Vận hành dịch vụ thực tế |

### 3. Tóm tắt các lớp/thành phần ADK chính

| Thành phần | Vai trò |
|-----------|---------|
| `Agent` | Định nghĩa tác tử đơn lẻ (tên, mô tả, chỉ thị, mô hình, công cụ) |
| `LoopAgent` | Trình điều phối thực thi tác tử con lặp đi lặp lại |
| `LiteLlm` | Giao diện thống nhất cho các nhà cung cấp LLM khác nhau |
| `ToolContext` | Truy cập state phiên, hành động từ hàm công cụ |
| `output_key` | Khóa lưu đầu ra tác tử vào state phiên |
| `escalate` | Kết thúc sớm vòng lặp hoặc chuỗi tác tử |
| `Runner` | Trình điều phối quản lý thực thi tác tử từ mã |
| `DatabaseSessionService` | Quản lý phiên bền vững dựa trên DB |
| `InMemoryArtifactService` | Quản lý artifact dựa trên bộ nhớ |
| `reasoning_engines.AdkApp` | Bọc tác tử ADK cho triển khai Vertex AI |

### 4. Mẫu luồng dữ liệu cốt lõi

```
Lưu với output_key → Tích lũy trong state → Tham chiếu qua {tên_biến} trong instruction
```

Mẫu này là cơ chế quan trọng nhất để truyền dữ liệu giữa các tác tử trong ADK.

---

## Bài tập thực hành

### Bài tập 1: Tác tử đánh giá mã (Sử dụng LoopAgent)

Tham khảo cấu trúc Email Refiner Agent để tạo **tác tử đánh giá mã**.

**Yêu cầu:**
- `SecurityReviewAgent`: Đánh giá lỗ hổng bảo mật
- `PerformanceReviewAgent`: Đề xuất tối ưu hiệu suất
- `StyleReviewAgent`: Đánh giá phong cách mã và khả năng đọc
- `ReviewSynthesizerAgent`: Tổng hợp tất cả đánh giá
- `ApprovalAgent`: Quyết định phê duyệt/từ chối cuối cùng (sử dụng công cụ escalate)

**Gợi ý:**
- Đặt `output_key` cho mỗi tác tử để lưu kết quả đánh giá vào state
- Cung cấp công cụ `escalate_review_complete` cho `ApprovalAgent`
- Đặt `max_iterations` của `LoopAgent` phù hợp

### Bài tập 2: Server API và client SSE

Mở rộng Travel Advisor Agent thêm **tính năng gợi ý nhà hàng**, và triển khai server API cùng client SSE.

**Yêu cầu:**
1. Thêm hàm công cụ `get_restaurant_recommendations(location, cuisine_type)`
2. Chạy server với `adk api_server`
3. Nhận phản hồi streaming thời gian thực với client SSE
4. Phân biệt sự kiện gọi công cụ và sự kiện phản hồi văn bản để hiển thị trên UI

### Bài tập 3: CLI tương tác với Runner

Tạo chương trình CLI tương tác với tác tử qua terminal sử dụng Runner.

**Yêu cầu:**
1. Sử dụng `DatabaseSessionService` để lưu bền vững lịch sử hội thoại
2. Khi khởi động chương trình, chọn tiếp tục phiên hiện có hoặc tạo phiên mới
3. Lưu ngôn ngữ ưu thích của người dùng trong `state` và sử dụng trong prompt
4. In ID phiên khi thoát bằng `Ctrl+C` để có thể tiếp tục lần sau

### Bài tập 4: Triển khai Vertex AI (Nâng cao)

Thực sự triển khai Travel Advisor Agent lên Vertex AI và sử dụng từ xa.

**Yêu cầu:**
1. Tạo dự án GCP và kích hoạt API Vertex AI
2. Tạo bucket GCS (cho staging)
3. Viết script triển khai tham khảo `deploy.py`
4. Tạo phiên từ xa và thực thi truy vấn tham khảo `remote.py`
5. So sánh và phân tích thời gian phản hồi giữa tác tử đã triển khai và thực thi cục bộ

**Lưu ý:**
- Có thể phát sinh phí GCP, nên nhớ xóa bằng `remote_app.delete(force=True)` sau khi test
- Không bao giờ hard-code API key trong mã; luôn truyền dưới dạng biến môi trường

---

> **Xem trước chương tiếp theo:** Chương tiếp theo bao gồm framework đánh giá tác tử, nơi chúng ta học cách đo lường và cải thiện chất lượng phản hồi tác tử một cách có hệ thống.
