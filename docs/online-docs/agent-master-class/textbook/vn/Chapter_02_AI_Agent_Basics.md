# Chapter 2: Nền tảng AI Agent

---

## Tổng quan chương

Trong chương này, chúng ta sẽ học từng bước cách xây dựng một AI agent từ đầu bằng OpenAI API. Bắt đầu từ một lệnh gọi API đơn giản, chúng ta dần dần thêm bộ nhớ hội thoại, định nghĩa các công cụ bên ngoài (Tool), và triển khai Function Calling để AI có thể thực thi các hàm thực tế, hoàn thành một agent hoàn chỉnh.

### Mục tiêu học tập

1. Thiết lập môi trường phát triển Python và xác nhận kết nối OpenAI API
2. Nắm vững cách sử dụng cơ bản của OpenAI Chat Completions API
3. Xây dựng chatbot duy trì ngữ cảnh bằng cách quản lý lịch sử hội thoại (bộ nhớ)
4. Định nghĩa Tool schema để thông báo cho AI về các hàm có sẵn
5. Triển khai Function Calling để AI thực sự thực thi các hàm được chọn
6. Truyền kết quả thực thi công cụ trở lại AI để tạo phản hồi cuối cùng

### Cấu trúc chương

| Phần | Chủ đề | Từ khóa chính |
|------|--------|--------------|
| 2.0 | Thiết lập dự án | uv, Python 3.13, OpenAI SDK |
| 2.2 | AI Agent đầu tiên | Chat Completions API, Prompt Engineering |
| 2.3 | Thêm bộ nhớ | Lịch sử hội thoại, mảng messages, vòng lặp while |
| 2.4 | Thêm công cụ | Tools Schema, JSON Schema, FUNCTION_MAP |
| 2.5 | Thêm Function Calling | Function Calling, tool_calls, process_ai_response |
| 2.6 | Kết quả công cụ | Tool Results, gọi đệ quy, hoàn thành vòng lặp Agent |

---

## 2.0 Thiết lập dự án (Setup)

### Chủ đề và mục tiêu

Thiết lập môi trường dự án Python để phát triển AI agent. Khởi tạo dự án bằng trình quản lý gói `uv`, cài đặt OpenAI Python SDK, và xác nhận API key được tải đúng cách trong môi trường Jupyter Notebook.

### Giải thích khái niệm cốt lõi

#### Trình quản lý gói uv

`uv` là trình quản lý gói Python thế hệ mới được viết bằng Rust, cung cấp tốc độ giải quyết và cài đặt phụ thuộc nhanh hơn nhiều so với các công cụ truyền thống như `pip` hay `poetry`. Trong dự án này, chúng ta sử dụng `uv` để tạo môi trường ảo và quản lý các gói.

#### Cấu trúc dự án

Dự án được tổ chức như sau:

```
my-first-agent/
├── .gitignore          # Danh sách file không được Git theo dõi
├── .python-version     # Chỉ định phiên bản Python (3.13)
├── README.md           # Mô tả dự án
├── main.ipynb          # Notebook chính (không gian viết code)
├── pyproject.toml      # Cấu hình dự án và định nghĩa phụ thuộc
└── uv.lock             # File khóa phụ thuộc
```

### Phân tích code

#### pyproject.toml - File cấu hình dự án

```toml
[project]
name = "my-first-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "openai>=1.98.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.0",
]
```

**Điểm chính:**

- `requires-python = ">=3.13"`: Yêu cầu Python 3.13 trở lên. Điều này để tận dụng các tính năng Python mới nhất.
- `dependencies`: Chỉ định `openai>=1.98.0` là phụ thuộc runtime. Đây là SDK Python chính thức để giao tiếp với OpenAI API.
- `[dependency-groups] dev`: Bao gồm `ipykernel` là phụ thuộc phát triển. Cần thiết để chạy code Python trong Jupyter Notebook.

#### .python-version

```
3.13
```

File này cho phép các công cụ như `uv` hoặc `pyenv` tự động nhận biết phiên bản Python sử dụng cho dự án.

#### main.ipynb - Xác minh API Key

```python
import os

print(os.getenv("OPENAI_API_KEY"))
```

Code này xác minh rằng OpenAI API key đã được thiết lập đúng trong biến môi trường. Vì lý do bảo mật, API key không bao giờ nên được hardcode trực tiếp trong mã nguồn mà phải được quản lý thông qua file `.env` hoặc biến môi trường hệ thống.

#### .gitignore - Cấu hình file bị bỏ qua

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv
.env
```

**Quan trọng:** File `.env` được bao gồm trong `.gitignore`. Điều này ngăn các thông tin nhạy cảm như API key bị tải lên kho Git. Đây là cấu hình bảo mật cực kỳ quan trọng.

### Điểm thực hành

1. Khởi tạo dự án bằng lệnh `uv init my-first-agent`
2. Cài đặt OpenAI SDK bằng `uv add openai`
3. Cài đặt Jupyter kernel bằng `uv add --dev ipykernel`
4. Tạo file `.env` và lưu API key theo định dạng `OPENAI_API_KEY=sk-...`
5. Chạy Jupyter Notebook và xác nhận API key được in ra đúng

---

## 2.2 AI Agent đầu tiên (Your First AI Agent)

### Chủ đề và mục tiêu

Tạo dạng AI agent cơ bản nhất bằng OpenAI Chat Completions API. Ở giai đoạn này, chúng ta thử nghiệm việc thông báo cho AI về các hàm có sẵn thông qua prompt engineering (dưới dạng văn bản thuần túy) và hướng dẫn nó chọn hàm phù hợp.

### Giải thích khái niệm cốt lõi

#### Chat Completions API

Chat Completions API của OpenAI là giao diện chính để tương tác với các mô hình AI hội thoại. Bạn gửi tin nhắn và AI tạo và trả về phản hồi. Mỗi tin nhắn bao gồm `role` (vai trò) và `content` (nội dung).

Các loại vai trò:
- `system`: Tin nhắn hệ thống chỉ dẫn hành vi của AI
- `user`: Tin nhắn do người dùng gửi
- `assistant`: Tin nhắn phản hồi do AI tạo ra
- `tool`: Tin nhắn truyền kết quả thực thi công cụ (được học trong các phần sau)

#### Chọn hàm thông qua Prompt Engineering

Ở giai đoạn này, chúng ta chưa sử dụng tính năng Function Calling chính thức của OpenAI. Thay vào đó, chúng ta sử dụng prompt (hướng dẫn văn bản) để yêu cầu AI "đây là các hàm có sẵn, hãy chọn hàm phù hợp." Đây là dạng nguyên thủy nhất của AI agent.

### Phân tích code

#### Chọn hàm dựa trên Prompt

```python
import openai

client = openai.OpenAI()

PROMPT = """
I have the following functions in my system.

`get_weather`
`get_currency`
`get_news`

All of them receive the name of a country as an argumet (i.e get_news('Spain'))

Please answer with the name of the function that you would like me to run.

Please say nothing else, just the name of the function with the arguments.

Answer the following question:

What is the weather in Greece?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": PROMPT}],
)

response
```

**Phân tích code:**

1. **`openai.OpenAI()`**: Tạo client OpenAI. Tự động đọc biến môi trường `OPENAI_API_KEY`.
2. **`PROMPT`**: Định nghĩa prompt nhiều dòng. Thông báo cho AI về các hàm có sẵn (`get_weather`, `get_currency`, `get_news`) và yêu cầu chọn hàm phù hợp với câu hỏi.
3. **`client.chat.completions.create()`**: Thực hiện lệnh gọi API. Tham số `model` chỉ định mô hình sử dụng, tham số `messages` truyền nội dung hội thoại.

#### Trích xuất tin nhắn từ phản hồi

```python
message = response.choices[0].message.content
message
```

**Kết quả đầu ra:**
```
"get_weather('Greece')"
```

**Hiểu biết cốt lõi:**

- `response.choices[0]`: Lấy lựa chọn đầu tiên từ phản hồi API (thường chỉ trả về một)
- `.message.content`: Trích xuất nội dung tin nhắn (văn bản) từ lựa chọn đó
- AI đã tuân theo hướng dẫn trong prompt và trả về văn bản dạng gọi hàm: `get_weather('Greece')`

#### Hạn chế của phương pháp này

Mặc dù phương pháp này hoạt động, nó có một số vấn đề:

- Không có gì đảm bảo rằng phản hồi của AI luôn ở định dạng nhất quán (ví dụ: `"get_weather('Greece')"` vs `"I would call get_weather with Greece"`)
- Cần xử lý chuỗi bổ sung để phân tích văn bản trả về và thực sự gọi hàm
- Khó truyền chính xác kiểu tham số và tính bắt buộc của hàm cho AI

Để giải quyết những hạn chế này, OpenAI cung cấp tính năng **Function Calling** chính thức, được học từ phần 2.4 trở đi.

### Điểm thực hành

1. Sửa prompt để hỏi câu hỏi khác (ví dụ: "What is the currency of Japan?")
2. Xác nhận AI trả về `get_currency('Japan')`
3. Xóa phần "Please say nothing else" khỏi prompt và quan sát phản hồi AI thay đổi như thế nào
4. So sánh sự khác biệt phản hồi khi sử dụng mô hình khác (ví dụ: `gpt-4o`)

---

## 2.3 Thêm bộ nhớ (Adding Memory)

### Chủ đề và mục tiêu

Xây dựng chatbot AI nhớ được các cuộc hội thoại trước đó. Sử dụng mảng `messages` để quản lý lịch sử hội thoại, cho phép đối thoại liên tục giữa người dùng và AI.

### Giải thích khái niệm cốt lõi

#### Nguyên lý hoạt động của bộ nhớ hội thoại

LLM (Large Language Model) về cơ bản là hệ thống **không lưu trạng thái (stateless)**. Mỗi lệnh gọi API là độc lập và không tự động nhớ các cuộc hội thoại trước đó. Do đó, để duy trì ngữ cảnh hội thoại, **tất cả tin nhắn trước đó phải được gửi cùng mỗi lệnh gọi API**.

Đây chính là vai trò của mảng `messages`. Mỗi khi người dùng gửi tin nhắn mới:

1. Thêm tin nhắn của người dùng vào mảng `messages`
2. Gửi toàn bộ mảng `messages` đến API
3. Thêm phản hồi của AI trở lại mảng `messages`
4. Trong lượt hội thoại tiếp theo, toàn bộ lịch sử này được gửi cùng nhau

```
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico!"},
    {"role": "user", "content": "What is my name?"},
    # AI có thể thấy lịch sử ở trên và trả lời "Your name is Nico."
]
```

### Phân tích code

#### Thiết lập ban đầu

```python
import openai

client = openai.OpenAI()
messages = []
```

Khởi tạo `messages` là mảng rỗng. Mảng này đóng vai trò **bộ nhớ** lưu trữ toàn bộ lịch sử hội thoại.

#### Định nghĩa hàm gọi AI

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": message})
    print(f"AI: {message}")
```

**Phân tích code:**

1. Toàn bộ mảng `messages` được truyền cho API để duy trì ngữ cảnh hội thoại
2. Phản hồi của AI (`message.content`) được trích xuất
3. Phản hồi được thêm vào mảng `messages` dưới dạng `{"role": "assistant", "content": message}` -- đây là hành động **lưu vào bộ nhớ**
4. Phản hồi được hiển thị trên màn hình

#### Vòng lặp hội thoại

```python
while True:
    message = input("Send a message to the LLM...")
    if message == "quit" or message == "q":
        break
    else:
        messages.append({"role": "user", "content": message})
        print(f"User: {message}")
        call_ai()
```

**Phân tích code:**

1. `while True`: Vòng lặp vô hạn để tiếp tục hội thoại
2. `input()`: Nhận tin nhắn từ người dùng
3. Nhập `"quit"` hoặc `"q"` để thoát vòng lặp
4. Tin nhắn người dùng được thêm vào `messages` dưới dạng `{"role": "user", "content": message}`, sau đó gọi `call_ai()`

#### Kết quả thực thi (Xác minh hoạt động bộ nhớ)

```
User: My name is Nico
AI: Nice to meet you, Nico! How can I assist you today?
User: What is my name?
AI: Your name is Nico.
User: I'm from Korea
AI: That's great! Korea has a rich culture and history. ...
User: What was the first question I asked you and what is the closest Island country to where I was born?
AI: The first question you asked was, "What is my name?" As for the closest island country to Korea, that would be Japan...
```

**Những kết quả này cho thấy:**

- AI nhớ tên người dùng ("Nico")
- AI nhớ quê hương người dùng ("Korea")
- AI thậm chí nhớ câu hỏi đầu tiên là gì
- Tất cả điều này có thể nhờ các cuộc hội thoại trước đó được tích lũy trong mảng `messages`

### Trực quan hóa cấu trúc mảng messages

```
Lệnh gọi API #1:
messages = [
    {"role": "user", "content": "My name is Nico"}
]

Lệnh gọi API #2:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"}
]

Lệnh gọi API #3:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"},
    {"role": "assistant", "content": "Your name is Nico."},
    {"role": "user", "content": "I'm from Korea"}
]
```

Lưu ý rằng toàn bộ lịch sử được gửi với mỗi lệnh gọi, do đó chi phí API (lượng token sử dụng) tăng lên khi cuộc hội thoại dài hơn.

### Điểm thực hành

1. Tiếp tục cuộc hội thoại dài và kiểm tra AI nhớ tốt đến mức nào
2. In trực tiếp mảng `messages` để kiểm tra cấu trúc bên trong
3. Reset bằng `messages = []` giữa cuộc hội thoại và xác nhận AI quên đối thoại trước đó
4. Thêm tin nhắn vai trò `system` để thay đổi tính cách AI (ví dụ: `{"role": "system", "content": "You are a pirate. Speak like a pirate."}`)

---

## 2.4 Thêm công cụ (Adding Tools)

### Chủ đề và mục tiêu

Sử dụng tính năng **Tools** chính thức của OpenAI để thông báo cấu trúc cho AI về các hàm có sẵn. Định nghĩa tên hàm, mô tả và tham số bằng JSON Schema, và quan sát cách `finish_reason` thay đổi thành `tool_calls` khi AI quyết định gọi công cụ.

### Giải thích khái niệm cốt lõi

#### So sánh phương pháp dựa trên Prompt và phương pháp dựa trên Tools

Trong phần 2.2, chúng ta sử dụng văn bản prompt để thông báo cho AI về các hàm có sẵn. Điều này không ổn định và khó phân tích. Tính năng **Tools** của OpenAI thay thế bằng JSON Schema có cấu trúc:

| Khía cạnh | Dựa trên Prompt (2.2) | Dựa trên Tools (2.4) |
|-----------|----------------------|---------------------|
| Phương pháp định nghĩa hàm | Văn bản ngôn ngữ tự nhiên | JSON Schema |
| Định dạng phản hồi | Văn bản tự do | Đối tượng tool_calls có cấu trúc |
| Định nghĩa tham số | Không rõ ràng | Kiểu dữ liệu, tính bắt buộc rõ ràng |
| Độ khó phân tích | Cao | Thấp (SDK xử lý) |

#### Mẫu FUNCTION_MAP

Khi AI trả về tên hàm, bạn cần tìm và thực thi hàm Python thực tế theo tên đó. Để làm điều này, chúng ta sử dụng **từ điển ánh xạ tên hàm (chuỗi) sang đối tượng hàm**:

```python
FUNCTION_MAP = {"get_weather": get_weather}
```

Mẫu này cho phép gọi động hàm `get_weather` bằng chuỗi `"get_weather"` được AI trả về.

### Phân tích code

#### Định nghĩa và ánh xạ hàm

```python
def get_weather(city):
    return "33 degrees celcius."


FUNCTION_MAP = {"get_weather": get_weather}
```

- `get_weather`: Hàm trả về thông tin thời tiết. Hiện tại trả về giá trị hardcode, nhưng trong production sẽ gọi API thời tiết.
- `FUNCTION_MAP`: Ánh xạ khóa chuỗi sang đối tượng hàm. Sau này khi AI phản hồi `"get_weather"`, chúng ta tìm hàm thực tế qua `FUNCTION_MAP["get_weather"]`.

#### Định nghĩa Tools Schema (Cốt lõi)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "A function to get the weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the weather of.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]
```

**Phân tích chi tiết cấu trúc Schema:**

1. **`"type": "function"`**: Chỉ định kiểu công cụ là hàm.

2. **Đối tượng `"function"`:**
   - `"name"`: Tên hàm. AI sử dụng tên này để yêu cầu gọi hàm.
   - `"description"`: Mô tả về hàm. AI sử dụng điều này để xác định khi nào sử dụng hàm. **Viết description tốt là cực kỳ quan trọng.**
   - `"parameters"`: Định nghĩa tham số theo định dạng JSON Schema.
     - `"type": "object"`: Chỉ ra tham số ở dạng đối tượng.
     - `"properties"`: Định nghĩa tên, kiểu và mô tả của mỗi tham số.
     - `"required"`: Chỉ định danh sách tham số bắt buộc dưới dạng mảng.

3. **`TOOLS` là một mảng.** Có thể định nghĩa nhiều công cụ và cung cấp cho AI.

#### Truyền Tools vào lệnh gọi API

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    print(response)
    message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": message})
    print(f"AI: {message}")
```

**Thay đổi:** Tham số `tools=TOOLS` đã được thêm vào. Bây giờ AI có thể gọi các hàm phù hợp trong cuộc hội thoại.

#### Phân tích kết quả thực thi

Với cuộc hội thoại bình thường:
```
User: my name is nico
AI: Nice to meet you, Nico! How can I assist you today?
```
- `finish_reason` là `'stop'` -- AI phản hồi bằng văn bản thông thường.
- `tool_calls` là `None`.

Với câu hỏi cần công cụ:
```
User: what is the weather in Spain
AI: None
```
- `finish_reason` đã thay đổi thành `'tool_calls'`!
- Mảng `tool_calls` chứa thông tin về hàm cần gọi:
  ```python
  tool_calls=[
      ChatCompletionMessageToolCall(
          id='call_yTID1R7DPur7eJMWlobM8tgu',
          function=Function(
              arguments='{"city":"Spain"}',
              name='get_weather'
          ),
          type='function'
      )
  ]
  ```
- `message.content` là `None` -- vì AI chọn gọi công cụ thay vì tạo văn bản.

**Đây là điểm mấu chốt:** Thay vì trực tiếp trả lời "cho tôi biết thời tiết", AI đã yêu cầu "hãy gọi hàm get_weather với đối số city='Spain'." Tuy nhiên, vì chưa có code xử lý yêu cầu này, `AI: None` được in ra.

### Điểm thực hành

1. Thêm schema hàm `get_news` vào `TOOLS`
2. Xen kẽ các câu hỏi cần công cụ và không cần công cụ, quan sát sự thay đổi của `finish_reason`
3. Thay đổi `description` và thử nghiệm hành vi chọn công cụ của AI thay đổi như thế nào
4. In chi tiết đối tượng `response` để tự kiểm tra cấu trúc `tool_calls`

---

## 2.5 Thêm Function Calling

### Chủ đề và mục tiêu

Triển khai hàm `process_ai_response` thực sự xử lý các lệnh gọi công cụ mà AI yêu cầu. Viết logic phân nhánh thực thi hàm tương ứng khi phản hồi AI chứa `tool_calls`, và xử lý như phản hồi văn bản thông thường nếu không.

### Giải thích khái niệm cốt lõi

#### Luồng hoàn chỉnh của Function Calling

```
Câu hỏi người dùng -> AI phán đoán -> Phản hồi tool_calls -> Thực thi hàm -> Thêm kết quả vào messages -> Gọi lại AI -> Phản hồi cuối cùng
```

Trong phần này, chúng ta triển khai đến "Thực thi hàm" và "Thêm kết quả vào messages." Phần "Gọi lại AI" được hoàn thành trong phần tiếp theo (2.6).

#### Cấu trúc phản hồi tool_calls

Khi AI quyết định sử dụng công cụ, đối tượng `message` của phản hồi bao gồm thông tin sau:

```python
message.tool_calls = [
    ChatCompletionMessageToolCall(
        id='call_yTID1R7DPur7eJMWlobM8tgu',    # ID duy nhất
        function=Function(
            name='get_weather',                   # Tên hàm cần gọi
            arguments='{"city":"Spain"}'           # Đối số dạng chuỗi JSON
        ),
        type='function'
    )
]
```

Điểm chú ý:
- `id`: Mỗi lệnh gọi công cụ được gán một ID duy nhất. ID này được sử dụng sau để khớp kết quả thuộc về lệnh gọi nào.
- `arguments`: Đây là **chuỗi JSON**, không phải từ điển Python, do đó phải được phân tích bằng `json.loads()`.
- `tool_calls` là **mảng**. AI có thể gọi nhiều hàm cùng lúc.

### Phân tích code

#### Thêm import

```python
import openai, json
```

Module `json` đã được thêm. Cần thiết để phân tích đối số hàm mà AI trả về dạng chuỗi JSON.

#### Hàm process_ai_response (Logic cốt lõi)

```python
from openai.types.chat import ChatCompletionMessage


def process_ai_response(message: ChatCompletionMessage):
    if message.tool_calls > 0:
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Calling function: {function_name} with {arguments}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

            function_to_run = FUNCTION_MAP.get(function_name)

            result = function_to_run(**arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result,
                }
            )
    else:
        messages.append({"role": "assistant", "content": message.content})
        print(f"AI: {message.content}")
```

**Phân tích chi tiết từng bước:**

**Bước 1: Quyết định phân nhánh**
```python
if message.tool_calls > 0:
```
Kiểm tra phản hồi AI có chứa `tool_calls` không. Nếu có, thực thi logic gọi công cụ; nếu không, thực thi logic phản hồi văn bản thông thường.

> Lưu ý: Câu điều kiện này sau đó được sửa thành `if message.tool_calls:` trong phần 2.6. Vì `None > 0` có thể gây ra `TypeError` trong Python. Kiểm tra truthy/falsy an toàn hơn.

**Bước 2: Thêm tin nhắn assistant vào lịch sử**
```python
messages.append(
    {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ],
    }
)
```

Ghi lại phản hồi gọi công cụ của AI vào mảng `messages`. **Điều này cực kỳ quan trọng.** OpenAI API xem lịch sử này trong lệnh gọi tiếp theo để hiểu rằng một lệnh gọi công cụ đã được thực hiện. Vì `content` có thể là `None`, chúng ta sử dụng `message.content or ""` để mặc định là chuỗi rỗng.

**Bước 3: Lặp qua mỗi lệnh gọi công cụ và thực thi**
```python
for tool_call in message.tool_calls:
    function_name = tool_call.function.name      # "get_weather"
    arguments = tool_call.function.arguments      # '{"city":"Spain"}'

    print(f"Calling function: {function_name} with {arguments}")

    try:
        arguments = json.loads(arguments)         # {"city": "Spain"}
    except json.JSONDecodeError:
        arguments = {}

    function_to_run = FUNCTION_MAP.get(function_name)  # Đối tượng hàm get_weather

    result = function_to_run(**arguments)          # get_weather(city="Spain")
```

- `json.loads()`: Chuyển đổi chuỗi JSON thành từ điển Python
- `try/except`: Code phòng thủ xử lý lỗi phân tích JSON
- `FUNCTION_MAP.get()`: Tìm hàm thực tế theo tên chuỗi
- `**arguments`: Giải nén từ điển thành đối số từ khóa. `{"city": "Spain"}` trở thành `city="Spain"`

**Bước 4: Thêm kết quả thực thi công cụ vào lịch sử**
```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": result,
    }
)
```

- `"role": "tool"`: Chỉ ra tin nhắn này là kết quả thực thi công cụ
- `"tool_call_id"`: ID khớp với kết quả thuộc về lệnh gọi công cụ nào. Thiếu ID này gây lỗi API
- `"content"`: Kết quả thực thi hàm (ở đây là "33 degrees celcius.")

#### Code bổ trợ để hiểu toán tử ** (Giải nén)

Commit bao gồm code thử nghiệm để hiểu toán tử `**`:

```python
a = '{"city": "Spain"}'

b = json.loads(a)    # b = {"city": "Spain"}

**b                   # Giải nén: city="Spain"

get_weather(city='Spain')
```

Code này là ví dụ học tập cho thấy cách chuỗi JSON được chuyển đổi thành đối số gọi hàm.

#### Hàm call_ai đã đơn giản hóa

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

Logic xử lý phản hồi từ `call_ai` trước đó đã được tách ra thành `process_ai_response` để code gọn gàng hơn.

### Điểm thực hành

1. Thêm `print(messages)` ở mỗi bước trong `process_ai_response` để theo dõi lịch sử thay đổi
2. Kiểm tra lỗi gì xảy ra khi AI cố gọi hàm không có trong `FUNCTION_MAP` (gợi ý: `None(**arguments)` gây `TypeError`)
3. Thêm hàm và schema `get_currency` để mở rộng agent hỗ trợ nhiều công cụ

---

## 2.6 Kết quả thực thi công cụ (Tool Results)

### Chủ đề và mục tiêu

Truyền kết quả thực thi công cụ trở lại AI để AI tạo phản hồi cuối cùng tự nhiên dựa trên kết quả đó. Điều này hoàn thành **vòng lặp agent**.

### Giải thích khái niệm cốt lõi

#### Vòng lặp Agent (Agent Loop)

Một AI agent hoàn chỉnh tạo thành vòng lặp sau:

```
Câu hỏi người dùng
    |
AI phán đoán ---> Nếu phản hồi thường -> Xuất văn bản cho người dùng (kết thúc vòng lặp)
    |
Nếu cần gọi công cụ
    |
Thực thi hàm -> Thêm kết quả vào messages
    |
Gọi lại AI (gọi call_ai lần nữa)
    |
AI phán đoán ---> Nếu phản hồi thường -> Xuất văn bản cho người dùng (kết thúc vòng lặp)
    |
Nếu cần gọi công cụ nữa -> Thực thi hàm lại... (lặp lại)
```

Điểm mấu chốt của vòng lặp này là **AI được gọi lại sau khi thực thi công cụ**. AI nhận kết quả thực thi công cụ và định dạng phù hợp cho người dùng.

#### Vấn đề của phần trước (2.5)

Trong 2.5, chúng ta chỉ triển khai thực thi công cụ và thêm kết quả vào `messages`. Tuy nhiên, chúng ta không truyền kết quả đó trở lại AI, nên AI không thể tạo câu trả lời cuối cùng. Phần này giải quyết vấn đề đó.

### Phân tích code

#### process_ai_response đã sửa (Phiên bản cuối cùng)

```python
from openai.types.chat import ChatCompletionMessage


def process_ai_response(message: ChatCompletionMessage):

    if message.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Calling function: {function_name} with {arguments}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

            function_to_run = FUNCTION_MAP.get(function_name)

            result = function_to_run(**arguments)

            print(f"Ran {function_name} with args {arguments} for a result of {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result,
                }
            )

        call_ai()
    else:
        messages.append({"role": "assistant", "content": message.content})
        print(f"AI: {message.content}")


def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

**Ba thay đổi chính:**

**1. Sửa điều kiện: `message.tool_calls > 0` -> `message.tool_calls`**

```python
# Trước (2.5)
if message.tool_calls > 0:

# Sau (2.6)
if message.tool_calls:
```

Trong Python, `None > 0` có thể gây `TypeError`. Vì `tool_calls` là falsy khi `None` và truthy khi có danh sách, cách này an toàn hơn.

**2. Thêm đầu ra debug**

```python
print(f"Ran {function_name} with args {arguments} for a result of {result}")
```

In kết quả thực thi hàm ra console để hỗ trợ debug.

**3. Gọi lại AI sau khi thực thi công cụ (Thay đổi quan trọng nhất)**

```python
        # Sau khi tất cả công cụ đã được thực thi
        call_ai()
```

Một dòng này hoàn thành vòng lặp agent. Sau khi thêm tất cả kết quả công cụ vào `messages`, `call_ai()` được gọi lại. AI sau đó nhận toàn bộ lịch sử bao gồm kết quả thực thi công cụ và tạo phản hồi cuối cùng.

**Theo dõi luồng gọi:**

```
call_ai()                          # Lần gọi thứ 1
  -> AI: trả về tool_calls
  -> process_ai_response()
    -> Thực thi công cụ, thêm kết quả vào messages
    -> call_ai()                    # Lần gọi thứ 2 (đệ quy)
      -> AI: trả về phản hồi văn bản thường
      -> process_ai_response()
        -> Nhánh else: in văn bản
```

Đây là mẫu **đệ quy tương hỗ**: `call_ai` -> `process_ai_response` -> `call_ai` -> `process_ai_response` -> ...

#### Kết quả thực thi (Hoạt động Agent hoàn chỉnh)

```
User: My name is Nico
AI: Hello, Nico! How can I assist you today?
User: What is my name
AI: Your name is Nico.
User: What is the weather in Spain
Calling function: get_weather with {"city":"Spain"}
Ran get_weather with args {'city': 'Spain'} for a result of 33 degrees celcius.
AI: The weather in Spain is 33 degrees Celsius. If you need more specific weather details for a particular city or region in Spain, just let me know!
```

**Phân tích quá trình hoạt động:**

1. Với câu hỏi "What is the weather in Spain", AI quyết định gọi `get_weather(city="Spain")`
2. Hàm được thực thi và trả về "33 degrees celcius."
3. Kết quả này được thêm vào `messages` và AI được gọi lại
4. AI biến đổi dữ liệu thô "33 degrees celcius." thành câu tự nhiên: "The weather in Spain is 33 degrees Celsius. If you need more specific weather details..."

#### Kiểm tra mảng messages cuối cùng

```python
messages
```

Đầu ra:
```python
[
    {'role': 'user', 'content': 'My name is Nico'},
    {'role': 'assistant', 'content': 'Hello, Nico! How can I assist you today?'},
    {'role': 'user', 'content': 'What is my name'},
    {'role': 'assistant', 'content': 'Your name is Nico.'},
    {'role': 'user', 'content': 'What is the weather in Spain'},
    {'role': 'assistant',
     'content': '',
     'tool_calls': [{'id': 'call_za6hozI93riBO1tzf0gdPOwt',
       'type': 'function',
       'function': {'name': 'get_weather', 'arguments': '{"city":"Spain"}'}}]},
    {'role': 'tool',
     'tool_call_id': 'call_za6hozI93riBO1tzf0gdPOwt',
     'name': 'get_weather',
     'content': '33 degrees celcius.'},
    {'role': 'assistant',
     'content': 'The weather in Spain is 33 degrees Celsius. If you need more specific weather details for a particular city or region in Spain, just let me know!'}
]
```

Mảng này cho thấy toàn bộ quá trình hoạt động của agent:
1. Hội thoại thông thường (user -> assistant)
2. Yêu cầu gọi công cụ (assistant with tool_calls)
3. Kết quả thực thi công cụ (tool)
4. Phản hồi cuối cùng dựa trên kết quả (assistant)

### Điểm thực hành

1. Thêm hàm `get_news` và `get_currency`, kiểm tra với câu hỏi phức hợp ("What is the weather and news in Korea?") để xem nhiều công cụ có được gọi đồng thời không
2. Suy nghĩ tại sao các lệnh gọi đệ quy không rơi vào vòng lặp vô hạn (gợi ý: khi AI nhận kết quả công cụ, nó phản hồi bằng văn bản thường, rơi vào nhánh `else`)
3. In mảng `messages` để xác nhận trực quan các tin nhắn của mỗi vai trò (user, assistant, tool) tích lũy như thế nào
4. Cố tình thay đổi kết quả thực thi công cụ thành thông báo lỗi và kiểm tra AI phản ứng như thế nào

---

## Tóm tắt điểm chính của chương

### 1. Bản chất của AI Agent

AI agent là sự kết hợp của **LLM + Công cụ + Vòng lặp**. LLM ra quyết định, công cụ thực thi hành động, và vòng lặp kết nối chúng lặp đi lặp lại.

### 2. Mảng messages CHÍNH LÀ bộ nhớ

LLM không duy trì trạng thái. Tích lũy các cuộc hội thoại trước đó trong mảng `messages` và gửi chúng mỗi lần chính là thực chất của "bộ nhớ."

### 3. Tools Schema là hợp đồng với AI

Khi các công cụ được định nghĩa qua JSON Schema, AI yêu cầu gọi công cụ theo định dạng có cấu trúc. Chất lượng của `description` ảnh hưởng trực tiếp đến độ chính xác phán đoán của AI.

### 4. Luồng cốt lõi của Function Calling

```
Câu hỏi người dùng -> AI phán đoán -> Trả về tool_calls -> Thực thi hàm -> Thêm kết quả vào messages -> Gọi lại AI -> Phản hồi cuối cùng
```

### 5. Điều kiện kết thúc vòng lặp Agent

Vòng lặp kết thúc khi AI phản hồi bằng văn bản thường mà không có lệnh gọi công cụ nào nữa. Điều này có nghĩa AI tự xác định rằng "không cần thêm công cụ nào nữa."

### 6. Tóm tắt luồng dữ liệu chính

```python
# Tin nhắn người dùng
{"role": "user", "content": "What is the weather in Spain?"}

# Yêu cầu gọi công cụ của AI
{"role": "assistant", "content": "", "tool_calls": [...]}

# Kết quả thực thi công cụ
{"role": "tool", "tool_call_id": "call_xxx", "name": "get_weather", "content": "33 degrees"}

# Phản hồi cuối cùng của AI
{"role": "assistant", "content": "The weather in Spain is 33 degrees Celsius."}
```

---

## Bài tập thực hành

### Bài tập 1: Agent đa công cụ (Cơ bản)

Triển khai cả ba hàm -- `get_weather`, `get_news`, `get_currency` -- và định nghĩa `TOOLS` schema để AI chọn công cụ phù hợp dựa trên câu hỏi.

**Yêu cầu:**
- Mỗi hàm có thể trả về kết quả hardcode
- Đăng ký cả ba hàm trong `FUNCTION_MAP`
- Xác nhận phản hồi đúng cho các câu hỏi như "What is the news in Japan?" và "What is the currency of Brazil?"

### Bài tập 2: Thêm System Prompt (Cơ bản)

Thêm tin nhắn vai trò `system` ở đầu mảng `messages` để thay đổi tính cách AI.

**Ví dụ:**
```python
messages = [
    {"role": "system", "content": "You are a helpful weather assistant. Always respond in Korean."}
]
```

### Bài tập 3: Tăng cường xử lý lỗi (Trung cấp)

Thêm xử lý lỗi cho các tình huống sau:
- Khi AI gọi hàm không tồn tại trong `FUNCTION_MAP`
- Khi ngoại lệ xảy ra trong quá trình thực thi hàm
- Khi `json.loads()` thất bại (đã có xử lý cơ bản, nhưng cải thiện để truyền thông báo lỗi cho AI)

**Gợi ý:** Khi lỗi xảy ra, bạn có thể đặt thông báo lỗi vào `content` của tin nhắn `"role": "tool"` để thông báo cho AI.

### Bài tập 4: Quản lý lịch sử hội thoại (Trung cấp)

Khi cuộc hội thoại dài hơn, lượng token sử dụng tăng lên. Triển khai một trong các chiến lược sau:
- Xóa tin nhắn cũ khi mảng `messages` vượt quá độ dài nhất định (nhưng giữ tin nhắn `system`)
- Ước tính tổng số token và đặt giới hạn

### Bài tập 5: Tích hợp API thực tế (Nâng cao)

Kết nối hàm `get_weather` với API thời tiết thực tế (ví dụ: OpenWeatherMap API) để trả về dữ liệu thời tiết thời gian thực. Xác nhận rằng khi giá trị trả về của hàm thay đổi, phản hồi của AI cũng thay đổi tương ứng.
