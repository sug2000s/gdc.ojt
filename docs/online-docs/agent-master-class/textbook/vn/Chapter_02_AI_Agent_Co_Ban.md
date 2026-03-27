# Chương 2: Cơ bản về AI Agent

---

## Tổng quan chương

Trong chương này, chúng ta sẽ học cách xây dựng AI Agent từ đầu bằng cách sử dụng OpenAI API theo từng bước. Bắt đầu từ một lệnh gọi API đơn giản, sau đó thêm bộ nhớ hội thoại, định nghĩa công cụ bên ngoài (Tool), và cuối cùng triển khai Function Calling để AI có thể thực thi các hàm thực tế, tạo ra một agent hoàn chỉnh.

### Mục tiêu học tập

1. Thiết lập môi trường phát triển Python và xác nhận kết nối OpenAI API
2. Nắm vững cách sử dụng cơ bản của OpenAI Chat Completions API
3. Quản lý lịch sử hội thoại (bộ nhớ) để tạo chatbot duy trì ngữ cảnh
4. Định nghĩa Tool schema để thông báo cho AI về các hàm có sẵn
5. Triển khai Function Calling để AI thực thi hàm đã chọn
6. Truyền kết quả thực thi Tool trở lại AI để tạo phản hồi cuối cùng

### Cấu trúc chương

| Phần | Chủ đề | Từ khóa chính |
|------|--------|---------------|
| 2.0 | Thiết lập dự án | uv, Python 3.13, OpenAI SDK |
| 2.2 | AI Agent đầu tiên | Chat Completions API, Prompt Engineering |
| 2.3 | Thêm bộ nhớ | Lịch sử hội thoại, mảng messages, vòng lặp while |
| 2.4 | Thêm công cụ | Tools schema, JSON Schema, FUNCTION_MAP |
| 2.5 | Thêm gọi hàm | Function Calling, tool_calls, process_ai_response |
| 2.6 | Kết quả thực thi công cụ | Tool Results, gọi đệ quy, hoàn thiện vòng lặp agent |

---

## 2.0 Thiết lập dự án (Setup)

### Chủ đề và mục tiêu

Thiết lập môi trường dự án Python để phát triển AI Agent. Sử dụng trình quản lý gói `uv` để khởi tạo dự án, cài đặt OpenAI Python SDK, và xác nhận API key được tải đúng cách trong môi trường Jupyter Notebook.

### Giải thích khái niệm chính

#### Trình quản lý gói uv

`uv` là trình quản lý gói Python thế hệ mới được viết bằng Rust, cung cấp tốc độ giải quyết phụ thuộc và cài đặt nhanh hơn nhiều so với `pip` hay `poetry` truyền thống. Trong dự án này, chúng ta sử dụng `uv` để tạo môi trường ảo và quản lý các gói.

#### Cấu trúc dự án

Dự án được tổ chức theo cấu trúc sau:

```
my-first-agent/
├── .gitignore          # Danh sách file không theo dõi bởi Git
├── .python-version     # Chỉ định phiên bản Python (3.13)
├── README.md           # Mô tả dự án
├── main.ipynb          # Notebook chính (không gian viết mã)
├── pyproject.toml      # Cấu hình dự án và định nghĩa phụ thuộc
└── uv.lock             # File khóa phụ thuộc
```

### Phân tích mã

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

**Điểm quan trọng:**

- `requires-python = ">=3.13"`: Yêu cầu Python 3.13 trở lên. Nhằm tận dụng các tính năng Python mới nhất.
- `dependencies`: Chỉ định `openai>=1.98.0` làm phụ thuộc runtime. Đây là SDK Python chính thức để giao tiếp với OpenAI API.
- `[dependency-groups] dev`: Bao gồm `ipykernel` làm phụ thuộc phát triển. Cần thiết để chạy mã Python trong Jupyter Notebook.

#### .python-version

```
3.13
```

File này giúp các công cụ như `uv` hay `pyenv` tự động nhận diện phiên bản Python sử dụng trong dự án.

#### main.ipynb - Xác nhận API Key

```python
import os

print(os.getenv("OPENAI_API_KEY"))
```

Mã này xác nhận rằng OpenAI API key đã được thiết lập đúng trong biến môi trường. Vì lý do bảo mật, API key không nên được hard-code trực tiếp trong mã mà phải được quản lý thông qua file `.env` hoặc biến môi trường hệ thống.

#### .gitignore - Cấu hình file bỏ qua

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

**Quan trọng:** File `.env` được bao gồm trong `.gitignore`. Điều này ngăn chặn thông tin nhạy cảm như API key bị tải lên kho Git. Đây là cấu hình rất quan trọng từ góc độ bảo mật.

### Điểm thực hành

1. Khởi tạo dự án bằng lệnh `uv init my-first-agent`
2. Cài đặt OpenAI SDK bằng lệnh `uv add openai`
3. Cài đặt Jupyter kernel bằng lệnh `uv add --dev ipykernel`
4. Tạo file `.env` và lưu API key theo định dạng `OPENAI_API_KEY=sk-...`
5. Chạy Jupyter Notebook và xác nhận API key được hiển thị đúng

---

## 2.2 AI Agent đầu tiên (Your First AI Agent)

### Chủ đề và mục tiêu

Tạo AI Agent ở dạng cơ bản nhất sử dụng OpenAI Chat Completions API. Ở bước này, chúng ta thử nghiệm phương pháp sử dụng Prompt Engineering để thông báo cho AI danh sách các hàm có sẵn dưới dạng văn bản, và hướng dẫn AI chọn hàm phù hợp.

### Giải thích khái niệm chính

#### Chat Completions API

Chat Completions API của OpenAI là giao diện chính để tương tác với mô hình AI hội thoại. Khi gửi tin nhắn, AI tạo phản hồi và trả về. Mỗi tin nhắn bao gồm `role` (vai trò) và `content` (nội dung).

Các loại vai trò (role):
- `system`: Tin nhắn hệ thống chỉ dẫn cách AI hành xử
- `user`: Tin nhắn do người dùng gửi
- `assistant`: Tin nhắn phản hồi do AI tạo
- `tool`: Tin nhắn truyền kết quả thực thi công cụ (học ở các phần sau)

#### Chọn hàm thông qua Prompt Engineering

Ở bước này, chúng ta chưa sử dụng tính năng Function Calling chính thức của OpenAI. Thay vào đó, chúng ta sử dụng phương pháp yêu cầu AI thông qua prompt (chỉ dẫn văn bản): "Có các hàm sau, hãy chọn hàm phù hợp". Đây là dạng nguyên thủy nhất của AI Agent.

### Phân tích mã

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

**Phân tích mã:**

1. **`openai.OpenAI()`**: Tạo client OpenAI. Lúc này tự động đọc biến môi trường `OPENAI_API_KEY`.
2. **`PROMPT`**: Định nghĩa prompt nhiều dòng. Thông báo cho AI danh sách hàm có sẵn (`get_weather`, `get_currency`, `get_news`) và yêu cầu chọn hàm phù hợp với câu hỏi.
3. **`client.chat.completions.create()`**: Thực hiện lệnh gọi API. Tham số `model` chỉ định model sử dụng, tham số `messages` truyền nội dung hội thoại.

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

- `response.choices[0]`: Lấy lựa chọn đầu tiên từ phản hồi API (thông thường chỉ trả về một lựa chọn)
- `.message.content`: Trích xuất nội dung tin nhắn (văn bản) của lựa chọn đó
- AI đã trả về văn bản dạng gọi hàm `get_weather('Greece')` theo chỉ dẫn của prompt

#### Hạn chế của phương pháp này

Phương pháp này hoạt động nhưng có nhiều vấn đề:

- Không đảm bảo phản hồi của AI luôn có định dạng nhất quán (ví dụ: `"get_weather('Greece')"` vs `"I would call get_weather with Greece"`)
- Cần xử lý chuỗi bổ sung để phân tích văn bản trả về và gọi hàm thực tế
- Khó truyền đạt chính xác kiểu tham số hoặc tính bắt buộc cho AI

Để giải quyết những hạn chế này, OpenAI cung cấp tính năng **Function Calling** chính thức, sẽ được học từ phần 2.4 trở đi.

### Điểm thực hành

1. Thử sửa prompt để đặt câu hỏi khác (ví dụ: "What is the currency of Japan?")
2. Xác nhận xem AI có trả về `get_currency('Japan')` không
3. Quan sát phản hồi AI thay đổi như thế nào khi xóa phần "Please say nothing else" trong prompt
4. So sánh sự khác biệt phản hồi khi sử dụng model khác (ví dụ: `gpt-4o`)

---

## 2.3 Thêm bộ nhớ (Adding Memory)

### Chủ đề và mục tiêu

Tạo chatbot AI ghi nhớ nội dung hội thoại trước đó. Sử dụng mảng `messages` để quản lý lịch sử hội thoại, cho phép cuộc hội thoại liên tục giữa người dùng và AI.

### Giải thích khái niệm chính

#### Nguyên lý bộ nhớ hội thoại

LLM (Large Language Model) về cơ bản là hệ thống **không lưu trạng thái (stateless)**. Mỗi lệnh gọi API là độc lập và không tự động ghi nhớ nội dung hội thoại trước đó. Do đó, để duy trì ngữ cảnh hội thoại, **tất cả tin nhắn trước đó phải được gửi cùng mỗi lần gọi API**.

Đây chính là vai trò của mảng `messages`. Mỗi khi người dùng gửi tin nhắn mới:

1. Thêm tin nhắn người dùng vào mảng `messages`
2. Gửi toàn bộ mảng `messages` tới API
3. Thêm phản hồi AI vào mảng `messages`
4. Trong cuộc hội thoại tiếp theo, toàn bộ lịch sử này được gửi cùng

```
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico!"},
    {"role": "user", "content": "What is my name?"},
    # AI có thể trả lời "Your name is Nico." nhờ lịch sử phía trên
]
```

### Phân tích mã

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

**Phân tích mã:**

1. Truyền toàn bộ `messages` cho API để duy trì ngữ cảnh hội thoại
2. Trích xuất phản hồi AI (`message.content`)
3. Thêm phản hồi vào mảng `messages` dưới dạng `{"role": "assistant", "content": message}` -- đây chính là hành động **lưu vào bộ nhớ**
4. Hiển thị phản hồi trên màn hình

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

**Phân tích mã:**

1. `while True`: Vòng lặp vô hạn để tiếp tục hội thoại
2. `input()`: Nhận tin nhắn từ người dùng
3. Nhập `"quit"` hoặc `"q"` để thoát vòng lặp
4. Thêm tin nhắn người dùng vào `messages` dưới dạng `{"role": "user", "content": message}` rồi gọi `call_ai()`

#### Kết quả thực thi (Xác nhận hoạt động bộ nhớ)

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

**Kết quả này cho thấy:**

- AI ghi nhớ tên người dùng ("Nico")
- AI ghi nhớ quê quán người dùng ("Korea")
- AI thậm chí ghi nhớ câu hỏi đầu tiên là gì
- Tất cả điều này khả thi nhờ lịch sử hội thoại trước đó được tích lũy trong mảng `messages`

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

Vì toàn bộ lịch sử được gửi mỗi lần gọi, chi phí API (số token sử dụng) tăng khi hội thoại dài hơn.

### Điểm thực hành

1. Tiếp tục hội thoại dài và kiểm tra AI ghi nhớ tốt đến mức nào
2. In trực tiếp mảng `messages` để xác nhận cấu trúc bên trong
3. Thử reset `messages = []` giữa hội thoại và xác nhận AI quên hội thoại trước đó
4. Thêm tin nhắn vai trò `system` để thay đổi tính cách AI (ví dụ: `{"role": "system", "content": "You are a pirate. Speak like a pirate."}`)

---

## 2.4 Thêm công cụ (Adding Tools)

### Chủ đề và mục tiêu

Sử dụng tính năng **Tools** chính thức của OpenAI để thông báo cho AI về các hàm có sẵn một cách có cấu trúc. Sử dụng JSON Schema để định nghĩa tên hàm, mô tả, tham số, và quan sát `finish_reason` thay đổi thành `tool_calls` khi AI quyết định gọi công cụ.

### Giải thích khái niệm chính

#### So sánh Prompt-based vs Tools-based

Ở phần 2.2, chúng ta đã sử dụng phương pháp thông báo danh sách hàm qua văn bản prompt. Phương pháp đó không ổn định và khó phân tích. Tính năng **Tools** của OpenAI thay thế bằng JSON Schema có cấu trúc:

| Tiêu chí | Dựa trên Prompt (2.2) | Dựa trên Tools (2.4) |
|----------|----------------------|----------------------|
| Cách định nghĩa hàm | Văn bản tự nhiên | JSON Schema |
| Định dạng phản hồi | Văn bản tự do | Đối tượng tool_calls có cấu trúc |
| Định nghĩa tham số | Không rõ ràng | Rõ ràng (kiểu, bắt buộc hay không) |
| Độ khó phân tích | Cao | Thấp (SDK xử lý) |

#### Mẫu FUNCTION_MAP

Khi AI trả về tên hàm, cần tìm và thực thi hàm Python thực tế tương ứng. Để làm điều này, sử dụng **dictionary ánh xạ tên hàm (chuỗi) với đối tượng hàm**:

```python
FUNCTION_MAP = {"get_weather": get_weather}
```

Mẫu này cho phép gọi động hàm `get_weather` thông qua chuỗi `"get_weather"` mà AI trả về.

### Phân tích mã

#### Định nghĩa hàm và ánh xạ

```python
def get_weather(city):
    return "33 degrees celcius."


FUNCTION_MAP = {"get_weather": get_weather}
```

- `get_weather`: Hàm trả về thông tin thời tiết thực tế. Hiện tại trả về giá trị hard-code, nhưng trong thực tế sẽ gọi API thời tiết.
- `FUNCTION_MAP`: Ánh xạ khóa chuỗi với đối tượng hàm. Sau này khi AI trả về `"get_weather"`, sẽ tìm hàm thực tế qua `FUNCTION_MAP["get_weather"]`.

#### Định nghĩa Tools Schema (Phần cốt lõi)

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

**Phân tích chi tiết cấu trúc schema:**

1. **`"type": "function"`**: Chỉ rõ kiểu công cụ là hàm.

2. **Đối tượng `"function"`:**
   - `"name"`: Tên hàm. AI sẽ yêu cầu gọi hàm bằng tên này.
   - `"description"`: Mô tả hàm. Được AI sử dụng để quyết định khi nào dùng hàm này. **Viết description tốt rất quan trọng.**
   - `"parameters"`: Định nghĩa tham số theo định dạng JSON Schema.
     - `"type": "object"`: Cho biết tham số ở dạng đối tượng.
     - `"properties"`: Định nghĩa tên, kiểu, mô tả của từng tham số.
     - `"required"`: Chỉ định danh sách tham số bắt buộc dưới dạng mảng.

3. **`TOOLS` là một mảng.** Có thể định nghĩa nhiều công cụ để cung cấp cho AI.

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

**Thay đổi:** Thêm tham số `tools=TOOLS`. Bây giờ AI có thể gọi hàm phù hợp trong quá trình hội thoại.

#### Phân tích kết quả thực thi

Trường hợp hội thoại thông thường:
```
User: my name is nico
AI: Nice to meet you, Nico! How can I assist you today?
```
- `finish_reason` là `'stop'` -- AI phản hồi bằng văn bản thông thường.
- `tool_calls` là `None`.

Trường hợp câu hỏi cần công cụ:
```
User: what is the weather in Spain
AI: None
```
- `finish_reason` đã đổi thành `'tool_calls'`!
- Mảng `tool_calls` chứa thông tin hàm cần gọi:
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
- `message.content` là `None` -- vì AI đã chọn gọi công cụ thay vì trả lời bằng văn bản.

**Đây là điểm cốt lõi:** AI không trả lời trực tiếp yêu cầu "cho biết thời tiết" mà yêu cầu "hãy gọi hàm get_weather với đối số city='Spain'". Tuy nhiên, vì chưa có mã xử lý yêu cầu này nên hiển thị `AI: None`.

### Điểm thực hành

1. Thử thêm schema hàm `get_news` vào `TOOLS`
2. Xen kẽ gửi câu hỏi cần và không cần công cụ, quan sát sự thay đổi `finish_reason`
3. Thay đổi `description` và thử nghiệm xem hành vi chọn công cụ của AI thay đổi thế nào
4. In chi tiết đối tượng `response` để trực tiếp xác nhận cấu trúc `tool_calls`

---

## 2.5 Thêm gọi hàm (Adding Function Calling)

### Chủ đề và mục tiêu

Triển khai hàm `process_ai_response` để xử lý thực tế các lệnh gọi công cụ mà AI yêu cầu. Viết logic phân nhánh: nếu phản hồi AI chứa `tool_calls` thì thực thi hàm tương ứng, nếu không thì xử lý như phản hồi văn bản thông thường.

### Giải thích khái niệm chính

#### Luồng tổng thể của Function Calling

```
Câu hỏi người dùng → AI phán đoán → Phản hồi tool_calls → Thực thi hàm → Thêm kết quả vào messages → Gọi lại AI → Phản hồi cuối cùng
```

Trong phần này, chúng ta triển khai đến bước "thực thi hàm" và "thêm kết quả vào messages". "Gọi lại AI" sẽ được hoàn thành ở phần tiếp theo (2.6).

#### Cấu trúc phản hồi tool_calls

Khi AI quyết định sử dụng công cụ, đối tượng `message` trong phản hồi chứa thông tin sau:

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

Lưu ý:
- `id`: Mỗi lệnh gọi công cụ có ID duy nhất. Sau này khi truyền kết quả, ID này dùng để khớp với lệnh gọi tương ứng.
- `arguments`: Là **chuỗi JSON**. Không phải dictionary Python nên cần phân tích bằng `json.loads()`.
- `tool_calls` là **mảng**. AI có thể gọi nhiều hàm cùng lúc.

### Phân tích mã

#### Thêm import

```python
import openai, json
```

Thêm module `json`. Cần thiết vì đối số hàm mà AI trả về là chuỗi JSON cần được phân tích.

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

**Bước 1: Phán đoán phân nhánh**
```python
if message.tool_calls > 0:
```
Kiểm tra phản hồi AI có `tool_calls` hay không. Nếu có thì thực thi logic gọi công cụ, nếu không thì thực thi logic phản hồi văn bản thông thường.

> Lưu ý: Điều kiện này sẽ được sửa thành `if message.tool_calls:` ở phần 2.6. Vì trong Python, `None > 0` có thể gây `TypeError`. Kiểm tra truthy/falsy an toàn hơn.

**Bước 2: Thêm tin nhắn assistant vào lịch sử**
```python
messages.append(
    {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [...]
    }
)
```

Ghi lại phản hồi gọi công cụ của AI vào mảng `messages`. **Điều này rất quan trọng.** OpenAI API sẽ xem lịch sử này ở lần gọi tiếp theo và hiểu rằng lệnh gọi công cụ đã được thực hiện. `content` có thể là `None` nên dùng `message.content or ""` để đặt chuỗi rỗng làm giá trị mặc định.

**Bước 3: Lặp qua từng lệnh gọi công cụ và thực thi**
```python
for tool_call in message.tool_calls:
    function_name = tool_call.function.name      # "get_weather"
    arguments = tool_call.function.arguments      # '{"city":"Spain"}'

    print(f"Calling function: {function_name} with {arguments}")

    try:
        arguments = json.loads(arguments)         # {"city": "Spain"}
    except json.JSONDecodeError:
        arguments = {}

    function_to_run = FUNCTION_MAP.get(function_name)  # đối tượng hàm get_weather

    result = function_to_run(**arguments)          # get_weather(city="Spain")
```

- `json.loads()`: Chuyển đổi chuỗi JSON thành dictionary Python
- `try/except`: Coding phòng thủ cho trường hợp phân tích JSON thất bại
- `FUNCTION_MAP.get()`: Tìm hàm thực tế bằng tên chuỗi
- `**arguments`: Giải nén dictionary thành đối số keyword. `{"city": "Spain"}` trở thành `city="Spain"`

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

- `"role": "tool"`: Cho biết tin nhắn này là kết quả thực thi công cụ
- `"tool_call_id"`: ID dùng để khớp kết quả với lệnh gọi công cụ nào. Thiếu ID này sẽ gây lỗi API
- `"content"`: Kết quả thực thi hàm (ở đây là "33 degrees celcius.")

#### Mã bổ trợ để hiểu toán tử ** (giải nén)

Commit bao gồm mã thử nghiệm để hiểu toán tử `**`:

```python
a = '{"city": "Spain"}'

b = json.loads(a)    # b = {"city": "Spain"}

**b                   # Giải nén: city="Spain"

get_weather(city='Spain')
```

Đây là ví dụ học tập cho thấy chuỗi JSON được chuyển đổi thành đối số gọi hàm như thế nào.

#### Hàm call_ai đơn giản hóa

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

Tách logic xử lý phản hồi từ `call_ai` trước đó sang `process_ai_response` để mã gọn gàng hơn.

### Điểm thực hành

1. Thêm `print(messages)` tại mỗi bước trong hàm `process_ai_response` để theo dõi sự thay đổi lịch sử
2. Xác nhận lỗi gì xảy ra khi AI cố gọi hàm không có trong `FUNCTION_MAP` (gợi ý: `None(**arguments)` gây `TypeError`)
3. Thêm hàm và schema `get_currency` để mở rộng hỗ trợ nhiều công cụ

---

## 2.6 Kết quả thực thi công cụ (Tool Results)

### Chủ đề và mục tiêu

Truyền kết quả thực thi công cụ trở lại AI để AI tạo phản hồi cuối cùng tự nhiên dựa trên kết quả đó. Qua đó hoàn thiện **vòng lặp agent**.

### Giải thích khái niệm chính

#### Vòng lặp Agent (Agent Loop)

Một AI Agent hoàn chỉnh hình thành vòng lặp sau:

```
Câu hỏi người dùng
    ↓
AI phán đoán ───→ Nếu phản hồi thông thường → Xuất văn bản cho người dùng (kết thúc vòng lặp)
    ↓
Nếu cần gọi công cụ
    ↓
Thực thi hàm → Thêm kết quả vào messages
    ↓
Gọi lại AI (gọi lại call_ai)
    ↓
AI phán đoán ───→ Nếu phản hồi thông thường → Xuất văn bản cho người dùng (kết thúc vòng lặp)
    ↓
Nếu lại cần gọi công cụ → Thực thi hàm lại... (lặp lại)
```

Cốt lõi của vòng lặp này là **gọi lại AI sau khi thực thi công cụ**. AI nhận kết quả từ công cụ và chuyển đổi thành dạng phù hợp để truyền đạt cho người dùng.

#### Vấn đề của phần trước (2.5)

Ở phần 2.5, chúng ta chỉ triển khai đến bước thực thi công cụ và thêm kết quả vào `messages`. Nhưng kết quả đó chưa được truyền lại cho AI nên AI không thể tạo câu trả lời cuối cùng. Phần này sẽ giải quyết vấn đề đó.

### Phân tích mã

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

**3 thay đổi quan trọng:**

**1. Sửa điều kiện: `message.tool_calls > 0` -> `message.tool_calls`**

```python
# Trước (2.5)
if message.tool_calls > 0:

# Sau (2.6)
if message.tool_calls:
```

Trong Python, `None > 0` có thể gây `TypeError`. Nếu `tool_calls` là `None` thì falsy, nếu có danh sách thì truthy, nên cách này an toàn hơn.

**2. Thêm xuất debug**

```python
print(f"Ran {function_name} with args {arguments} for a result of {result}")
```

Xuất kết quả thực thi hàm ra console để hỗ trợ gỡ lỗi.

**3. Gọi lại AI sau khi thực thi công cụ (Thay đổi quan trọng nhất)**

```python
        # Sau khi tất cả công cụ thực thi xong
        call_ai()
```

Dòng này hoàn thiện vòng lặp agent. Sau khi thêm tất cả kết quả công cụ vào `messages`, gọi lại `call_ai()`. Khi đó AI nhận toàn bộ lịch sử bao gồm kết quả thực thi công cụ và tạo phản hồi cuối cùng.

**Theo dõi luồng gọi:**

```
call_ai()                          # Lần gọi 1
  → AI: trả về tool_calls
  → process_ai_response()
    → Thực thi công cụ, thêm kết quả vào messages
    → call_ai()                    # Lần gọi 2 (đệ quy)
      → AI: trả về phản hồi văn bản thông thường
      → process_ai_response()
        → Nhánh else: xuất văn bản
```

Đây là mẫu **gọi đệ quy tương hỗ**: `call_ai` -> `process_ai_response` -> `call_ai` -> `process_ai_response` -> ...

#### Kết quả thực thi (Agent hoạt động hoàn chỉnh)

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
3. Kết quả được thêm vào `messages` và AI được gọi lại
4. AI chuyển đổi dữ liệu thô "33 degrees celcius." thành câu tự nhiên "The weather in Spain is 33 degrees Celsius. If you need more specific weather details..."

#### Xác nhận mảng messages cuối cùng

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

Mảng này thể hiện toàn bộ quá trình hoạt động của agent:
1. Hội thoại thông thường (user -> assistant)
2. Yêu cầu gọi công cụ (assistant with tool_calls)
3. Kết quả thực thi công cụ (tool)
4. Phản hồi cuối cùng dựa trên kết quả (assistant)

### Điểm thực hành

1. Thêm hàm `get_news` và `get_currency`, thử câu hỏi phức hợp ("What is the weather and news in Korea?") và xác nhận nhiều công cụ được gọi đồng thời
2. Suy nghĩ tại sao gọi đệ quy không rơi vào vòng lặp vô hạn (gợi ý: khi AI nhận kết quả công cụ, nó phản hồi bằng văn bản thông thường nên rơi vào nhánh `else`)
3. In mảng `messages` để trực quan xác nhận tin nhắn của từng vai trò (user, assistant, tool) tích lũy như thế nào
4. Cố tình thay đổi kết quả thực thi công cụ thành thông báo lỗi và kiểm tra AI phản ứng thế nào

---

## Tổng kết chương

### 1. Bản chất của AI Agent

AI Agent là sự kết hợp của **LLM + Công cụ + Vòng lặp**. LLM phán đoán, công cụ thực thi, và vòng lặp kết nối chúng lặp đi lặp lại.

### 2. Mảng messages chính là bộ nhớ

LLM không lưu trạng thái. Tích lũy hội thoại trước đó trong mảng `messages` và gửi mỗi lần chính là thực chất của "bộ nhớ".

### 3. Tools Schema là hợp đồng với AI

Khi định nghĩa công cụ bằng JSON Schema, AI yêu cầu gọi công cụ ở dạng có cấu trúc. Chất lượng của `description` ảnh hưởng trực tiếp đến độ chính xác phán đoán của AI.

### 4. Luồng cốt lõi của Function Calling

```
Câu hỏi người dùng → AI phán đoán → Trả về tool_calls → Thực thi hàm → Thêm kết quả vào messages → Gọi lại AI → Phản hồi cuối cùng
```

### 5. Điều kiện kết thúc vòng lặp Agent

Khi AI phản hồi bằng văn bản thông thường mà không cần gọi công cụ nữa, vòng lặp kết thúc. Đây là AI tự phán đoán rằng "không cần công cụ nữa".

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

Triển khai đầy đủ ba hàm `get_weather`, `get_news`, `get_currency` và định nghĩa `TOOLS` schema để AI chọn công cụ phù hợp theo câu hỏi.

**Yêu cầu:**
- Mỗi hàm có thể trả về kết quả hard-code
- Đăng ký cả ba hàm trong `FUNCTION_MAP`
- Xác nhận phản hồi đúng với các câu hỏi như "What is the news in Japan?", "What is the currency of Brazil?"

### Bài tập 2: Thêm System Prompt (Cơ bản)

Thêm tin nhắn vai trò `system` vào đầu mảng `messages` để thay đổi tính cách AI.

**Ví dụ:**
```python
messages = [
    {"role": "system", "content": "You are a helpful weather assistant. Always respond in Korean."}
]
```

### Bài tập 3: Tăng cường xử lý lỗi (Trung cấp)

Thêm xử lý lỗi cho các tình huống sau:
- AI gọi hàm không tồn tại trong `FUNCTION_MAP`
- Xảy ra ngoại lệ trong quá trình thực thi hàm
- `json.loads()` thất bại (đã có xử lý cơ bản nhưng cải thiện để truyền thông báo lỗi cho AI)

**Gợi ý:** Khi xảy ra lỗi, có thể đặt thông báo lỗi vào `content` của tin nhắn `"role": "tool"` để thông báo cho AI.

### Bài tập 4: Quản lý lịch sử hội thoại (Trung cấp)

Khi hội thoại dài hơn, số token sử dụng tăng lên. Triển khai một trong các chiến lược sau:
- Khi độ dài mảng `messages` vượt quá giới hạn, xóa tin nhắn cũ (nhưng giữ lại tin nhắn `system`)
- Ước tính tổng số token và đặt giới hạn

### Bài tập 5: Tích hợp API thực tế (Nâng cao)

Tích hợp hàm `get_weather` với API thời tiết thực tế (ví dụ: OpenWeatherMap API) để trả về dữ liệu thời tiết thời gian thực. Xác nhận phản hồi AI thay đổi phù hợp khi giá trị trả về của hàm thay đổi.
