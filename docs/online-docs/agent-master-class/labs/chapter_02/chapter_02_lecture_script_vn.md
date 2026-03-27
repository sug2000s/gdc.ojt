# Chapter 02: Cơ bản về AI Agent — Kịch bản bài giảng

---

## Mở đầu (2 phút)

> Được rồi, hôm nay chúng ta sẽ **tự tay xây dựng AI Agent từ đầu**.
>
> Khi dùng ChatGPT, bạn nói "tìm trên mạng đi", "đọc file này đi" là nó tự làm, đúng không?
> Đó chính là Agent. AI **tự phán đoán và chọn công cụ để sử dụng**.
>
> Hôm nay chúng ta sẽ xây dựng đúng thứ đó.
> Nhưng thay vì làm một lần, chúng ta sẽ chia thành **5 giai đoạn**, thêm từng phần một.
>
> ```
> 2.2 Chọn hàm qua prompt       (Cách nguyên thủy)
> 2.3 Thêm bộ nhớ                (Lịch sử hội thoại)
> 2.4 Định nghĩa công cụ         (JSON Schema)
> 2.5 Thực thi hàm               (Function Calling)
> 2.6 Vòng lặp agent hoàn chỉnh  (Phản hồi kết quả)
> ```
>
> Kết thúc buổi học, bạn sẽ có một AI Agent hoạt động do chính tay mình tạo ra. Bắt đầu thôi.

---

## 2.0 Setup & Environment (3 phút)

### Trước khi chạy cell

> Đầu tiên, thiết lập môi trường.
> File `.env` chứa API key, và chúng ta load bằng `load_dotenv`.
>
> Chúng ta sẽ dùng model GPT của Azure OpenAI.
> Ba thứ quan trọng:
> - `OPENAI_API_KEY` — khóa xác thực
> - `OPENAI_BASE_URL` — địa chỉ endpoint Azure
> - `OPENAI_MODEL_NAME` — tên model sử dụng

### Sau khi chạy cell

> Nếu chỉ hiện 8 ký tự đầu của key, vậy là đúng.
> Không được hiện toàn bộ — API key tuyệt đối không được lộ ra ngoài.
>
> Bị lỗi? Kiểm tra đường dẫn file `.env`. Nó phải nằm ở thư mục cha với `../`.
>
> Xong hết rồi? Tiếp tục nào.

---

## 2.2 AI Agent đầu tiên (10 phút)

### Khái niệm

> Câu hỏi đầu tiên.
> **Làm sao AI biết "hỏi thời tiết thì phải dùng get_weather"?**
>
> Có một cách rất nguyên thủy.
> Đơn giản là **liệt kê danh sách hàm trong prompt và bảo "chọn đi"**.
>
> Code này làm chính xác điều đó.

### Giải thích code (trước khi chạy)

> Cùng đọc prompt.
>
> ```
> "Hệ thống có các hàm sau: get_weather, get_currency, get_news"
> "Tham số là tên quốc gia"
> "Chỉ trả lời tên hàm. Không nói gì thêm"
> "Câu hỏi: Thời tiết Hy Lạp thế nào?"
> ```
>
> Dòng quan trọng nhất: **"Please say nothing else, just the name of the function"**
> Không có dòng này, AI sẽ thêm giải thích.
>
> Chạy thử nào.

### Sau khi chạy cell

> Xem kết quả. Toàn bộ đối tượng `response` hiện ra.
> Thứ duy nhất chúng ta cần là: `choices[0].message.content`
>
> Chạy cell tiếp...
>
> `'get_weather("Greece")'` — nó trả lời đúng dạng gọi hàm.
>
> **Điểm quan trọng ở đây:**
> Đây không phải lệnh gọi hàm thật. Nó chỉ là một **chuỗi ký tự**.
> AI chỉ trả lời bằng text rằng "tôi nghĩ nên gọi hàm này."
>
> Muốn thực thi thật? Phải dùng `eval()` hoặc tự parse.
> Không ổn định. Format có thể thay đổi, hoặc code nguy hiểm có thể xuất hiện.
>
> Đó là lý do OpenAI tạo tính năng **Tool Calling** chính thức.
> Chúng ta sẽ làm ở phần 2.4. Bây giờ làm bài tập trước.

### Bài tập 2.2 (5 phút)

> **Bài tập 1**: Đổi câu hỏi thành `"What is the currency of Japan?"`.
> Kiểm tra xem có ra `get_currency('Japan')` không.
>
> **Bài tập 2**: Xóa dòng `"Please say nothing else"` khỏi prompt.
> AI trả lời khác thế nào? Có thể sẽ ra kiểu:
> `"I would call get_currency('Japan') because..."`
> Một dòng trong prompt thay đổi hoàn toàn định dạng output.
>
> **Bài tập 3**: Nếu có thời gian, thử đổi tên model và so sánh kết quả.
>
> (Sau 5 phút) Xong rồi thì tiếp nhé.

---

## 2.3 Thêm bộ nhớ (10 phút)

### Khái niệm

> Bây giờ là khái niệm quan trọng.
>
> **LLM không có trí nhớ.**
>
> Khi dùng ChatGPT, có vẻ như nó nhớ cuộc trò chuyện trước,
> nhưng thực tế **toàn bộ lịch sử hội thoại được gửi lại mỗi lần**.
>
> Mảng `messages` này chính là "bộ nhớ" của AI.
> Chúng ta phải tự quản lý nó.

### Giải thích code

> Cùng xem code.
>
> `messages = []` — bắt đầu bằng mảng rỗng.
>
> Hàm `call_ai()`:
> 1. Gửi toàn bộ `messages` đến API
> 2. Nhận phản hồi
> 3. Thêm vào `messages` dưới dạng `{"role": "assistant", ...}`
>
> Vòng lặp `while` bên dưới:
> 1. Nhận input từ người dùng
> 2. Thêm vào `messages` dưới dạng `{"role": "user", ...}`
> 3. Gọi `call_ai()`
>
> Chỉ có vậy thôi. Vì **gửi toàn bộ messages mỗi lần**, AI hoạt động như thể "nhớ" cuộc trò chuyện trước.

### Chạy cell

> Chạy thử nào. Cho AI biết tên bạn, rồi hỏi "Tên tôi là gì?"
> Nó nhớ, đúng không?
>
> Bây giờ nhấn `q` để thoát, rồi kiểm tra nội dung `messages` ở cell tiếp.
>
> `[user]`, `[assistant]`, `[user]`, `[assistant]`... xếp theo thứ tự, đúng không?
> Đây là bộ nhớ của AI. Mảng này chính là thứ đi vào **context window**.

### Bài tập 2.3 (5 phút)

> **Bài tập 1**: Trò chuyện 10 lần trở lên. Nó có nhớ hết không?
>
> **Bài tập 2**: Chạy `messages = []` ở giữa cuộc trò chuyện.
> Rồi hỏi lại "Tên tôi là gì?" — nó sẽ không biết. Vì bộ nhớ đã bị xóa.
>
> **Bài tập 3** (thú vị): Thử cái này:
> ```python
> messages = [{"role": "system", "content": "You are a pirate. Speak like a pirate."}]
> ```
> Thêm system message sẽ thay đổi tính cách AI.
> Nó sẽ trả lời kiểu "Arrr! What be ye name, matey?"
>
> Quy ước là đặt system message ở `messages[0]`. Đây là cách thiết lập vai trò, tính cách và quy tắc cho AI.

---

## 2.4 Thêm công cụ (15 phút)

### Khái niệm

> Nhớ ở 2.2, chúng ta cho AI chọn hàm qua prompt?
> Vấn đề là gì?
>
> 1. Kết quả trả về dạng chuỗi, nên **parse không ổn định**
> 2. AI có thể trả lời sai format
> 3. AI không biết chính xác cần tham số gì
>
> OpenAI giải quyết bằng tính năng **Tools**.
> Bạn định nghĩa tên hàm, mô tả, tham số bằng **JSON Schema** rồi truyền vào API.
> AI sẽ trả về dạng có cấu trúc: "gọi hàm này với tham số này."

### Giải thích code — Định nghĩa TOOLS

> Cùng xem mảng `TOOLS`.
>
> ```python
> {
>     "type": "function",
>     "function": {
>         "name": "get_weather",          # tên hàm
>         "description": "...",            # mô tả AI sẽ đọc
>         "parameters": {                  # JSON Schema
>             "type": "object",
>             "properties": {
>                 "city": {
>                     "type": "string",
>                     "description": "..."
>                 }
>             },
>             "required": ["city"]
>         }
>     }
> }
> ```
>
> **description là yếu tố then chốt.** AI đọc mô tả này để quyết định "có nên dùng hàm này không".
> Nếu description sai, AI sẽ không dùng hoặc dùng sai công cụ.
>
> `FUNCTION_MAP` là dictionary liên kết tên với hàm Python thực tế.
> Sau này khi AI nói "gọi get_weather", chúng ta tìm ở đây rồi thực thi.

### Giải thích code — call_ai()

> Xem `call_ai()`, có một điểm khác so với trước:
>
> ```python
> tools=TOOLS  # dòng này được thêm vào
> ```
>
> Với dòng này, AI trả về theo hai chế độ:
> - **Trả lời bình thường**: `finish_reason = 'stop'`, `tool_calls = None`
> - **Cần công cụ**: `finish_reason = 'tool_calls'`, `tool_calls = [...]`

### Chạy cell — Test 1

> Chạy Test 1. `"My name is Nico"` — chỉ là hội thoại thường.
>
> Xem kết quả:
> - `finish_reason: stop` — không cần công cụ
> - `tool_calls: None` — không có công cụ nào cần gọi
> - `content: "Nice to meet you, Nico!"` — phản hồi text bình thường
>
> AI đã phán đoán: "Câu này không cần dùng công cụ cũng trả lời được."

### Chạy cell — Test 2

> Chạy Test 2. `"What is the weather in Spain?"` — câu hỏi thời tiết.
>
> Kết quả:
> - `finish_reason: tool_calls` — **cần công cụ!**
> - `tool_calls: [...]` — yêu cầu gọi `get_weather` với `Spain`
> - `content: None` — không có phản hồi text
>
> **Đây là điểm khác biệt với 2.2.**
> Ở 2.2, chúng ta nhận **chuỗi ký tự** `"get_weather('Spain')"`,
> nhưng giờ nó trả về **đối tượng có cấu trúc**. Không cần parse.
>
> Nhưng! Vẫn còn thiếu một thứ.
> AI nói "gọi get_weather", nhưng **thực tế chưa thực thi**.
> Phần tiếp theo sẽ làm việc đó.

### Bài tập 2.4 (5 phút)

> **Bài tập 1**: Thêm công cụ `get_news` vào `TOOLS`.
> Format giống `get_weather` — chỉ đổi name và description.
>
> **Bài tập 2**: Luân phiên hỏi câu thường như "Hello" và "Cho tôi xem tin tức",
> rồi quan sát `finish_reason` thay đổi.
>
> **Bài tập 3**: Cố tình đổi `description` thành sai.
> Ví dụ: `"A function to order pizza"` — AI có dùng hàm này cho câu hỏi thời tiết không?

---

## 2.5 Thêm Function Calling (15 phút)

### Khái niệm

> Ở 2.4, AI yêu cầu "gọi get_weather".
> Nhưng thực tế chưa bao giờ thực thi hàm.
>
> Trong phần này, chúng ta tạo hàm **`process_ai_response`**.
> Nhận phản hồi từ AI và:
> 1. Nếu có `tool_calls` → tìm và thực thi hàm
> 2. Nếu không → chỉ in text

### Giải thích code — process_ai_response

> Hàm này là trọng tâm của phần này. Cùng xem kỹ.
>
> **Trường hợp 1: khi có tool_calls** (AI muốn dùng công cụ)
>
> ```
> Bước 1: Thêm message tool_calls của AI vào messages
> Bước 2: Duyệt qua từng tool_call
>          - Trích xuất tên hàm (function_name)
>          - Trích xuất tham số (arguments) → parse JSON
>          - Tìm hàm trong FUNCTION_MAP và thực thi
>          - Thêm kết quả vào messages (role: "tool")
> ```
>
> **Trường hợp 2: khi không có tool_calls** (hội thoại thường)
>
> ```
> Chỉ thêm vào messages và in ra
> ```
>
> Chi tiết quan trọng: **kết quả được thêm với `role: "tool"`.**
> Kèm theo `tool_call_id` để khớp kết quả với yêu cầu tương ứng.

### Chạy cell

> Chạy thử. `"What is the weather in Spain?"`
>
> Output:
> ```
> Calling function: get_weather with {"city": "Spain"}
> ```
>
> Hàm đã được thực thi! Giờ kiểm tra messages.
>
> ```
> messages[0]: user — "What is the weather in Spain?"
> messages[1]: assistant — (has tool_calls)
> messages[2]: tool — "33 degrees celcius."
> ```
>
> Ba message đã được xếp chồng.
>
> **Nhưng có vấn đề!**
> Người dùng chưa nhận được câu trả lời. AI chưa **nói cho người dùng biết "33 độ"**.
>
> Tại sao? Vì sau khi thực thi hàm, chúng ta **chưa gọi AI lại lần nữa**.
> Kết quả công cụ đã được thêm vào messages, nhưng AI chưa có cơ hội xem và tạo phản hồi.
>
> Giải quyết vấn đề này là nội dung phần 2.6.

---

## 2.6 Kết quả công cụ — Vòng lặp Agent hoàn chỉnh (15 phút)

### Khái niệm

> Cuối cùng, mảnh ghép cuối.
>
> Tổng kết những gì đã xây dựng:
>
> ```
> Câu hỏi người dùng → AI phán đoán → tool_calls → thực thi hàm → thêm kết quả vào messages
> ```
>
> Chỉ cần **thêm một dòng** là có Agent hoàn chỉnh:
>
> ```
> Câu hỏi người dùng → AI phán đoán → tool_calls → thực thi hàm → thêm kết quả vào messages
>     → ★ Gọi AI lại lần nữa → phản hồi text cuối cùng
> ```
>
> Đây chính là **Agent Loop**.

### Giải thích code — điểm thay đổi

> Cùng xem code. Gần như giống hệt 2.5, chỉ **thêm đúng một dòng**.
>
> Bên trong `process_ai_response`, sau khi thêm kết quả tool vào messages:
>
> ```python
> # KEY: Call AI again so it can use the tool result!
> call_ai()
> ```
>
> Dòng đó là tất cả.
>
> `call_ai()` → `process_ai_response()` → nếu có thêm `tool_calls` → lại `call_ai()`...
> **Chạy đệ quy.** Khi không cần công cụ nữa, nó trả lời bằng text và dừng.
>
> Đây là bản chất của Agent Loop:
> **"AI ơi, tôi sẽ cho bạn chạy cho đến khi dùng hết các công cụ cần thiết."**

### Chạy cell

> Chạy thử. Hãy trò chuyện.
>
> `"What is the weather in Seoul?"`
>
> ```
> Calling function: get_weather with {"city": "Seoul"}
> AI: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> **Cuối cùng người dùng đã nhận được câu trả lời!**
>
> Bên trong đã xảy ra những gì:
> 1. AI yêu cầu `get_weather("Seoul")` qua `tool_calls`
> 2. Chúng ta thực thi hàm và nhận kết quả `"33 degrees celcius."`
> 3. Thêm kết quả vào messages
> 4. **Gọi AI lại lần nữa** → AI xem kết quả và tạo phản hồi bằng ngôn ngữ tự nhiên
>
> Bây giờ nhấn `q` để thoát, rồi kiểm tra messages.
>
> ```
> [0] user: What is the weather in Seoul?
> [1] assistant: (has tool_calls)
> [2] tool: 33 degrees celcius.
> [3] assistant: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> Bốn message đã được xếp chồng. Đây là một chu kỳ hoàn chỉnh của Agent Loop.

### Bài tập 2.6 — Bài tập cuối (15 phút)

> Đây là bài tập cuối cùng. Có đủ thời gian nên hãy thử sức.
>
> **Bài tập 1**: Thêm hàm `get_news` và `get_currency` + tool schema.
> Dùng hàm mock:
> ```python
> def get_news(country):
>     return f"Big news in {country} today!"
>
> def get_currency(country):
>     return f"The currency rate for {country} is 1,350 KRW."
> ```
> Đừng quên thêm vào cả `FUNCTION_MAP` và `TOOLS`.
>
> **Bài tập 2**: Thử hội thoại nhiều lượt.
> `"Thời tiết Hàn Quốc thế nào?"` → `"Tin tức ở đó thì sao?"` → AI có nhớ "ở đó" là Hàn Quốc không?
>
> **Bài tập 3**: Thêm `print(messages)` vào trong `process_ai_response`
> để theo dõi messages tích lũy thế nào ở mỗi bước.
>
> **Bài tập 4** (thử thách): Nếu AI gọi hàm không có trong `FUNCTION_MAP` thì sao?
> `function_to_run` sẽ là `None` và chương trình sẽ crash.
> Thêm xử lý lỗi:
> ```python
> if function_to_run is None:
>     result = f"Error: function '{function_name}' not found."
> else:
>     result = function_to_run(**arguments)
> ```

---

## Kết thúc (3 phút)

> Tổng kết những gì đã làm hôm nay.
>
> Agent chúng ta xây dựng hoạt động theo luồng sau:
>
> ```
> ┌──────────────────────────────────────────────────────────┐
> │  Câu hỏi người dùng                                     │
> │       ↓                                                  │
> │  AI phán đoán: cần công cụ? ──Không──→ Phản hồi text → Xong │
> │       │ Có                                               │
> │       ↓                                                  │
> │  Trả về tool_calls                                       │
> │       ↓                                                  │
> │  Thực thi hàm (process_ai_response)                      │
> │       ↓                                                  │
> │  Thêm kết quả vào messages                               │
> │       ↓                                                  │
> │  Gọi AI lại ──→ (lặp lại nếu cần thêm công cụ)          │
> └──────────────────────────────────────────────────────────┘
> ```
>
> Đây là cấu trúc cơ bản của AI Agent.
> Dù là LangChain, CrewAI, hay AutoGen — bên trong đều chạy vòng lặp này.
> Thứ các bạn xây dựng hôm nay chính là cốt lõi đó.
>
> Chương tiếp theo, chúng ta sẽ phát triển thêm —
> kết hợp nhiều công cụ để xử lý các tác vụ phức tạp hơn.
>
> Cảm ơn các bạn. Làm tốt lắm.
