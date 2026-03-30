# Chapter 14: Xây dựng Chatbot LangGraph — Kịch bản bài giảng

---

## Mở đầu (2 phút)

> Được rồi, chương trước chúng ta đã học các building block cơ bản của LangGraph.
>
> Hôm nay chúng ta sẽ sử dụng chúng để xây dựng một **chatbot thực tế** từng bước.
>
> Bắt đầu từ chatbot đơn giản, rồi thêm tính năng từng cái một:
>
> ```
> 14.0 Chatbot cơ bản      (MessagesState + init_chat_model)
> 14.1 Tool Node             (Tool Calling + ReAct Pattern)
> 14.2 Bộ nhớ                (SQLite Checkpointer)
> 14.3 Human-in-the-loop     (interrupt + Command resume)
> 14.4 Du hành thời gian     (State Fork)
> ```
>
> Mỗi bước xây dựng trên bước trước.
> Cuối cùng, chúng ta sẽ có chatbot có bộ nhớ, cho phép con người can thiệp, và có thể quay lại quá khứ.
>
> Bắt đầu thôi.

---

## 14.0 Setup & Environment (2 phút)

### Trước khi chạy cell

> Đầu tiên, kiểm tra môi trường.
> Chúng ta load API key từ file `.env` và kiểm tra phiên bản `langgraph` và `langchain`.
>
> Cùng môi trường với Chapter 13. Không cần package bổ sung.

### Sau khi chạy cell

> Nếu API key và phiên bản hiển thị đúng, chúng ta đã sẵn sàng.
> Nếu bị lỗi, kiểm tra `uv sync` hoặc `pip install`.

---

## 14.0 Chatbot cơ bản — MessagesState (8 phút)

### Khái niệm

> Ở Chapter 13, chúng ta định nghĩa state thủ công bằng `TypedDict`.
> Cho chatbot, có thứ tiện hơn: **`MessagesState`**.
>
> `MessagesState` là gì?
> - Có sẵn trường `messages: list`
> - **Reducer được tích hợp sẵn**, nên tin nhắn tự động tích lũy
> - Nhớ `Annotated[list, operator.add]` ở Chapter 13 không? Đã bao gồm rồi
>
> Khởi tạo LLM cũng đơn giản:
> ```python
> llm = init_chat_model("openai:gpt-4o-mini")
> ```
> `init_chat_model` khởi tạo bất kỳ LLM nào trong một dòng với format provider:tên_model.
>
> Cấu trúc đồ thị đơn giản nhất có thể:
> ```
> START -> chatbot -> END
> ```

### Giải thích code (trước khi chạy)

> Cùng xem code.
>
> ```python
> class State(MessagesState):
>     pass
> ```
>
> Chỉ cần kế thừa `MessagesState`. Trường `messages` tự động được bao gồm.
>
> ```python
> def chatbot(state: State):
>     response = llm.invoke(state["messages"])
>     return {"messages": [response]}
> ```
>
> Node `chatbot`:
> 1. Lấy `messages` từ state và truyền cho LLM
> 2. Thêm response của LLM vào `messages`
>
> Nhờ reducer, tin nhắn cũ không biến mất — response được nối thêm vào.
>
> Edge: `START -> chatbot -> END`. Đồ thị tuyến tính đơn giản nhất.
> Chạy thử nào.

### Sau khi chạy

> Xem output:
> ```
> human: how are you?
> ai: I'm just a computer program, so I don't have feelings...
> ```
>
> Tin nhắn `human` và response `ai` được xếp chồng trong list `messages`.
>
> Đây là chatbot cơ bản nhất.
> Nhưng có vấn đề — **lịch sử hội thoại không được lưu.**
> Khi `invoke()` kết thúc, state biến mất. Chúng ta sẽ giải quyết ở 14.2.

### Bài tập 14.0 (3 phút)

> **Bài tập 1**: Nếu dùng `TypedDict` với `messages: list` thay vì `MessagesState` thì sao? Kiểm tra điều gì xảy ra khi không có reducer.
>
> **Bài tập 2**: Đổi provider trong `init_chat_model`. Thử `"anthropic:claude-sonnet-4-20250514"` chẳng hạn.
>
> **Bài tập 3**: Thêm trường `system_prompt: str` vào `State` và triển khai system prompt động.

---

## 14.1 Tool Node — Tool Calling + ReAct Pattern (10 phút)

### Khái niệm

> Chatbot cơ bản chỉ trả lời từ kiến thức LLM.
> Còn thời tiết thời gian thực hay truy vấn database — khi cần **công cụ bên ngoài** thì sao?
>
> Đó là **ReAct pattern**:
> 1. LLM quyết định "Tôi cần gọi công cụ này" và tạo tool_calls
> 2. Công cụ thực thi và trả kết quả
> 3. LLM xem kết quả và tạo response cuối cùng
>
> Cấu trúc đồ thị:
> ```
> START -> chatbot -> [có tool_calls?] -> tools -> chatbot -> ... -> END
> ```
>
> Ba thành phần chính:
> - `@tool` — chuyển hàm Python thành công cụ LLM có thể gọi
> - `ToolNode` — node chịu trách nhiệm thực thi công cụ
> - `tools_condition` — chuyển đến `tools` nếu có tool_calls, ngược lại đến `END`

### Giải thích code (trước khi chạy)

> Cùng đi qua code từng bước.
>
> **Bước 1: Định nghĩa công cụ**
> ```python
> @tool
> def get_weather(city: str):
>     """Gets weather in city"""
>     return f"The weather in {city} is sunny."
> ```
> Decorator `@tool` biến hàm này thành công cụ LLM có thể gọi.
> **Docstring rất quan trọng** — LLM đọc nó để quyết định khi nào dùng công cụ này.
>
> **Bước 2: Gắn công cụ vào LLM**
> ```python
> llm_with_tools = llm.bind_tools(tools=[get_weather])
> ```
> Báo cho LLM "bạn có quyền truy cập công cụ này".
>
> **Bước 3: Xây dựng đồ thị**
> ```python
> graph_builder.add_conditional_edges("chatbot", tools_condition)
> graph_builder.add_edge("tools", "chatbot")
> ```
>
> `tools_condition` là hàm tích hợp sẵn của LangGraph.
> Nếu response LLM có `tool_calls`, trả về `"tools"`; ngược lại `"__end__"`.
>
> Kết quả công cụ quay lại `chatbot` vì LLM cần xem kết quả để soạn câu trả lời cuối.
>
> Chạy thử nào.

### Sau khi chạy — câu hỏi cần công cụ

> ```
> human: what is the weather in machupichu
> ai:                          <- chỉ có tool_calls, content trống
> tool: The weather in Machupichu is sunny.
> ai: The weather in Machupicchu is sunny.
> ```
>
> Thấy luồng xử lý không?
> 1. User hỏi về thời tiết
> 2. LLM quyết định cần `get_weather` và tạo tool_calls
> 3. `tools_condition` phát hiện tool_calls và chuyển đến node `tools`
> 4. `ToolNode` thực thi `get_weather("Machupichu")`
> 5. Kết quả quay lại `chatbot`, LLM tạo response cuối cùng
>
> Đây là **vòng lặp ReAct**. Reasoning (suy luận) + Acting (hành động).

### Sau khi chạy — câu hỏi không cần công cụ

> ```
> human: hello, how are you?
> ai: Hello! I'm just a computer program...
> ```
>
> Lần này không có message `tool`.
> LLM quyết định có thể trả lời mà không cần công cụ.
> `tools_condition` chuyển thẳng đến `END`.
>
> **Cùng đồ thị, đường đi khác nhau tùy theo input.** Cùng nguyên lý với conditional edges ở Chapter 13.

### Bài tập 14.1 (3 phút)

> **Bài tập 1**: Thêm công cụ khác như `get_time`. LLM có chọn đúng công cụ cho từng tình huống không?
>
> **Bài tập 2**: Thay đổi docstring của công cụ. Nó ảnh hưởng thế nào đến hành vi chọn công cụ của LLM?
>
> **Bài tập 3**: Thay `tools_condition` bằng hàm điều kiện tùy chỉnh.

---

## 14.2 Bộ nhớ — SQLite Checkpointer (10 phút)

### Khái niệm

> Nhớ vấn đề ở 14.0 không? State biến mất sau khi `invoke()` kết thúc.
>
> Để chatbot nhớ các cuộc hội thoại trước, cần **checkpointer**.
>
> Checkpointer làm gì:
> - Lưu state vào database sau mỗi lần thực thi node
> - Cùng session (thread_id), load state trước đó và tiếp tục
>
> ```python
> conn = sqlite3.connect("memory.db")
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> **`thread_id`** là chìa khóa:
> - Cùng thread_id nghĩa là hội thoại tiếp tục (bộ nhớ được giữ)
> - Khác thread_id nghĩa là cuộc hội thoại hoàn toàn mới
>
> Hãy nghĩ thread_id như "phòng chat" trong ứng dụng nhắn tin thực tế.

### Giải thích code (trước khi chạy)

> Code gần như giống hệt chatbot công cụ ở 14.1.
> Chỉ thay đổi hai dòng:
>
> ```python
> conn = sqlite3.connect("memory.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> Chỉ cần truyền `checkpointer` cho `compile()` là kích hoạt bộ nhớ.
>
> Và khi gọi `invoke()`:
> ```python
> config = {"configurable": {"thread_id": "1"}}
> result = graph.invoke({"messages": [...]}, config=config)
> ```
>
> `config` bao gồm `thread_id` để xác định session.
> Chạy thử nào.

### Sau khi chạy — thread_id="1", hội thoại đầu tiên

> ```
> Hello Alice! How can I assist you today?
> ```
>
> Chúng ta đã cho biết tên. Hãy hỏi về nó ở cell tiếp với cùng thread_id.

### Sau khi chạy — thread_id="1", tiếp tục

> ```
> Your name is Alice.
> ```
>
> **Nó nhớ!** Cùng thread_id, nên hội thoại trước vẫn trong bộ nhớ.

### Sau khi chạy — thread_id="2", hội thoại mới

> ```
> I'm sorry, but I don't have access to personal information about you...
> ```
>
> Khác thread_id, nên đây là **cuộc hội thoại hoàn toàn mới**. Không biết tên.
>
> Đây là cốt lõi của checkpointer:
> - Cùng thread nghĩa là bộ nhớ được giữ
> - Khác thread nghĩa là cách ly hoàn toàn

### Sau khi chạy — get_state_history

> ```
> next: (), messages: 4
> next: ('chatbot',), messages: 3
> next: ('__start__',), messages: 2
> ...
> ```
>
> `get_state_history()` hiển thị tất cả snapshot cho thread đó.
> Trạng thái trước và sau mỗi lần thực thi node đều được ghi lại.
>
> `next` trống nghĩa là đồ thị đã hoàn thành tại thời điểm đó,
> `('chatbot',)` nghĩa là ngay trước khi node chatbot thực thi.
>
> Lịch sử này trở nên quan trọng ở 14.4 Du hành thời gian.

### Bài tập 14.2 (3 phút)

> **Bài tập 1**: Trò chuyện nhiều lần với cùng thread_id. Bộ nhớ có được giữ không?
>
> **Bài tập 2**: Thử chạy streaming với `stream_mode="updates"`.
>
> **Bài tập 3**: Phân tích từng snapshot từ `get_state_history()` để theo dõi thay đổi state.

---

## 14.3 Human-in-the-loop (10 phút)

### Khái niệm

> Chatbot chúng ta xây cho đến giờ là **hoàn toàn tự động**.
> User yêu cầu, LLM xử lý, xong.
>
> Nhưng trong thực tế, **con người thường cần can thiệp**:
> - Review code do AI tạo
> - Phê duyệt quyết định quan trọng
> - Cung cấp phản hồi về output của AI
>
> Giải pháp của LangGraph: **`interrupt()` và `Command(resume=...)`**
>
> Luồng xử lý:
> 1. Trong quá trình chạy đồ thị, `interrupt()` được gọi và thực thi tạm dừng
> 2. User cung cấp phản hồi
> 3. `Command(resume=phản_hồi)` truyền phản hồi và tiếp tục thực thi
>
> Vì checkpointer lưu trạng thái tạm dừng, chúng ta có thể tiếp tục ngay chỗ dừng lại.

### Giải thích code (trước khi chạy)

> Ví dụ này là **chatbot viết thơ**. LLM viết thơ và nhận phản hồi từ con người.
>
> Chìa khóa là công cụ `get_human_feedback`:
> ```python
> @tool
> def get_human_feedback(poem: str):
>     """Asks the user for feedback on the poem."""
>     feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
>     return feedback
> ```
>
> Khi `interrupt()` được gọi:
> 1. Thực thi đồ thị **dừng lại**
> 2. Giá trị truyền vào được hiển thị cho user
> 3. Khi tiếp tục với `Command(resume=...)`, giá trị đó trở thành giá trị trả về của `interrupt()`
>
> Xem system prompt của LLM:
> ```
> ALWAYS ASK FOR FEEDBACK FIRST.
> Only after you receive positive feedback you can return the final poem.
> ```
>
> LLM phải luôn yêu cầu phản hồi sau khi viết thơ, và chỉ trả kết quả cuối sau phản hồi tích cực.
> Chạy thử nào.

### Sau khi chạy — Bước 1: Yêu cầu viết thơ

> ```
> Next: ('tools',)
> ```
>
> `next` là `('tools',)`. Nghĩa là chúng ta đang **tạm dừng** tại node tools.
> LLM đã gọi `get_human_feedback`, và `interrupt()` đã dừng thực thi.
>
> Bây giờ đến lượt user cung cấp phản hồi.

### Sau khi chạy — Bước 2: Phản hồi tiêu cực

> ```python
> Command(resume="It is too long! Make it shorter, 4 lines max.")
> ```
>
> Chúng ta đưa phản hồi tiêu cực, nên LLM sửa lại bài thơ và hỏi phản hồi lần nữa.
> `next` vẫn là `('tools',)` — đang chờ interrupt tiếp.

### Sau khi chạy — Bước 3: Phản hồi tích cực

> ```python
> Command(resume="It looks great!")
> ```
>
> Lần này phản hồi tích cực, nên LLM trả bài thơ cuối cùng.
> `next` là `()` — đồ thị hoàn thành.
>
> Xem toàn bộ luồng hội thoại:
> ```
> human: Please make a poem about Python code.
> ai:                          <- viết thơ và gọi công cụ phản hồi
> tool: It is too long!...     <- phản hồi con người (tiêu cực)
> ai:                          <- sửa lại và hỏi lần nữa
> tool: It looks great!        <- phản hồi con người (tích cực)
> ai: Here's the final poem... <- kết quả cuối cùng
> ```
>
> **Con người nằm bên trong vòng lặp.** Đó là lý do gọi là Human-in-the-loop.

### Bài tập 14.3 (3 phút)

> **Bài tập 1**: Đưa phản hồi tiêu cực nhiều lần liên tiếp. LLM phản ứng thế nào?
>
> **Bài tập 2**: Truyền dictionary (dữ liệu có cấu trúc) vào `Command(resume=...)`.
>
> **Bài tập 3**: Thiết kế pipeline với nhiều điểm interrupt, như review rồi phê duyệt rồi deploy.

---

## 14.4 Du hành thời gian — State Fork (10 phút)

### Khái niệm

> Chúng ta nói checkpointer lưu tất cả state, đúng không?
> Vậy **có thể quay lại một thời điểm trong quá khứ không?**
>
> Được. Đó là **Du hành thời gian (Time Travel)**.
>
> Các API chính:
> - `get_state_history()` — lấy tất cả snapshot theo thứ tự thời gian
> - `update_state()` — sửa checkpoint quá khứ để tạo nhánh mới
>
> Tại sao hữu ích?
> - **Debug**: quay lại thời điểm lỗi xảy ra và phân tích nguyên nhân
> - **A/B testing**: phân nhánh từ cùng thời điểm với input khác nhau
> - **Rollback**: hoàn tác thực thi sai và bắt đầu lại

### Giải thích code (trước khi chạy)

> Chúng ta xây chatbot đơn giản và trò chuyện hai lần.
>
> 1. "I live in Europe. My city is Valencia." tiếp theo là response về Valencia
> 2. "What are some good restaurants near me?" tiếp theo là gợi ý nhà hàng ở Valencia
>
> Sau đó dùng `get_state_history()` xem tất cả snapshot,
> và **fork** từ thời điểm quá khứ bằng cách đổi "Valencia" thành "Zagreb".
>
> Chạy thử nào.

### Sau khi chạy — bắt đầu hội thoại

> ```
> Valencia is a beautiful city located on the eastern coast of Spain...
> ```
>
> Hội thoại bình thường.

### Sau khi chạy — câu hỏi tiếp theo

> ```
> Response dựa trên Valencia:
> La Pepica — Famous for its paella...
> ```
>
> Gợi ý nhà hàng dựa trên Valencia.

### Sau khi chạy — lịch sử state

> ```
> Snapshot 0: next=(), messages=4
> Snapshot 1: next=('chatbot',), messages=3
> ...
> Snapshot 5: next=('__start__',), messages=0
> ```
>
> Tổng cộng 6 snapshot. Mỗi cái là một thời điểm cụ thể trong quá trình chạy đồ thị.
> Chúng ta sẽ tìm **snapshot ngay sau khi user đề cập thành phố**.

### Sau khi chạy — fork (Valencia sang Zagreb)

> ```python
> graph.update_state(
>     fork_config,
>     {"messages": [HumanMessage(content="I live in Europe. My city is Zagreb.")]},
> )
> result_fork = graph.invoke(None, config=fork_config)
> ```
>
> Cập nhật state dùng config của snapshot quá khứ (`fork_config`),
> rồi `invoke(None)` để chạy lại từ thời điểm đó.
>
> Kết quả giờ hiện response dựa trên Zagreb.
> **Chúng ta đã fork từ thời điểm quá khứ trong cùng cuộc hội thoại.**
>
> Cuộc hội thoại gốc không bị ảnh hưởng. Fork tạo nhánh mới, không ghi đè nhánh hiện tại.

### Bài tập 14.4 (3 phút)

> **Bài tập 1**: Trò chuyện dài hơn, rồi fork từ giữa chừng.
>
> **Bài tập 2**: Hỏi câu khác ở nhánh fork và so sánh kết quả với nhánh gốc.
>
> **Bài tập 3**: Nghĩ về các tình huống thực tế mà du hành thời gian hữu ích (A/B testing, debug, rollback).

---

## Hướng dẫn bài tập tổng hợp (3 phút)

> Cuối notebook có 4 bài tập tổng hợp.
>
> **Bài 1** (★★☆): Chatbot đa công cụ — 3 công cụ: `get_weather`, `get_time`, `get_news`
> **Bài 2** (★★☆): Lưu giữ hội thoại — quản lý session bằng thread_id
> **Bài 3** (★★★): Code review HITL — review bởi con người qua interrupt()
> **Bài 4** (★★★): A/B test du hành thời gian — so sánh các nhánh từ cùng thời điểm
>
> Bài 1–2 là cơ bản, bài 3–4 là thử thách.
> Phân bổ thời gian: 10 phút cho bài dễ, 15 phút cho bài khó.

---

## Kết thúc (2 phút)

> Tổng kết những gì chúng ta đã học hôm nay.
>
> | Khái niệm | Điểm chính |
> |------------|------------|
> | MessagesState | Reducer tin nhắn tích hợp sẵn, chuẩn cho state chatbot |
> | init_chat_model | Khởi tạo một dòng với provider:tên_model |
> | @tool + ToolNode | Tích hợp công cụ bên ngoài vào đồ thị |
> | tools_condition | Tự động định tuyến dựa trên có tool_calls hay không |
> | SqliteSaver | Lưu state dựa trên DB, cách ly session qua thread_id |
> | interrupt() | Tạm dừng thực thi đồ thị cho con người can thiệp |
> | Command(resume) | Tiếp tục thực thi với phản hồi |
> | get_state_history | Lấy toàn bộ lịch sử snapshot |
> | update_state | Tạo nhánh mới (fork) từ checkpoint quá khứ |
>
> Từ chatbot cơ bản đến công cụ, bộ nhớ, con người can thiệp, và du hành thời gian.
> Các pattern này là building block cốt lõi của AI agent trong production.
>
> Cảm ơn các bạn. Làm tốt lắm.
