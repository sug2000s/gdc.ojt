# Chapter 18: Kiến trúc Multi-Agent — Kịch bản bài giảng

---

## Mở đầu (2 phút)

> Được rồi, hôm nay chúng ta sẽ học **Kiến trúc Multi-Agent**.
>
> Cho đến giờ, một agent duy nhất xử lý mọi thứ, đúng không?
> Nhưng trong thực tế, bạn thường cần **nhiều agent phối hợp với nhau**.
>
> Ví dụ: nếu khách hàng hỏi bằng tiếng Hàn, agent tiếng Hàn trả lời.
> Nếu hỏi bằng tiếng Tây Ban Nha, agent tiếng Tây Ban Nha tiếp nhận.
>
> Nội dung hôm nay:
>
> ```
> 18.1 Network Architecture    (P2P — agent tự chuyển giao)
> 18.2 Supervisor Architecture  (bộ điều phối trung tâm)
> 18.3 Supervisor as Tools      (agent được đóng gói thành tool)
> ```
>
> Chúng ta sẽ giải cùng một bài toán với ba kiến trúc khác nhau.
> Trải nghiệm từng cái và so sánh ưu nhược điểm. Bắt đầu thôi.

---

## 18.0 Setup & Environment (3 phút)

### Trước khi chạy cell

> Đầu tiên, kiểm tra môi trường.
> Chúng ta load API key từ file `.env` và kiểm tra phiên bản `langgraph` và `langchain`.
>
> Các package cần thiết hôm nay:
> - `langgraph >= 1.1` — hỗ trợ multi-agent
> - `langchain >= 1.2` — tích hợp LLM
> - `langchain-openai` — mô hình OpenAI

### Sau khi chạy cell

> Nếu phiên bản hiển thị đúng, chúng ta đã sẵn sàng.
> Nếu bị lỗi, kiểm tra `uv sync` hoặc `pip install`.

---

## 18.1 Network Architecture — Chuyển giao P2P giữa các Agent (15 phút)

### Khái niệm

> Kiến trúc đầu tiên là phương pháp **Network (P2P)**.
>
> Không có bộ điều phối trung tâm. Mỗi agent **tự quyết định**
> khi nào chuyển giao cuộc hội thoại cho agent khác.
>
> ```
> korean_agent ◄──► greek_agent
>       ▲               ▲
>       └──► spanish_agent ◄──┘
> ```
>
> Ba khái niệm chính:
>
> 1. **`handoff_tool`** — công cụ chuyển giao. Dùng `Command(goto=..., graph=Command.PARENT)` để chuyển agent ở cấp đồ thị cha.
> 2. **`make_agent()` factory** — tạo agent có cấu trúc giống nhau nhưng tham số khác nhau.
> 3. **Subgraph** — mỗi agent chạy vòng lặp ReAct độc lập riêng.
>
> `Command.PARENT` là chìa khóa. Nó cho phép subgraph nhảy đến node khác trong đồ thị cha.

### Giải thích code (trước khi chạy)

> Cùng xem code. Được tổ chức thành 4 bước.
>
> **Bước 1 — Định nghĩa State:**
> ```python
> class AgentsState(MessagesState):
>     current_agent: str
>     transfered_by: str
> ```
> Mở rộng `MessagesState` để theo dõi agent hiện tại và agent đã chuyển giao.
>
> **Bước 2 — `make_agent()` factory:**
> Mỗi agent có prompt và tools khác nhau nhưng cấu trúc giống nhau.
> Vòng lặp `agent → tools_condition → tools → agent`.
> Đây chính là pattern ReAct mà chúng ta đã thấy trước đó.
>
> **Bước 3 — `handoff_tool`:**
> ```python
> return Command(
>     update={"current_agent": transfer_to},
>     goto=transfer_to,
>     graph=Command.PARENT,  # Chuyển ở cấp đồ thị cha!
> )
> ```
> Không có `graph=Command.PARENT`, việc chuyển đổi sẽ chỉ xảy ra bên trong subgraph.
> Có nó, chúng ta nhảy đến node agent khác trong đồ thị cha.
>
> Cũng có cơ chế bảo vệ vòng lặp vô hạn — agent không thể tự chuyển cho chính mình.
>
> **Bước 4 — Đồ thị cấp cao nhất:**
> Mỗi agent được đăng ký như một node với `destinations` chỉ định các mục tiêu chuyển giao.
> `START → korean_agent` đặt điểm vào mặc định.
>
> Chạy thôi.

### Sau khi chạy — Tin nhắn tiếng Hàn

> Chúng ta gửi "Xin chào! Tôi có vấn đề về tài khoản" bằng tiếng Hàn.
>
> `korean_agent` xử lý trực tiếp. Không cần chuyển giao.
> Tin nhắn tiếng Hàn đến agent tiếng Hàn — tự nhiên nó tự xử lý.

### Sau khi chạy — Tin nhắn tiếng Tây Ban Nha

> Bây giờ là tiếng Tây Ban Nha: "Hola! Necesito ayuda con mi cuenta."
>
> Xem output:
> ```
> [korean_agent] current_agent=spanish_agent    ← phát hiện, chuyển giao!
> [spanish_agent] current_agent=spanish_agent    ← trả lời bằng tiếng TBN
> ```
>
> `korean_agent` phát hiện tiếng Tây Ban Nha và gọi `handoff_tool`
> để chuyển sang `spanish_agent`. Một quyết định tự chủ.
>
> Đây là đặc trưng của kiến trúc Network.
> Mỗi agent tự quyết định: "Nếu tôi không xử lý được, tôi sẽ chuyển tiếp."

### Exercise 18.1 (5 phút)

> **Exercise 1**: Gửi tin nhắn tiếng Hy Lạp và quan sát luồng chuyển giao.
>
> **Exercise 2**: Thêm agent tiếng Nhật. Nhớ cập nhật docstring của `handoff_tool` nữa.
>
> **Exercise 3**: Xóa `Command.PARENT` và xem lỗi gì xảy ra.

---

## 18.2 Supervisor Architecture — Bộ điều phối trung tâm (15 phút)

### Khái niệm

> Kiến trúc thứ hai là phương pháp **Supervisor**.
>
> Trong Network, mọi agent đều có logic routing.
> Supervisor thì khác. **Một bộ điều phối trung tâm** xử lý tất cả routing.
>
> ```
>          Supervisor
>         /    |     \
>    korean  greek  spanish
>         \    |     /
>          Supervisor  ← quay lại
> ```
>
> Agent chỉ tập trung vào vai trò của mình. Không cần biết về routing.
>
> Các khái niệm chính:
> - **Structured Output** — `SupervisorOutput(next_agent, reasoning)` cho routing an toàn
> - **Kiểu `Literal`** — giới hạn giá trị hợp lệ để ngăn routing sai
> - **Đồ thị tuần hoàn** — agent → supervisor → agent → ... → `__end__`
> - **Trường `reasoning`** — theo dõi lý do chọn agent (rất hữu ích cho debugging)

### Giải thích code (trước khi chạy)

> Cùng xem code.
>
> **Schema output của Supervisor:**
> ```python
> class SupervisorOutput(BaseModel):
>     next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
>     reasoning: str
> ```
> `Literal` giới hạn giá trị hợp lệ. LLM không thể trả về giá trị rác.
> `__end__` cho phép kết thúc cuộc hội thoại.
>
> **Agent factory — `make_agent()`:**
> So sánh với 18.1. Không có tham số `tools`!
> Agent không cần công cụ chuyển giao. Chỉ đơn giản trả lời.
> Cấu trúc đơn giản hơn nhiều.
>
> **Node Supervisor:**
> ```python
> structured_llm = llm.with_structured_output(SupervisorOutput)
> ```
> `with_structured_output` buộc LLM phải trả lời theo format `SupervisorOutput`.
> Sau đó `Command(goto=response.next_agent)` xử lý routing.
>
> **Cấu trúc đồ thị — tuần hoàn:**
> ```
> START → supervisor → agent → supervisor → agent → ... → END
> ```
> Sau khi agent trả lời, nó quay lại supervisor.
> Khi supervisor trả về `__end__`, cuộc hội thoại kết thúc.
>
> Chạy thôi.

### Sau khi chạy — Tin nhắn tiếng Hàn

> Xem output:
> ```
> Supervisor → korean_agent (reason: ...)
> ```
>
> Supervisor phát hiện tiếng Hàn và routing đến `korean_agent`.
> Trường `reasoning` cho biết lý do. Rất hữu ích cho debugging.
>
> Sau khi agent trả lời, nó quay lại supervisor,
> supervisor trả về `__end__` để kết thúc.
>
> So với Network: code agent đơn giản hơn nhiều.
> Nhưng supervisor phải gánh toàn bộ quyết định.

### Sau khi chạy — Tin nhắn tiếng Tây Ban Nha

> Bây giờ tiếng Tây Ban Nha. Supervisor routing đến `spanish_agent`.
>
> Agent không cần xác định liệu nó có phải agent đúng cho ngôn ngữ đó không.
> Supervisor xử lý tất cả. Đây là lợi ích của việc phân tách trách nhiệm.

### Exercise 18.2 (5 phút)

> **Exercise 1**: In trường `reasoning` để phân tích cơ sở quyết định của supervisor.
>
> **Exercise 2**: Khi thêm agent, so sánh code cần thay đổi trong Network vs Supervisor.
>
> **Exercise 3**: Xóa option `__end__` khỏi `SupervisorOutput` và xem điều gì xảy ra.

---

## 18.3 Supervisor as Tools — Agent đóng gói thành Tool (10 phút)

### Khái niệm

> Kiến trúc thứ ba. **Cấu trúc gọn gàng nhất** trong tất cả.
>
> Ý tưởng cốt lõi: biến agent thành hàm `@tool`.
> Supervisor dùng `bind_tools` để gắn các tool agent,
> và gọi chúng thông qua LLM tool calling một cách tự nhiên.
>
> ```
> Supervisor ──tools_condition──► ToolNode
>                                ├ korean_agent
>                                ├ greek_agent
>                                └ spanish_agent
> ```
>
> Không cần logic routing riêng. LLM tự chọn tool phù hợp.
> Không Structured Output, không Command, không handoff_tool.
>
> Thêm agent = thêm một hàm `@tool`. Xong.

### Giải thích code (trước khi chạy)

> Cùng xem code. Đơn giản đến bất ngờ.
>
> **Agent = hàm @tool:**
> ```python
> @tool
> def korean_agent(message: str) -> str:
>     """Transfer to Korean customer support agent."""
>     response = llm.invoke(...)
>     return response.content
> ```
>
> Docstring rất quan trọng. LLM đọc mô tả này để quyết định khi nào gọi tool.
>
> **Supervisor:**
> ```python
> llm_with_tools = llm.bind_tools(agent_tools)
> ```
> Gắn các tool agent vào LLM. System prompt yêu cầu "routing đến agent ngôn ngữ phù hợp."
>
> **Đồ thị — cấu trúc ReAct:**
> ```
> START → supervisor → tools_condition → ToolNode → supervisor → ... → END
> ```
>
> Thực ra đây chính là pattern ReAct từ Chapter 15.
> Điểm khác biệt duy nhất là tool ở đây là **agent** thay vì hàm thông thường.
>
> Chạy thôi.

### Sau khi chạy — Tin nhắn tiếng Hàn

> Xem kết quả:
> ```
> 고객님, 비밀번호 변경을 도와드리겠습니다...
> ```
>
> Supervisor gọi tool `korean_agent`,
> và phản hồi bằng tiếng Hàn được trả về.
>
> So với 18.2 — code ít hơn một nửa.
> Không schema Structured Output, không Command, không thiết lập đồ thị tuần hoàn.

### Sau khi chạy — Tin nhắn tiếng Tây Ban Nha

> Tương tự với tiếng Tây Ban Nha. Tool `spanish_agent` được gọi.
>
> LLM tự nhiên quyết định "khách hàng này nói tiếng Tây Ban Nha,
> vậy tôi sẽ gọi spanish_agent" thông qua tool calling.
>
> Bạn có thể cảm nhận ưu điểm của kiến trúc này — ít code nhất, cấu trúc gọn nhất.

### Exercise 18.3 (5 phút)

> **Exercise 1**: Thêm agent tiếng Nhật dưới dạng `@tool`. Cảm nhận sự thay đổi code đơn giản thế nào.
>
> **Exercise 2**: So sánh ba kiến trúc (Network, Supervisor, Supervisor+Tools):
> - Phương pháp routing, độ phức tạp agent, khả năng mở rộng, dễ debug
>
> **Exercise 3**: Thiết kế kịch bản thực tế ngoài routing ngôn ngữ (ví dụ: routing theo phòng ban, phân loại theo tech stack).

---

## Tổng kết so sánh kiến trúc (3 phút)

> Cùng so sánh ba kiến trúc.
>
> | | Network (18.1) | Supervisor (18.2) | Supervisor+Tools (18.3) |
> |--|---------|------------|------------------|
> | **Routing** | Mỗi agent tự quyết | Supervisor trung tâm | LLM tool calling |
> | **Độ phức tạp agent** | Cao (logic handoff) | Thấp (chỉ trả lời) | Tối thiểu (hàm @tool) |
> | **Khả năng mở rộng** | Sửa tất cả agent | Chỉ sửa supervisor | Chỉ thêm tool |
> | **Debugging** | Khó | Theo dõi reasoning | Theo dõi tool call |
> | **Phù hợp khi** | Ít agent, cần tự chủ | Quy mô vừa, cần kiểm soát | Quy mô lớn, cấu trúc gọn |
>
> Trong thực tế, **Supervisor+Tools (18.3)** được sử dụng phổ biến nhất.
> Nhưng Network phù hợp khi agent cần phối hợp tự chủ,
> và Supervisor phù hợp khi bạn cần kiểm soát routing chi tiết.
>
> Chọn đúng kiến trúc cho tình huống mới là điều quan trọng.

---

## Hướng dẫn bài tập tổng hợp (2 phút)

> Có 3 bài tập tổng hợp ở cuối notebook.
>
> **Bài tập 1** (★★☆): Kiến trúc Network 4 ngôn ngữ — thêm agent tiếng Nhật
> **Bài tập 2** (★★★): Supervisor routing theo phòng ban — billing, tech, general
> **Bài tập 3** (★★★): Supervisor+Tools với tool chuyên dụng — weather, calculator, search
>
> Dành 15 phút cho Bài tập 1, 20 phút mỗi bài cho Bài tập 2-3.

---

## Kết thúc (2 phút)

> Tổng kết những gì chúng ta đã học hôm nay.
>
> | Khái niệm | Điểm chính |
> |-----------|-----------|
> | Network (P2P) | `handoff_tool` + `Command.PARENT` cho chuyển giao tự chủ |
> | Supervisor | Structured Output cho routing trung tâm, đồ thị tuần hoàn |
> | Supervisor+Tools | Agent thành `@tool`, tái sử dụng pattern ReAct |
> | `make_agent()` | Agent factory — cùng cấu trúc, khác tham số |
> | `Command.PARENT` | Nhảy từ subgraph lên đồ thị cha |
> | `destinations` | Khai báo mục tiêu chuyển giao trong đồ thị |
>
> Multi-agent là tinh hoa của LangGraph.
> Khi xây dựng hệ thống agent trong production, bạn sẽ dùng một trong ba pattern này.
>
> Cảm ơn các bạn. Bài học hôm nay kết thúc tại đây.
