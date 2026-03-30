# Chapter 17: Kiểm thử Workflow LangGraph — Kịch bản giảng dạy

---

## Mở đầu (2 phút)

> Được rồi, cho đến nay chúng ta đã xây dựng các workflow với LangGraph.
>
> Nhưng làm sao để xác minh rằng các graph của chúng ta **thực sự hoạt động đúng**?
>
> Hôm nay chúng ta sẽ học cách **kiểm thử có hệ thống các workflow LangGraph bằng pytest**.
>
> Lộ trình như sau:
>
> ```
> 17.0 Setup
> 17.1 Graph xử lý email          (Dựa trên quy tắc, tất định)
> 17.2 Cơ bản Pytest               (%%writefile, parametrize)
> 17.3 Kiểm thử đơn vị từng node  (graph.nodes, update_state)
> 17.4 Chuyển sang AI node         (with_structured_output)
> 17.5 Chiến lược kiểm thử AI     (Range-based assertions)
> 17.6 LLM-as-a-Judge             (Golden examples, điểm tương đồng)
> ```
>
> Sự tiến hóa cốt lõi cần ghi nhớ:
> **Khớp chính xác --> Kiểm tra phạm vi --> LLM làm giám khảo**
>
> Code dựa trên quy tắc có thể kiểm thử bằng `assert ==`,
> nhưng output AI thay đổi mỗi lần, nên cần chiến lược mới.
>
> Hãy cùng trải nghiệm từng bước. Bắt đầu thôi.

---

## 17.0 Setup & Environment (2 phút)

### Trước khi chạy cell

> Đầu tiên, kiểm tra môi trường.
> Tải API key từ `.env` và xác minh phiên bản `langgraph`, `langchain`, `pytest`.
>
> Cùng môi trường như các chapter trước, thêm `pytest`.

### Sau khi chạy cell

> Nếu API key và phiên bản hiển thị đúng, chúng ta đã sẵn sàng.
> Nếu thiếu `pytest`, cài đặt bằng `pip install pytest`.

---

## 17.1 Graph xử lý email — Rule-based Email Classifier (8 phút)

### Khái niệm

> Chúng ta bắt đầu bằng việc xây dựng một **graph dễ kiểm thử**.
>
> Tại sao bắt đầu với quy tắc?
> Vì logic dựa trên quy tắc là **tất định (deterministic)**.
> Cùng input luôn cho cùng output. Kiểm thử rất đơn giản.
>
> Cấu trúc graph là tuyến tính:
>
> ```
> START --> categorize_email --> assign_priority --> generate_response --> END
> ```
>
> Ba node:
> - `categorize_email` — phân loại theo từ khóa (urgent / spam / normal)
> - `assign_priority` — gán mức ưu tiên theo danh mục (high / low / medium)
> - `generate_response` — trả về phản hồi mẫu theo danh mục
>
> Tất cả dùng `if-elif-else`. Không có LLM.

### Giải thích code (trước khi chạy)

> Hãy xem code.
>
> Định nghĩa State:
>
> ```python
> class EmailState(TypedDict):
>     email: str
>     category: str
>     priority: str
>     response: str
> ```
>
> Khi email đi qua graph, `category`, `priority` và `response` được điền dần.
>
> Hàm `categorize_email`:
>
> ```python
> def categorize_email(state: EmailState) -> dict:
>     email = state["email"].lower()
>     if "urgent" in email or "asap" in email:
>         return {"category": "urgent"}
>     elif "offer" in email or "discount" in email:
>         return {"category": "spam"}
>     return {"category": "normal"}
> ```
>
> Chuyển thành chữ thường, rồi tìm từ khóa.
> "urgent" hoặc "asap" thì urgent, "offer" hoặc "discount" thì spam, còn lại normal.
>
> `assign_priority` ánh xạ danh mục sang mức ưu tiên:
> urgent --> high, spam --> low, còn lại --> medium.
>
> `generate_response` trả về phản hồi mẫu theo danh mục.
>
> Các cạnh graph là tuyến tính:
>
> ```python
> builder.add_edge(START, "categorize_email")
> builder.add_edge("categorize_email", "assign_priority")
> builder.add_edge("assign_priority", "generate_response")
> builder.add_edge("generate_response", END)
> ```
>
> Pipeline thẳng. Không phân nhánh.

### Sau khi chạy

> ```
> "URGENT: Server is down, fix ASAP!" --> category: urgent, priority: high
> "Special offer! 50% discount today!" --> category: spam, priority: low
> "Hi, I have a question about my order." --> category: normal, priority: medium
> ```
>
> Hoạt động như mong đợi.
> Bây giờ hãy thay kiểm tra thủ công bằng **kiểm thử tự động**.

---

## 17.2 Cơ bản Pytest — parametrize + %%writefile (8 phút)

### Khái niệm

> Có một pattern để chạy pytest trong notebook. Ba bước:
>
> 1. `%%writefile main.py` — lưu code cần kiểm thử vào file `.py`
> 2. `%%writefile tests.py` — lưu code kiểm thử vào file `.py`
> 3. `!pytest tests.py -v` — chạy kiểm thử
>
> **`%%writefile` là gì?**
> Đó là lệnh magic của Jupyter. Lưu nội dung cell trực tiếp vào file.
> Chỉ cần đặt `%%writefile tên_file` ở dòng đầu tiên của cell.
>
> Tại sao cần thiết?
> pytest chỉ chạy file `.py`. Nó không đọc được cell notebook trực tiếp.
> Nên chúng ta dùng `%%writefile` để xuất code ra file.
>
> Và `@pytest.mark.parametrize` — decorator cho phép chạy một hàm kiểm thử
> với nhiều bộ input và giá trị kỳ vọng.
> Một hàm, sáu trường hợp, mười trường hợp — tất cả cùng lúc.

### Giải thích code (trước khi chạy)

> Đầu tiên, `%%writefile main.py` lưu code graph email vào file.
> Code giống hệt phần 17.1, không thay đổi.
>
> Tiếp theo, `%%writefile tests.py`:
>
> ```python
> @pytest.mark.parametrize(
>     "email, expected_category, expected_priority",
>     [
>         ("URGENT: Server down!", "urgent", "high"),
>         ("Fix this ASAP please", "urgent", "high"),
>         ("Special offer! Buy now!", "spam", "low"),
>         ("Get 50% discount today", "spam", "low"),
>         ("Hi, I have a question", "normal", "medium"),
>         ("Meeting tomorrow at 3pm", "normal", "medium"),
>     ],
> )
> def test_email_pipeline(email, expected_category, expected_priority):
>     graph = build_email_graph()
>     result = graph.invoke({"email": email})
>     assert result["category"] == expected_category
>     assert result["priority"] == expected_priority
>     assert len(result["response"]) > 0
> ```
>
> `@pytest.mark.parametrize`:
> - Tham số đầu: tên các tham số (chuỗi phân tách bằng dấu phẩy)
> - Tham số hai: danh sách test case (danh sách các tuple)
> - 6 tuple nghĩa là test chạy 6 lần
>
> `assert` làm test thất bại khi điều kiện là False.
> `result["category"] == expected_category` — phải khớp chính xác.
>
> Bên dưới có các test node riêng lẻ:
>
> ```python
> def test_categorize_urgent():
>     result = categorize_email({"email": "This is URGENT!", ...})
>     assert result["category"] == "urgent"
> ```
>
> Gọi trực tiếp hàm node. Đây là **nền tảng của kiểm thử đơn vị**.

### Sau khi chạy

> `!pytest tests.py -v` cho kết quả:
>
> ```
> tests.py::test_email_pipeline[URGENT: Server down!-urgent-high]   PASSED
> tests.py::test_email_pipeline[Fix this ASAP please-urgent-high]   PASSED
> ...
> tests.py::test_priority_mapping                                    PASSED
> tests.py::test_response_templates                                  PASSED
> ```
>
> Tất cả PASSED. Xanh toàn bộ!
>
> `-v` nghĩa là verbose. Mỗi test case được hiển thị trên một dòng.
> Nhờ parametrize, giá trị input xuất hiện trong tên test.
>
> Đây là ưu điểm của kiểm thử dựa trên quy tắc: **`assert ==` để xác minh chính xác.**

---

## 17.3 Kiểm thử đơn vị Node — graph.nodes + update_state (8 phút)

### Khái niệm

> Ở 17.2, chúng ta kiểm thử toàn bộ graph end-to-end.
> Nhưng thực tế, thường cần kiểm thử **từng node riêng biệt**.
>
> Hai phương pháp:
>
> **Phương pháp 1: `graph.nodes["name"].invoke(state)`**
> - Gọi trực tiếp một node cụ thể từ graph đã compile
> - Tương đương với gọi hàm node trực tiếp
>
> **Phương pháp 2: `graph.update_state(config, values, as_node="name")`**
> - Tiêm state như thể một node cụ thể đã tạo ra nó
> - Sau đó chỉ chạy **các node còn lại** (thực thi một phần)
> - Cần checkpointer `MemorySaver`

### Giải thích code — Phương pháp 1 (trước khi chạy)

> ```python
> test_state = {"email": "URGENT: Need help ASAP", "category": "", "priority": "", "response": ""}
>
> cat_result = graph.nodes["categorize_email"].invoke(test_state)
> print(f"categorize_email result: {cat_result}")
> ```
>
> `graph.nodes` là dictionary. Key là tên node, value là đối tượng node.
> `.invoke(state)` chạy chỉ node đó.
>
> Quan trọng: **không chạy toàn bộ graph.**
> Chỉ `categorize_email` thực thi và trả về kết quả.
>
> Điều này cho phép xác minh "hàm node của tôi có trả về giá trị đúng không?" một cách độc lập.

### Sau khi chạy — Phương pháp 1

> ```
> categorize_email result: {'category': 'urgent'}
> assign_priority result: {'priority': 'high'}
> generate_response result: {'response': 'This email has been classified as spam.'}
> ```
>
> Mỗi node hoạt động độc lập.

### Giải thích code — Phương pháp 2: update_state (trước khi chạy)

> Đầu tiên, tạo graph có checkpointer:
>
> ```python
> from langgraph.checkpoint.memory import MemorySaver
> graph_mem = builder.compile(checkpointer=MemorySaver())
> ```
>
> `MemorySaver` là checkpointer trong bộ nhớ. Nhẹ hơn SQLite, phù hợp cho kiểm thử.
>
> Sau đó:
>
> ```python
> config2 = {"configurable": {"thread_id": "test_partial_2"}}
>
> # Chạy toàn bộ graph trước
> graph_mem.invoke({"email": "Hello, normal email"}, config=config2)
>
> # Tiêm state cưỡng bức như thể categorize_email trả về "urgent"
> graph_mem.update_state(
>     config2,
>     {"category": "urgent"},
>     as_node="categorize_email",
> )
> ```
>
> `as_node="categorize_email"` — đây là điểm mấu chốt!
> Nghĩa là "xử lý giá trị này như thể categorize_email đã tạo ra nó."
>
> Chúng ta đã cưỡng bức thay đổi category từ "normal" thành "urgent".
>
> ```python
> result_partial = graph_mem.invoke(None, config=config2)
> ```
>
> `invoke(None)` — không có input mới, chỉ **chạy các node còn lại**.
> `assign_priority --> generate_response` được thực thi.
> Vì category là "urgent", priority trở thành "high".

### Sau khi chạy — Phương pháp 2

> ```
> State sau khi tiêm: category=urgent
> Kết quả thực thi một phần: category=urgent, priority=high
> Assert thành công! Thực thi một phần với update_state thành công
> ```
>
> Tại sao điều này hữu ích?
>
> Khi bạn nghi ngờ node thứ 3 có bug,
> không cần chạy hai node đầu.
> **Tiêm state mong muốn, chỉ kiểm thử node thứ 3.**
>
> Tiết kiệm thời gian và debug nhanh hơn.

---

## 17.4 Chuyển sang AI Node — LLM + with_structured_output (8 phút)

### Khái niệm

> Đây là nơi mô hình thay đổi.
>
> Cho đến giờ, mọi thứ dựa trên quy tắc và tất định.
> Bây giờ chúng ta **thay hàm node bằng AI, giữ nguyên cấu trúc graph**.
>
> Cấu trúc graph không đổi:
> ```
> START --> categorize_email --> assign_priority --> generate_response --> END
> ```
>
> Chỉ logic bên trong mỗi node thay đổi.
>
> Kỹ thuật cốt lõi: **`with_structured_output`**
>
> LLM tự nhiên trả về văn bản tự do.
> Nhưng graph cần chính xác `"urgent"`, `"spam"`, hoặc `"normal"`.
>
> `with_structured_output(PydanticModel)` ép output LLM vào Pydantic model.
> Dù LLM nói gì, nó được parse thành các trường và kiểu đã định nghĩa.

### Giải thích code (trước khi chạy)

> Schema output Pydantic:
>
> ```python
> class CategoryOutput(BaseModel):
>     category: Literal["urgent", "spam", "normal"] = Field(
>         description="The email category"
>     )
> ```
>
> `Literal["urgent", "spam", "normal"]` — chỉ cho phép 3 giá trị này!
> Ngay cả khi LLM muốn nói "critical", nó phải chọn một trong ba giá trị này.
>
> `Field(description=...)` giải thích cho LLM trường này đại diện cho gì.
>
> Cùng pattern cho `PriorityOutput` và `ResponseOutput`.
>
> Tạo Structured LLM:
>
> ```python
> category_llm = llm.with_structured_output(CategoryOutput)
> priority_llm = llm.with_structured_output(PriorityOutput)
> response_llm = llm.with_structured_output(ResponseOutput)
> ```
>
> Bây giờ `category_llm.invoke("...")` trả về đối tượng `CategoryOutput`,
> không phải chuỗi — một đối tượng có thuộc tính `.category`.
>
> Hàm AI node:
>
> ```python
> def ai_categorize_email(state: EmailState) -> dict:
>     result = category_llm.invoke(
>         f"Classify this email into one of: urgent, spam, normal.\n\nEmail: {state['email']}"
>     )
>     return {"category": result.category}
> ```
>
> So sánh quy tắc vs AI:
> - Quy tắc: `if "urgent" in email` — so khớp từ khóa
> - AI: LLM hiểu ngữ cảnh và đưa ra phán đoán
>
> "Server bị sập" không có từ khóa "urgent", nhưng AI vẫn phân loại thành urgent.

### Sau khi chạy

> ```
> "URGENT: Production database is corrupted" --> Category: urgent, Priority: high
> "Congratulations! You won $1,000,000!" --> Category: spam, Priority: low
> ```
>
> AI phân loại đúng.
>
> Nhưng có vấn đề! **Chạy lại cùng email có thể cho kết quả khác.**
> Output AI là **bất tất định (non-deterministic)**.
>
> Điều này nghĩa là `assert ==` sẽ không hoạt động đáng tin cậy.
> Chúng ta cần chiến lược kiểm thử mới.

---

## 17.5 Chiến lược kiểm thử AI — Range-based Assertions (8 phút)

### Khái niệm

> Output AI thay đổi mỗi lần.
> Vậy kiểm thử thế nào?
>
> **Xác thực bằng phạm vi (range)!**
>
> Bốn chiến lược:
>
> 1. **Phạm vi giá trị hợp lệ**: `assert result in ["urgent", "spam", "normal"]`
>    - Giá trị gì cũng được, miễn là một trong ba, OK
>
> 2. **Phạm vi độ dài**: `assert 20 <= len(response) <= 1000`
>    - Không quá ngắn, không quá dài, OK
>
> 3. **Ngưỡng chất lượng tối thiểu**: `assert score >= threshold`
>    - Điểm trên ngưỡng, OK
>
> 4. **Tính nhất quán**: Chạy cùng input N lần, kiểm tra đa số đồng ý
>    - Ít nhất 2 trên 3 lần trùng nhau, OK
>
> Vì không biết giá trị chính xác, chúng ta xác minh
> "ít nhất phải nằm trong phạm vi này."

### Giải thích code (trước khi chạy)

> `%%writefile tests_ai.py` tạo file test.
>
> ```python
> VALID_CATEGORIES = {"urgent", "spam", "normal"}
> VALID_PRIORITIES = {"high", "medium", "low"}
> ```
>
> Tập giá trị hợp lệ được định nghĩa trước.
>
> ```python
> @pytest.fixture
> def ai_graph():
>     return build_ai_email_graph()
> ```
>
> `@pytest.fixture` — hàm tạo đối tượng tái sử dụng cho các test.
> Nhiều hàm test nhận `ai_graph` làm tham số sẽ được tự động tiêm.
>
> ```python
> def test_output_in_valid_range(ai_graph, email):
>     result = ai_graph.invoke({"email": email})
>     assert result["category"] in VALID_CATEGORIES
>     assert result["priority"] in VALID_PRIORITIES
>     assert len(result["response"]) > 10
>     assert len(result["response"]) < 2000
> ```
>
> Không phải `assert ==` mà `assert in`!
> Không phải "có phải urgent không?" mà "có phải một trong urgent, spam, normal không?"
>
> Test tính nhất quán:
>
> ```python
> def test_consistency_over_runs(ai_graph):
>     email = "URGENT: Production is completely down!"
>     categories = []
>     for _ in range(3):
>         result = ai_graph.invoke({"email": email})
>         categories.append(result["category"])
>
>     from collections import Counter
>     most_common_count = Counter(categories).most_common(1)[0][1]
>     assert most_common_count >= 2
> ```
>
> Chạy 3 lần, kết quả xuất hiện nhiều nhất phải ít nhất 2 lần.
> Tức 66% nhất quán. "2 trên 3 phải đồng ý."

### Sau khi chạy

> ```
> tests_ai.py::test_output_in_valid_range[URGENT: Server...]     PASSED
> tests_ai.py::test_clear_cases_match_expected[CRITICAL...]       PASSED
> tests_ai.py::test_response_length_range                          PASSED
> tests_ai.py::test_consistency_over_runs                          PASSED
> ```
>
> Tất cả PASSED!
>
> Điểm quan trọng — so sánh kiểm thử quy tắc vs AI:
>
> | Dựa trên quy tắc | Dựa trên AI |
> |-------------------|-------------|
> | `assert ==` khớp chính xác | `assert in` kiểm tra phạm vi |
> | Chạy một lần là đủ | N lần cho tính nhất quán |
> | Luôn cùng kết quả | Có thể khác mỗi lần |
>
> Chiến lược kiểm thử hoàn toàn khác nhau.

---

## 17.6 LLM-as-a-Judge — Golden Examples + Similarity Scoring (10 phút)

### Khái niệm

> Được rồi, phương pháp kiểm thử cuối cùng và mạnh mẽ nhất.
>
> Ở 17.5, xác thực phạm vi kiểm tra "giá trị có hợp lệ không?"
> Nhưng làm sao kiểm tra "**chất lượng** phản hồi có tốt không?"
>
> Để người đọc từng phản hồi? Không hiệu quả.
> Nên chúng ta để **LLM đóng vai giám khảo (Judge)**.
>
> Pattern gồm ba phần:
>
> 1. **Golden Examples** — phản hồi lý tưởng được chuẩn bị cho mỗi danh mục
> 2. **Judge LLM** — so sánh phản hồi được tạo với golden example, cho điểm tương đồng
> 3. **Threshold** — điểm từ 70 trở lên là đạt
>
> Ví von:
> - Golden Example = đáp án mẫu
> - Judge LLM = giám khảo chấm điểm
> - Threshold = điểm đạt yêu cầu

### Giải thích code — Golden Examples (trước khi chạy)

> ```python
> RESPONSE_EXAMPLES = {
>     "urgent": (
>         "Thank you for alerting us. We have escalated this to our on-call team "
>         "and will provide an update within 1 hour..."
>     ),
>     "spam": (
>         "This message has been identified as unsolicited commercial email..."
>     ),
>     "normal": (
>         "Thank you for your email. We have received your message and a team "
>         "member will respond within 24 hours..."
>     ),
> }
> ```
>
> Viết sẵn "phản hồi lý tưởng" cho mỗi danh mục.
> Chúng ta sẽ đánh giá phản hồi AI tạo ra **tương tự** bao nhiêu với những phản hồi này.

### Giải thích code — SimilarityScoreOutput + Hàm Judge

> ```python
> class SimilarityScoreOutput(BaseModel):
>     score: int = Field(gt=0, lt=100, description="Similarity score between 1 and 99")
>     reasoning: str = Field(description="Brief explanation of the score")
> ```
>
> Cấu trúc Judge LLM trả về:
> - `score` — điểm tương đồng từ 1 đến 99
> - `reasoning` — giải thích cho điểm số
>
> `gt=0, lt=100` — ràng buộc Pydantic. Phải lớn hơn 0 và nhỏ hơn 100.
>
> ```python
> judge_llm = llm.with_structured_output(SimilarityScoreOutput)
> ```
>
> Lại `with_structured_output`! Lần này cấu trúc hóa điểm và lý do.
>
> Hàm Judge:
>
> ```python
> def judge_response(generated: str, golden: str) -> SimilarityScoreOutput:
>     prompt = f"""You are an expert quality evaluator. Compare the generated response
> with the golden (ideal) response and score their similarity.
>
> Consider:
> - Tone and professionalism (30%)
> - Key information coverage (40%)
> - Appropriate length and format (30%)
>
> Golden Response:
> {golden}
>
> Generated Response:
> {generated}
>
> Score from 1 to 99..."""
>     return judge_llm.invoke(prompt)
> ```
>
> Prompt chỉ rõ tiêu chí đánh giá:
> - Giọng văn và tính chuyên nghiệp 30%
> - Bao quát thông tin cốt lõi 40%
> - Độ dài và định dạng phù hợp 30%
>
> Judge LLM chấm điểm theo các tiêu chí này.

### Sau khi chạy — Test Judge

> ```
> Category: urgent
> AI Response: "We have received your critical alert..."
> Similarity Score: 82
> Reasoning: "Both responses acknowledge urgency and promise timely action..."
> ```
>
> Điểm 82! Trên ngưỡng 70, nên PASS.

### Giải thích code — pytest với Judge

> Xem `%%writefile tests_judge.py`:
>
> ```python
> THRESHOLD = 70
>
> def test_response_quality_above_threshold(ai_graph, email, expected_category):
>     result = ai_graph.invoke({"email": email})
>     golden = RESPONSE_EXAMPLES[expected_category]
>     score_result = judge_response(result["response"], golden)
>
>     assert score_result.score >= THRESHOLD
> ```
>
> So sánh phản hồi AI tạo ra với golden example.
> Nếu điểm Judge từ 70 trở lên, test đạt.
>
> Còn hai test trường hợp cực đoan:
>
> ```python
> def test_judge_perfect_match():
>     golden = RESPONSE_EXAMPLES["urgent"]
>     score_result = judge_response(golden, golden)
>     assert score_result.score >= 90
> ```
>
> Đưa golden example vào chính nó phải cho 90+ điểm. (So sánh với chính mình.)
>
> ```python
> def test_judge_poor_match():
>     poor_response = "lol ok whatever"
>     score_result = judge_response(poor_response, golden)
>     assert score_result.score < 40
> ```
>
> Phản hồi tệ phải dưới 40 điểm.
>
> Điều này xác thực chính Judge. "Judge có đang phán xét đúng không?"

### Sau khi chạy — pytest

> ```
> tests_judge.py::test_response_quality_above_threshold[EMERGENCY...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[FREE GIFT...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[Hi, could...]   PASSED
> tests_judge.py::test_judge_perfect_match                               PASSED
> tests_judge.py::test_judge_poor_match                                  PASSED
> ```
>
> Tất cả đạt!

---

## Kết thúc — Sự tiến hóa của chiến lược kiểm thử (4 phút)

> Hãy tổng kết những gì đã học hôm nay.
>
> **Sự tiến hóa của chiến lược kiểm thử:**
>
> ```
> 17.1-17.2  Dựa trên quy tắc  -->  assert == (khớp chính xác)
> 17.4-17.5  Dựa trên AI        -->  assert in (xác thực phạm vi)
> 17.6       LLM Judge           -->  assert score >= threshold (xác thực chất lượng)
> ```
>
> Khi chuyển từ hệ thống tất định sang bất tất định,
> kiểm thử trở nên **nới lỏng hơn nhưng thực tế hơn**.
>
> Tổng hợp công cụ cốt lõi:
>
> | Công cụ | Mục đích |
> |---------|----------|
> | `%%writefile` | Tạo file .py từ notebook |
> | `@pytest.mark.parametrize` | Test nhiều trường hợp cùng lúc |
> | `graph.nodes["name"].invoke()` | Kiểm thử đơn vị từng node |
> | `update_state(as_node="...")` | Tiêm state cho thực thi một phần |
> | `with_structured_output` | Cấu trúc hóa output LLM |
> | `SimilarityScoreOutput` | Schema điểm Judge |
>
> Trước khi sang chapter tiếp, hãy làm Final Exercises.
> Đặc biệt **Bài tập 2 (Benchmark độ chính xác phân loại AI)** là
> pattern được sử dụng nhiều nhất trong thực tế.
>
> Cảm ơn các bạn đã nỗ lực!
