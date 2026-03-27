# Chapter 13: Cơ bản LangGraph — Kịch bản bài giảng

---

## Mở đầu (2 phút)

> Được rồi, hôm nay chúng ta sẽ học **LangGraph** từ đầu.
>
> Cho đến giờ, các agent chúng ta xây dựng đều dùng vòng lặp code thủ công, đúng không?
> LangGraph cho phép thiết kế chúng dưới dạng cấu trúc **Đồ thị (Graph)**.
>
> Node là "việc cần làm", Edge là "tiếp theo là gì".
> Kết hợp chúng lại, bạn có thể xây dựng các workflow AI phức tạp một cách gọn gàng.
>
> Nội dung hôm nay:
>
> ```
> 13.1 Đồ thị đầu tiên        (Node, Edge, START, END)
> 13.2 Quản lý State           (Đọc/Sửa State)
> 13.4 Đa Schema               (Tách Input/Output/Internal)
> 13.5 Reducer                  (Tích lũy State)
> 13.6 Node Caching             (CachePolicy)
> 13.7 Conditional Edges        (Phân nhánh động)
> 13.8 Send API                 (Xử lý song song động)
> 13.9 Command                  (Định tuyến trong Node)
> ```
>
> Nhìn có vẻ nhiều, nhưng chúng ta sẽ xây dựng từng bước một. Bắt đầu thôi.

---

## 13.0 Setup & Environment (3 phút)

### Trước khi chạy cell

> Đầu tiên, kiểm tra môi trường.
> Chúng ta load API key từ file `.env` và kiểm tra phiên bản `langgraph` và `langchain`.
>
> Các package cần thiết hôm nay:
> - `langgraph >= 0.6.6` — framework cốt lõi
> - `langchain[openai] >= 0.3.27` — tích hợp OpenAI
> - `grandalf` — trực quan hóa đồ thị

### Sau khi chạy cell

> Nếu phiên bản hiển thị đúng, chúng ta đã sẵn sàng.
> Nếu bị lỗi, kiểm tra `uv sync` hoặc `pip install`.

---

## 13.1 Đồ thị đầu tiên (10 phút)

### Khái niệm

> LangGraph có ba thành phần cốt lõi:
>
> 1. **State** — dữ liệu chia sẻ trong toàn bộ đồ thị. Định nghĩa bằng `TypedDict`.
> 2. **Node** — hàm thực thi. Nhận state làm đầu vào.
> 3. **Edge** — kết nối giữa các node. Xác định thứ tự thực thi.
>
> Và hai node đặc biệt:
> - `START` — điểm bắt đầu của đồ thị
> - `END` — điểm kết thúc của đồ thị
>
> Chỉ cần năm thứ này, bạn có thể xây dựng bất kỳ đồ thị nào.

### Giải thích code (trước khi chạy)

> Cùng xem code.
>
> ```python
> class State(TypedDict):
>     hello: str
> ```
>
> State có một trường chuỗi `hello`.
>
> ```python
> graph_builder = StateGraph(State)
> ```
>
> Truyền state schema vào `StateGraph` để tạo graph builder.
>
> Ba hàm node chỉ đơn giản `print`. Chưa thay đổi state.
>
> Sau đó `add_node` đăng ký node, và `add_edge` kết nối chúng:
>
> ```
> START -> node_one -> node_two -> node_three -> END
> ```
>
> Cuối cùng, `compile()` → `invoke()` để chạy.
> Chạy thử nào.

### Sau khi chạy

> `node_one`, `node_two`, `node_three` in theo thứ tự, đúng không?
> Đây là **đồ thị tuyến tính** cơ bản nhất.
>
> Ở cell tiếp theo, `draw_mermaid_png()` trực quan hóa đồ thị.
> Bạn thấy các mũi tên nối theo thứ tự — đó là đồ thị chúng ta đã xây.

### Bài tập 13.1 (5 phút)

> **Bài tập 1**: Bỏ edge `START`. Lỗi gì xảy ra?
>
> **Bài tập 2**: Thay đổi thứ tự node. Thứ tự `add_edge` quyết định luồng thực thi.
>
> **Bài tập 3**: Thay vì `add_node("node_one", node_one)`, thử `add_node(node_one)`.
> Tên hàm có tự động thành tên node không?

---

## 13.2 Graph State (10 phút)

### Khái niệm

> Bây giờ là phần quan trọng. **Cách node đọc và sửa state.**
>
> Quy tắc rất đơn giản:
> 1. Node nhận `state` làm đầu vào
> 2. Trả về dictionary để cập nhật state
> 3. **Mặc định là ghi đè (overwrite)**
>
> Các key không được trả về giữ nguyên giá trị trước đó.

### Giải thích code

> State giờ có hai trường: `hello` (chuỗi) và `a` (boolean).
>
> `node_one` cập nhật cả hai:
> ```python
> return {"hello": "from node one.", "a": True}
> ```
>
> `node_two` chỉ cập nhật `hello`. Không đụng đến `a`.
> Vậy `a` thì sao? **Giá trị trước đó `True` được giữ nguyên.**
>
> Chạy thử nào.

### Sau khi chạy

> Xem output:
> ```
> node_one {'hello': 'world'}           ← nhận input ban đầu
> node_two {'hello': 'from node one.', 'a': True}  ← đã được node_one cập nhật
> node_three {'hello': 'from node two.', 'a': True} ← a vẫn là True
> ```
>
> Thấy `a` vẫn `True` không? Vì `node_two`, `node_three` không trả về `a`.
> **Chỉ key được trả về mới bị ghi đè; phần còn lại giữ nguyên.** Đó là chiến lược mặc định.
>
> Kết quả cuối: `{'hello': 'from node three.', 'a': True}`

### Bài tập 13.2 (5 phút)

> **Bài tập 1**: Thêm `"a": False` vào input ban đầu. `node_one` nhìn thấy gì?
>
> **Bài tập 2**: Trả về key không có trong `State`. Ví dụ: `return {"unknown": 123}`
>
> **Bài tập 3**: Đổi `a` thành `False` trong `node_two`. Kiểm tra `node_three` thấy gì.

---

## 13.4 Multiple Schemas — Đa Schema (10 phút)

### Khái niệm

> Trong ứng dụng thực tế, các tình huống này xảy ra rất nhiều:
>
> - Input từ user khác với dữ liệu xử lý nội bộ
> - Output cuối chỉ nên hiển thị một phần state nội bộ
> - Một số node cần state riêng tư mà node khác không thấy
>
> LangGraph cho phép **tách ba schema**:
>
> | Tham số | Vai trò |
> |---------|---------|
> | Tham số đầu tiên | State nội bộ đầy đủ (Private) |
> | `input_schema` | Hình thái input bên ngoài |
> | `output_schema` | Hình thái output bên ngoài |

### Giải thích code

> `PrivateState` là nội bộ (`a`, `b`).
> `InputState` là input bên ngoài (`hello`).
> `OutputState` là output bên ngoài (`bye`).
>
> Mỗi node dùng **schema khác nhau làm type hint**.
> `node_one` chỉ thấy `InputState`, `node_two` thấy `PrivateState`.
>
> Chạy thử nào.

### Sau khi chạy

> Kiểm tra output:
> ```
> node_one -> {'hello': 'world'}       ← chỉ thấy InputState
> node_two -> {}                        ← PrivateState nhưng a, b chưa được set
> node_three -> {'a': 1}                ← node_two đã set a
> node_four -> {'a': 1, 'b': 1}        ← toàn bộ PrivateState
> {'secret': True}                      ← MegaPrivate
> ```
>
> **Kết quả cuối: `{'bye': 'world'}`**
>
> `a`, `b`, `secret` không xuất hiện trong output.
> Vì `OutputState` chỉ định nghĩa `bye`, nên chỉ trả về đó.
>
> Điều này quan trọng cho thiết kế API. Dữ liệu xử lý nội bộ không bị lộ ra ngoài.

### Bài tập 13.4 (5 phút)

> **Bài tập 1**: Bỏ `output_schema`. Giá trị trả về thay đổi thế nào?
>
> **Bài tập 2**: `invoke({"hello": "world", "extra": 123})` — điều gì xảy ra với field không tồn tại?
>
> **Bài tập 3**: Nghĩ xem tại sao điều này quan trọng từ góc độ bảo mật.

---

## 13.5 Reducer Functions (10 phút)

### Khái niệm

> Ở 13.2 chúng ta nói chiến lược mặc định là "ghi đè", đúng không?
>
> Nhưng hãy nghĩ về **tin nhắn chat**.
> Nếu tin nhắn trước biến mất mỗi khi có tin mới, thì hỏng rồi.
> Tin nhắn cần được **tích lũy**.
>
> Đó là thứ **Reducer** giải quyết.
>
> ```python
> messages: Annotated[list[str], operator.add]
> ```
>
> Dòng này có nghĩa:
> "Khi `messages` được cập nhật, đừng ghi đè — hãy **nối thêm** vào list hiện tại."
>
> `operator.add` thực hiện phép `+` cho list.

### Sau khi chạy

> Kết quả: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`
>
> Input ban đầu `["Hello!"]` được **kết hợp** với `["Hello, nice to meet you!"]` từ `node_one`.
>
> **Không có reducer?** Chỉ còn `["Hello, nice to meet you!"]`, còn `["Hello!"]` sẽ biến mất.
>
> Trong ứng dụng chat, reducer là bắt buộc — bạn cần xây dựng lịch sử hội thoại.

### Bài tập 13.5 (5 phút)

> **Bài tập 1**: Thêm tin nhắn trong `node_two`. Ba tin nhắn có tích lũy không?
>
> **Bài tập 2**: Bỏ `Annotated`, dùng `messages: list[str]` thuần. So sánh kết quả.
>
> **Bài tập 3**: Viết reducer tùy chỉnh. Ví dụ: loại bỏ trùng lặp, chỉ giữ 5 tin gần nhất.

---

## 13.6 Node Caching (7 phút)

### Khái niệm

> Một số node tốn chi phí chạy cao — gọi LLM, gọi API bên ngoài.
> Nếu cùng input cho cùng kết quả, **cache lại**.
>
> ```python
> cache_policy=CachePolicy(ttl=20)  # cache 20 giây
> ```
>
> `ttl` là Time-To-Live. Sau thời gian này, cache hết hạn và node chạy lại.

### Chạy cell

> `node_two` trả về thời gian hiện tại.
> Chạy 6 lần cách nhau 5 giây — tổng 30 giây.
> Với `ttl=20`:
>
> - **Lần 1–4** (0–20 giây): cùng thời gian — cache hit!
> - **Lần 5–6** (sau 20 giây): thời gian mới — cache hết hạn, chạy lại
>
> Cùng kiểm tra. (Mất khoảng 30 giây)

### Sau khi chạy

> Thấy không? Vài lần đầu hiện cùng thời gian, sau đó thay đổi.
> Rất hữu ích để giảm chi phí gọi API.

---

## 13.7 Conditional Edges (15 phút)

### Khái niệm

> Cho đến giờ, chúng ta chỉ xây **đồ thị tuyến tính**. A → B → C.
>
> Nhưng thực tế, bạn cần "đường đi khác nhau dựa trên điều kiện".
>
> `add_conditional_edges` xác định node tiếp theo dựa trên giá trị trả về của **hàm phân nhánh**.
>
> ```python
> add_conditional_edges(
>     node_nguồn,
>     hàm_phân_nhánh,         # kiểm tra state, trả về giá trị
>     {giá_trị: node_đích}    # giá trị trả về → ánh xạ node
> )
> ```

### Giải thích code

> Xem hàm `decide_path`:
>
> ```python
> def decide_path(state: State):
>     return state["seed"] % 2 == 0  # True hoặc False
> ```
>
> `seed` chẵn → `True`, lẻ → `False`.
>
> Ánh xạ:
> ```python
> {True: "node_one", False: "node_two"}
> ```
>
> Vậy:
> - seed=42 (chẵn) → `True` → `node_one` → `node_two` → ...
> - seed=7 (lẻ) → `False` → thẳng đến `node_two` → ...
>
> Sau `node_two`, lại có conditional edge nữa. Tái sử dụng cùng `decide_path`.

### Chạy — seed=42

> Số chẵn:
> ```
> node_one -> {'seed': 42}
> node_two -> {'seed': 42}
> node_three -> {'seed': 42}
> ```
> Đường đi: `START → node_one → node_two → node_three → END`.

### Chạy — seed=7

> Số lẻ:
> ```
> node_two -> {'seed': 7}
> node_four -> {'seed': 7}
> ```
> Đường đi: `START → node_two → node_four → END`.
>
> Cùng một đồ thị, nhưng đường đi hoàn toàn khác dựa trên input!
> Trực quan hóa đồ thị cho thấy rõ các nhánh.

### Bài tập 13.7 (5 phút)

> **Bài tập 1**: Thử nhiều giá trị `seed` khác nhau và quan sát đường đi thay đổi.
>
> **Bài tập 2**: Chuyển sang trả trực tiếp tên node:
> ```python
> def decide_path(state) -> Literal["node_three", "node_four"]:
>     if state["seed"] % 2 == 0:
>         return "node_three"
>     else:
>         return "node_four"
> ```
>
> **Bài tập 3**: Thiết kế conditional edge với 3 nhánh trở lên.

---

## 13.8 Send API (15 phút)

### Khái niệm

> Conditional edge quyết định "đi đâu".
> **Send API** đi xa hơn một bước:
>
> **Chạy cùng một node nhiều lần đồng thời với các input khác nhau.**
>
> ```python
> Send("node_two", word)  # chạy node_two với word làm input
> ```
>
> Khi hàm `dispatcher` trả về list các đối tượng `Send`,
> bấy nhiêu instance `node_two` sẽ chạy **song song**.
>
> Đây là pattern **Map-Reduce**:
> - Map: chia dữ liệu và xử lý từng phần
> - Reduce: thu thập và gộp kết quả (Reducer!)

### Giải thích code

> Điểm quan trọng:
>
> ```python
> def node_two(word: str):  # nhận str, không phải State!
> ```
>
> Giá trị truyền qua Send API là **input tùy chỉnh, không phải State**.
> Mỗi từ được truyền riêng lẻ cho `node_two`.
>
> Trường `output` có reducer `Annotated[..., operator.add]`,
> nên tất cả kết quả chạy song song tự động được gộp lại.
>
> `dispatcher`:
> ```python
> def dispatcher(state):
>     return [Send("node_two", word) for word in state["words"]]
> ```
> 6 từ → 6 đối tượng `Send` → 6 `node_two` chạy song song.

### Sau khi chạy

> Kết quả:
> ```
> hello -> 5 letters
> world -> 5 letters
> how   -> 3 letters
> are   -> 3 letters
> you   -> 3 letters
> doing -> 5 letters
> ```
>
> 6 từ được xử lý riêng lẻ, kết quả gộp vào list `output`.
>
> Ứng dụng thực tế?
> - Tóm tắt nhiều tài liệu đồng thời
> - Thu thập dữ liệu từ nhiều API cùng lúc
> - Xử lý song song nhiều yêu cầu từ user

### Bài tập 13.8 (5 phút)

> **Bài tập 1**: Tăng lên 20 từ.
>
> **Bài tập 2**: Thêm `time.sleep(1)` vào `node_two`. Tổng thời gian là 1 giây hay 6 giây?
>
> **Bài tập 3**: Tự triển khai một use case thực tế.

---

## 13.9 Command (10 phút)

### Khái niệm

> Cho đến giờ:
> - Cập nhật state = node trả về dictionary
> - Định tuyến = `add_conditional_edges` + hàm phân nhánh
>
> Hai thứ này **tách biệt**.
>
> **Command** gộp chúng **thành một**:
>
> ```python
> Command(
>     goto="account_support",       # đi đâu
>     update={"reason": "..."},     # cập nhật state
> )
> ```
>
> Bên trong node: "cập nhật state + quyết định node tiếp theo" trong một lần.
> Không cần `add_conditional_edges`, không cần hàm phân nhánh.

### Giải thích code

> Xem `triage_node`:
>
> ```python
> def triage_node(state) -> Command[Literal["account_support", "tech_support"]]:
>     return Command(
>         goto="account_support",
>         update={"transfer_reason": "The user wants to change password."},
>     )
> ```
>
> **Kiểu trả về `Command[Literal[...]]` là chìa khóa.**
> Type hint này cho LangGraph biết các đường đi có thể,
> nên đồ thị hoạt động mà không cần `add_edge`.
>
> Chú ý **không có edge** sau `triage_node` trong cấu hình đồ thị.
> Command xác định node tiếp theo tại thời điểm chạy.

### Sau khi chạy

> ```
> account_support running
> Result: {'transfer_reason': 'The user wants to change password.'}
> ```
>
> `triage_node` dùng Command để:
> 1. Cập nhật `transfer_reason`
> 2. Định tuyến đến `account_support`
>
> Trực quan hóa đồ thị cho thấy phân nhánh từ `triage_node` đến cả hai node.
> Chỉ nhờ type hint mà được như vậy.

### Bài tập 13.9 (5 phút)

> **Bài tập 1**: Thêm điều kiện để có thể định tuyến đến `tech_support`.
>
> **Bài tập 2**: `Command` vs `add_conditional_edges` — ưu nhược điểm?
>
> **Bài tập 3**: Triển khai định tuyến nhiều bước như hệ thống hỗ trợ khách hàng thực tế.

---

## Hướng dẫn bài tập tổng hợp (3 phút)

> Cuối notebook có 5 bài tập tổng hợp.
>
> **Bài 1** (★☆☆): Đồ thị counter — mỗi node tăng counter thêm 1
> **Bài 2** (★★☆): Chat simulator — tích lũy tin nhắn bằng reducer
> **Bài 3** (★★☆): Định tuyến theo tuổi — conditional edges
> **Bài 4** (★★★): Chuyển chữ hoa — Send API
> **Bài 5** (★★★): Router hỗ trợ — Command
>
> Bài 1–2 là cơ bản, bài 3–5 là thử thách.
> Phân bổ thời gian: 10 phút cho bài dễ, 15 phút cho bài khó.

---

## Kết thúc (3 phút)

> Tổng kết những gì chúng ta đã học hôm nay.
>
> | Khái niệm | Điểm chính |
> |------------|------------|
> | StateGraph | Xương sống của đồ thị dựa trên state |
> | Node / Edge | Việc cần làm / Thứ tự |
> | State | Mặc định ghi đè; Reducer cho phép tích lũy |
> | Multiple Schemas | Tách input/output/internal |
> | CachePolicy | Cache theo node để tối ưu hiệu suất |
> | Conditional Edges | Phân nhánh động dựa trên state |
> | Send API | Thực thi song song động (Map-Reduce) |
> | Command | Định tuyến + cập nhật state trong một lần |
>
> Đây là những building block cơ bản của LangGraph.
> Chương tiếp theo, chúng ta sẽ dùng chúng để xây dựng một **chatbot** thực tế.
>
> Cảm ơn các bạn. Làm tốt lắm.
