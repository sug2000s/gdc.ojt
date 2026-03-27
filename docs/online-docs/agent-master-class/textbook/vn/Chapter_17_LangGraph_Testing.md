# Chapter 17: Kiểm thử Workflow LangGraph

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ học **cách kiểm thử một cách có hệ thống các workflow AI agent được xây dựng bằng LangGraph**. Bắt đầu từ một đồ thị đơn giản dựa trên quy tắc, chúng ta chuyển sang các node dựa trên AI (LLM), sau đó từng bước tìm hiểu cách xác minh một cách đáng tin cậy các phản hồi AI không xác định (non-deterministic).

### Mục tiêu học tập

- Xây dựng workflow xử lý email sử dụng `StateGraph` của LangGraph
- Thiết lập framework kiểm thử đồ thị với `pytest`
- Kiểm thử đơn vị từng node riêng lẻ và kiểm thử thực thi một phần (Partial Execution)
- Hiểu quá trình chuyển đổi từ node dựa trên quy tắc sang node AI (LLM)
- Xây dựng chiến lược kiểm thử phù hợp với đặc tính không xác định của phản hồi AI
- Đánh giá chất lượng phản hồi AI sử dụng mô hình LLM-as-a-Judge

### Cấu trúc dự án

```
workflow-testing/
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py          # Định nghĩa workflow LangGraph
├── tests.py         # Mã kiểm thử pytest
└── README.md
```

---

## 2. Mô tả chi tiết từng phần

---

### 17.0 Introduction -- Thiết lập dự án ban đầu

**Chủ đề và Mục tiêu:** Thiết lập môi trường dự án Python cho các bài thực hành kiểm thử.

#### Các khái niệm chính

Trong phần này, chúng ta tạo một dự án Python mới sử dụng trình quản lý gói `uv`. Các dependency được định nghĩa trong `pyproject.toml`, và vai trò của các thư viện chính như sau:

| Gói | Phiên bản | Mục đích |
|-----|-----------|----------|
| `langchain[openai]` | 0.3.27 | Framework tích hợp LLM |
| `langgraph` | 0.6.6 | Bộ xây dựng đồ thị workflow |
| `langgraph-checkpoint-sqlite` | 2.0.11 | Lưu trữ checkpoint trạng thái |
| `pytest` | 8.4.2 | Framework kiểm thử Python |
| `python-dotenv` | 1.1.1 | Tải biến môi trường (.env) |

#### Phân tích mã nguồn

```toml
# pyproject.toml
[project]
name = "workflow-testing"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]
```

**Điểm cần lưu ý:**
- Yêu cầu Python 3.13 trở lên. File `.python-version` chỉ định `3.13`, vì vậy `uv` tự động sử dụng đúng phiên bản Python.
- `pytest` được bao gồm trong dependency của dự án. Điều này cho thấy kiểm thử là thành phần cốt lõi của quá trình phát triển.
- `grandalf` là thư viện để trực quan hóa đồ thị.

#### Điểm thực hành

1. Tạo dự án bằng lệnh `uv init workflow-testing`.
2. Thêm dependency bằng các lệnh như `uv add langgraph pytest langchain[openai]`.
3. Chạy `uv sync` để xác minh tất cả các gói được cài đặt chính xác.

---

### 17.1 Email Graph -- Xây dựng Workflow xử lý email

**Chủ đề và Mục tiêu:** Xây dựng workflow 3 bước phân loại email, gán mức ưu tiên và tạo phản hồi sử dụng `StateGraph` của LangGraph.

#### Các khái niệm chính

Workflow của LangGraph bao gồm ba yếu tố cốt lõi:

1. **State (Trạng thái):** Cấu trúc dữ liệu được chia sẻ trong toàn bộ workflow. Được định nghĩa bằng `TypedDict`, mỗi node đọc và ghi một phần của trạng thái.
2. **Node (Nút):** Hàm nhận trạng thái làm đầu vào, xử lý và trả về cập nhật trạng thái.
3. **Edge (Cạnh):** Kết nối định nghĩa thứ tự thực thi giữa các node.

Luồng workflow được xây dựng trong phần này:

```
START --> categorize_email --> assing_priority --> draft_response --> END
```

#### Phân tích mã nguồn

**Bước 1: Định nghĩa trạng thái**

```python
from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class EmailState(TypedDict):
    email: str
    category: Literal["spam", "normal", "urgent"]
    priority_score: int
    response: str
```

`EmailState` định nghĩa schema dữ liệu mà workflow sẽ xử lý. Kiểu `Literal` ràng buộc `category` chỉ có thể nhận một trong các giá trị "spam", "normal" hoặc "urgent". Trạng thái bao gồm email gốc (`email`), kết quả phân loại (`category`), điểm ưu tiên (`priority_score`) và phản hồi được tạo (`response`).

**Bước 2: Định nghĩa hàm node**

```python
def categorize_email(state: EmailState):
    email = state["email"].lower()

    if "urgent" in email or "asap" in email:
        category = "urgent"
    elif "offer" in email or "discount" in email:
        category = "spam"
    else:
        category = "normal"

    return {
        "category": category,
    }
```

Node `categorize_email` phân loại danh mục dựa trên từ khóa có trong nội dung email. Nó sử dụng logic dựa trên quy tắc đơn giản: nếu có "urgent" hoặc "asap" thì là khẩn cấp, nếu có "offer" hoặc "discount" thì là spam, còn lại là bình thường.

**Điểm chính:** Mỗi hàm node nhận toàn bộ trạng thái (`EmailState`) làm tham số nhưng trả về dictionary chỉ chứa các trường mà nó thay đổi. LangGraph sẽ **hợp nhất (merge)** giá trị trả về này vào trạng thái hiện có.

```python
def assing_priority(state: EmailState):
    scores = {
        "urgent": 10,
        "normal": 5,
        "spam": 1,
    }
    return {
        "priority_score": scores[state["category"]],
    }


def draft_response(state: EmailState) -> EmailState:
    responses = {
        "urgent": "I will answer you as fast as i can",
        "normal": "I'll get back to you soon",
        "spam": "Go away!",
    }
    return {
        "response": responses[state["category"]],
    }
```

`assing_priority` gán điểm cố định theo danh mục, và `draft_response` tạo thông điệp phản hồi cố định theo danh mục. Tại thời điểm này, tất cả logic đều xác định (deterministic) -- cùng một đầu vào luôn đảm bảo cùng một đầu ra.

**Bước 3: Lắp ráp và thực thi đồ thị**

```python
graph_builder = StateGraph(EmailState)

graph_builder.add_node("categorize_email", categorize_email)
graph_builder.add_node("assing_priority", assing_priority)
graph_builder.add_node("draft_response", draft_response)

graph_builder.add_edge(START, "categorize_email")
graph_builder.add_edge("categorize_email", "assing_priority")
graph_builder.add_edge("assing_priority", "draft_response")
graph_builder.add_edge("draft_response", END)

graph = graph_builder.compile()

result = graph.invoke({"email": "i have an offer for you!"})
print(result)
```

Các node được đăng ký trong `StateGraph`, edge chỉ định thứ tự, và `compile()` tạo ra đồ thị có thể thực thi. Truyền trạng thái ban đầu (nội dung email) vào `invoke()` sẽ chạy toàn bộ workflow tuần tự.

#### Điểm thực hành

1. Gọi `graph.invoke()` với nhiều văn bản email khác nhau ngoài "i have an offer for you!" và kiểm tra kết quả.
2. Suy nghĩ về những phần nào cần sửa đổi để thêm danh mục mới (ví dụ: "important").
3. Thử sử dụng phân nhánh có điều kiện (`add_conditional_edges`) để bỏ qua `draft_response` cho email spam.

---

### 17.2 Pytest -- Giới thiệu Framework kiểm thử

**Chủ đề và Mục tiêu:** Viết các bài kiểm thử tự động cho workflow LangGraph sử dụng `pytest`. Học kiểm thử tham số hóa với `@pytest.mark.parametrize`.

#### Các khái niệm chính

**pytest** là framework kiểm thử hàng đầu của Python. Các tính năng chính:

- Hàm có tên bắt đầu bằng `test_` tự động được nhận dạng là kiểm thử
- Xác minh ngắn gọn với câu lệnh `assert`
- `@pytest.mark.parametrize` chạy cùng một logic kiểm thử lặp lại với nhiều giá trị đầu vào khác nhau

**Kiểm thử tham số hóa (Parameterized Test)** là kỹ thuật then chốt để loại bỏ mã kiểm thử trùng lặp. Một hàm kiểm thử duy nhất có thể bao phủ nhiều kịch bản.

#### Phân tích mã nguồn

Đầu tiên, xóa mã thực thi trực tiếp (invoke + print) khỏi `main.py`:

```python
# Mã đã xóa (cuối main.py)
# result = graph.invoke({"email": "i have an offer for you!"})
# print(result)
```

Tách mã production và mã kiểm thử là nguyên tắc cơ bản. `main.py` chỉ chịu trách nhiệm định nghĩa đồ thị, trong khi thực thi và xác minh được thực hiện trong `tests.py`.

```python
# tests.py
import pytest
from main import graph


@pytest.mark.parametrize(
    "email, expected_category, expected_score",
    [
        ("this is urgent!", "urgent", 10),
        ("i wanna talk to you", "normal", 5),
        ("i have an offer for you", "spam", 1),
    ],
)
def test_full_graph(email, expected_category, expected_score):

    result = graph.invoke({"email": email})

    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

**Giải thích mã:**

1. `from main import graph`: Import đồ thị đã biên dịch từ `main.py`.
2. `@pytest.mark.parametrize`: Đối số đầu tiên của decorator là tên tham số (chuỗi phân tách bằng dấu phẩy), đối số thứ hai là danh sách các trường hợp kiểm thử.
3. Mỗi tuple `("this is urgent!", "urgent", 10)` đại diện cho một trường hợp kiểm thử.
4. `graph.invoke()` chạy toàn bộ workflow, và `assert` xác minh kết quả mong đợi.

Bài kiểm thử này chạy thành 3 trường hợp kiểm thử độc lập:
- Email khẩn cấp -> category="urgent", priority_score=10
- Email bình thường -> category="normal", priority_score=5
- Email spam -> category="spam", priority_score=1

#### Điểm thực hành

1. Chạy `pytest tests.py -v` trong terminal để xem từng trường hợp kiểm thử được thực thi riêng biệt. (`-v` là chế độ chi tiết)
2. Cố tình đặt giá trị mong đợi sai để xem thông báo lỗi. Tìm hiểu pytest cung cấp những thông tin gì.
3. Thêm các trường hợp biên: "URGENT offer" khi cả hai từ khóa đều xuất hiện thì kết quả là gì?

---

### 17.3 Testing Nodes -- Kiểm thử đơn vị node và thực thi một phần

**Chủ đề và Mục tiêu:** Ngoài việc thực thi toàn bộ đồ thị, học các kỹ thuật kiểm thử nâng cao: (1) kiểm thử từng node riêng biệt, và (2) chèn trạng thái trung gian vào đồ thị để thực thi một phần từ một điểm cụ thể.

#### Các khái niệm chính

Có ba cấp độ kiểm thử workflow:

| Cấp độ kiểm thử | Mô tả | Trường hợp sử dụng |
|-----------------|-------|---------------------|
| **Kiểm thử toàn bộ đồ thị** | Chạy từ đầu đến cuối với `graph.invoke()` | Kiểm thử tích hợp, xác minh E2E |
| **Kiểm thử node riêng lẻ** | Chạy một node cụ thể với `graph.nodes["tên_node"].invoke()` | Kiểm thử đơn vị, xác minh logic node |
| **Kiểm thử thực thi một phần** | Chèn trạng thái trung gian với `graph.update_state()` rồi tiếp tục thực thi | Tái tạo kịch bản cụ thể, gỡ lỗi |

**MemorySaver (Checkpointer):** Để thực thi một phần, đồ thị phải có khả năng lưu trạng thái. `MemorySaver` là checkpointer dựa trên bộ nhớ, lưu trạng thái thực thi của đồ thị theo từng `thread_id`.

#### Phân tích mã nguồn

**Thêm Checkpointer (main.py):**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# ... (định nghĩa node, edge bỏ qua) ...

graph = graph_builder.compile(checkpointer=checkpointer)
```

Khi `checkpointer` được truyền vào `compile()`, đồ thị tự động lưu trạng thái sau khi thực thi mỗi node. Sau đó bạn có thể truy vấn hoặc sửa đổi trạng thái của một lần thực thi cụ thể thông qua `thread_id`.

**Kiểm thử toàn bộ đồ thị đã sửa đổi:**

```python
def test_full_graph(email, expected_category, expected_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

Đồ thị sử dụng checkpointer bắt buộc phải cung cấp `thread_id` trong `config`.

**Kiểm thử node riêng lẻ:**

```python
def test_individual_nodes():

    # Chạy riêng node categorize_email
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # Chạy riêng node assing_priority
    result = graph.nodes["assing_priority"].invoke({"category": "spam"})
    assert result["priority_score"] == 1

    # Chạy riêng node draft_response
    result = graph.nodes["draft_response"].invoke({"category": "spam"})
    assert "Go away" in result["response"]
```

`graph.nodes` là dictionary chứa các node đã đăng ký. Mỗi node có phương thức `invoke()` và có thể được thực thi độc lập bằng cách chỉ truyền trạng thái cần thiết cho hàm node đó. Điều này có nghĩa:

- `categorize_email` chỉ cần trường `email`
- `assing_priority` chỉ cần trường `category`
- `draft_response` chỉ cần trường `category`

Bằng cách cô lập đầu vào và đầu ra của mỗi node để kiểm thử, bạn có thể nhanh chóng xác định node nào có lỗi khi vấn đề xảy ra.

**Kiểm thử thực thi một phần:**

```python
def test_partial_execution():

    # Bước 1: Chèn trực tiếp trạng thái trung gian
    graph.update_state(
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        values={
            "email": "please check out this offer",
            "category": "spam",
        },
        as_node="categorize_email",  # Đặt trạng thái như thể node này đã thực thi
    )

    # Bước 2: Tiếp tục thực thi từ trạng thái đã chèn
    result = graph.invoke(
        None,  # Tiếp tục từ trạng thái hiện có mà không có đầu vào mới
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        interrupt_after="draft_response",
    )

    assert result["priority_score"] == 1
```

Các hành vi chính của bài kiểm thử này:

1. `update_state()` chèn trạng thái như thể node `categorize_email` đã hoàn thành. `as_node="categorize_email"` có nghĩa là "trạng thái này là đầu ra của node categorize_email."
2. `graph.invoke(None, ...)` tiếp tục thực thi từ trạng thái đã lưu mà không có đầu vào. Nó bắt đầu từ `assing_priority`, node tiếp theo sau `categorize_email`.
3. `interrupt_after="draft_response"` dừng thực thi sau khi `draft_response` chạy.

Kỹ thuật này hữu ích trong các tình huống sau:
- Khi các node phía trước có chi phí thực thi cao (ví dụ: gọi LLM)
- Khi bạn chỉ muốn xác minh hành vi tại một trạng thái trung gian cụ thể
- Khi bạn cần tạo các trường hợp biên một cách nhân tạo

#### Điểm thực hành

1. In ra các key có trong `graph.nodes`.
2. Thay đổi `as_node` trong `update_state()` thành `"assing_priority"` và thử chạy từ một node muộn hơn.
3. Kiểm tra điều gì xảy ra khi gọi `invoke(None, ...)` với `thread_id` không tồn tại.

---

### 17.4 AI Nodes -- Chuyển đổi từ dựa trên quy tắc sang dựa trên LLM

**Chủ đề và Mục tiêu:** Thay thế các node dựa trên quy tắc cứng bằng các node AI sử dụng LLM (GPT-4o). Sử dụng `BaseModel` của Pydantic và `with_structured_output` của LangChain để cấu trúc đầu ra LLM.

#### Các khái niệm chính

Hạn chế của hệ thống dựa trên quy tắc:
- Không thể xử lý email khẩn cấp không chứa từ "urgent"
- Phải thêm điều kiện if/elif thủ công cho mỗi mẫu mới
- Khó bao phủ các biểu đạt đa dạng của ngôn ngữ tự nhiên

Sử dụng LLM cho phép phân loại dựa trên **ngữ nghĩa (semantics)** của ngôn ngữ tự nhiên. Tuy nhiên, vì đầu ra LLM là văn bản tự do, **Structured Output (Đầu ra có cấu trúc)** được sử dụng để ép buộc nó thành định dạng có thể xử lý lập trình.

**Mẫu Structured Output:**
1. Định nghĩa schema đầu ra mong muốn bằng Pydantic `BaseModel`
2. Bao bọc LLM bằng `llm.with_structured_output(Model)`
3. Ép buộc LLM luôn trả về JSON khớp với schema đó

#### Phân tích mã nguồn

**Khởi tạo LLM và Định nghĩa Schema đầu ra:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class EmailClassificationOuput(BaseModel):
    category: Literal["spam", "normal", "urgent"] = Field(
        description="Category of the email",
    )


class PriorityScoreOutput(BaseModel):
    priority_score: int = Field(
        description="Priority score from 1 to 10",
        ge=1,
        le=10,
    )
```

`EmailClassificationOuput` ép buộc LLM trả về một trong "spam", "normal" hoặc "urgent". `PriorityScoreOutput` bao gồm xác thực `ge` (lớn hơn hoặc bằng) và `le` (nhỏ hơn hoặc bằng) để đảm bảo trả về số nguyên từ 1 đến 10.

**categorize_email dựa trên AI:**

```python
def categorize_email(state: EmailState):
    s_llm = llm.with_structured_output(EmailClassificationOuput)

    result = s_llm.invoke(
        f"""Classify this email into one of three categories:
        - urgent: time-sensitive, requires immediate attention
        - normal: regular business communication
        - spam: promotional, marketing, or unwanted content

        Email: {state['email']}"""
    )

    return {
        "category": result.category,
    }
```

Thay vì so khớp từ khóa, tiêu chí phân loại được truyền cho LLM qua prompt. `s_llm` được trả về bởi `with_structured_output()` luôn trả về instance `EmailClassificationOuput`. Bạn có thể truy cập giá trị một cách an toàn về kiểu dữ liệu với `result.category`.

**assing_priority dựa trên AI:**

```python
def assing_priority(state: EmailState):
    s_llm = llm.with_structured_output(PriorityScoreOutput)

    result = s_llm.invoke(
        f"""Assign a priority score from 1-10 for this {state['category']} email.
        Consider:
        - Category: {state['category']}
        - Email content: {state['email']}

        Guidelines:
        - Urgent emails: usually 8-10
        - Normal emails: usually 4-7
        - Spam emails: usually 1-3"""
    )

    return {"priority_score": result.priority_score}
```

Thay vì ánh xạ cố định (`urgent=10, normal=5, spam=1`), LLM xem xét tổng hợp nội dung email và danh mục để gán điểm linh hoạt trong phạm vi 1-10. Prompt bao gồm hướng dẫn chỉ ra phạm vi điểm theo từng danh mục.

**draft_response dựa trên AI:**

```python
def draft_response(state: EmailState) -> EmailState:
    result = llm.invoke(
        f"""Draft a brief, professional response for this {state['category']} email.

        Original email: {state['email']}
        Category: {state['category']}
        Priority: {state['priority_score']}/10

        Guidelines:
        - Urgent: Acknowledge urgency, promise immediate attention
        - Normal: Professional acknowledgment, standard timeline
        - Spam: Brief notice that message was filtered

        Keep response under 2 sentences."""
    )
    return {
        "response": result.content,
    }
```

Node này sử dụng LLM thông thường không có structured output, vì phản hồi là văn bản tự do. Phản hồi văn bản của LLM được truy cập qua `result.content`.

**So sánh dựa trên quy tắc vs dựa trên AI:**

| Hạng mục | Dựa trên quy tắc (17.1) | Dựa trên AI (17.4) |
|----------|-------------------------|---------------------|
| Phương pháp phân loại | So khớp từ khóa | Hiểu ngữ nghĩa |
| Gán điểm | Giá trị cố định theo danh mục | Giá trị linh hoạt dựa trên ngữ cảnh |
| Tạo phản hồi | Mẫu cố định | Tạo động |
| Tính xác định | Xác định (cùng đầu vào = cùng đầu ra) | Không xác định (cùng đầu vào có thể cho đầu ra khác) |
| Độ khó kiểm thử | Dễ (so sánh giá trị chính xác) | Khó (cần so sánh phạm vi, ngữ nghĩa) |

#### Điểm thực hành

1. Kiểm thử với email như "Please help me, my server is down and clients are complaining!" - khẩn cấp nhưng không chứa từ khóa "urgent". So sánh sự khác biệt giữa kết quả dựa trên quy tắc và dựa trên AI.
2. Thêm danh mục mới (ví dụ: "inquiry") vào `EmailClassificationOuput` và sửa đổi prompt.
3. Thay đổi phạm vi `ge`, `le` trong `Field` để xác minh rằng xác thực Pydantic hoạt động.

---

### 17.5 Testing AI Nodes -- Chiến lược kiểm thử cho node AI

**Chủ đề và Mục tiêu:** Học các chiến lược kiểm thử hiệu quả cho đầu ra không xác định của các node dựa trên AI (LLM). Chuyển từ so sánh giá trị chính xác sang so sánh dựa trên phạm vi.

#### Các khái niệm chính

Giới thiệu node AI làm hỏng các bài kiểm thử hiện có. Lý do:

1. **Phân loại danh mục:** LLM phân loại đúng, nhưng có thể có sự khác biệt diễn giải nhỏ ngay cả với cùng đầu vào.
2. **Điểm ưu tiên:** Trả về phạm vi (ví dụ: 8-10) thay vì giá trị cố định (ví dụ: 10).
3. **Văn bản phản hồi:** Các câu khác nhau được tạo ra mỗi lần.

Do đó, nguyên tắc chính cho kiểm thử node AI là:

> **Xác minh phạm vi chấp nhận được thay vì giá trị chính xác.**

#### Phân tích mã nguồn

**Thêm tải biến môi trường:**

```python
import dotenv
dotenv.load_dotenv()
```

Vì node AI gọi API OpenAI, `OPENAI_API_KEY` phải được tải từ file `.env`. Quan trọng là mã này phải được đặt **ở đầu file**. Khi `from main import graph` được thực thi, `init_chat_model("openai:gpt-4o")` trong `main.py` được gọi, nên biến môi trường phải được tải trước đó.

**Kiểm thử toàn bộ đồ thị -- Chuyển sang dựa trên phạm vi:**

```python
@pytest.mark.parametrize(
    "email, expected_category, min_score, max_score",
    [
        ("this is urgent!", "urgent", 8, 10),
        ("i wanna talk to you", "normal", 4, 7),
        ("i have an offer for you", "spam", 1, 3),
    ],
)
def test_full_graph(email, expected_category, min_score, max_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert min_score <= result["priority_score"] <= max_score
```

Các thay đổi:
- Phạm vi `min_score` và `max_score` thay thế `expected_score` đơn lẻ
- `assert min_score <= result["priority_score"] <= max_score` thay thế `assert result["priority_score"] == expected_score`
- Danh mục vẫn có thể so sánh chính xác vì được ép buộc bởi Structured Output

**Kiểm thử node riêng lẻ đã sửa đổi:**

```python
def test_individual_nodes():

    # categorize_email -- vẫn cho phép so sánh giá trị chính xác
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # assing_priority -- chuyển sang so sánh phạm vi, thêm trường email
    result = graph.nodes["assing_priority"].invoke(
        {"category": "spam", "email": "buy this pot."}
    )
    assert 1 <= result["priority_score"] <= 3

    # draft_response -- đã comment (chưa có phương pháp xác minh phù hợp)
    # result = graph.nodes["draft_response"].invoke({"category": "spam"})
    # assert "Go away" in result["response"]
```

Điểm cần lưu ý:
- Trường `email` đã được thêm vào `assing_priority`. Phiên bản AI cũng tham chiếu nội dung email trong prompt.
- `draft_response` bị comment. Vì AI tạo phản hồi khác nhau mỗi lần, xác minh từ khóa như `"Go away" in result["response"]` không khả thi. Vấn đề này được giải quyết ở 17.6.

**Kiểm thử thực thi một phần đã sửa đổi:**

```python
def test_partial_execution():
    # ... (phần update_state giống nhau) ...

    result = graph.invoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after="draft_response",
    )
    assert 1 <= result["priority_score"] <= 3  # Phạm vi thay vì giá trị cố định 1
```

#### Điểm thực hành

1. Cố tình đặt phạm vi hẹp (ví dụ: `min_score=10, max_score=10`) và quan sát tần suất kiểm thử AI thất bại.
2. Chạy cùng bài kiểm thử 10 lần liên tiếp và kiểm tra phân bố kết quả: `pytest tests.py -v --count=10` (cần plugin pytest-repeat)
3. Kiểm tra đầu ra node `draft_response` nhiều lần và suy nghĩ về phương pháp xác minh nào sẽ phù hợp.

---

### 17.6 Testing AI Responses -- Mẫu LLM-as-a-Judge

**Chủ đề và Mục tiêu:** Triển khai mẫu **LLM-as-a-Judge (LLM làm giám khảo)** để xác minh chất lượng phản hồi AI dạng tự do. Kiểm thử văn bản do AI tạo thông qua đánh giá độ tương đồng dựa trên ví dụ.

#### Các khái niệm chính

Lý do kiểm thử `draft_response` bị comment ở 17.5 là vì AI tạo ra văn bản khác nhau mỗi lần. So khớp chuỗi chính xác như "Go away!" là không thể.

Mẫu **LLM-as-a-Judge** là kỹ thuật tiêu biểu để giải quyết vấn đề này:

1. Định nghĩa trước các **golden examples (ví dụ mẫu)** phản hồi lý tưởng cho mỗi danh mục.
2. Truyền phản hồi AI cần kiểm thử cùng với các ví dụ cho **LLM giám khảo**.
3. LLM giám khảo trả về điểm tương đồng (similarity score).
4. Nếu điểm trên ngưỡng, bài kiểm thử đạt.

Ưu điểm của mẫu này:
- Có thể đánh giá ngữ nghĩa văn bản dạng tự do
- Tiêu chí đánh giá có thể dễ dàng điều chỉnh bằng cách thêm/sửa đổi ví dụ
- Linh hoạt và bền vững hơn so khớp từ khóa

#### Phân tích mã nguồn

**Schema đầu ra điểm tương đồng:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class SimilarityScoreOutput(BaseModel):
    similarity_score: int = Field(
        description="How similar is the response to the examples?",
        gt=0,
        lt=100,
    )
```

Định nghĩa schema để LLM giám khảo trả về điểm tương đồng từ 0 đến 100. `gt=0, lt=100` ép buộc phạm vi 1-99, loại trừ 0 và 100.

**Ví dụ phản hồi (Golden Examples):**

```python
RESPONSE_EXAMPLES = {
    "urgent": [
        "Thank you for your urgent message. We are addressing this immediately and will respond as soon as possible.",
        "We've received your urgent request and are prioritizing it. Our team is on it right away.",
        "This urgent matter has our immediate attention. We'll respond promptly.",
    ],
    "normal": [
        "Thank you for your email. We'll review it and get back to you within 24-48 hours.",
        "We've received your message and will respond soon. Thank you for reaching out.",
        "Thank you for contacting us. We'll process your request and respond shortly.",
        "Thank you for the update. I will review the information and follow up as needed.",
        "Thank you for the update on the project status. I will review and follow up by the end of the week.",
        "Thanks for sharing this update. We'll review and respond accordingly.",
    ],
    "spam": [
        "This message has been flagged as spam and filtered.",
        "This email has been identified as promotional content.",
        "This message has been marked as spam.",
    ],
}
```

Nhiều ví dụ cho mỗi danh mục cung cấp "phản hồi mong muốn trông như thế nào." Ví dụ càng đa dạng, độ chính xác đánh giá càng cao. Danh mục `normal` có nhiều ví dụ nhất vì phản hồi cho email thông thường có thể đa dạng nhất.

**Hàm giám khảo (Judge Function):**

```python
def judge_response(response: str, category: str):

    s_llm = llm.with_structured_output(SimilarityScoreOutput)

    examples = RESPONSE_EXAMPLES[category]
    result = s_llm.invoke(
        f"""
        Score how similar this response is to the examples.

        Category: {category}

        Examples:
        {"\n".join(examples)}

        Response to evaluate:
        {response}

        Scoring criteria:
        - 90-100: Very similar in tone, content, and intent
        - 70-89: Similar with minor differences
        - 50-69: Moderately similar, captures main idea
        - 30-49: Some similarity but missing key elements
        - 0-29: Very different or inappropriate
    """
    )

    return result.similarity_score
```

Cách hàm `judge_response` hoạt động:

1. Lấy các ví dụ phù hợp với danh mục.
2. Truyền cả ví dụ và phản hồi cần đánh giá cho LLM giám khảo.
3. Bao gồm tiêu chí đánh giá rõ ràng (rubric) trong prompt.
4. Nhận điểm số nguyên qua structured output.

**Sử dụng trong mã kiểm thử:**

```python
def test_individual_nodes():

    # ... (kiểm thử categorize_email, assing_priority không đổi) ...

    # draft_response -- xác minh bằng LLM-as-a-Judge
    result = graph.nodes["draft_response"].invoke(
        {
            "category": "spam",
            "email": "Get rich quick!!! I have a pyramid scheme for you!",
            "priority_score": 1,
        }
    )

    similarity_score = judge_response(result["response"], "spam")
    assert similarity_score >= 70
```

Trạng thái đầy đủ (bao gồm category, email và priority_score) được truyền cho node `draft_response`, sau đó phản hồi được tạo được truyền cho `judge_response` để đánh giá tương đồng. Bài kiểm thử đạt nếu ngưỡng từ 70 trở lên.

**Ý nghĩa của ngưỡng 70:**
- Theo rubric, 70-89 nghĩa là "tương tự với khác biệt nhỏ"
- Nếu quá cao (ví dụ: 90), kiểm thử trở nên không ổn định do khác biệt biểu đạt nhỏ
- Nếu quá thấp (ví dụ: 40), phản hồi chất lượng thấp cũng đạt
- 70 là điểm cân bằng cho phép "ý đồ và giọng điệu phù hợp nhưng biểu đạt có thể khác"

#### Điểm thực hành

1. Đặt ngưỡng lần lượt là 50, 70 và 90 và quan sát sự thay đổi độ ổn định kiểm thử.
2. Thêm/xóa ví dụ trong `RESPONSE_EXAMPLES` và xem kết quả giám định thay đổi như thế nào.
3. Thêm kiểm thử áp dụng `judge_response` cho danh mục `urgent` và `normal`.
4. Thay thế LLM giám khảo bằng mô hình khác (ví dụ: `gpt-4o-mini`) và thử nghiệm đánh đổi chi phí-độ chính xác.

---

## 3. Tổng kết trọng tâm chương

### Quá trình phát triển chiến lược kiểm thử

```
17.1 Đồ thị dựa trên quy tắc    -->  17.2 Kiểm thử so sánh giá trị chính xác
        |                                      |
17.4 Đồ thị dựa trên AI          -->  17.5 Kiểm thử dựa trên phạm vi
        |                                      |
                                       17.6 Kiểm thử LLM-as-a-Judge
```

### Tóm tắt nguyên tắc chính

| Nguyên tắc | Mô tả |
|------------|-------|
| **Tách biệt cấp độ kiểm thử** | Kiểm thử toàn bộ đồ thị, node riêng lẻ và thực thi một phần một cách riêng biệt. |
| **So sánh chính xác đầu ra xác định** | Danh mục được ép buộc bởi Structured Output có thể so sánh bằng `==`. |
| **So sánh đầu ra không xác định bằng phạm vi** | Số được LLM tạo được xác minh bằng `min <= value <= max`. |
| **Dùng LLM giám định văn bản tự do** | Sử dụng LLM khác làm giám khảo để đánh giá tương đồng ngữ nghĩa. |
| **Golden Examples** | Định nghĩa trước các ví dụ phản hồi lý tưởng để sử dụng làm tiêu chí đánh giá. |
| **Nêu rõ tiêu chí đánh giá trong prompt** | Truyền đạt rõ ràng ý nghĩa các khoảng điểm cho LLM giám khảo. |
| **Thực thi một phần với checkpointer** | Sử dụng `MemorySaver` và `update_state()` để kiểm thử từ các điểm cụ thể. |

### Tổng kết công nghệ sử dụng

| Công nghệ | Mục đích |
|-----------|----------|
| `langgraph.StateGraph` | Định nghĩa đồ thị workflow |
| `langgraph.checkpoint.memory.MemorySaver` | Checkpointing trạng thái trong bộ nhớ |
| `pytest` + `@pytest.mark.parametrize` | Kiểm thử tham số hóa |
| `pydantic.BaseModel` + `Field` | Định nghĩa và xác thực schema đầu ra LLM |
| `langchain.chat_models.init_chat_model` | Khởi tạo LLM |
| `llm.with_structured_output()` | Ép buộc đầu ra có cấu trúc |

---

## 4. Bài tập thực hành

### Bài tập 1: Thêm danh mục mới (Độ khó: Trung bình)

Thêm danh mục `"inquiry"` (yêu cầu thông tin) vào phân loại email.

- Thêm `"inquiry"` vào `category` Literal trong `EmailState`
- Phản ánh điều này trong `EmailClassificationOuput`
- Thêm hướng dẫn "inquiry: questions or information requests" vào prompt phân loại
- Thêm "Inquiry emails: usually 5-7" vào prompt `PriorityScoreOutput`
- Thêm 3 ví dụ trở lên cho danh mục `"inquiry"` trong `RESPONSE_EXAMPLES`
- Thêm trường hợp kiểm thử cho danh mục mới vào `parametrize` của `test_full_graph`

### Bài tập 2: Đồ thị phân nhánh có điều kiện (Độ khó: Trung bình)

Sửa đổi đồ thị để email spam bỏ qua node `draft_response` và đi thẳng đến END.

- Sử dụng `add_conditional_edges` để phân nhánh sau `assing_priority` dựa trên danh mục
- Nếu là spam, đi đến END; nếu không, đi đến `draft_response`
- Cập nhật mã kiểm thử cho phù hợp với thay đổi này. Đối với email spam, trường `response` không nên tồn tại.

### Bài tập 3: Nâng cao LLM giám khảo (Độ khó: Cao)

Cải thiện `judge_response` hiện tại.

- Thay vì điểm tương đồng đơn, tạo schema đánh giá nhiều chiều (giọng điệu, tính chuyên nghiệp, tính phù hợp, độ dài) riêng biệt.
- Tính điểm cuối cùng bằng trung bình điểm mỗi chiều.
- Cho phép trọng số khác nhau theo từng chiều (ví dụ: tính phù hợp 40%, giọng điệu 30%, tính chuyên nghiệp 20%, độ dài 10%).
- Khi kiểm thử thất bại, xuất ra chiều nào có điểm thấp.

### Bài tập 4: Phân tích độ ổn định kiểm thử (Độ khó: Cao)

Chạy cùng bài kiểm thử 20 lần để phân tích độ ổn định kiểm thử AI.

- Cài đặt plugin `pytest-repeat`.
- Chạy với `pytest tests.py --count=20 -v`.
- Thống kê tỷ lệ đạt/không đạt cho mỗi bài kiểm thử.
- Nếu có trường hợp thất bại, phân tích nguyên nhân (vấn đề ngưỡng, vấn đề prompt, hay vấn đề mô hình).
- Viết báo cáo về những điều chỉnh cần thiết để nâng độ ổn định kiểm thử trên 95%.
