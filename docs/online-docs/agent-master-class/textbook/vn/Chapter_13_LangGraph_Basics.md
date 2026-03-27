# Chapter 13: Cơ bản LangGraph (LangGraph Fundamentals)

## Tổng quan chương

Trong chương này, chúng ta học từng bước từ đầu các kiến thức cơ bản về **LangGraph**, một framework cốt lõi trong hệ sinh thái LangChain. LangGraph là framework cho phép thiết kế và thực thi các ứng dụng dựa trên LLM theo cấu trúc **đồ thị (Graph)**, giúp xây dựng các workflow tác tử AI phức tạp một cách trực quan.

Qua chương này, bạn sẽ học:

- Thiết lập ban đầu và cấu hình môi trường cho dự án LangGraph
- Cấu trúc cơ bản của đồ thị: Node và Edge
- Quản lý State đồ thị và truyền dữ liệu giữa các node
- Tách biệt đầu vào/đầu ra sử dụng Multiple Schemas
- Chiến lược hợp nhất trạng thái qua hàm Reducer
- Tối ưu hiệu suất sử dụng Node Caching
- Điều khiển luồng động qua Conditional Edges
- Xử lý song song động sử dụng Send API
- Định tuyến nội bộ node sử dụng đối tượng Command

### Môi trường dự án

| Mục | Phiên bản/Chi tiết |
|-----|-------------------|
| Python | >= 3.13 |
| LangGraph | >= 0.6.6 |
| LangChain | >= 0.3.27 (bao gồm OpenAI) |
| Công cụ phát triển | Jupyter Notebook (ipykernel) |
| Quản lý gói | uv (dựa trên pyproject.toml) |

---

## 13.0 Introduction - Thiết lập ban đầu dự án

### Chủ đề và mục tiêu
Tạo dự án Python mới để học LangGraph và cài đặt các phụ thuộc cần thiết.

### Khái niệm chính

Để bắt đầu dự án LangGraph, chúng ta thiết lập thư mục dự án mới có tên `hello-langgraph`. Dự án này sử dụng trình quản lý gói **uv** và thực hành được tiến hành trong môi trường Jupyter Notebook.

#### Phụ thuộc dự án (`pyproject.toml`)

```toml
[project]
name = "hello-langgraph"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf>=0.8",
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.6",
    "langgraph-checkpoint-sqlite>=2.0.11",
    "langgraph-cli[inmem]>=0.4.0",
    "python-dotenv>=1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Vai trò của từng phụ thuộc:

| Gói | Vai trò |
|-----|---------|
| `langgraph` | Framework workflow dựa trên đồ thị (cốt lõi) |
| `langchain[openai]` | Tích hợp LangChain và OpenAI |
| `grandalf` | Hỗ trợ trực quan hóa đồ thị |
| `langgraph-checkpoint-sqlite` | Lưu trữ checkpoint trạng thái (SQLite) |
| `langgraph-cli[inmem]` | Công cụ CLI LangGraph (chế độ in-memory) |
| `python-dotenv` | Quản lý biến môi trường (tệp .env) |
| `ipykernel` | Kernel Jupyter Notebook (chỉ dùng cho phát triển) |

### Điểm thực hành
- Khởi tạo dự án sử dụng `uv` và cài đặt phụ thuộc.
- Sử dụng `.gitignore` để loại trừ môi trường ảo, tệp cache, v.v. khỏi quản lý phiên bản.
- Xác nhận môi trường Jupyter Notebook hoạt động bình thường.

---

## 13.1 Your First Graph - Xây dựng đồ thị đầu tiên

### Chủ đề và mục tiêu
Hiểu các thành phần cơ bản nhất của LangGraph -- **StateGraph**, **Node**, và **Edge** -- và xây dựng đồ thị đầu tiên.

### Khái niệm chính

Đồ thị trong LangGraph gồm ba yếu tố cốt lõi:

1. **State (Trạng thái)**: Cấu trúc dữ liệu chia sẻ trên toàn bộ đồ thị. Định nghĩa sử dụng `TypedDict`.
2. **Node (Nút)**: Hàm riêng lẻ được thực thi trong đồ thị. Mỗi node nhận state làm đầu vào.
3. **Edge (Cạnh)**: Kết nối giữa các node. Xác định thứ tự thực thi.

LangGraph cũng cung cấp hai node đặc biệt:
- **`START`**: Điểm vào của đồ thị. Chỉ ra node nào thực thi đầu tiên.
- **`END`**: Điểm ra của đồ thị. Node kết nối đây thực thi cuối cùng.

### Phân tích mã

#### Bước 1: Import và định nghĩa State

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    hello: str

graph_builder = StateGraph(State)
```

- `StateGraph` là lớp cốt lõi để tạo đồ thị dựa trên trạng thái.
- `State` kế thừa `TypedDict` để định nghĩa schema trạng thái sử dụng trong đồ thị.
- `graph_builder` là instance của `StateGraph`, qua đó chúng ta thêm node và edge.

#### Bước 2: Định nghĩa hàm Node

```python
def node_one(state: State):
    print("node_one")

def node_two(state: State):
    print("node_two")

def node_three(state: State):
    print("node_three")
```

- Mỗi hàm node phải nhận `state` làm tham số.
- Ở giai đoạn này, chúng ta chưa sửa đổi state, chỉ in ra để xác nhận thực thi.

#### Bước 3: Xây dựng đồ thị

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

Mã này tạo đồ thị tuyến tính sau:

```
START -> node_one -> node_two -> node_three -> END
```

- `add_node(tên, hàm)`: Đăng ký node vào đồ thị.
- `add_edge(nguồn, đích)`: Thêm edge kết nối hai node.

### Điểm thực hành
- Kiểm tra lỗi gì xảy ra khi bỏ `START` và `END`.
- Thay đổi thứ tự node và quan sát luồng thực thi thay đổi ra sao.
- Kiểm tra xem tên hàm có tự động được sử dụng khi bỏ đối số đầu tiên (tên chuỗi) trong `add_node` không.

---

## 13.2 Graph State - Quản lý trạng thái đồ thị

### Chủ đề và mục tiêu
Hiểu cách node **đọc và sửa đổi** state, và theo dõi state thay đổi như thế nào khi đi qua các node.

### Khái niệm chính

State là cốt lõi của thực thi đồ thị trong LangGraph. Mỗi node:

1. Nhận state hiện tại làm **đầu vào**.
2. **Trả về** dictionary để cập nhật state.
3. Giá trị trả về **ghi đè** state hiện có (hành vi mặc định).

Điểm quan trọng là giá trị của các khóa mà node không trả về vẫn được giữ nguyên.

### Phân tích mã

#### Mở rộng định nghĩa State

```python
class State(TypedDict):
    hello: str
    a: bool

graph_builder = StateGraph(State)
```

State giờ có hai trường: `hello` (chuỗi) và `a` (boolean).

#### Đọc và sửa đổi State trong Node

```python
def node_one(state: State):
    print("node_one", state)
    return {
        "hello": "from node one.",
        "a": True,
    }

def node_two(state: State):
    print("node_two", state)
    return {"hello": "from node two."}

def node_three(state: State):
    print("node_three", state)
    return {"hello": "from node three."}
```

Điểm chính:
- `node_one` cập nhật cả hai trường `hello` và `a`.
- `node_two` chỉ cập nhật `hello`. Giá trị trước của `a` (`True`) được giữ.
- `node_three` cũng chỉ cập nhật `hello`. `a` vẫn là `True`.

#### Biên dịch và thực thi đồ thị

```python
graph = graph_builder.compile()

result = graph.invoke(
    {
        "hello": "world",
    },
)
```

- `compile()`: Biên dịch graph builder thành đồ thị thực thi được.
- `invoke()`: Truyền state ban đầu để thực thi đồ thị.

#### Theo dõi kết quả thực thi

```
node_one {'hello': 'world'}
node_two {'hello': 'from node one.', 'a': True}
node_three {'hello': 'from node two.', 'a': True}
```

| Thời điểm | hello | a |
|-----------|-------|---|
| Đầu vào ban đầu | `"world"` | (không có) |
| Sau node_one | `"from node one."` | `True` |
| Sau node_two | `"from node two."` | `True` |
| Sau node_three | `"from node three."` | `True` |

Kết quả cuối: `{'hello': 'from node three.', 'a': True}`

**Chiến lược cập nhật state mặc định là "ghi đè".** Giá trị node trả về thay thế giá trị hiện có cho khóa đó. Các khóa không được trả về vẫn giữ nguyên.

### Điểm thực hành
- Thử bao gồm giá trị `a` trong đầu vào ban đầu và xem nó xuất hiện thế nào trong node.
- Kiểm tra điều gì xảy ra khi node trả về khóa không tồn tại trong `state`.
- In trực tiếp đối tượng `graph` để xem sơ đồ trực quan của đồ thị.

---

## 13.4 Multiple Schemas - Đa Schema

### Chủ đề và mục tiêu
Học cách tách biệt **schema đầu vào**, **schema đầu ra**, và **schema nội bộ (Private)** trong một đồ thị duy nhất.

### Khái niệm chính

Trong ứng dụng thực tế, các yêu cầu sau thường phát sinh:

- Dạng **dữ liệu đầu vào** nhận từ người dùng khác với dữ liệu xử lý nội bộ.
- Dữ liệu cuối cùng **trả về người dùng** chỉ nên là một phần của state nội bộ.
- Một số node cần **state riêng tư** mà chỉ chúng truy cập được.

LangGraph giải quyết điều này bằng cách chỉ định ba schema cho `StateGraph`:

| Tham số | Vai trò |
|---------|---------|
| Đối số đầu tiên (State) | State đầy đủ nội bộ (Private State) |
| `input_schema` | Dạng đầu vào truyền từ bên ngoài vào đồ thị |
| `output_schema` | Dạng đầu ra đồ thị trả về bên ngoài |

### Phân tích mã

#### Định nghĩa đa Schema

```python
class PrivateState(TypedDict):
    a: int
    b: int

class InputState(TypedDict):
    hello: str

class OutputState(TypedDict):
    bye: str

class MegaPrivate(TypedDict):
    secret: bool

graph_builder = StateGraph(
    PrivateState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

Trong cấu hình này:
- Bên ngoài chỉ có thể cung cấp đầu vào dạng `{"hello": "world"}`.
- Nội bộ sử dụng trường `a`, `b` cho tính toán.
- Đầu ra cuối chỉ trả về dạng `{"bye": "world"}`.
- `MegaPrivate` là state siêu riêng tư chỉ được sử dụng bởi node cụ thể.

#### Các Node sử dụng Schema khác nhau

```python
def node_one(state: InputState) -> InputState:
    print("node_one ->", state)
    return {"hello": "world"}

def node_two(state: PrivateState) -> PrivateState:
    print("node_two ->", state)
    return {"a": 1}

def node_three(state: PrivateState) -> PrivateState:
    print("node_three ->", state)
    return {"b": 1}

def node_four(state: PrivateState) -> OutputState:
    print("node_four ->", state)
    return {"bye": "world"}

def node_five(state: OutputState):
    return {"secret": True}

def node_six(state: MegaPrivate):
    print(state)
```

Lưu ý mỗi node sử dụng schema khác nhau làm type hint. Điều này biểu đạt rõ ràng **mỗi node quan tâm đến dữ liệu nào**.

#### Xây dựng và thực thi đồ thị

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)
graph_builder.add_node("node_five", node_five)
graph_builder.add_node("node_six", node_six)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", "node_four")
graph_builder.add_edge("node_four", "node_five")
graph_builder.add_edge("node_five", "node_six")
graph_builder.add_edge("node_six", END)
```

#### Phân tích kết quả thực thi

```
node_one -> {'hello': 'world'}
node_two -> {}
node_three -> {'a': 1}
node_four -> {'a': 1, 'b': 1}
{'secret': True}
```

Giá trị trả về cuối: `{'bye': 'world'}`

Quan sát chính:
- `node_one` chỉ thấy `InputState`, nên nhận `{'hello': 'world'}`.
- `node_two` thấy `PrivateState`, nhưng vì `a`, `b` chưa được đặt nên là `{}`.
- `node_three` thấy `{'a': 1}` mà `node_two` đã đặt.
- `node_four` thấy toàn bộ PrivateState `{'a': 1, 'b': 1}`.
- **Đầu ra cuối chỉ chứa trường `bye` được định nghĩa trong `OutputState`**. State nội bộ (`a`, `b`, `secret`) không bị lộ ra bên ngoài.

### Điểm thực hành
- Kiểm tra giá trị trả về thay đổi thế nào khi không chỉ định `output_schema`.
- Kiểm tra điều gì xảy ra khi truyền trường không tồn tại trong `input_schema` vào `invoke()`.
- Suy nghĩ tại sao tách biệt schema quan trọng trong môi trường production thực tế (bảo mật, thiết kế API, v.v.).

---

## 13.5 Reducer Functions - Hàm Reducer

### Chủ đề và mục tiêu
Học cách **tích lũy** state sử dụng **hàm Reducer** thay vì chiến lược "ghi đè" mặc định.

### Khái niệm chính

Mặc định, cập nhật state của LangGraph **thay thế hoàn toàn** giá trị trước bằng giá trị mới. Tuy nhiên trong nhiều trường hợp (đặc biệt lịch sử tin nhắn chat), giá trị cần được **tích lũy**.

**Hàm Reducer** sử dụng type hint `Annotated` để tùy chỉnh chiến lược cập nhật cho trường cụ thể:

```
Annotated[kiểu, hàm_reducer]
```

Hàm reducer nhận hai đối số:
- `old`: Giá trị hiện tại của state
- `new`: Giá trị mới node trả về

Và trả về **giá trị cuối cùng được lưu**.

### Phân tích mã

#### Định nghĩa hàm Reducer và áp dụng vào State

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
import operator

def update_function(old, new):
    return old + new

class State(TypedDict):
    # messages: Annotated[list[str], update_function]
    messages: Annotated[list[str], operator.add]

graph_builder = StateGraph(State)
```

Điểm chính:
- `Annotated[list[str], operator.add]` có nghĩa "khi trường `messages` được cập nhật, **nối thêm** list mới vào list hiện có".
- `operator.add` là hàm tích hợp Python, thực hiện phép toán `+` (nối) trên list.
- `update_function` đã comment là hàm reducer tùy chỉnh có hành vi tương tự. Bạn có thể tự tạo hoặc dùng hàm có sẵn như `operator.add`.

#### Định nghĩa Node

```python
def node_one(state: State):
    return {
        "messages": ["Hello, nice to meet you!"],
    }

def node_two(state: State):
    return {}

def node_three(state: State):
    return {}
```

- Chỉ `node_one` thêm mục mới vào `messages`.
- `node_two` và `node_three` trả về dictionary rỗng, nên không thay đổi state.

#### Thực thi và kết quả

```python
graph = graph_builder.compile()

graph.invoke(
    {"messages": ["Hello!"]},
)
```

Kết quả: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`

| Thời điểm | messages |
|-----------|----------|
| Đầu vào ban đầu | `["Hello!"]` |
| Sau node_one | `["Hello!"] + ["Hello, nice to meet you!"]` = `["Hello!", "Hello, nice to meet you!"]` |
| node_two, node_three | Không thay đổi |

**Nếu không có reducer**, giá trị trả về của `node_one` `["Hello, nice to meet you!"]` sẽ thay thế hoàn toàn giá trị ban đầu `["Hello!"]`. Nhờ reducer, hai list đã được **kết hợp**.

### Điểm thực hành
- Viết hàm reducer tùy chỉnh (ví dụ: chỉ giữ giá trị lớn nhất, loại bỏ trùng lặp, v.v.).
- Thêm tin nhắn trong `node_two` và xác nhận tích lũy hoạt động đúng.
- Chạy cùng mã không có reducer và so sánh kết quả khác nhau thế nào.
- Suy nghĩ tại sao reducer là cần thiết trong ứng dụng chat.

---

## 13.6 Node Caching - Cache Node

### Chủ đề và mục tiêu
Học cách cache kết quả thực thi node cụ thể sử dụng **CachePolicy** và sử dụng giá trị đã cache mà không cần tính lại trong khoảng thời gian nhất định.

### Khái niệm chính

Một số node có chi phí thực thi cao (ví dụ: gọi API bên ngoài, gọi LLM), hoặc trả về cùng kết quả cho cùng đầu vào. Trong các trường hợp này, **caching** có thể tối ưu hiệu suất.

LangGraph cung cấp chính sách cache theo node:

| Thành phần | Vai trò |
|-----------|---------|
| `CachePolicy(ttl=giây)` | Đặt thời gian cache hợp lệ (Time-To-Live) tính bằng giây |
| `InMemoryCache()` | Kho cache dựa trên bộ nhớ |
| `graph_builder.compile(cache=...)` | Kết nối kho cache khi biên dịch |

### Phân tích mã

#### Import và định nghĩa State

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from datetime import datetime

class State(TypedDict):
    time: str

graph_builder = StateGraph(State)
```

#### Định nghĩa Node - Node đích cache

```python
def node_one(state: State):
    return {}

def node_two(state: State):
    return {"time": f"{datetime.now()}"}

def node_three(state: State):
    return {}
```

`node_two` trả về thời gian hiện tại. Khi cache được áp dụng, thời gian đã ghi trước đó được trả về nguyên trạng trong khoảng TTL.

#### Áp dụng chính sách Cache

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node(
    "node_two",
    node_two,
    cache_policy=CachePolicy(ttl=20),  # Cache trong 20 giây
)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

Điểm chính: Chỉ `node_two` được chỉ định `cache_policy=CachePolicy(ttl=20)`. Kết quả node này được **cache trong 20 giây**.

#### Biên dịch với Cache và thực thi lặp

```python
import time

graph = graph_builder.compile(cache=InMemoryCache())

print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
```

Mã này chạy đồ thị 6 lần cách nhau 5 giây. Vì `node_two` có `ttl=20`:

- **0~20 giây đầu**: Kết quả thực thi đầu tiên (thời gian) được cache và cùng thời gian được trả về.
- **Sau 20 giây**: Cache hết hạn, `node_two` chạy lại và ghi thời gian mới.

### Điểm thực hành
- Thay đổi giá trị `ttl` và quan sát thời điểm cache hết hạn.
- Nghiên cứu xem có thể sử dụng kho cache khác thay vì `InMemoryCache` không.
- Suy nghĩ về các tình huống thực tế mà caching hữu ích (ví dụ: giới hạn tốc độ API bên ngoài, giảm chi phí).
- Cũng suy nghĩ về trường hợp caching có thể gây vấn đề (ví dụ: khi cần dữ liệu thời gian thực).

---

## 13.7 Conditional Edges - Cạnh có điều kiện

### Chủ đề và mục tiêu
Triển khai logic phân nhánh **chọn node tiếp theo động** dựa trên state sử dụng **Conditional Edges**.

### Khái niệm chính

Cho đến nay, tất cả đường đi trong đồ thị đều cố định (luồng tuyến tính). Tuy nhiên trong ứng dụng thực tế, thường cần chọn đường đi khác nhau dựa trên state.

Phương thức **`add_conditional_edges`** cho phép:
1. Thực thi **hàm định tuyến (routing function)** sau node cụ thể.
2. Xác định node tiếp theo động dựa trên giá trị trả về của hàm định tuyến.

```
add_conditional_edges(
    node_nguồn,
    hàm_định_tuyến,
    dictionary_ánh_xạ   # {giá_trị_trả_về: node_đích}
)
```

### Phân tích mã

#### Định nghĩa State và Node

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal

class State(TypedDict):
    seed: int

graph_builder = StateGraph(State)

def node_one(state: State):
    print("node_one ->", state)
    return {}

def node_two(state: State):
    print("node_two ->", state)
    return {}

def node_three(state: State):
    print("node_three ->", state)
    return {}

def node_four(state: State):
    print("node_four ->", state)
    return {}
```

#### Định nghĩa hàm định tuyến

Mã cho thấy hai cách tiếp cận hàm định tuyến:

**Cách 1: Trả về chuỗi (đã comment)**
```python
# def decide_path(state: State) -> Literal["node_three", "node_four"]:
#     if state["seed"] % 2 == 0:
#         return "node_three"
#     else:
#         return "node_four"
```
Cách này trả về tên node trực tiếp. Type hint `Literal` chỉ rõ các giá trị trả về có thể.

**Cách 2: Trả về giá trị tùy ý + dictionary ánh xạ (thực tế sử dụng)**
```python
def decide_path(state: State):
    return state["seed"] % 2 == 0  # Trả về True hoặc False
```
Hàm định tuyến trả về giá trị tùy ý như `True`/`False`, và dictionary ánh xạ chuyển đổi chúng thành node thực tế.

#### Xây dựng cạnh có điều kiện

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)

# Phân nhánh có điều kiện từ START
graph_builder.add_conditional_edges(
    START,
    decide_path,
    {
        True: "node_one",     # Nếu seed chẵn, đến node_one
        False: "node_two",    # Nếu seed lẻ, đến node_two
        "hello": END,         # Nếu trả về "hello", kết thúc
    },
)

graph_builder.add_edge("node_one", "node_two")

# Phân nhánh có điều kiện từ node_two
graph_builder.add_conditional_edges(
    "node_two",
    decide_path,
    {
        True: "node_three",
        False: "node_four",
        "hello": END,
    },
)

graph_builder.add_edge("node_four", END)
graph_builder.add_edge("node_three", END)
```

Luồng đồ thị này:

```
             ┌─ True ──> node_one ──> node_two ─┬─ True ──> node_three ──> END
START ───────┤                                  ├─ False ─> node_four ───> END
             ├─ False ─> node_two ──────────────┘
             └─ "hello" ──> END
```

### Điểm thực hành
- Thay đổi giá trị `seed` đa dạng và quan sát đường thực thi khác nhau thế nào.
- Chuyển sang cách trả về tên node trực tiếp (Cách 1) không có dictionary ánh xạ.
- Thiết kế cạnh có điều kiện với 3 nhánh trở lên.
- Tạo workflow phức tạp bằng cách kết hợp cạnh có điều kiện và cạnh thường.

---

## 13.8 Send API - Xử lý song song động

### Chủ đề và mục tiêu
Học cách **tạo instance node động** khi chạy và **thực thi chúng song song** sử dụng **Send API**.

### Khái niệm chính

Cạnh có điều kiện quyết định "đi đến node nào", nhưng **Send API** tiến xa hơn một bước:

1. Có thể thực thi **cùng node nhiều lần** song song.
2. Mỗi instance có thể nhận **đầu vào khác nhau**.
3. **Số lượng instance được xác định khi chạy**.

Điều này tương tự mẫu Map-Reduce:
- **Map**: Chia dữ liệu và áp dụng cùng xử lý cho từng phần
- **Reduce**: Gom kết quả lại (sử dụng kết hợp với hàm Reducer)

### Phân tích mã

#### Import và định nghĩa State

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.types import Send
import operator
from typing import Union

class State(TypedDict):
    words: list[str]
    output: Annotated[list[dict[str, Union[str, int]]], operator.add]

graph_builder = StateGraph(State)
```

Điểm chính:
- `words`: Danh sách từ cần xử lý
- `output`: List **tích lũy** kết quả xử lý cho mỗi từ. Sử dụng `Annotated` với reducer `operator.add` để kết hợp kết quả.
- `Send` được import. Đây là cốt lõi của xử lý song song động.

#### Định nghĩa Node

```python
def node_one(state: State):
    print(f"I want to count {len(state['words'])} words in my state.")

def node_two(word: str):
    return {
        "output": [
            {
                "word": word,
                "letters": len(word),
            }
        ]
    }
```

Sự khác biệt quan trọng:
- `node_one` là node thông thường nhận toàn bộ `State`.
- **`node_two` nhận `word` (chuỗi) riêng lẻ thay vì `State`.** Đây là đầu vào tùy chỉnh truyền qua Send API.
- `node_two` thêm kết quả vào list `output`. Nhờ reducer (`operator.add`), kết quả từ tất cả thực thi song song tự động được kết hợp.

#### Hàm Dispatcher và xây dựng đồ thị

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)

def dispatcher(state: State):
    return [Send("node_two", word) for word in state["words"]]

graph_builder.add_edge(START, "node_one")
graph_builder.add_conditional_edges("node_one", dispatcher, ["node_two"])
graph_builder.add_edge("node_two", END)
```

Phân tích chính:

1. **Hàm `dispatcher`**: Tạo đối tượng `Send` cho mỗi từ trong `state["words"]`.
   - `Send("node_two", word)`: "Thực thi `node_two` với `word` làm đầu vào"
   - Vì trả về list, `node_two` được thực thi song song **nhiều lần bằng số từ**.

2. **Truyền list cho `add_conditional_edges`**: `["node_two"]` là danh sách node đích có thể.

#### Kết quả thực thi

```python
graph.invoke(
    {
        "words": ["hello", "world", "how", "are", "you", "doing"],
    }
)
```

Đầu ra:
```
I want to count 6 words in my state.
```

Kết quả:
```python
{
    'words': ['hello', 'world', 'how', 'are', 'you', 'doing'],
    'output': [
        {'word': 'hello', 'letters': 5},
        {'word': 'world', 'letters': 5},
        {'word': 'how', 'letters': 3},
        {'word': 'are', 'letters': 3},
        {'word': 'you', 'letters': 3},
        {'word': 'doing', 'letters': 5}
    ]
}
```

6 instance của `node_two` mỗi cái xử lý một từ, và kết quả được kết hợp vào list `output` bởi reducer `operator.add`.

### Điểm thực hành
- Tăng kích thước danh sách từ và quan sát sự khác biệt hiệu suất.
- Thêm `time.sleep` vào `node_two` để cảm nhận hiệu quả thực thi song song.
- Viết mã tạo cùng kết quả mà không dùng Send API và so sánh.
- Suy nghĩ về trường hợp sử dụng thực tế (ví dụ: tóm tắt nhiều tài liệu đồng thời, thu thập dữ liệu từ nhiều nguồn, v.v.).

---

## 13.9 Command - Đối tượng Command

### Chủ đề và mục tiêu
Học cách thực hiện **cập nhật state và định tuyến đồng thời** từ bên trong node sử dụng đối tượng **Command**.

### Khái niệm chính

Trong các phương pháp đã học:
- Cập nhật state: Node trả về dictionary
- Định tuyến: `add_conditional_edges` + hàm định tuyến riêng

Hai điều này **tách biệt**. Đối tượng **Command** **thống nhất** chúng:

```python
Command(
    goto="node_đích",             # Node tiếp theo để đi đến
    update={"key": "value"},      # Cập nhật state
)
```

Ưu điểm của cách này:
- Logic định tuyến nằm bên trong node, trực quan hơn.
- Cập nhật state và định tuyến được xử lý nguyên tử (atomic).
- Không cần hàm định tuyến riêng hoặc cạnh có điều kiện.

### Phân tích mã

#### Import và định nghĩa State

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import Command

class State(TypedDict):
    transfer_reason: str

graph_builder = StateGraph(State)
```

#### Định nghĩa Node - Node Router trả về Command

```python
from typing import Literal

def triage_node(state: State) -> Command[Literal["account_support", "tech_support"]]:
    return Command(
        goto="account_support",
        update={
            "transfer_reason": "The user wants to change password.",
        },
    )

def tech_support(state: State):
    return {}

def account_support(state: State):
    print("account_support running")
    return {}
```

Phân tích chính:

1. **Kiểu trả về của `triage_node`**: `Command[Literal["account_support", "tech_support"]]`
   - Chỉ ra qua kiểu rằng node trả về `Command` và đích có thể là `"account_support"` hoặc `"tech_support"`.
   - Nhờ type hint này, LangGraph biết đường đi có thể **mà không cần `add_edge` hoặc `add_conditional_edges`**.

2. **Đối tượng `Command`**:
   - `goto="account_support"`: Di chuyển đến node `account_support` tiếp theo
   - `update={"transfer_reason": "The user wants to change password."}`: Cập nhật `transfer_reason` trong state

#### Xây dựng đồ thị

```python
graph_builder.add_node("triage_node", triage_node)
graph_builder.add_node("tech_support", tech_support)
graph_builder.add_node("account_support", account_support)

graph_builder.add_edge(START, "triage_node")
# Không cần add_edge sau triage_node! Command xử lý định tuyến.

graph_builder.add_edge("tech_support", END)
graph_builder.add_edge("account_support", END)
```

Lưu ý: Không có edge nào được định nghĩa sau `triage_node`. `goto` của đối tượng `Command` xác định node tiếp theo khi chạy.

Cấu trúc đồ thị:
```
                         ┌──> tech_support ────> END
START ──> triage_node ───┤
                         └──> account_support ──> END
```

#### Kết quả thực thi

```python
graph = graph_builder.compile()
graph.invoke({})
```

Đầu ra:
```
account_support running
```

Kết quả: `{'transfer_reason': 'The user wants to change password.'}`

`triage_node` dùng `Command` để:
1. Cập nhật `transfer_reason` và
2. Định tuyến đến `account_support`.

### Điểm thực hành
- Sửa `triage_node` để định tuyến đến `tech_support` dựa trên điều kiện.
- So sánh ưu nhược điểm của `Command` và `add_conditional_edges`.
- Triển khai định tuyến nhiều giai đoạn như hệ thống hỗ trợ khách hàng thực tế sử dụng `Command`.
- Nghiên cứu xem `goto` trong `Command` có thể chỉ định nhiều node không.

---

## Tóm tắt chương (Key Takeaways)

### 1. Cấu trúc cơ bản LangGraph
- **StateGraph**: Lớp cốt lõi cho đồ thị dựa trên trạng thái
- **Node**: Hàm nhận state, xử lý và trả về cập nhật
- **Edge**: Kết nối giữa các node (xác định thứ tự thực thi)
- **START / END**: Điểm vào và ra của đồ thị

### 2. Quản lý State
- Schema state được định nghĩa bằng `TypedDict`.
- Chiến lược cập nhật mặc định là **ghi đè (overwrite)**.
- Sử dụng `Annotated` với hàm reducer cho phép áp dụng chiến lược **tích lũy (accumulate)**.
- `operator.add` là reducer được sử dụng phổ biến nhất cho nối list.

### 3. Đa Schema
- `input_schema`: Hạn chế dạng đầu vào bên ngoài
- `output_schema`: Hạn chế dạng đầu ra bên ngoài
- State nội bộ không bị lộ ra bên ngoài, có lợi cho bảo mật và thiết kế API.

### 4. Caching
- Đặt chính sách cache theo node với `CachePolicy(ttl=giây)`.
- Kích hoạt với `InMemoryCache()` và `compile(cache=...)`.
- Có thể cải thiện đáng kể hiệu suất cho các phép toán tốn kém (gọi API, v.v.).

### 5. Điều khiển luồng
| Phương thức | Đặc điểm | Khi nào sử dụng |
|------------|----------|----------------|
| `add_edge` | Đường đi cố định | Luôn cùng node tiếp theo |
| `add_conditional_edges` | Định tuyến động dựa trên hàm | Thay đổi đường đi dựa trên state |
| `Send` API | Thực thi song song động | Chạy cùng node với đầu vào khác nhiều lần |
| `Command` | Định tuyến nội bộ node + cập nhật state | Xử lý định tuyến và thay đổi state cùng lúc |

### 6. Nguyên tắc thiết kế cốt lõi
- Đồ thị được xây dựng **khai báo (declarative)**: Định nghĩa node và edge trước, biên dịch và thực thi sau.
- State được xử lý như thể **bất biến (immutable)**: Node trả về dictionary mới để cập nhật state.
- **Tách biệt mối quan tâm**: Mỗi node chỉ có một trách nhiệm.

---

## Bài tập thực hành

### Bài tập 1: Đồ thị cơ bản (Độ khó: thấp)

Tạo đồ thị tuyến tính với 4 node (`start_node`, `process_a`, `process_b`, `end_node`). Thêm trường `counter: int` vào state và cho mỗi node tăng `counter` lên 1. Giá trị `counter` cuối cùng phải là 4.

**Gợi ý**: Không có reducer thì sẽ ghi đè. Đọc giá trị hiện tại trong mỗi node và trả về giá trị +1.

### Bài tập 2: Tích lũy tin nhắn chat (Độ khó: trung bình)

Tạo trình mô phỏng chat đơn giản sử dụng reducer:
- State: `messages: Annotated[list[str], operator.add]`
- `user_node`: Thêm `["Người dùng: Xin chào"]`
- `assistant_node`: Thêm `["Trợ lý: Tôi có thể giúp gì?"]`
- `user_reply_node`: Thêm `["Người dùng: Cho tôi biết thời tiết"]`

`messages` cuối cùng phải chứa 3 tin nhắn theo thứ tự.

### Bài tập 3: Định tuyến có điều kiện (Độ khó: trung bình)

Tạo đồ thị phân nhánh theo đường đi khác nhau dựa trên tuổi người dùng:
- State: `age: int`, `message: str`
- Sau node `check_age`, phân nhánh có điều kiện:
  - Dưới 18: `minor_node` -> "Bạn là vị thành niên."
  - 18 đến dưới 65: `adult_node` -> "Bạn là người trưởng thành."
  - 65 trở lên: `senior_node` -> "Bạn đủ điều kiện ưu đãi người cao tuổi."

### Bài tập 4: Sử dụng Send API (Độ khó: cao)

Tạo đồ thị nhận câu đầu vào và chuyển mỗi từ thành chữ hoa đồng thời:
- State: `sentence: str`, `results: Annotated[list[str], operator.add]`
- `splitter_node`: Tách câu thành các từ
- `upper_node`: Chuyển từ riêng lẻ thành chữ hoa (thực thi song song qua Send API)
- Đầu vào: `{"sentence": "hello world from langgraph"}`
- Đầu ra mong đợi: `{"sentence": "...", "results": ["HELLO", "WORLD", "FROM", "LANGGRAPH"]}`

### Bài tập 5: Tác tử dựa trên Command (Độ khó: cao)

Triển khai router hỗ trợ khách hàng đơn giản sử dụng đối tượng Command:
- State: `query: str`, `department: str`, `response: str`
- `router_node`: Định tuyến với Command dựa trên nội dung truy vấn
  - Chứa "hoàn tiền" hoặc "thanh toán" -> `billing_node`
  - Chứa "lỗi" hoặc "bug" -> `tech_node`
  - Còn lại -> `general_node`
- Mỗi node phòng ban đặt thông báo hướng dẫn phù hợp trong `response`

**Bonus**: Sử dụng type hint của `Command` (`Command[Literal[...]]`) chính xác để tất cả đường đi có thể hiển thị khi trực quan hóa đồ thị.
