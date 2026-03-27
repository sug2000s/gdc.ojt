# Chapter 13: LangGraph Co Ban (LangGraph Fundamentals)

## Tong quan chuong

Trong chuong nay, chung ta se hoc tung buoc tu dau cac kien thuc co ban ve **LangGraph** - framework cot loi cua he sinh thai LangChain. LangGraph la framework cho phep thiet ke va thuc thi ung dung dua tren LLM theo cau truc **do thi (Graph)**, giup xay dung cac workflow AI agent phuc tap mot cach truc quan.

Qua chuong nay, ban se hoc duoc:

- Thiet lap ban dau va cau hinh moi truong du an LangGraph
- Cau truc co ban cua do thi (Graph): Node va Edge
- Quan ly trang thai (State) cua do thi va truyen du lieu giua cac node
- Su dung da schema (Multiple Schemas) de tach biet dau vao/dau ra
- Chien luoc gop trang thai bang ham Reducer
- Toi uu hieu suat bang Node Caching
- Dieu khien luong dong bang Conditional Edges
- Xu ly song song dong voi Send API
- Dinh tuyen ben trong node bang doi tuong Command

### Moi truong du an

| Hang muc | Phien ban/Noi dung |
|----------|---------------------|
| Python | >= 3.13 |
| LangGraph | >= 0.6.6 |
| LangChain | >= 0.3.27 (bao gom OpenAI) |
| Cong cu phat trien | Jupyter Notebook (ipykernel) |
| Quan ly goi | uv (dua tren pyproject.toml) |

---

## 13.0 Introduction - Thiet lap ban dau du an

### Chu de va Muc tieu
Tao du an Python moi de hoc LangGraph va cai dat cac phu thuoc can thiet.

### Giai thich khai niem cot loi

De bat dau du an LangGraph, chung ta tao thu muc du an moi co ten `hello-langgraph`. Du an nay su dung **uv** package manager va thuc hanh trong moi truong Jupyter Notebook.

#### Phu thuoc du an (`pyproject.toml`)

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

Vai tro cua tung phu thuoc:

| Goi | Vai tro |
|-----|---------|
| `langgraph` | Framework workflow dua tren do thi (cot loi) |
| `langchain[openai]` | Tich hop LangChain va OpenAI |
| `grandalf` | Ho tro truc quan hoa do thi |
| `langgraph-checkpoint-sqlite` | Luu checkpoint trang thai (SQLite) |
| `langgraph-cli[inmem]` | Cong cu CLI LangGraph (che do in-memory) |
| `python-dotenv` | Quan ly bien moi truong (file .env) |
| `ipykernel` | Kernel Jupyter Notebook (chi cho phat trien) |

### Diem thuc hanh
- Su dung `uv` de khoi tao du an va cai dat phu thuoc.
- Dung `.gitignore` de loai tru moi truong ao, file cache khoi version control.
- Kiem tra moi truong Jupyter Notebook hoat dong binh thuong.

---

## 13.1 Your First Graph - Tao do thi dau tien

### Chu de va Muc tieu
Hieu cac thanh phan co ban nhat cua LangGraph: **StateGraph**, **Node**, **Edge** va xay dung do thi dau tien.

### Giai thich khai niem cot loi

Do thi trong LangGraph gom ba yeu to cot loi:

1. **State (trang thai)**: Cau truc du lieu duoc chia se tren toan bo do thi. Dinh nghia bang `TypedDict`.
2. **Node (nut)**: Cac ham rieng le duoc thuc thi trong do thi. Moi node nhan trang thai lam dau vao.
3. **Edge (canh)**: Ket noi giua cac node. Xac dinh thu tu thuc thi.

LangGraph cung cung cap hai node dac biet:
- **`START`**: Diem bat dau do thi. Cho biet node nao chay dau tien.
- **`END`**: Diem ket thuc do thi. Node ket noi voi day se chay cuoi cung.

### Phan tich code

#### Buoc 1: Import va dinh nghia trang thai

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    hello: str

graph_builder = StateGraph(State)
```

- `StateGraph` la lop cot loi de tao do thi dua tren trang thai.
- `State` ke thua `TypedDict` de dinh nghia schema trang thai su dung trong do thi.
- `graph_builder` la instance cua `StateGraph`, qua do them node va edge.

#### Buoc 2: Dinh nghia ham node

```python
def node_one(state: State):
    print("node_one")

def node_two(state: State):
    print("node_two")

def node_three(state: State):
    print("node_three")
```

- Moi ham node bat buoc nhan `state` lam tham so.
- O buoc nay chua sua doi trang thai, chi in ra de xac nhan thuc thi.

#### Buoc 3: Cau hinh do thi

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

Code nay tao do thi tuyen tinh:

```
START -> node_one -> node_two -> node_three -> END
```

- `add_node(ten, ham)`: Dang ky node vao do thi.
- `add_edge(xuat_phat, dich)`: Them edge ket noi hai node.

### Diem thuc hanh
- Bo `START` va `END` xem loi gi xay ra.
- Thay doi thu tu node va quan sat luong thuc thi thay doi the nao.
- Thu bo tham so ten (chuoi) trong `add_node` xem ten ham co tu dong duoc su dung khong.

---

## 13.2 Graph State - Quan ly trang thai do thi

### Chu de va Muc tieu
Hieu cach node **doc va sua doi** trang thai, va theo doi trang thai thay doi the nao khi di qua cac node.

### Giai thich khai niem cot loi

Trang thai (State) la cot loi cua viec thuc thi do thi trong LangGraph. Moi node:

1. Nhan trang thai hien tai lam **dau vao**.
2. Tra ve dictionary de **cap nhat** trang thai.
3. Gia tri tra ve se **ghi de (overwrite)** len trang thai hien tai (hanh vi mac dinh).

Diem quan trong la gia tri cua cac key ma node khong tra ve se duoc giu nguyen.

### Phan tich code

#### Mo rong dinh nghia trang thai

```python
class State(TypedDict):
    hello: str
    a: bool

graph_builder = StateGraph(State)
```

Bay gio trang thai co hai truong: `hello` (chuoi) va `a` (boolean).

#### Doc va sua doi trang thai trong node

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

Diem cot loi:
- `node_one` cap nhat ca hai truong `hello` va `a`.
- `node_two` chi cap nhat `hello`. `a` giu nguyen gia tri truoc (`True`).
- `node_three` cung chi cap nhat `hello`. `a` van la `True`.

#### Bien dich va thuc thi do thi

```python
graph = graph_builder.compile()

result = graph.invoke(
    {
        "hello": "world",
    },
)
```

- `compile()`: Bien dich graph builder thanh do thi co the thuc thi.
- `invoke()`: Truyen trang thai ban dau va thuc thi do thi.

#### Theo doi ket qua thuc thi

```
node_one {'hello': 'world'}
node_two {'hello': 'from node one.', 'a': True}
node_three {'hello': 'from node two.', 'a': True}
```

| Thoi diem | hello | a |
|-----------|-------|---|
| Dau vao ban dau | `"world"` | (khong co) |
| Sau khi node_one chay | `"from node one."` | `True` |
| Sau khi node_two chay | `"from node two."` | `True` |
| Sau khi node_three chay | `"from node three."` | `True` |

Ket qua cuoi cung: `{'hello': 'from node three.', 'a': True}`

**Chien luoc cap nhat trang thai mac dinh la "ghi de".** Gia tri cua key ma node tra ve se thay the gia tri cu. Key khong tra ve se duoc giu nguyen.

### Diem thuc hanh
- Thu truyen gia tri `a` trong dau vao ban dau. No hien thi the nao trong node?
- Thu tra ve key khong co trong `state` tu node, xem dieu gi xay ra.
- In truc tiep doi tuong `graph` de xem so do do thi truc quan.

---

## 13.4 Multiple Schemas - Da schema

### Chu de va Muc tieu
Hoc cach tach biet **schema dau vao**, **schema dau ra**, va **schema noi bo (Private)** trong mot do thi.

### Giai thich khai niem cot loi

Trong ung dung thuc te, cac yeu cau sau thuong xuat hien:

- Hinh thuc **du lieu dau vao** tu nguoi dung khac voi du lieu xu ly noi bo.
- **Du lieu tra ve** cuoi cung cho nguoi dung chi nen la mot phan cua trang thai noi bo.
- Can **trang thai bi mat** ma chi mot so node truy cap duoc.

LangGraph giai quyet bang cach chi dinh ba schema cho `StateGraph`:

| Tham so | Vai tro |
|---------|---------|
| Tham so dau tien (State) | Trang thai noi bo tong the (Private State) |
| `input_schema` | Hinh thuc dau vao tu ben ngoai |
| `output_schema` | Hinh thuc dau ra tra ve ben ngoai |

### Phan tich code

#### Dinh nghia da schema

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

Trong cau hinh nay:
- Ben ngoai chi co the truyen dau vao dang `{"hello": "world"}`.
- Noi bo su dung cac truong `a`, `b` de tinh toan.
- Dau ra cuoi cung chi tra ve dang `{"bye": "world"}`.
- `MegaPrivate` la trang thai sieu bi mat chi mot so node su dung.

#### Cac node su dung cac schema khac nhau

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

Chu y rang moi node su dung type hint schema khac nhau. Dieu nay the hien ro rang **moi node quan tam den du lieu nao**.

#### Cau hinh va thuc thi do thi

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

#### Phan tich ket qua thuc thi

```
node_one -> {'hello': 'world'}
node_two -> {}
node_three -> {'a': 1}
node_four -> {'a': 1, 'b': 1}
{'secret': True}
```

Gia tri tra ve cuoi cung: `{'bye': 'world'}`

Quan sat cot loi:
- `node_one` chi thay `InputState` nen nhan `{'hello': 'world'}`.
- `node_two` thay `PrivateState` nhung `a`, `b` chua duoc dat nen nhan `{}`.
- `node_three` thay `{'a': 1}` ma `node_two` da dat.
- `node_four` thay toan bo PrivateState `{'a': 1, 'b': 1}`.
- **Dau ra cuoi cung chi chua truong `bye` dinh nghia trong `OutputState`**. Trang thai noi bo (`a`, `b`, `secret`) khong bi lo ra ben ngoai.

### Diem thuc hanh
- Thu bo `output_schema` va xem gia tri tra ve thay doi the nao.
- Thu truyen truong khong co trong `input_schema` vao `invoke()` va xem dieu gi xay ra.
- Suy nghi tai sao tach biet schema quan trong trong moi truong production (bao mat, thiet ke API...).

---

## 13.5 Reducer Functions - Ham Reducer

### Chu de va Muc tieu
Hoc cach su dung **ham Reducer** de **tich luy (accumulate)** trang thai thay vi chien luoc "ghi de" mac dinh.

### Giai thich khai niem cot loi

Mac dinh, cap nhat trang thai trong LangGraph la gia tri moi **thay the hoan toan** gia tri cu. Nhung trong nhieu truong hop (dac biet la lich su tin nhan chat) can **tich luy** gia tri.

**Ham Reducer** su dung type hint `Annotated` de tuy chinh chien luoc cap nhat cho truong cu the:

```
Annotated[kieu, ham_reducer]
```

Ham reducer nhan hai tham so:
- `old`: Gia tri hien tai trong trang thai
- `new`: Gia tri moi ma node tra ve

Va tra ve **gia tri cuoi cung se duoc luu**.

### Phan tich code

#### Dinh nghia ham reducer va ap dung vao trang thai

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

Diem cot loi:
- `Annotated[list[str], operator.add]` co nghia la "khi truong `messages` duoc cap nhat, **noi them (concatenate)** list moi vao list cu".
- `operator.add` la ham co san cua Python, thuc hien phep `+` (ket hop) tren list.
- `update_function` duoc comment la ham reducer tuy chinh co cung hanh vi. Co the tu viet hoac dung ham co san nhu `operator.add`.

#### Dinh nghia node

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

- Chi `node_one` them muc moi vao `messages`.
- `node_two`, `node_three` tra ve dictionary trong nen khong thay doi trang thai.

#### Thuc thi va ket qua

```python
graph = graph_builder.compile()

graph.invoke(
    {"messages": ["Hello!"]},
)
```

Ket qua: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`

| Thoi diem | messages |
|-----------|----------|
| Dau vao ban dau | `["Hello!"]` |
| Sau khi node_one chay | `["Hello!"] + ["Hello, nice to meet you!"]` = `["Hello!", "Hello, nice to meet you!"]` |
| node_two, node_three | Khong thay doi |

**Neu khong co reducer**, gia tri tra ve cua `node_one` `["Hello, nice to meet you!"]` se **thay the hoan toan** gia tri ban dau `["Hello!"]`. Nho reducer, hai list duoc **ket hop**.

### Diem thuc hanh
- Viet ham reducer tuy chinh (vi du: chi giu gia tri lon nhat, loai bo trung lap...).
- Them message tu `node_two` va xac nhan tich luy hoat dong dung.
- Chay cung code khong co reducer va so sanh ket qua.
- Suy nghi tai sao reducer la bat buoc trong ung dung chat.

---

## 13.6 Node Caching - Cache node

### Chu de va Muc tieu
Hoc cach su dung **CachePolicy** de cache ket qua thuc thi cua node cu the va su dung gia tri da cache trong thoi gian nhat dinh ma khong can tinh lai.

### Giai thich khai niem cot loi

Mot so node co chi phi thuc thi cao (vi du: goi API ben ngoai, goi LLM) hoac tra ve cung ket qua cho cung dau vao. Trong nhung truong hop nay, **caching** giup toi uu hieu suat.

LangGraph cung cap chinh sach cache theo tung node:

| Thanh phan | Vai tro |
|------------|---------|
| `CachePolicy(ttl=giay)` | Dat thoi gian hieu luc cache (Time-To-Live) tinh bang giay |
| `InMemoryCache()` | Kho luu tru cache dua tren bo nho |
| `graph_builder.compile(cache=...)` | Ket noi kho cache khi bien dich |

### Phan tich code

#### Import va dinh nghia trang thai

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

#### Dinh nghia node - Node duoc cache

```python
def node_one(state: State):
    return {}

def node_two(state: State):
    return {"time": f"{datetime.now()}"}

def node_three(state: State):
    return {}
```

`node_two` tra ve thoi gian hien tai. Khi cache duoc ap dung, trong thoi gian TTL, thoi gian da ghi truoc do se duoc tra ve nguyen ven.

#### Ap dung chinh sach cache

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node(
    "node_two",
    node_two,
    cache_policy=CachePolicy(ttl=20),  # Cache trong 20 giay
)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

Cot loi: Chi `node_two` duoc chi dinh `cache_policy=CachePolicy(ttl=20)`. Ket qua cua node nay se duoc **cache trong 20 giay**.

#### Bien dich voi cache va chay lap lai

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

Code nay chay do thi 6 lan cach nhau 5 giay. Vi `node_two` co `ttl=20`:

- **Tu dau den 20 giay**: Ket qua lan chay dau (thoi gian) duoc cache, tra ve cung thoi gian.
- **Sau 20 giay**: Cache het han, `node_two` chay lai va ghi thoi gian moi.

### Diem thuc hanh
- Thay doi gia tri `ttl` va quan sat thoi diem cache het han.
- Tim hieu xem co the su dung kho cache khac ngoai `InMemoryCache` khong.
- Nghi ve cac tinh huong thuc te ma caching huu ich (vi du: gioi han rate API ben ngoai, tiet kiem chi phi).
- Nghi ve truong hop caching co the gay van de (vi du: can du lieu thoi gian thuc).

---

## 13.7 Conditional Edges - Canh dieu kien

### Chu de va Muc tieu
Su dung **Conditional Edges** de hien thuc logic phan nhanh (branching) **dong chon node tiep theo** dua tren trang thai.

### Giai thich khai niem cot loi

Cac do thi truoc day deu co duong di co dinh (luong tuyen tinh). Nhung trong ung dung thuc te, thuong can chon duong di khac nhau tuy theo trang thai.

Phuong thuc **`add_conditional_edges`** cho phep:
1. Sau mot node cu the, thuc thi **ham phan nhanh (routing function)**.
2. Chon dong node tiep theo dua tren gia tri tra ve cua ham phan nhanh.

```
add_conditional_edges(
    node_xuat_phat,
    ham_phan_nhanh,
    dictionary_anh_xa   # {gia_tri_tra_ve: node_dich}
)
```

### Phan tich code

#### Dinh nghia trang thai va node

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

#### Dinh nghia ham phan nhanh

Code cho thay hai cach viet ham phan nhanh:

**Cach 1: Tra ve truc tiep ten node (da bi comment)**
```python
# def decide_path(state: State) -> Literal["node_three", "node_four"]:
#     if state["seed"] % 2 == 0:
#         return "node_three"
#     else:
#         return "node_four"
```
Cach nay tra ve truc tiep ten node. Type hint `Literal` chi ro cac gia tri tra ve co the.

**Cach 2: Tra ve gia tri bat ky + dictionary anh xa (dang su dung)**
```python
def decide_path(state: State):
    return state["seed"] % 2 == 0  # Tra ve True hoac False
```
Ham phan nhanh tra ve gia tri bat ky nhu `True`/`False`, va dictionary anh xa chuyen doi thanh node thuc te.

#### Cau hinh conditional edges

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)

# Phan nhanh dieu kien tu START
graph_builder.add_conditional_edges(
    START,
    decide_path,
    {
        True: "node_one",     # seed chan thi di node_one
        False: "node_two",    # seed le thi di node_two
        "hello": END,         # Tra ve "hello" thi ket thuc
    },
)

graph_builder.add_edge("node_one", "node_two")

# Phan nhanh dieu kien tu node_two
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

Luong do thi:

```
             +-- True --> node_one --> node_two --+-- True --> node_three --> END
START -------+                                   +-- False -> node_four ---> END
             +-- False -> node_two ---------------+
             +-- "hello" -> END
```

### Diem thuc hanh
- Thay doi gia tri `seed` va quan sat duong di thuc thi thay doi the nao.
- Chuyen sang cach tra ve truc tiep ten node (cach 1) khong dung dictionary anh xa.
- Thiet ke conditional edge co 3 nhanh tro len.
- Ket hop conditional edges va edge thuong de tao workflow phuc tap.

---

## 13.8 Send API - Xu ly song song dong

### Chu de va Muc tieu
Hoc cach su dung **Send API** de **dong tao nhieu instance node** tai thoi diem chay va **thuc thi song song**.

### Giai thich khai niem cot loi

Conditional edges quyet dinh "di den node nao", con **Send API** tien xa hon:

1. Co the thuc thi **cung mot node nhieu lan** song song.
2. Truyen **dau vao khac nhau** cho moi instance.
3. **So luong instance duoc quyet dinh tai thoi diem chay**.

Dieu nay tuong tu pattern Map-Reduce:
- **Map**: Chia du lieu va ap dung cung xu ly cho tung phan
- **Reduce**: Gom ket qua lai (ket hop voi ham Reducer)

### Phan tich code

#### Import va dinh nghia trang thai

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

Diem cot loi:
- `words`: Danh sach tu can xu ly
- `output`: List tich luy ket qua xu ly moi tu. Su dung `Annotated` va reducer `operator.add` de gop ket qua.
- Import `Send` - cot loi cua xu ly song song dong.

#### Dinh nghia node

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

Diem khac biet quan trong:
- `node_one` la node thong thuong, nhan toan bo `State`.
- **`node_two` nhan `word` (chuoi) rieng le thay vi `State`.** Day la dau vao tuy chinh duoc truyen qua Send API.
- `node_two` tra ve ket qua trong list `output`. Nho reducer (`operator.add`), tat ca ket qua tu cac instance song song duoc tu dong gop lai.

#### Ham dispatcher va cau hinh do thi

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)

def dispatcher(state: State):
    return [Send("node_two", word) for word in state["words"]]

graph_builder.add_edge(START, "node_one")
graph_builder.add_conditional_edges("node_one", dispatcher, ["node_two"])
graph_builder.add_edge("node_two", END)
```

Phan tich cot loi:

1. **Ham `dispatcher`**: Tao doi tuong `Send` cho moi tu trong `state["words"]`.
   - `Send("node_two", word)`: "Thuc thi `node_two` voi dau vao la `word`"
   - Tra ve list nen `node_two` se chay song song **theo so luong tu**.

2. **Truyen list vao `add_conditional_edges`**: `["node_two"]` la danh sach node dich co the. LangGraph dung thong tin nay de xac thuc do thi.

#### Ket qua thuc thi

```python
graph.invoke(
    {
        "words": ["hello", "world", "how", "are", "you", "doing"],
    }
)
```

Xuat ra:
```
I want to count 6 words in my state.
```

Ket qua:
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

6 instance cua `node_two` moi cai xu ly mot tu, va ket qua duoc gop vao list `output` nho reducer `operator.add`.

### Diem thuc hanh
- Tang kich thuoc danh sach tu va quan sat su khac biet hieu suat.
- Them `time.sleep` vao `node_two` de cam nhan hieu qua cua xu ly song song.
- Viet code tuong duong khong dung Send API va so sanh.
- Nghi ve truong hop su dung thuc te (vi du: tom tat nhieu tai lieu dong thoi, thu thap du lieu tu nhieu nguon...).

---

## 13.9 Command - Doi tuong Command

### Chu de va Muc tieu
Hoc cach su dung doi tuong **Command** de **cap nhat trang thai va dinh tuyen dong thoi** tu ben trong node.

### Giai thich khai niem cot loi

Trong cac phuong phap da hoc:
- Cap nhat trang thai: Node tra ve dictionary
- Dinh tuyen: `add_conditional_edges` + ham phan nhanh rieng

Hai viec nay **tach roi** nhau. Doi tuong **Command** **thong nhat** ca hai:

```python
Command(
    goto="node_dich",           # Node tiep theo can di den
    update={"key": "value"},    # Cap nhat trang thai
)
```

Uu diem cua cach nay:
- Logic dinh tuyen nam ngay trong node, truc quan hon.
- Cap nhat trang thai va dinh tuyen duoc xu ly nguyen tu (atomic).
- Khong can ham phan nhanh rieng hay conditional edges.

### Phan tich code

#### Import va dinh nghia trang thai

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import Command

class State(TypedDict):
    transfer_reason: str

graph_builder = StateGraph(State)
```

#### Dinh nghia node - Node router tra ve Command

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

Phan tich cot loi:

1. **Kieu tra ve cua `triage_node`**: `Command[Literal["account_support", "tech_support"]]`
   - Chi ro node nay tra ve `Command` voi cac dich co the la `"account_support"` hoac `"tech_support"`.
   - Nho type hint nay, LangGraph biet duoc cac duong di co the **ma khong can `add_edge` hay `add_conditional_edges`**.

2. **Doi tuong `Command`**:
   - `goto="account_support"`: Di den node `account_support` tiep theo
   - `update={"transfer_reason": "The user wants to change password."}`: Cap nhat `transfer_reason` trong trang thai

#### Cau hinh do thi

```python
graph_builder.add_node("triage_node", triage_node)
graph_builder.add_node("tech_support", tech_support)
graph_builder.add_node("account_support", account_support)

graph_builder.add_edge(START, "triage_node")
# triage_node khong can add_edge! Command xu ly dinh tuyen.

graph_builder.add_edge("tech_support", END)
graph_builder.add_edge("account_support", END)
```

Diem dang chu y: Edge sau `triage_node` **khong duoc dinh nghia**. `goto` trong doi tuong `Command` quyet dinh node tiep theo tai thoi diem chay.

Cau truc do thi:
```
                         +---> tech_support -----> END
START ---> triage_node --+
                         +---> account_support --> END
```

#### Ket qua thuc thi

```python
graph = graph_builder.compile()
graph.invoke({})
```

Xuat ra:
```
account_support running
```

Ket qua: `{'transfer_reason': 'The user wants to change password.'}`

`triage_node` qua `Command`:
1. Cap nhat `transfer_reason`
2. Dinh tuyen den `account_support`

### Diem thuc hanh
- Sua `triage_node` de dinh tuyen den `tech_support` theo dieu kien.
- So sanh uu nhuoc diem giua `Command` va cach `add_conditional_edges`.
- Hien thuc nhieu buoc dinh tuyen nhu he thong ho tro khach hang thuc te bang `Command`.
- Tim hieu xem co the chi dinh nhieu node trong `goto` cua `Command` khong.

---

## Tong hop cot loi chuong (Key Takeaways)

### 1. Cau truc co ban cua LangGraph
- **StateGraph**: Lop cot loi cua do thi dua tren trang thai
- **Node**: Ham nhan trang thai, xu ly va tra ve cap nhat
- **Edge**: Ket noi giua cac node (xac dinh thu tu thuc thi)
- **START / END**: Diem vao va diem ket thuc do thi

### 2. Quan ly trang thai (State)
- Dinh nghia schema trang thai bang `TypedDict`.
- Chien luoc cap nhat mac dinh la **ghi de (overwrite)**.
- Su dung `Annotated` va ham reducer de ap dung chien luoc **tich luy (accumulate)**.
- `operator.add` la reducer pho bien nhat cho viec ket hop list.

### 3. Da schema
- `input_schema`: Gioi han hinh thuc dau vao tu ben ngoai
- `output_schema`: Gioi han hinh thuc dau ra tra ve ben ngoai
- Trang thai noi bo khong bi lo ra ben ngoai, co loi cho bao mat va thiet ke API.

### 4. Caching
- Dat chinh sach cache theo tung node bang `CachePolicy(ttl=giay)`.
- Kich hoat bang `InMemoryCache()` kem `compile(cache=...)`.
- Co the cai thien dang ke hieu suat cho cac phep tinh chi phi cao (goi API...).

### 5. Dieu khien luong

| Cach thuc | Dac diem | Khi nao su dung |
|-----------|---------|-----------------|
| `add_edge` | Duong di co dinh | Luon di den cung node tiep theo |
| `add_conditional_edges` | Dinh tuyen dong dua tren ham phan nhanh | Thay doi duong di theo trang thai |
| `Send` API | Thuc thi song song dong | Chay cung node voi dau vao khac nhau nhieu lan |
| `Command` | Dinh tuyen noi bo node + cap nhat trang thai | Xu ly dinh tuyen va thay doi trang thai trong mot lan |

### 6. Nguyen tac thiet ke cot loi
- Do thi duoc cau hinh **khai bao (declarative)**: Dinh nghia node va edge truoc, bien dich va thuc thi sau.
- Trang thai duoc xu ly nhu **bat bien (immutable)**: Node tra ve dictionary moi de cap nhat trang thai.
- **Tach biet moi quan tam**: Moi node chi co mot trach nhiem duy nhat.

---

## Bai tap thuc hanh (Practice Exercises)

### Bai tap 1: Do thi co ban (Do kho: 1/3)

Tao do thi tuyen tinh gom 4 node (`start_node`, `process_a`, `process_b`, `end_node`). Dat truong `counter: int` trong trang thai va moi node tang `counter` len 1. Gia tri `counter` cuoi cung phai la 4.

**Goi y**: Neu khong dung reducer thi se bi ghi de. Moi node doc gia tri hien tai va tra ve gia tri +1.

### Bai tap 2: Tich luy tin nhan chat (Do kho: 2/3)

Su dung reducer de tao trinh mo phong chat don gian:
- Trang thai: `messages: Annotated[list[str], operator.add]`
- `user_node`: Them `["Nguoi dung: Xin chao"]`
- `assistant_node`: Them `["Tro ly: Toi co the giup gi?"]`
- `user_reply_node`: Them `["Nguoi dung: Cho toi biet thoi tiet"]`

`messages` cuoi cung phai chua 3 tin nhan theo dung thu tu.

### Bai tap 3: Dinh tuyen dieu kien (Do kho: 2/3)

Tao do thi phan nhanh theo tuoi nguoi dung:
- Trang thai: `age: int`, `message: str`
- Phan nhanh dieu kien sau node `check_age`:
  - Duoi 18 tuoi: `minor_node` -> "Ban la vi thanh nien."
  - Tu 18 den duoi 65: `adult_node` -> "Ban la nguoi truong thanh."
  - Tu 65 tro len: `senior_node` -> "Ban thuoc doi tuong uu dai nguoi cao tuoi."

### Bai tap 4: Su dung Send API (Do kho: 3/3)

Tao do thi nhan cau nhap vao va chuyen dong thoi tung tu thanh chu hoa:
- Trang thai: `sentence: str`, `results: Annotated[list[str], operator.add]`
- `splitter_node`: Tach cau thanh cac tu
- `upper_node`: Chuyen tung tu thanh chu hoa (chay song song bang Send API)
- Dau vao: `{"sentence": "hello world from langgraph"}`
- Dau ra mong doi: `{"sentence": "...", "results": ["HELLO", "WORLD", "FROM", "LANGGRAPH"]}`

### Bai tap 5: Agent dua tren Command (Do kho: 3/3)

Su dung doi tuong Command de hien thuc router tu van khach hang don gian:
- Trang thai: `query: str`, `department: str`, `response: str`
- `router_node`: Dinh tuyen bang Command theo noi dung truy van
  - Chua "hoan tien" hoac "thanh toan" -> `billing_node`
  - Chua "loi" hoac "bug" -> `tech_node`
  - Con lai -> `general_node`
- Moi node bo phan dat thong bao huong dan phu hop vao `response`

**Bonus**: Su dung chinh xac type hint cua `Command` (`Command[Literal[...]]`) de khi truc quan hoa do thi, tat ca duong di co the deu duoc hien thi.
