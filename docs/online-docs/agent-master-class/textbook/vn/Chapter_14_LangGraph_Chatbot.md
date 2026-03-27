# Chapter 14: Xay dung Chatbot voi LangGraph

## Tong quan chuong

Trong chuong nay, chung ta se hoc cach xay dung mot chatbot AI thuc te theo tung buoc su dung **LangGraph**. LangGraph la thu vien cot loi trong he sinh thai LangChain, cho phep thiet ke cac workflow agent AI phuc tap thong qua mo hinh **Stateful Graph (Do thi co trang thai)**.

Cac chu de chinh xuyen suot chuong nay bao gom:

- **Cau truc chatbot co ban**: Xay dung khung chatbot su dung StateGraph va MessagesState
- **Tich hop cong cu (Tool)**: Mo hinh de LLM goi cong cu ben ngoai va su dung ket qua
- **Bo nho (Memory)**: Luu tru trang thai hoi thoai thong qua checkpointer dua tren SQLite
- **Human-in-the-loop**: Mo hinh interrupt de tich hop phan hoi cua con nguoi vao workflow
- **Du hanh thoi gian (Time Travel)**: Kham pha lich su trang thai va thuc thi nhanh thong qua fork
- **Cong cu phat trien (DevTools)**: Chuyen doi sang cau truc production cho LangGraph Studio

Moi phan xay dung dua tren ma nguon cua phan truoc, cuoi cung hoan thanh mot he thong agent day du bao gom goi cong cu, bo nho, su can thiep cua con nguoi va quan ly trang thai.

---

## 14.0 LangGraph Chatbot (Cau truc co ban)

### Chu de va Muc tieu

Muc tieu cua phan nay la tao cau truc chatbot co ban nhat trong LangGraph. Chung ta loai bo cac do thi dinh tuyen phuc tap dua tren `StateGraph`, `Command`, va `TypedDict` tu cac chuong truoc va tai cau truc hoan toan thanh mot **chatbot don gian dua tren tin nhan**.

### Cac khai niem cot loi

#### MessagesState la gi?

LangGraph cung cap mot lop trang thai duoc dinh nghia san goi la `MessagesState`. Day la cong cu quan ly trang thai duoc toi uu hoa cho phat trien chatbot, ben trong chua mot truong danh sach goi la `messages`. Danh sach nay tu dong tich luy cac loai tin nhan khac nhau cua LangChain nhu `HumanMessage`, `AIMessage`, va `ToolMessage`.

Diem khac biet lon nhat so voi cach dinh nghia trang thai truc tiep bang `TypedDict` truoc day la `MessagesState` ho tro hanh vi **them vao (append)** theo mac dinh. Nghia la, khi mot node tra ve `{"messages": [new_message]}`, tin nhan moi se duoc them vao danh sach tin nhan hien tai.

#### init_chat_model

Ham `init_chat_model` tu `langchain.chat_models` cho phep khoi tao cac nha cung cap LLM khac nhau thong qua giao dien thong nhat. Ban chi can truyen mot chuoi theo dinh dang `nha_cung_cap:ten_model`, vi du `"openai:gpt-4o-mini"`.

### Phan tich ma nguon

#### Buoc 1: Import va Khoi tao LLM

```python
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph.message import MessagesState

llm = init_chat_model("openai:gpt-4o-mini")
```

Chung ta loai bo `TypedDict` va `Command` da su dung trong ma truoc va thay vao do import `MessagesState` va `init_chat_model`. `init_chat_model` la ham khoi tao model chat da nang cua LangChain, trong do nha cung cap va ten model duoc phan cach bang dau hai cham.

#### Buoc 2: Dinh nghia State

```python
class State(MessagesState):
    custom_stuff: str

graph_builder = StateGraph(State)
```

Chung ta dinh nghia lop `State` rieng bang cach ke thua tu `MessagesState`. Vi `MessagesState` da bao gom truong `messages`, chung ta chi can khai bao them cac truong bo sung khi can thiet (o day la `custom_stuff`). Day la cach dinh nghia trang thai chuan cho chatbot LangGraph.

#### Buoc 3: Dinh nghia Node Chatbot

```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

Day la ham node chatbot cot loi. Nguyen ly hoat dong rat don gian:
1. Lay danh sach `messages` tu trang thai hien tai.
2. Truyen toan bo lich su tin nhan cho LLM de tao phan hoi.
3. Tra ve phan hoi da tao duoc them vao danh sach `messages`.

Nho logic reducer cua `MessagesState`, danh sach `messages` trong gia tri tra ve se duoc **them vao** cac tin nhan trang thai hien tai.

#### Buoc 4: Xay dung va Thuc thi Do thi

```python
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "how are you?"},
        ]
    }
)
```

Cau truc do thi rat don gian:
- `START` -> `chatbot` -> `END`

Chung ta truyen tin nhan khoi tao dang dictionary cho `graph.invoke()`. `{"role": "user", "content": "how are you?"}` se duoc LangChain tu dong chuyen doi thanh doi tuong `HumanMessage`.

Ket qua thuc thi tra ve mot dictionary trang thai chua `HumanMessage` va `AIMessage`.

### Diem thuc hanh

- Thu nghiem dinh nghia trang thai truc tiep bang `TypedDict` thay vi ke thua tu `MessagesState` va xem co gi khac biet.
- Thu thay doi nha cung cap cua `init_chat_model` thanh `"anthropic:claude-sonnet-4-20250514"` v.v. va chay cung do thi voi cac LLM khac nhau.
- Nghi cach su dung truong `custom_stuff` de tiem system prompt dong.

---

## 14.1 Tool Nodes (Node Cong cu)

### Chu de va Muc tieu

Trong phan nay, chung ta ket noi **cong cu ben ngoai (Tool)** vao chatbot. Chung ta hien thuc **mo hinh ReAct (Reasoning + Acting)** trong do LLM goi cong cu theo yeu cau cua nguoi dung va tao phan hoi cuoi cung su dung ket qua cong cu.

### Cac khai niem cot loi

#### Co che Tool Calling

Cac LLM hien dai nhu gpt-4o-mini cua OpenAI ho tro **function calling (goi cong cu)**. Khi ban cung cap cho LLM danh sach cac cong cu kha dung, LLM se chon cong cu phu hop dua tren cau hoi cua nguoi dung, xay dung cac tham so va tao yeu cau goi. LLM khong truc tiep thuc thi cong cu -- no bieu thi **y dinh (intent)** "hay goi cong cu nay voi cac tham so nhu the nay."

#### ToolNode va tools_condition

Module `langgraph.prebuilt` cung cap hai tien ich chinh:

- **`ToolNode`**: Node duoc xay dung san chiu trach nhiem thuc thi cong cu. No phat hien `tool_calls` duoc LLM tra ve, thuc thi cac cong cu tuong ung va tra ve ket qua duoi dang `ToolMessage`.
- **`tools_condition`**: Ham dinh tuyen co dieu kien. Neu phan hoi LLM chua `tool_calls`, no dinh tuyen den node `"tools"`; neu khong, no dinh tuyen den `END`.

#### Canh co dieu kien (Conditional Edges)

`add_conditional_edges` them cac canh xac dinh dong node tiep theo dua tren dau ra cua node truoc. Khi su dung voi `tools_condition`, no tu nhien hien thuc luong di chuyen den node cong cu chi khi LLM yeu cau goi cong cu, va ket thuc hoi thoai trong truong hop nguoc lai.

### Phan tich ma nguon

#### Buoc 1: Import moi

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
```

`ToolNode` va `tools_condition` den tu module prebuilt cua LangGraph, va decorator `@tool` den tu `langchain_core.tools`.

#### Buoc 2: Dinh nghia cong cu

```python
@tool
def get_weather(city: str):
    """Gets weather in city"""
    return f"The weather in {city} is sunny."
```

Decorator `@tool` chuyen doi ham Python thong thuong thanh cong cu LangChain. **Docstring** cua ham tro thanh mo ta cong cu, va **type hint tham so** cua ham duoc tu dong chuyen doi thanh schema dau vao cua cong cu. LLM su dung thong tin nay de quyet dinh co goi cong cu hay khong va truyen tham so gi.

#### Buoc 3: Bind cong cu vao LLM

```python
llm_with_tools = llm.bind_tools(tools=[get_weather])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

Phuong thuc `bind_tools()` thong bao cho LLM ve danh sach cong cu kha dung. Bay gio LLM co the tra ve yeu cau goi cong cu (`tool_calls`) thay vi van ban. Node chatbot su dung `llm_with_tools` thay vi `llm` thong thuong.

#### Buoc 4: Tai cau truc do thi

```python
tool_node = ToolNode(
    tools=[get_weather],
)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()
```

Cau truc do thi thay doi dang ke:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

Cac thay doi chinh:
1. Them `ToolNode` lam node `"tools"`.
2. Thay vi ket noi truc tiep tu `chatbot` den `END`, thiet lap **dinh tuyen co dieu kien** voi `add_conditional_edges`.
3. Them canh tu node `"tools"` den `"chatbot"` de tao **vong lap**.

Nho cau truc nay, LLM co the goi cong cu lap di lap lai nhieu lan khi can. Sau khi nhan ket qua cong cu, no quay lai node `chatbot` de tao phan hoi cuoi cung hoac goi them cong cu.

#### Buoc 5: Kiem tra luong thuc thi

```python
graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "what is the weather in machupichu"},
        ]
    }
)
```

Xem luong tin nhan trong ket qua thuc thi:

1. `HumanMessage`: "what is the weather in machupichu"
2. `AIMessage` (chua tool_calls): Yeu cau goi `get_weather(city="machupichu")`
3. `ToolMessage`: "The weather in machupichu is sunny."
4. `AIMessage`: "The weather in Machu Picchu is sunny." (phan hoi cuoi cung)

Ban co the thay LLM to chuc ket qua goi cong cu thanh ngon ngu tu nhien va chuyen den nguoi dung cuoi.

### Diem thuc hanh

- Dang ky nhieu cong cu dong thoi va xac minh xem LLM co chon cong cu phu hop cho tung tinh huong khong.
- Thu nghiem thay doi docstring cua cong cu va quan sat hanh vi chon cong cu cua LLM thay doi nhu the nao.
- Thay the `tools_condition` bang ham dieu kien tuy chinh de hien thuc logic dinh tuyen phuc tap hon.

---

## 14.2 Bo nho (Memory)

### Chu de va Muc tieu

Trong phan nay, chung ta them **bo nho luu tru** vao chatbot. Do thi LangGraph co ban mat trang thai sau khi cuoc goi `invoke()` ket thuc, nhung bang cach su dung **Checkpointer**, ban co the luu trang thai hoi thoai vao co so du lieu va duy tri ngu canh truoc do trong cac hoi thoai tiep theo.

### Cac khai niem cot loi

#### Checkpointer

Checkpointer cua LangGraph la co che tu dong luu trang thai tai **moi buoc** cua viec thuc thi do thi. Dieu nay cho phep:

- **Tinh lien tuc hoi thoai**: Khi goi `invoke()` nhieu lan voi cung `thread_id`, ngu canh hoi thoai truoc do duoc duy tri.
- **Lich su trang thai**: Tat ca cac buoc trung gian cua viec thuc thi do thi co the duoc truy van sau.
- **Phuc hoi loi**: Ngay ca khi xay ra loi trong qua trinh thuc thi, ban co the tiep tuc tu checkpoint cuoi cung.

#### SqliteSaver

`SqliteSaver` la hien thuc checkpointer su dung co so du lieu SQLite lam backend. Mac du checkpointer dua tren PostgreSQL duoc khuyen nghi cho moi truong production, SQLite tien loi cho muc dich phat trien va hoc tap.

#### thread_id va config

Khi su dung checkpointer, ban phai chi dinh `thread_id` trong `config`. `thread_id` la dinh danh phan biet cac phien hoi thoai -- su dung cung `thread_id` tiep tuc cung hoi thoai, trong khi su dung `thread_id` khac bat dau hoi thoai moi.

#### Streaming bat dong bo (Async Streaming)

Phan nay cung gioi thieu mo hinh streaming bat dong bo su dung `graph.astream()` thay vi `graph.invoke()`. Thiet lap `stream_mode="updates"` chuyen su kien khi ket qua thuc thi cua moi node xay ra, cho phep ban giam sat qua trinh thuc thi do thi theo thoi gian thuc.

### Phan tich ma nguon

#### Buoc 1: Ket noi SQLite va Thiet lap Checkpointer

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(
    "memory.db",
    check_same_thread=False,
)
```

Tao ket noi co so du lieu SQLite. `check_same_thread=False` la thiet lap de su dung SQLite an toan trong moi truong da luong. Tuy chon nay can thiet de hoat dong trong vong lap su kien bat dong bo cua Jupyter notebook.

Ngoai ra, dependency `aiosqlite` da duoc them vao `pyproject.toml`:

```toml
dependencies = [
    "aiosqlite>=0.21.0",
    ...
]
```

Day la goi ho tro cac thao tac bat dong bo cua `SqliteSaver`.

#### Buoc 2: Ket noi Checkpointer vao Do thi

```python
graph = graph_builder.compile(
    checkpointer=SqliteSaver(conn),
)
```

Chi can truyen tham so `checkpointer` cho phuong thuc `compile()` la kich hoat tinh nang bo nho. Tat ca cac lan thuc thi do thi sau do se tu dong luu trang thai vao SQLite.

#### Buoc 3: Thuc thi Streaming

```python
async for event in graph.astream(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the weather in berlin, budapest and bratislava.",
            },
        ]
    },
    stream_mode="updates",
):
    print(event)
```

`astream()` la bo tao bat dong bo, stream ket qua thuc thi cua moi node theo thoi gian thuc. Su dung `stream_mode="updates"` chuyen hieu qua chi **phan thay doi** duoi dang su kien thay vi toan bo trang thai.

Kich hoat phan `config` bi comment cho phep ban chi dinh `thread_id` de tiep tuc hoi thoai:

```python
config={
    "configurable": {
        "thread_id": "2",
    },
},
```

#### Buoc 4: Truy van Lich su Trang thai

```python
for state in graph.get_state_history(
    {
        "configurable": {
            "thread_id": "2",
        },
    }
):
    print(state.next)
```

`get_state_history()` tra ve tat ca cac snapshot trang thai cho mot thread cu the theo thu tu thoi gian nguoc. Moi snapshot chua cac gia tri trang thai tai thoi diem do va thong tin ve node tiep theo se duoc thuc thi (`next`). Tinh nang nay la nen tang cho tinh nang du hanh thoi gian trong phan tiep theo.

### Diem thuc hanh

- Xac minh rang hoi thoai thuc su tiep tuc bang cach goi `invoke()` nhieu lan voi cung `thread_id`.
- Xac nhan rang goi voi `thread_id` khac bat dau hoi thoai hoan toan moi.
- Phan tich ket qua cua `get_state_history()` de theo doi trang thai thay doi nhu the nao sau moi lan thuc thi node.
- So sanh dau ra khac nhau nhu the nao khi thay doi `stream_mode` thanh `"values"`.

---

## 14.3 Human-in-the-loop (Su can thiep cua con nguoi)

### Chu de va Muc tieu

Trong phan nay, chung ta hien thuc mot trong nhung tinh nang manh me nhat cua LangGraph: mo hinh **Human-in-the-loop**. Thay vi AI xu ly moi thu tu dong, mo hinh nay nhan **phan xet hoac phan hoi cua con nguoi** giua chung de tiep tuc workflow.

Chung ta su dung ham `interrupt` va lop `Command` cua LangGraph cho muc dich nay.

### Cac khai niem cot loi

#### Ham interrupt

`interrupt()` la ham **tam dung** viec thuc thi do thi. Khi duoc goi ben trong node cong cu, viec thuc thi do thi dung lai va trang thai hien tai duoc luu vao checkpointer. Nha phat trien co the gui cau hoi den nguoi dung tai diem gian doan va, sau khi nhan phan hoi cua nguoi dung, tiep tuc thuc thi voi `Command(resume=...)`.

```python
feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
return feedback
```

Gia tri truyen cho `interrupt()` tro thanh tin nhan hien thi cho nguoi dung khi gian doan, va gia tri truyen qua `Command(resume=...)` tro thanh gia tri tra ve cua `interrupt()`.

#### Lop Command

`Command` la doi tuong lenh de tiep tuc do thi da tam dung. Khi ban truyen phan hoi cua nguoi dung trong tham so `resume` va cung cap no cho `graph.invoke()`, do thi tiep tuc tu diem gian doan voi phan hoi tro thanh gia tri tra ve cua `interrupt()`.

#### Snapshot trang thai va next

Ban co the truy van snapshot trang thai hien tai cua do thi thong qua `graph.get_state(config)`. Thuoc tinh `next` cua snapshot tra ve ten cua node tiep theo se duoc thuc thi duoi dang tuple.
- `('tools',)`: Bi gian doan tai node cong cu, dang cho
- `()`: Viec thuc thi do thi da hoan thanh

### Phan tich ma nguon

#### Buoc 1: Import interrupt va Command

```python
from langgraph.types import interrupt, Command
```

Import hai lop chinh tu module `types` cua LangGraph.

#### Buoc 2: Dinh nghia cong cu Phan hoi con nguoi

```python
@tool
def get_human_feedback(poem: str):
    """
    Asks the user for feedback on the poem.
    Use this before returning the final response.
    """
    feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
    return feedback
```

Hoat dong cua cong cu nay la diem mau chot:
1. LLM tao bai tho va goi cong cu nay.
2. Khi `interrupt()` duoc goi, viec thuc thi do thi tam dung.
3. Khi nguoi dung cung cap phan hoi, `interrupt()` tra ve phan hoi do.
4. Phan hoi duoc chuyen den LLM duoi dang `ToolMessage`.

#### Buoc 3: Node Chatbot voi System Prompt

```python
def chatbot(state: State):
    response = llm_with_tools.invoke(
        f"""
        You are an expert in making poems.

        Use the `get_human_feedback` tool to get feedback on your poem.

        Only after you receive positive feedback you can return the final poem.

        ALWAYS ASK FOR FEEDBACK FIRST.

        Here is the conversation history:

        {state["messages"]}
    """
    )
    return {
        "messages": [response],
    }
```

System prompt dua ra chi dan ro rang cho LLM:
- Vai tro chuyen gia lam tho
- Phai su dung cong cu `get_human_feedback` de nhan phan hoi
- Chi tra ve bai tho cuoi cung sau khi nhan phan hoi tich cuc

Thiet ke prompt nay quyet dinh su thanh cong cua mo hinh Human-in-the-loop.

#### Buoc 4: Thuc thi lan dau (Den khi gian doan)

```python
config = {
    "configurable": {
        "thread_id": "3",
    },
}

result = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "Please make a poem about Python code."},
        ]
    },
    config=config,
)
```

Khi thuc thi, LLM tao bai tho va goi cong cu `get_human_feedback`. Do thi bi gian doan boi `interrupt()`, va trang thai hoi thoai den thoi diem nay duoc luu vao checkpointer.

#### Buoc 5: Kiem tra trang thai

```python
snapshot = graph.get_state(config)
snapshot.next  # ('tools',)
```

Khi `next` tra ve `('tools',)`, nghia la do thi bi gian doan tai node cong cu va dang cho phan hoi cua nguoi dung.

#### Buoc 6: Cung cap phan hoi va Tiep tuc

```python
response = Command(resume="It looks great!")

result = graph.invoke(
    response,
    config=config,
)
for message in result["messages"]:
    message.pretty_print()
```

Khi phan hoi tich cuc duoc truyen voi `Command(resume="It looks great!")`:
1. `interrupt()` tra ve `"It looks great!"`.
2. Gia tri nay duoc chuyen den LLM duoi dang `ToolMessage`.
3. LLM xac nhan phan hoi tich cuc va tra ve bai tho cuoi cung.

Trong dau ra thuc te, ban co the thay toan bo luong hoi thoai:
- Bai tho dau tien duoc tao -> Phan hoi "It is too long! Make shorter." -> Phien ban ngan hon duoc tao -> Phan hoi "It looks great!" -> Bai tho cuoi cung duoc tra ve

#### Buoc 7: Xac nhan hoan thanh

```python
snapshot = graph.get_state(config)
snapshot.next  # ()
```

Tuple rong `()` cho biet viec thuc thi do thi da hoan thanh.

### Diem thuc hanh

- Quan sat cach LLM phan ung khi phan hoi tieu cuc duoc cung cap nhieu lan lien tiep.
- So sanh voi phien ban ma cong cu tu dong tra ve phan hoi khong co `interrupt()` de cam nhan su khac biet cua Human-in-the-loop.
- Thiet ke workflow phuc tap voi nhieu diem interrupt (vi du: xem xet -> phe duyet -> trien khai pipeline).
- Thu nghiem truyen du lieu co cau truc (dictionary) cho `Command(resume=...)`.

---

## 14.4 Du hanh thoi gian (Time Travel)

### Chu de va Muc tieu

Trong phan nay, chung ta hoc tinh nang du hanh thoi gian su dung **lich su trang thai** duoc luu boi checkpointer cua LangGraph de quay lai mot thoi diem cu the trong qua khu va **fork (phan nhanh)** viec thuc thi do thi.

Tinh nang nay duoc su dung trong nhieu tinh huong thuc te nhu debug, A/B testing, va rollback trai nghiem nguoi dung.

### Cac khai niem cot loi

#### Lich su trang thai

Do thi co checkpointer duoc kich hoat se luu snapshot trang thai **sau moi lan thuc thi node**. Goi `get_state_history(config)` truy xuat tat ca cac snapshot cho mot thread cu the theo thu tu thoi gian nguoc. Moi snapshot chua:

- `values`: Trang thai day du tai thoi diem do (danh sach tin nhan, v.v.)
- `next`: Node tiep theo se duoc thuc thi
- `config`: Cau hinh xac dinh snapshot (bao gom checkpoint_id)

#### Fork trang thai (State Fork)

Su dung `graph.update_state()`, ban co the tao **nhanh moi** dua tren mot checkpoint cu the trong qua khu. Vi du, quay lai thoi diem nguoi dung noi "Toi song o Valencia" va doi thanh "Toi song o Zagreb" se khien LLM tao phan hoi dua tren Zagreb trong nhanh moi.

#### checkpoint_id

Moi snapshot trang thai co mot `checkpoint_id` duy nhat. Bao gom ID nay trong `config` khi goi `graph.invoke()` cho phep ban tiep tuc thuc thi do thi tu checkpoint do.

### Phan tich ma nguon

Trong phan nay, chung ta loai bo tat ca cac goi cong cu va Human-in-the-loop truoc do va quay lai cau truc chatbot don gian. Dieu nay de tap trung vao khai niem du hanh thoi gian.

#### Buoc 1: Chatbot don gian hoa

```python
class State(MessagesState):
    pass

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
    }

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(
    checkpointer=SqliteSaver(conn),
)
```

Chatbot don gian khong co node cong cu, chi kich hoat bo nho. `State` ke thua truc tiep tu `MessagesState` khong co truong bo sung.

#### Buoc 2: Chay hoi thoai

```python
config = {
    "configurable": {
        "thread_id": "0_x",
    },
}

result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "I live in Europe now. And the city I live in is Valencia.",
            },
        ]
    },
    config=config,
)
```

Nguoi dung noi "Toi song o chau Au, va thanh pho cua toi la Valencia." LLM tao phan hoi cho dieu nay.

#### Buoc 3: Kham pha lich su trang thai

```python
state_history = graph.get_state_history(config)

for state_snapshot in list(state_history):
    print(state_snapshot.next)
    print(state_snapshot.values["messages"])
    print("=========\n")
```

Chung ta duyet qua tat ca cac snapshot trang thai, in node `next` va noi dung tin nhan tai moi thoi diem. Dieu nay cho phep ban hieu toan bo luong thuc thi do thi theo thu tu thoi gian.

#### Buoc 4: Chon diem Fork

```python
state_history = graph.get_state_history(config)
to_fork = list(state_history)[-5]
to_fork.config
```

Chung ta chon mot snapshot tai chi muc cu the tu lich su trang thai. `config` cua snapshot nay chua `checkpoint_id` cho thoi diem do.

#### Buoc 5: Sua doi trang thai (Tao Fork)

```python
from langchain_core.messages import HumanMessage

graph.update_state(
    to_fork.config,
    {
        "messages": [
            HumanMessage(
                content="I live in Europe now. And the city I live in is Zagreb.",
                id="25169a3d-cc86-4a5f-9abd-03d575089a9f",
            )
        ]
    },
)
```

Cac diem chinh cua `update_state()`:
- **Tham so thu nhat**: `config` cua thoi diem can fork (bao gom checkpoint_id)
- **Tham so thu hai**: Cac gia tri trang thai can sua doi
- **Message ID**: Chi dinh cung ID se **thay the** tin nhan hien tai; su dung ID moi se **them** tin nhan.

O day chung ta thay the "Valencia" bang "Zagreb" de tao nhanh moi.

#### Buoc 6: Tiep tuc thuc thi tu trang thai da Fork

```python
result = graph.invoke(
    None,
    {
        "configurable": {
            "thread_id": "0_x",
            "checkpoint_ns": "",
            "checkpoint_id": "1f08d808-b408-6ca2-8004-f964cbac5a14",
        }
    },
)

for message in result["messages"]:
    message.pretty_print()
```

Trong `graph.invoke(None, config)`:
- Khi tham so thu nhat la `None`, thuc thi tiep tuc tu trang thai hien tai ma khong co dau vao moi.
- Chi dinh `checkpoint_id` cu the trong `config` bat dau tu checkpoint do.

Dieu nay khien LLM tao phan hoi moi dua tren tin nhan da sua doi "Toi song o Zagreb."

### Diem thuc hanh

- Sau khi co nhieu hoi thoai, thu nghiem quay lai mot diem giua va hoi cau hoi khac.
- Xac minh truong hop dat ID tin nhan khac trong `update_state()` dan den viec them thay vi thay the.
- Tao nhieu fork khac nhau tu cung checkpoint de hien thuc kich ban A/B testing.
- Phan tich cach cac trang thai da fork duoc ghi lai trong lich su su dung `get_state_history()`.

---

## 14.5 Cong cu phat trien (DevTools)

### Chu de va Muc tieu

Trong phan nay, chung ta chuyen doi prototype dua tren Jupyter notebook sang **cau truc production** va cau hinh de su dung voi **LangGraph Studio** (LangGraph DevTools).

LangGraph Studio la cong cu phat trien cung cap debug truc quan, theo doi trang thai, du hanh thoi gian va nhieu hon thong qua GUI.

### Cac khai niem cot loi

#### Tu Jupyter sang Python Script

Mac du Jupyter notebook huu ich cho prototyping nhanh trong qua trinh phat trien, ban can chuyen doi sang Python script chuan (`.py`) de trien khai thuc te va tich hop DevTools. Trong qua trinh nay:

- Hop nhat ma dua tren cell cua notebook thanh mot script duy nhat
- Ket hop lai tat ca cac tinh nang tu cac phan truoc: Human-in-the-loop, goi cong cu, v.v.
- Export doi tuong `graph` o cap module de co the tham chieu tu ben ngoai

#### File cau hinh langgraph.json

`langgraph.json` la file cau hinh de LangGraph Studio nhan dien du an. File nay dinh nghia cac dependency, bien moi truong va diem vao cua do thi.

### Phan tich ma nguon

#### Buoc 1: Cau hinh langgraph.json

```json
{
    "dependencies": [
        "langchain_openai",
        "./main.py"
    ],
    "env": "./.env",
    "graphs": {
        "mr_poet": "./main.py:graph"
    }
}
```

Y nghia cua moi truong:
- **`dependencies`**: Cac goi va module can thiet cho du an. `langchain_openai` la goi tich hop OpenAI, va `./main.py` la file script noi do thi duoc dinh nghia.
- **`env`**: Duong dan den file bien moi truong. Tai thong tin bi mat nhu API key OpenAI tu file `.env`.
- **`graphs`**: Danh sach cac do thi hien thi cho DevTools. Dang ky bien `graph` tu `main.py` duoi ten `"mr_poet"`.

#### Buoc 2: main.py - Ma production tich hop

```python
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
```

Tat ca cac module can thiet duoc import. Tat ca cac tinh nang da hoc trong cac phan truoc (cong cu, checkpointer, interrupt) duoc tich hop vao mot file duy nhat.

#### Buoc 3: Dinh nghia cong cu (Bao gom Human-in-the-loop)

```python
@tool
def get_human_feedback(poem: str):
    """
    Get human feedback on a poem.
    Use this to get feedback on a poem.
    The user will tell you if the poem is ready or if it needs more work.
    """
    response = interrupt({"poem": poem})
    return response["feedback"]

tools = [get_human_feedback]
```

Cong cu Human-in-the-loop tu Phan 14.3 da duoc cai tien nhe. Dictionary duoc truyen cho `interrupt()`, va phan hoi cung duoc nhan dang dictionary (`response["feedback"]`), cho phep trao doi du lieu co cau truc.

#### Buoc 4: Dinh nghia LLM va State

```python
llm = init_chat_model("openai:gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class State(MessagesState):
    pass
```

#### Buoc 5: Node Chatbot (Voi System Prompt)

```python
def chatbot(state: State) -> State:
    response = llm_with_tools.invoke(
        f"""
    You are an expert at making poems.

    You are given a topic and need to write a poem about it.

    Use the `get_human_feedback` tool to get feedback on your poem.

    Only after the user says the poem is ready, you should return the poem.

    Here is the conversation history:
    {state['messages']}
    """
    )
    return {
        "messages": [response],
    }
```

System prompt tuong tu Phan 14.3 duoc su dung, voi viec them type hint (`-> State`) de tang tinh ro rang cua ma.

#### Buoc 6: Xay dung Do thi

```python
tool_node = ToolNode(
    tools=tools,
)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

conn = sqlite3.connect("memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

graph = graph_builder.compile(name="mr_poet")
```

Cau truc do thi cuoi cung:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

Cac diem dang chu y:
- `compile(name="mr_poet")` gan ten cho do thi. Ten nay lien ket voi truong `graphs` trong `langgraph.json`.
- Checkpointer dua tren SQLite duoc thiet lap thong qua file `memory.db`.
- Bien `graph` duoc dinh nghia o cap module, co the tham chieu tu ben ngoai qua `./main.py:graph`.

### Diem thuc hanh

- Thu chay LangGraph Studio voi lenh `langgraph dev` (hoac `langgraph up`).
- Kiem tra truc quan cau truc do thi trong LangGraph Studio va theo doi qua trinh thuc thi cua moi node.
- Su dung tinh nang du hanh thoi gian cua Studio de quay lai trang thai qua khu, sua doi chung va tao nhanh moi.
- Thu nghiem dang ky nhieu do thi trong `langgraph.json` de quan ly chung dong thoi.

---

## Tong ket chuong

### 1. Cau truc co ban cua LangGraph Chatbot
- Quan ly hoi thoai dua tren tin nhan voi lop trang thai ke thua tu `MessagesState`.
- Dinh nghia node va canh voi `StateGraph` va tao do thi thuc thi duoc voi `compile()`.
- Su dung cac nha cung cap LLM khac nhau thong qua giao dien thong nhat voi `init_chat_model()`.

### 2. Mo hinh tich hop cong cu
- Chuyen doi ham Python thanh cong cu LangChain voi decorator `@tool`.
- Bind cong cu vao LLM voi `llm.bind_tools()` va tu dong hoa luong thuc thi cong cu voi `ToolNode` va `tools_condition`.
- Dinh tuyen dong dua tren viec LLM co goi cong cu hay khong thong qua canh co dieu kien (`add_conditional_edges`).

### 3. Bo nho va Checkpointer
- Kich hoat luu tru trang thai bang cach truyen `SqliteSaver` cho `compile(checkpointer=...)`.
- Phan biet phien hoi thoai voi `thread_id`; su dung cung ID tiep tuc hoi thoai.
- Streaming thoi gian thuc co the voi `astream(stream_mode="updates")`.

### 4. Human-in-the-loop
- Tam dung thuc thi do thi va cho dau vao nguoi dung voi ham `interrupt()`.
- Chuyen phan hoi cua nguoi dung voi `Command(resume=...)` de tiep tuc thuc thi.
- Kiem tra trang thai tam dung hien tai voi `get_state(config).next`.

### 5. Du hanh thoi gian
- Truy van tat ca snapshot trang thai voi `get_state_history(config)`.
- Sua doi trang thai qua khu de tao fork voi `update_state(checkpoint_config, new_values)`.
- Tiep tuc thuc thi tu checkpoint cu the voi `graph.invoke(None, checkpoint_config)`.
- Chi dinh cung message ID thay the; chi dinh ID khac them vao.

### 6. Tich hop DevTools
- Dinh nghia cai dat du an voi `langgraph.json`.
- Chuyen doi tu Jupyter notebook sang Python script de ap dung cau truc production.
- Thuc hien debug truc quan, theo doi trang thai va du hanh thoi gian qua GUI thong qua LangGraph Studio.

---

## Bai tap thuc hanh

### Bai tap 1: Chatbot da cong cu (Co ban)

Hien thuc chatbot voi 3 cong cu tro len nhu thoi tiet, ty gia hoi doai va tim kiem tin tuc. LLM nen chon cong cu phu hop dua tren cau hoi cua nguoi dung va, khi can thiet, goi nhieu cong cu tuan tu de tao phan hoi cuoi cung.

**Yeu cau:**
- Dinh nghia it nhat 3 ham `@tool`
- Xay dung do thi su dung `ToolNode` va `tools_condition`
- Test cac kich ban yeu cau goi da cong cu

### Bai tap 2: Workflow phe duyet (Trung cap)

Hien thuc workflow 3 giai doan: tao tai lieu -> xem xet -> phe duyet. Moi giai doan yeu cau su phe duyet cua con nguoi thong qua Human-in-the-loop truoc khi chuyen sang giai doan tiep theo.

**Yeu cau:**
- Nhieu diem phe duyet su dung `interrupt()`
- Logic quay lai giai doan truoc khi bi tu choi
- Luu tru trang thai thong qua checkpointer

### Bai tap 3: Trinh debug du hanh thoi gian (Nang cao)

Hien thuc trinh debug tuong tac kham pha lich su hoi thoai, quay lai cac diem cu the va tao nhanh voi dau vao khac nhau.

**Yeu cau:**
- Hien thi truc quan toan bo lich su su dung `get_state_history()`
- Giao dien cho nguoi dung chon checkpoint cu the va sua doi trang thai
- Tao fork va tiep tuc thuc thi su dung `update_state()`
- Chuc nang so sanh ket qua giua nhanh goc va nhanh da fork

### Bai tap 4: Trien khai DevTools (Nang cao)

Mo rong `main.py` tu Phan 14.5 de tao cau truc quan ly nhieu do thi trong mot du an duy nhat.

**Yeu cau:**
- Dang ky 2 do thi khac nhau tro len trong `langgraph.json`
- Moi do thi su dung cong cu va system prompt khac nhau
- Xac minh va chay ca hai do thi trong LangGraph Studio
- Quan ly bien moi truong thong qua file `.env`
