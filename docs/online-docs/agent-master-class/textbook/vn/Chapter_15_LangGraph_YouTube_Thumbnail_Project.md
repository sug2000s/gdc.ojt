# Chapter 15: Du an tu dong tao thumbnail YouTube bang LangGraph

---

## 1. Tong quan chuong

Trong chuong nay, chung ta se su dung **LangGraph** de xay dung mot pipeline hoan chinh tu dong tao thumbnail tu video YouTube. Du an nay vuot xa viec don thuan tao hinh anh - bao gom toan bo workflow tu trich xuat audio tu video, nhan dang giong noi (Transcription), tom tat van ban, tao hinh anh bang AI, den tao thumbnail chat luong cao cuoi cung co phan hoi cua nguoi dung, tat ca duoc thiet ke dua tren do thi.

### Muc tieu hoc tap cot loi cua du an

| Section | Chu de | Cong nghe cot loi |
|---------|--------|-------------------|
| 15.0 | Gioi thieu du an va thiet lap moi truong | uv, pyproject.toml, quan ly phu thuoc |
| 15.1 | Trich xuat audio va nhan dang giong noi | ffmpeg, OpenAI Whisper API, StateGraph |
| 15.2 | Node tom tat song song | Send API, pattern Map-Reduce, Annotated reducer |
| 15.3 | Node tao sketch thumbnail | GPT Image API, tao hinh anh song song |
| 15.4 | Vong lap phan hoi cua nguoi | interrupt, Command, InMemorySaver |
| 15.5 | Tao thumbnail HD va trien khai | LangGraph CLI, langgraph.json, trien khai production |

### So do luong do thi tong the

```
[START]
   |
   v
[extract_audio] -----> Trich xuat mp3 tu mp4 bang ffmpeg
   |
   v
[transcribe_audio] --> Chuyen giong noi -> van ban bang OpenAI Whisper
   |
   v (conditional: dispatch_summarizers)
[summarize_chunk] x N --> Chia van ban thanh chunk va tom tat song song
   |
   v
[mega_summary] --------> Tong hop tat ca tom tat chunk thanh mot tom tat cuoi
   |
   v (conditional: dispatch_artists)
[generate_thumbnails] x 5 --> Tao song song 5 sketch thumbnail khac nhau
   |
   v
[human_feedback] ------> Thu thap phan hoi nguoi dung bang interrupt
   |
   v
[generate_hd_thumbnail] -> Tao thumbnail chat luong cao cuoi cung co phan hoi
   |
   v
[END]
```

---

## 2. Mo ta chi tiet tung section

---

### 15.0 Gioi thieu du an va thiet lap moi truong

**Commit:** `c8e3d16` "15.0 Introduction"

#### Chu de va Muc tieu

Day la buoc xay dung nen tang du an. Su dung cong cu quan ly goi Python **uv** de khoi tao du an va thiet lap tat ca phu thuoc can thiet. Bao gom `ipykernel` lam phu thuoc phat trien de bat dau phat trien trong moi truong Jupyter Notebook.

#### Giai thich khai niem cot loi

**uv package manager**: Package manager hien dai thay the `pip` va `venv` trong he sinh thai Python. Duoc viet bang Rust nen rat nhanh, quan ly du an dua tren `pyproject.toml`. File `uv.lock` co dinh (lock) chinh xac phien ban phu thuoc, dam bao moi truong co the tai tao.

**pyproject.toml**: File cau hinh chuan cua du an Python. Dinh nghia metadata du an va phu thuoc theo cach khai bao.

#### Phan tich code

```toml
[project]
name = "youtube-thumbnail-maker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.7",
    "langgraph-cli[inmem]>=0.4.2",
    "langsmith>=0.4.28",
    "openai[aiohttp]>=1.107.3",
    "python-dotenv>=1.1.1",
    "tiktoken>=0.11.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

**Phan tich phu thuoc:**

- **`langchain[openai]`**: Tich hop OpenAI cua framework LangChain. `[openai]` la extras, cai them cac goi lien quan OpenAI.
- **`langgraph`**: Thu vien cot loi LangGraph. Cong cu cot loi de xay dung workflow agent dua tren do thi.
- **`langgraph-cli[inmem]`**: Cong cu CLI LangGraph. `[inmem]` ho tro checkpointer in-memory. Dung khi chay server phat trien local.
- **`langsmith`**: Nen tang observability LangSmith. Giam sat va debug viec thuc thi do thi.
- **`openai[aiohttp]`**: OpenAI Python SDK. `[aiohttp]` them ho tro HTTP bat dong bo. Su dung cho Whisper API va Image Generation API.
- **`python-dotenv`**: Tien ich tai bien moi truong (API key...) tu file `.env`.
- **`tiktoken`**: Tokenizer cua OpenAI. Su dung khi can tinh so token cua van ban.
- **`ipykernel`**: Phu thuoc phat trien cho phep su dung kernel Python trong Jupyter Notebook.

#### Diem thuc hanh

1. Tao du an moi bang lenh `uv init youtube-thumbnail-maker`.
2. Them phu thuoc bang `uv add langchain[openai] langgraph openai[aiohttp]`...
3. Tao file `.env` va thiet lap `OPENAI_API_KEY`.
4. Chay moi truong Jupyter bang `uv run jupyter notebook`.

---

### 15.1 Trich xuat audio va nhan dang giong noi

**Commit:** `4133ae6` "15.1 Audio Extraction and Transcription"

#### Chu de va Muc tieu

Trich xuat audio (mp3) tu file video YouTube (mp4) va su dung model Whisper cua OpenAI de chuyen giong noi thanh van ban. Trong buoc nay, hoc **cau truc co ban cua StateGraph** va khai niem **Node va Edge**.

#### Giai thich khai niem cot loi

**StateGraph**: Lop cot loi cua LangGraph, dinh nghia do thi truyen du lieu giua cac node thong qua trang thai (State). Moi node nhan trang thai lam dau vao va tra ve cap nhat cho mot phan cua trang thai.

**Dinh nghia State bang TypedDict**: Su dung `TypedDict` cua Python de dinh nghia schema trang thai chia se tren toan bo do thi mot cach an toan ve kieu. Day co the coi la "hop dong du lieu (data contract)" cua do thi.

**ffmpeg**: Cong cu dong lenh manh me de xu ly audio/video. Thuc thi tu Python qua `subprocess.run()` nhu tien trinh ben ngoai.

**OpenAI Whisper API**: Model nhan dang giong noi (Speech-to-Text). Ho tro nhieu ngon ngu, va co the tang do chinh xac nhan dien thuat ngu chuyen nganh qua tham so `prompt`.

#### Phan tich code

**Buoc 1: Dinh nghia State**

```python
from langgraph.graph import END, START, StateGraph
from typing import TypedDict
import subprocess
from openai import OpenAI


class State(TypedDict):
    video_file: str       # Duong dan file video dau vao
    audio_file: str       # Duong dan file audio da trich xuat
    transcription: str    # Van ban ket qua nhan dang giong noi
```

`State` la cau truc du lieu chia se tren toan bo do thi. Moi node doc mot phan hoac toan bo State va tra ve dictionary chi chua phan can cap nhat. Gia tri tra ve duoc **gop (merge)** vao State hien tai.

**Buoc 2: Node trich xuat audio**

```python
def extract_audio(state: State):
    output_file = state["video_file"].replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file,
    ]
    subprocess.run(command)
    return {
        "audio_file": output_file,
    }
```

Diem dang chu y trong node nay:

- **`-filter:a "atempo=2.0"`**: Tang toc do phat audio len 2 lan. Whisper API co gioi han kich thuoc file upload (25MB), nen tang toc audio la thu thuat thuc te de giam kich thuoc file. Khong anh huong dang ke den chat luong nhan dang giong noi.
- **`-y`**: Co tu dong cho phep ghi de khi file dau ra da ton tai.
- **Gia tri tra ve**: Chi tra ve `{"audio_file": output_file}` de chi cap nhat truong `audio_file` trong State.

**Buoc 3: Node nhan dang giong noi**

```python
def transcribe_audio(state: State):
    client = OpenAI()
    with open(state["audio_file"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en",
            prompt="Netherlands, Rotterdam, Amsterdam, The Hage",
        )
        return {
            "transcription": transcription,
        }
```

- **`model="whisper-1"`**: Su dung model nhan dang giong noi Whisper cua OpenAI.
- **`response_format="text"`**: Nhan ket qua dang van ban thuan. Cung ho tro `"json"`, `"verbose_json"`, `"srt"`, `"vtt"`...
- **`language="en"`**: Chi dinh ngon ngu la tieng Anh de tang do chinh xac.
- **Tham so `prompt`**: Day la tinh nang rat huu ich. Cung cap truoc cac danh tu rieng (ten thanh pho...) du kien xuat hien trong video de giup Whisper nhan dien chinh xac.

**Buoc 4: Cau hinh va thuc thi do thi**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_edge("transcribe_audio", END)

graph = graph_builder.compile()
```

Cau hinh do thi LangGraph gom ba buoc:

1. **Them node** (`add_node`): Dang ky ham can thuc thi kem ten.
2. **Them edge** (`add_edge`): Dinh nghia thu tu thuc thi (huong) giua cac node. `START` va `END` la node dac biet danh dau diem bat dau va ket thuc do thi.
3. **Bien dich** (`compile`): Chuyen do thi thanh dang co the thuc thi.

```python
graph.invoke({"video_file": "netherlands.mp4"})
```

Truyen trang thai ban dau vao `invoke()` de thuc thi do thi. Do thi chay theo thu tu `START` -> `extract_audio` -> `transcribe_audio` -> `END`, va tra ve trang thai cuoi cung.

#### Diem thuc hanh

1. Kiem tra `ffmpeg` da cai tren he thong chua (`ffmpeg -version`).
2. Thu pipeline voi video test ngan (1-2 phut) truoc.
3. Thay doi gia tri `atempo` (1.5, 2.0, 3.0) va so sanh su khac biet chat luong nhan dang giong noi.
4. Thu them danh tu rieng khac vao tham so `prompt` va quan sat su thay doi do chinh xac.

---

### 15.2 Node tom tat song song (Summarizer Nodes)

**Commit:** `a664f18` "15.2 Summarizer Nodes"

#### Chu de va Muc tieu

Chia van ban transcription dai thanh nhieu chunk va tom tat **song song** tung chunk theo **pattern Map-Reduce**. Hoc cach xu ly song song dong voi `Send` API va `Annotated` reducer cua LangGraph.

#### Giai thich khai niem cot loi

**Pattern Map-Reduce**: Pattern xu ly phan tan co dien cho du lieu lon.
- **Buoc Map**: Chia du lieu thanh nhieu phan va xu ly doc lap (tom tat song song).
- **Buoc Reduce**: Gop cac ket qua rieng le lai (mega_summary o section tiep theo).

**Send API**: Co che cot loi cho xu ly song song dong trong LangGraph. Tra ve list `Send("ten_node", du_lieu)` se khien node do chay song song cho tung du lieu.

**Annotated reducer**: Chi dinh ham reducer cho kieu bang `Annotated[list[str], operator.add]`. Khi nhieu node dong thoi tra ve gia tri cho cung truong State, dinh nghia cach ket hop cac gia tri. `operator.add` noi (concatenate) cac list lai.

#### Phan tich code

**Mo rong State: Annotated reducer**

```python
from langgraph.types import Send
from typing_extensions import Annotated
import operator
import textwrap
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")


class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]  # Them reducer!
```

Viec su dung `Annotated[list[str], operator.add]` cho truong `summaries` la diem cot loi. Neu khong co, khi nhieu node song song cung ghi vao `summaries`, chi gia tri cuoi cung con lai. Voi reducer `operator.add`, gia tri tra ve tu moi node duoc **tich luy** vao list hien tai.

> **Luu y**: Phai su dung `typing_extensions.Annotated` thay vi `typing.Annotated`. Voi `typing.Annotated`, cu phap goi ham `Annotated(list[str], operator.add)` se gay loi `TypeError: Cannot instantiate typing.Annotated`. Bat buoc dung cu phap ngoac vuong `Annotated[list[str], operator.add]`.

**Ham dispatcher: Phan nhanh song song dong**

```python
def dispatch_summarizers(state: State):
    transcription = state["transcription"]
    chunks = []
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append({"id": i + 1, "chunk": chunk})
    return [Send("summarize_chunk", chunk) for chunk in chunks]
```

Nguyen ly hoat dong cua ham nay:

1. **Chia van ban**: `textwrap.wrap(transcription, 500)` chia van ban dai thanh cac doan toi da 500 ky tu. Ton trong ranh gioi tu nen tu khong bi cat giua chung.
2. **Tao chunk**: Gan ID duy nhat cho tung chunk de theo doi thu tu.
3. **Tra ve list Send**: `Send("summarize_chunk", chunk)` tra ve list, LangGraph se thuc thi node `summarize_chunk` **song song** cho tung chunk.

Ham nay duoc su dung nhu router cua `add_conditional_edges`. Conditional edge thong thuong tra ve ten node (chuoi), nhung tra ve doi tuong `Send` thi tro thanh phan nhanh song song dong.

**Node tom tat**

```python
def summarize_chunk(chunk):
    chunk_id = chunk["id"]
    chunk = chunk["chunk"]

    response = llm.invoke(
        f"""
        Please summarize the following text.

        Text: {chunk}
        """
    )
    summary = f"[Chunk {chunk_id}] {response.content}"
    return {
        "summaries": [summary],
    }
```

Diem dang chu y:

- **Tham so khong phai `state`**: Node duoc goi qua `Send` nhan truc tiep du lieu truyen tu `Send` lam tham so, khong phai State thong thuong. O day la dictionary `{"id": ..., "chunk": ...}`.
- **Tra ve trong list**: `"summaries": [summary]` bat buoc phai boc trong list. Vi reducer `operator.add` noi cac list, nen gia tri don cung phai boc trong list de hoat dong dung.
- **Tien to `[Chunk N]`**: Gan ID chunk cho moi tom tat de sau nay co the xac dinh thu tu.

**Cau hinh do thi: Conditional edges**

```python
graph_builder.add_node("summarize_chunk", summarize_chunk)

graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", END)
```

Tham so thu ba `["summarize_chunk"]` cua `add_conditional_edges` chi ro danh sach node co the dat den tu conditional edge nay. LangGraph su dung thong tin nay khi xac thuc do thi.

#### Diem thuc hanh

1. Thay doi kich thuoc chunk (hien tai 500) va quan sat su thay doi chat luong tom tat.
2. Thu hien thuc xu ly tuan tu (vong for) thay vi `Send` roi so sanh thoi gian thuc thi.
3. Thu nghiem voi reducer khac. Vi du co the su dung ham tuy chinh thay vi `operator.add`.
4. Doi model LLM tu `gpt-4o-mini` sang `gpt-4o` va so sanh chat luong tom tat.

---

### 15.3 Node tao sketch thumbnail (Thumbnail Sketcher Nodes)

**Commit:** `13e2688` "15.3 Thumbnail Sketcher Nodes"

#### Chu de va Muc tieu

Tong hop cac tom tat rieng le thanh mot tom tat tong hop (`mega_summary`) roi dua tren do tao **song song** 5 sketch thumbnail khac nhau. Su dung OpenAI Image Generation API (`gpt-image-1`) va ap dung pattern Map-Reduce cho viec tao hinh anh.

#### Giai thich khai niem cot loi

**Mega Summary (Tom tat tong hop)**: Buoc Reduce cua Map-Reduce. Gop nhieu tom tat chunk thanh mot tom tat duy nhat chua noi dung cot loi cua toan bo video. Tom tat nay la nen tang tao thumbnail.

**OpenAI Image Generation API**: Su dung model `gpt-image-1` de tao hinh anh tu prompt van ban. Dieu khien chat luong tao (`low`/`medium`/`high`) bang tham so `quality` va muc loc noi dung bang tham so `moderation`.

**Map-Reduce kep**: Du an nay su dung Map-Reduce **hai lan**.
1. Lan 1: Chunk van ban -> Tom tat song song -> Tom tat tong hop
2. Lan 2: Tom tat tong hop -> Tao thumbnail song song -> Nguoi dung chon

#### Phan tich code

**Mo rong State**

```python
class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]    # Them moi
    thumbnail_sketches: Annotated[list[str], operator.add]   # Them moi
    final_summary: str                                        # Them moi
```

Cac truong them moi:
- **`thumbnail_prompts`**: Luu prompt da dung tao tung thumbnail. De sau nay tai su dung prompt cua thumbnail nguoi dung chon.
- **`thumbnail_sketches`**: Luu duong dan file hinh anh thumbnail da tao.
- **`final_summary`**: Luu van ban tom tat tong hop.

**Node tom tat tong hop (mega_summary)**

```python
def mega_summary(state: State):
    all_summaries = "\n".join(state["summaries"])

    prompt = f"""
        You are given multiple summaries of different chunks from a video transcription.

        Please create a comprehensive final summary that combines all the key points.

        Individual summaries:

        {all_summaries}
    """

    response = llm.invoke(prompt)

    return {
        "final_summary": response.content,
    }
```

Node nay chay sau khi tat ca `summarize_chunk` hoan thanh. Trong `state["summaries"]` da tich luy tat ca tom tat chunk nho reducer `operator.add`. Gop chung thanh mot prompt va yeu cau LLM tao tom tat tong hop.

**Artist dispatcher: Phan nhanh song song lan hai**

```python
def dispatch_artists(state: State):
    return [
        Send(
            "generate_thumbnails",
            {
                "id": i,
                "summary": state["final_summary"],
            },
        )
        for i in [1, 2, 3, 4, 5]
    ]
```

Cung pattern nhu `dispatch_summarizers`, nhung lan nay tao 5 tac vu co dinh. Moi tac vu nhan cung `final_summary` nhung co `id` khac nhau. Nho tinh chat xac suat cua LLM, tu cung tom tat se tao ra nhung concept hinh anh khac nhau moi lan.

**Node tao thumbnail**

```python
def generate_thumbnails(args):
    concept_id = args["id"]
    summary = args["summary"]

    prompt = f"""
    Based on this video summary, create a detailed visual prompt for a YouTube thumbnail.

    Create a detailed prompt for generating a thumbnail image that would attract viewers. Include:
        - Main visual elements
        - Color scheme
        - Text overlay suggestions
        - Overall composition

    Summary: {summary}
    """

    response = llm.invoke(prompt)
    thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    filename = f"thumbnail_{concept_id}.jpg"

    with open(filename, "wb") as file:
        file.write(image_bytes)

    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [filename],
    }
```

Node nay hoat dong theo hai buoc:

1. **Tao prompt**: Yeu cau LLM (`gpt-4o-mini`) tao prompt hinh anh dua tren tom tat video. Chi thi bao gom yeu to hinh anh chinh, phoi mau, text overlay, bo cuc tong the.
2. **Tao hinh anh**: Truyen prompt da tao cho model `gpt-image-1` de tao hinh anh thuc te.

Cac tham so chinh:
- **`quality="low"`**: O buoc sketch nen tao chat luong thap cho nhanh. Tiet kiem chi phi va thoi gian.
- **`moderation="low"`**: Dat loc noi dung long de cho phep ket qua sang tao.
- **`size="auto"`**: De model tu quyet dinh kich thuoc hinh anh.
- **`b64_json`**: Hinh anh tra ve dang Base64 encoded JSON. Giai ma va luu thanh file JPG.

**Cau hinh do thi hoan chinh**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", END)

graph = graph_builder.compile()
```

Luong do thi:
1. `summarize_chunk` -> `mega_summary`: Khi tat ca tom tat song song hoan thanh, chuyen den node tom tat tong hop.
2. `mega_summary` -> `dispatch_artists` -> `generate_thumbnails` x 5: Khi tom tat tong hop xong, tao 5 thumbnail song song.

#### Diem thuc hanh

1. Thay doi so luong thumbnail tao (hien tai 5).
2. Sua prompt tao hinh anh de yeu cau phong cach cu the (minh hoa, anh thuc, toi gian...).
3. Thay doi tham so `quality` thanh `"medium"` hoac `"high"` va so sanh chat luong va thoi gian tao.
4. So sanh 5 thumbnail da tao de xac nhan tinh chat xac suat cua prompt.

---

### 15.4 Vong lap phan hoi cua nguoi (Human Feedback)

**Commit:** `910cdef` "15.4 Human Feedback"

#### Chu de va Muc tieu

Hien thuc pattern **Human-in-the-Loop** cho phep nguoi dung chon mot trong 5 sketch thumbnail ma AI tao va cung cap phan hoi bo sung. Su dung ham `interrupt` va lop `Command` cua LangGraph cung voi checkpointer `InMemorySaver`.

#### Giai thich khai niem cot loi

**Human-in-the-Loop (HITL)**: Pattern chen phan doan hoac dau vao cua con nguoi vao giua workflow AI. Khong tu dong hoa hoan toan, ma nhan duoc su xem xet va phe duyet cua con nguoi tai cac diem quyet dinh quan trong. Dieu nay can thiet de nang cao chat luong dau ra cua AI va phan anh chinh xac y dinh cua nguoi dung.

**interrupt()**: Ham **tam dung** thuc thi do thi trong LangGraph. Khi ham nay duoc goi, do thi ngung chay va trang thai hien tai duoc luu vao checkpointer. Sau khi nhan phan hoi tu nguoi dung, tiep tuc thuc thi bang `Command(resume=response)`.

**Command**: Lop truyen phan hoi cua nguoi dung den do thi da tam dung va tiep tuc thuc thi.

**InMemorySaver**: Checkpointer luu trang thai do thi trong bo nho. `interrupt` bat buoc phai co checkpointer moi hoat dong. Vi khi tam dung roi tiep tuc do thi, phai luu trang thai truoc do o dau day.

**thread_id**: ID duy nhat de xac dinh cung mot cuoc hoi thoai/phien. Checkpointer su dung `thread_id` lam khoa de luu va khoi phuc trang thai.

#### Phan tich code

**Import va thiet lap moi**

```python
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
```

**Mo rong State**

```python
class State(TypedDict):
    # ... cac truong hien co ...
    user_feedback: str    # Phan hoi chinh sua cua nguoi dung
    chosen_prompt: str    # Prompt cua thumbnail nguoi dung chon
```

**Node phan hoi nguoi dung**

```python
def human_feedback(state: State):
    answer = interrupt(
        {
            "chosen_thumbnail": "Which thumbnail do you like the most?",
            "feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
        }
    )
    user_feedback = answer["user_feedback"]
    chosen_prompt = answer["chosen_prompt"]
    return {
        "user_feedback": user_feedback,
        "chosen_prompt": state["thumbnail_prompts"][chosen_prompt - 1],
    }
```

Cach hoat dong cua ham `interrupt()`:

1. Dictionary truyen vao `interrupt()` la **cau hoi/thong bao huong dan** hien thi cho nguoi dung. Dictionary nay nam trong gia tri tra ve cua do thi, client (UI) co the hien thi.
2. Thuc thi do thi **dung tai day**. Toan bo trang thai den thoi diem nay (duong dan 5 hinh thumbnail, prompt...) duoc luu vao checkpointer.
3. Khi nguoi dung cung cap phan hoi, `interrupt()` tra ve phan hoi do va thuc thi tiep tuc.

Trong gia tri tra ve, `state["thumbnail_prompts"][chosen_prompt - 1]` lay prompt goc tuong ung voi so nguoi dung chon (1-5). Prompt nay la nen tang tao thumbnail HD.

**Node tao thumbnail HD**

```python
def generate_hd_thumbnail(state: State):
    chosen_prompt = state["chosen_prompt"]
    user_feedback = state["user_feedback"]

    prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail
    prompt and create an enhanced version that incorporates the user's specific feedback.

    ORIGINAL PROMPT:
    {chosen_prompt}

    USER FEEDBACK TO INCORPORATE:
    {user_feedback}

    Create an enhanced prompt that:
        1. Maintains the core concept from the original prompt
        2. Specifically addresses and implements the user's feedback requests
        3. Adds professional YouTube thumbnail specifications:
            - High contrast and bold visual elements
            - Clear focal points that draw the eye
            - Professional lighting and composition
            - Optimal text placement and readability with generous padding from edges
            - Colors that pop and grab attention
            - Elements that work well at small thumbnail sizes
            - IMPORTANT: Always ensure adequate white space/padding between any text
              and the image borders
    """

    response = llm.invoke(prompt)
    final_thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=final_thumbnail_prompt,
        quality="high",           # Doi sang chat luong cao!
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    with open("thumbnail_final.jpg", "wb") as file:
        file.write(image_bytes)
```

Diem thiet ke cua node nay:

1. **Tang cuong prompt**: Phan anh phan hoi vao prompt goc nguoi dung chon va them cac thong so thumbnail YouTube chuyen nghiep (do tuong phan cao, diem tap trung thi giac, khoang cach text...).
2. **`quality="high"`**: Khac voi `"low"` o buoc sketch, day la san pham cuoi cung nen tao chat luong cao.
3. **Ten file co dinh**: Luu thanh `thumbnail_final.jpg` de chi ro la san pham cuoi cung.

**Bien dich do thi (them checkpointer)**

```python
graph = graph_builder.compile(checkpointer=memory)
```

Truyen `checkpointer=memory` de kich hoat chuc nang luu tru trang thai cho do thi. Khong co dieu nay, `interrupt()` se khong hoat dong.

**Thuc thi: Goi 2 buoc**

```python
# Buoc 1: Bat dau thuc thi do thi (tam dung tai human_feedback)
config = {
    "configurable": {
        "thread_id": "1",
    },
}

graph.invoke(
    {"video_file": "netherlands.mp4"},
    config=config,
)

# Buoc 2: Tiep tuc voi phan hoi nguoi dung
response = {
    "user_feedback": "Make sure the fella is smiling, remove any mention of audible, "
                     "or any logo, and give it a photo realistic, 3d style.",
    "chosen_prompt": 1,
}

graph.invoke(
    Command(resume=response),
    config=config,
)
```

`invoke` lan dau chay tu `extract_audio` -> `transcribe_audio` -> `summarize_chunk` x N -> `mega_summary` -> `generate_thumbnails` x 5 -> `human_feedback` roi tam dung. Nguoi dung xem 5 thumbnail da tao roi cung cap so thu tu va phan hoi chinh sua.

`invoke` lan hai truyen `Command(resume=response)`, `interrupt()` tra ve `response` va node `human_feedback` hoan thanh. Tiep theo `generate_hd_thumbnail` chay va tao thumbnail chat luong cao cuoi cung.

> **Quan trong**: Ca hai lan goi `invoke` phai su dung **cung `config`** (cung `thread_id`). Vi checkpointer quan ly trang thai dua tren `thread_id` lam khoa.

#### Diem thuc hanh

1. Thu cac phan hoi da dang ("sang hon", "xoa text", "phong cach minh hoa"...).
2. Thay doi `thread_id` va chay phien rieng biet.
3. Sua thong bao truyen vao `interrupt()` de cung cap huong dan chi tiet hon cho nguoi dung.
4. Doi checkpointer thanh `SqliteSaver` de thu nghiem luu tru vinh vien.

---

### 15.5 Tao thumbnail HD va trien khai production

**Commit:** `257d1b8` "15.5 HD Thumbnail Generation"

#### Chu de va Muc tieu

Chuyen doi code phat trien trong Jupyter Notebook thanh **dang co the trien khai production**. Tao module doc lap `graph.py` va cau hinh de phuc vu bang **LangGraph CLI** thong qua file cau hinh `langgraph.json`.

#### Giai thich khai niem cot loi

**LangGraph CLI (langgraph-cli)**: Cong cu CLI cho phep chay do thi LangGraph tren server local hoac trien khai len cloud. Chay `langgraph dev` de khoi dong server phat trien local, cho phep tuong tac voi do thi qua REST API.

**langgraph.json**: File cau hinh cua LangGraph CLI. Dinh nghia do thi duoc dinh nghia trong file nao, phu thuoc la gi, bien moi truong tai tu dau.

**Chuyen tu Notebook sang module**: O giai doan phat trien, Jupyter Notebook tien loi, nhung khi trien khai phai chuyen sang file `.py`. Trong qua trinh nay, chinh ly cau truc code va module hoa.

#### Phan tich code

**File cau hinh langgraph.json**

```json
{
    "dependencies": [
        "graph.py"
    ],
    "graphs": {
        "mr_thumbs": "./graph.py:graph"
    },
    "env": ".env"
}
```

- **`dependencies`**: Chi dinh file phu thuoc cua du an. O day chi dinh chinh `graph.py` la phu thuoc. Thong thuong neu co `pyproject.toml` thi phu thuoc se tu dong duoc cai dat.
- **`graphs`**: Dinh nghia do thi can phuc vu theo cap ten-duong dan.
  - `"mr_thumbs"`: Ten dich vu cua do thi. Truy cap qua ten nay trong API endpoint.
  - `"./graph.py:graph"`: Chi dinh duong dan file va ten bien, phan cach bang dau hai cham. Phuc vu bien `graph` trong file `graph.py`.
- **`env`**: Duong dan file `.env` de tai bien moi truong.

**graph.py: Code production hoan chinh**

```python
graph = graph_builder.compile(name="mr_thumbs")
```

Diem khac biet so voi phien ban Notebook:
- Bo `checkpointer=memory` vi LangGraph CLI tu dong quan ly checkpointer.
- Dat `name="mr_thumbs"` de tang tinh nhan dien cho do thi.

**Cau truc tong the cua graph.py**:

```python
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
import operator
import base64

llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[str], operator.add]
    final_summary: str
    user_feedback: str
    chosen_prompt: str

# --- Cac ham node (extract_audio, transcribe_audio, ...) ---

# --- Cau hinh do thi ---
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", END)

graph = graph_builder.compile(name="mr_thumbs")
```

**Cach chay LangGraph CLI**

```bash
# Chay server phat trien
langgraph dev

# Hoac chay che do in-memory
langgraph dev --no-browser
```

Khi chay, server local khoi dong va co the xem do thi truc quan, thuc thi va cung cap phan hoi tai diem `interrupt` trong LangGraph Studio UI.

#### Diem thuc hanh

1. Chay lenh `langgraph dev` de khoi dong server local va xem do thi trong LangGraph Studio.
2. Nhap `video_file` trong Studio UI va thuc thi do thi.
3. Cung cap phan hoi qua Studio UI tai diem `interrupt` va kiem tra ket qua.
4. Thu them node moi vao do thi (vi du: them watermark, resize hinh anh).

---

## 3. Tong hop cot loi chuong

### Cac pattern cot loi LangGraph

| Pattern | Mo ta | Truong hop su dung |
|---------|-------|---------------------|
| **StateGraph** | Workflow do thi lay trang thai lam trung tam | Nen tang cua moi du an LangGraph |
| **Send (Map-Reduce)** | Phan nhanh song song dong va thu thap ket qua | Tom tat song song chunk van ban, tao nhieu hinh anh |
| **Annotated reducer** | Chien luoc gop ket qua tu cac node song song | Tich luy list bang `operator.add` |
| **interrupt / Command** | Pattern Human-in-the-Loop | Lua chon cua nguoi dung, thu thap phan hoi |
| **Checkpointer** | Luu tru vinh vien trang thai do thi | Ho tro interrupt, quan ly phien |
| **conditional_edges** | Dinh tuyen dieu kien | Phan nhanh dong dua tren Send |

### Cong cu va API ben ngoai da su dung

| Cong cu/API | Muc dich |
|-------------|----------|
| **ffmpeg** | Trich xuat audio tu video, dieu chinh toc do |
| **OpenAI Whisper** (`whisper-1`) | Chuyen giong noi -> van ban |
| **OpenAI Chat** (`gpt-4o-mini`) | Tom tat van ban, tao prompt |
| **OpenAI Image** (`gpt-image-1`) | Tao hinh anh tu van ban |
| **LangGraph CLI** | Trien khai dich vu do thi |

### Nguyen tac thiet ke kien truc

1. **Tang do phuc tap dan dan**: Bat dau tu do thi don gian 2 node, mo rong dan thanh do thi phuc tap 7 node.
2. **Thiet ke tiet kiem chi phi**: Su dung `quality="low"` o buoc sketch, chi ap dung `quality="high"` cho san pham cuoi de toi uu chi phi API.
3. **Module hoa**: Thiet ke moi node co mot trach nhiem ro rang, viec sua doi node rieng le khong anh huong den node khac.
4. **Human-in-the-Loop**: Khong tu dong hoa hoan toan, ma hien thuc workflow thuc te phan anh phan doan cua nguoi dung tai diem quyet dinh cot loi.

---

## 4. Bai tap thuc hanh

### Bai tap 1: Co ban - Them ho tro da ngon ngu (Do kho: 2/5)

Sua node `transcribe_audio`, them truong `language` vao State de nguoi dung co the chi dinh ngon ngu cua video. Test voi cac ngon ngu khac nhau nhu tieng Han (`"ko"`), tieng Nhat (`"ja"`)...

```python
class State(TypedDict):
    video_file: str
    language: str  # Them moi
    # ...
```

### Bai tap 2: Trung cap - Cai thien chat luong tom tat (Do kho: 3/5)

Hien tai `textwrap.wrap` don thuan chia van ban theo so ky tu. Cai tien `dispatch_summarizers` de chia chunk theo so token su dung `tiktoken`. Ngoai ra, them mot it overlap giua cac chunk de giam mat ngu canh.

### Bai tap 3: Trung cap - Chon phong cach thumbnail (Do kho: 3/5)

Them `interrupt` bo sung **truoc** node `human_feedback` de nguoi dung chon phong cach thumbnail mong muon (photo realism, minh hoa, toi gian, 3D rendering...) truoc. Phan anh phong cach da chon vao prompt cua `generate_thumbnails`.

### Bai tap 4: Nang cao - Vong lap phan hoi lap lai (Do kho: 4/5)

Hien tai chi nhan phan hoi mot lan. Hien thuc **vong lap phan hoi lap lai** de khi nguoi dung khong hai long voi thumbnail HD cuoi cung, co the nhan them phan hoi va tao lai. Thiet ke conditional edge de lap chu ky `generate_hd_thumbnail` -> `interrupt` -> `generate_hd_thumbnail` cho den khi nguoi dung phan hoi "hoan thanh".

### Bai tap 5: Nang cao - Mo rong toan bo pipeline (Do kho: 5/5)

Them cac tinh nang sau de mo rong pipeline:

1. **Nhap URL YouTube**: Thay vi `video_file`, nhan URL YouTube va them node tu dong tai xuong bang `yt-dlp`...
2. **Che do A/B testing**: Tao 2 thumbnail cuoi cung de cung cap cho A/B test.
3. **Hau xu ly hinh anh**: Su dung thu vien Pillow de tu dong them watermark logo kenh vao thumbnail da tao.
4. **Luu ket qua**: Them node chinh ly va luu tat ca ket qua trung gian (tom tat, prompt, hinh anh) thanh file JSON.

---

## Phu luc: Tham khao API chinh

### LangGraph StateGraph API

```python
from langgraph.graph import END, START, StateGraph

# Tao do thi
graph_builder = StateGraph(State)

# Them node
graph_builder.add_node("ten", ham)

# Them edge (tuan tu)
graph_builder.add_edge("node_bat_dau", "node_ket_thuc")

# Conditional edge (su dung ham router)
graph_builder.add_conditional_edges("node_bat_dau", ham_router, ["node_co_the_1", "node_co_the_2"])

# Bien dich
graph = graph_builder.compile(checkpointer=checkpointer, name="ten_do_thi")

# Thuc thi
result = graph.invoke(trang_thai_ban_dau, config={"configurable": {"thread_id": "1"}})
```

### LangGraph Send API

```python
from langgraph.types import Send

# Tra ve list Send trong ham dispatcher
def dispatcher(state: State):
    return [Send("ten_node", du_lieu) for du_lieu in danh_sach_du_lieu]
```

### LangGraph interrupt / Command

```python
from langgraph.types import interrupt, Command

# Tam dung thuc thi trong node
def my_node(state: State):
    answer = interrupt({"cau_hoi": "Thong bao hien thi cho nguoi dung"})
    # answer la gia tri truyen tu Command(resume=phan_hoi)
    return {"field": answer["key"]}

# Tiep tuc thuc thi
graph.invoke(Command(resume=phan_hoi), config=config)
```

### OpenAI Image Generation API

```python
from openai import OpenAI
import base64

client = OpenAI()
result = client.images.generate(
    model="gpt-image-1",
    prompt="mo ta hinh anh",
    quality="low" | "medium" | "high",
    moderation="low" | "auto",
    size="auto" | "1024x1024" | "1792x1024",
)
image_bytes = base64.b64decode(result.data[0].b64_json)
```
