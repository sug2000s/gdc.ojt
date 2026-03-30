# Chapter 15: LangGraph Project Pipeline — Kich ban bai giang

---

## Mo dau (2 phut)

> Duoc roi, o Chapter 14 chung ta da xay dung chatbot va hoc cac pattern cot loi cua LangGraph.
>
> Hom nay chung ta se **ket hop tat ca** thanh mot pipeline hoan chinh, end-to-end.
>
> Du an la **Blog Post Generator** — pipeline tu dong viet bai blog.
>
> ```
> 15.0 Setup
> 15.1 Pipeline co ban          (Do thi tuyen tinh 2 node)
> 15.2 Node viet song song      (Send API + Map-Reduce)
> 15.3 Human-in-the-loop        (interrupt + Command resume)
> 15.4 Pipeline hoan chinh      (Tat ca pattern ket hop)
> 15.5 Cau truc deploy san xuat (langgraph.json, graph.py)
> ```
>
> Giong nhu Chapter 14, moi phan xay dung tren phan truoc.
> Cuoi cung, chung ta se co pipeline tu dong nghien cuu chu de, viet song song cac phan, nhan review tu nguoi dung, va tao ban cuoi cung.
>
> Bat dau thoi.

---

## 15.0 Setup & Environment (2 phut)

### Truoc khi chay cell

> Dau tien, kiem tra moi truong.
> Chung ta load API key tu `.env` va kiem tra phien ban `langgraph` va `langchain`.
>
> Cung moi truong voi Chapter 14. Khong can package bo sung.

### Sau khi chay cell

> Neu API key va phien ban hien thi dung, chung ta da san sang.
> Neu bi loi, kiem tra `uv sync` hoac `pip install`.

---

## 15.1 Pipeline co ban — Do thi tuyen tinh 2 node (8 phut)

### Khai niem

> Chung ta bat dau voi pipeline don gian nhat.
>
> Cau truc:
> ```
> START → get_topic_info → write_draft → END
> ```
>
> **Khac gi voi chatbot?**
> O Chapter 14, chatbot su dung `MessagesState` — tin nhan tich luy trong cuoc hoi thoai.
> Pipeline su dung **`TypedDict` de thiet ke state rieng**.
> Vi pipeline khong phai cuoc hoi thoai — ma la **du lieu bien doi tung buoc**.
>
> Xem state:
> ```python
> class PipelineState(TypedDict):
>     topic: str             # Dau vao: chu de blog
>     background_info: str   # Dau ra buoc 1: nghien cuu nen
>     draft: str             # Dau ra buoc 2: ban nhap
> ```
>
> Moi node dien vao mot truong trong state.
> `get_topic_info` dien `background_info`, `write_draft` dien `draft`.
> Du lieu chay qua nhu mot duong ong.

### Giai thich code (truoc khi chay)

> Xem code.
>
> ```python
> def get_topic_info(state: PipelineState):
>     topic = state["topic"]
>     response = llm.invoke(f"Provide a concise background summary about: {topic}...")
>     return {"background_info": response.content}
> ```
>
> Pattern cua ham node giong het Chapter 13 va 14:
> 1. Lay du lieu can thiet tu state
> 2. Goi LLM
> 3. Tra ket qua vao truong state
>
> `write_draft` cung theo pattern tuong tu. Nhan `background_info` va tao `draft`.
>
> Lap rap do thi cung quen thuoc:
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_draft", write_draft)
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_edge("get_topic_info", "write_draft")
> graph_builder.add_edge("write_draft", END)
> ```
>
> Canh tuyen tinh noi cac node theo thu tu. Pipeline co ban nhat.
>
> Chay thoi.

### Sau khi chay cell

> Xem ket qua.
> `background_info` chua thong tin nghien cuu ve chu de,
> va `draft` chua bai blog duoc viet dua tren nghien cuu do.
>
> Chi 2 node da cho chung ta pipeline "nghien cuu → viet".
>
> Nhung co van de — bai blog la **mot khoi lon**.
> Thuc te, chung ta muon viet nhieu phan rieng biet.
> Vi vay tiep theo chung ta them **viet song song**.

---

## 15.2 Node viet song song — Send API + Map-Reduce (15 phut)

### Khai niem

> Phan nay la **diem noi bat** cua Chapter 15.
>
> Chung ta chia bai blog thanh N phan, **viet moi phan dong thoi**, roi gop lai.
>
> ```
> get_topic_info → dispatch_writers ──→ write_section (x3) → combine_sections
>                                  ├─→ write_section
>                                  └─→ write_section
> ```
>
> Ba khai niem moi xuat hien:
>
> 1. **`Send` API** — gui dong cac node song song. Chung ta da hoc o Chapter 13.
> 2. **`Annotated[list[str], operator.add]`** — reducer tu dong gop ket qua song song
> 3. **`with_structured_output()`** — LLM tao dau ra co cau truc dang Pydantic model
>
> Di qua tung cai.

### Giai thich with_structured_output

> Dau tien, `with_structured_output`.
>
> Truoc gio, phan hoi cua LLM luon la chuoi ky tu — chung ta lay text qua `response.content`.
>
> Nhung chung ta muon **tieu de phan va diem chinh** cua bai blog duoi dang du lieu co cau truc.
>
> Nen chung ta dinh nghia hinh dang dau ra bang Pydantic model:
>
> ```python
> class SectionPlan(BaseModel):
>     title: str = Field(description="Section title")
>     key_points: list[str] = Field(description="Key points to cover")
>
> class BlogOutline(BaseModel):
>     sections: list[SectionPlan] = Field(description="List of sections")
> ```
>
> Roi bao LLM xuat ra theo format nay:
>
> ```python
> planner = llm.with_structured_output(BlogOutline)
> outline = planner.invoke("Tao outline voi 3 phan")
> ```
>
> Bay gio `outline.sections` la danh sach cac doi tuong `SectionPlan`.
> Khong can phan tich chuoi — truy cap truc tiep `.title` va `.key_points`!
>
> Day la dieu bien LLM thanh **cong cu lap trinh duoc**.

### Giai thich dispatch_writers — Day KHONG phai la node!

> Bay gio, day la phan rat quan trong.
>
> **`dispatch_writers` KHONG phai la node!**
>
> Xem code:
>
> ```python
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> ```
>
> No nam trong `add_conditional_edges` lam tham so thu hai.
> Day la vi tri giong nhu `tools_condition` o Chapter 14.
> No la **ham dinh tuyen canh (routing function)**.
>
> No khong duoc dang ky bang `add_node`. No khong hien thi la node trong do thi.
>
> **No lam gi?**
>
> Sau khi `get_topic_info` ket thuc, LangGraph hoi "di dau tiep?" va
> goi `dispatch_writers` de quyet dinh.
>
> Ham nay tra ve danh sach cac doi tuong `Send`:
>
> ```python
> def dispatch_writers(state: PipelineState):
>     planner = llm.with_structured_output(BlogOutline)
>     outline = planner.invoke("Tao outline")
>     return [
>         Send("write_section", {
>             "topic": state["topic"],
>             "section_title": section.title,
>             "section_key_points": section.key_points,
>         })
>         for section in outline.sections
>     ]
> ```
>
> Moi doi tuong `Send` tao mot **phien ban thuc thi doc lap** cua `write_section`.
> Neu co 3 phan, 3 `write_section` chay **dong thoi**.
>
> Vi du nhu the nay:
> - `dispatch_writers` la **trung tam phan loai buu kien**
> - No phan loai goi hang (du lieu phan) va gui moi goi cho mot nguoi giao hang khac (`write_section`)
> - Trung tam phan loai khong giao hang (no khong phai node)

### Giai thich pattern Map-Reduce

> Ket qua song song duoc gop nhu the nao?
>
> Xem dinh nghia state:
>
> ```python
> class PipelineState(TypedDict):
>     sections: Annotated[list[str], operator.add]  # Reducer!
> ```
>
> `operator.add` la reducer.
> Moi `write_section` tra ve `{"sections": ["noi dung phan"]}`,
> va LangGraph tu dong **noi cac danh sach lai**.
>
> 3 node moi cai tra ve 1 phan tu → `sections` co 3 phan tu.
>
> Day la pattern **Map-Reduce**:
> - **Map** = gui bang `Send`, xu ly tung cai song song
> - **Reduce** = gop ket qua bang reducer `operator.add`
>
> Node `combine_sections` nhan `state["sections"]` (da gop du 3) va
> ghep thanh mot `combined_draft` duy nhat.

### Chay code

> Chay thoi.

### Sau khi chay cell

> Xem ket qua:
> - "3 sections written in parallel!" — 3 phan duoc viet dong thoi
> - Moi phan bat dau bang cau truc `## Tieu de`
> - `combined_draft` chua ban nhap da gop
>
> Day khong phai mot node viet 3 lan lien tiep.
> `Send` gui 3 phien ban chay **song song**.
>
> Trong san xuat, ngay ca voi 10 hay 20 phan, pattern nay van mo rong duoc.

---

## 15.3 Human-in-the-loop — interrupt + Command resume (10 phut)

### Khai niem

> O 15.2, ban nhap duoc tao tu dong.
> Nhung khong nen xuat ban truc tiep — **nguoi can review**.
>
> Chung ta ap dung `interrupt` va `Command` tu Chapter 14.3 vao day.
>
> ```
> write_draft → human_feedback(interrupt!) → [phan hoi nguoi dung] → finalize_post → END
> ```
>
> Ba diem chinh:
> - `interrupt(value)` — tam dung do thi va hien thi ban nhap cho nguoi dung
> - `Command(resume=feedback)` — tiep tuc voi phan hoi cua nguoi dung
> - `SqliteSaver` — checkpointer bao toan state giua cac lan tam dung (bat buoc!)

### Giai thich code (truoc khi chay)

> Xem state:
>
> ```python
> class ReviewState(TypedDict):
>     topic: str
>     draft: str
>     feedback: str
>     final_post: str
> ```
>
> Them hai truong `feedback` va `final_post`.
>
> Node `human_feedback` la then chot:
>
> ```python
> def human_feedback(state: ReviewState):
>     feedback = interrupt(
>         f"DRAFT FOR REVIEW\n{state['draft'][:500]}...\n"
>         f"Please provide your feedback:"
>     )
>     return {"feedback": feedback}
> ```
>
> Khi `interrupt()` duoc goi:
> 1. Do thi **tam dung** — state tu dong luu vao SQLite
> 2. Tham so cua `interrupt()` duoc tra ve cho nguoi dung (xem truoc ban nhap)
> 3. Doi cho den khi nguoi dung cung cap phan hoi
>
> De tiep tuc:
> ```python
> result = graph.invoke(
>     Command(resume="Make it more concise. Add code examples."),
>     config=config,
> )
> ```
>
> Gia tri trong `Command(resume=...)` tro thanh **gia tri tra ve** cua `interrupt()`.
> Tuc la `feedback = "Make it more concise..."` duoc luu vao state.
>
> Node `finalize_post` gui ban nhap + phan hoi cho LLM de tao ban cuoi cung.
>
> Checkpointer la **bat buoc**:
> ```python
> conn = sqlite3.connect("pipeline_review.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> Su dung `interrupt` ma khong co checkpointer se bao loi.
> State phai duoc luu o dau do trong khi do thi tam dung.

### Chay cell — Buoc 1: Viet ban nhap

> Chay cell dau tien.

### Sau khi chay (Buoc 1)

> `snapshot.next` la `('human_feedback',)`.
> Do thi dang tam dung tai node `human_feedback` vi `interrupt`.
>
> Xem truoc ban nhap duoc hien thi, dang doi phan hoi.

### Chay cell — Buoc 2: Cung cap phan hoi

> Tiep tuc voi `Command(resume="Make it more concise. Add a code example.")`.

### Sau khi chay (Buoc 2)

> `Status: COMPLETE` — pipeline hoan thanh.
> `snapshot.next` la tuple rong `()`.
>
> `final_post` chua ban cuoi cung da duoc sua theo phan hoi.
> No se ngan gon hon va co them vi du code.
>
> Day la cau truc co ban cua hop tac AI + nguoi.
> AI tao ban nhap, nguoi chi dao huong di, AI hoan thien.

---

## 15.4 Pipeline hoan chinh — Tat ca pattern ket hop (10 phut)

### Khai niem

> Bay gio chung ta **gop tat ca** tu 15.1 den 15.3 thanh mot pipeline.
>
> ```
> [START]
>    |
> [get_topic_info] ───── LLM nghien cuu nen
>    |
> [dispatch_writers] ──── with_structured_output + Send API (router!)
>    |
> [write_section] x N ─── Viet song song (Map)
>    |
> [combine_sections] ──── Gop ban nhap (Reduce)
>    |
> [human_feedback] ───── interrupt() review
>    |
> [finalize_post] ────── Ap dung phan hoi, ban cuoi cung
>    |
> [END]
> ```
>
> Code trong dai nhung moi phan deu da hoc roi.
> Khong co gi moi. Chung ta chi **ket hop** chung lai.

### Giai thich state

> State ke tat ca cau chuyen:
>
> ```python
> class FullPipelineState(TypedDict):
>     topic: str                                      # Dau vao
>     background_info: str                             # Tu 15.1
>     sections: Annotated[list[str], operator.add]     # Tu 15.2 (Map-Reduce)
>     combined_draft: str                              # Tu 15.2
>     feedback: str                                    # Tu 15.3 (HITL)
>     final_post: str                                  # Tu 15.3
> ```
>
> Sau truong the hien toan bo luong du lieu cua pipeline.
> Moi node dien vao tung cai mot.

### Giai thich lap rap do thi

> Xem code lap rap do thi:
>
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_section", write_section)
> graph_builder.add_node("combine_sections", combine_sections)
> graph_builder.add_node("human_feedback", human_feedback)
> graph_builder.add_node("finalize_post", finalize_post)
> ```
>
> **5 node** duoc dang ky.
> `dispatch_writers` **khong co o day**! Vi no khong phai node.
>
> ```python
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> graph_builder.add_edge("write_section", "combine_sections")
> graph_builder.add_edge("combine_sections", "human_feedback")
> graph_builder.add_edge("human_feedback", "finalize_post")
> graph_builder.add_edge("finalize_post", END)
> ```
>
> `dispatch_writers` duoc dang ky lam **ham dinh tuyen** trong `add_conditional_edges`.
> Sau khi `get_topic_info` chay, router nay tra ve cac doi tuong `Send` de gui cac node song song.
>
> Checkpointer cung bat buoc:
> ```python
> conn = sqlite3.connect("pipeline_full.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```

### Chay cell — Buoc 1: Thuc thi pipeline

> Chay thoi. Chu de la "Building AI Agents with LangGraph".

### Sau khi chay (Buoc 1)

> Xem dau ra:
> - `Dispatching 3 parallel writers...` — 3 phan duoc gui song song
> - `Paused at: ('human_feedback',)` — tam dung tai buoc review
> - `Sections written: 3` — 3 phan hoan thanh
> - Xem truoc ban nhap duoc hien thi
>
> Den day tu dong: nghien cuu chu de → tao outline → viet 3 phan song song → gop ban nhap.
> Bay gio dang doi nguoi review.

### Chay cell — Buoc 2: Cung cap phan hoi

> Dua phan hoi:
> ```python
> Command(resume="Add a practical code example in each section. Make the tone more engaging.")
> ```

### Sau khi chay (Buoc 2)

> `Status: COMPLETE` — hoan thanh!
>
> Ban cuoi cung da duoc ap dung phan hoi.
> Vi du code duoc them va giong van thay doi.
>
> Day la **pipeline viet AI hoan chinh**:
> 1. AI nghien cuu chu de
> 2. Len ke hoach cau truc
> 3. Viet cac phan song song
> 4. Nguoi review
> 5. AI ap dung phan hoi va tao ban cuoi cung
>
> 6 don vi (chinh xac: 5 node + 1 router), ket hop 3 pattern LangGraph.

---

## 15.5 Cau truc deploy san xuat (5 phut)

### Khai niem

> Lam prototype trong notebook thi tot, nhung deploy len san xuat thi sao?
>
> LangGraph cung cap cau truc chuan cho viec deploy san xuat.
>
> ```
> my_pipeline/
> ├── langgraph.json       # Cau hinh diem vao
> ├── graph.py             # Dinh nghia do thi
> ├── state.py             # Schema state
> ├── nodes.py             # Cac ham node
> ├── prompts.py           # Template prompt LLM
> └── requirements.txt     # Dependencies
> ```
>
> Chung ta **tach code notebook thanh cac file rieng**.
>
> Then chot la `langgraph.json`:
> ```json
> {
>     "dependencies": ["."],
>     "graphs": {
>         "blog_pipeline": "./graph.py:graph"
>     },
>     "env": ".env"
> }
> ```
>
> `"blog_pipeline": "./graph.py:graph"` — khai bao bien `graph` trong `graph.py` la diem vao.
>
> `graph.py` chua cung code nhu chung ta da viet trong notebook.
> Tao `StateGraph`, dang ky node, noi canh, `compile()`.
>
> Chi khac duong dan import.

### Cac tuy chon deploy

> Co 3 phuong thuc deploy:
>
> | Phuong thuc | Mo ta |
> |-------------|-------|
> | `langgraph dev` | Phat trien local + Studio UI (http://localhost:8123) |
> | `langgraph up` | Chay trong Docker container |
> | LangGraph Cloud | Deploy quan ly voi tich hop LangSmith |
>
> Chay `langgraph dev` se mo Studio UI de ban co the **test do thi truc quan**.
> Click vao node de xem dau vao/dau ra, va cung cap phan hoi truc tiep tai cac diem interrupt.
>
> Cho san xuat, su dung `langgraph up` cho Docker, hoac
> deploy len LangGraph Cloud de expose thanh API endpoint.

---

## Tong ket (3 phut)

> Tong ket nhung gi chung ta da xay dung hom nay.
>
> **Blog Post Generator Pipeline:**
> 1. `get_topic_info` — LLM nghien cuu chu de
> 2. `dispatch_writers` — ham router, tao outline bang `with_structured_output`, gui bang `Send`
> 3. `write_section` x N — viet phan song song (Map)
> 4. `combine_sections` — gop ket qua (Reduce)
> 5. `human_feedback` — nguoi review bang `interrupt()`
> 6. `finalize_post` — ap dung phan hoi, ban cuoi cung
>
> **Cac pattern LangGraph da su dung:**
>
> | Pattern | Su dung o dau |
> |---------|--------------|
> | `TypedDict` state | Thiet ke luong du lieu pipeline |
> | `Send` API | Gui phan song song (Map) |
> | `operator.add` reducer | Tu dong gop ket qua song song (Reduce) |
> | `with_structured_output` | Dau ra LLM co cau truc (outline) |
> | `conditional_edges` | Ham router cho phan nhanh dong |
> | `interrupt` + `Command` | Workflow review cua nguoi |
> | `SqliteSaver` | Bao toan state giua cac interrupt |
>
> O Chapter 13 chung ta hoc co ban, o Chapter 14 ap dung vao chatbot,
> va hom nay o Chapter 15 chung ta da xay dung **pipeline cap san xuat ket hop tat ca**.
>
> Mot dieu can nho:
> **`dispatch_writers` KHONG phai node — no la ham router.**
> No duoc dang ky voi `add_conditional_edges` va tra ve cac doi tuong `Send` de gui cac node song song.
>
> Hen gap lai o chuong tiep theo.
