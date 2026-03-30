# Chapter 16: Các mẫu kiến truc Workflow — Kich ban bai giang

---

## Mo dau (2 phut)

> Duoc roi, hom nay chung ta se hoc **5 mau kien truc Workflow**.
>
> Chuong truoc, chung ta da hoc cac building block co ban cua LangGraph.
> Hom nay, chung ta ket hop chung thanh **cac mau thuong dung trong du an thuc te**.
>
> Noi dung hom nay:
>
> ```
> 16.1 Prompt Chaining        (Goi LLM tuan tu)
> 16.2 Prompt Chaining + Gate (Kiem tra chat luong va thu lai)
> 16.3 Routing                (Dinh tuyen dong theo do kho)
> 16.4 Parallelization        (Thuc thi song song + tong hop)
> 16.5 Orchestrator-Workers   (Phan phoi tac vu dong)
> ```
>
> Moi mau xay dung tren mau truoc do.
> Chaining la nen tang, sau do them Gate, Routing, Parallelization,
> va cuoi cung ket hop tat ca thanh Orchestrator-Workers.
>
> Bat dau thoi.

---

## 16.0 Setup & Environment (2 phut)

### Truoc khi chay cell

> Dau tien, kiem tra moi truong.
> Chung ta load API key va ten model tu file `.env`.
>
> Cac package can thiet hom nay:
> - `langgraph >= 1.1` — framework cot loi
> - `langchain >= 1.2` — tich hop LLM
> - `pydantic` — schema dau ra co cau truc

### Sau khi chay cell

> Neu `OPENAI_API_KEY`, `OPENAI_BASE_URL`, va `OPENAI_MODEL_NAME` hien thi dung, chung ta da san sang.
> Cell tiep theo cung kiem tra phien ban `langgraph` va `langchain`.

---

## 16.1 Prompt Chaining — Chuoi tuan tu (10 phut)

### Khai niem

> Day la mau co ban nhat. **Noi nhieu loi goi LLM theo trinh tu.**
>
> Dau ra cua buoc truoc tro thanh dau vao cua buoc sau.
>
> ```
> START → list_ingredients → create_recipe → describe_plating → END
> ```
>
> Chung ta dung vi du cong thuc nau an:
> 1. Ten mon an → tao **danh sach nguyen lieu**
> 2. Danh sach nguyen lieu → tao **huong dan nau**
> 3. Huong dan nau → tao **mo ta trinh bay dia**
>
> Ky thuat chinh la `with_structured_output()`.
> O node dau tien, chung ta **ep phan hoi LLM thanh Pydantic model**
> de node tiep theo nhan duoc cau truc dam bao.

### Giai thich code (truoc khi chay)

> Cung xem code.
>
> Dau tien, dinh nghia State:
> ```python
> class State(TypedDict):
>     dish: str                    # dau vao: ten mon an
>     ingredients: list[dict]      # dau ra buoc 1
>     recipe_steps: str            # dau ra buoc 2
>     plating_instructions: str    # dau ra buoc 3
> ```
>
> Sau do dinh nghia Pydantic model cho dau ra co cau truc:
> ```python
> class Ingredient(BaseModel):
>     name: str
>     quantity: str
>     unit: str
> ```
>
> Phan chinh trong node `list_ingredients`:
> ```python
> structured_llm = llm.with_structured_output(IngredientsOutput)
> ```
>
> Dieu nay ep LLM tra ve **cau truc JSON chinh xac** thay vi van ban tu do.
> `name`, `quantity`, `unit` duoc dam bao co mat.
>
> Hai node con lai (`create_recipe`, `describe_plating`) dung `invoke()` binh thuong.
> Chung dua ket qua buoc truoc vao prompt.
>
> Cau hinh do thi don gian — `add_edge` noi theo thu tu.
> Chay thu nao.

### Sau khi chay

> Cung xem ket qua.
>
> Phan `=== Ingredients ===` hien thi nguyen lieu co cau truc:
> `chickpeas: 1 can`, `tahini: 1/4 cup`, v.v.
> Nho `with_structured_output()` ma co cau truc gon gang.
>
> Phia duoi, Recipe va Plating la van ban tu do.
> Diem chinh la moi buoc dung ket qua buoc truoc lam dau vao.
>
> **Uu diem cua Chaining**: chia tac vu phuc tap thanh cac buoc
> giup moi prompt don gian hon va chat luong dau ra tot hon.

### Bai tap 16.1 (3 phut)

> **Bai tap 1**: Doi `dish` thanh `"bibimbap"` hoac `"sushi"` va so sanh ket qua.
>
> **Bai tap 2**: So sanh su khac biet giua node dung `with_structured_output()` va node dung `invoke()` binh thuong.
>
> **Bai tap 3**: Them buoc thu 4 (vi du: goi y ket hop ruou vang).

---

## 16.2 Prompt Chaining + Gate — Cong kiem tra dieu kien (8 phut)

### Khai niem

> Chung ta them **kiem tra chat luong** vao mau Chaining 16.1.
>
> Neu dau ra LLM khong dat tieu chuan? **Chay lai buoc truoc.**
>
> ```
> START → list_ingredients → [gate: 3~8?] → Yes → create_recipe → ...
>                              ↑              → No ─┘ (thu lai)
> ```
>
> Ham gate don gian: tra ve `True` de qua, `False` de thu lai.
>
> Day la ung dung thuc te cua **conditional edges** tu 13.7.
> Trong `add_conditional_edges`, `True` di tiep, `False` quay lai chinh no.

### Giai thich code (truoc khi chay)

> Code gan giong 16.1, nhung them ham gate:
>
> ```python
> def gate(state: State):
>     count = len(state["ingredients"])
>     if count > 8 or count < 3:
>         print(f"  GATE FAIL: {count} ingredients (need 3-8). Retrying...")
>         return False
>     print(f"  GATE PASS: {count} ingredients")
>     return True
> ```
>
> Duoi 3 hoac tren 8 nguyen lieu la that bai — goi lai `list_ingredients`.
>
> Phan chinh cua do thi:
> ```python
> graph_builder.add_conditional_edges(
>     "list_ingredients",
>     gate,
>     {True: "create_recipe", False: "list_ingredients"},
> )
> ```
>
> Khi `False`, no quay lai chinh no — tao thanh **vong lap thu lai**.
> Chay thu nao.

### Sau khi chay

> `Generated 8 ingredients` → `GATE PASS: 8 ingredients`
>
> Lan nay qua ngay tu lan dau. Vi prompt yeu cau "5-8" nen thuong qua.
>
> Nhung neu thay doi dieu kien gate thanh `len == 5`?
> Ban se thay nhieu lan thu lai.
>
> **Luu y**: co nguy co vong lap vo han.
> Trong thuc te, luon them truong `retry_count` vao State va gioi han so lan thu lai toi da.

### Bai tap 16.2 (3 phut)

> **Bai tap 1**: Doi dieu kien gate thanh `len(ingredients) == 5` va quan sat so lan thu lai.
>
> **Bai tap 2**: Them truong `retry_count` vao State va gioi han toi da 3 lan.
>
> **Bai tap 3**: Nghi ve cac tinh huong ma mau gate huu ich — kiem tra cu phap code, xac minh ngon ngu dich, kiem tra truong bat buoc, v.v.

---

## 16.3 Routing — Dinh tuyen dong (10 phut)

### Khai niem

> Mau thu ba. **Dinh tuyen den cac duong di khac nhau dua tren dau vao.**
>
> LLM danh gia do kho cau hoi va dinh tuyen den cac node model khac nhau:
>
> ```
> START → assess_difficulty → easy   → dumb_node   (GPT-3.5)
>                           → medium → average_node (GPT-4o)
>                           → hard   → smart_node   (GPT-5)
> ```
>
> Hai ky thuat chinh:
>
> 1. **Structured Output + Literal**: Gioi han phan hoi LLM chi co `"easy"`, `"medium"`, hoac `"hard"`
> 2. **Command**: Tu 13.9 — dinh tuyen + cap nhat state trong mot lan
>
> Rat huu ich de toi uu chi phi trong thuc te.
> Khong can dung model dat cho cau hoi de.

### Giai thich code (truoc khi chay)

> Xem `DifficultyResponse`:
> ```python
> class DifficultyResponse(BaseModel):
>     difficulty_level: Literal["easy", "medium", "hard"]
> ```
>
> `Literal` ep LLM chi tra ve mot trong ba gia tri nay.
>
> Node `assess_difficulty`:
> ```python
> def assess_difficulty(state: State):
>     structured_llm = llm.with_structured_output(DifficultyResponse)
>     response = structured_llm.invoke(...)
>     level = response.difficulty_level
>     goto_map = {"easy": "dumb_node", "medium": "average_node", "hard": "smart_node"}
>     return Command(goto=goto_map[level], update={"difficulty": level})
> ```
>
> `Command` xu ly dong thoi **di dau** va **cap nhat state**.
> Khong can `add_conditional_edges`.
>
> Chu y tham so `destinations` moi trong cau hinh do thi:
> ```python
> graph_builder.add_node(
>     "assess_difficulty", assess_difficulty,
>     destinations=("dumb_node", "average_node", "smart_node"),
> )
> ```
>
> Cho LangGraph biet cac dich den co the khi dung Command.
>
> Chung ta se test voi hai cau hoi — mot de, mot kho.

### Chay — cau hoi de

> `"What is the capital of France?"` — thu do cua Phap.
>
> ```
> Difficulty: easy → dumb_node
> Model: gpt-3.5 (simulated)
> Answer: The capital of France is Paris.
> ```
>
> Cau hoi don gian, danh gia `easy`, dinh tuyen den `dumb_node`.

### Chay — cau hoi kho

> `"Explain the economic implications of quantum computing on global supply chains"`
>
> ```
> Difficulty: hard → smart_node
> Model: gpt-5 (simulated)
> ```
>
> Cau hoi phuc tap, danh gia `hard`, dinh tuyen den `smart_node`.
>
> O day chung ta mo phong voi cung mot model, nhung trong thuc te, dung model khac nhau
> co the **giam chi phi dang ke**. Model re cho cau de, model cao cap cho cau kho.

### Bai tap 16.3 (3 phut)

> **Bai tap 1**: Thu cac cau hoi voi do kho khac nhau va xac minh dinh tuyen dung.
>
> **Bai tap 2**: Kiem tra truong `model_used` de xac minh duong di thuc te.
>
> **Bai tap 3**: Thiet ke tinh huong dinh tuyen den cac prompt template khac nhau thay vi model khac nhau.

---

## 16.4 Parallelization — Thuc thi song song (10 phut)

### Khai niem

> Mau thu tu. **Chay cac loi goi LLM doc lap dong thoi.**
>
> ```
>         → get_summary ────────┐
>         → get_sentiment ──────┤
> START → → get_key_points ─────┤→ get_final_analysis → END
>         → get_recommendation ─┘
> ```
>
> Day con goi la mau **Fan-out / Fan-in**:
> - Fan-out: 4 node khoi dong dong thoi tu START
> - Fan-in: ca 4 phai hoan thanh truoc khi hop nhat vao mot
>
> Trien khai dieu nay trong LangGraph don gian den bat ngo.
> Noi edge tu `START` den nhieu node — chung **tu dong chay song song**.
> Noi nhieu node den mot — no **tu dong join** (doi tat ca hoan thanh).
>
> Khac voi Send API o 13.8: o day chung ta chay **cac ham khac nhau** dong thoi.
> Send API chay **cung mot ham voi dau vao khac nhau** dong thoi.

### Giai thich code (truoc khi chay)

> State co 6 truong:
> ```python
> class State(TypedDict):
>     document: str          # tai lieu dau vao
>     summary: str           # node song song 1
>     sentiment: str         # node song song 2
>     key_points: str        # node song song 3
>     recommendation: str    # node song song 4
>     final_analysis: str    # ket qua tong hop
> ```
>
> Ca 4 node song song deu doc cung `document` nhung **ghi vao truong khac nhau**.
> Khong trung lap, nen khong xung dot khi chay dong thoi.
>
> Phan chinh cua cau hinh do thi:
> ```python
> # Fan-out: START den 4 node
> graph_builder.add_edge(START, "get_summary")
> graph_builder.add_edge(START, "get_sentiment")
> graph_builder.add_edge(START, "get_key_points")
> graph_builder.add_edge(START, "get_recommendation")
>
> # Fan-in: 4 node den 1
> graph_builder.add_edge("get_summary", "get_final_analysis")
> graph_builder.add_edge("get_sentiment", "get_final_analysis")
> graph_builder.add_edge("get_key_points", "get_final_analysis")
> graph_builder.add_edge("get_recommendation", "get_final_analysis")
> ```
>
> Chi can vay la co song song + join. Chay thu nao.

### Sau khi chay

> Xem output:
> ```
> [parallel] get_key_points started
> [parallel] get_recommendation started
> [parallel] get_sentiment started
> [parallel] get_summary started
> ```
>
> Ca 4 bat dau **gan nhu dong thoi**. Thu tu ngau nhien la bang chung cua thuc thi song song.
>
> Sau do:
> ```
> [join] get_final_analysis started
> ```
>
> `get_final_analysis` chi chay sau khi ca 4 hoan thanh.
>
> Final Analysis ket hop summary, sentiment, key_points, va recommendation
> thanh bai phan tich tong hop.
>
> **Goc do hieu suat**: thuc thi tuan tu thi thoi gian goi LLM cong lai.
> Thuc thi song song chi mat **thoi gian cua loi goi cham nhat**. Co the nhanh hon 3-4 lan.

### Bai tap 16.4 (3 phut)

> **Bai tap 1**: Them node song song thu 5 (vi du: `get_risks`) va xac minh no duoc bao gom trong tong hop.
>
> **Bai tap 2**: Chay voi `graph.stream()` de quan sat truc tiep viec bat dau song song.
>
> **Bai tap 3**: So sanh tong thoi gian thuc thi giua tuan tu va song song.

---

## 16.5 Orchestrator-Workers — Phan phoi tac vu dong (10 phut)

### Khai niem

> Mau cuoi cung. Manh nhat, ket hop tat ca cac mau truoc.
>
> ```
> START → orchestrator → [Send x N] → worker x N → synthesizer → END
> ```
>
> Y tuong cot loi:
> 1. **Orchestrator**: phan tich chu de va tao danh sach cac section (khong biet truoc so luong)
> 2. **Send API**: dong khoi dong N worker song song, moi worker mot section
> 3. **Synthesizer**: gop tat ca ket qua worker thanh bao cao cuoi cung
>
> Khac voi 16.4 Parallelization:
> - 16.4: **so node co dinh** (4 node dinh nghia trong code)
> - 16.5: **so node dong** (LLM quyet dinh, Send API thuc thi)
>
> Day la **su ket hop thuc te** cua 13.8 Send API + 16.4 Parallelization.

### Giai thich code (truoc khi chay)

> Xem State:
> ```python
> class State(TypedDict):
>     topic: str
>     sections: list[str]
>     results: Annotated[list[dict], operator.add]  # reducer!
>     final_report: str
> ```
>
> `results` co reducer `operator.add`.
> Khi cac worker tra ve ket qua song song, chung duoc tu dong tich luy.
>
> **Orchestrator**:
> ```python
> def orchestrator(state: State):
>     structured_llm = llm.with_structured_output(Sections)
>     response = structured_llm.invoke(
>         f"Break down this topic into 3-5 research sections: {state['topic']}"
>     )
>     return {"sections": response.sections}
> ```
>
> LLM chia chu de thanh 3-5 section. Co cau truc voi `with_structured_output()`.
>
> **Dispatcher** (Send API):
> ```python
> def dispatch_workers(state: State):
>     return [Send("worker", section) for section in state["sections"]]
> ```
>
> Tra ve so doi tuong `Send` bang so section. 5 section = 5 worker song song.
>
> **Worker**:
> ```python
> def worker(section: str):  # nhan str, khong phai State!
>     response = llm.invoke(f"Write a brief paragraph about: {section}")
>     return {"results": [{"section": section, "content": response.content}]}
> ```
>
> Nhu da hoc o 13.8, gia tri truyen qua Send API la **dau vao tuy chinh, khong phai State**.
>
> **Synthesizer**: gop tat ca ket qua worker thanh bao cao cuoi cung.
>
> Chay thu nao.

### Sau khi chay

> ```
> Orchestrator: 5 sections
>   - Introduction to AI in Healthcare
>   - Applications of AI in Clinical Settings
>   - Ethical Considerations and Challenges
>   - Impact on Patient Outcomes
>   - Future Trends and Predictions
> ```
>
> Orchestrator chia chu de thanh 5 section.
>
> Sau do 5 worker chay song song:
> ```
> Worker done: Introduction to AI in Healthcare...
> Worker done: Future Trends and Predictions...
> Worker done: Impact on Patient Outcomes...
> ```
>
> Chu y thu tu ngau nhien — do la thuc thi song song.
>
> Cuoi cung, Final Report la bao cao hoan chinh gop 5 section.
>
> **Day la suc manh cua mau Orchestrator-Workers.**
> Chi can doi chu de va cau truc bao cao hoan toan khac duoc tao tu dong.
> LLM quyet dinh ca so luong va noi dung cac section.

### Bai tap 16.5 (3 phut)

> **Bai tap 1**: Doi `topic` de xem LLM tao cac section khac nhau khong.
>
> **Bai tap 2**: Them `time.sleep(1)` vao worker de cam nhan hieu qua thuc thi song song.
>
> **Bai tap 3**: Sua worker de tra ve dau ra co cau truc dung `BaseModel`.

---

## Huong dan bai tap tong hop (3 phut)

> Cuoi notebook co 4 bai tap tong hop.
>
> **Bai 1** (★★☆): Chuoi dich + gate — Han Quoc→Anh→Nhat dich tuan tu, thu lai neu duoi 50 ky tu
> **Bai 2** (★★☆): Dinh tuyen theo cam xuc — phan tich cam xuc tin nhan nguoi dung, dinh tuyen den positive/negative/neutral
> **Bai 3** (★★★): Phan tich code review song song — 4 goc do (bao mat/hieu suat/doc duoc/testing) dong thoi, roi tong hop
> **Bai 4** (★★★): Orchestrator-Workers blog — chu de→tao muc luc→worker theo section→tong hop
>
> Bai 1-2 la co ban, bai 3-4 la thu thach.
> Phan bo thoi gian: 10 phut cho bai de, 15 phut cho bai kho.

---

## Ket thuc (2 phut)

> Tong ket 5 mau chung ta da hoc hom nay.
>
> | Mau | Cau truc | Diem chinh |
> |-----|----------|------------|
> | Prompt Chaining | A → B → C | Noi tuan tu, structured output dam bao on dinh |
> | Chaining + Gate | A → [kiem tra] → B / ↩ A | Kiem tra chat luong va thu lai, chu y vong lap vo han |
> | Routing | A → nhanh → B hoac C hoac D | Command + Literal cho dinh tuyen an toan |
> | Parallelization | Fan-out → Fan-in | Chi noi edge la tu dong song song, hieu suat tang 3-4 lan |
> | Orchestrator-Workers | O → Send x N → S | Song song dong, Send API + Reducer tich luy ket qua |
>
> Day la **5 mau thuong dung nhat trong thuc te**.
> Hau het cac workflow AI co the xay dung tu su ket hop cac mau nay.
>
> Chuong tiep theo, chung ta se hoc cac mau nang cao hon.
> Cam on cac ban. Lam tot lam.
