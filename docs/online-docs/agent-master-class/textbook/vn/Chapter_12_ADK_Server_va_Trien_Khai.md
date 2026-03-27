# Chapter 12: ADK Server va Trien Khai

---

## 12.0 Tong quan chuong

Trong chuong nay, chung ta se hoc cach xay dung AI agent bang Google ADK (Agent Development Kit), van hanh chung tren server, va cuoi cung trien khai len Google Cloud Vertex AI.

Xuyen suot chuong nay, chung ta se lam viec voi hai du an agent:

1. **Email Refiner Agent** - He thong dua tren LoopAgent, noi nhieu agent chuyen biet hop tac de cai thien email lap di lap lai
2. **Travel Advisor Agent** - Agent co van du lich dua tren cong cu, cung cap thong tin thoi tiet, ty gia va dia diem du lich

Thong qua hai du an nay, chung ta se hoc cac chu de cot loi sau:

| Section | Chu de | Khai niem cot loi |
|---------|--------|-------------------|
| 12.0 | Introduction | Cau truc du an ADK, dinh nghia Agent, thiet ke prompt |
| 12.1 | LoopAgent | Agent lap, output_key, escalate, ToolContext |
| 12.3 | API Server | API server tich hop ADK, REST endpoint, quan ly phien |
| 12.4 | Server Sent Events | SSE streaming, xu ly phan hoi thoi gian thuc |
| 12.6 | Runner | Lop Runner, DatabaseSessionService, thuc thi bang code |
| 12.7 | Deployment to VertexAI | Trien khai Vertex AI, reasoning_engines, thuc thi tu xa |

---

## 12.0 Introduction - Cau truc du an ADK va dinh nghia Agent

### Chu de va Muc tieu

Hieu cau truc co ban cua du an agent dua tren ADK va thiet ke cac thanh phan cua he thong da agent goi la Email Refiner.

### Giai thich khai niem cot loi

#### 1) Cau truc thu muc du an ADK

ADK tuan theo quy tac cau truc thu muc cu the. Trong goi agent phai co `agent.py` va `__init__.py`, va `__init__.py` phai import module `agent` de ADK tu dong nhan dien agent.

```
email-refiner-agent/
├── .python-version          # Phien ban Python (3.13)
├── pyproject.toml           # Dinh nghia phu thuoc du an
├── uv.lock                  # File khoa phu thuoc
├── README.md
└── email_refiner/           # Goi agent
    ├── __init__.py          # Dang ky module agent
    ├── agent.py             # Dinh nghia agent
    └── prompt.py            # Tap hop prompt va mo ta
```

**Vai tro cua `__init__.py`:**

```python
from . import agent
```

Dong nay rat quan trong. Framework ADK tu dong tim kiem module `agent` trong goi, va phai import tuong minh trong `__init__.py` de co che agent discovery cua ADK hoat dong.

#### 2) Thiet lap phu thuoc (pyproject.toml)

```toml
[project]
name = "email-refiner-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "google-adk>=1.12.0",
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Cac phu thuoc chinh:
- **`google-adk`**: Thu vien cot loi Google Agent Development Kit
- **`google-genai`**: Client Google Generative AI
- **`litellm`**: Thu vien cho phep su dung nhieu nha cung cap LLM (OpenAI, Anthropic, Google...) qua giao dien thong nhat

#### 3) Thiet ke da agent chuyen biet

Email Refiner gom 5 agent chuyen biet. Moi agent dam nhan mot khia canh khac nhau cua viec cai thien email:

```python
from google.adk.agents import Agent, LoopAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o-mini")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
)

literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
)
```

**Vai tro cua tung agent:**

| Agent | Vai tro | Nhiem vu cot loi |
|-------|---------|------------------|
| ClarityEditorAgent | Bien tap vien su ro rang | Loai bo su mo ho, xoa cum tu trung lap, lam gon cau |
| ToneStylistAgent | Chuyen gia phong cach | Duy tri giong am ap, tu tin, giu tinh chuyen nghiep |
| PersuationAgent | Chien luoc gia thuyet phuc | Tang cuong CTA, cau truc luan diem, loai bo dien dat bi dong |
| EmailSynthesizerAgent | Nguoi tong hop email | Tich hop tat ca cai tien thanh mot email thong nhat |
| LiteraryCriticAgent | Nha phe binh van hoc | Danh gia chat luong cuoi cung va quyet dinh duyet/lam lai |

#### 4) Pattern thiet ke prompt

Prompt duoc tach thanh `description` (mo ta vai tro agent) va `instruction` (chi thi chi tiet), tuan theo nguyen tac tach biet moi quan tam (Separation of Concerns).

```python
# Mo ta (description) - Dinh nghia ngan gon agent la gi
CLARITY_EDITOR_DESCRIPTION = "Expert editor focused on clarity and simplicity."

# Chi thi (instruction) - Mo ta chi tiet cach agent hoat dong
CLARITY_EDITOR_INSTRUCTION = """
You are an expert editor focused on clarity and simplicity. Your job is to
eliminate ambiguity, redundancy, and make every sentence crisp and clear.

Take the email draft and improve it for clarity:
- Remove redundant phrases
- Simplify complex sentences
- Eliminate ambiguity
- Make every sentence clear and direct

Provide your improved version with focus on clarity.
"""
```

Dac biet chu y **pattern pipeline**. Instruction cua moi agent su dung bien template de tham chieu dau ra cua agent truoc:

```python
TONE_STYLIST_INSTRUCTION = """
...
Here's the clarity-improved version:
{clarity_output}
"""

PERSUASION_STRATEGIST_INSTRUCTION = """
...
Here's the tone-improved version:
{tone_output}
"""

EMAIL_SYNTHESIZER_INSTRUCTION = """
...
Clarity version: {clarity_output}
Tone version: {tone_output}
Persuasion version: {persuasion_output}

Synthesize the best elements from all versions into one polished final email.
"""
```

Cac bien `{clarity_output}`, `{tone_output}` nay duoc ket noi voi `output_key` se hoc trong section tiep theo.

### Diem thuc hanh

1. Tu tao cau truc thu muc du an ADK va tao cau truc import module agent trong `__init__.py`.
2. Doc prompt cua tung agent va ve so do luong pipeline cai thien email.
3. Su dung `LiteLlm` de doi sang model khac (vi du: `anthropic/claude-3-haiku`).

---

## 12.1 LoopAgent - Agent lap va Escalation

### Chu de va Muc tieu

Su dung `LoopAgent` cua ADK de xay dung he thong nhieu agent hop tac lap di lap lai. Hoc cach chia se du lieu giua cac agent qua `output_key` va co che ket thuc vong lap qua `escalate`.

### Giai thich khai niem cot loi

#### 1) output_key - Truyen du lieu giua cac agent

`output_key` chi dinh ten khoa luu dau ra cua agent vao trang thai phien (state). Cac bien template `{clarity_output}`, `{tone_output}` trong prompt duoc dien gia tri chinh tu `output_key` nay.

```python
MODEL = LiteLlm(model="openai/gpt-4o")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
    output_key="clarity_output",    # Dau ra luu vao state["clarity_output"]
    model=MODEL,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
    output_key="tone_output",       # Dau ra luu vao state["tone_output"]
    model=MODEL,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
    output_key="persuasion_output", # Dau ra luu vao state["persuasion_output"]
    model=MODEL,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
    output_key="synthesized_output", # Dau ra luu vao state["synthesized_output"]
    model=MODEL,
)
```

**Luong du lieu:**

```
Dau vao email nguoi dung
    |
    v
ClarityEditorAgent ---- output_key="clarity_output" -------> Luu vao state
    |
    v
ToneStylistAgent ------ output_key="tone_output" ----------> Luu vao state
    |                    (instruction tham chieu {clarity_output})
    v
PersuationAgent ------- output_key="persuasion_output" ----> Luu vao state
    |                    (instruction tham chieu {tone_output})
    v
EmailSynthesizerAgent - output_key="synthesized_output" ---> Luu vao state
    |                    (tham chieu ca 3 output)
    v
LiteraryCriticAgent --- Danh gia chat luong
    |                    (instruction tham chieu {synthesized_output})
    +-- Khong dat --> Bat dau lai vong lap
    +-- Dat ------> escalate de ket thuc vong lap
```

#### 2) ToolContext va escalate - Co che ket thuc vong lap

`LoopAgent` mac dinh lap vo han (hoac den `max_iterations`). De thoat khoi vong lap khi thoa dieu kien, su dung co che `escalate`.

```python
from google.adk.tools.tool_context import ToolContext

async def escalate_email_complete(tool_context: ToolContext):
    """Use this tool only when the email is good to go."""
    tool_context.actions.escalate = True
    return "Email optimization complete."
```

**Diem cot loi:**
- `ToolContext` la doi tuong context ma ADK tu dong inject khi thuc thi cong cu.
- Dat `tool_context.actions.escalate = True` se ket thuc vong lap hien tai ngay lap tuc.
- Cong cu nay chi duoc gan cho `LiteraryCriticAgent`, nen chi khi nha phe binh hai long voi chat luong email thi vong lap moi ket thuc.

```python
literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
    tools=[
        escalate_email_complete,   # Gan cong cu escalate
    ],
    model=MODEL,
)
```

#### 3) Cau hinh LoopAgent

Gom tat ca sub-agent vao `LoopAgent` de hoan thanh cau truc thuc thi lap:

```python
email_refiner_agent = LoopAgent(
    name="EmailRefinerAgent",
    max_iterations=50,                    # Toi da 50 vong lap (bao ve)
    description=EMAIL_OPTIMIZER_DESCRIPTION,
    sub_agents=[
        clarity_agent,                     # 1. Cai thien su ro rang
        tone_stylist_agent,                # 2. Dieu chinh giong dieu
        persuation_agent,                  # 3. Tang suc thuyet phuc
        email_synthesizer_agent,           # 4. Tong hop
        literary_critic_agent,             # 5. Danh gia cuoi (co the escalate)
    ],
)

root_agent = email_refiner_agent
```

**Tam quan trong cua bien `root_agent`:** Framework ADK tu dong tim bien co ten `root_agent` lam diem vao (entry point). Bat buoc phai su dung ten nay.

#### 4) Tang cuong prompt - Dam bao LLM thuc su goi cong cu

Trong thuc te, LLM co the "noi" rang se goi cong cu nhung khong thuc su goi. De ngan chan dieu nay, prompt duoc tang cuong:

```python
LITERARY_CRITIC_INSTRUCTION = """
...
2. If the email meets professional standards and communicates effectively:
   - Call the `escalate_email_complete` tool, CALL IT DONT JUST SAY YOU ARE
     GOING TO CALL IT. CALL THE THING!
   - Provide your final positive assessment of the email
...
## Tool Usage:
When the email is ready, CALL the tool: `escalate_email_complete()`
...
"""
```

Su dung chu in hoa va nhan manh de chi thi ro rang cho LLM phai thuc thi goi cong cu la ky thuat prompt engineering rat huu ich trong thuc te.

### Diem thuc hanh

1. Giam `max_iterations` xuong 3 va chay, quan sat hanh vi khi vong lap dat toi da.
2. Thu bo `escalate = True` va chi tra ve gia tri, xem dieu gi xay ra.
3. Xoa `output_key` va chay, xac nhan rang agent tiep theo khong the tham chieu ket qua truoc.

---

## 12.3 API Server - API Server tich hop ADK

### Chu de va Muc tieu

Hoc cach su dung web server tich hop cua ADK de phuc vu agent duoi dang REST API. Tao Travel Advisor Agent moi va tuong tac qua API server.

### Giai thich khai niem cot loi

#### 1) Travel Advisor Agent - Agent dua tren cong cu

Xay dung agent co van du lich su dung cong cu (tool) de trinh dien API server:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

MODEL = LiteLlm(model="openai/gpt-4o")


async def get_weather(tool_context: ToolContext, location: str):
    """Get current weather information for a location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind": "12 km/h",
        "forecast": "Mild weather with occasional clouds expected throughout the day",
    }


async def get_exchange_rate(
    tool_context: ToolContext, from_currency: str, to_currency: str, amount: float
):
    """Get exchange rate between two currencies.
    Args should always be from_currency str, to_currency str, amount flot
    """
    mock_rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("USD", "KRW"): 1325.00,
        ("EUR", "USD"): 1.09,
        ("EUR", "GBP"): 0.86,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
        ("KRW", "USD"): 0.00075,
    }

    rate = mock_rates.get((from_currency, to_currency), 1.0)
    converted_amount = amount * rate

    return {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "amount": amount,
        "exchange_rate": rate,
        "converted_amount": converted_amount,
        "timestamp": "2024-03-15 10:30:00 UTC",
    }


async def get_local_attractions(
    tool_context: ToolContext, location: str, category: str = "all"
):
    """Get popular attractions and points of interest for a location."""
    attractions = {
        "Paris": [
            {"name": "Eiffel Tower", "type": "landmark", "rating": 4.8,
             "description": "Iconic iron lattice tower"},
            {"name": "Louvre Museum", "type": "museum", "rating": 4.7,
             "description": "World's largest art museum"},
        ],
        "Tokyo": [
            {"name": "Tokyo Tower", "type": "landmark", "rating": 4.5,
             "description": "Communications and observation tower"},
            {"name": "Senso-ji", "type": "temple", "rating": 4.6,
             "description": "Ancient Buddhist temple"},
        ],
        "default": [
            {"name": "City Center", "type": "area", "rating": 4.2,
             "description": "Main downtown area"},
        ],
    }

    location_attractions = attractions.get(location, attractions["default"])

    if category != "all":
        location_attractions = [
            a for a in location_attractions if a["type"] == category
        ]

    return {
        "location": location,
        "category": category,
        "attractions": location_attractions,
        "total_count": len(location_attractions),
    }
```

**Pattern thiet ke ham cong cu:**
- Tat ca ham cong cu duoc dinh nghia la ham `async` bat dong bo.
- Tham so dau tien bat buoc la `tool_context: ToolContext` (ADK tu dong inject).
- Docstring dong vai tro giai thich muc dich cong cu cho LLM.
- Gia tri tra ve la dang dictionary, LLM se dien giai va phan hoi nguoi dung.

Dang ky agent:

```python
travel_advisor_agent = Agent(
    name="TravelAdvisorAgent",
    description=TRAVEL_ADVISOR_DESCRIPTION,
    instruction=TRAVEL_ADVISOR_INSTRUCTION,
    tools=[
        get_weather,
        get_exchange_rate,
        get_local_attractions,
    ],
    model=MODEL,
)

root_agent = travel_advisor_agent
```

#### 2) Chay API Server tich hop ADK

ADK co the chay web server tich hop ngay bang lenh `adk api_server`. Server nay dua tren FastAPI va tu dong cung cap cac REST endpoint de tuong tac voi agent.

```bash
# Chay tu thu muc cha chua du an agent
adk api_server email-refiner-agent/
```

Sau khi server khoi dong, truy cap tai `http://127.0.0.1:8000`.

#### 3) Tuong tac voi agent qua REST API

**Tao phien:**

```python
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"

# Tao phien moi
response = requests.post(
    f"{BASE_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions"
)
print(response.json())
# Nhan phan hoi chua Session ID
```

Pattern endpoint tao phien cua ADK API server:
```
POST /apps/{ten_app}/users/{user_ID}/sessions
```

**Gui tin nhan (che do dong bo):**

```python
SESSION_ID = "ce085ce3-9637-4eca-b7a1-b0be58fa39f1"  # ID nhan khi tao phien

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "Yes, I want to know the currency exchange rate"}],
        "role": "user",
    },
}
response = requests.post(f"{BASE_URL}/run", json=message)
print(response.json())
```

**Phan tich phan hoi:**

```python
data = response.json()

for event in data:
    content = event.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

Phan hoi la mang su kien, trong `content.parts` cua moi su kien:
- `functionCall`: Thong tin cong cu ma agent da goi
- `text`: Phan hoi van ban cua agent

#### 4) Cap nhat phu thuoc

Them phu thuoc cho chuc nang API server va evaluation:

```toml
dependencies = [
    "google-adk[eval]>=1.12.0",   # Them [eval] extra
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",            # HTTP client (goi API)
    "sseclient-py>=1.8.0",        # SSE client (section tiep theo)
]
```

### Diem thuc hanh

1. Chay server bang `adk api_server` va truy cap `http://127.0.0.1:8000/docs` de xem Swagger UI tu dong tao.
2. Tao phien va gui nhieu tin nhan lien tiep, kiem tra ngu canh hoi thoai co duoc duy tri khong.
3. Thu gui API request den agent `email_refiner` voi `APP_NAME` khac.

---

## 12.4 Server Sent Events (SSE) - Phan hoi streaming thoi gian thuc

### Chu de va Muc tieu

Hoc cach su dung endpoint `/run_sse` thay cho endpoint dong bo `/run` de xu ly phan hoi streaming thoi gian thuc dua tren Server-Sent Events.

### Giai thich khai niem cot loi

#### 1) SSE (Server-Sent Events) la gi?

SSE la giao thuc dua tren HTTP de streaming du lieu mot chieu tu server den client theo thoi gian thuc. Khac voi WebSocket, SSE su dung ket noi HTTP thong thuong nen don gian hon de hien thuc.

**So sanh che do dong bo va SSE:**

| Dac diem | `/run` (dong bo) | `/run_sse` (streaming) |
|----------|-------------------|------------------------|
| Cach phan hoi | Tra ve toan bo phan hoi mot lan | Gui theo tung su kien thoi gian thuc |
| Trai nghiem nguoi dung | Cho den khi phan hoi hoan tat | Theo doi tien trinh thoi gian thuc |
| Quan sat goi cong cu | Bao gom trong ket qua | Quan sat qua trinh goi thoi gian thuc |
| Phu hop cho | Phan hoi ngan, xu ly backend | Phan hoi dai, frontend UI |

#### 2) Hien thuc SSE client

```python
import sseclient
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"
SESSION_ID = "3f673a5a-04ab-4edb-af23-6f42449a970b"

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "What is the weather there?"}],
        "role": "user",
    },
    "streaming": True,              # Co kich hoat streaming
}

response = requests.post(
    f"{BASE_URL}/run_sse",           # Endpoint danh rieng cho SSE
    json=message,
    stream=True,                     # Kich hoat che do streaming cua requests
)

client = sseclient.SSEClient(response)

for event in client.events():
    data = json.loads(event.data)
    content = data.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

**Diem khac biet so voi che do dong bo:**

1. **Them `"streaming": True` vao tin nhan yeu cau** - Thong bao server ve che do streaming.
2. **Doi endpoint**: Su dung `/run_sse` thay vi `/run`
3. **Tuy chon `stream=True`**: Kich hoat che do streaming trong `requests.post()`
4. **Boc bang `sseclient.SSEClient`**: Phan tich phan hoi thanh luong su kien SSE
5. **Vong lap su kien**: Xu ly tung su kien qua `client.events()`

#### 3) Cau truc su kien SSE

Moi su kien SSE chua truong `data` dang JSON:

```json
{
    "content": {
        "parts": [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"}
                }
            }
        ]
    }
}
```

Hoac phan hoi van ban:

```json
{
    "content": {
        "parts": [
            {
                "text": "Thoi tiet hien tai o Paris la 22 do..."
            }
        ]
    }
}
```

### Diem thuc hanh

1. Gui cung cau hoi bang che do dong bo (`/run`) va SSE (`/run_sse`), so sanh thoi gian phan hoi va trai nghiem.
2. Quan sat rang su kien goi cong cu (functionCall) den truoc phan hoi van ban khi nhan SSE.
3. Thu tu viet code phan tich HTTP stream truc tiep thay vi dung `sseclient-py` (su dung `response.iter_lines()`).

---

## 12.6 Runner - Thuc thi agent truc tiep tu code

### Chu de va Muc tieu

Hoc cach su dung lop `Runner` de thuc thi agent truc tiep tu code Python thuan, khong can ADK CLI hay API server. Dong thoi tim hieu `DatabaseSessionService` cho quan ly phien luu tru va `InMemoryArtifactService`.

### Giai thich khai niem cot loi

#### 1) Vai tro cua Runner

`Runner` la orchestrator cot loi cho viec thuc thi agent. Su dung khi muon chay agent truc tiep trong code ma khong can API server. Runner quan ly:
- Luong thuc thi agent
- Quan ly trang thai phien
- Quan ly artifact (file...)
- Streaming su kien

#### 2) Session Service va Artifact Service

```python
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService

# Artifact service: Dua tren bo nho (luu tam file...)
in_memory_service_py = InMemoryArtifactService()

# Session service: Dua tren SQLite DB (luu phien lau dai)
session_service = DatabaseSessionService(db_url="sqlite:///./session.db")
```

**Uu diem cua `DatabaseSessionService`:**
- Du lieu phien duoc luu lau dai vao file SQLite (`session.db`).
- Sau khi khoi dong lai server van co the tiep tuc hoi thoai truoc.
- Doi `db_url` sang PostgreSQL de su dung trong moi truong production.

#### 3) Tao phien va khoi tao trang thai

```python
session = await session_service.create_session(
    app_name="weather_agent",
    user_id="u_123",
    state={
        "user_name": "nico",    # Luu ten nguoi dung vao trang thai ban dau
    },
)
```

Co the dat gia tri ban dau trong dictionary `state`. Gia tri nay duoc tham chieu nhu bien template trong instruction cua agent:

```python
# prompt.py
TRAVEL_ADVISOR_INSTRUCTION = """
You are a helpful travel advisor agent...

You call the user by their name:

Their name is {user_name}
...
"""
```

`{user_name}` tu dong duoc thay the bang gia tri `"user_name"` trong state cua phien. Day la chuc nang **prompt template dua tren trang thai** cua ADK.

#### 4) Thuc thi agent qua Runner

```python
from google.genai import types
from google.adk.runners import Runner

# Tao Runner
runner = Runner(
    agent=travel_advisor_agent,           # Agent can thuc thi
    session_service=session_service,      # Dich vu quan ly phien
    app_name="weather_agent",             # Ten app (phai khop voi session service)
    artifact_service=in_memory_service_py, # Dich vu quan ly artifact
)

# Tao tin nhan nguoi dung
message = types.Content(
    role="user",
    parts=[
        types.Part(text="Im going to Vietnam, tell me all about it."),
    ],
)

# Thuc thi streaming bat dong bo
async for event in runner.run_async(
    user_id="u_123",
    session_id=session.id,
    new_message=message
):
    if event.is_final_response():
        print(event.content.parts[0].text)
    else:
        print(event.get_function_calls())
        print(event.get_function_responses())
```

**Pattern xu ly su kien:**
- `event.is_final_response()`: Kiem tra co phai phan hoi van ban cuoi cung khong
- `event.get_function_calls()`: Kiem tra su kien goi cong cu
- `event.get_function_responses()`: Kiem tra su kien phan hoi tu cong cu

#### 5) Phan tich ket qua thuc thi

Ket qua thuc thi thuc te cho thay ro qua trinh hoat dong cua agent:

```
# Buoc 1: Agent goi 3 cong cu dong thoi (goi cong cu song song)
[FunctionCall(name='get_weather', args={'location': 'Vietnam'}),
 FunctionCall(name='get_exchange_rate', args={'from_currency': 'USD', 'to_currency': 'VND', 'amount': 1}),
 FunctionCall(name='get_local_attractions', args={'location': 'Vietnam'})]

# Buoc 2: Nhan phan hoi tu cong cu
[FunctionResponse(name='get_weather', response=<dict len=6>),
 FunctionResponse(name='get_exchange_rate', response=<dict len=6>),
 FunctionResponse(name='get_local_attractions', response={
     'error': "Invoking `get_local_attractions()` failed as the following
     mandatory input parameters are not present: category..."
 })]

# Buoc 3: Phan hoi cuoi (tong hop ket qua cong cu thanh ngon ngu tu nhien)
Hello Nico! Here's some information to help you prepare for your trip to Vietnam:

### Weather in Vietnam
- **Current Temperature:** 22°C
- **Condition:** Partly cloudy
...
```

Diem dang chu y:
1. Agent doc `{user_name}` tu trang thai phien de chao "Hello Nico!"
2. Goi 3 cong cu **song song** de thu thap thong tin hieu qua.
3. `get_local_attractions` gap loi thieu tham so `category`, nhung agent tu xu ly va truc tiep tao thong tin dia diem du lich Viet Nam chung.

### Diem thuc hanh

1. Doi `DatabaseSessionService` thanh `InMemorySessionService` va xac nhan phien khong duoc duy tri sau khi khoi dong lai.
2. Them `"preferred_language": "Korean"` vao `state` va su dung trong prompt de agent tra loi bang tieng Han.
3. Tim cach su dung phuong thuc dong bo `run` thay vi `run_async`.
4. Su dung `output_key` de luu phan hoi agent vao trang thai phien va tham chieu trong hoi thoai tiep theo.

---

## 12.7 Deployment to Vertex AI - Trien khai len dam may

### Chu de va Muc tieu

Hoc cach trien khai ADK agent da xay dung len Vertex AI Agent Engine cua Google Cloud de van hanh trong moi truong production.

### Giai thich khai niem cot loi

#### 1) Vertex AI Agent Engine la gi?

Vertex AI Agent Engine (truoc la Reasoning Engine) la dich vu cua Google Cloud de host va quan ly AI agent. Khi trien khai ADK agent len cloud:
- Khong can quan ly ha tang server
- Tu dong scaling
- Su dung tinh nang bao mat va giam sat cua Google Cloud
- Quan ly phien va thuc thi tu xa

#### 2) Script trien khai (deploy.py)

```python
import dotenv

dotenv.load_dotenv()

import os
import vertexai
import vertexai.agent_engines
from vertexai.preview import reasoning_engines
from travel_advisor_agent.agent import travel_advisor_agent

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"
BUCKET = "gs://nico-awesome-weather_agent"

# Khoi tao Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,         # Bucket GCS de staging file trien khai
)

# Boc ADK agent thanh AdkApp
app = reasoning_engines.AdkApp(
    agent=travel_advisor_agent,
    enable_tracing=True,            # Kich hoat theo doi thuc thi
)

# Trien khai len Vertex AI
remote_app = vertexai.agent_engines.create(
    display_name="Travel Advisor Agent",
    agent_engine=app,
    requirements=[                  # Goi Python can thiet
        "google-cloud-aiplatform[adk,agent_engines]",
        "litellm",
    ],
    extra_packages=["travel_advisor_agent"],  # Bao gom goi agent
    env_vars={                      # Truyen bien moi truong
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    },
)
```

**Phan tich chi tiet qua trinh trien khai:**

| Buoc | Code | Mo ta |
|------|------|-------|
| 1. Thiet lap moi truong | `dotenv.load_dotenv()` | Tai bien moi truong tu file `.env` (API key...) |
| 2. Khoi tao Vertex AI | `vertexai.init(...)` | Thiet lap project, region, staging bucket |
| 3. Boc app | `reasoning_engines.AdkApp(...)` | Boc ADK agent thanh dinh dang tuong thich Vertex AI |
| 4. Trien khai | `agent_engines.create(...)` | Thuc thi trien khai thuc te len cloud |

**Vai tro cua tham so `extra_packages`:**
Bao gom thu muc goi local (`travel_advisor_agent`) vao goi trien khai. Nhu vay code agent co the duoc import trong moi truong cloud.

**Quan ly secret qua `env_vars`:**
Truyen thong tin nhay cam nhu API key qua bien moi truong. Khong nen hard-code truc tiep trong code vi ly do bao mat.

#### 3) Phu thuoc bo sung

Goi them cho trien khai:

```toml
dependencies = [
    "cloudpickle>=3.1.1",                                    # Serialization doi tuong
    "google-adk[eval]>=1.12.0",
    "google-cloud-aiplatform[adk,agent-engines]>=1.111.0",   # Vertex AI SDK
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",
    "sseclient-py>=1.8.0",
]
```

- **`cloudpickle`**: Su dung de serialize doi tuong Python va gui len cloud
- **`google-cloud-aiplatform[adk,agent-engines]`**: Bao gom chuc nang ADK va Agent Engine cua Vertex AI

#### 4) Quan ly va thuc thi agent tu xa (remote.py)

```python
import vertexai
from vertexai import agent_engines

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# Truy van danh sach trien khai
# deployments = agent_engines.list()
# for deployment in deployments:
#     print(deployment)

# Lay remote app theo deployment ID cu the
DEPLOYMENT_ID = "projects/23382131925/locations/europe-southwest1/reasoningEngines/2153529862441140224"
remote_app = agent_engines.get(DEPLOYMENT_ID)

# Xoa trien khai (force=True de cuong che xoa)
remote_app.delete(force=True)
```

**Tao phien tu xa va streaming query:**

```python
# Tao phien tu xa
# remote_session = remote_app.create_session(user_id="u_123")
# print(remote_session["id"])

SESSION_ID = "5724511082748313600"

# Gui streaming query den agent tu xa
# for event in remote_app.stream_query(
#     user_id="u_123",
#     session_id=SESSION_ID,
#     message="I'm going to Laos, any tips?",
# ):
#     print(event, "\n", "=" * 50)
```

**Tom tat API thuc thi tu xa:**

| Phuong thuc | Muc dich |
|-------------|----------|
| `agent_engines.list()` | Truy van tat ca danh sach trien khai |
| `agent_engines.get(id)` | Lay trien khai cu the |
| `remote_app.create_session(user_id=...)` | Tao phien tu xa |
| `remote_app.stream_query(...)` | Query theo kieu streaming |
| `remote_app.delete(force=True)` | Xoa trien khai |

### Diem thuc hanh

1. Tao GCP project va GCS bucket, thu trien khai agent thuc te.
2. Trien khai voi `enable_tracing=True` va kiem tra tracing log tren Google Cloud Console.
3. So sanh thoi gian phan hoi giua `remote_app.stream_query()` va chay Runner local.
4. Tao phien voi nhieu user ID va kiem tra session isolation hoat dong dung.

---

## Tong hop cot loi chuong

### 1. Kien truc ADK Agent

```
                    ┌─────────────────────┐
                    │      ADK Agent      │
                    │                     │
                    │  - name             │
                    │  - description      │
                    │  - instruction      │
                    │  - model            │
                    │  - tools            │
                    │  - output_key       │
                    │  - sub_agents       │
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
         │  Agent  │   │ LoopAgent │   │  Runner   │
         │ (don)   │   │ (lap)     │   │ (thuc thi)│
         └─────────┘   └───────────┘   └───────────┘
```

### 2. So sanh cac che do thuc thi

| Cach thuc thi | Mo ta | Khi nao su dung |
|---------------|-------|-----------------|
| `adk web` | Test agent qua web UI | Test nhanh khi phat trien |
| `adk api_server` | Chay REST API server | Ket noi frontend, dich vu local |
| `Runner` (che do code) | Thuc thi truc tiep tu code Python | Tich hop vao ung dung tuy chinh |
| Trien khai Vertex AI | Moi truong production tren cloud | Van hanh dich vu thuc te |

### 3. Tong hop cac lop/thanh phan ADK cot loi

| Thanh phan | Vai tro |
|------------|---------|
| `Agent` | Dinh nghia agent don (ten, mo ta, chi thi, model, cong cu) |
| `LoopAgent` | Orchestrator thuc thi lap cac sub-agent |
| `LiteLlm` | Su dung nhieu nha cung cap LLM qua giao dien thong nhat |
| `ToolContext` | Truy cap trang thai phien, hanh dong tu ham cong cu |
| `output_key` | Khoa luu dau ra agent vao trang thai phien |
| `escalate` | Ket thuc som vong lap hoac chuoi agent |
| `Runner` | Orchestrator quan ly thuc thi agent trong code |
| `DatabaseSessionService` | Quan ly phien luu tru dua tren DB |
| `InMemoryArtifactService` | Quan ly artifact dua tren bo nho |
| `reasoning_engines.AdkApp` | Boc ADK agent thanh dinh dang trien khai Vertex AI |

### 4. Pattern luong du lieu cot loi

```
Luu qua output_key -> Tich luy trong state -> Tham chieu qua {ten_bien} trong instruction
```

Pattern nay la co che quan trong nhat de truyen du lieu giua cac agent trong ADK.

---

## Bai tap thuc hanh

### Bai tap 1: Agent review code (su dung LoopAgent)

Tham khao cau truc Email Refiner Agent de tao **agent review code**.

**Yeu cau:**
- `SecurityReviewAgent`: Kiem tra lo hong bao mat
- `PerformanceReviewAgent`: De xuat toi uu hieu suat
- `StyleReviewAgent`: Kiem tra phong cach code va do doc
- `ReviewSynthesizerAgent`: Tong hop tat ca review
- `ApprovalAgent`: Quyet dinh duyet/tu choi cuoi cung (su dung cong cu escalate)

**Goi y:**
- Dat `output_key` cho tung agent de luu ket qua review vao state
- Gan cong cu `escalate_review_complete` cho `ApprovalAgent`
- Dat `max_iterations` cua `LoopAgent` phu hop

### Bai tap 2: API server va SSE client

Mo rong Travel Advisor Agent them **chuc nang goi y nha hang** va hien thuc API server cung SSE client.

**Yeu cau:**
1. Them ham cong cu `get_restaurant_recommendations(location, cuisine_type)`
2. Chay server bang `adk api_server`
3. Nhan phan hoi streaming thoi gian thuc qua SSE client
4. Phan biet su kien goi cong cu va su kien phan hoi van ban de hien thi tren UI

### Bai tap 3: CLI hoi thoai su dung Runner

Tao chuong trinh CLI hoi thoai voi agent su dung Runner.

**Yeu cau:**
1. Su dung `DatabaseSessionService` de luu tru lich su hoi thoai lau dai
2. Khi khoi dong chuong trinh, cho phep chon tiep tuc phien cu hoac tao phien moi
3. Luu ngon ngu uu tien cua nguoi dung vao `state` va su dung trong prompt
4. Khi thoat bang `Ctrl+C`, in session ID de co the tiep tuc lan sau

### Bai tap 4: Trien khai Vertex AI (nang cao)

Trien khai Travel Advisor Agent thuc te len Vertex AI va su dung tu xa.

**Yeu cau:**
1. Tao GCP project va kich hoat Vertex AI API
2. Tao GCS bucket (cho staging)
3. Tham khao `deploy.py` de viet script trien khai
4. Tham khao `remote.py` de tao phien tu xa va thuc thi query
5. So sanh thoi gian phan hoi giua agent da trien khai va chay local

**Luu y:**
- Co the phat sinh phi GCP, nen nho xoa bang `remote_app.delete(force=True)` sau khi test
- Khong hard-code API key trong code, bat buoc truyen qua bien moi truong

---

> **Gioi thieu chuong tiep theo:** Chuong tiep theo se gioi thieu framework danh gia agent (Evaluation), hoc cach do luong va cai thien chat luong phan hoi cua agent mot cach he thong.
