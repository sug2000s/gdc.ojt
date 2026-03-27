# Chapter 12: ADK 서버와 배포

---

## 12.0 챕터 개요

이번 챕터에서는 Google ADK(Agent Development Kit)를 활용하여 AI 에이전트를 구축하고, 이를 서버로 운영하며, 최종적으로 Google Cloud Vertex AI에 배포하는 전체 과정을 학습한다.

챕터 전반에 걸쳐 두 가지 에이전트 프로젝트를 다룬다:

1. **Email Refiner Agent** - 여러 전문 에이전트가 협력하여 이메일을 반복적으로 개선하는 LoopAgent 기반 시스템
2. **Travel Advisor Agent** - 날씨, 환율, 관광지 정보를 제공하는 도구 기반 여행 어드바이저 에이전트

이 두 프로젝트를 통해 다음 핵심 주제를 학습한다:

| 섹션 | 주제 | 핵심 개념 |
|------|------|-----------|
| 12.0 | Introduction | ADK 프로젝트 구조, Agent 정의, 프롬프트 설계 |
| 12.1 | LoopAgent | 반복 에이전트, output_key, escalate, ToolContext |
| 12.3 | API Server | ADK 내장 API 서버, REST 엔드포인트, 세션 관리 |
| 12.4 | Server Sent Events | SSE 스트리밍, 실시간 응답 처리 |
| 12.6 | Runner | Runner 클래스, DatabaseSessionService, 코드 모드 실행 |
| 12.7 | Deployment to VertexAI | Vertex AI 배포, reasoning_engines, 원격 실행 |

---

## 12.0 Introduction - ADK 프로젝트 구조와 에이전트 정의

### 주제 및 목표

ADK 기반 에이전트 프로젝트의 기본 구조를 이해하고, 이메일 리파이너(Email Refiner)라는 다중 에이전트 시스템의 각 구성 요소를 설계한다.

### 핵심 개념 설명

#### 1) ADK 프로젝트 디렉토리 구조

ADK는 특정 디렉토리 구조 규칙을 따른다. 에이전트 패키지 안에 `agent.py`와 `__init__.py`가 반드시 존재해야 하며, `__init__.py`에서 `agent` 모듈을 임포트해야 ADK가 에이전트를 자동으로 인식한다.

```
email-refiner-agent/
├── .python-version          # Python 버전 (3.13)
├── pyproject.toml           # 프로젝트 의존성 정의
├── uv.lock                  # 의존성 잠금 파일
├── README.md
└── email_refiner/           # 에이전트 패키지
    ├── __init__.py          # 에이전트 모듈 등록
    ├── agent.py             # 에이전트 정의
    └── prompt.py            # 프롬프트 및 설명 모음
```

**`__init__.py`의 역할:**

```python
from . import agent
```

이 한 줄이 매우 중요하다. ADK 프레임워크는 패키지 내의 `agent` 모듈을 자동 탐색하는데, `__init__.py`에서 이를 명시적으로 임포트해야 ADK의 에이전트 디스커버리(discovery)가 작동한다.

#### 2) 의존성 설정 (pyproject.toml)

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

주요 의존성:
- **`google-adk`**: Google Agent Development Kit 핵심 라이브러리
- **`google-genai`**: Google Generative AI 클라이언트
- **`litellm`**: 다양한 LLM 제공자(OpenAI, Anthropic, Google 등)를 통합 인터페이스로 사용할 수 있게 해주는 라이브러리

#### 3) 다중 전문 에이전트 설계

이메일 리파이너는 5개의 전문 에이전트로 구성된다. 각 에이전트는 이메일 개선의 서로 다른 측면을 담당한다:

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

**각 에이전트의 역할:**

| 에이전트 | 역할 | 핵심 임무 |
|----------|------|-----------|
| ClarityEditorAgent | 명확성 편집자 | 모호함 제거, 중복 문구 삭제, 문장 간결화 |
| ToneStylistAgent | 톤 스타일리스트 | 따뜻하고 자신감 있는 톤 유지, 전문성 유지 |
| PersuationAgent | 설득 전략가 | CTA 강화, 논증 구조화, 수동적 표현 제거 |
| EmailSynthesizerAgent | 이메일 종합가 | 모든 개선 사항을 하나의 이메일로 통합 |
| LiteraryCriticAgent | 문학 비평가 | 최종 품질 검토 및 승인/재작업 결정 |

#### 4) 프롬프트 설계 패턴

프롬프트는 `description`(에이전트 역할 설명)과 `instruction`(상세 지시사항)으로 분리하여 관리한다. 이는 관심사 분리(Separation of Concerns) 원칙을 따르는 것이다.

```python
# 설명 (description) - 에이전트가 무엇인지 간단히 정의
CLARITY_EDITOR_DESCRIPTION = "Expert editor focused on clarity and simplicity."

# 지시사항 (instruction) - 에이전트가 어떻게 동작해야 하는지 상세 기술
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

특히 주목할 점은 **파이프라인 패턴**이다. 각 에이전트의 instruction에서 이전 에이전트의 출력을 참조하는 템플릿 변수를 사용한다:

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

이 `{clarity_output}`, `{tone_output}` 등의 변수는 다음 섹션에서 배울 `output_key`와 연결된다.

### 실습 포인트

1. ADK 프로젝트 디렉토리를 직접 생성하고 `__init__.py`에서 agent 모듈을 임포트하는 구조를 만들어 보자.
2. 각 에이전트의 프롬프트를 읽고, 이메일 개선 파이프라인의 흐름을 도식으로 그려 보자.
3. `LiteLlm`을 사용하여 OpenAI 모델 외에 다른 모델(예: `anthropic/claude-3-haiku`)로 교체해 보자.

---

## 12.1 LoopAgent - 반복 에이전트와 에스컬레이션

### 주제 및 목표

ADK의 `LoopAgent`를 사용하여 여러 에이전트가 반복적으로 협업하는 시스템을 구축한다. `output_key`를 통한 에이전트 간 데이터 공유와, `escalate`를 통한 루프 종료 메커니즘을 학습한다.

### 핵심 개념 설명

#### 1) output_key - 에이전트 간 데이터 전달

`output_key`는 에이전트의 출력을 세션 상태(state)에 저장하는 키 이름을 지정한다. 이전 섹션에서 본 프롬프트의 `{clarity_output}`, `{tone_output}` 등의 템플릿 변수가 바로 이 `output_key`를 통해 채워진다.

```python
MODEL = LiteLlm(model="openai/gpt-4o")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
    output_key="clarity_output",    # 출력이 state["clarity_output"]에 저장
    model=MODEL,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
    output_key="tone_output",       # 출력이 state["tone_output"]에 저장
    model=MODEL,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
    output_key="persuasion_output", # 출력이 state["persuasion_output"]에 저장
    model=MODEL,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
    output_key="synthesized_output", # 출력이 state["synthesized_output"]에 저장
    model=MODEL,
)
```

**데이터 흐름:**

```
사용자 이메일 입력
    │
    ▼
ClarityEditorAgent ──── output_key="clarity_output" ──────► state에 저장
    │
    ▼
ToneStylistAgent ────── output_key="tone_output" ──────────► state에 저장
    │                    (instruction에서 {clarity_output} 참조)
    ▼
PersuationAgent ─────── output_key="persuasion_output" ────► state에 저장
    │                    (instruction에서 {tone_output} 참조)
    ▼
EmailSynthesizerAgent ─ output_key="synthesized_output" ───► state에 저장
    │                    (3개의 output 모두 참조)
    ▼
LiteraryCriticAgent ─── 품질 판단
    │                    (instruction에서 {synthesized_output} 참조)
    ├── 불합격 → 루프 다시 시작
    └── 합격 → escalate로 루프 종료
```

#### 2) ToolContext와 escalate - 루프 종료 메커니즘

`LoopAgent`는 기본적으로 무한 반복(또는 `max_iterations`까지)하는데, 특정 조건에서 루프를 탈출하려면 `escalate` 메커니즘을 사용한다.

```python
from google.adk.tools.tool_context import ToolContext

async def escalate_email_complete(tool_context: ToolContext):
    """Use this tool only when the email is good to go."""
    tool_context.actions.escalate = True
    return "Email optimization complete."
```

**핵심 포인트:**
- `ToolContext`는 ADK가 도구 실행 시 자동으로 주입하는 컨텍스트 객체이다.
- `tool_context.actions.escalate = True`를 설정하면 현재 루프를 즉시 종료한다.
- 이 도구는 `LiteraryCriticAgent`에게만 부여되어, 비평가가 이메일 품질에 만족했을 때만 루프가 종료된다.

```python
literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
    tools=[
        escalate_email_complete,   # escalate 도구 부여
    ],
    model=MODEL,
)
```

#### 3) LoopAgent 구성

모든 서브 에이전트를 `LoopAgent`로 묶어 반복 실행 구조를 완성한다:

```python
email_refiner_agent = LoopAgent(
    name="EmailRefinerAgent",
    max_iterations=50,                    # 최대 50회 반복 (안전장치)
    description=EMAIL_OPTIMIZER_DESCRIPTION,
    sub_agents=[
        clarity_agent,                     # 1. 명확성 개선
        tone_stylist_agent,                # 2. 톤 조정
        persuation_agent,                  # 3. 설득력 강화
        email_synthesizer_agent,           # 4. 종합
        literary_critic_agent,             # 5. 최종 검토 (escalate 가능)
    ],
)

root_agent = email_refiner_agent
```

**`root_agent` 변수의 중요성:** ADK 프레임워크는 `root_agent`라는 이름의 변수를 자동 탐색하여 진입점(entry point) 에이전트로 사용한다. 반드시 이 이름을 사용해야 한다.

#### 4) 프롬프트 강화 - LLM에게 도구 호출을 확실히 지시하기

실제 운영에서 LLM이 도구를 호출하겠다고 "말만" 하고 실제로 호출하지 않는 문제가 발생할 수 있다. 이를 방지하기 위해 프롬프트를 강화했다:

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

이처럼 대문자와 강조 표현으로 LLM에게 도구 호출의 실행을 명확히 지시하는 것은 실전에서 매우 유용한 프롬프트 엔지니어링 기법이다.

### 실습 포인트

1. `max_iterations`를 3으로 낮추고 실행하여, 루프가 최대 반복 횟수에 도달할 때의 동작을 관찰해 보자.
2. `escalate_email_complete` 함수에서 `escalate = True` 대신 반환값만 주면 어떻게 되는지 확인해 보자.
3. `output_key`를 제거하고 실행하면 다음 에이전트가 이전 결과를 참조하지 못하는 것을 확인해 보자.

---

## 12.3 API Server - ADK 내장 API 서버

### 주제 및 목표

ADK의 내장 웹 서버를 활용하여 에이전트를 REST API로 서빙하는 방법을 학습한다. 새로운 Travel Advisor Agent를 만들어 API 서버를 통해 상호작용한다.

### 핵심 개념 설명

#### 1) Travel Advisor Agent - 도구 기반 에이전트

API 서버 시연을 위해 도구(tool)를 활용하는 여행 어드바이저 에이전트를 새로 구축한다:

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
            # ... 더 많은 관광지 데이터
        ],
        "Tokyo": [
            {"name": "Tokyo Tower", "type": "landmark", "rating": 4.5,
             "description": "Communications and observation tower"},
            {"name": "Senso-ji", "type": "temple", "rating": 4.6,
             "description": "Ancient Buddhist temple"},
            # ... 더 많은 관광지 데이터
        ],
        "default": [
            {"name": "City Center", "type": "area", "rating": 4.2,
             "description": "Main downtown area"},
            # ... 기본 관광지 데이터
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

**도구 함수 설계 패턴:**
- 모든 도구 함수는 `async` 비동기 함수로 정의한다.
- 첫 번째 매개변수는 반드시 `tool_context: ToolContext`이다 (ADK가 자동 주입).
- docstring이 LLM에게 도구의 용도를 설명하는 역할을 한다.
- 반환값은 딕셔너리 형태로, LLM이 해석하여 사용자에게 응답한다.

에이전트 등록:

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

#### 2) ADK 내장 API 서버 실행

ADK는 `adk api_server` 명령어로 내장 웹 서버를 즉시 실행할 수 있다. 이 서버는 FastAPI 기반이며, 에이전트와 상호작용할 수 있는 REST 엔드포인트를 자동으로 제공한다.

```bash
# 에이전트 프로젝트가 있는 상위 디렉토리에서 실행
adk api_server email-refiner-agent/
```

서버가 시작되면 `http://127.0.0.1:8000`에서 접근 가능하다.

#### 3) REST API를 통한 에이전트 상호작용

**세션 생성:**

```python
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"

# 새 세션 생성
response = requests.post(
    f"{BASE_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions"
)
print(response.json())
# 세션 ID가 포함된 응답을 받는다
```

ADK API 서버의 세션 생성 엔드포인트 패턴:
```
POST /apps/{앱_이름}/users/{사용자_ID}/sessions
```

**메시지 전송 (동기 모드):**

```python
SESSION_ID = "ce085ce3-9637-4eca-b7a1-b0be58fa39f1"  # 세션 생성 시 받은 ID

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

**응답 파싱:**

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

응답은 이벤트 배열 형태이며, 각 이벤트의 `content.parts`에서:
- `functionCall`: 에이전트가 호출한 도구 정보
- `text`: 에이전트의 텍스트 응답

#### 4) 의존성 업데이트

API 서버 기능과 평가(eval) 기능을 사용하기 위해 의존성이 추가되었다:

```toml
dependencies = [
    "google-adk[eval]>=1.12.0",   # [eval] extra 추가
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",            # HTTP 클라이언트 (API 호출용)
    "sseclient-py>=1.8.0",        # SSE 클라이언트 (다음 섹션)
]
```

### 실습 포인트

1. `adk api_server` 명령어로 서버를 실행하고, 브라우저에서 `http://127.0.0.1:8000/docs`에 접속하여 자동 생성된 Swagger UI를 확인해 보자.
2. 세션을 생성하고, 여러 메시지를 순차적으로 보내 대화 맥락이 유지되는지 확인해 보자.
3. 다른 `APP_NAME`으로 `email_refiner` 에이전트에도 API 요청을 보내 보자.

---

## 12.4 Server Sent Events (SSE) - 실시간 스트리밍 응답

### 주제 및 목표

동기식 `/run` 엔드포인트 대신 `/run_sse` 엔드포인트를 사용하여 Server-Sent Events 기반의 실시간 스트리밍 응답을 처리하는 방법을 학습한다.

### 핵심 개념 설명

#### 1) SSE(Server-Sent Events)란?

SSE는 서버에서 클라이언트로 실시간 데이터를 단방향으로 스트리밍하는 HTTP 기반 프로토콜이다. WebSocket과 달리 일반 HTTP 연결을 사용하므로 구현이 간단하다.

**동기 모드 vs SSE 모드 비교:**

| 특성 | `/run` (동기) | `/run_sse` (스트리밍) |
|------|---------------|----------------------|
| 응답 방식 | 전체 응답 한 번에 반환 | 이벤트 단위로 실시간 전송 |
| 사용자 경험 | 응답 완료까지 대기 | 실시간으로 진행 상황 확인 가능 |
| 도구 호출 관찰 | 결과에 포함 | 실시간으로 호출 과정 관찰 가능 |
| 적합한 상황 | 짧은 응답, 백엔드 처리 | 긴 응답, 프론트엔드 UI |

#### 2) SSE 클라이언트 구현

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
    "streaming": True,              # 스트리밍 활성화 플래그
}

response = requests.post(
    f"{BASE_URL}/run_sse",           # SSE 전용 엔드포인트
    json=message,
    stream=True,                     # requests의 스트리밍 모드 활성화
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

**동기 모드와의 코드 차이점:**

1. **요청 메시지에 `"streaming": True` 추가** - 서버에 스트리밍 모드를 알린다.
2. **엔드포인트 변경**: `/run` 대신 `/run_sse` 사용
3. **`stream=True` 옵션**: `requests.post()`에 스트리밍 모드 활성화
4. **`sseclient.SSEClient`로 래핑**: 응답을 SSE 이벤트 스트림으로 파싱
5. **이벤트 루프**: `client.events()`로 이벤트를 하나씩 처리

#### 3) SSE 이벤트 구조

각 SSE 이벤트는 JSON 형식의 `data` 필드를 포함한다:

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

또는 텍스트 응답:

```json
{
    "content": {
        "parts": [
            {
                "text": "파리의 현재 날씨는 22도이며..."
            }
        ]
    }
}
```

### 실습 포인트

1. 동기 모드(`/run`)와 SSE 모드(`/run_sse`)로 같은 질문을 보내고, 응답 시간과 사용자 경험의 차이를 비교해 보자.
2. SSE 이벤트를 수신하면서 도구 호출(functionCall)이 텍스트 응답보다 먼저 도착하는 것을 관찰해 보자.
3. `sseclient-py` 대신 직접 HTTP 스트림을 파싱하는 코드를 작성해 보자 (`response.iter_lines()` 활용).

---

## 12.6 Runner - 코드에서 직접 에이전트 실행하기

### 주제 및 목표

ADK CLI나 API 서버 없이, 순수 Python 코드에서 `Runner` 클래스를 사용하여 에이전트를 직접 실행하는 방법을 학습한다. `DatabaseSessionService`를 통한 영속적 세션 관리와 `InMemoryArtifactService`를 함께 다룬다.

### 핵심 개념 설명

#### 1) Runner의 역할

`Runner`는 에이전트 실행의 핵심 오케스트레이터이다. API 서버 없이 코드 내에서 직접 에이전트를 실행하고 싶을 때 사용한다. Runner는 다음을 관리한다:
- 에이전트 실행 흐름
- 세션 상태 관리
- 아티팩트(파일 등) 관리
- 이벤트 스트리밍

#### 2) 세션 서비스와 아티팩트 서비스

```python
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService

# 아티팩트 서비스: 메모리 기반 (파일 등 임시 저장)
in_memory_service_py = InMemoryArtifactService()

# 세션 서비스: SQLite DB 기반 (영속적 세션 저장)
session_service = DatabaseSessionService(db_url="sqlite:///./session.db")
```

**`DatabaseSessionService`의 장점:**
- 세션 데이터가 SQLite 파일(`session.db`)에 영속적으로 저장된다.
- 서버 재시작 후에도 이전 대화를 이어갈 수 있다.
- `db_url`을 PostgreSQL 등으로 변경하면 프로덕션 환경에서도 사용 가능하다.

#### 3) 세션 생성과 상태 초기화

```python
session = await session_service.create_session(
    app_name="weather_agent",
    user_id="u_123",
    state={
        "user_name": "nico",    # 초기 상태에 사용자 이름 저장
    },
)
```

`state` 딕셔너리에 초기값을 설정할 수 있다. 이 값은 에이전트의 instruction에서 템플릿 변수로 참조된다:

```python
# prompt.py
TRAVEL_ADVISOR_INSTRUCTION = """
You are a helpful travel advisor agent...

You call the user by their name:

Their name is {user_name}
...
"""
```

`{user_name}`은 세션 state의 `"user_name"` 값으로 자동 치환된다. 이것이 ADK의 **상태 기반 프롬프트 템플릿** 기능이다.

#### 4) Runner를 통한 에이전트 실행

```python
from google.genai import types
from google.adk.runners import Runner

# Runner 생성
runner = Runner(
    agent=travel_advisor_agent,           # 실행할 에이전트
    session_service=session_service,      # 세션 관리 서비스
    app_name="weather_agent",             # 앱 이름 (세션 서비스와 일치해야 함)
    artifact_service=in_memory_service_py, # 아티팩트 관리 서비스
)

# 사용자 메시지 생성
message = types.Content(
    role="user",
    parts=[
        types.Part(text="Im going to Vietnam, tell me all about it."),
    ],
)

# 비동기 스트리밍 실행
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

**이벤트 처리 패턴:**
- `event.is_final_response()`: 최종 텍스트 응답인지 확인
- `event.get_function_calls()`: 도구 호출 이벤트 확인
- `event.get_function_responses()`: 도구 응답 이벤트 확인

#### 5) 실행 결과 분석

실제 실행 결과를 보면 에이전트의 동작 과정을 명확히 관찰할 수 있다:

```
# 1단계: 에이전트가 3개의 도구를 동시에 호출 (병렬 도구 호출)
[FunctionCall(name='get_weather', args={'location': 'Vietnam'}),
 FunctionCall(name='get_exchange_rate', args={'from_currency': 'USD', 'to_currency': 'VND', 'amount': 1}),
 FunctionCall(name='get_local_attractions', args={'location': 'Vietnam'})]

# 2단계: 도구 응답 수신
[FunctionResponse(name='get_weather', response=<dict len=6>),
 FunctionResponse(name='get_exchange_rate', response=<dict len=6>),
 FunctionResponse(name='get_local_attractions', response={
     'error': "Invoking `get_local_attractions()` failed as the following
     mandatory input parameters are not present: category..."
 })]

# 3단계: 최종 응답 (도구 결과를 종합하여 자연어로 답변)
Hello Nico! Here's some information to help you prepare for your trip to Vietnam:

### Weather in Vietnam
- **Current Temperature:** 22°C
- **Condition:** Partly cloudy
...
```

주목할 점:
1. 에이전트가 세션 상태에서 `{user_name}`을 읽어 "Hello Nico!"로 인사한다.
2. 3개의 도구를 **병렬로** 호출하여 효율적으로 정보를 수집한다.
3. `get_local_attractions`에서 `category` 매개변수 누락 에러가 발생했지만, 에이전트가 스스로 대처하여 일반적인 베트남 관광지 정보를 직접 생성했다.

### 실습 포인트

1. `DatabaseSessionService`를 `InMemorySessionService`로 교체하고, 서버 재시작 후 세션이 유지되지 않는 것을 확인해 보자.
2. `state`에 `"preferred_language": "Korean"` 같은 값을 추가하고, 프롬프트에서 이를 활용하여 한국어로 답변하게 만들어 보자.
3. `run_async` 대신 동기식 `run` 메서드를 사용하는 방법을 찾아보자.
4. `output_key`를 사용하여 에이전트 응답을 세션 상태에 저장하고, 다음 대화에서 참조해 보자.

---

## 12.7 Deployment to Vertex AI - 클라우드 배포

### 주제 및 목표

구축한 ADK 에이전트를 Google Cloud의 Vertex AI Agent Engine에 배포하여 프로덕션 환경에서 운영하는 방법을 학습한다.

### 핵심 개념 설명

#### 1) Vertex AI Agent Engine이란?

Vertex AI Agent Engine(구 Reasoning Engine)은 Google Cloud에서 AI 에이전트를 호스팅하고 관리하는 서비스이다. ADK로 만든 에이전트를 클라우드에 배포하면:
- 서버 인프라 관리 불필요
- 자동 스케일링
- Google Cloud의 보안 및 모니터링 기능 활용
- 원격 세션 관리 및 실행

#### 2) 배포 스크립트 (deploy.py)

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

# Vertex AI 초기화
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,         # 배포 파일 스테이징용 GCS 버킷
)

# ADK 에이전트를 AdkApp으로 래핑
app = reasoning_engines.AdkApp(
    agent=travel_advisor_agent,
    enable_tracing=True,            # 실행 추적 활성화
)

# Vertex AI에 배포
remote_app = vertexai.agent_engines.create(
    display_name="Travel Advisor Agent",
    agent_engine=app,
    requirements=[                  # 필요한 Python 패키지
        "google-cloud-aiplatform[adk,agent_engines]",
        "litellm",
    ],
    extra_packages=["travel_advisor_agent"],  # 에이전트 패키지 포함
    env_vars={                      # 환경 변수 전달
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    },
)
```

**배포 과정 상세 분석:**

| 단계 | 코드 | 설명 |
|------|------|------|
| 1. 환경 설정 | `dotenv.load_dotenv()` | `.env` 파일에서 API 키 등 환경 변수 로드 |
| 2. Vertex AI 초기화 | `vertexai.init(...)` | 프로젝트, 리전, 스테이징 버킷 설정 |
| 3. 앱 래핑 | `reasoning_engines.AdkApp(...)` | ADK 에이전트를 Vertex AI 호환 형태로 래핑 |
| 4. 배포 | `agent_engines.create(...)` | 클라우드에 실제 배포 실행 |

**`extra_packages` 매개변수의 역할:**
로컬 패키지 디렉토리(`travel_advisor_agent`)를 배포 번들에 포함시킨다. 이렇게 해야 에이전트 코드가 클라우드 환경에서도 임포트 가능하다.

**`env_vars`를 통한 시크릿 관리:**
API 키 같은 민감한 정보를 환경 변수로 전달한다. 코드에 직접 하드코딩하지 않는 것이 보안상 중요하다.

#### 3) 추가 의존성

배포를 위해 추가된 패키지:

```toml
dependencies = [
    "cloudpickle>=3.1.1",                                    # 객체 직렬화
    "google-adk[eval]>=1.12.0",
    "google-cloud-aiplatform[adk,agent-engines]>=1.111.0",   # Vertex AI SDK
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",
    "sseclient-py>=1.8.0",
]
```

- **`cloudpickle`**: Python 객체를 직렬화하여 클라우드로 전송하는 데 사용
- **`google-cloud-aiplatform[adk,agent-engines]`**: Vertex AI의 ADK 및 Agent Engine 기능 포함

#### 4) 원격 에이전트 관리 및 실행 (remote.py)

```python
import vertexai
from vertexai import agent_engines

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# 배포 목록 조회
# deployments = agent_engines.list()
# for deployment in deployments:
#     print(deployment)

# 특정 배포 ID로 원격 앱 가져오기
DEPLOYMENT_ID = "projects/23382131925/locations/europe-southwest1/reasoningEngines/2153529862441140224"
remote_app = agent_engines.get(DEPLOYMENT_ID)

# 배포 삭제 (force=True로 강제 삭제)
remote_app.delete(force=True)
```

**원격 세션 생성 및 스트리밍 쿼리:**

```python
# 원격 세션 생성
# remote_session = remote_app.create_session(user_id="u_123")
# print(remote_session["id"])

SESSION_ID = "5724511082748313600"

# 원격 에이전트에 스트리밍 쿼리 전송
# for event in remote_app.stream_query(
#     user_id="u_123",
#     session_id=SESSION_ID,
#     message="I'm going to Laos, any tips?",
# ):
#     print(event, "\n", "=" * 50)
```

**원격 실행 API 요약:**

| 메서드 | 용도 |
|--------|------|
| `agent_engines.list()` | 모든 배포 목록 조회 |
| `agent_engines.get(id)` | 특정 배포 가져오기 |
| `remote_app.create_session(user_id=...)` | 원격 세션 생성 |
| `remote_app.stream_query(...)` | 스트리밍 방식으로 쿼리 |
| `remote_app.delete(force=True)` | 배포 삭제 |

### 실습 포인트

1. GCP 프로젝트와 GCS 버킷을 생성하고, 실제로 에이전트를 배포해 보자.
2. `enable_tracing=True`로 배포 후, Google Cloud Console에서 트레이싱 로그를 확인해 보자.
3. `remote_app.stream_query()`와 로컬 Runner 실행의 응답 시간을 비교해 보자.
4. 여러 사용자 ID로 세션을 생성하고, 세션 격리가 올바르게 동작하는지 확인해 보자.

---

## 챕터 핵심 정리

### 1. ADK 에이전트 아키텍처

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
         │ (단일)  │   │ (반복)    │   │ (실행기)  │
         └─────────┘   └───────────┘   └───────────┘
```

### 2. 실행 모드 비교

| 실행 방식 | 설명 | 사용 시점 |
|-----------|------|-----------|
| `adk web` | 웹 UI로 에이전트 테스트 | 개발 중 빠른 테스트 |
| `adk api_server` | REST API 서버 실행 | 프론트엔드 연동, 로컬 서비스 |
| `Runner` (코드 모드) | Python 코드에서 직접 실행 | 커스텀 애플리케이션 통합 |
| Vertex AI 배포 | 클라우드 프로덕션 환경 | 실서비스 운영 |

### 3. 핵심 ADK 클래스/컴포넌트 정리

| 컴포넌트 | 역할 |
|----------|------|
| `Agent` | 단일 에이전트 정의 (이름, 설명, 지시, 모델, 도구) |
| `LoopAgent` | 서브 에이전트를 반복 실행하는 오케스트레이터 |
| `LiteLlm` | 다양한 LLM 제공자를 통합 인터페이스로 사용 |
| `ToolContext` | 도구 함수에서 세션 상태, 액션 접근 |
| `output_key` | 에이전트 출력을 세션 상태에 저장하는 키 |
| `escalate` | 루프 또는 에이전트 체인을 조기 종료 |
| `Runner` | 코드에서 에이전트 실행을 관리하는 오케스트레이터 |
| `DatabaseSessionService` | DB 기반 영속적 세션 관리 |
| `InMemoryArtifactService` | 메모리 기반 아티팩트 관리 |
| `reasoning_engines.AdkApp` | ADK 에이전트를 Vertex AI 배포 형태로 래핑 |

### 4. 데이터 흐름 핵심 패턴

```
output_key로 저장 → state에 축적 → instruction의 {변수명}으로 참조
```

이 패턴은 ADK에서 에이전트 간 데이터를 전달하는 가장 중요한 메커니즘이다.

---

## 실습 과제

### 과제 1: 코드 리뷰 에이전트 (LoopAgent 활용)

Email Refiner Agent의 구조를 참고하여 **코드 리뷰 에이전트**를 만들어 보자.

**요구사항:**
- `SecurityReviewAgent`: 보안 취약점 검토
- `PerformanceReviewAgent`: 성능 최적화 제안
- `StyleReviewAgent`: 코드 스타일 및 가독성 검토
- `ReviewSynthesizerAgent`: 모든 리뷰를 종합
- `ApprovalAgent`: 최종 승인/반려 결정 (escalate 도구 사용)

**힌트:**
- 각 에이전트에 `output_key`를 설정하여 리뷰 결과를 state에 저장
- `ApprovalAgent`에 `escalate_review_complete` 도구를 부여
- `LoopAgent`의 `max_iterations`를 적절히 설정

### 과제 2: API 서버와 SSE 클라이언트

Travel Advisor Agent를 확장하여 **레스토랑 추천 기능**을 추가하고, API 서버와 SSE 클라이언트를 구현하자.

**요구사항:**
1. `get_restaurant_recommendations(location, cuisine_type)` 도구 함수 추가
2. `adk api_server`로 서버 실행
3. SSE 클라이언트로 실시간 스트리밍 응답 수신
4. 도구 호출 이벤트와 텍스트 응답 이벤트를 구분하여 UI에 표시

### 과제 3: Runner를 활용한 대화형 CLI

Runner를 사용하여 터미널에서 대화형으로 에이전트와 소통하는 CLI 프로그램을 만들어 보자.

**요구사항:**
1. `DatabaseSessionService`를 사용하여 대화 기록 영속 저장
2. 프로그램 시작 시 기존 세션 이어가기 또는 새 세션 생성 선택
3. `state`에 사용자 선호 언어를 저장하고 프롬프트에서 활용
4. `Ctrl+C`로 종료 시 세션 ID를 출력하여 다음에 이어갈 수 있도록 구현

### 과제 4: Vertex AI 배포 (심화)

Travel Advisor Agent를 실제 Vertex AI에 배포하고 원격으로 사용해 보자.

**요구사항:**
1. GCP 프로젝트 생성 및 Vertex AI API 활성화
2. GCS 버킷 생성 (스테이징용)
3. `deploy.py`를 참고하여 배포 스크립트 작성
4. `remote.py`를 참고하여 원격 세션 생성 및 쿼리 실행
5. 배포된 에이전트의 응답 시간을 로컬 실행과 비교 분석

**주의사항:**
- GCP 요금이 발생할 수 있으므로 테스트 후 반드시 `remote_app.delete(force=True)`로 삭제
- API 키를 코드에 하드코딩하지 말고 반드시 환경 변수로 전달

---

> **다음 챕터 예고:** 다음 챕터에서는 에이전트 평가(Evaluation) 프레임워크를 다루며, 에이전트의 응답 품질을 체계적으로 측정하고 개선하는 방법을 학습한다.
