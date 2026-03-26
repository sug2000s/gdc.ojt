# Chapter 07: OpenAI Agents SDK 기초와 Streamlit 통합

---

## 1. 챕터 개요

이 챕터에서는 **OpenAI Agents SDK**(`openai-agents`)를 활용하여 AI 에이전트를 구축하는 기초부터 실전 UI 통합까지를 다룬다. 단순한 에이전트 생성에서 시작하여, 스트리밍 이벤트 처리, 세션 메모리를 통한 대화 유지, 에이전트 간 핸드오프(Handoff), 구조화된 출력(Structured Output), 그래프 시각화, 그리고 최종적으로 Streamlit을 이용한 웹 UI 구축까지 점진적으로 학습한다.

### 학습 목표

- OpenAI Agents SDK의 핵심 구성요소(`Agent`, `Runner`, `function_tool`)를 이해한다
- 스트리밍 응답에서 이벤트를 처리하는 두 가지 방법(고수준/저수준)을 익힌다
- `SQLiteSession`을 통한 세션 기반 메모리 관리를 구현한다
- 멀티 에이전트 시스템에서 Handoff 패턴을 설계한다
- Pydantic `BaseModel`을 활용한 구조화된 출력과 에이전트 그래프 시각화를 학습한다
- Streamlit 프레임워크의 기본 위젯과 데이터 흐름(Data Flow) 모델을 이해한다

### 프로젝트 구조

```
chatgpt-clone/
├── .gitignore
├── .python-version          # Python 3.13.3
├── pyproject.toml           # 프로젝트 의존성 설정
├── uv.lock                  # uv 패키지 매니저 락 파일
├── dummy-agent.ipynb        # 에이전트 실험용 Jupyter 노트북
├── main.py                  # Streamlit 웹 애플리케이션
├── ai-memory.db             # SQLite 세션 메모리 DB
└── README.md
```

### 핵심 의존성

```toml
[project]
name = "chatgpt-clone"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    "python-dotenv>=1.1.1",
    "streamlit>=1.48.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

- **`openai-agents[viz]`**: OpenAI Agents SDK (시각화 확장 포함)
- **`graphviz`**: 에이전트 그래프를 SVG로 시각화
- **`streamlit`**: 웹 UI 프레임워크
- **`python-dotenv`**: 환경변수(.env) 관리
- **`ipykernel`**: Jupyter 노트북 커널 (개발용)

---

## 2. 각 섹션별 상세 설명

---

### 2.1 섹션 7.0 - Introduction (프로젝트 초기 설정)

**커밋**: `fbc2f97`

#### 주제 및 목표

프로젝트의 기본 골격을 잡는 단계이다. `uv` 패키지 매니저를 사용하여 Python 3.13 기반의 프로젝트를 초기화하고, 필요한 의존성을 설치한다.

#### 핵심 개념 설명

**uv 패키지 매니저**

이 프로젝트는 `pip` 대신 `uv`를 사용한다. `uv`는 Rust로 작성된 초고속 Python 패키지 매니저로, `pyproject.toml`과 `uv.lock` 파일을 통해 의존성을 관리한다. `.python-version` 파일에 `3.13.3`이 명시되어 있어, `uv`가 자동으로 해당 Python 버전을 사용한다.

**프로젝트 초기화 명령어 (참고)**

```bash
uv init chatgpt-clone
cd chatgpt-clone
uv add "openai-agents[viz]" python-dotenv streamlit
uv add --dev ipykernel
```

**openai-agents SDK란?**

OpenAI에서 공식 제공하는 에이전트 프레임워크로, 다음과 같은 핵심 기능을 제공한다:

| 구성요소 | 설명 |
|----------|------|
| `Agent` | 에이전트를 정의하는 클래스. 이름, 지시문(instructions), 도구(tools) 등을 설정 |
| `Runner` | 에이전트를 실행하는 클래스. 동기/비동기/스트리밍 실행 지원 |
| `function_tool` | Python 함수를 에이전트가 사용할 수 있는 도구로 변환하는 데코레이터 |
| `SQLiteSession` | SQLite 기반의 세션 메모리 관리 |
| `ItemHelpers` | 스트리밍 이벤트에서 메시지를 추출하는 유틸리티 |

#### 실습 포인트

1. `uv`를 설치하고 프로젝트를 초기화해 본다
2. `pyproject.toml`의 의존성 구조를 살펴본다
3. Jupyter 노트북에서 `.venv` 커널을 선택하여 개발 환경을 확인한다

---

### 2.2 섹션 7.2 - Stream Events (스트리밍 이벤트 처리)

**커밋**: `996dae4`

#### 주제 및 목표

에이전트의 응답을 실시간으로 스트리밍 처리하는 방법을 학습한다. 고수준 이벤트 처리와 저수준(raw) 이벤트 처리, 두 가지 접근 방식을 모두 다룬다.

#### 핵심 개념 설명

**에이전트와 도구 정의**

먼저 간단한 에이전트와 도구를 정의한다:

```python
from agents import Agent, Runner, function_tool, ItemHelpers


@function_tool
def get_weather(city: str):
    """Get weather by city"""
    return "30 degrees"


agent = Agent(
    name="Assistant Agent",
    instructions="You are a helpful assistant. Use tools when needed to answer questions",
    tools=[get_weather],
)
```

핵심 포인트:
- `@function_tool` 데코레이터는 일반 Python 함수를 에이전트의 도구로 변환한다
- 함수의 **docstring**이 도구의 설명(description)으로 사용된다 -- 에이전트가 이 설명을 보고 언제 도구를 사용할지 판단한다
- 함수의 **타입 힌트**(예: `city: str`)가 도구의 파라미터 스키마로 자동 변환된다
- `Agent`에 `tools` 리스트로 도구를 전달한다

**방법 1: 고수준 이벤트 처리 (run_item_stream_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        continue
    elif event.type == "agent_updated_stream_event":
        print("Agent updated to", event.new_agent.name)
    elif event.type == "run_item_stream_event":
        if event.item.type == "tool_call_item":
            print(event.item.raw_item.to_dict())
        elif event.item.type == "tool_call_output_item":
            print(event.item.output)
        elif event.item.type == "message_output_item":
            print(ItemHelpers.text_message_output(event.item))
    print("=" * 20)
```

이 방식에서는 스트리밍 이벤트를 **세 가지 카테고리**로 분류하여 처리한다:

| 이벤트 타입 | 설명 |
|------------|------|
| `raw_response_event` | 가공되지 않은 원시 응답 (이 방식에서는 무시) |
| `agent_updated_stream_event` | 현재 활성 에이전트가 변경되었을 때 발생 |
| `run_item_stream_event` | 실행 항목(메시지, 도구 호출 등)이 생성되었을 때 발생 |

`run_item_stream_event` 내부의 항목 타입:

| 항목 타입 | 설명 |
|----------|------|
| `tool_call_item` | 에이전트가 도구를 호출할 때 |
| `tool_call_output_item` | 도구 실행 결과가 반환될 때 |
| `message_output_item` | 에이전트의 텍스트 응답 |

**방법 2: 저수준 이벤트 처리 (raw_response_event)**

```python
stream = Runner.run_streamed(
    agent, "Hello how are you? What is the weather in the capital of Spain?"
)

message = ""
args = ""

async for event in stream.stream_events():

    if event.type == "raw_response_event":
        event_type = event.data.type
        if event_type == "response.output_text.delta":
            message += event.data.delta
            print(message)
        elif event_type == "response.function_call_arguments.delta":
            args += event.data.delta
            print(args)
        elif event_type == "response.completed":
            message = ""
            args = ""
```

이 방식은 `raw_response_event`를 직접 처리하여 **토큰 단위의 실시간 스트리밍**을 구현한다:

| raw 이벤트 타입 | 설명 |
|----------------|------|
| `response.output_text.delta` | 텍스트 응답의 토큰 조각(delta) |
| `response.function_call_arguments.delta` | 도구 호출 인자의 조각(delta) |
| `response.completed` | 하나의 응답이 완료됨 |

실행 결과를 보면, 도구 호출 인자가 점진적으로 구성되는 과정을 볼 수 있다:

```
{"
{"city
{"city":"
{"city":"Madrid
{"city":"Madrid"}
```

그 후 텍스트 응답도 토큰 단위로 누적된다:

```
Hello
Hello!
Hello! I'm
Hello! I'm doing
Hello! I'm doing well
...
Hello! I'm doing well, thank you. The weather in Madrid, the capital of Spain, is currently 30 degrees Celsius. How can I assist you further?
```

#### 두 방식의 비교

| 특성 | 고수준 (run_item) | 저수준 (raw_response) |
|------|-------------------|----------------------|
| 세밀도 | 항목(item) 단위 | 토큰(delta) 단위 |
| 용도 | 로직 처리, 상태 관리 | 실시간 UI 업데이트 |
| 복잡도 | 낮음 | 높음 (직접 문자열 누적 필요) |
| ChatGPT 같은 UI | 부적합 | 적합 (타이핑 효과) |

#### 실습 포인트

1. 두 가지 스트리밍 방식을 각각 실행하고 출력 차이를 비교한다
2. `delta`를 누적하여 전체 메시지를 복원하는 로직을 이해한다
3. `response.completed` 이벤트에서 `message`과 `args`를 초기화하는 이유를 생각해 본다 -- 하나의 실행에서 여러 응답이 발생할 수 있기 때문이다

---

### 2.3 섹션 7.3 - Session Memory (세션 메모리)

**커밋**: `35a1fe4`

#### 주제 및 목표

에이전트가 이전 대화 내용을 기억하도록 `SQLiteSession`을 사용한 세션 메모리를 구현한다. 이를 통해 멀티턴(multi-turn) 대화가 가능해진다.

#### 핵심 개념 설명

**세션 메모리가 필요한 이유**

기본적으로 `Runner.run()`을 호출할 때마다 에이전트는 이전 대화를 전혀 기억하지 못한다. 매 호출이 독립적인 새로운 대화이다. 사용자가 "내 이름은 Nico야"라고 말한 후, 다음 호출에서 "내 이름이 뭐였지?"라고 물어도 대답할 수 없다.

`SQLiteSession`은 대화 이력을 SQLite 데이터베이스에 자동으로 저장하고, 다음 호출 시 자동으로 불러와서 에이전트에게 제공한다.

**SQLiteSession 설정**

```python
from agents import Agent, Runner, function_tool, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")
```

- 첫 번째 인자 `"user_1"`: **세션 식별자**. 같은 식별자를 사용하면 같은 대화 이력을 공유한다
- 두 번째 인자 `"ai-memory.db"`: SQLite 데이터베이스 파일 경로

**세션을 활용한 대화 실행**

```python
result = await Runner.run(
    agent,
    "What was my name again?",
    session=session,
)

print(result.final_output)
```

`Runner.run()`에 `session=session` 파라미터를 전달하면, SDK가 자동으로:
1. 해당 세션의 이전 대화 이력을 DB에서 불러온다
2. 새로운 사용자 메시지와 함께 에이전트에게 전달한다
3. 에이전트의 응답을 DB에 저장한다

**세션 데이터 확인**

```python
await session.get_items()
```

이 메서드는 세션에 저장된 전체 대화 이력을 리스트로 반환한다. 반환되는 데이터의 구조를 살펴보면:

```python
[
    {'content': 'Hello how are you? My name is Nico', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'I live in Spain', 'role': 'user'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What is the weather in the third biggest city of the country i live on', 'role': 'user'},
    {'arguments': '{"city":"Valencia"}', 'name': 'get_weather', 'type': 'function_call', ...},
    {'call_id': '...', 'output': '30 degrees', 'type': 'function_call_output'},
    {'id': 'msg_...', 'content': [...], 'role': 'assistant', ...},
    {'content': 'What was my name again?', 'role': 'user'},
    {'id': 'msg_...', 'content': [{'text': 'Your name is Nico.', ...}], 'role': 'assistant', ...}
]
```

주목할 점:
- 대화 이력에는 사용자 메시지, 에이전트 응답뿐 아니라 **도구 호출(`function_call`)과 결과(`function_call_output`)**도 함께 저장된다
- 에이전트는 "스페인의 세 번째로 큰 도시"라는 맥락 의존적 질문에서 이전 대화의 "나는 스페인에 살아"라는 정보를 활용하여 **Valencia**를 정확히 추론했다
- "내 이름이 뭐였지?"라는 질문에도 이전 대화에서 "내 이름은 Nico야"라는 정보를 기억하여 정확히 대답했다

#### 실습 포인트

1. `SQLiteSession`의 세션 ID를 바꿔가며 서로 다른 대화가 독립적으로 유지되는지 확인한다
2. `ai-memory.db` 파일을 삭제하면 대화 이력이 초기화되는 것을 확인한다
3. `await session.get_items()`로 저장된 대화 이력의 구조를 직접 살펴본다
4. 맥락 의존적 질문(예: "그 나라의 수도는?")을 만들어 세션 메모리가 제대로 동작하는지 테스트한다

---

### 2.4 섹션 7.4 - Handoffs (에이전트 간 핸드오프)

**커밋**: `b23f34f`

#### 주제 및 목표

여러 전문 에이전트를 정의하고, 메인 에이전트가 사용자의 질문을 분석하여 적절한 전문 에이전트에게 **핸드오프(위임)**하는 멀티 에이전트 패턴을 학습한다.

#### 핵심 개념 설명

**핸드오프(Handoff)란?**

핸드오프는 하나의 에이전트가 다른 에이전트에게 대화의 제어권을 넘기는 것이다. 이는 마치 고객센터에서 일반 상담원이 전문 상담원에게 전화를 연결해 주는 것과 같다. OpenAI Agents SDK에서는 `handoffs` 파라미터를 통해 이를 구현한다.

**전문 에이전트 정의**

```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("user_1", "ai-memory.db")


geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
)
economics_agent = Agent(
    name="Economics Expert Agent",
    instructions="You are a expert in economics, you answer questions related to them.",
    handoff_description="Use this to answer economics questions.",
)
```

각 전문 에이전트에는 두 가지 중요한 설정이 있다:

| 파라미터 | 역할 |
|---------|------|
| `instructions` | 에이전트 자신이 따를 지시문. 해당 에이전트가 실행될 때 시스템 프롬프트로 사용 |
| `handoff_description` | **다른 에이전트(메인 에이전트)가** 이 에이전트에게 위임할지 판단할 때 참고하는 설명 |

**메인 에이전트 (오케스트레이터)**

```python
main_agent = Agent(
    name="Main Agent",
    instructions="You are a user facing agent. Transfer to the agent most capable of answering the user's question.",
    handoffs=[
        economics_agent,
        geaography_agent,
    ],
)
```

- `handoffs` 리스트에 위임 가능한 에이전트들을 등록한다
- `instructions`에서 "Transfer to the agent most capable..."라고 명시하여, 메인 에이전트가 라우터(router) 역할을 하도록 한다
- 메인 에이전트는 사용자의 질문 내용과 각 에이전트의 `handoff_description`을 비교하여 가장 적합한 에이전트를 선택한다

**실행 및 결과 확인**

```python
result = await Runner.run(
    main_agent,
    "Why do countries sell bonds?",
    session=session,
)

print(result.last_agent.name)
print(result.final_output)
```

출력:
```
Economics Expert Agent
Countries sell bonds as a way to raise funds for various purposes...
```

- `result.last_agent.name`을 통해 최종적으로 응답을 생성한 에이전트가 누구인지 확인할 수 있다
- "국가는 왜 채권을 파는가?"라는 질문에 대해 메인 에이전트가 이를 경제 관련 질문으로 판단하고 `Economics Expert Agent`에게 핸드오프했다

**핸드오프 흐름 요약**

```
사용자: "Why do countries sell bonds?"
    |
    v
[Main Agent] -- 질문 분석 --> 경제 관련 질문으로 판단
    |
    v  (handoff)
[Economics Expert Agent] -- 답변 생성 --> "Countries sell bonds..."
    |
    v
사용자에게 응답 전달
```

#### 실습 포인트

1. 지리 관련 질문("세계에서 가장 긴 강은?")과 경제 관련 질문("인플레이션이란?")을 각각 보내고, 어떤 에이전트가 응답하는지 확인한다
2. 두 분야에 걸치는 모호한 질문("한국의 GDP에 지리가 미치는 영향은?")을 보내고 어떤 에이전트가 선택되는지 관찰한다
3. 새로운 전문 에이전트(예: 역사 전문가)를 추가하여 핸드오프 대상을 확장해 본다
4. `handoff_description`을 수정하여 라우팅 결과가 어떻게 달라지는지 실험한다

---

### 2.5 섹션 7.5 - Viz and Structured Outputs (시각화 및 구조화된 출력)

**커밋**: `45d261a`

#### 주제 및 목표

에이전트 시스템의 구조를 그래프로 시각화하는 방법과, Pydantic `BaseModel`을 사용하여 에이전트의 출력을 정해진 구조로 강제하는 방법을 학습한다.

#### 핵심 개념 설명

**구조화된 출력 (Structured Output)**

기본적으로 에이전트는 자유형식의 텍스트를 반환한다. 하지만 프로그래밍적으로 응답을 처리해야 할 때는 일정한 구조가 필요하다. `output_type` 파라미터에 Pydantic 모델을 지정하면, 에이전트가 반드시 해당 구조에 맞는 JSON을 반환하도록 강제할 수 있다.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    background_explanation: str
```

이 모델을 에이전트에 적용한다:

```python
geaography_agent = Agent(
    name="Geo Expert Agent",
    instructions="You are a expert in geography, you answer questions related to them.",
    handoff_description="Use this to answer geography related questions.",
    tools=[
        get_weather,
    ],
    output_type=Answer,
)
```

실행 결과:
```
answer="The capital of Thailand's northern province, Chiang Mai, is Chiang Mai City."
background_explanation="Chiang Mai is both a city and a province in northern Thailand..."
```

- 에이전트의 응답이 자유 텍스트가 아닌, `answer`와 `background_explanation` 필드를 가진 구조화된 객체로 반환된다
- 이를 통해 `result.final_output.answer`처럼 프로그래밍적으로 특정 필드에 접근할 수 있다

**에이전트 그래프 시각화**

```python
from agents.extensions.visualization import draw_graph

draw_graph(main_agent)
```

`draw_graph()` 함수는 에이전트 시스템의 구조를 **SVG 그래프**로 시각화한다. 생성되는 그래프에는 다음 요소가 포함된다:

| 노드 | 색상 | 의미 |
|------|------|------|
| `__start__` | 하늘색 (ellipse) | 실행 시작점 |
| `__end__` | 하늘색 (ellipse) | 실행 종료점 |
| `Main Agent` | 연노랑 (rectangle) | 메인 에이전트 |
| `Economics Expert Agent` | 흰색 (rounded rect) | 전문 에이전트 |
| `Geo Expert Agent` | 흰색 (rounded rect) | 전문 에이전트 |
| `get_weather` | 연초록 (ellipse) | 도구 (function tool) |

그래프의 엣지(화살표)는 핸드오프 관계와 도구 사용 관계를 나타낸다:
- **실선 화살표**: 에이전트 간 핸드오프 방향
- **점선 화살표**: 에이전트와 도구 간의 호출/반환 관계

**의존성 추가**

시각화 기능을 사용하려면 `graphviz` 패키지가 필요하다:

```toml
dependencies = [
    "graphviz>=0.21",
    "openai-agents[viz]>=0.2.6",
    ...
]
```

#### 실습 포인트

1. `draw_graph(main_agent)`를 실행하여 에이전트 시스템의 구조를 시각화한다
2. `Answer` 모델에 필드를 추가하거나 수정하여 출력 구조를 커스터마이즈한다
3. 새로운 전문 에이전트를 추가한 후 그래프가 어떻게 변하는지 확인한다
4. `output_type`이 없는 에이전트와 있는 에이전트의 `result.final_output` 타입 차이를 비교한다

---

### 2.6 섹션 7.8 - Welcome To Streamlit (Streamlit 기초)

**커밋**: `e763a74`

#### 주제 및 목표

Streamlit 프레임워크를 소개하고, 다양한 UI 위젯을 활용하여 웹 인터페이스를 구축하는 기초를 학습한다. 또한 `trace` 기능을 통한 에이전트 실행 추적도 간략히 다룬다.

#### 핵심 개념 설명

**Streamlit이란?**

Streamlit은 Python만으로 웹 애플리케이션을 빠르게 만들 수 있는 프레임워크이다. HTML, CSS, JavaScript 없이도 데이터 시각화나 AI 데모 앱을 구축할 수 있다. 실행 명령:

```bash
streamlit run main.py
```

**기본 위젯 활용**

```python
import streamlit as st
import time


st.header("Hello world!")

st.button("Click me please!")

st.text_input(
    "Write your API KEY",
    max_chars=20,
)

st.feedback("faces")
```

| 위젯 | 설명 |
|------|------|
| `st.header()` | 제목 텍스트 표시 |
| `st.button()` | 클릭 가능한 버튼 |
| `st.text_input()` | 텍스트 입력 필드 |
| `st.feedback()` | 피드백 위젯 (표정 아이콘) |

**사이드바**

```python
with st.sidebar:
    st.badge("Badge 1")
```

`st.sidebar` 컨텍스트 매니저를 사용하면 왼쪽 사이드바에 위젯을 배치할 수 있다.

**탭 레이아웃**

```python
tab1, tab2, tab3 = st.tabs(["Agent", "Chat", "Outpu"])

with tab1:
    st.header("Agent")
with tab2:
    st.header("Agent 2")
with tab3:
    st.header("Agent 3")
```

`st.tabs()`로 탭 인터페이스를 생성하고, 각 탭의 컨텍스트 안에서 콘텐츠를 배치한다.

**채팅 인터페이스**

```python
with st.chat_message("ai"):
    st.text("Hello!")
    with st.status("Agent is using tool") as status:
        time.sleep(1)
        status.update(label="Agent is searching the web....")
        time.sleep(2)
        status.update(label="Agent is reading the page....")
        time.sleep(3)
        status.update(state="complete")

with st.chat_message("human"):
    st.text("Hi!")


st.chat_input(
    "Write a message for the assistant.",
    accept_file=True,
)
```

| 위젯 | 설명 |
|------|------|
| `st.chat_message("ai")` | AI 역할의 채팅 메시지 버블 |
| `st.chat_message("human")` | 사용자 역할의 채팅 메시지 버블 |
| `st.status()` | 진행 상태 표시 위젯 (로딩 상태, 완료 상태 등) |
| `st.chat_input()` | 채팅 입력 필드 (파일 첨부 지원 가능) |

`st.status()`는 에이전트가 도구를 사용하는 과정을 사용자에게 시각적으로 보여주기 위한 핵심 위젯이다. `status.update()`로 레이블을 실시간으로 변경하고, `state="complete"`로 완료 상태를 표시한다.

**trace 기능 (노트북 측)**

```python
from agents import trace

with trace("user_111111"):
    result = await Runner.run(
        main_agent,
        "What is the capital of Colombia's northen province.",
        session=session,
    )
    result = await Runner.run(
        main_agent,
        "What is the capital of Cambodia's northen province.",
        session=session,
    )
```

`trace()` 컨텍스트 매니저를 사용하면 여러 `Runner.run()` 호출을 하나의 추적 단위로 묶을 수 있다. 이는 OpenAI의 대시보드에서 에이전트 실행 과정을 디버깅하고 모니터링할 때 유용하다.

#### 실습 포인트

1. `streamlit run main.py`로 앱을 실행하고 각 위젯의 동작을 확인한다
2. `st.status()`의 라벨을 변경하고 시간 간격을 조정하여 UX를 실험한다
3. `st.chat_message()`의 역할을 "ai", "human" 외에 커스텀 이름으로도 설정해 본다
4. 사이드바에 다양한 위젯을 추가해 본다

---

### 2.7 섹션 7.9 - Streamlit Data Flow (Streamlit 데이터 흐름)

**커밋**: `8c438f5`

#### 주제 및 목표

Streamlit의 핵심 동작 원리인 **데이터 흐름(Data Flow)** 모델과 **세션 상태(`st.session_state`)** 관리를 학습한다.

#### 핵심 개념 설명

**Streamlit의 데이터 흐름 모델**

Streamlit의 가장 중요한 특성: **사용자가 위젯과 상호작용할 때마다 전체 스크립트(`main.py`)가 위에서 아래로 다시 실행된다.** 이것이 Streamlit의 "Data Flow" 모델이다.

예를 들어:
1. 사용자가 텍스트를 입력하면 -> `main.py` 전체가 재실행
2. 버튼을 클릭하면 -> `main.py` 전체가 재실행
3. 체크박스를 토글하면 -> `main.py` 전체가 재실행

이 특성 때문에, 스크립트 내의 일반 변수는 매번 초기화된다. 따라서 상호작용 간에 상태를 유지하려면 `st.session_state`를 사용해야 한다.

**session_state를 이용한 상태 관리**

```python
import streamlit as st

if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

st.header("Hello!")

name = st.text_input("What is your name?")

if name:
    st.write(f"Hello {name}")
    st.session_state["is_admin"] = True


print(st.session_state["is_admin"])
```

**코드 흐름 분석:**

1. **초기 상태 확인**: `"is_admin"` 키가 `st.session_state`에 없으면 `False`로 초기화한다. 이 패턴은 첫 실행에서만 초기화하고, 이후 재실행에서는 기존 값을 유지하기 위한 것이다.

2. **위젯 렌더링**: `st.text_input()`은 텍스트 입력 필드를 표시하고, 사용자가 입력한 값을 `name` 변수에 반환한다.

3. **조건부 처리**: `name`에 값이 있으면(빈 문자열이 아니면) 인사 메시지를 표시하고 `is_admin`을 `True`로 변경한다.

4. **상태 확인**: `print()`는 서버 콘솔에 현재 `is_admin` 상태를 출력한다.

**session_state의 핵심 포인트:**

```
[첫 번째 실행]
- st.session_state["is_admin"] = False  (초기화)
- name = ""  (아직 입력 없음)
- print(False)

[사용자가 "Nico" 입력 -> 전체 스크립트 재실행]
- "is_admin" in st.session_state == True  (이미 존재하므로 초기화 건너뜀)
- name = "Nico"
- st.write("Hello Nico")
- st.session_state["is_admin"] = True
- print(True)

[사용자가 입력을 지움 -> 전체 스크립트 재실행]
- "is_admin" in st.session_state == True  (여전히 존재)
- st.session_state["is_admin"]는 이전 실행에서 True로 설정됨 (유지!)
- name = ""
- if name: 조건 불만족 -> write 실행 안 됨
- print(True)  <-- 여전히 True! session_state는 재실행 간 유지됨
```

**일반 변수 vs session_state**

| 특성 | 일반 변수 | `st.session_state` |
|------|----------|-------------------|
| 재실행 시 | 초기값으로 리셋 | 이전 값 유지 |
| 용도 | 일시적 계산 | 상태 보존 (대화 이력, 설정 등) |
| 스코프 | 현재 실행 | 브라우저 세션 전체 |

이 개념은 이후 ChatGPT 클론을 만들 때 매우 중요하다:
- 대화 이력을 `st.session_state`에 저장해야 재실행 시에도 이전 대화가 유지된다
- 에이전트 인스턴스나 세션 객체도 `st.session_state`에 보관한다

#### 실습 포인트

1. `main.py`를 실행하고, 이름을 입력한 후 터미널의 `print()` 출력을 관찰한다
2. 이름을 입력했다가 지운 후 `is_admin` 값이 어떻게 되는지 확인한다
3. `st.session_state`에 대화 이력 리스트를 저장하고, 새 메시지가 추가될 때마다 유지되는지 실험한다
4. 브라우저 탭을 새로고침하면 `session_state`가 초기화되는 것을 확인한다

---

## 3. 챕터 핵심 정리

### OpenAI Agents SDK 핵심 구성요소

| 구성요소 | 역할 | 주요 파라미터 |
|---------|------|-------------|
| `Agent` | 에이전트 정의 | `name`, `instructions`, `tools`, `handoffs`, `output_type`, `handoff_description` |
| `Runner.run()` | 동기적 실행 | `agent`, `input`, `session` |
| `Runner.run_streamed()` | 스트리밍 실행 | `agent`, `input` |
| `@function_tool` | 도구 정의 데코레이터 | docstring이 설명, 타입힌트가 스키마 |
| `SQLiteSession` | 세션 메모리 | `session_id`, `db_path` |
| `draw_graph()` | 에이전트 그래프 시각화 | `agent` |
| `trace()` | 실행 추적 | `trace_name` |

### 스트리밍 이벤트 계층 구조

```
stream.stream_events()
├── raw_response_event          # 토큰 단위 원시 이벤트
│   ├── response.output_text.delta
│   ├── response.function_call_arguments.delta
│   └── response.completed
├── agent_updated_stream_event  # 에이전트 전환 이벤트
└── run_item_stream_event       # 항목 단위 이벤트
    ├── tool_call_item
    ├── tool_call_output_item
    └── message_output_item
```

### 멀티 에이전트 패턴: 핸드오프

```
[사용자 입력]
     |
     v
[Main Agent (오케스트레이터)]
     |
     ├──(경제 질문)--> [Economics Expert Agent]
     ├──(지리 질문)--> [Geo Expert Agent]
     └──(기타)-------> [직접 응답 또는 추가 에이전트]
```

### Streamlit 핵심 원리

1. **Data Flow**: 위젯 상호작용마다 전체 스크립트 재실행
2. **session_state**: 재실행 간 상태 유지를 위한 딕셔너리
3. **채팅 UI**: `st.chat_message()`, `st.chat_input()`, `st.status()`

---

## 4. 실습 과제

### 과제 1: 기본 에이전트 만들기 (난이도: ★☆☆)

**목표**: 기본적인 에이전트를 만들고 도구를 활용하게 하라.

- `calculate` 도구를 만들어 두 숫자의 사칙연산을 수행하게 하라
- `@function_tool` 데코레이터를 사용하여 `operation: str`, `a: float`, `b: float` 파라미터를 정의하라
- 에이전트에게 "What is 123 * 456 + 789?"를 물어보고 도구를 사용하는지 확인하라

### 과제 2: 스트리밍으로 타이핑 효과 구현 (난이도: ★★☆)

**목표**: 저수준 스트리밍 이벤트를 활용하여 ChatGPT와 유사한 타이핑 효과를 구현하라.

- `Runner.run_streamed()`와 `raw_response_event`를 사용한다
- `response.output_text.delta`를 받을 때마다 터미널에 한 글자씩 출력한다 (`print(delta, end="", flush=True)`)
- 도구 호출 시 도구 이름과 인자를 실시간으로 표시하라

### 과제 3: 멀티턴 대화 에이전트 (난이도: ★★☆)

**목표**: `SQLiteSession`을 활용하여 멀티턴 대화를 구현하라.

- 사용자의 이름, 좋아하는 색, 좋아하는 음식을 기억하는 에이전트를 만들라
- 첫 번째 호출에서 이름을 알려주고, 두 번째 호출에서 색을 알려주고, 세 번째 호출에서 "내가 좋아하는 것들을 정리해줘"라고 요청하라
- 세 번째 응답에서 이전 정보가 모두 포함되는지 확인하라

### 과제 4: 3개 이상의 전문 에이전트 핸드오프 시스템 (난이도: ★★★)

**목표**: 다양한 분야의 전문 에이전트를 갖춘 라우팅 시스템을 구축하라.

- 최소 3개의 전문 에이전트를 만들라 (예: 과학, 역사, 요리)
- 각 전문 에이전트에 `output_type`으로 구조화된 출력을 설정하라 (예: `answer`, `confidence_level`, `sources`)
- `draw_graph()`로 시스템 구조를 시각화하라
- 다양한 질문을 보내고 올바른 에이전트로 라우팅되는지 테스트하라

### 과제 5: Streamlit ChatGPT 클론 UI (난이도: ★★★)

**목표**: Streamlit을 사용하여 ChatGPT와 유사한 대화형 UI를 구축하라.

- `st.session_state`에 대화 이력을 `messages` 리스트로 관리하라
- `st.chat_input()`으로 사용자 입력을 받고, `st.chat_message()`로 대화를 표시하라
- 사이드바에 "New Chat" 버튼을 만들어 대화를 초기화하는 기능을 구현하라
- (보너스) `st.status()`를 활용하여 에이전트가 "생각 중..."인 상태를 표시하라

---

## 부록: 참고 자료

- [OpenAI Agents SDK 공식 문서](https://openai.github.io/openai-agents-python/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Pydantic 공식 문서](https://docs.pydantic.dev/)
- [uv 패키지 매니저](https://docs.astral.sh/uv/)
