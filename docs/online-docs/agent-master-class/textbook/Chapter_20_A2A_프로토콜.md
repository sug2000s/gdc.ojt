# Chapter 20: A2A (Agent-to-Agent) 프로토콜

---

## 1. 챕터 개요

본 챕터에서는 **A2A(Agent-to-Agent) 프로토콜**을 학습한다. A2A 프로토콜은 서로 다른 AI 에이전트들이 **네트워크를 통해 통신**할 수 있도록 하는 개방형 표준 프로토콜이다. 이 프로토콜을 통해 서로 다른 프레임워크(Google ADK, LangGraph 등)로 구축된 에이전트들이 하나의 시스템처럼 협력할 수 있다.

### 학습 목표

- A2A 프로토콜의 개념과 필요성을 이해한다
- Google ADK의 A2A 유틸리티(`to_a2a`)를 활용하여 에이전트를 A2A 서버로 변환하는 방법을 익힌다
- `RemoteA2aAgent`를 사용하여 원격 에이전트를 서브 에이전트로 연결하는 방법을 학습한다
- Agent Card의 구조와 역할을 이해한다
- A2A 프로토콜을 직접 구현하여 LangGraph 에이전트를 A2A 서버로 만드는 방법을 익힌다
- 프레임워크에 관계없이 에이전트 간 통신이 가능함을 확인한다

### 프로젝트 구조

```
a2a/
├── .python-version          # Python 3.13
├── pyproject.toml           # 프로젝트 의존성 정의
├── remote_adk_agent/        # A2A 서버로 동작하는 원격 ADK 에이전트
│   └── agent.py
├── user-facing-agent/       # 사용자와 직접 소통하는 루트 에이전트
│   └── user_facing_agent/
│       ├── __init__.py
│       └── agent.py
└── langraph_agent/          # LangGraph 기반 A2A 서버 에이전트
    ├── graph.py
    └── server.py
```

### 핵심 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `google-adk[a2a]` | 1.15.1 | Google ADK + A2A 확장 |
| `google-genai` | 1.40.0 | Google Generative AI |
| `langchain[openai]` | 0.3.27 | LangChain + OpenAI 통합 |
| `langgraph` | 0.6.8 | LangGraph 상태 그래프 |
| `litellm` | 1.77.7 | 다양한 LLM 통합 인터페이스 |
| `fastapi[standard]` | 0.118.0 | 웹 서버 프레임워크 |
| `uvicorn` | 0.37.0 | ASGI 서버 |
| `python-dotenv` | 1.1.1 | 환경 변수 관리 |

---

## 2. 섹션별 상세 설명

---

### 20.0 Introduction - 프로젝트 초기 설정

#### 주제 및 목표

A2A 프로토콜 학습을 위한 프로젝트 환경을 구성한다. Python 3.13 기반의 새로운 프로젝트를 생성하고, 필요한 모든 의존성을 설치한다.

#### 핵심 개념 설명

**A2A 프로토콜이란?**

A2A(Agent-to-Agent)는 Google이 주도하여 개발한 개방형 프로토콜로, 서로 다른 AI 에이전트 시스템 간의 상호 운용성(interoperability)을 제공한다. 기존에는 하나의 프레임워크 내에서만 에이전트 간 통신이 가능했지만, A2A를 사용하면 **프레임워크에 관계없이** 에이전트들이 네트워크를 통해 메시지를 주고받을 수 있다.

A2A 프로토콜의 핵심 구성 요소는 다음과 같다:

1. **Agent Card**: 에이전트의 메타데이터를 담은 JSON 문서. 에이전트의 이름, 설명, 능력(capabilities), 지원하는 입출력 형식 등을 정의한다. `/.well-known/agent-card.json` 경로에서 접근할 수 있다.
2. **Message**: 에이전트 간 주고받는 메시지의 표준 형식.
3. **Transport**: 메시지 전송 방식 (예: JSON-RPC).

**왜 별도의 `a2a/` 디렉토리인가?**

A2A 프로토콜은 여러 개의 독립적인 에이전트 서버를 실행해야 하므로, 기존 프로젝트와 분리된 새로운 프로젝트 구조를 생성한다. 각 에이전트는 별도의 포트에서 독립적으로 실행된다.

#### 코드 분석

```toml
# a2a/pyproject.toml
[project]
name = "a2a"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi[standard]==0.118.0",
    "google-adk[a2a]==1.15.1",
    "google-genai==1.40.0",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.8",
    "litellm==1.77.7",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

주목할 점:
- `google-adk[a2a]`: Google ADK를 설치할 때 `[a2a]` extras를 포함한다. 이것이 A2A 관련 유틸리티(`to_a2a`, `RemoteA2aAgent` 등)를 포함시킨다.
- `fastapi`와 `uvicorn`: A2A 프로토콜 서버를 직접 구현할 때 사용하는 웹 프레임워크이다.
- `litellm`: 다양한 LLM 제공자(OpenAI, Anthropic, Google 등)를 하나의 인터페이스로 사용할 수 있게 해주는 라이브러리이다.

#### 실습 포인트

1. `uv`를 사용하여 프로젝트를 초기화하고 의존성을 설치해보자:
   ```bash
   cd a2a
   uv sync
   ```
2. `.python-version` 파일을 확인하여 Python 3.13이 지정되어 있는지 확인하자.
3. `.env` 파일을 생성하고 필요한 API 키(OpenAI 등)를 설정하자.

---

### 20.1 A2A Using ADK - ADK를 활용한 A2A 에이전트 생성

#### 주제 및 목표

Google ADK의 `to_a2a` 유틸리티를 사용하여 일반 ADK 에이전트를 A2A 프로토콜 서버로 변환하는 방법을 학습한다. 또한 사용자와 직접 소통하는 "루트 에이전트"를 생성한다.

#### 핵심 개념 설명

**두 가지 에이전트의 역할**

이 섹션에서는 두 가지 유형의 에이전트를 생성한다:

1. **원격 에이전트 (Remote Agent)**: A2A 서버로 동작하며, 특정 분야(예: 역사)에 대한 전문 지식을 제공한다. 별도의 포트(8001)에서 독립적으로 실행된다.
2. **사용자 대면 에이전트 (User-Facing Agent)**: 사용자와 직접 대화하며, 필요에 따라 원격 에이전트에게 작업을 위임한다.

**`to_a2a` 함수**

`google.adk.a2a.utils.agent_to_a2a` 모듈의 `to_a2a` 함수는 일반 ADK `Agent` 객체를 A2A 프로토콜을 지원하는 웹 애플리케이션으로 변환해준다. 이 함수가 하는 일은 다음과 같다:
- Agent Card를 자동으로 생성하여 `/.well-known/agent-card.json`에 노출
- JSON-RPC 기반의 메시지 수신 엔드포인트 생성
- 에이전트의 응답을 A2A 프로토콜 형식으로 변환

**LiteLlm 모델**

`LiteLlm`은 Google ADK에서 OpenAI, Anthropic 등 다양한 LLM 제공자의 모델을 사용할 수 있게 해주는 어댑터이다. `"openai/gpt-4o"`와 같이 `제공자/모델명` 형식으로 지정한다.

#### 코드 분석

**원격 ADK 에이전트 (A2A 서버)**

```python
# a2a/remote_adk_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)

app = to_a2a(agent, port=8001)
```

핵심 포인트:
- `load_dotenv()`를 import 직후 바로 호출한다. 이는 이후의 import문에서 환경 변수가 필요할 수 있기 때문이다.
- `Agent`를 생성할 때 `name`과 `description`을 지정한다. 이 정보는 Agent Card에 자동으로 포함된다.
- `to_a2a(agent, port=8001)`로 에이전트를 A2A 서버 앱으로 변환한다. `port=8001`은 이 서버가 사용할 포트 번호이다.
- 반환된 `app` 객체는 ASGI 애플리케이션으로, `uvicorn`을 통해 실행할 수 있다.

**사용자 대면 에이전트 (루트 에이전트)**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[],
)
```

핵심 포인트:
- 이 에이전트는 `to_a2a`를 사용하지 않는다. Google ADK의 웹 UI(`adk web`)를 통해 직접 실행된다.
- `sub_agents=[]`로 아직 서브 에이전트가 연결되어 있지 않다. 이후 섹션에서 원격 에이전트를 연결할 것이다.
- 변수 이름이 `root_agent`인 것에 주목하자. 이는 ADK 웹 UI가 인식하는 규약이다.

**`__init__.py` 파일**

```python
# a2a/user-facing-agent/user_facing_agent/__init__.py
from . import agent
```

이 파일은 ADK가 패키지 내의 `agent` 모듈을 자동으로 로드할 수 있게 해준다.

#### 실습 포인트

1. 원격 에이전트를 먼저 실행해보자:
   ```bash
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001
   ```
2. 브라우저에서 `http://localhost:8001/.well-known/agent-card.json`에 접속하여 자동 생성된 Agent Card를 확인해보자.
3. 사용자 대면 에이전트를 ADK 웹 UI로 실행해보자:
   ```bash
   cd a2a
   adk web user-facing-agent
   ```

---

### 20.2 A2A For Dummies - 에이전트에 도구 추가

#### 주제 및 목표

원격 에이전트에 도구(tool)를 추가하여, A2A를 통해 에이전트가 도구를 사용하는 과정이 올바르게 작동하는지 확인한다. 또한 `sub_agents` 대신 `tools`를 사용하는 패턴을 학습한다.

#### 핵심 개념 설명

**에이전트에 도구 추가하기**

A2A 프로토콜에서 원격 에이전트는 단순한 텍스트 응답뿐만 아니라 도구(tool)를 활용한 복잡한 작업도 수행할 수 있다. 도구가 추가된 에이전트는 사용자의 요청에 따라 도구를 호출하고, 그 결과를 바탕으로 응답을 생성한다.

**`sub_agents`에서 `tools`로의 변경**

기존에 빈 리스트(`sub_agents=[]`)였던 부분이 `tools=[dummy_tool]`로 변경되었다. 이는 에이전트의 역할을 명확히 한다:
- `sub_agents`: 다른 에이전트에게 작업을 위임
- `tools`: 에이전트가 직접 사용할 수 있는 함수형 도구

#### 코드 분석

```python
# a2a/remote_adk_agent/agent.py (변경 부분)
def dummy_tool(hello: str):
    """Dummy Tool. Helps the agent"""
    return "world"


agent = Agent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    model=LiteLlm("openai/gpt-4o"),
    tools=[dummy_tool],
)

app = to_a2a(agent, port=8001)
```

핵심 포인트:
- `dummy_tool`은 간단한 Python 함수이다. ADK는 함수의 **이름**, **매개변수 타입 힌트**, **독스트링**을 자동으로 파싱하여 LLM이 이해할 수 있는 도구 정의를 생성한다.
- 함수의 매개변수 `hello: str`에 타입 힌트가 반드시 필요하다. LLM이 어떤 값을 전달해야 하는지 알아야 하기 때문이다.
- 독스트링 `"""Dummy Tool. Helps the agent"""`는 LLM에게 이 도구의 용도를 설명하는 역할을 한다.
- `tools=[dummy_tool]`로 도구를 에이전트에 등록한다. 에이전트는 필요할 때 이 도구를 자동으로 호출할 수 있다.

**A2A를 통한 도구 호출 흐름**

```
사용자 → 루트 에이전트 → [A2A 프로토콜] → 원격 에이전트 → dummy_tool 호출 → 응답 반환
```

중요한 점은 도구 호출이 **원격 에이전트 서버 내부에서** 일어난다는 것이다. 루트 에이전트는 원격 에이전트가 어떤 도구를 사용하는지 알 필요가 없다. A2A 프로토콜은 에이전트의 **내부 구현을 캡슐화**한다.

#### 실습 포인트

1. `dummy_tool`을 실제 유용한 도구(예: 위키피디아 검색, 날짜 계산 등)로 교체해보자.
2. 도구의 독스트링을 변경해보며 LLM의 도구 호출 패턴이 어떻게 달라지는지 관찰하자.
3. 여러 개의 도구를 `tools=[tool1, tool2, tool3]`처럼 등록해보자.

---

### 20.3 RemoteA2aAgent - 원격 에이전트 연결

#### 주제 및 목표

`RemoteA2aAgent`를 사용하여 루트 에이전트가 원격 A2A 서버에 있는 에이전트를 서브 에이전트로 활용하는 방법을 학습한다. 이것이 A2A 프로토콜의 **핵심 사용 패턴**이다.

#### 핵심 개념 설명

**RemoteA2aAgent란?**

`RemoteA2aAgent`는 Google ADK에서 제공하는 클래스로, 원격 A2A 서버에서 실행 중인 에이전트를 마치 로컬 서브 에이전트처럼 사용할 수 있게 해준다. 내부적으로는 HTTP를 통해 A2A 프로토콜 메시지를 주고받지만, 사용하는 입장에서는 일반 서브 에이전트와 동일하게 취급할 수 있다.

**AGENT_CARD_WELL_KNOWN_PATH**

A2A 프로토콜의 표준에 따르면, 에이전트의 메타데이터(Agent Card)는 `/.well-known/agent-card.json` 경로에 위치해야 한다. `AGENT_CARD_WELL_KNOWN_PATH` 상수가 바로 이 경로 문자열이다. 이를 통해 `RemoteA2aAgent`는 원격 에이전트의 정보(이름, 설명, 능력, 메시지 수신 URL 등)를 자동으로 가져올 수 있다.

**에이전트 위임(Delegation) 패턴**

```
사용자: "나폴레옹에 대해 알려줘"
    ↓
루트 에이전트 (StudentHelperAgent)
    ↓ description을 보고 역사 관련 질문임을 판단
    ↓ history_agent에게 위임
    ↓
[A2A 프로토콜 - HTTP 통신]
    ↓
원격 에이전트 (HistoryHelperAgent, port 8001)
    ↓ 답변 생성
    ↓
[A2A 프로토콜 - HTTP 응답]
    ↓
루트 에이전트 → 사용자에게 답변 전달
```

루트 에이전트는 각 서브 에이전트의 `description`을 기반으로 어떤 에이전트에게 작업을 위임할지 결정한다. 따라서 `description`을 명확하고 구체적으로 작성하는 것이 매우 중요하다.

#### 코드 분석

```python
# a2a/user-facing-agent/user_facing_agent/agent.py
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)

history_agent = RemoteA2aAgent(
    name="HistoryHelperAgent",
    description="An agent that can help students with their history homework",
    agent_card=f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
    ],
)
```

핵심 포인트:

1. **RemoteA2aAgent 생성**:
   - `name`: 원격 에이전트의 이름. 원격 서버에 등록된 이름과 일치해야 한다.
   - `description`: 루트 에이전트가 위임 결정을 내릴 때 참고하는 설명.
   - `agent_card`: Agent Card의 전체 URL. `f"http://127.0.0.1:8001{AGENT_CARD_WELL_KNOWN_PATH}"`는 `http://127.0.0.1:8001/.well-known/agent-card.json`이 된다.

2. **sub_agents에 연결**:
   - `sub_agents=[history_agent]`로 원격 에이전트를 서브 에이전트 목록에 추가한다.
   - 루트 에이전트는 이제 역사 관련 질문을 받으면 `history_agent`에게 A2A 프로토콜을 통해 위임할 수 있다.

3. **투명한 추상화**:
   - `RemoteA2aAgent`는 `Agent`의 서브클래스처럼 동작한다. 루트 에이전트 입장에서는 로컬 에이전트와 원격 에이전트의 차이가 없다.
   - 네트워크 통신, 프로토콜 변환 등의 복잡한 작업은 `RemoteA2aAgent`가 내부적으로 처리한다.

#### 실습 포인트

1. 두 개의 터미널을 열어 각각 원격 에이전트와 루트 에이전트를 실행해보자:
   ```bash
   # 터미널 1: 원격 에이전트
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # 터미널 2: 루트 에이전트
   cd a2a
   adk web user-facing-agent
   ```
2. ADK 웹 UI에서 역사 관련 질문과 일반 질문을 각각 해보자. 어떤 질문이 원격 에이전트로 위임되는지 관찰하자.
3. `description`을 수정해보며 위임 결정이 어떻게 달라지는지 실험하자.

---

### 20.5 SendMessageResponse - LangGraph 기반 A2A 서버 직접 구현

#### 주제 및 목표

Google ADK가 아닌 **LangGraph**로 만든 에이전트를 A2A 서버로 동작시키는 방법을 학습한다. `to_a2a` 유틸리티 없이 A2A 프로토콜을 **직접 구현**하여, 프로토콜의 내부 동작을 깊이 이해한다. 이를 통해 **프레임워크에 관계없이** 어떤 에이전트든 A2A 프로토콜을 지원할 수 있음을 확인한다.

#### 핵심 개념 설명

**LangGraph로 에이전트 만들기**

LangGraph는 LangChain 생태계의 상태 기반 그래프 프레임워크이다. `StateGraph`를 사용하여 에이전트의 실행 흐름을 그래프로 정의한다. 각 노드는 하나의 처리 단계이고, 엣지는 흐름의 방향을 나타낸다.

**A2A 프로토콜 직접 구현**

`to_a2a` 없이 A2A 서버를 만들려면 다음 두 가지 엔드포인트를 구현해야 한다:

1. **`GET /.well-known/agent-card.json`**: Agent Card를 반환하는 엔드포인트. 에이전트의 메타데이터를 JSON 형식으로 제공한다.
2. **`POST /messages`**: 메시지를 수신하고 응답을 반환하는 엔드포인트. JSON-RPC 형식의 요청을 받아 처리한다.

**Agent Card 구조**

Agent Card는 A2A 프로토콜에서 에이전트의 "명함" 역할을 한다:

| 필드 | 설명 |
|------|------|
| `name` | 에이전트 이름 |
| `description` | 에이전트 설명 |
| `url` | 메시지를 보낼 URL |
| `protocolVersion` | A2A 프로토콜 버전 |
| `capabilities` | 에이전트의 능력 |
| `defaultInputModes` | 지원하는 입력 형식 |
| `defaultOutputModes` | 지원하는 출력 형식 |
| `skills` | 에이전트의 기술 목록 |
| `preferredTransport` | 선호하는 전송 방식 |

**SendMessageResponse 구조**

A2A 프로토콜에서 메시지 응답은 JSON-RPC 형식을 따른다. 응답의 `result` 객체에는 `kind`, `message_id`, `role`, `parts` 등의 필드가 포함된다.

#### 코드 분석

**LangGraph 그래프 정의**

```python
# a2a/langraph_agent/graph.py
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph


llm = init_chat_model("openai:gpt-4o")


class ConversationState(MessagesState):
    pass


def call_model(state: ConversationState) -> ConversationState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(ConversationState)
graph_builder.add_node("llm", call_model)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()
```

핵심 포인트:
- `init_chat_model("openai:gpt-4o")`: LangChain의 통합 모델 초기화 함수. `제공자:모델명` 형식을 사용한다 (LiteLlm의 `제공자/모델명`과 다름에 주목).
- `ConversationState(MessagesState)`: `MessagesState`를 상속하여 대화 메시지 목록을 상태로 관리한다.
- `call_model`: LLM을 호출하고 응답을 메시지 목록에 추가하는 노드 함수이다.
- 그래프 구조는 단순하다: `START → llm → END`. 사용자 메시지가 들어오면 LLM을 호출하고 종료한다.
- `graph_builder.compile()`로 실행 가능한 그래프 객체를 생성한다.

**A2A 서버 구현 (FastAPI)**

```python
# a2a/langraph_agent/server.py
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from graph import graph

app = FastAPI()


def run_graph(message: str):
    result = graph.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content


@app.get("/.well-known/agent-card.json")
def get_agent_card():
    return {
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "An agent that can help students with their philosophy homework",
        "name": "PhilosophyHelperAgent",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "skills": [
            {
                "description": "An agent that can help students with their philosophy homework",
                "id": "PhilosophyHelperAgent",
                "name": "model",
                "tags": ["llm"],
            },
        ],
        "supportsAuthenticatedExtendedCard": False,
        "url": "http://localhost:8002/messages",
        "version": "0.0.1",
    }


@app.post("/messages")
async def handle_message(req: Request):
    body = await req.json()
    messages = body.get("params").get("message").get("parts")
    messages.reverse()
    message_text = ""
    for message in messages:
        text = message.get("text")
        message_text += f"{text}\n"
    response = run_graph(message_text)
    return {
        "id": "message_1",
        "jsonrpc": "2.0",
        "result": {
            "kind": "message",
            "message_id": "239827493847289374",
            "role": "agent",
            "parts": [
                {"kind": "text", "text": response},
            ],
        },
    }
```

핵심 포인트:

1. **`run_graph` 함수**:
   - LangGraph 그래프를 호출하는 래퍼 함수이다.
   - 사용자 메시지를 `{"role": "user", "content": message}` 형식으로 변환하여 그래프에 전달한다.
   - 결과에서 마지막 메시지(LLM의 응답)의 `content`를 추출하여 반환한다.

2. **Agent Card 엔드포인트** (`GET /.well-known/agent-card.json`):
   - `protocolVersion: "0.3.0"`: A2A 프로토콜 버전 0.3.0을 사용한다.
   - `url: "http://localhost:8002/messages"`: 메시지를 수신할 URL을 명시한다. `RemoteA2aAgent`는 이 URL로 메시지를 전송한다.
   - `skills`: 이 에이전트가 제공하는 기술 목록을 정의한다.
   - `preferredTransport: "JSONRPC"`: JSON-RPC 방식으로 통신함을 명시한다.

3. **메시지 처리 엔드포인트** (`POST /messages`):
   - A2A 프로토콜의 JSON-RPC 요청을 파싱한다.
   - 요청 구조: `body.params.message.parts` 경로에서 메시지 파트들을 추출한다.
   - `messages.reverse()`: 메시지 파트의 순서를 뒤집어 최신 메시지가 먼저 오도록 한다.
   - 각 파트에서 `text`를 추출하여 하나의 문자열로 합친다.
   - 응답은 JSON-RPC 형식을 따른다: `jsonrpc: "2.0"`, `result` 객체에 에이전트의 응답이 포함된다.
   - `result.parts`는 A2A 메시지 파트 형식으로, `kind: "text"`와 실제 텍스트를 포함한다.

**루트 에이전트에 철학 에이전트 추가**

```python
# a2a/user-facing-agent/user_facing_agent/agent.py (추가 부분)
philosophy_agent = RemoteA2aAgent(
    name="PhilosophyHelperAgent",
    description="An agent that can help students with their philosophy homework",
    agent_card=f"http://127.0.0.1:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = Agent(
    name="StudentHelperAgent",
    description="An agent that can help students with their homework",
    model=LiteLlm("openai/gpt-4o"),
    sub_agents=[
        history_agent,
        philosophy_agent,
    ],
)
```

핵심 포인트:
- 철학 에이전트는 포트 8002에서 실행된다.
- 루트 에이전트의 `sub_agents`에 두 개의 원격 에이전트가 포함되어 있다.
- 루트 에이전트는 질문의 내용에 따라 역사 에이전트(포트 8001) 또는 철학 에이전트(포트 8002)에게 자동으로 위임한다.
- **핵심**: 역사 에이전트는 ADK + `to_a2a`로 만들었고, 철학 에이전트는 LangGraph + FastAPI로 직접 구현했다. 프레임워크가 다르지만 A2A 프로토콜 덕분에 동일한 방식으로 통신한다.

#### 실습 포인트

1. 세 개의 터미널을 열어 모든 에이전트를 동시에 실행해보자:
   ```bash
   # 터미널 1: 역사 에이전트 (ADK 기반)
   cd a2a/remote_adk_agent
   uvicorn agent:app --port 8001

   # 터미널 2: 철학 에이전트 (LangGraph 기반)
   cd a2a/langraph_agent
   uvicorn server:app --port 8002

   # 터미널 3: 루트 에이전트
   cd a2a
   adk web user-facing-agent
   ```
2. ADK 웹 UI에서 다양한 질문을 해보자:
   - "제2차 세계대전의 원인은?" → 역사 에이전트로 위임
   - "소크라테스의 철학이란?" → 철학 에이전트로 위임
   - "오늘 날씨 어때?" → 루트 에이전트가 직접 응답
3. `http://localhost:8002/.well-known/agent-card.json`에 접속하여 직접 구현한 Agent Card를 확인해보자.
4. 각 서버의 로그를 관찰하여 A2A 메시지가 실제로 전달되는 과정을 추적해보자.

---

## 3. 챕터 핵심 정리

### A2A 프로토콜의 핵심 원리

| 개념 | 설명 |
|------|------|
| **A2A 프로토콜** | 서로 다른 프레임워크의 에이전트들이 네트워크를 통해 통신할 수 있게 하는 개방형 표준 |
| **Agent Card** | 에이전트의 메타데이터를 담은 JSON 문서 (`/.well-known/agent-card.json`) |
| **to_a2a** | ADK 에이전트를 A2A 서버로 자동 변환하는 유틸리티 함수 |
| **RemoteA2aAgent** | 원격 A2A 서버의 에이전트를 로컬 서브 에이전트처럼 사용하게 해주는 ADK 클래스 |
| **JSON-RPC** | A2A 메시지 교환에 사용되는 통신 프로토콜 |

### 아키텍처 패턴

```
┌─────────────────────────────────────────────────────────┐
│                    사용자 (User)                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│            루트 에이전트 (StudentHelperAgent)              │
│            - Google ADK + LiteLlm                       │
│            - adk web으로 실행                             │
└──────────┬──────────────────────────────┬───────────────┘
           │ A2A (port 8001)              │ A2A (port 8002)
           ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────┐
│ 역사 에이전트          │    │ 철학 에이전트              │
│ (HistoryHelperAgent) │    │ (PhilosophyHelperAgent)  │
│ - Google ADK         │    │ - LangGraph + FastAPI    │
│ - to_a2a 사용         │    │ - A2A 직접 구현           │
│ - port 8001          │    │ - port 8002              │
└──────────────────────┘    └──────────────────────────┘
```

### 핵심 요점 5가지

1. **프레임워크 독립성**: A2A 프로토콜을 사용하면 Google ADK, LangGraph, LangChain 등 서로 다른 프레임워크로 만든 에이전트들이 하나의 시스템처럼 협력할 수 있다.

2. **Agent Card는 필수**: 모든 A2A 에이전트는 `/.well-known/agent-card.json`에 자신의 메타데이터를 노출해야 한다. 이를 통해 다른 에이전트가 자동으로 연결 정보를 얻는다.

3. **`to_a2a`로 간편 변환**: Google ADK를 사용한다면 `to_a2a()` 한 줄로 에이전트를 A2A 서버로 변환할 수 있다. Agent Card와 메시지 처리 엔드포인트가 자동으로 생성된다.

4. **직접 구현도 가능**: A2A 프로토콜은 표준 HTTP + JSON-RPC 기반이므로, 어떤 웹 프레임워크로든 직접 구현할 수 있다. Agent Card 엔드포인트와 메시지 처리 엔드포인트만 구현하면 된다.

5. **description이 라우팅의 핵심**: 루트 에이전트는 각 서브 에이전트의 `description`을 기반으로 작업 위임을 결정한다. 명확하고 구체적인 설명이 정확한 라우팅을 보장한다.

---

## 4. 실습 과제

### 과제 1: 새로운 전문 에이전트 추가 (기초)

수학 숙제를 도와주는 `MathHelperAgent`를 Google ADK + `to_a2a`를 사용하여 만들고, 포트 8003에서 실행되도록 구성하라. 루트 에이전트의 `sub_agents`에 추가하여 세 개의 전문 에이전트가 협력하도록 하라.

**요구사항:**
- 에이전트 이름: `MathHelperAgent`
- 포트: 8003
- 수학 관련 도구 최소 1개 추가 (예: 계산기 함수)
- 루트 에이전트에서 수학 질문 시 해당 에이전트로 위임 확인

### 과제 2: LangGraph 기반 A2A 서버 확장 (중급)

섹션 20.5에서 만든 LangGraph 기반 철학 에이전트를 확장하여 다음 기능을 추가하라:

**요구사항:**
- 대화 이력 관리: 여러 번의 메시지 교환에서 이전 대화 내용을 기억하도록 구현
- 에러 처리: 잘못된 요청 형식에 대한 적절한 에러 응답 반환
- Agent Card에 `capabilities` 필드를 활용하여 지원하는 기능 명시

**힌트:**
- 세션별 대화 이력을 저장하기 위해 딕셔너리를 활용할 수 있다
- JSON-RPC 에러 응답 형식을 참고하라

### 과제 3: 완전히 다른 프레임워크로 A2A 에이전트 만들기 (고급)

FastAPI와 직접 HTTP 호출(예: `httpx`로 OpenAI API 직접 호출)만을 사용하여, 어떤 AI 프레임워크도 사용하지 않고 A2A 에이전트를 구현하라. 이를 통해 A2A 프로토콜이 정말로 프레임워크에 독립적임을 증명하라.

**요구사항:**
- Google ADK, LangChain, LangGraph 등을 사용하지 않을 것
- FastAPI + `httpx`(또는 `requests`)로 OpenAI API를 직접 호출
- Agent Card와 메시지 처리 엔드포인트를 직접 구현
- 루트 에이전트의 `RemoteA2aAgent`로 연결하여 정상 동작 확인

**힌트:**
```python
import httpx

async def call_openai(message: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": message}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]
```

### 과제 4: Agent Card 탐색기 만들기 (심화)

주어진 URL에서 Agent Card를 가져와 읽기 쉽게 출력하고, 해당 에이전트에게 테스트 메시지를 보내는 CLI 도구를 만들어라.

**요구사항:**
- `python explorer.py http://localhost:8001` 형식으로 실행
- Agent Card의 모든 필드를 보기 좋게 출력
- 사용자 입력을 받아 해당 에이전트에게 A2A 메시지를 전송하고 응답을 출력
- JSON-RPC 형식에 맞는 요청/응답 처리

---

> **참고**: 이 챕터의 모든 코드는 `a2a/` 디렉토리에 위치하며, 실행 전 `.env` 파일에 필요한 API 키를 설정해야 한다. `uv sync`로 의존성을 설치한 후 각 에이전트를 별도의 터미널에서 실행하라.
