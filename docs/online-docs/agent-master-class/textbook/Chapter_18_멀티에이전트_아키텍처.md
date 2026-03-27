# Chapter 18: 멀티 에이전트 아키텍처 (Multi-Agent Architectures)

---

## 1. 챕터 개요

이 챕터에서는 LangGraph를 활용하여 **여러 AI 에이전트가 협력하는 멀티 에이전트 시스템**을 설계하고 구현하는 방법을 학습한다. 단일 에이전트의 한계를 넘어, 각각 전문화된 역할을 가진 에이전트들이 서로 소통하며 복잡한 작업을 처리하는 아키텍처를 단계별로 구축해 나간다.

### 학습 목표

- 멀티 에이전트 시스템의 필요성과 핵심 개념을 이해한다
- **네트워크 아키텍처**: 에이전트 간 직접 통신(P2P) 방식을 구현한다
- **슈퍼바이저 아키텍처**: 중앙 조정자가 에이전트를 관리하는 방식을 구현한다
- **슈퍼바이저-도구 아키텍처**: 에이전트를 도구로 캡슐화하는 고급 패턴을 익힌다
- **사전 구축 에이전트(Prebuilt)**: `langgraph-supervisor` 라이브러리로 간결하게 구현하는 방법을 배운다
- LangGraph Studio를 통한 그래프 시각화 방법을 학습한다

### 프로젝트 구조

```
multi-agent-architectures/
├── .python-version          # Python 3.13
├── pyproject.toml           # 프로젝트 의존성 정의
├── main.ipynb               # 메인 실습 노트북
├── graph.py                 # LangGraph Studio용 그래프 정의
├── langgraph.json           # LangGraph Studio 설정
└── uv.lock                  # 의존성 잠금 파일
```

### 주요 의존성

```toml
[project]
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "langgraph-supervisor==0.0.29",
    "langgraph-swarm==0.0.14",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]
```

---

## 2. 섹션별 상세 설명

---

### 18.0 Introduction - 프로젝트 초기화 및 기본 임포트

**주제 및 목표**: 멀티 에이전트 프로젝트의 기반을 마련하고, 핵심 라이브러리를 임포트한다.

#### 핵심 개념 설명

멀티 에이전트 시스템을 구축하기 위해서는 LangGraph의 여러 핵심 모듈을 이해해야 한다. 이 섹션에서는 프로젝트를 초기화하고 필요한 모든 라이브러리를 불러온다.

#### 코드 분석

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

각 임포트의 역할을 살펴보자:

| 모듈 | 역할 |
|------|------|
| `StateGraph` | 상태 기반 그래프를 생성하는 핵심 클래스. 노드와 엣지를 정의하여 에이전트의 실행 흐름을 구성한다. |
| `START`, `END` | 그래프의 시작점과 종료점을 나타내는 특수 노드 상수. |
| `Command` | 에이전트 간 전환(handoff)을 제어하는 명령 객체. `goto`로 다음 노드를, `update`로 상태 변경을 지정한다. |
| `MessagesState` | 메시지 목록을 관리하는 사전 정의된 상태 클래스. `messages` 키를 기본 제공한다. |
| `ToolNode` | 도구 호출을 처리하는 사전 구축 노드. LLM이 도구 사용을 결정하면 이 노드가 실제 실행을 담당한다. |
| `tools_condition` | LLM 응답에 도구 호출이 포함되어 있는지 판단하는 조건부 라우팅 함수. |
| `@tool` | 일반 Python 함수를 LLM이 호출 가능한 도구로 변환하는 데코레이터. |
| `init_chat_model` | `"provider:model_name"` 형식의 문자열로 다양한 LLM을 초기화하는 유틸리티. |

#### 실습 포인트

- `uv`를 사용하여 프로젝트를 초기화하고 의존성을 설치해 보라: `uv sync`
- `.env` 파일에 `OPENAI_API_KEY`를 설정해야 한다
- Python 3.13 환경이 필요하다

---

### 18.1 Network Architecture - 네트워크 아키텍처

**주제 및 목표**: 에이전트들이 서로 직접 대화를 전달(handoff)할 수 있는 **네트워크(P2P) 아키텍처**를 구현한다.

#### 핵심 개념 설명

네트워크 아키텍처에서는 **중앙 조정자 없이** 각 에이전트가 자율적으로 판단하여 다른 에이전트에게 대화를 넘길 수 있다. 이 예제에서는 한국어, 그리스어, 스페인어 고객 지원 에이전트를 만들어 고객이 사용하는 언어에 맞는 에이전트로 자동 전환하는 시스템을 구축한다.

```
┌──────────────┐     handoff     ┌──────────────┐
│ korean_agent │ ◄─────────────► │ greek_agent  │
└──────┬───────┘                 └──────┬───────┘
       │                                │
       │          handoff               │
       └────────►┌──────────────┐◄──────┘
                 │spanish_agent │
                 └──────────────┘
```

이 구조에서 각 에이전트는:
1. 자신이 이해하는 언어로 응답을 시도한다
2. 자신이 이해하지 못하는 언어가 감지되면 `handoff_tool`을 사용하여 적절한 에이전트로 전환한다

#### 코드 분석

**1단계: 상태 정의**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str

llm = init_chat_model("openai:gpt-4o")
```

`MessagesState`를 확장하여 현재 활성 에이전트(`current_agent`)와 전환을 수행한 에이전트(`transfered_by`)를 추적하는 커스텀 상태를 정의한다. 이를 통해 어떤 에이전트가 대화를 처리 중인지, 누가 전환했는지 알 수 있다.

**2단계: 에이전트 팩토리 함수**

```python
def make_agent(prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node(
        "tools",
        ToolNode(tools=tools),
    )

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    return agent_builder.compile()
```

`make_agent`는 **에이전트 팩토리 함수**로, 동일한 구조의 에이전트를 매개변수만 바꿔서 생성할 수 있게 해준다. 각 에이전트는 내부적으로 하나의 완전한 `StateGraph` 서브그래프를 구성한다:

- `agent` 노드: LLM에 프롬프트와 대화 이력을 전달하여 응답을 생성한다
- `tools` 노드: `ToolNode`가 도구 호출을 실행한다
- `tools_condition`: LLM이 도구를 호출하면 `tools` 노드로, 아니면 `END`로 라우팅한다
- `tools` -> `agent` 엣지: 도구 실행 결과를 다시 에이전트에 전달하여 추가 판단을 가능하게 한다

이 구조는 각 에이전트가 **독립적인 ReAct(Reasoning + Acting) 루프**를 가지게 해준다.

**3단계: 핸드오프 도구 정의**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    """
    Handoff to another agent.

    Use this tool when the customer speaks a language that you don't understand.

    Possible values for `transfer_to`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Possible values for `transfered_by`:
    - `korean_agent`
    - `greek_agent`
    - `spanish_agent`

    Args:
        transfer_to: The agent to transfer the conversation to
        transfered_by: The agent that transferred the conversation
    """
    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

`handoff_tool`은 네트워크 아키텍처의 **핵심 메커니즘**이다. 주요 포인트:

- `Command` 객체를 반환하여 그래프의 실행 흐름을 직접 제어한다
- `goto=transfer_to`: 지정된 에이전트 노드로 실행을 이동시킨다
- `graph=Command.PARENT`: **부모 그래프**에서 이동을 수행한다. 각 에이전트는 서브그래프이므로, 이 옵션 없이는 서브그래프 내부에서만 이동이 발생한다. `Command.PARENT`를 통해 최상위 그래프 수준에서 에이전트 간 전환이 가능해진다
- `update`로 상태를 갱신하여 현재 에이전트 정보를 추적한다
- 도구의 docstring에 가능한 값들을 명시하여 LLM이 올바른 값을 사용하도록 유도한다

**4단계: 그래프 조립**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
)

graph_builder.add_edge(START, "korean_agent")

graph = graph_builder.compile()
```

최상위 그래프에서 세 개의 에이전트 노드를 등록한다. 각 노드의 값은 `make_agent`가 반환한 컴파일된 서브그래프이다. `START -> korean_agent`로 설정했으므로 모든 대화는 한국어 에이전트에서 시작된다. 사용자가 스페인어로 말하면 한국어 에이전트가 이를 감지하고 `handoff_tool`을 사용하여 스페인어 에이전트로 전환한다.

#### 실습 포인트

- 대화 시작 에이전트를 `spanish_agent`로 바꿔보고, 한국어로 메시지를 보내 자동 전환이 되는지 확인해 보라
- 새로운 언어 에이전트(예: 일본어)를 추가해 보라. `handoff_tool`의 docstring도 함께 수정해야 한다
- `Command.PARENT`를 제거하면 어떤 오류가 발생하는지 실험해 보라

---

### 18.2 Network Visualization - 네트워크 시각화

**주제 및 목표**: LangGraph Studio를 사용하여 멀티 에이전트 그래프를 시각적으로 확인하고, 실행 흐름을 디버깅한다. 또한 자기 자신에게 전환하는 무한 루프 버그를 방어한다.

#### 핵심 개념 설명

복잡한 멀티 에이전트 시스템을 개발할 때, 그래프의 구조를 시각적으로 확인하는 것은 매우 중요하다. LangGraph Studio는 그래프의 노드와 엣지를 시각화하고, 실시간 실행 흐름을 추적할 수 있는 도구이다.

그러나 시각화를 위해서는 그래프가 어떤 노드에서 어떤 노드로 이동할 수 있는지 미리 선언해야 한다. `Command`를 사용한 동적 라우팅에서는 `destinations` 매개변수를 통해 이를 명시한다.

#### 코드 분석

**1단계: graph.py 분리 및 destinations 추가**

노트북 코드를 `graph.py`로 분리하면서, 각 에이전트 노드에 `destinations` 매개변수를 추가한다:

```python
graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You're a Korean customer support agent. You only speak and understand Korean.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "spanish_agent"),
)
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You're a Greek customer support agent. You only speak and understand Greek.",
        tools=[handoff_tool],
    ),
    destinations=("korean_agent", "spanish_agent"),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "korean_agent"),
)
```

`destinations` 매개변수는 해당 노드에서 `Command`를 통해 이동 가능한 대상 노드를 선언한다. 이는 **실행 로직에는 영향을 주지 않으며**, LangGraph Studio의 시각화 도구가 정확한 그래프 구조를 표시하기 위해 사용된다.

**2단계: 자기 자신에게 전환하는 버그 방어**

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    # ... (docstring 생략)
    if transfer_to == transfered_by:
        return {
            "error": "Stop trying to transfer to yourself and answer the question or i will fire you."
        }

    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

에이전트 프롬프트에도 방어 문구를 추가한다:

```python
response = llm_with_tools.invoke(
    f"""
    {prompt}

    You have a tool called 'handoff_tool' use it to transfer to other agent,
    don't use it to transfer to yourself.

    Conversation History:
    {state["messages"]}
    """
)
```

LLM이 가끔 자기 자신에게 전환을 시도하여 **무한 루프**에 빠질 수 있다. 이를 두 가지 레이어로 방어한다:
1. **프롬프트 레벨**: "자기 자신에게 전환하지 마라"라고 명시적으로 지시한다
2. **코드 레벨**: `transfer_to == transfered_by` 체크로 자기 자신에게의 전환을 거부하고 에러 메시지를 반환한다

**3단계: LangGraph Studio 설정**

```json
{
    "dependencies": [
        "./graph.py"
    ],
    "graphs": {
        "agent": "./graph.py:graph"
    },
    "env": ".env"
}
```

`langgraph.json` 설정 파일은 LangGraph Studio에 다음을 알려준다:
- `dependencies`: 필요한 Python 파일 목록
- `graphs`: 시각화할 그래프 객체의 위치 (`파일경로:변수명` 형식)
- `env`: 환경 변수 파일 경로

**4단계: 스트리밍 실행**

```python
for event in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hola! Necesito ayuda con mi cuenta.",
            }
        ]
    },
    stream_mode="updates",
):
    print(event)
```

`stream_mode="updates"`를 사용하면 각 노드의 상태 업데이트를 실시간으로 받을 수 있다. 실행 결과에서 한국어 에이전트가 스페인어를 감지하고 스페인어 에이전트로 전환한 후, 스페인어 에이전트가 스페인어로 응답하는 전체 흐름을 확인할 수 있다:

```
{'korean_agent': {'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
{'spanish_agent': {'messages': [...], 'current_agent': 'spanish_agent', 'transfered_by': 'korean_agent'}}
```

#### 실습 포인트

- LangGraph CLI로 스튜디오를 실행해 보라: `langgraph dev`
- `destinations`를 제거한 상태에서 스튜디오를 열어보고, 시각화가 어떻게 달라지는지 비교해 보라
- 그리스어 메시지를 보내 전환 흐름을 확인해 보라

---

### 18.3 Supervisor Architecture - 슈퍼바이저 아키텍처

**주제 및 목표**: 중앙 **슈퍼바이저** 노드가 대화를 분석하고 적절한 에이전트에게 라우팅하는 아키텍처를 구현한다.

#### 핵심 개념 설명

네트워크 아키텍처에서는 각 에이전트가 자율적으로 전환을 결정했다. 슈퍼바이저 아키텍처는 이와 다르게 **하나의 중앙 조정자(Supervisor)**가 모든 라우팅 결정을 담당한다.

```
                    ┌─────────────┐
          ┌────────►│  Supervisor │◄────────┐
          │         └──────┬──────┘         │
          │                │                │
          │         ┌──────┼──────┐         │
          │         ▼      ▼      ▼         │
    ┌─────┴──┐  ┌───┴───┐  ┌──┴──────┐     │
    │ korean │  │ greek │  │ spanish │─────┘
    │ _agent │  │_agent │  │ _agent  │
    └────────┘  └───────┘  └─────────┘
```

이 아키텍처의 장점:
- **단일 진입점**: 모든 요청이 슈퍼바이저를 거치므로 라우팅 로직이 중앙화된다
- **일관성**: 에이전트들은 라우팅에 대해 신경 쓸 필요 없이 자신의 전문 영역에만 집중한다
- **제어 용이**: 슈퍼바이저의 프롬프트 하나만 수정하면 전체 라우팅 전략을 변경할 수 있다

#### 코드 분석

**1단계: 구조화된 출력 모델 정의**

```python
from typing import Literal
from pydantic import BaseModel

class SupervisorOutput(BaseModel):
    next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
    reasoning: str
```

슈퍼바이저의 판단 결과를 `Pydantic` 모델로 정의한다:
- `next_agent`: `Literal` 타입으로 가능한 값을 제한하여 LLM이 유효한 에이전트 이름만 반환하도록 강제한다. `"__end__"`는 대화를 종료하겠다는 의미이다.
- `reasoning`: 슈퍼바이저가 해당 에이전트를 선택한 이유를 설명한다. 디버깅과 투명성에 유용하다.

**2단계: 상태 확장**

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
    reasoning: str
```

슈퍼바이저의 판단 이유를 추적하기 위해 `reasoning` 필드를 상태에 추가한다.

**3단계: 에이전트 간소화**

```python
def make_agent(prompt, tools):
    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}
    # ... (그래프 구성은 동일)
```

네트워크 아키텍처와 달리, 에이전트 프롬프트에서 `handoff_tool` 관련 지시문이 **제거**되었다. 슈퍼바이저가 라우팅을 담당하므로 에이전트는 단순히 자신의 언어로 응답만 하면 된다. `tools=[]`로 빈 도구 목록을 전달한다.

**4단계: 슈퍼바이저 노드 구현**

```python
def supervisor(state: AgentState):
    structured_llm = llm.with_structured_output(SupervisorOutput)
    response = structured_llm.invoke(
        f"""
        You are a supervisor that routes conversations to the appropriate language agent.

        Analyse the customers request and the conversation history and decide which
        agent should handle the conversation.

        The options for the next agent are:
        - greek_agent
        - spanish_agent
        - korean_agent

        <CONVERSATION_HISTORY>
        {state.get("messages", [])}
        </CONVERSATION_HISTORY>

        IMPORTANT:

        Never transfer to the same agent twice in a row.

        If an agent has replied end the conversation by returning __end__
    """
    )
    return Command(
        goto=response.next_agent,
        update={"reasoning": response.reasoning},
    )
```

슈퍼바이저의 핵심 메커니즘:

- `llm.with_structured_output(SupervisorOutput)`: LLM에게 반드시 `SupervisorOutput` 스키마에 맞는 JSON을 반환하도록 강제한다. 이를 통해 파싱 오류 없이 안정적인 라우팅이 가능하다.
- `<CONVERSATION_HISTORY>` XML 태그: 대화 이력을 명확하게 구분하여 LLM이 맥락을 정확히 파악하도록 한다.
- `"Never transfer to the same agent twice in a row"`: 무한 루프 방지를 위한 프롬프트 제약.
- `"If an agent has replied end the conversation by returning __end__"`: 에이전트가 이미 응답했으면 대화를 종료하여 불필요한 반복을 막는다.
- `Command(goto=response.next_agent)`: 슈퍼바이저는 서브그래프가 아닌 최상위 그래프의 노드이므로 `graph=Command.PARENT`가 불필요하다.

**5단계: 그래프 조립 - 순환 구조**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "supervisor",
    supervisor,
    destinations=(
        "korean_agent",
        "spanish_agent",
        "greek_agent",
        END,
    ),
)

graph_builder.add_node("korean_agent", make_agent(
    prompt="You're a Korean customer support agent. You only speak and understand Korean.",
    tools=[],
))
graph_builder.add_node("greek_agent", make_agent(
    prompt="You're a Greek customer support agent. You only speak and understand Greek.",
    tools=[],
))
graph_builder.add_node("spanish_agent", make_agent(
    prompt="You're a Spanish customer support agent. You only speak and understand Spanish.",
    tools=[],
))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("korean_agent", "supervisor")
graph_builder.add_edge("spanish_agent", "supervisor")
graph_builder.add_edge("greek_agent", "supervisor")

graph = graph_builder.compile()
```

그래프의 흐름을 정리하면:

1. `START` -> `supervisor`: 모든 대화는 슈퍼바이저에서 시작
2. `supervisor` -> `{agent}` 또는 `END`: 슈퍼바이저가 `Command`로 적절한 에이전트나 종료로 라우팅
3. `{agent}` -> `supervisor`: 에이전트가 응답을 완료하면 다시 슈퍼바이저로 돌아온다

이 순환 구조가 **슈퍼바이저 아키텍처의 핵심**이다. 슈퍼바이저는 에이전트 응답 후 대화를 종료할지(`__end__`), 다른 에이전트에게 추가로 라우팅할지 판단할 수 있다.

#### 네트워크 vs 슈퍼바이저 비교

| 특성 | 네트워크 | 슈퍼바이저 |
|------|----------|------------|
| 라우팅 결정 | 각 에이전트가 자율적으로 | 중앙 슈퍼바이저가 담당 |
| 에이전트 간 연결 | P2P (직접 연결) | 허브-스포크 (슈퍼바이저 경유) |
| 에이전트 복잡도 | 높음 (라우팅 로직 포함) | 낮음 (전문 영역만 담당) |
| 확장성 | 에이전트 추가 시 모든 에이전트 수정 | 슈퍼바이저만 수정 |
| 디버깅 | 어려움 | 용이 (reasoning 추적 가능) |

#### 실습 포인트

- `reasoning` 필드를 출력하여 슈퍼바이저가 어떤 근거로 라우팅을 결정하는지 분석해 보라
- 에이전트를 추가할 때 네트워크 아키텍처와 슈퍼바이저 아키텍처에서 각각 어떤 코드를 수정해야 하는지 비교해 보라
- `SupervisorOutput`에서 `__end__` 옵션을 제거하면 어떤 일이 벌어지는지 실험해 보라

---

### 18.4 Supervisor As Tools - 에이전트를 도구로 캡슐화

**주제 및 목표**: 에이전트를 **LLM 도구(tool)로 캡슐화**하여 슈퍼바이저가 도구 호출 메커니즘을 통해 자연스럽게 에이전트를 사용하도록 한다.

#### 핵심 개념 설명

이전 섹션의 슈퍼바이저 아키텍처에서는 `structured_output`을 사용하여 라우팅했다. 이번 섹션에서는 **각 에이전트를 도구로 변환**하고, 슈퍼바이저가 `bind_tools` + `ToolNode` 메커니즘을 통해 에이전트를 호출하도록 리팩토링한다.

```
                ┌──────────────┐
    START ────► │  Supervisor  │ ────► END
                └──────┬───────┘
                       │
                 tools_condition
                       │
                ┌──────▼───────┐
                │   ToolNode   │
                │ ┌──────────┐ │
                │ │korean_ag.│ │
                │ │spanish_ag│ │
                │ │greek_ag. │ │
                │ └──────────┘ │
                └──────────────┘
```

이 접근법의 장점:
- LLM의 **기존 도구 호출 능력**을 활용하므로 별도의 라우팅 로직 불필요
- `@tool` 데코레이터의 `description`으로 자연스러운 라우팅 기준을 제공
- `ToolNode`가 자동으로 적절한 에이전트 도구를 실행

#### 코드 분석

**1단계: 에이전트-도구 팩토리 함수**

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

def make_agent_tool(tool_name, tool_description, system_prompt, tools):

    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(
            f"""
        {system_prompt}

        Conversation History:
        {state["messages"]}
        """
        )
        return {"messages": [response]}

    agent_builder = StateGraph(AgentsState)

    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node("tools", ToolNode(tools=tools))

    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
    agent_builder.add_edge("agent", END)

    agent = agent_builder.compile()

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
    )
    def agent_tool(state: Annotated[dict, InjectedState]):
        result = agent.invoke(state)
        return result["messages"][-1].content

    return agent_tool
```

이 함수는 이전 `make_agent`와 구조적으로 비슷하지만, 핵심적인 차이가 있다:

- **반환값이 컴파일된 그래프가 아닌 `@tool` 함수**이다
- `@tool(name_or_callable=tool_name, description=tool_description)`: 도구의 이름과 설명을 매개변수로 받아 동적으로 도구를 생성한다
- `Annotated[dict, InjectedState]`: `InjectedState`는 LangGraph의 특수 어노테이션으로, **현재 그래프 상태를 도구 함수에 자동 주입**한다. LLM은 이 매개변수를 인식하지 못하며(도구 스키마에 노출되지 않음), 실행 시 LangGraph가 자동으로 상태를 전달한다.
- `result["messages"][-1].content`: 에이전트의 최종 응답 텍스트만 추출하여 반환한다

**2단계: 도구 목록 생성**

```python
tools = [
    make_agent_tool(
        tool_name="korean_agent",
        tool_description="Use this when the user is speaking korean",
        system_prompt="You're a korean customer support agent you speak in korean",
        tools=[],
    ),
    make_agent_tool(
        tool_name="spanish_agent",
        tool_description="Use this when the user is speaking spanish",
        system_prompt="You're a spanish customer support agent you speak in spanish",
        tools=[],
    ),
    make_agent_tool(
        tool_name="greek_agent",
        tool_description="Use this when the user is speaking greek",
        system_prompt="You're a greek customer support agent you speak in greek",
        tools=[],
    ),
]
```

각 에이전트 도구의 `tool_description`이 LLM의 라우팅 기준이 된다. LLM은 사용자의 언어를 감지하고, 자연스럽게 해당 언어의 에이전트 도구를 호출한다.

**3단계: 슈퍼바이저 간소화**

```python
def supervisor(state: AgentState):
    llm_with_tools = llm.bind_tools(tools=tools)
    result = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [result],
    }
```

이전 버전과 비교하면 **극적으로 간소화**되었다:
- `structured_output` 대신 `bind_tools`를 사용한다
- 복잡한 프롬프트 대신 단순히 메시지를 전달한다
- LLM이 자체적으로 도구 설명을 읽고 적절한 도구를 선택한다

**4단계: 그래프 조립**

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("tools", ToolNode(tools=tools))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", tools_condition)
graph_builder.add_edge("tools", "supervisor")
graph_builder.add_edge("supervisor", END)

graph = graph_builder.compile()
```

이전 슈퍼바이저 아키텍처에서는 각 에이전트가 별도 노드였지만, 이제는 **2개의 노드**만 있다:
- `supervisor`: LLM이 도구 호출을 결정하는 노드
- `tools`: `ToolNode`가 에이전트 도구를 실행하는 노드

`tools_condition`이 LLM 응답에 도구 호출이 포함되어 있으면 `tools` 노드로, 아니면 `END`로 라우팅한다. 이는 기본적인 ReAct 에이전트 패턴과 동일하며, 도구가 에이전트라는 점만 다르다.

#### 세 가지 아키텍처 비교

| 특성 | 네트워크 | 슈퍼바이저 | 슈퍼바이저+도구 |
|------|----------|------------|-----------------|
| 그래프 노드 수 | 에이전트 수만큼 | 에이전트 수 + 1 | 2 (supervisor + tools) |
| 라우팅 메커니즘 | Command + handoff_tool | structured_output | bind_tools + ToolNode |
| 에이전트 구현 | 서브그래프 노드 | 서브그래프 노드 | @tool 함수 내부 |
| 코드 복잡도 | 중간 | 높음 | 낮음 |

#### 실습 포인트

- `InjectedState`를 제거하고 실행해 보라. LLM이 에이전트에 어떤 인자를 전달하려 하는지 관찰해 보라
- 에이전트 도구의 `description`을 더 상세하게 작성하여 라우팅 정확도가 향상되는지 실험해 보라
- 하나의 에이전트에 실제 도구(예: 검색 도구)를 추가하여 중첩 도구 호출이 작동하는지 확인해 보라

---

### 18.5 Prebuilt Agents - 사전 구축 에이전트

**주제 및 목표**: `langgraph-supervisor` 라이브러리의 `create_supervisor`와 `create_react_agent`를 사용하여 **최소한의 코드**로 멀티 에이전트 슈퍼바이저 시스템을 구현한다.

#### 핵심 개념 설명

지금까지 수동으로 구현했던 모든 패턴(에이전트 팩토리, 슈퍼바이저, 도구 기반 라우팅)을 LangGraph가 **사전 구축 모듈**로 제공한다. `langgraph-supervisor`와 `langgraph-swarm` 패키지를 사용하면 몇 줄의 코드로 강력한 멀티 에이전트 시스템을 구축할 수 있다.

추가 의존성:
```toml
"langgraph-supervisor==0.0.29",
"langgraph-swarm==0.0.14",
```

#### 코드 분석

**1단계: 임포트 간소화**

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
```

이전 섹션에서 필요했던 `StateGraph`, `Command`, `ToolNode`, `tools_condition` 등이 모두 사라졌다. `create_react_agent`와 `create_supervisor` 두 함수만으로 충분하다.

**2단계: 전문 에이전트 생성**

```python
MODEL = "openai:gpt-5"

history_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="history_agent",
    prompt="You are a history expert. You only answer questions about history.",
)
geography_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="geography_agent",
    prompt="You are a geography expert. You only answer questions about geography.",
)
maths_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="maths_agent",
    prompt="You are a maths expert. You only answer questions about maths.",
)
philosophy_agent = create_react_agent(
    model=MODEL,
    tools=[],
    name="philosophy_agent",
    prompt="You are a philosophy expert. You only answer questions about philosophy.",
)
```

`create_react_agent`는 LangGraph에서 제공하는 사전 구축 함수로, ReAct 패턴의 에이전트를 단 한 줄로 생성한다:

- `model`: 모델 문자열 또는 초기화된 모델 객체
- `tools`: 에이전트가 사용할 도구 목록
- `name`: 에이전트의 고유 이름 (슈퍼바이저가 라우팅 시 사용)
- `prompt`: 에이전트의 시스템 프롬프트

이전 섹션에서 수동으로 구현했던 `StateGraph` + `agent` 노드 + `tools` 노드 + 엣지 연결이 모두 이 한 줄에 캡슐화되어 있다.

**3단계: 슈퍼바이저 생성**

```python
supervisor = create_supervisor(
    agents=[
        history_agent,
        maths_agent,
        geography_agent,
        philosophy_agent,
    ],
    model=init_chat_model(MODEL),
    prompt="""
    You are a supervisor that routes student questions to the appropriate subject expert.
    You manage a history agent, geography agent, maths agent, and philosophy agent.
    Analyze the student's question and assign it to the correct expert based on the subject matter:
        - history_agent: For historical events, dates, historical figures
        - geography_agent: For locations, rivers, mountains, countries
        - maths_agent: For mathematics, calculations, algebra, geometry
        - philosophy_agent: For philosophical concepts, ethics, logic
    """,
).compile()
```

`create_supervisor`가 내부적으로 수행하는 작업:
1. 각 에이전트를 도구로 변환한다 (`transfer_to_{agent_name}` 형식)
2. 슈퍼바이저 노드를 생성하고 도구를 바인딩한다
3. `ToolNode`와 조건부 엣지를 설정한다
4. 에이전트 실행 후 슈퍼바이저로 돌아오는 `transfer_back_to_supervisor` 도구를 자동 추가한다

`.compile()`을 호출하여 실행 가능한 그래프로 컴파일한다.

**4단계: 실행 및 검증**

```python
questions = [
    "When was Madrid founded?",
    "What is the capital of France and what river runs through it?",
    "What is 15% of 240?",
    "Tell me about the Battle of Waterloo",
    "What are the highest mountains in Asia?",
    "If I have a rectangle with length 8 and width 5, what is its area and perimeter?",
    "Who was Alexander the Great?",
    "What countries border Switzerland?",
    "Solve for x: 2x + 10 = 30",
]

for question in questions:
    result = supervisor.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        }
    )
    if result["messages"]:
        for message in result["messages"]:
            message.pretty_print()
```

실행 흐름을 살펴보면:

1. 사용자 질문이 슈퍼바이저에 전달된다
2. 슈퍼바이저가 `transfer_to_{agent_name}` 도구를 호출한다
3. 해당 에이전트가 질문에 답변한다
4. 에이전트가 자동으로 `transfer_back_to_supervisor`를 호출하여 슈퍼바이저로 돌아온다
5. 슈퍼바이저가 최종 응답을 반환한다

실행 결과 예시:
```
Human Message: When was Madrid founded?
Ai Message (supervisor): transfer_to_history_agent 호출
Tool Message: Successfully transferred to history_agent
Ai Message (history_agent): Madrid originated in the mid-9th century...
Ai Message (history_agent): Transferring back to supervisor
Ai Message (supervisor): 최종 응답 전달
```

#### 수동 구현 vs Prebuilt 비교

```python
# 수동 구현 (18.4): ~60줄
def make_agent_tool(...): ...
def supervisor(...): ...
graph_builder = StateGraph(...)
graph_builder.add_node(...)
# ... 여러 설정 코드

# Prebuilt (18.5): ~15줄
agent = create_react_agent(model=MODEL, tools=[], name="agent", prompt="...")
supervisor = create_supervisor(agents=[...], model=..., prompt="...").compile()
```

#### 실습 포인트

- `philosophy_agent`에 실제 도구를 추가하여 외부 데이터를 활용하는 에이전트를 만들어 보라
- 슈퍼바이저의 프롬프트를 한국어로 변경하고 정상 작동하는지 확인해 보라
- 여러 질문에 대해 슈퍼바이저가 올바른 에이전트를 선택하는 비율을 측정해 보라
- `langgraph-swarm` 패키지도 함께 설치되었다. Swarm 패턴에 대해 조사하고 Supervisor 패턴과 비교해 보라

---

## 3. 챕터 핵심 정리

### 멀티 에이전트 아키텍처 3가지 패턴

| 패턴 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **네트워크 (P2P)** | 에이전트가 `Command` + `handoff_tool`로 직접 전환 | 중앙 병목 없음, 자율성 높음 | 에이전트 추가 시 모든 에이전트 수정 필요 |
| **슈퍼바이저** | 중앙 노드가 `structured_output`으로 라우팅 | 제어 용이, 디버깅 쉬움 | 슈퍼바이저 프롬프트 복잡도 증가 가능 |
| **슈퍼바이저+도구** | 에이전트를 `@tool`로 캡슐화, `bind_tools`로 라우팅 | 코드 간결, LLM 기본 능력 활용 | 에이전트 내부 상태 접근 제한 |

### 핵심 LangGraph 개념

1. **`Command`**: 그래프의 실행 흐름을 프로그래밍적으로 제어하는 객체
   - `goto`: 다음 실행 노드 지정
   - `update`: 상태 업데이트
   - `graph=Command.PARENT`: 부모 그래프 수준에서 이동

2. **`InjectedState`**: 도구 함수에 현재 그래프 상태를 자동 주입하는 어노테이션. LLM 스키마에는 노출되지 않는다.

3. **`destinations`**: `add_node`의 매개변수로, `Command` 기반 동적 라우팅에서 가능한 대상 노드를 선언한다. 시각화 전용이며 실행 로직에 영향을 주지 않는다.

4. **서브그래프 패턴**: `make_agent`가 반환하는 컴파일된 그래프를 다른 그래프의 노드로 사용하여 계층적 구조를 만든다.

5. **`create_react_agent` / `create_supervisor`**: 위의 모든 패턴을 캡슐화한 사전 구축 함수.

### 무한 루프 방어 전략

멀티 에이전트 시스템에서 가장 흔한 문제는 **에이전트 간 무한 전환**이다. 이를 방지하기 위한 전략:

1. **프롬프트 제약**: "자기 자신에게 전환하지 마라" 명시
2. **코드 레벨 검증**: `transfer_to == transfered_by` 체크
3. **슈퍼바이저의 종료 조건**: `__end__` 옵션 제공
4. **구조적 제약**: `Literal` 타입으로 유효한 대상만 허용

---

## 4. 실습 과제

### 과제 1: 다국어 고객 지원 시스템 확장 (난이도: 중)

네트워크 아키텍처(18.1)를 기반으로 다음을 구현하라:
- 일본어, 중국어 에이전트를 추가한다
- `handoff_tool`의 docstring과 `destinations`를 적절히 수정한다
- 각 에이전트에 간단한 FAQ 도구를 추가하여 "배송 상태 확인", "환불 요청" 등의 기능을 구현한다

### 과제 2: 슈퍼바이저 아키텍처 비교 실험 (난이도: 중)

동일한 시나리오(학생 질문 라우팅)에 대해:
1. 18.3의 `structured_output` 슈퍼바이저
2. 18.4의 도구 기반 슈퍼바이저
3. 18.5의 `create_supervisor`

세 가지 방식으로 각각 구현하고, 동일한 질문 세트에 대해:
- 정확한 에이전트로 라우팅되는 비율
- 응답 시간
- 토큰 사용량

을 측정하여 비교 보고서를 작성하라.

### 과제 3: 계층적 멀티 슈퍼바이저 (난이도: 상)

`create_supervisor`를 중첩하여 다음과 같은 2단계 계층 구조를 구현하라:

```
                    ┌─────────────────┐
                    │  Main Supervisor│
                    └───┬─────────┬───┘
                        │         │
              ┌─────────▼──┐  ┌──▼──────────┐
              │Science Sup.│  │Humanities S.│
              └──┬──────┬──┘  └──┬──────┬───┘
                 │      │        │      │
              ┌──▼┐  ┌──▼┐   ┌──▼┐  ┌──▼──┐
              │물리│  │화학│   │역사│  │철학  │
              └───┘  └───┘   └───┘  └─────┘
```

- Main Supervisor: 과학/인문학 분야를 판별
- Science Supervisor: 물리/화학 에이전트 관리
- Humanities Supervisor: 역사/철학 에이전트 관리

### 과제 4: Swarm 아키텍처 탐구 (난이도: 상)

`langgraph-swarm` 패키지를 사용하여:
1. Swarm 아키텍처가 무엇인지 조사한다
2. 네트워크 아키텍처와의 차이점을 분석한다
3. 동일한 고객 지원 시나리오를 Swarm 패턴으로 구현한다
4. 네트워크, 슈퍼바이저, Swarm 세 가지 패턴의 장단점을 정리한다

---

## 부록: 주요 API 레퍼런스

### Command

```python
Command(
    goto="node_name",           # 이동할 노드
    update={"key": "value"},    # 상태 업데이트
    graph=Command.PARENT,       # 부모 그래프에서 이동 (서브그래프 내부에서 사용 시)
)
```

### create_react_agent

```python
agent = create_react_agent(
    model="openai:gpt-4o",      # 모델 문자열 또는 초기화된 모델
    tools=[tool1, tool2],        # 사용할 도구 목록
    name="agent_name",           # 에이전트 고유 이름
    prompt="시스템 프롬프트",      # 에이전트 역할 정의
)
```

### create_supervisor

```python
supervisor = create_supervisor(
    agents=[agent1, agent2],     # 관리할 에이전트 목록
    model=init_chat_model(...),  # 슈퍼바이저용 모델
    prompt="라우팅 규칙 프롬프트",  # 슈퍼바이저 지시사항
).compile()                      # 반드시 compile() 호출
```

### InjectedState

```python
from langgraph.prebuilt import InjectedState
from typing import Annotated

@tool
def my_tool(state: Annotated[dict, InjectedState]):
    # state에는 현재 그래프 상태가 자동 주입됨
    # LLM은 이 매개변수를 인식하지 못함
    return state["messages"][-1].content
```
