# Chapter 18: 멀티 에이전트 아키텍처 — 강사용 해설서

> 이 문서는 강사가 Chapter 18을 **미리 이해**하기 위한 해설서입니다.
> 강의 대본(lecture_script)과 별개로, 코드를 한 줄 한 줄 쉽게 풀어서 설명합니다.

---

## 전체 진화 흐름 (먼저 큰 그림)

```
18.1  Network (P2P)       에이전트끼리 직접 대화를 넘김     "동료끼리 전화 돌리기"
       ↓
18.2  Supervisor           중앙 관리자가 배분               "콜센터 상담원 배정"
       ↓
18.3  Supervisor as Tools  에이전트가 도구로 캡슐화         "원클릭 자동 배정"
```

각 섹션이 **라우팅 방식만 다를 뿐 같은 문제(다국어 고객 지원)**를 풀고 있습니다.
그래서 비교하면서 가르치면 장단점이 명확하게 보입니다.

### 비유: 회사 조직도

| 아키텍처 | 비유 |
|----------|------|
| **Network** | 직원 3명이 있는 작은 사무실. 한국어 담당이 스페인어 고객 전화를 받으면 "이거 네 거야" 하고 옆 사람에게 직접 넘김 |
| **Supervisor** | 콜센터. 모든 전화가 먼저 **팀장(슈퍼바이저)**에게 간다. 팀장이 "이건 한국어팀으로" 배정. 상담원은 자기 언어만 신경 쓰면 됨 |
| **Supervisor as Tools** | 자동 콜센터. 팀장이 버튼(도구)만 누르면 해당 상담원이 자동 호출됨. 가장 깔끔 |

---

## 18.1 Network Architecture — "동료끼리 전화 돌리기"

### 전체 흐름 그림 (ASCII)

```
                    ┌─────────────────────────────────────────────┐
                    │              부모 그래프 (Parent)             │
                    │                                             │
  START ──► korean_agent ◄──────► greek_agent                    │
                    ▲                   ▲                         │
                    └────► spanish_agent ◄───┘                    │
                    │                                             │
                    └─────────────────────────────────────────────┘

  각 에이전트 내부 (서브그래프):
  START ──► agent ──[tool_calls?]──► tools ──► agent ──► ...
                         │
                         └── No ──► END
```

핵심 아이디어:
- 각 에이전트는 **독립적인 서브그래프** (자기만의 ReAct 루프)
- `handoff_tool`이 `Command(graph=Command.PARENT)`로 **부모 그래프 수준에서** 다른 에이전트로 전환
- 중앙 조정자 없이 에이전트끼리 자율적으로 넘김

### import 정리

```python
import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

| import | 뭐하는 놈 |
|--------|----------|
| `StateGraph` | 그래프(워크플로우)를 만드는 설계도 |
| `START, END` | 시작점, 끝점 (특수 노드) |
| `Command` | 그래프 라우팅 명령. `goto`로 다음 노드 지정, `graph=Command.PARENT`로 부모 그래프에서 전환 |
| `MessagesState` | 채팅 전용 상태. `messages` 리스트가 내장 |
| `ToolNode` | 도구 실행을 담당하는 미리 만들어진 노드 |
| `tools_condition` | "AI가 도구를 부르고 싶어해?" 판단하는 라우팅 함수 |
| `@tool` | 일반 함수를 "AI가 호출할 수 있는 도구"로 변환 |
| `init_chat_model` | LLM을 초기화하는 함수 |

### LLM 초기화

```python
llm = init_chat_model(f"openai:{os.getenv('OPENAI_MODEL_NAME')}")
```

`.env`에서 모델명(gpt-5.1) 읽어서 LLM 객체 생성. 모든 에이전트가 이 하나의 LLM을 공유한다.

### 상태 정의 — AgentsState

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
```

`MessagesState`를 상속하면 `messages: list`가 자동 생성된다.
여기에 2개 필드를 추가:

| 필드 | 역할 |
|------|------|
| `current_agent` | 지금 대화를 처리 중인 에이전트 이름 |
| `transfered_by` | 누가 이 대화를 넘겼는지 (무한 루프 방어용) |

### 에이전트 팩토리 — `make_agent()` (가장 중요!)

```python
def make_agent(prompt, tools):
```

**같은 구조의 에이전트를 매개변수만 바꿔서 찍어내는 공장 함수.**
한국어 에이전트, 그리스어 에이전트, 스페인어 에이전트 모두 구조는 동일하고 prompt만 다르다.

```python
    def agent_node(state: AgentsState):
        llm_with_tools = llm.bind_tools(tools)
```

`bind_tools(tools)` — LLM에게 "너 이 도구들 쓸 수 있어"라고 알려주는 것.
여기서 `tools`에는 `handoff_tool`이 들어간다.

```python
        system_msg = f"{prompt}\n\nIf the customer writes in a language you cannot assist with, use the handoff_tool to transfer to the correct agent. Do not transfer to yourself."
        messages = [{"role": "system", "content": system_msg}] + state["messages"]
```

시스템 메시지를 만들어서 대화 기록 앞에 붙인다.
핵심 지시: "네가 못하는 언어면 `handoff_tool`로 넘겨라. **자기 자신에게 넘기지 마라.**"

```python
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
```

LLM이 응답 생성. 도구를 호출할 수도 있고, 직접 답할 수도 있다.

```python
    agent_builder = StateGraph(AgentsState)
    agent_builder.add_node("agent", agent_node)
    agent_builder.add_node("tools", ToolNode(tools=tools))
```

에이전트 내부에 **서브그래프**를 만든다. 노드 2개: `agent`(LLM 호출)와 `tools`(도구 실행).

```python
    agent_builder.add_edge(START, "agent")
    agent_builder.add_conditional_edges("agent", tools_condition)
    agent_builder.add_edge("tools", "agent")
```

서브그래프 구조:
- `START → agent` — 시작하면 바로 LLM 호출
- `agent → [tool_calls?]` — AI가 도구를 호출하고 싶으면 `tools`로, 아니면 `END`로
- `tools → agent` — 도구 결과를 받아서 다시 LLM이 판단

이것이 **ReAct 루프** (Chapter 14.1과 동일한 패턴!)

```python
    return agent_builder.compile()
```

서브그래프를 컴파일해서 반환. 이게 하나의 "에이전트"가 된다.

### 핸드오프 도구 — `handoff_tool` (전환의 핵심!)

```python
@tool
def handoff_tool(transfer_to: str, transfered_by: str):
    """
    Handoff to another agent.
    Use this when the customer speaks a language you cannot assist with.

    Possible values for transfer_to: korean_agent, greek_agent, spanish_agent
    Possible values for transfered_by: korean_agent, greek_agent, spanish_agent
    """
```

`@tool` 데코레이터로 AI가 호출할 수 있는 도구로 만든다.
**docstring이 매우 중요** — AI가 이걸 읽고 "어떤 에이전트로 넘길까?" 판단한다.
`Possible values`를 명시해서 AI가 엉뚱한 값을 넣지 않도록 한다.

```python
    if transfer_to == transfered_by:
        return {"error": "Cannot transfer to yourself. Please respond to the customer directly."}
```

**무한 루프 방어!** 자기 자신에게 넘기려고 하면 에러 반환.
예: 한국어 에이전트가 "korean_agent로 넘겨줘" → 차단.

```python
    return Command(
        update={
            "current_agent": transfer_to,
            "transfered_by": transfered_by,
        },
        goto=transfer_to,
        graph=Command.PARENT,
    )
```

**이 3줄이 네트워크 아키텍처의 핵심 메커니즘:**

| 파라미터 | 역할 |
|----------|------|
| `update={...}` | 상태(State)를 업데이트. 누가 넘겼는지, 지금 누구한테 가는지 기록 |
| `goto=transfer_to` | 다음에 실행할 노드 이름 (예: `"spanish_agent"`) |
| `graph=Command.PARENT` | **부모 그래프 수준에서** 전환! 이게 없으면 서브그래프 안에서만 돌아서 에러 |

> **비유:** 서브그래프는 "부서 내부"이고, `Command.PARENT`는 "회사 전체 레벨에서 다른 부서로 보내줘"라는 의미.

### 최상위 그래프 조립

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "korean_agent",
    make_agent(
        prompt="You are a helpful Korean-speaking customer support agent. Please respond to customers in Korean.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "spanish_agent"),
)
```

`make_agent()`로 만든 서브그래프를 **부모 그래프의 노드**로 등록.

| 파라미터 | 역할 |
|----------|------|
| `"korean_agent"` | 노드 이름 |
| `make_agent(...)` | 이 노드가 실행할 서브그래프 |
| `destinations=(...)` | 이 노드에서 갈 수 있는 다른 노드 목록. `Command(goto=)`가 여기 있는 이름만 사용 가능 |

그리스어, 스페인어 에이전트도 동일한 방식으로 추가:

```python
graph_builder.add_node(
    "greek_agent",
    make_agent(
        prompt="You are a helpful Greek-speaking customer support agent. Please respond to customers in Greek.",
        tools=[handoff_tool],
    ),
    destinations=("korean_agent", "spanish_agent"),
)
graph_builder.add_node(
    "spanish_agent",
    make_agent(
        prompt="You are a helpful Spanish-speaking customer support agent. Please respond to customers in Spanish.",
        tools=[handoff_tool],
    ),
    destinations=("greek_agent", "korean_agent"),
)
```

각 에이전트의 `destinations`에 **자기 자신은 빠져있다** — 이것도 자기 전환 방지의 일부.

```python
graph_builder.add_edge(START, "korean_agent")
```

시작 노드는 항상 `korean_agent`. 한국어가 아닌 메시지가 오면 korean_agent가 감지해서 넘긴다.

```python
graph = graph_builder.compile()
```

### 실행 흐름 예시

**한국어 메시지 (직접 처리):**

```
User: "안녕하세요! 계정 문제가 있어요."
  → korean_agent가 받음
  → 한국어다! 내가 처리하지
  → 직접 응답 (handoff 없음)
```

**스페인어 메시지 (전환 발생):**

```
User: "Hola! Necesito ayuda con mi cuenta."
  → korean_agent가 받음
  → 스페인어네? 나는 못해
  → handoff_tool(transfer_to="spanish_agent", transfered_by="korean_agent") 호출
  → Command(goto="spanish_agent", graph=Command.PARENT) 실행
  → spanish_agent가 받아서 스페인어로 응답
```

---

## 18.2 Supervisor Architecture — "콜센터 팀장이 배정"

### 전체 흐름 그림 (ASCII)

```
                    ┌──────────────────────┐
                    │                      │
  START ──► supervisor ──► korean_agent ──►│
                │                          │
                ├──► greek_agent ──────────►│
                │                          │
                ├──► spanish_agent ────────►│
                │                          │
                └──► __end__ (END)         │
                    └──────────────────────┘

  순환 흐름: agent 실행 후 다시 supervisor로 돌아옴
  supervisor가 __end__를 선택하면 종료
```

핵심 아이디어:
- **슈퍼바이저가 모든 라우팅 결정**을 담당
- 에이전트는 라우팅 로직 없이 **자기 역할만 수행**
- `with_structured_output`으로 안전한 라우팅 (엉뚱한 값 차단)
- `reasoning` 필드로 "왜 이 에이전트를 골랐는지" 추적 가능

### 18.1과의 차이점

| | Network (18.1) | Supervisor (18.2) |
|--|----------------|-------------------|
| 라우팅 주체 | 각 에이전트가 직접 | 슈퍼바이저가 중앙에서 |
| 에이전트 복잡도 | 높음 (handoff 로직 포함) | 낮음 (응답만) |
| 도구 | handoff_tool 필요 | 도구 없음 |
| 에이전트 추가 시 | 모든 에이전트 수정 | 슈퍼바이저만 수정 |

### import 정리

```python
import os
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
```

| import | 뭐하는 놈 |
|--------|----------|
| `Literal` | 허용되는 값을 제한. `Literal["a", "b"]`면 "a" 또는 "b"만 가능 |
| `BaseModel` | Pydantic 모델. LLM 출력을 구조화된 객체로 강제 변환할 때 사용 |
| 나머지 | 18.1과 동일 |

### 슈퍼바이저 출력 스키마 — `SupervisorOutput`

```python
class SupervisorOutput(BaseModel):
    next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
    reasoning: str
```

이게 **안전한 라우팅의 비결**.

| 필드 | 역할 |
|------|------|
| `next_agent` | 다음에 실행할 에이전트. `Literal`로 4가지 값만 허용 |
| `reasoning` | 왜 이 에이전트를 골랐는지 설명. 디버깅에 매우 유용 |

`Literal`이 하는 일: LLM이 `"japanese_agent"` 같은 엉뚱한 값을 반환하면 **자동으로 차단**.
`"__end__"`를 포함시켜서 대화 종료도 슈퍼바이저가 결정한다.

### 상태 정의

```python
class AgentsState(MessagesState):
    current_agent: str
    transfered_by: str
    reasoning: str
```

18.1과 동일하지만 `reasoning` 필드가 추가됨. 슈퍼바이저의 판단 근거를 저장.

### 에이전트 팩토리 — 18.1보다 훨씬 단순!

```python
def make_agent(prompt):
    def agent_node(state: AgentsState):
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
```

18.1과 비교하면:
- `tools` 매개변수 **없음** — 에이전트는 도구를 안 씀
- `bind_tools()` **없음** — 라우팅은 슈퍼바이저가 하니까
- 에이전트는 그냥 **시스템 프롬프트 + 대화 기록을 LLM에 넘기고 응답만 반환**

```python
    agent_builder = StateGraph(AgentsState)
    agent_builder.add_node("agent", agent_node)
    agent_builder.add_edge(START, "agent")
    agent_builder.add_edge("agent", END)
    return agent_builder.compile()
```

서브그래프 구조도 단순: `START → agent → END`. 도구 노드 없음, 조건부 엣지 없음.

> **비유:** 18.1의 에이전트는 "전화 받고 + 다른 사람에게 돌리는 것까지" 했다면,
> 18.2의 에이전트는 "전화 받고 대답만". 돌리는 건 팀장(슈퍼바이저)이 한다.

### 슈퍼바이저 노드 — 라우팅의 두뇌

```python
def supervisor(state: AgentsState):
    structured_llm = llm.with_structured_output(SupervisorOutput)
```

`with_structured_output()` — LLM의 응답을 **반드시 `SupervisorOutput` 형태로** 변환.
자유 텍스트가 아니라 `{"next_agent": "korean_agent", "reasoning": "..."}`처럼 구조화된 JSON이 나온다.

```python
    response = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": """You are a supervisor that routes conversations to the appropriate language agent.

Analyse the customer's request and decide which agent should handle the conversation.

Options: greek_agent, spanish_agent, korean_agent

Rules:
- Never transfer to the same agent twice in a row.
- If an agent has already replied, end the conversation by returning __end__"""
            },
```

시스템 프롬프트에서 슈퍼바이저에게 규칙을 명확히 지시:
- 같은 에이전트 연속 배정 금지
- 이미 응답했으면 `__end__`로 종료

```python
            {
                "role": "user",
                "content": f"Conversation so far: {state.get('messages', [])}"
            }
        ]
    )
```

대화 기록 전체를 user 메시지로 넘겨서 슈퍼바이저가 판단할 수 있게 한다.

```python
    print(f"  Supervisor → {response.next_agent} (reason: {response.reasoning[:60]})")
    return Command(
        goto=response.next_agent,
        update={"reasoning": response.reasoning},
    )
```

| 파라미터 | 역할 |
|----------|------|
| `goto=response.next_agent` | 다음에 실행할 노드. `"korean_agent"`, `"__end__"` 등 |
| `update={"reasoning": ...}` | 상태에 판단 근거 저장 |

18.1과 달리 `graph=Command.PARENT` **없음** — 슈퍼바이저는 이미 부모 그래프에 있으니까.

### 그래프 조립 — 순환 구조

```python
graph_builder = StateGraph(AgentsState)

graph_builder.add_node(
    "supervisor", supervisor,
    destinations=("korean_agent", "spanish_agent", "greek_agent", END),
)
```

슈퍼바이저 노드가 갈 수 있는 곳: 3개 에이전트 + END.

```python
graph_builder.add_node("korean_agent", make_agent(
    prompt="You are a helpful Korean-speaking customer support agent. Please respond in Korean.",
))
graph_builder.add_node("greek_agent", make_agent(
    prompt="You are a helpful Greek-speaking customer support agent. Please respond in Greek.",
))
graph_builder.add_node("spanish_agent", make_agent(
    prompt="You are a helpful Spanish-speaking customer support agent. Please respond in Spanish.",
))
```

에이전트 등록. 18.1과 달리 `tools` 매개변수 없고, `destinations`도 없다.

```python
graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("korean_agent", "supervisor")
graph_builder.add_edge("spanish_agent", "supervisor")
graph_builder.add_edge("greek_agent", "supervisor")
```

**순환 구조의 핵심!**

```
START → supervisor → agent → supervisor → agent → ... → __end__
```

모든 에이전트가 실행 후 **다시 슈퍼바이저로 돌아온다**.
슈퍼바이저가 `__end__`를 선택하면 그때 비로소 종료.

### 실행 흐름 예시

**한국어 메시지:**

```
User: "비밀번호를 바꾸고 싶어요."
  → supervisor: 한국어네 → korean_agent로 라우팅 (reasoning: "한국어로 작성됨")
  → korean_agent: "비밀번호 변경 방법은..."
  → 다시 supervisor로 돌아옴
  → supervisor: 이미 응답했으니 → __end__ (reasoning: "이미 처리 완료")
  → 종료
```

**스페인어 메시지:**

```
User: "Hola! Necesito ayuda con mi cuenta."
  → supervisor: 스페인어네 → spanish_agent로 라우팅
  → spanish_agent: "Hola, claro, te ayudo..."
  → 다시 supervisor로 돌아옴
  → supervisor: → __end__
  → 종료
```

---

## 18.3 Supervisor as Tools — "버튼 하나로 자동 배정"

### 전체 흐름 그림 (ASCII)

```
  START ──► supervisor ──[tool_calls?]──► ToolNode ──► supervisor ──► ...
                  │                       ├ korean_agent()
                  │                       ├ greek_agent()
                  └── No ──► END          └ spanish_agent()
```

핵심 아이디어:
- 에이전트를 `@tool` 함수로 캡슐화
- 슈퍼바이저가 `bind_tools` + `tools_condition`으로 자동 호출
- **별도 라우팅 로직 불필요** — LLM의 tool calling이 알아서 선택
- Chapter 14.1의 ReAct 패턴과 구조가 동일!

### 18.2와의 차이점

| | Supervisor (18.2) | Supervisor as Tools (18.3) |
|--|-------------------|---------------------------|
| 라우팅 방법 | `with_structured_output` + `Command` | LLM의 `tool_calls` (자동) |
| 에이전트 형태 | 서브그래프 | `@tool` 함수 |
| 라우팅 코드 | supervisor 함수에 직접 작성 | 없음 (LLM이 알아서) |
| 에이전트 추가 | 노드 추가 + 엣지 추가 + Literal 수정 | `@tool` 함수 추가 + 리스트에 추가 |

### import 정리

```python
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
```

| import | 뭐하는 놈 |
|--------|----------|
| `ToolNode` | 도구 실행 담당 노드. 여기서 에이전트(@tool)를 실행 |
| `tools_condition` | AI가 tool_calls를 했으면 → tools, 안 했으면 → END |
| `@tool` | 에이전트를 도구로 변환! 이게 이 아키텍처의 핵심 |
| 나머지 | 이전 섹션과 동일 |

**18.2에서 쓰던 `Command`, `Literal`, `BaseModel`이 전부 사라졌다!** 그만큼 단순해진 것.

### 상태 정의 — 가장 단순

```python
class State(MessagesState):
    pass
```

추가 필드 없음. `messages`만 있으면 충분.
`current_agent`, `transfered_by`, `reasoning` 같은 추적 필드가 다 빠졌다.

### 에이전트를 도구로 캡슐화 — 이 아키텍처의 핵심!

```python
@tool
def korean_agent(message: str) -> str:
    """Transfer to Korean customer support agent. Use when the customer speaks Korean."""
    response = llm.invoke(
        f"You're a Korean customer support agent. Respond in Korean.\nCustomer: {message}"
    )
    return response.content
```

**에이전트가 그냥 함수다!**

- `@tool` — AI가 호출할 수 있는 도구로 변환
- `"""docstring"""` — AI가 이걸 읽고 "한국어 고객이니까 이 도구를 쓰자" 판단
- `message: str` — 고객 메시지를 받아서
- `llm.invoke(...)` — 해당 언어로 응답 생성
- `return response.content` — 텍스트 응답 반환

그리스어, 스페인어도 동일 패턴:

```python
@tool
def greek_agent(message: str) -> str:
    """Transfer to Greek customer support agent. Use when the customer speaks Greek."""
    response = llm.invoke(
        f"You're a Greek customer support agent. Respond in Greek.\nCustomer: {message}"
    )
    return response.content


@tool
def spanish_agent(message: str) -> str:
    """Transfer to Spanish customer support agent. Use when the customer speaks Spanish."""
    response = llm.invoke(
        f"You're a Spanish customer support agent. Respond in Spanish.\nCustomer: {message}"
    )
    return response.content
```

> **비교:** 18.1에서는 `make_agent()` 팩토리로 서브그래프를 만들고, `handoff_tool`에 `Command.PARENT`를 넣고... 복잡했다.
> 18.3에서는 `@tool` 하나면 끝.

### 슈퍼바이저 = LLM + 도구

```python
agent_tools = [korean_agent, greek_agent, spanish_agent]
llm_with_tools = llm.bind_tools(agent_tools)
```

3개의 에이전트 도구를 LLM에 바인딩. LLM이 "이 3개 도구를 쓸 수 있어"를 알게 된다.

```python
def supervisor(state: State):
    response = llm_with_tools.invoke(
        [
            {
                "role": "system",
                "content": "You are a customer support supervisor. "
                "Route the customer to the appropriate language agent using the tools.",
            }
        ]
        + state["messages"]
    )
    return {"messages": [response]}
```

슈퍼바이저 함수가 매우 단순해졌다:
1. 시스템 프롬프트 + 대화 기록을 LLM에 넘김
2. LLM이 알아서 적절한 도구(에이전트)를 선택하여 `tool_calls` 생성
3. 응답을 messages에 추가

**18.2에서 직접 짰던 라우팅 로직(`with_structured_output`, `Command`)이 전부 사라졌다!**
LLM의 tool calling 기능이 라우팅을 대신한다.

### 그래프 — 깔끔한 ReAct 구조

```python
graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("tools", ToolNode(tools=agent_tools))

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges("supervisor", tools_condition)
graph_builder.add_edge("tools", "supervisor")

graph = graph_builder.compile()
```

**Chapter 14.1의 ReAct 패턴과 완전히 동일한 구조!**

```
14.1: START → chatbot → [tool_calls?] → tools → chatbot → ...
18.3: START → supervisor → [tool_calls?] → tools → supervisor → ...
```

다른 점은 도구가 `get_weather` 같은 일반 도구가 아니라 **에이전트 함수**라는 것뿐.

### 실행 흐름 예시

**한국어 메시지:**

```
User: "비밀번호를 변경하고 싶습니다."
  → supervisor(LLM): 한국어네 → tool_calls: korean_agent(message="비밀번호를 변경하고 싶습니다.")
  → tools_condition: tool_calls 있으니 → tools 노드로
  → ToolNode: korean_agent() 실행 → "비밀번호 변경 방법은..." 반환
  → 다시 supervisor로: 도구 결과를 자연어로 정리
  → tools_condition: tool_calls 없으니 → END
```

---

## 3가지 아키텍처 비교표

| | Network (18.1) | Supervisor (18.2) | Supervisor+Tools (18.3) |
|--|---------|------------|------------------|
| **라우팅 방식** | 각 에이전트가 `handoff_tool`로 자율 전환 | 중앙 슈퍼바이저가 `with_structured_output`으로 결정 | LLM의 `tool_calls`가 자동 선택 |
| **에이전트 형태** | 서브그래프 (ReAct 루프) | 서브그래프 (응답만) | `@tool` 함수 |
| **에이전트 복잡도** | 높음 (handoff 로직 포함) | 낮음 (응답만) | 최소 (함수 하나) |
| **라우팅 코드** | `Command(graph=Command.PARENT)` | `Command(goto=)` + `SupervisorOutput` | 없음 (LLM이 알아서) |
| **상태 필드** | `current_agent`, `transfered_by` | + `reasoning` | `messages`만 |
| **에이전트 추가 시** | 모든 에이전트의 destinations 수정 + docstring 수정 | 슈퍼바이저의 Literal + 프롬프트 수정 | `@tool` 함수 추가 + 리스트에 추가 |
| **디버깅** | 어려움 (분산 결정) | 쉬움 (`reasoning` 추적) | 쉬움 (`tool_calls` 추적) |
| **적합한 경우** | 에이전트 소수, 자율성 필요 | 중규모, 제어 필요 | 대규모, 깔끔한 구조 |
| **비유** | 직원끼리 전화 돌리기 | 콜센터 팀장 배정 | 자동 콜센터 |

---

## 핵심 개념 정리표

| 섹션 | 핵심 개념 | 핵심 코드 | 비유 |
|------|----------|----------|------|
| **18.1** | P2P 핸드오프 | `Command(goto=..., graph=Command.PARENT)` | 동료끼리 전화 돌리기 |
| **18.1** | 에이전트 팩토리 | `make_agent(prompt, tools)` → 서브그래프 | 같은 틀로 찍어내는 공장 |
| **18.1** | 자기 전환 방어 | `if transfer_to == transfered_by` | 자기한테 전화 돌리기 금지 |
| **18.2** | 구조화 출력 | `llm.with_structured_output(SupervisorOutput)` | 지정된 양식으로만 답변 |
| **18.2** | Literal 제한 | `Literal["korean_agent", ..., "__end__"]` | 4지선다 (엉뚱한 답 차단) |
| **18.2** | 순환 그래프 | `agent → supervisor → agent → ...` | 팀장에게 보고 후 재배정 |
| **18.3** | 에이전트 = 도구 | `@tool def korean_agent(message)` | 버튼 하나로 호출 |
| **18.3** | ReAct 재활용 | `supervisor → tools_condition → ToolNode` | 14.1 패턴 그대로 |
| **공통** | MessagesState | 메시지 자동 누적 (append) | 채팅 기록 |
| **공통** | 서브그래프 | 에이전트 = 독립적인 작은 그래프 | 부서 = 독립 조직 |
