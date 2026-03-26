# Chapter 14: LangGraph 챗봇 구축

## 챕터 개요

이 챕터에서는 **LangGraph**를 활용하여 실용적인 AI 챗봇을 단계적으로 구축하는 방법을 학습한다. LangGraph는 LangChain 생태계의 핵심 라이브러리로, **상태 기반 그래프(Stateful Graph)** 패턴을 통해 복잡한 AI 에이전트 워크플로우를 설계할 수 있게 해준다.

챕터 전체를 관통하는 핵심 주제는 다음과 같다:

- **기본 챗봇 구조**: StateGraph와 MessagesState를 활용한 챗봇의 뼈대 구축
- **도구(Tool) 통합**: LLM이 외부 도구를 호출하고 결과를 활용하는 패턴
- **메모리(Memory)**: SQLite 기반 체크포인터를 통한 대화 상태 영속화
- **Human-in-the-loop**: 사람의 피드백을 워크플로우에 통합하는 인터럽트 패턴
- **타임 트래블(Time Travel)**: 상태 히스토리 탐색과 포크(fork)를 통한 분기 실행
- **개발 도구(DevTools)**: LangGraph Studio를 위한 프로덕션 구조 전환

각 섹션은 이전 섹션의 코드를 발전시키는 형태로 구성되어 있으며, 최종적으로 도구 호출, 메모리, 사람의 개입, 상태 관리가 모두 포함된 완전한 에이전트 시스템을 완성한다.

---

## 14.0 LangGraph 챗봇 (기본 구조)

### 주제 및 목표

이 섹션의 목표는 LangGraph의 가장 기본적인 챗봇 구조를 만드는 것이다. 이전 챕터에서 학습했던 `StateGraph`, `Command`, `TypedDict` 기반의 복잡한 라우팅 그래프를 걷어내고, **메시지 기반의 심플한 챗봇**으로 완전히 재구성한다.

### 핵심 개념 설명

#### MessagesState란?

LangGraph는 `MessagesState`라는 사전 정의된 상태 클래스를 제공한다. 이것은 챗봇 개발에 최적화된 상태 관리 도구로, 내부적으로 `messages`라는 리스트 필드를 가지고 있다. 이 리스트에는 `HumanMessage`, `AIMessage`, `ToolMessage` 등 LangChain의 다양한 메시지 타입이 자동으로 축적된다.

기존에 `TypedDict`로 직접 상태를 정의하던 방식과의 가장 큰 차이점은, `MessagesState`가 메시지의 **추가(append)** 동작을 기본으로 지원한다는 것이다. 즉, 노드가 `{"messages": [new_message]}`를 반환하면 기존 메시지 리스트에 새 메시지가 추가된다.

#### init_chat_model

`langchain.chat_models`의 `init_chat_model` 함수는 다양한 LLM 프로바이더를 통합된 인터페이스로 초기화할 수 있게 해준다. `"openai:gpt-4o-mini"`와 같이 `프로바이더:모델명` 형식의 문자열을 전달하면 된다.

### 코드 분석

#### 1단계: 임포트 및 LLM 초기화

```python
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph.message import MessagesState

llm = init_chat_model("openai:gpt-4o-mini")
```

이전 코드에서 사용하던 `TypedDict`와 `Command`를 제거하고, 대신 `MessagesState`와 `init_chat_model`을 임포트한다. `init_chat_model`은 LangChain의 범용 채팅 모델 초기화 함수로, 프로바이더와 모델명을 콜론으로 구분하여 지정한다.

#### 2단계: 상태 정의

```python
class State(MessagesState):
    custom_stuff: str

graph_builder = StateGraph(State)
```

`MessagesState`를 상속받아 자체 `State` 클래스를 정의한다. `MessagesState`에는 이미 `messages` 필드가 포함되어 있으므로, 필요에 따라 추가 필드(여기서는 `custom_stuff`)만 선언하면 된다. 이 패턴은 LangGraph 챗봇의 표준적인 상태 정의 방식이다.

#### 3단계: 챗봇 노드 정의

```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

핵심이 되는 챗봇 노드 함수이다. 동작 원리는 간단하다:
1. 현재 상태에서 `messages` 리스트를 꺼낸다.
2. LLM에 전체 메시지 히스토리를 전달하여 응답을 생성한다.
3. 생성된 응답을 `messages` 리스트에 추가하여 반환한다.

`MessagesState`의 리듀서(reducer) 로직 덕분에, 반환값의 `messages` 리스트는 기존 상태의 메시지에 **추가**된다.

#### 4단계: 그래프 구성 및 실행

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

그래프 구조는 매우 단순하다:
- `START` -> `chatbot` -> `END`

`graph.invoke()`에 초기 메시지를 딕셔너리 형태로 전달한다. `{"role": "user", "content": "how are you?"}`는 LangChain이 자동으로 `HumanMessage` 객체로 변환한다.

실행 결과로 `HumanMessage`와 `AIMessage`가 포함된 상태 딕셔너리가 반환된다.

### 실습 포인트

- `MessagesState`를 상속받지 않고 `TypedDict`로 직접 상태를 정의하면 어떤 차이가 생기는지 실험해 보라.
- `init_chat_model`의 프로바이더를 `"anthropic:claude-sonnet-4-20250514"` 등으로 바꿔서 다른 LLM으로 동일한 그래프를 실행해 보라.
- `custom_stuff` 필드를 활용하여 시스템 프롬프트를 동적으로 주입하는 방법을 고안해 보라.

---

## 14.1 도구 노드 (Tool Nodes)

### 주제 및 목표

이 섹션에서는 챗봇에 **외부 도구(Tool)**를 연결한다. LLM이 사용자 요청에 따라 도구를 호출하고, 도구의 실행 결과를 받아 최종 응답을 생성하는 **ReAct(Reasoning + Acting) 패턴**을 구현한다.

### 핵심 개념 설명

#### Tool Calling 메커니즘

OpenAI의 gpt-4o-mini 같은 최신 LLM은 **function calling(도구 호출)** 기능을 지원한다. LLM에게 사용 가능한 도구 목록을 알려주면, LLM은 사용자의 질문에 따라 적절한 도구를 선택하고 인자를 구성하여 호출 요청을 생성한다. 이때 LLM이 직접 도구를 실행하는 것이 아니라, "이 도구를 이런 인자로 호출해 달라"는 **의도(intent)**를 표현하는 것이다.

#### ToolNode와 tools_condition

LangGraph의 `langgraph.prebuilt` 모듈은 두 가지 핵심 유틸리티를 제공한다:

- **`ToolNode`**: 도구 실행을 담당하는 사전 구축된 노드. LLM이 반환한 `tool_calls`를 감지하여 해당 도구를 실행하고, 결과를 `ToolMessage`로 반환한다.
- **`tools_condition`**: 조건부 라우팅 함수. LLM 응답에 `tool_calls`가 있으면 `"tools"` 노드로, 없으면 `END`로 라우팅한다.

#### 조건부 엣지 (Conditional Edges)

`add_conditional_edges`는 이전 노드의 출력에 따라 다음 노드를 동적으로 결정하는 엣지를 추가한다. `tools_condition`과 함께 사용하면, LLM이 도구 호출을 요청했을 때만 도구 노드로 이동하고, 그렇지 않으면 대화를 종료하는 흐름을 자연스럽게 구현할 수 있다.

### 코드 분석

#### 1단계: 새로운 임포트

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
```

`ToolNode`와 `tools_condition`은 LangGraph의 프리빌트 모듈에서, `@tool` 데코레이터는 `langchain_core.tools`에서 가져온다.

#### 2단계: 도구 정의

```python
@tool
def get_weather(city: str):
    """Gets weather in city"""
    return f"The weather in {city} is sunny."
```

`@tool` 데코레이터를 사용하여 일반 Python 함수를 LangChain 도구로 변환한다. 함수의 **docstring**이 도구의 설명으로 사용되고, 함수의 **파라미터 타입 힌트**가 도구의 입력 스키마로 자동 변환된다. LLM은 이 정보를 기반으로 도구를 호출할지, 어떤 인자를 넘길지 결정한다.

#### 3단계: LLM에 도구 바인딩

```python
llm_with_tools = llm.bind_tools(tools=[get_weather])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

`bind_tools()` 메서드를 통해 LLM에게 사용 가능한 도구 목록을 알려준다. 이제 LLM은 응답 시 텍스트 대신 도구 호출 요청(`tool_calls`)을 반환할 수 있다. 챗봇 노드에서는 기존 `llm` 대신 `llm_with_tools`를 사용한다.

#### 4단계: 그래프 재구성

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

그래프 구조가 크게 변경된다:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

핵심 변경사항:
1. `ToolNode`를 `"tools"` 노드로 추가한다.
2. `chatbot`에서 `END`로의 직접 연결 대신, `add_conditional_edges`로 **조건부 라우팅**을 설정한다.
3. `"tools"` 노드에서 `"chatbot"`으로의 엣지를 추가하여 **루프**를 형성한다.

이 구조 덕분에 LLM은 필요한 만큼 도구를 반복 호출할 수 있다. 도구 결과를 받은 후 다시 `chatbot` 노드로 돌아가 최종 응답을 생성하거나, 추가 도구를 호출할 수 있다.

#### 5단계: 실행 흐름 확인

```python
graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "what is the weather in machupichu"},
        ]
    }
)
```

실행 결과에서 메시지 흐름을 살펴보면:

1. `HumanMessage`: "what is the weather in machupichu"
2. `AIMessage` (tool_calls 포함): `get_weather(city="machupichu")` 호출 요청
3. `ToolMessage`: "The weather in machupichu is sunny."
4. `AIMessage`: "The weather in Machu Picchu is sunny." (최종 응답)

LLM이 도구 호출 결과를 자연어로 정리하여 최종 사용자에게 전달하는 것을 확인할 수 있다.

### 실습 포인트

- 여러 개의 도구를 동시에 등록하고, LLM이 상황에 맞게 적절한 도구를 선택하는지 확인해 보라.
- 도구의 docstring을 변경하면 LLM의 도구 선택 행동이 어떻게 달라지는지 실험해 보라.
- `tools_condition`을 커스텀 조건 함수로 교체하여 더 복잡한 라우팅 로직을 구현해 보라.

---

## 14.2 메모리 (Memory)

### 주제 및 목표

이 섹션에서는 챗봇에 **영속적인 메모리**를 추가한다. 기본 LangGraph 그래프는 `invoke()` 호출이 끝나면 상태가 사라지지만, **체크포인터(Checkpointer)**를 사용하면 대화 상태를 데이터베이스에 저장하여 이후 대화에서도 이전 맥락을 유지할 수 있다.

### 핵심 개념 설명

#### 체크포인터 (Checkpointer)

LangGraph의 체크포인터는 그래프 실행의 **각 단계(step)**마다 상태를 자동으로 저장하는 메커니즘이다. 이를 통해:

- **대화 지속성**: 동일한 `thread_id`로 여러 번 `invoke()`를 호출하면 이전 대화 맥락이 유지된다.
- **상태 히스토리**: 그래프 실행의 모든 중간 단계를 나중에 조회할 수 있다.
- **에러 복구**: 실행 중 오류가 발생하더라도 마지막 체크포인트에서 재개할 수 있다.

#### SqliteSaver

`SqliteSaver`는 SQLite 데이터베이스를 백엔드로 사용하는 체크포인터 구현체이다. 프로덕션 환경에서는 PostgreSQL 기반의 체크포인터를 사용하는 것이 권장되지만, 개발 및 학습 목적으로는 SQLite가 간편하다.

#### thread_id와 config

체크포인터를 사용할 때는 반드시 `config`에 `thread_id`를 지정해야 한다. `thread_id`는 대화 세션을 구분하는 식별자로, 같은 `thread_id`를 사용하면 동일한 대화를 이어가고, 다른 `thread_id`를 사용하면 새로운 대화가 시작된다.

#### 비동기 스트리밍 (Async Streaming)

이 섹션에서는 `graph.invoke()` 대신 `graph.astream()`을 사용한 비동기 스트리밍 패턴도 소개된다. `stream_mode="updates"`를 설정하면 각 노드의 실행 결과가 발생할 때마다 이벤트로 전달되어, 실시간으로 그래프 실행 과정을 모니터링할 수 있다.

### 코드 분석

#### 1단계: SQLite 연결 및 체크포인터 설정

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect(
    "memory.db",
    check_same_thread=False,
)
```

SQLite 데이터베이스 연결을 생성한다. `check_same_thread=False`는 멀티스레드 환경에서 SQLite를 안전하게 사용하기 위한 설정이다. Jupyter 노트북의 비동기 이벤트 루프에서 동작하려면 이 옵션이 필요하다.

또한 `pyproject.toml`에 `aiosqlite` 의존성이 추가되었다:

```toml
dependencies = [
    "aiosqlite>=0.21.0",
    ...
]
```

이는 `SqliteSaver`의 비동기 연산을 지원하기 위한 패키지이다.

#### 2단계: 그래프에 체크포인터 연결

```python
graph = graph_builder.compile(
    checkpointer=SqliteSaver(conn),
)
```

`compile()` 메서드에 `checkpointer` 파라미터를 전달하는 것만으로 메모리 기능이 활성화된다. 이후 그래프의 모든 실행은 자동으로 SQLite에 상태가 저장된다.

#### 3단계: 스트리밍으로 실행

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

`astream()`은 비동기 제너레이터로, 각 노드의 실행 결과를 실시간으로 스트리밍한다. `stream_mode="updates"`를 사용하면 전체 상태가 아닌 **변경된 부분만** 이벤트로 전달되어 효율적이다.

주석 처리된 `config` 부분을 활성화하면 `thread_id`를 지정하여 대화를 이어갈 수 있다:

```python
config={
    "configurable": {
        "thread_id": "2",
    },
},
```

#### 4단계: 상태 히스토리 조회

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

`get_state_history()`는 특정 스레드의 모든 상태 스냅샷을 시간 역순으로 반환한다. 각 스냅샷에는 해당 시점의 상태값과 다음에 실행될 노드(`next`) 정보가 포함되어 있다. 이 기능은 다음 섹션의 타임 트래블 기능의 기초가 된다.

### 실습 포인트

- 동일한 `thread_id`로 여러 번 `invoke()`를 호출하여 대화가 실제로 이어지는지 확인해 보라.
- 다른 `thread_id`로 호출하면 완전히 새로운 대화가 시작되는지 확인해 보라.
- `get_state_history()`의 결과를 분석하여 각 노드 실행 후 상태가 어떻게 변화하는지 추적해 보라.
- `stream_mode`를 `"values"`로 변경하면 출력이 어떻게 달라지는지 비교해 보라.

---

## 14.3 Human-in-the-loop (사람 개입)

### 주제 및 목표

이 섹션에서는 LangGraph의 가장 강력한 기능 중 하나인 **Human-in-the-loop** 패턴을 구현한다. AI가 자동으로 처리하는 것이 아니라, 중간에 **사람의 판단이나 피드백**을 받아 워크플로우를 계속 진행하는 패턴이다.

이를 위해 LangGraph의 `interrupt` 함수와 `Command` 클래스를 활용한다.

### 핵심 개념 설명

#### interrupt 함수

`interrupt()`는 그래프의 실행을 **일시 중단**하는 함수이다. 도구 노드 내부에서 호출되면, 그래프 실행이 멈추고 현재 상태가 체크포인터에 저장된다. 개발자는 중단 시점에 사용자에게 질문을 전달하고, 사용자의 응답을 받은 후 `Command(resume=...)`로 실행을 재개할 수 있다.

```python
feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
return feedback
```

`interrupt()`에 전달한 값은 중단 시 사용자에게 보여줄 메시지가 되고, `Command(resume=...)`로 전달한 값이 `interrupt()`의 반환값이 된다.

#### Command 클래스

`Command`는 중단된 그래프를 재개하기 위한 명령 객체이다. `resume` 파라미터에 사용자의 응답을 담아 `graph.invoke()`에 전달하면, 그래프가 중단된 지점에서 `interrupt()`의 반환값으로 해당 응답이 들어가며 실행이 계속된다.

#### 상태 스냅샷과 next

`graph.get_state(config)`를 통해 현재 그래프의 상태 스냅샷을 조회할 수 있다. 스냅샷의 `next` 속성은 다음에 실행될 노드의 이름을 튜플로 반환한다.
- `('tools',)`: 도구 노드에서 인터럽트되어 대기 중
- `()`: 그래프 실행이 완료됨

### 코드 분석

#### 1단계: interrupt와 Command 임포트

```python
from langgraph.types import interrupt, Command
```

LangGraph의 `types` 모듈에서 두 핵심 클래스를 임포트한다.

#### 2단계: 사람 피드백 도구 정의

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

이 도구의 동작 방식이 핵심이다:
1. LLM이 시(poem)를 생성하여 이 도구를 호출한다.
2. `interrupt()`가 호출되면 그래프 실행이 중단된다.
3. 사용자가 피드백을 제공하면 `interrupt()`가 해당 피드백을 반환한다.
4. 피드백이 `ToolMessage`로 LLM에게 전달된다.

#### 3단계: 시스템 프롬프트가 포함된 챗봇 노드

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

시스템 프롬프트에서 LLM에게 명확한 지시를 내린다:
- 시를 만드는 전문가 역할
- 반드시 `get_human_feedback` 도구를 사용하여 피드백을 받을 것
- 긍정적인 피드백을 받은 후에만 최종 시를 반환할 것

이 프롬프트 설계가 Human-in-the-loop 패턴의 성공 여부를 좌우한다.

#### 4단계: 첫 번째 실행 (중단까지)

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

실행하면 LLM이 시를 생성하고 `get_human_feedback` 도구를 호출한다. `interrupt()`에 의해 그래프가 중단되고, 현재까지의 대화 상태가 체크포인터에 저장된다.

#### 5단계: 상태 확인

```python
snapshot = graph.get_state(config)
snapshot.next  # ('tools',)
```

`next`가 `('tools',)`를 반환하면 도구 노드에서 인터럽트되어 사용자의 응답을 기다리고 있다는 의미이다.

#### 6단계: 피드백 제공 및 재개

```python
response = Command(resume="It looks great!")

result = graph.invoke(
    response,
    config=config,
)
for message in result["messages"]:
    message.pretty_print()
```

`Command(resume="It looks great!")`로 긍정적 피드백을 전달하면:
1. `interrupt()`가 `"It looks great!"`를 반환한다.
2. 이 값이 `ToolMessage`로 LLM에게 전달된다.
3. LLM은 긍정적 피드백을 확인하고 최종 시를 반환한다.

실제 출력에서 전체 대화 흐름을 확인할 수 있다:
- 첫 번째 시 생성 -> 피드백 "It is too long! Make shorter." -> 짧은 버전 생성 -> 피드백 "It looks great!" -> 최종 시 반환

#### 7단계: 완료 확인

```python
snapshot = graph.get_state(config)
snapshot.next  # ()
```

빈 튜플 `()`는 그래프 실행이 완료되었음을 나타낸다.

### 실습 포인트

- 부정적인 피드백을 여러 번 연속으로 제공하면 LLM이 어떻게 반응하는지 관찰해 보라.
- `interrupt()` 없이 도구가 자동으로 피드백을 반환하는 버전과 비교하여 Human-in-the-loop의 차이를 체감해 보라.
- 여러 개의 인터럽트 포인트를 가진 복잡한 워크플로우를 설계해 보라 (예: 검토 -> 승인 -> 배포 파이프라인).
- `Command(resume=...)`에 구조화된 데이터(딕셔너리)를 전달하는 방법을 실험해 보라.

---

## 14.4 타임 트래블 (Time Travel)

### 주제 및 목표

이 섹션에서는 LangGraph의 체크포인터가 저장한 **상태 히스토리**를 활용하여, 과거의 특정 시점으로 돌아가 그래프 실행을 **분기(fork)**하는 타임 트래블 기능을 학습한다.

이 기능은 디버깅, A/B 테스팅, 사용자 경험 롤백 등 다양한 실무 시나리오에서 활용된다.

### 핵심 개념 설명

#### 상태 히스토리

체크포인터가 활성화된 그래프는 **매 노드 실행마다** 상태 스냅샷을 저장한다. `get_state_history(config)`를 호출하면 특정 스레드의 모든 스냅샷을 시간 역순으로 조회할 수 있다. 각 스냅샷에는:

- `values`: 해당 시점의 전체 상태 (메시지 리스트 등)
- `next`: 다음에 실행될 노드
- `config`: 해당 스냅샷을 식별하는 설정 (checkpoint_id 포함)

#### 상태 포크 (State Fork)

`graph.update_state()`를 사용하면 과거의 특정 체크포인트를 기반으로 **새로운 분기**를 생성할 수 있다. 예를 들어, 사용자가 "Valencia에 산다"고 말한 시점으로 돌아가 "Zagreb에 산다"로 변경하면, 새로운 분기에서 LLM이 Zagreb 기준의 응답을 생성한다.

#### checkpoint_id

각 상태 스냅샷은 고유한 `checkpoint_id`를 가진다. 이 ID를 `config`에 포함시켜 `graph.invoke()`를 호출하면, 해당 체크포인트에서 그래프 실행을 재개할 수 있다.

### 코드 분석

이 섹션에서는 이전의 도구 호출과 Human-in-the-loop을 모두 제거하고, 간단한 챗봇 구조로 돌아간다. 이는 타임 트래블 개념 자체에 집중하기 위함이다.

#### 1단계: 단순화된 챗봇

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

도구 노드 없이, 메모리만 활성화된 단순한 챗봇이다. `State`는 `MessagesState`를 그대로 상속하며 추가 필드가 없다.

#### 2단계: 대화 실행

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

사용자가 "유럽에 살고, 도시는 Valencia"라고 말한다. LLM은 이에 대한 응답을 생성한다.

#### 3단계: 상태 히스토리 탐색

```python
state_history = graph.get_state_history(config)

for state_snapshot in list(state_history):
    print(state_snapshot.next)
    print(state_snapshot.values["messages"])
    print("=========\n")
```

모든 상태 스냅샷을 순회하며, 각 시점의 `next` 노드와 메시지 내용을 출력한다. 이를 통해 그래프 실행의 전체 흐름을 시간순으로 파악할 수 있다.

#### 4단계: 포크할 지점 선택

```python
state_history = graph.get_state_history(config)
to_fork = list(state_history)[-5]
to_fork.config
```

상태 히스토리에서 특정 인덱스의 스냅샷을 선택한다. 이 스냅샷의 `config`에는 해당 시점의 `checkpoint_id`가 포함되어 있다.

#### 5단계: 상태 수정 (포크 생성)

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

`update_state()`의 핵심 포인트:
- **첫 번째 인자**: 포크할 시점의 `config` (checkpoint_id 포함)
- **두 번째 인자**: 수정할 상태 값
- **메시지 ID**: 동일한 ID를 지정하면 기존 메시지를 **교체**하고, 새 ID를 사용하면 메시지를 **추가**한다.

여기서는 "Valencia"를 "Zagreb"로 교체하여 새로운 분기를 생성한다.

#### 6단계: 포크된 상태에서 실행 재개

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

`graph.invoke(None, config)`에서:
- 첫 번째 인자가 `None`이면 새로운 입력 없이 기존 상태에서 실행을 재개한다.
- `config`에 특정 `checkpoint_id`를 지정하면 해당 체크포인트에서 시작한다.

이렇게 하면 LLM이 "Zagreb에 산다"는 수정된 메시지를 기반으로 새로운 응답을 생성한다.

### 실습 포인트

- 여러 번 대화를 진행한 후, 중간 지점으로 돌아가 다른 질문을 하면 어떻게 되는지 실험해 보라.
- `update_state()`에서 메시지 ID를 다르게 설정하여, 교체 대신 추가가 되는 경우를 확인해 보라.
- 같은 체크포인트에서 여러 개의 서로 다른 포크를 생성하여 A/B 테스팅 시나리오를 구현해 보라.
- `get_state_history()`로 포크된 상태가 히스토리에 어떻게 기록되는지 분석해 보라.

---

## 14.5 개발 도구 (DevTools)

### 주제 및 목표

이 섹션에서는 Jupyter 노트북 기반의 프로토타입을 **프로덕션 구조**로 전환하고, **LangGraph Studio** (LangGraph DevTools)를 활용하기 위한 설정을 수행한다.

LangGraph Studio는 그래프의 시각적 디버깅, 상태 추적, 타임 트래블 등을 GUI로 제공하는 개발 도구이다.

### 핵심 개념 설명

#### Jupyter에서 Python 스크립트로

개발 과정에서 Jupyter 노트북은 빠른 프로토타이핑에 유용하지만, 실제 배포와 DevTools 연동을 위해서는 표준 Python 스크립트(`.py`)로 전환해야 한다. 이 과정에서:

- 노트북의 셀 단위 코드를 하나의 스크립트로 통합
- Human-in-the-loop, 도구 호출 등 이전 섹션의 모든 기능을 다시 결합
- `graph` 객체를 모듈 레벨에서 export하여 외부에서 참조 가능하게 만듦

#### langgraph.json 설정 파일

`langgraph.json`은 LangGraph Studio가 프로젝트를 인식하기 위한 설정 파일이다. 이 파일에서 의존성, 환경변수, 그래프 엔트리포인트를 정의한다.

### 코드 분석

#### 1단계: langgraph.json 설정

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

각 필드의 의미:
- **`dependencies`**: 프로젝트에 필요한 패키지와 모듈. `langchain_openai`는 OpenAI 연동 패키지이고, `./main.py`는 그래프가 정의된 스크립트 파일이다.
- **`env`**: 환경변수 파일 경로. OpenAI API 키 등의 비밀 정보를 `.env` 파일에서 로드한다.
- **`graphs`**: DevTools에 노출할 그래프 목록. `"mr_poet"`이라는 이름으로 `main.py` 파일의 `graph` 변수를 등록한다.

#### 2단계: main.py - 통합된 프로덕션 코드

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

모든 필요한 모듈을 임포트한다. 이전 섹션들에서 학습한 모든 기능(도구, 체크포인터, 인터럽트)이 하나의 파일에 통합된다.

#### 3단계: 도구 정의 (Human-in-the-loop 포함)

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

14.3 섹션의 Human-in-the-loop 도구가 약간 개선되었다. `interrupt()`에 딕셔너리를 전달하고, 응답도 딕셔너리 형태(`response["feedback"]`)로 받아 구조화된 데이터 교환이 가능해졌다.

#### 4단계: LLM 및 상태 정의

```python
llm = init_chat_model("openai:gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

class State(MessagesState):
    pass
```

#### 5단계: 챗봇 노드 (시스템 프롬프트 포함)

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

14.3 섹션과 유사한 시스템 프롬프트를 사용하며, 타입 힌트(`-> State`)가 추가되어 코드의 명확성이 향상되었다.

#### 6단계: 그래프 구성

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

최종 그래프 구조:

```
START -> chatbot -> [tools_condition] -> tools -> chatbot -> ... -> END
```

주목할 점:
- `compile(name="mr_poet")`으로 그래프에 이름을 부여한다. 이 이름은 `langgraph.json`의 `graphs` 필드와 연동된다.
- `memory.db` 파일을 통해 SQLite 기반 체크포인터가 설정된다.
- 모듈 레벨에서 `graph` 변수가 정의되어, `./main.py:graph`로 외부에서 참조 가능하다.

### 실습 포인트

- `langgraph dev`(또는 `langgraph up`) 명령어로 LangGraph Studio를 실행해 보라.
- LangGraph Studio에서 그래프 구조를 시각적으로 확인하고, 각 노드의 실행 과정을 추적해 보라.
- Studio의 타임 트래블 기능을 사용하여 과거 상태로 돌아가고, 상태를 수정하여 새로운 분기를 만들어 보라.
- `langgraph.json`에 여러 개의 그래프를 등록하여 동시에 관리하는 방법을 실험해 보라.

---

## 챕터 핵심 정리

### 1. LangGraph 챗봇의 기본 구조
- `MessagesState`를 상속한 상태 클래스로 메시지 기반 대화를 관리한다.
- `StateGraph`로 노드와 엣지를 정의하고, `compile()`로 실행 가능한 그래프를 생성한다.
- `init_chat_model()`을 통해 다양한 LLM 프로바이더를 통합된 인터페이스로 사용한다.

### 2. 도구 통합 패턴
- `@tool` 데코레이터로 Python 함수를 LangChain 도구로 변환한다.
- `llm.bind_tools()`로 LLM에 도구를 바인딩하고, `ToolNode`와 `tools_condition`으로 도구 실행 흐름을 자동화한다.
- 조건부 엣지(`add_conditional_edges`)를 통해 LLM의 도구 호출 여부에 따라 동적으로 라우팅한다.

### 3. 메모리와 체크포인터
- `SqliteSaver`를 `compile(checkpointer=...)`에 전달하여 상태 영속화를 활성화한다.
- `thread_id`로 대화 세션을 구분하며, 같은 ID를 사용하면 대화가 이어진다.
- `astream(stream_mode="updates")`으로 실시간 스트리밍이 가능하다.

### 4. Human-in-the-loop
- `interrupt()` 함수로 그래프 실행을 일시 중단하고 사용자 입력을 대기한다.
- `Command(resume=...)` 으로 사용자의 응답을 전달하여 실행을 재개한다.
- `get_state(config).next`로 현재 중단 상태를 확인할 수 있다.

### 5. 타임 트래블
- `get_state_history(config)`로 모든 상태 스냅샷을 조회한다.
- `update_state(checkpoint_config, new_values)`로 과거 상태를 수정하여 포크를 생성한다.
- `graph.invoke(None, checkpoint_config)`로 특정 체크포인트에서 실행을 재개한다.
- 메시지 ID를 동일하게 지정하면 교체, 다르게 지정하면 추가가 된다.

### 6. DevTools 통합
- `langgraph.json`으로 프로젝트 설정을 정의한다.
- Jupyter 노트북에서 Python 스크립트로 전환하여 프로덕션 구조를 갖춘다.
- LangGraph Studio를 통해 시각적 디버깅, 상태 추적, 타임 트래블을 GUI로 수행한다.

---

## 실습 과제

### 과제 1: 다중 도구 챗봇 (기본)

날씨, 환율, 뉴스 검색 등 3개 이상의 도구를 가진 챗봇을 구현하라. LLM이 사용자의 질문에 따라 적절한 도구를 선택하고, 필요시 여러 도구를 순차적으로 호출하여 최종 응답을 생성하도록 하라.

**요구사항:**
- 최소 3개의 `@tool` 함수 정의
- `ToolNode`와 `tools_condition`을 활용한 그래프 구성
- 다중 도구 호출이 필요한 질문 시나리오 테스트

### 과제 2: 승인 워크플로우 (중급)

문서 작성 -> 검토 -> 승인의 3단계 워크플로우를 구현하라. 각 단계에서 Human-in-the-loop을 통해 사람의 승인을 받아야 다음 단계로 넘어간다.

**요구사항:**
- `interrupt()`를 활용한 다중 승인 포인트
- 각 단계에서 거절 시 이전 단계로 돌아가는 로직
- 체크포인터를 통한 상태 영속화

### 과제 3: 타임 트래블 디버거 (고급)

대화 히스토리를 탐색하고, 특정 시점으로 돌아가 다른 입력으로 분기를 생성하는 인터랙티브 디버거를 구현하라.

**요구사항:**
- `get_state_history()`로 전체 히스토리를 시각적으로 표시
- 사용자가 특정 체크포인트를 선택하여 상태를 수정할 수 있는 인터페이스
- `update_state()`를 활용한 포크 생성 및 실행 재개
- 원본과 포크된 분기의 결과를 비교하는 기능

### 과제 4: DevTools 배포 (고급)

14.5에서 만든 `main.py`를 확장하여, 여러 개의 그래프를 하나의 프로젝트에서 관리하는 구조를 만들어라.

**요구사항:**
- 2개 이상의 서로 다른 그래프를 `langgraph.json`에 등록
- 각 그래프가 서로 다른 도구와 시스템 프롬프트를 사용
- LangGraph Studio에서 두 그래프를 모두 확인하고 실행
- `.env` 파일을 통한 환경변수 관리
