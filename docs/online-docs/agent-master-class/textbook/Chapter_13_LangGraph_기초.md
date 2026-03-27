# Chapter 13: LangGraph 기초 (LangGraph Fundamentals)

## 챕터 개요

이 챕터에서는 LangChain 생태계의 핵심 프레임워크인 **LangGraph**의 기초를 처음부터 단계적으로 학습한다. LangGraph는 LLM 기반 애플리케이션을 **그래프(Graph)** 구조로 설계하고 실행할 수 있게 해주는 프레임워크로, 복잡한 AI 에이전트 워크플로우를 직관적으로 구성할 수 있다.

이 챕터를 통해 다음을 배우게 된다:

- LangGraph 프로젝트의 초기 설정 및 환경 구성
- 그래프(Graph)의 기본 구조: 노드(Node)와 엣지(Edge)
- 그래프 상태(State) 관리와 노드 간 데이터 전달
- 다중 스키마(Multiple Schemas)를 활용한 입출력 분리
- Reducer 함수를 통한 상태 병합 전략
- 노드 캐싱(Node Caching)을 활용한 성능 최적화
- 조건부 엣지(Conditional Edges)를 통한 동적 흐름 제어
- Send API를 활용한 동적 병렬 처리
- Command 객체를 활용한 노드 내부 라우팅

### 프로젝트 환경

| 항목 | 버전/내용 |
|------|-----------|
| Python | >= 3.13 |
| LangGraph | >= 0.6.6 |
| LangChain | >= 0.3.27 (OpenAI 포함) |
| 개발 도구 | Jupyter Notebook (ipykernel) |
| 패키지 관리 | uv (pyproject.toml 기반) |

---

## 13.0 Introduction - 프로젝트 초기 설정

### 주제 및 목표
LangGraph 학습을 위한 새로운 Python 프로젝트를 생성하고, 필요한 의존성을 설치한다.

### 핵심 개념 설명

LangGraph 프로젝트를 시작하기 위해 `hello-langgraph`라는 새로운 프로젝트 디렉토리를 구성한다. 이 프로젝트는 **uv** 패키지 매니저를 사용하며, Jupyter Notebook 환경에서 실습을 진행한다.

#### 프로젝트 의존성 (`pyproject.toml`)

```toml
[project]
name = "hello-langgraph"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf>=0.8",
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.6",
    "langgraph-checkpoint-sqlite>=2.0.11",
    "langgraph-cli[inmem]>=0.4.0",
    "python-dotenv>=1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

각 의존성의 역할:

| 패키지 | 역할 |
|--------|------|
| `langgraph` | 그래프 기반 워크플로우 프레임워크 (핵심) |
| `langchain[openai]` | LangChain과 OpenAI 통합 |
| `grandalf` | 그래프 시각화 지원 |
| `langgraph-checkpoint-sqlite` | 상태 체크포인트 저장 (SQLite) |
| `langgraph-cli[inmem]` | LangGraph CLI 도구 (인메모리 모드) |
| `python-dotenv` | 환경변수 관리 (.env 파일) |
| `ipykernel` | Jupyter Notebook 커널 (개발 전용) |

### 실습 포인트
- `uv`를 사용하여 프로젝트를 초기화하고 의존성을 설치해 본다.
- `.gitignore`를 통해 가상환경, 캐시 파일 등을 버전 관리에서 제외한다.
- Jupyter Notebook 환경이 정상적으로 동작하는지 확인한다.

---

## 13.1 Your First Graph - 첫 번째 그래프 만들기

### 주제 및 목표
LangGraph의 가장 기본적인 구성 요소인 **StateGraph**, **노드(Node)**, **엣지(Edge)**를 이해하고 첫 번째 그래프를 구성한다.

### 핵심 개념 설명

LangGraph에서 그래프는 세 가지 핵심 요소로 구성된다:

1. **State (상태)**: 그래프 전체에서 공유되는 데이터 구조. `TypedDict`를 사용하여 정의한다.
2. **Node (노드)**: 그래프 내에서 실행되는 개별 함수. 각 노드는 상태를 입력으로 받는다.
3. **Edge (엣지)**: 노드 간의 연결. 실행 순서를 결정한다.

또한 LangGraph는 두 가지 특수 노드를 제공한다:
- **`START`**: 그래프의 시작점. 어떤 노드가 가장 먼저 실행될지를 나타낸다.
- **`END`**: 그래프의 종료점. 여기에 연결된 노드가 마지막으로 실행된다.

### 코드 분석

#### 1단계: 임포트 및 상태 정의

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    hello: str

graph_builder = StateGraph(State)
```

- `StateGraph`는 상태 기반 그래프를 생성하는 핵심 클래스이다.
- `State`는 `TypedDict`를 상속하여 그래프에서 사용할 상태의 스키마를 정의한다.
- `graph_builder`는 `StateGraph`의 인스턴스로, 이것을 통해 노드와 엣지를 추가한다.

#### 2단계: 노드 함수 정의

```python
def node_one(state: State):
    print("node_one")

def node_two(state: State):
    print("node_two")

def node_three(state: State):
    print("node_three")
```

- 각 노드 함수는 반드시 `state`를 매개변수로 받는다.
- 이 단계에서는 아직 상태를 수정하지 않고, 단순히 실행 확인용 출력만 한다.

#### 3단계: 그래프 구성

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

이 코드는 다음과 같은 선형 그래프를 만든다:

```
START -> node_one -> node_two -> node_three -> END
```

- `add_node(이름, 함수)`: 그래프에 노드를 등록한다.
- `add_edge(출발, 도착)`: 두 노드를 연결하는 엣지를 추가한다.

### 실습 포인트
- `START`와 `END`를 생략하면 어떤 에러가 발생하는지 확인해 보라.
- 노드의 순서를 변경하여 실행 흐름이 어떻게 달라지는지 관찰하라.
- `add_node`에서 첫 번째 인자(문자열 이름)를 생략하면 함수 이름이 자동으로 사용되는지 테스트해 보라.

---

## 13.2 Graph State - 그래프 상태 관리

### 주제 및 목표
노드가 상태를 **읽고 수정**하는 방법을 이해하고, 상태가 노드를 거치면서 어떻게 변화하는지 추적한다.

### 핵심 개념 설명

LangGraph에서 상태(State)는 그래프 실행의 핵심이다. 각 노드는:

1. 현재 상태를 **입력**으로 받는다.
2. 딕셔너리를 **반환**하여 상태를 업데이트한다.
3. 반환된 값은 기존 상태에 **덮어쓰기(overwrite)** 된다 (기본 동작).

중요한 점은 노드가 반환하지 않은 키의 값은 그대로 유지된다는 것이다.

### 코드 분석

#### 상태 정의 확장

```python
class State(TypedDict):
    hello: str
    a: bool

graph_builder = StateGraph(State)
```

이제 상태에 `hello`(문자열)와 `a`(불리언) 두 개의 필드가 있다.

#### 노드에서 상태 읽기와 수정

```python
def node_one(state: State):
    print("node_one", state)
    return {
        "hello": "from node one.",
        "a": True,
    }

def node_two(state: State):
    print("node_two", state)
    return {"hello": "from node two."}

def node_three(state: State):
    print("node_three", state)
    return {"hello": "from node three."}
```

핵심 포인트:
- `node_one`은 `hello`와 `a` 두 필드 모두를 업데이트한다.
- `node_two`는 `hello`만 업데이트한다. `a`는 이전 값(`True`)이 유지된다.
- `node_three`도 `hello`만 업데이트한다. `a`는 여전히 `True`이다.

#### 그래프 컴파일 및 실행

```python
graph = graph_builder.compile()

result = graph.invoke(
    {
        "hello": "world",
    },
)
```

- `compile()`: 그래프 빌더를 실행 가능한 그래프로 컴파일한다.
- `invoke()`: 초기 상태를 전달하여 그래프를 실행한다.

#### 실행 결과 추적

```
node_one {'hello': 'world'}
node_two {'hello': 'from node one.', 'a': True}
node_three {'hello': 'from node two.', 'a': True}
```

| 시점 | hello | a |
|------|-------|---|
| 초기 입력 | `"world"` | (없음) |
| node_one 실행 후 | `"from node one."` | `True` |
| node_two 실행 후 | `"from node two."` | `True` |
| node_three 실행 후 | `"from node three."` | `True` |

최종 결과: `{'hello': 'from node three.', 'a': True}`

**기본 상태 업데이트 전략은 "덮어쓰기"이다.** 노드가 반환한 키의 값이 기존 값을 대체한다. 반환하지 않은 키는 유지된다.

### 실습 포인트
- 초기 입력에 `a` 값을 포함하여 전달해 보라. 노드에서 어떻게 보이는가?
- 노드에서 `state`에 없는 키를 반환하면 어떻게 되는지 테스트해 보라.
- `graph` 객체를 직접 출력하면 그래프의 시각적 다이어그램을 확인할 수 있다.

---

## 13.4 Multiple Schemas - 다중 스키마

### 주제 및 목표
하나의 그래프에서 **입력 스키마**, **출력 스키마**, **내부(Private) 스키마**를 분리하여 사용하는 방법을 학습한다.

### 핵심 개념 설명

실제 애플리케이션에서는 다음과 같은 요구사항이 자주 발생한다:

- 사용자에게 받는 **입력 데이터**의 형태와, 내부적으로 처리하는 데이터의 형태가 다르다.
- 최종적으로 사용자에게 **반환하는 데이터**는 내부 상태의 일부만이어야 한다.
- 일부 노드만 접근할 수 있는 **비공개 상태**가 필요하다.

LangGraph는 `StateGraph`에 세 가지 스키마를 지정하여 이를 해결한다:

| 매개변수 | 역할 |
|----------|------|
| 첫 번째 인자 (State) | 내부 전체 상태 (Private State) |
| `input_schema` | 외부에서 그래프로 전달하는 입력 형태 |
| `output_schema` | 그래프가 외부로 반환하는 출력 형태 |

### 코드 분석

#### 다중 스키마 정의

```python
class PrivateState(TypedDict):
    a: int
    b: int

class InputState(TypedDict):
    hello: str

class OutputState(TypedDict):
    bye: str

class MegaPrivate(TypedDict):
    secret: bool

graph_builder = StateGraph(
    PrivateState,
    input_schema=InputState,
    output_schema=OutputState,
)
```

이 구성에서:
- 외부에서는 `{"hello": "world"}` 형태로만 입력할 수 있다.
- 내부적으로는 `a`, `b` 필드를 사용하여 계산한다.
- 최종 출력은 `{"bye": "world"}` 형태로만 반환된다.
- `MegaPrivate`는 특정 노드만 사용하는 초비공개 상태이다.

#### 다양한 스키마를 사용하는 노드들

```python
def node_one(state: InputState) -> InputState:
    print("node_one ->", state)
    return {"hello": "world"}

def node_two(state: PrivateState) -> PrivateState:
    print("node_two ->", state)
    return {"a": 1}

def node_three(state: PrivateState) -> PrivateState:
    print("node_three ->", state)
    return {"b": 1}

def node_four(state: PrivateState) -> OutputState:
    print("node_four ->", state)
    return {"bye": "world"}

def node_five(state: OutputState):
    return {"secret": True}

def node_six(state: MegaPrivate):
    print(state)
```

각 노드가 다른 스키마를 타입 힌트로 사용하고 있음에 주목하라. 이는 각 노드가 **어떤 데이터에 관심이 있는지**를 명시적으로 표현한다.

#### 그래프 구성 및 실행

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)
graph_builder.add_node("node_five", node_five)
graph_builder.add_node("node_six", node_six)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", "node_four")
graph_builder.add_edge("node_four", "node_five")
graph_builder.add_edge("node_five", "node_six")
graph_builder.add_edge("node_six", END)
```

#### 실행 결과 분석

```
node_one -> {'hello': 'world'}
node_two -> {}
node_three -> {'a': 1}
node_four -> {'a': 1, 'b': 1}
{'secret': True}
```

최종 반환값: `{'bye': 'world'}`

핵심 관찰:
- `node_one`은 `InputState`만 볼 수 있으므로 `{'hello': 'world'}`를 받는다.
- `node_two`는 `PrivateState`를 보지만, 아직 `a`, `b`가 설정되지 않았으므로 `{}`이다.
- `node_three`는 `node_two`가 설정한 `{'a': 1}`을 볼 수 있다.
- `node_four`는 `{'a': 1, 'b': 1}` 전체 PrivateState를 볼 수 있다.
- **최종 출력은 `OutputState`에 정의된 `bye` 필드만 포함**한다. 내부 상태(`a`, `b`, `secret`)는 외부로 노출되지 않는다.

### 실습 포인트
- `output_schema`를 지정하지 않으면 반환값이 어떻게 달라지는지 확인하라.
- `input_schema`에 없는 필드를 `invoke()`에 전달하면 어떻게 되는지 테스트하라.
- 스키마 분리가 실제 프로덕션 환경에서 왜 중요한지 생각해 보라 (보안, API 설계 등).

---

## 13.5 Reducer Functions - 리듀서 함수

### 주제 및 목표
기본 "덮어쓰기" 전략 대신, **리듀서(Reducer) 함수**를 사용하여 상태를 **누적(accumulate)** 하는 방법을 학습한다.

### 핵심 개념 설명

기본적으로 LangGraph의 상태 업데이트는 새로운 값이 이전 값을 **완전히 대체**한다. 하지만 많은 경우(특히 채팅 메시지 기록 등)에는 값을 **누적**해야 한다.

**Reducer 함수**는 `Annotated` 타입 힌트를 사용하여 특정 필드의 업데이트 전략을 커스터마이즈한다:

```
Annotated[타입, 리듀서_함수]
```

리듀서 함수는 두 개의 인자를 받는다:
- `old`: 현재 상태의 값
- `new`: 노드가 반환한 새로운 값

그리고 **최종적으로 저장될 값**을 반환한다.

### 코드 분석

#### 리듀서 함수 정의와 상태 적용

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Annotated
import operator

def update_function(old, new):
    return old + new

class State(TypedDict):
    # messages: Annotated[list[str], update_function]
    messages: Annotated[list[str], operator.add]

graph_builder = StateGraph(State)
```

핵심 포인트:
- `Annotated[list[str], operator.add]`는 "`messages` 필드가 업데이트될 때 기존 리스트에 새 리스트를 **이어 붙여라(concatenate)**"라는 의미이다.
- `operator.add`는 파이썬 내장 함수로, 리스트에 대해 `+` 연산(결합)을 수행한다.
- 주석 처리된 `update_function`은 동일한 동작을 하는 커스텀 리듀서 함수이다. 직접 만들 수도, `operator.add` 같은 기존 함수를 사용할 수도 있다.

#### 노드 정의

```python
def node_one(state: State):
    return {
        "messages": ["Hello, nice to meet you!"],
    }

def node_two(state: State):
    return {}

def node_three(state: State):
    return {}
```

- `node_one`만 `messages`에 새 항목을 추가한다.
- `node_two`, `node_three`는 빈 딕셔너리를 반환하므로 상태를 변경하지 않는다.

#### 실행 및 결과

```python
graph = graph_builder.compile()

graph.invoke(
    {"messages": ["Hello!"]},
)
```

결과: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`

| 시점 | messages |
|------|----------|
| 초기 입력 | `["Hello!"]` |
| node_one 실행 후 | `["Hello!"] + ["Hello, nice to meet you!"]` = `["Hello!", "Hello, nice to meet you!"]` |
| node_two, node_three | 변경 없음 |

**리듀서가 없었다면** `node_one`의 반환값 `["Hello, nice to meet you!"]`이 초기값 `["Hello!"]`를 완전히 대체했을 것이다. 리듀서 덕분에 두 리스트가 **결합**되었다.

### 실습 포인트
- 커스텀 리듀서 함수를 작성해 보라 (예: 최대값만 유지, 중복 제거 등).
- `node_two`에서도 메시지를 추가하여 누적이 정상적으로 동작하는지 확인하라.
- 리듀서 없이 동일한 코드를 실행하면 결과가 어떻게 달라지는지 비교하라.
- 채팅 애플리케이션에서 리듀서가 왜 필수적인지 생각해 보라.

---

## 13.6 Node Caching - 노드 캐싱

### 주제 및 목표
**CachePolicy**를 사용하여 특정 노드의 실행 결과를 캐싱하고, 일정 시간 동안 재계산 없이 캐시된 값을 사용하는 방법을 학습한다.

### 핵심 개념 설명

일부 노드는 실행 비용이 높거나(예: 외부 API 호출, LLM 호출), 동일한 입력에 대해 같은 결과를 반환한다. 이런 경우 **캐싱**을 통해 성능을 최적화할 수 있다.

LangGraph는 노드 단위의 캐시 정책을 제공한다:

| 구성 요소 | 역할 |
|-----------|------|
| `CachePolicy(ttl=초)` | 캐시 유효 기간(Time-To-Live)을 초 단위로 설정 |
| `InMemoryCache()` | 메모리 기반 캐시 저장소 |
| `graph_builder.compile(cache=...)` | 컴파일 시 캐시 저장소 연결 |

### 코드 분석

#### 임포트 및 상태 정의

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from datetime import datetime

class State(TypedDict):
    time: str

graph_builder = StateGraph(State)
```

#### 노드 정의 - 캐싱 대상 노드

```python
def node_one(state: State):
    return {}

def node_two(state: State):
    return {"time": f"{datetime.now()}"}

def node_three(state: State):
    return {}
```

`node_two`는 현재 시간을 반환한다. 캐싱이 적용되면, TTL 기간 동안은 이전에 기록한 시간이 그대로 반환된다.

#### 캐시 정책 적용

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node(
    "node_two",
    node_two,
    cache_policy=CachePolicy(ttl=20),  # 20초 동안 캐시 유지
)
graph_builder.add_node("node_three", node_three)

graph_builder.add_edge(START, "node_one")
graph_builder.add_edge("node_one", "node_two")
graph_builder.add_edge("node_two", "node_three")
graph_builder.add_edge("node_three", END)
```

핵심: `node_two`에만 `cache_policy=CachePolicy(ttl=20)`을 지정했다. 이 노드의 결과는 **20초 동안 캐시**된다.

#### 캐시를 활성화한 컴파일 및 반복 실행

```python
import time

graph = graph_builder.compile(cache=InMemoryCache())

print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
print(graph.invoke({}))
time.sleep(5)
```

이 코드는 5초 간격으로 6번 그래프를 실행한다. `node_two`의 `ttl=20`이므로:

- **처음~20초**: 첫 번째 실행 결과(시간)가 캐시되어 동일한 시간이 반환된다.
- **20초 이후**: 캐시가 만료되어 `node_two`가 다시 실행되고 새로운 시간이 기록된다.

### 실습 포인트
- `ttl` 값을 변경하며 캐시 만료 타이밍을 관찰하라.
- `InMemoryCache` 대신 다른 캐시 저장소를 사용할 수 있는지 조사하라.
- 캐시가 유용한 실제 시나리오를 생각해 보라 (예: 외부 API 요율 제한, 비용 절감).
- 캐시가 문제가 될 수 있는 경우도 생각해 보라 (예: 실시간 데이터가 필요한 경우).

---

## 13.7 Conditional Edges - 조건부 엣지

### 주제 및 목표
**조건부 엣지(Conditional Edges)**를 사용하여 상태에 따라 **동적으로 다음 노드를 선택**하는 분기(branching) 로직을 구현한다.

### 핵심 개념 설명

지금까지의 그래프는 모든 경로가 고정되어 있었다(선형 흐름). 하지만 실제 애플리케이션에서는 상태에 따라 다른 경로를 선택해야 하는 경우가 많다.

**`add_conditional_edges`** 메서드를 사용하면:
1. 특정 노드 이후에 **분기 함수(routing function)**를 실행한다.
2. 분기 함수의 반환값에 따라 다음 노드를 동적으로 결정한다.

```
add_conditional_edges(
    출발_노드,
    분기_함수,
    매핑_딕셔너리   # {반환값: 목적지_노드}
)
```

### 코드 분석

#### 상태 및 노드 정의

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal

class State(TypedDict):
    seed: int

graph_builder = StateGraph(State)

def node_one(state: State):
    print("node_one ->", state)
    return {}

def node_two(state: State):
    print("node_two ->", state)
    return {}

def node_three(state: State):
    print("node_three ->", state)
    return {}

def node_four(state: State):
    print("node_four ->", state)
    return {}
```

#### 분기 함수 정의

코드에는 두 가지 방식의 분기 함수가 보인다:

**방식 1: 문자열 반환 (주석 처리됨)**
```python
# def decide_path(state: State) -> Literal["node_three", "node_four"]:
#     if state["seed"] % 2 == 0:
#         return "node_three"
#     else:
#         return "node_four"
```
이 방식은 노드 이름을 직접 반환한다. `Literal` 타입 힌트로 가능한 반환값을 명시한다.

**방식 2: 임의의 값 반환 + 매핑 딕셔너리 (실제 사용)**
```python
def decide_path(state: State):
    return state["seed"] % 2 == 0  # True 또는 False 반환
```
분기 함수가 `True`/`False` 같은 임의의 값을 반환하고, 매핑 딕셔너리에서 실제 노드로 변환한다.

#### 조건부 엣지 구성

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)
graph_builder.add_node("node_three", node_three)
graph_builder.add_node("node_four", node_four)

# START에서 조건부 분기
graph_builder.add_conditional_edges(
    START,
    decide_path,
    {
        True: "node_one",     # seed가 짝수이면 node_one으로
        False: "node_two",    # seed가 홀수이면 node_two로
        "hello": END,         # "hello" 반환 시 종료
    },
)

graph_builder.add_edge("node_one", "node_two")

# node_two에서도 조건부 분기
graph_builder.add_conditional_edges(
    "node_two",
    decide_path,
    {
        True: "node_three",
        False: "node_four",
        "hello": END,
    },
)

graph_builder.add_edge("node_four", END)
graph_builder.add_edge("node_three", END)
```

이 그래프의 흐름:

```
             ┌─ True ──> node_one ──> node_two ─┬─ True ──> node_three ──> END
START ───────┤                                  ├─ False ─> node_four ───> END
             ├─ False ─> node_two ──────────────┘
             └─ "hello" ──> END
```

### 실습 포인트
- `seed` 값을 다양하게 변경하며 실행 경로가 어떻게 달라지는지 관찰하라.
- 매핑 딕셔너리 없이 노드 이름을 직접 반환하는 방식(방식 1)으로 변경해 보라.
- 3개 이상의 분기를 가진 조건부 엣지를 설계해 보라.
- 조건부 엣지와 일반 엣지를 혼합하여 복잡한 워크플로우를 만들어 보라.

---

## 13.8 Send API - 동적 병렬 처리

### 주제 및 목표
**Send API**를 사용하여 런타임에 **동적으로 노드 인스턴스를 생성**하고 **병렬로 실행**하는 방법을 학습한다.

### 핵심 개념 설명

조건부 엣지는 "어떤 노드로 갈 것인가"를 결정하지만, **Send API**는 한 단계 더 나아가:

1. **동일한 노드를 여러 번** 병렬로 실행할 수 있다.
2. 각 인스턴스에 **서로 다른 입력**을 전달할 수 있다.
3. 실행할 인스턴스의 **수가 런타임에 결정**된다.

이는 Map-Reduce 패턴과 유사하다:
- **Map**: 데이터를 분할하여 각각에 동일한 처리를 적용
- **Reduce**: 결과를 모아 합침 (Reducer 함수와 결합하여 사용)

### 코드 분석

#### 임포트 및 상태 정의

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.types import Send
import operator
from typing import Union

class State(TypedDict):
    words: list[str]
    output: Annotated[list[dict[str, Union[str, int]]], operator.add]

graph_builder = StateGraph(State)
```

핵심 포인트:
- `words`: 처리할 단어 목록
- `output`: 각 단어의 처리 결과를 **누적**하는 리스트. `Annotated`와 `operator.add` 리듀서를 사용하여 결과를 합친다.
- `Send`를 임포트한다. 이것이 동적 병렬 처리의 핵심이다.

#### 노드 정의

```python
def node_one(state: State):
    print(f"I want to count {len(state['words'])} words in my state.")

def node_two(word: str):
    return {
        "output": [
            {
                "word": word,
                "letters": len(word),
            }
        ]
    }
```

중요한 차이점:
- `node_one`은 일반적인 노드로 전체 `State`를 받는다.
- **`node_two`는 `State`가 아닌 개별 `word`(문자열)를 받는다.** Send API를 통해 전달되는 커스텀 입력이다.
- `node_two`는 결과를 `output` 리스트에 추가한다. 리듀서(`operator.add`) 덕분에 병렬 실행된 모든 결과가 자동으로 합쳐진다.

#### 디스패처 함수와 그래프 구성

```python
graph_builder.add_node("node_one", node_one)
graph_builder.add_node("node_two", node_two)

def dispatcher(state: State):
    return [Send("node_two", word) for word in state["words"]]

graph_builder.add_edge(START, "node_one")
graph_builder.add_conditional_edges("node_one", dispatcher, ["node_two"])
graph_builder.add_edge("node_two", END)
```

핵심 분석:

1. **`dispatcher` 함수**: `state["words"]`의 각 단어에 대해 `Send` 객체를 생성한다.
   - `Send("node_two", word)`: "`node_two`를 `word`를 입력으로 하여 실행하라"
   - 리스트를 반환하므로 **단어 수만큼** `node_two`가 병렬 실행된다.

2. **`add_conditional_edges`에 리스트 전달**: `["node_two"]`는 가능한 목적지 노드 목록이다.

#### 실행 결과

```python
graph.invoke(
    {
        "words": ["hello", "world", "how", "are", "you", "doing"],
    }
)
```

출력:
```
I want to count 6 words in my state.
```

결과:
```python
{
    'words': ['hello', 'world', 'how', 'are', 'you', 'doing'],
    'output': [
        {'word': 'hello', 'letters': 5},
        {'word': 'world', 'letters': 5},
        {'word': 'how', 'letters': 3},
        {'word': 'are', 'letters': 3},
        {'word': 'you', 'letters': 3},
        {'word': 'doing', 'letters': 5}
    ]
}
```

6개의 `node_two` 인스턴스가 각각 하나의 단어를 처리하고, 그 결과가 `operator.add` 리듀서에 의해 `output` 리스트로 합쳐졌다.

### 실습 포인트
- 단어 목록의 크기를 늘려 보고 성능 차이를 관찰하라.
- `node_two`에 `time.sleep`을 추가하여 병렬 실행의 효과를 체감하라.
- Send API를 사용하지 않고 동일한 결과를 만드는 코드를 작성해 비교하라.
- 실제 활용 사례를 생각해 보라 (예: 여러 문서를 동시에 요약, 여러 소스에서 데이터 수집 등).

---

## 13.9 Command - 커맨드 객체

### 주제 및 목표
**Command** 객체를 사용하여 노드 내부에서 **상태 업데이트와 라우팅을 동시에** 수행하는 방법을 학습한다.

### 핵심 개념 설명

지금까지 배운 방법들에서는:
- 상태 업데이트: 노드가 딕셔너리를 반환
- 라우팅: `add_conditional_edges` + 별도의 분기 함수

이 두 가지가 **분리**되어 있었다. **Command** 객체는 이 둘을 **하나로 통합**한다:

```python
Command(
    goto="목적지_노드",       # 다음으로 이동할 노드
    update={"key": "value"},  # 상태 업데이트
)
```

이 방식의 장점:
- 라우팅 로직이 노드 내부에 있어 더 직관적이다.
- 상태 업데이트와 라우팅을 원자적(atomic)으로 처리한다.
- 별도의 분기 함수나 조건부 엣지가 필요 없다.

### 코드 분석

#### 임포트 및 상태 정의

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import Command

class State(TypedDict):
    transfer_reason: str

graph_builder = StateGraph(State)
```

#### 노드 정의 - Command를 반환하는 라우터 노드

```python
from typing import Literal

def triage_node(state: State) -> Command[Literal["account_support", "tech_support"]]:
    return Command(
        goto="account_support",
        update={
            "transfer_reason": "The user wants to change password.",
        },
    )

def tech_support(state: State):
    return {}

def account_support(state: State):
    print("account_support running")
    return {}
```

핵심 분석:

1. **`triage_node`의 반환 타입**: `Command[Literal["account_support", "tech_support"]]`
   - 이 노드가 `Command`를 반환하며, 가능한 목적지가 `"account_support"` 또는 `"tech_support"`임을 타입으로 명시한다.
   - 이 타입 힌트 덕분에 LangGraph는 **`add_edge`나 `add_conditional_edges` 없이도** 가능한 경로를 알 수 있다.

2. **`Command` 객체**:
   - `goto="account_support"`: 다음에 `account_support` 노드로 이동
   - `update={"transfer_reason": "The user wants to change password."}`: 상태의 `transfer_reason`을 업데이트

#### 그래프 구성

```python
graph_builder.add_node("triage_node", triage_node)
graph_builder.add_node("tech_support", tech_support)
graph_builder.add_node("account_support", account_support)

graph_builder.add_edge(START, "triage_node")
# triage_node에는 add_edge가 필요 없다! Command가 라우팅을 처리한다.

graph_builder.add_edge("tech_support", END)
graph_builder.add_edge("account_support", END)
```

주목할 점: `triage_node` 이후의 엣지는 **정의하지 않았다**. `Command` 객체의 `goto`가 런타임에 다음 노드를 결정하기 때문이다.

그래프 구조:
```
                         ┌──> tech_support ────> END
START ──> triage_node ───┤
                         └──> account_support ──> END
```

#### 실행 결과

```python
graph = graph_builder.compile()
graph.invoke({})
```

출력:
```
account_support running
```

결과: `{'transfer_reason': 'The user wants to change password.'}`

`triage_node`가 `Command`를 통해:
1. `transfer_reason`을 업데이트하고
2. `account_support`로 라우팅했다.

### 실습 포인트
- `triage_node`에서 조건에 따라 `tech_support`로 라우팅하도록 수정해 보라.
- `Command`와 `add_conditional_edges` 방식의 장단점을 비교해 보라.
- 실제 고객 지원 시스템처럼 여러 단계의 라우팅을 `Command`로 구현해 보라.
- `Command`에서 `goto`에 여러 노드를 지정할 수 있는지 조사해 보라.

---

## 챕터 핵심 정리 (Key Takeaways)

### 1. LangGraph의 기본 구조
- **StateGraph**: 상태 기반 그래프의 핵심 클래스
- **Node**: 상태를 받아 처리하고 업데이트를 반환하는 함수
- **Edge**: 노드 간의 연결 (실행 순서 결정)
- **START / END**: 그래프의 진입점과 종료점

### 2. 상태(State) 관리
- `TypedDict`로 상태 스키마를 정의한다.
- 기본 업데이트 전략은 **덮어쓰기(overwrite)**이다.
- `Annotated`와 리듀서 함수를 사용하면 **누적(accumulate)** 전략을 적용할 수 있다.
- `operator.add`는 리스트 결합에 가장 많이 사용되는 리듀서이다.

### 3. 다중 스키마
- `input_schema`: 외부 입력 형태 제한
- `output_schema`: 외부 출력 형태 제한
- 내부 상태는 외부에 노출되지 않아 보안과 API 설계에 유리하다.

### 4. 캐싱
- `CachePolicy(ttl=초)`로 노드별 캐시 정책을 설정한다.
- `InMemoryCache()`와 함께 `compile(cache=...)`로 활성화한다.
- 비용이 높은 연산(API 호출 등)의 성능을 크게 개선할 수 있다.

### 5. 흐름 제어
| 방식 | 특징 | 사용 시점 |
|------|------|-----------|
| `add_edge` | 고정 경로 | 항상 동일한 다음 노드 |
| `add_conditional_edges` | 분기 함수 기반 동적 라우팅 | 상태에 따라 경로 변경 |
| `Send` API | 동적 병렬 실행 | 동일 노드를 다른 입력으로 여러 번 실행 |
| `Command` | 노드 내부 라우팅 + 상태 업데이트 | 라우팅과 상태 변경을 한 번에 처리 |

### 6. 핵심 설계 원칙
- 그래프는 **선언적(declarative)**으로 구성한다: 먼저 노드와 엣지를 정의하고, 나중에 컴파일하여 실행한다.
- 상태는 **불변(immutable)**처럼 다룬다: 노드는 새로운 딕셔너리를 반환하여 상태를 업데이트한다.
- **관심사의 분리**: 각 노드는 하나의 책임만 가진다.

---

## 실습 과제 (Practice Exercises)

### 과제 1: 기본 그래프 (난이도: ★☆☆)

4개의 노드(`start_node`, `process_a`, `process_b`, `end_node`)로 구성된 선형 그래프를 만들어라. 상태에 `counter: int` 필드를 두고, 각 노드가 `counter`를 1씩 증가시키도록 구현하라. 최종 `counter` 값이 4가 되어야 한다.

**힌트**: 리듀서를 사용하지 않으면 덮어쓰기가 된다. 각 노드에서 현재 값을 읽어 +1한 값을 반환하라.

### 과제 2: 채팅 메시지 누적 (난이도: ★★☆)

리듀서를 활용하여 간단한 채팅 시뮬레이터를 만들어라:
- 상태: `messages: Annotated[list[str], operator.add]`
- `user_node`: `["사용자: 안녕하세요"]` 추가
- `assistant_node`: `["어시스턴트: 무엇을 도와드릴까요?"]` 추가
- `user_reply_node`: `["사용자: 날씨 알려주세요"]` 추가

최종 `messages`에 3개의 메시지가 순서대로 포함되어야 한다.

### 과제 3: 조건부 라우팅 (난이도: ★★☆)

사용자의 나이에 따라 다른 경로로 분기하는 그래프를 만들어라:
- 상태: `age: int`, `message: str`
- `check_age` 노드 이후 조건부 분기:
  - 18세 미만: `minor_node` -> "미성년자입니다."
  - 18세 이상 65세 미만: `adult_node` -> "성인입니다."
  - 65세 이상: `senior_node` -> "경로 우대 대상입니다."

### 과제 4: Send API 활용 (난이도: ★★★)

문장을 입력받아 각 단어를 동시에 대문자로 변환하는 그래프를 만들어라:
- 상태: `sentence: str`, `results: Annotated[list[str], operator.add]`
- `splitter_node`: 문장을 단어로 분리
- `upper_node`: 개별 단어를 대문자로 변환 (Send API로 병렬 실행)
- 입력: `{"sentence": "hello world from langgraph"}`
- 기대 출력: `{"sentence": "...", "results": ["HELLO", "WORLD", "FROM", "LANGGRAPH"]}`

### 과제 5: Command 기반 에이전트 (난이도: ★★★)

Command 객체를 사용하여 간단한 고객 상담 라우터를 구현하라:
- 상태: `query: str`, `department: str`, `response: str`
- `router_node`: 쿼리 내용에 따라 Command로 라우팅
  - "환불" 또는 "결제" 포함 -> `billing_node`
  - "오류" 또는 "버그" 포함 -> `tech_node`
  - 그 외 -> `general_node`
- 각 부서 노드는 `response`에 적절한 안내 메시지를 설정

**보너스**: `Command`의 타입 힌트(`Command[Literal[...]]`)를 정확히 사용하여, 그래프 시각화 시 모든 가능한 경로가 표시되도록 하라.
