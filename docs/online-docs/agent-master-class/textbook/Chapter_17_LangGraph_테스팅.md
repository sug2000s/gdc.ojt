# Chapter 17: LangGraph 워크플로우 테스팅

---

## 1. 챕터 개요

이 챕터에서는 **LangGraph로 구축한 AI 에이전트 워크플로우를 체계적으로 테스트하는 방법**을 학습한다. 단순한 규칙 기반 그래프에서 시작하여, AI(LLM) 기반 노드로 전환한 뒤, 비결정적(non-deterministic)인 AI 응답을 어떻게 신뢰성 있게 검증할 수 있는지까지 단계적으로 다룬다.

### 학습 목표

- LangGraph의 `StateGraph`를 활용한 이메일 처리 워크플로우 구축
- `pytest`를 활용한 그래프 테스트 프레임워크 구성
- 개별 노드 단위 테스트 및 부분 실행(Partial Execution) 테스트
- 규칙 기반 노드를 AI(LLM) 노드로 전환하는 과정 이해
- AI 응답의 비결정적 특성에 맞는 테스트 전략 수립
- LLM-as-a-Judge 패턴을 활용한 AI 응답 품질 평가

### 프로젝트 구조

```
workflow-testing/
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py          # LangGraph 워크플로우 정의
├── tests.py         # pytest 테스트 코드
└── README.md
```

---

## 2. 섹션별 상세 설명

---

### 17.0 Introduction -- 프로젝트 초기 설정

**주제 및 목표:** 테스트 실습을 위한 Python 프로젝트 환경을 구성한다.

#### 핵심 개념 설명

이 섹션에서는 `uv` 패키지 매니저를 사용하여 새로운 Python 프로젝트를 생성한다. `pyproject.toml`에 프로젝트 의존성을 정의하며, 핵심 라이브러리들의 역할은 다음과 같다:

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `langchain[openai]` | 0.3.27 | LLM 연동 프레임워크 |
| `langgraph` | 0.6.6 | 워크플로우 그래프 빌더 |
| `langgraph-checkpoint-sqlite` | 2.0.11 | 상태 체크포인트 저장 |
| `pytest` | 8.4.2 | Python 테스트 프레임워크 |
| `python-dotenv` | 1.1.1 | 환경변수(.env) 로딩 |

#### 코드 분석

```toml
# pyproject.toml
[project]
name = "workflow-testing"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]
```

**주목할 점:**
- Python 3.13 이상을 요구한다. `.python-version` 파일에 `3.13`이 명시되어 있어 `uv`가 자동으로 올바른 Python 버전을 사용한다.
- `pytest`가 프로젝트 의존성에 포함되어 있다. 이는 테스트가 개발 과정의 핵심 구성 요소임을 보여준다.
- `grandalf`는 그래프 시각화를 위한 라이브러리이다.

#### 실습 포인트

1. `uv init workflow-testing` 명령으로 프로젝트를 생성해 본다.
2. `uv add langgraph pytest langchain[openai]` 등의 명령으로 의존성을 추가해 본다.
3. `uv sync`를 실행하여 모든 패키지가 정상적으로 설치되는지 확인한다.

---

### 17.1 Email Graph -- 이메일 처리 워크플로우 구축

**주제 및 목표:** LangGraph의 `StateGraph`를 활용하여 이메일을 분류하고, 우선순위를 매기고, 응답을 생성하는 3단계 워크플로우를 구축한다.

#### 핵심 개념 설명

LangGraph의 워크플로우는 세 가지 핵심 요소로 구성된다:

1. **State (상태):** 워크플로우 전체에서 공유되는 데이터 구조. `TypedDict`로 정의하며, 각 노드가 상태의 일부를 읽고 쓴다.
2. **Node (노드):** 상태를 입력받아 처리한 뒤 상태 업데이트를 반환하는 함수.
3. **Edge (엣지):** 노드 간의 실행 순서를 정의하는 연결.

이 섹션에서 구축하는 워크플로우의 흐름:

```
START --> categorize_email --> assing_priority --> draft_response --> END
```

#### 코드 분석

**1단계: 상태 정의**

```python
from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class EmailState(TypedDict):
    email: str
    category: Literal["spam", "normal", "urgent"]
    priority_score: int
    response: str
```

`EmailState`는 워크플로우가 처리할 데이터의 스키마를 정의한다. `Literal` 타입을 사용하여 `category`가 "spam", "normal", "urgent" 중 하나만 가질 수 있도록 제약한다. 이메일 원문(`email`), 분류 결과(`category`), 우선순위 점수(`priority_score`), 생성된 응답(`response`)이 상태에 포함된다.

**2단계: 노드 함수 정의**

```python
def categorize_email(state: EmailState):
    email = state["email"].lower()

    if "urgent" in email or "asap" in email:
        category = "urgent"
    elif "offer" in email or "discount" in email:
        category = "spam"
    else:
        category = "normal"

    return {
        "category": category,
    }
```

`categorize_email` 노드는 이메일 본문에 포함된 키워드를 기반으로 카테고리를 분류한다. 단순한 규칙 기반 로직으로, "urgent"이나 "asap"이 포함되면 긴급, "offer"이나 "discount"이 포함되면 스팸, 나머지는 일반으로 분류한다.

**핵심:** 각 노드 함수는 전체 상태(`EmailState`)를 매개변수로 받지만, 자신이 변경하는 필드만 포함하는 딕셔너리를 반환한다. LangGraph가 이 반환값을 기존 상태에 **병합(merge)** 한다.

```python
def assing_priority(state: EmailState):
    scores = {
        "urgent": 10,
        "normal": 5,
        "spam": 1,
    }
    return {
        "priority_score": scores[state["category"]],
    }


def draft_response(state: EmailState) -> EmailState:
    responses = {
        "urgent": "I will answer you as fast as i can",
        "normal": "I'll get back to you soon",
        "spam": "Go away!",
    }
    return {
        "response": responses[state["category"]],
    }
```

`assing_priority`는 카테고리별 고정 점수를 매기고, `draft_response`는 카테고리별 고정 응답 메시지를 생성한다. 이 시점에서는 모든 로직이 결정적(deterministic)이다 -- 동일한 입력에 항상 동일한 출력을 보장한다.

**3단계: 그래프 조립 및 실행**

```python
graph_builder = StateGraph(EmailState)

graph_builder.add_node("categorize_email", categorize_email)
graph_builder.add_node("assing_priority", assing_priority)
graph_builder.add_node("draft_response", draft_response)

graph_builder.add_edge(START, "categorize_email")
graph_builder.add_edge("categorize_email", "assing_priority")
graph_builder.add_edge("assing_priority", "draft_response")
graph_builder.add_edge("draft_response", END)

graph = graph_builder.compile()

result = graph.invoke({"email": "i have an offer for you!"})
print(result)
```

`StateGraph`에 노드를 등록하고, 엣지로 순서를 지정한 뒤, `compile()`로 실행 가능한 그래프를 생성한다. `invoke()`에 초기 상태(이메일 본문)를 넘기면 전체 워크플로우가 순차적으로 실행된다.

#### 실습 포인트

1. "i have an offer for you!" 외에 다양한 이메일 텍스트로 `graph.invoke()`를 호출하여 결과를 확인해 보라.
2. 새로운 카테고리(예: "important")를 추가하려면 어떤 부분들을 수정해야 하는지 생각해 보라.
3. 조건 분기(`add_conditional_edges`)를 사용하여 스팸 이메일은 `draft_response`를 건너뛰도록 수정해 보라.

---

### 17.2 Pytest -- 테스트 프레임워크 도입

**주제 및 목표:** `pytest`를 사용하여 LangGraph 워크플로우의 자동화된 테스트를 작성한다. `@pytest.mark.parametrize`를 활용한 매개변수화 테스트를 배운다.

#### 핵심 개념 설명

**pytest**는 Python의 대표적인 테스트 프레임워크이다. 주요 특징:

- 함수 이름이 `test_`로 시작하면 자동으로 테스트로 인식
- `assert` 문으로 간결하게 검증
- `@pytest.mark.parametrize`로 동일한 테스트 로직을 여러 입력값에 대해 반복 실행

**매개변수화 테스트(Parameterized Test)** 는 테스트 코드의 중복을 제거하는 핵심 기법이다. 하나의 테스트 함수로 여러 시나리오를 커버할 수 있다.

#### 코드 분석

먼저 `main.py`에서 직접 실행 코드(invoke + print)를 제거한다:

```python
# 제거된 코드 (main.py 하단)
# result = graph.invoke({"email": "i have an offer for you!"})
# print(result)
```

프로덕션 코드와 테스트 코드를 분리하는 것이 기본 원칙이다. `main.py`는 그래프 정의만 담당하고, 실행 및 검증은 `tests.py`에서 수행한다.

```python
# tests.py
import pytest
from main import graph


@pytest.mark.parametrize(
    "email, expected_category, expected_score",
    [
        ("this is urgent!", "urgent", 10),
        ("i wanna talk to you", "normal", 5),
        ("i have an offer for you", "spam", 1),
    ],
)
def test_full_graph(email, expected_category, expected_score):

    result = graph.invoke({"email": email})

    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

**코드 해설:**

1. `from main import graph`: `main.py`에서 컴파일된 그래프를 임포트한다.
2. `@pytest.mark.parametrize`: 데코레이터의 첫 번째 인자는 매개변수 이름들(쉼표로 구분된 문자열), 두 번째 인자는 테스트 케이스 리스트이다.
3. 각 튜플 `("this is urgent!", "urgent", 10)`은 하나의 테스트 케이스를 나타낸다.
4. `graph.invoke()`로 전체 워크플로우를 실행하고, `assert`로 기대 결과를 검증한다.

이 테스트는 3개의 독립된 테스트 케이스로 실행된다:
- 긴급 이메일 -> category="urgent", priority_score=10
- 일반 이메일 -> category="normal", priority_score=5
- 스팸 이메일 -> category="spam", priority_score=1

#### 실습 포인트

1. 터미널에서 `pytest tests.py -v`를 실행하여 각 테스트 케이스가 개별적으로 실행되는 것을 확인해 보라. (`-v`는 verbose 모드)
2. 의도적으로 기대값을 틀리게 바꿔서 실패 메시지를 확인해 보라. pytest가 어떤 정보를 제공하는지 파악한다.
3. 엣지 케이스를 추가해 보라: "URGENT offer" 처럼 두 키워드가 모두 포함된 경우 어떤 결과가 나오는가?

---

### 17.3 Testing Nodes -- 노드 단위 테스트 및 부분 실행

**주제 및 목표:** 전체 그래프 실행 외에, (1) 개별 노드를 독립적으로 테스트하고, (2) 그래프의 중간 상태를 주입하여 특정 지점부터 부분 실행하는 고급 테스트 기법을 학습한다.

#### 핵심 개념 설명

워크플로우 테스트에는 세 가지 수준이 있다:

| 테스트 수준 | 설명 | 용도 |
|------------|------|------|
| **전체 그래프 테스트** | `graph.invoke()`로 처음부터 끝까지 실행 | 통합 테스트, E2E 검증 |
| **개별 노드 테스트** | `graph.nodes["노드명"].invoke()`로 특정 노드만 실행 | 단위 테스트, 노드 로직 검증 |
| **부분 실행 테스트** | `graph.update_state()`로 중간 상태를 주입한 뒤 이어서 실행 | 특정 시나리오 재현, 디버깅 |

**MemorySaver(체크포인터):** 부분 실행을 위해서는 그래프가 상태를 저장할 수 있어야 한다. `MemorySaver`는 메모리 기반 체크포인터로, 각 `thread_id`별로 그래프의 실행 상태를 저장한다.

#### 코드 분석

**체크포인터 추가 (main.py):**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# ... (노드, 엣지 정의 생략) ...

graph = graph_builder.compile(checkpointer=checkpointer)
```

`compile()` 시 `checkpointer`를 전달하면, 그래프가 각 노드 실행 후 상태를 자동으로 저장한다. 이후 `thread_id`를 통해 특정 실행의 상태를 조회하거나 수정할 수 있다.

**전체 그래프 테스트 수정:**

```python
def test_full_graph(email, expected_category, expected_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert result["priority_score"] == expected_score
```

체크포인터를 사용하는 그래프는 반드시 `config`에 `thread_id`를 제공해야 한다.

**개별 노드 테스트:**

```python
def test_individual_nodes():

    # categorize_email 노드만 단독 실행
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # assing_priority 노드만 단독 실행
    result = graph.nodes["assing_priority"].invoke({"category": "spam"})
    assert result["priority_score"] == 1

    # draft_response 노드만 단독 실행
    result = graph.nodes["draft_response"].invoke({"category": "spam"})
    assert "Go away" in result["response"]
```

`graph.nodes`는 등록된 노드들의 딕셔너리이다. 각 노드는 `invoke()` 메서드를 가지며, 해당 노드 함수에 필요한 상태만 전달하면 독립적으로 실행할 수 있다. 이를 통해:

- `categorize_email`은 `email` 필드만 필요
- `assing_priority`는 `category` 필드만 필요
- `draft_response`는 `category` 필드만 필요

이렇게 각 노드의 입출력을 격리하여 테스트하면, 문제가 발생했을 때 어떤 노드에 버그가 있는지 빠르게 파악할 수 있다.

**부분 실행 테스트:**

```python
def test_partial_execution():

    # 1단계: 중간 상태를 직접 주입
    graph.update_state(
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        values={
            "email": "please check out this offer",
            "category": "spam",
        },
        as_node="categorize_email",  # 이 노드가 실행된 것처럼 상태를 설정
    )

    # 2단계: 주입된 상태부터 이어서 실행
    result = graph.invoke(
        None,  # 새 입력 없이 기존 상태에서 이어서 실행
        config={
            "configurable": {
                "thread_id": "1",
            },
        },
        interrupt_after="draft_response",
    )

    assert result["priority_score"] == 1
```

이 테스트의 핵심 동작:

1. `update_state()`로 `categorize_email` 노드가 이미 실행 완료된 것처럼 상태를 주입한다. `as_node="categorize_email"`은 "이 상태가 categorize_email 노드의 출력이다"라는 의미이다.
2. `graph.invoke(None, ...)`으로 입력 없이 저장된 상태에서 이어서 실행한다. `categorize_email` 다음 노드인 `assing_priority`부터 실행된다.
3. `interrupt_after="draft_response"`로 `draft_response` 실행 후 중단한다.

이 기법은 다음과 같은 상황에서 유용하다:
- 앞쪽 노드의 실행 비용이 높을 때(예: LLM 호출)
- 특정 중간 상태에서의 동작만 검증하고 싶을 때
- 엣지 케이스를 인위적으로 만들어야 할 때

#### 실습 포인트

1. `graph.nodes`에 어떤 키들이 있는지 출력해 보라.
2. `update_state()`의 `as_node`를 `"assing_priority"`로 바꿔서 더 뒤쪽 노드부터 실행해 보라.
3. 존재하지 않는 `thread_id`로 `invoke(None, ...)`을 호출하면 어떻게 되는지 확인해 보라.

---

### 17.4 AI Nodes -- 규칙 기반에서 LLM 기반으로 전환

**주제 및 목표:** 하드코딩된 규칙 기반 노드를 LLM(GPT-4o)을 활용한 AI 노드로 교체한다. Pydantic의 `BaseModel`과 LangChain의 `with_structured_output`을 활용하여 LLM 출력을 구조화한다.

#### 핵심 개념 설명

규칙 기반 시스템의 한계:
- "urgent"라는 단어가 없지만 긴급한 이메일을 처리하지 못함
- 새로운 패턴마다 if/elif 조건을 수동으로 추가해야 함
- 자연어의 다양한 표현을 포괄하기 어려움

LLM을 사용하면 자연어의 **의미(semantics)** 를 이해하여 분류할 수 있다. 다만 LLM의 출력은 자유 형식 텍스트이므로, **Structured Output(구조화 출력)** 을 사용하여 프로그래밍적으로 처리 가능한 형태로 강제한다.

**Structured Output 패턴:**
1. Pydantic `BaseModel`로 원하는 출력 스키마를 정의
2. `llm.with_structured_output(Model)`로 LLM을 감싸기
3. LLM이 반드시 해당 스키마에 맞는 JSON을 반환하도록 강제

#### 코드 분석

**LLM 초기화 및 출력 스키마 정의:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class EmailClassificationOuput(BaseModel):
    category: Literal["spam", "normal", "urgent"] = Field(
        description="Category of the email",
    )


class PriorityScoreOutput(BaseModel):
    priority_score: int = Field(
        description="Priority score from 1 to 10",
        ge=1,
        le=10,
    )
```

`EmailClassificationOuput`은 LLM이 반드시 "spam", "normal", "urgent" 중 하나를 반환하도록 강제한다. `PriorityScoreOutput`은 1~10 사이의 정수를 반환하도록 `ge`(greater than or equal)와 `le`(less than or equal) 검증을 포함한다.

**AI 기반 categorize_email:**

```python
def categorize_email(state: EmailState):
    s_llm = llm.with_structured_output(EmailClassificationOuput)

    result = s_llm.invoke(
        f"""Classify this email into one of three categories:
        - urgent: time-sensitive, requires immediate attention
        - normal: regular business communication
        - spam: promotional, marketing, or unwanted content

        Email: {state['email']}"""
    )

    return {
        "category": result.category,
    }
```

이전의 키워드 매칭 대신, LLM에 분류 기준을 프롬프트로 전달한다. `with_structured_output()`이 반환하는 `s_llm`은 항상 `EmailClassificationOuput` 인스턴스를 반환한다. `result.category`로 타입 안전하게 값에 접근할 수 있다.

**AI 기반 assing_priority:**

```python
def assing_priority(state: EmailState):
    s_llm = llm.with_structured_output(PriorityScoreOutput)

    result = s_llm.invoke(
        f"""Assign a priority score from 1-10 for this {state['category']} email.
        Consider:
        - Category: {state['category']}
        - Email content: {state['email']}

        Guidelines:
        - Urgent emails: usually 8-10
        - Normal emails: usually 4-7
        - Spam emails: usually 1-3"""
    )

    return {"priority_score": result.priority_score}
```

고정 매핑(`urgent=10, normal=5, spam=1`) 대신, LLM이 이메일 내용과 카테고리를 종합적으로 고려하여 1~10 범위 내에서 유동적인 점수를 부여한다. 프롬프트에 가이드라인을 포함하여 카테고리별 점수 범위를 안내한다.

**AI 기반 draft_response:**

```python
def draft_response(state: EmailState) -> EmailState:
    result = llm.invoke(
        f"""Draft a brief, professional response for this {state['category']} email.

        Original email: {state['email']}
        Category: {state['category']}
        Priority: {state['priority_score']}/10

        Guidelines:
        - Urgent: Acknowledge urgency, promise immediate attention
        - Normal: Professional acknowledgment, standard timeline
        - Spam: Brief notice that message was filtered

        Keep response under 2 sentences."""
    )
    return {
        "response": result.content,
    }
```

이 노드는 구조화 출력 없이 일반 LLM을 사용한다. 응답이 자유 형식 텍스트이기 때문이다. `result.content`로 LLM의 텍스트 응답에 접근한다.

**규칙 기반 vs AI 기반 비교:**

| 항목 | 규칙 기반 (17.1) | AI 기반 (17.4) |
|------|-----------------|----------------|
| 분류 방식 | 키워드 매칭 | 의미 기반 이해 |
| 점수 산정 | 카테고리별 고정값 | 컨텍스트 기반 유동값 |
| 응답 생성 | 고정 템플릿 | 동적 생성 |
| 결정성 | 결정적 (동일 입력 = 동일 출력) | 비결정적 (동일 입력이라도 다른 출력 가능) |
| 테스트 난이도 | 쉬움 (정확한 값 비교) | 어려움 (범위, 의미 비교 필요) |

#### 실습 포인트

1. "Please help me, my server is down and clients are complaining!"과 같이 "urgent" 키워드 없이도 긴급한 이메일을 테스트해 보라. 규칙 기반과 AI 기반의 결과 차이를 비교한다.
2. `EmailClassificationOuput`에 새로운 카테고리(예: "inquiry")를 추가하고 프롬프트를 수정해 보라.
3. `Field`의 `ge`, `le` 범위를 변경하여 Pydantic 검증이 작동하는지 확인해 보라.

---

### 17.5 Testing AI Nodes -- AI 노드에 맞는 테스트 전략

**주제 및 목표:** AI(LLM) 기반 노드의 비결정적 출력을 효과적으로 테스트하기 위한 전략을 학습한다. 정확한 값 비교에서 범위 기반 비교로 전환한다.

#### 핵심 개념 설명

AI 노드를 도입하면 기존 테스트가 깨진다. 그 이유:

1. **카테고리 분류:** LLM이 올바르게 분류하지만, 동일한 입력에도 미세한 해석 차이가 있을 수 있다.
2. **우선순위 점수:** 고정값(예: 10)이 아닌 범위(예: 8~10)로 반환된다.
3. **응답 텍스트:** 매번 다른 문장이 생성된다.

따라서 AI 노드 테스트의 핵심 원칙은 다음과 같다:

> **정확한 값(exact value) 대신 허용 범위(acceptable range)를 검증한다.**

#### 코드 분석

**환경 변수 로딩 추가:**

```python
import dotenv
dotenv.load_dotenv()
```

AI 노드가 OpenAI API를 호출하므로, `.env` 파일에서 `OPENAI_API_KEY`를 로드해야 한다. 이 코드가 **파일 최상단에** 위치하는 것이 중요하다. `from main import graph`가 실행될 때 `main.py`의 `init_chat_model("openai:gpt-4o")`이 호출되므로, 그 전에 환경 변수가 로드되어 있어야 한다.

**전체 그래프 테스트 -- 범위 기반으로 전환:**

```python
@pytest.mark.parametrize(
    "email, expected_category, min_score, max_score",
    [
        ("this is urgent!", "urgent", 8, 10),
        ("i wanna talk to you", "normal", 4, 7),
        ("i have an offer for you", "spam", 1, 3),
    ],
)
def test_full_graph(email, expected_category, min_score, max_score):
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}}
    )
    assert result["category"] == expected_category
    assert min_score <= result["priority_score"] <= max_score
```

변경 사항:
- `expected_score` 단일값 대신 `min_score`, `max_score` 범위를 사용
- `assert result["priority_score"] == expected_score` 대신 `assert min_score <= result["priority_score"] <= max_score`
- 카테고리는 Structured Output으로 강제되므로 여전히 정확한 값 비교 가능

**개별 노드 테스트 수정:**

```python
def test_individual_nodes():

    # categorize_email -- 여전히 정확한 값 비교 가능
    result = graph.nodes["categorize_email"].invoke(
        {"email": "check out this offer"}
    )
    assert result["category"] == "spam"

    # assing_priority -- 범위 비교로 전환, email 필드 추가
    result = graph.nodes["assing_priority"].invoke(
        {"category": "spam", "email": "buy this pot."}
    )
    assert 1 <= result["priority_score"] <= 3

    # draft_response -- 주석 처리 (아직 적절한 검증 방법이 없음)
    # result = graph.nodes["draft_response"].invoke({"category": "spam"})
    # assert "Go away" in result["response"]
```

주목할 점:
- `assing_priority`에 `email` 필드가 추가되었다. AI 버전은 프롬프트에서 이메일 내용도 참조하기 때문이다.
- `draft_response`는 주석 처리되었다. AI가 매번 다른 응답을 생성하므로 `"Go away" in result["response"]` 같은 키워드 검증이 불가능하다. 이 문제는 17.6에서 해결한다.

**부분 실행 테스트 수정:**

```python
def test_partial_execution():
    # ... (update_state 부분 동일) ...

    result = graph.invoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after="draft_response",
    )
    assert 1 <= result["priority_score"] <= 3  # 고정값 1 대신 범위
```

#### 실습 포인트

1. 범위를 일부러 좁게 설정(예: `min_score=10, max_score=10`)하여 AI 테스트가 얼마나 자주 실패하는지 관찰해 보라.
2. 동일한 테스트를 10번 연속 실행하여 결과의 분포를 확인해 보라: `pytest tests.py -v --count=10` (pytest-repeat 플러그인 필요)
3. `draft_response` 노드의 출력을 여러 번 확인하여, 어떤 검증 방법이 적절할지 고민해 보라.

---

### 17.6 Testing AI Responses -- LLM-as-a-Judge 패턴

**주제 및 목표:** 자유 형식 AI 응답의 품질을 검증하기 위해 **LLM-as-a-Judge(LLM을 심사자로 활용)** 패턴을 구현한다. 예시 기반 유사도 평가를 통해 AI 생성 텍스트를 테스트한다.

#### 핵심 개념 설명

17.5에서 `draft_response` 테스트가 주석 처리된 이유는, AI가 매번 다른 텍스트를 생성하기 때문이다. "Go away!"라는 정확한 문자열 매칭은 불가능하다.

**LLM-as-a-Judge** 패턴은 이 문제를 해결하는 대표적인 기법이다:

1. 카테고리별 **이상적인 응답 예시(golden examples)** 를 미리 정의한다.
2. 테스트 대상 AI 응답을 예시와 함께 **심사용 LLM**에게 전달한다.
3. 심사 LLM이 유사도 점수(similarity score)를 반환한다.
4. 점수가 임계값(threshold) 이상이면 테스트 통과로 판정한다.

이 패턴의 장점:
- 자유 형식 텍스트도 의미적으로 평가 가능
- 예시를 추가/수정하여 평가 기준을 쉽게 조정 가능
- 키워드 매칭보다 유연하고 견고함

#### 코드 분석

**유사도 점수 출력 스키마:**

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


class SimilarityScoreOutput(BaseModel):
    similarity_score: int = Field(
        description="How similar is the response to the examples?",
        gt=0,
        lt=100,
    )
```

심사 LLM이 0~100 사이의 유사도 점수를 반환하도록 스키마를 정의한다. `gt=0, lt=100`으로 0과 100을 제외한 1~99 범위를 강제한다.

**응답 예시 정의 (Golden Examples):**

```python
RESPONSE_EXAMPLES = {
    "urgent": [
        "Thank you for your urgent message. We are addressing this immediately and will respond as soon as possible.",
        "We've received your urgent request and are prioritizing it. Our team is on it right away.",
        "This urgent matter has our immediate attention. We'll respond promptly.",
    ],
    "normal": [
        "Thank you for your email. We'll review it and get back to you within 24-48 hours.",
        "We've received your message and will respond soon. Thank you for reaching out.",
        "Thank you for contacting us. We'll process your request and respond shortly.",
        "Thank you for the update. I will review the information and follow up as needed.",
        "Thank you for the update on the project status. I will review and follow up by the end of the week.",
        "Thanks for sharing this update. We'll review and respond accordingly.",
    ],
    "spam": [
        "This message has been flagged as spam and filtered.",
        "This email has been identified as promotional content.",
        "This message has been marked as spam.",
    ],
}
```

각 카테고리별로 "이런 식의 응답이 바람직하다"는 예시를 여러 개 제공한다. 예시가 다양할수록 심사의 정확도가 높아진다. `normal` 카테고리에 예시가 가장 많은 것은, 일반 이메일에 대한 응답이 가장 다양할 수 있기 때문이다.

**심사 함수 (Judge Function):**

```python
def judge_response(response: str, category: str):

    s_llm = llm.with_structured_output(SimilarityScoreOutput)

    examples = RESPONSE_EXAMPLES[category]
    result = s_llm.invoke(
        f"""
        Score how similar this response is to the examples.

        Category: {category}

        Examples:
        {"\n".join(examples)}

        Response to evaluate:
        {response}

        Scoring criteria:
        - 90-100: Very similar in tone, content, and intent
        - 70-89: Similar with minor differences
        - 50-69: Moderately similar, captures main idea
        - 30-49: Some similarity but missing key elements
        - 0-29: Very different or inappropriate
    """
    )

    return result.similarity_score
```

`judge_response` 함수의 동작:

1. 카테고리에 맞는 예시들을 가져온다.
2. 심사 LLM에게 예시와 평가 대상 응답을 함께 전달한다.
3. 명확한 평가 기준(rubric)을 프롬프트에 포함한다.
4. 구조화 출력으로 정수 점수를 반환받는다.

**테스트 코드에서 활용:**

```python
def test_individual_nodes():

    # ... (categorize_email, assing_priority 테스트 동일) ...

    # draft_response -- LLM-as-a-Judge로 검증
    result = graph.nodes["draft_response"].invoke(
        {
            "category": "spam",
            "email": "Get rich quick!!! I have a pyramid scheme for you!",
            "priority_score": 1,
        }
    )

    similarity_score = judge_response(result["response"], "spam")
    assert similarity_score >= 70
```

`draft_response` 노드에 완전한 상태(category, email, priority_score 모두 포함)를 전달한 뒤, 생성된 응답을 `judge_response`에 넘겨 유사도를 평가한다. 임계값 70 이상이면 테스트 통과이다.

**임계값 70의 의미:**
- 평가 기준표에 따르면 70~89는 "유사하되 약간의 차이가 있음"
- 너무 높으면(예: 90) 미세한 표현 차이로 테스트가 불안정해짐
- 너무 낮으면(예: 40) 품질이 낮은 응답도 통과시킴
- 70은 "의도와 톤이 맞지만 표현이 다를 수 있음"을 허용하는 균형점

#### 실습 포인트

1. 임계값을 50, 70, 90으로 각각 설정하여 테스트 안정성의 변화를 관찰해 보라.
2. `RESPONSE_EXAMPLES`에 예시를 추가/제거하여 심사 결과가 어떻게 변하는지 확인해 보라.
3. `judge_response`를 `urgent`와 `normal` 카테고리에도 적용하는 테스트를 추가해 보라.
4. 심사 LLM을 다른 모델(예: `gpt-4o-mini`)로 교체하여 비용과 정확도 트레이드오프를 실험해 보라.

---

## 3. 챕터 핵심 정리

### 테스트 전략 진화 과정

```
17.1 규칙 기반 그래프    -->  17.2 정확한 값 비교 테스트
        |                              |
17.4 AI 기반 그래프      -->  17.5 범위 기반 테스트
        |                              |
                             17.6 LLM-as-a-Judge 테스트
```

### 핵심 원칙 요약

| 원칙 | 설명 |
|------|------|
| **테스트 수준 분리** | 전체 그래프, 개별 노드, 부분 실행을 각각 테스트한다. |
| **결정적 출력은 정확히 비교** | Structured Output으로 강제한 카테고리 등은 `==`로 비교한다. |
| **비결정적 출력은 범위로 비교** | LLM이 생성하는 숫자는 `min <= value <= max`로 검증한다. |
| **자유 형식 텍스트는 LLM이 심사** | 다른 LLM을 심사자로 활용하여 의미적 유사도를 평가한다. |
| **Golden Examples** | 이상적인 응답 예시를 미리 정의하여 평가 기준으로 사용한다. |
| **프롬프트에 평가 기준 명시** | 심사 LLM에게 점수 구간별 의미를 명확히 전달한다. |
| **체크포인터로 부분 실행** | `MemorySaver`와 `update_state()`로 특정 지점부터 테스트할 수 있다. |

### 기술 스택 정리

| 기술 | 용도 |
|------|------|
| `langgraph.StateGraph` | 워크플로우 그래프 정의 |
| `langgraph.checkpoint.memory.MemorySaver` | 인메모리 상태 체크포인팅 |
| `pytest` + `@pytest.mark.parametrize` | 매개변수화 테스트 |
| `pydantic.BaseModel` + `Field` | LLM 출력 스키마 정의 및 검증 |
| `langchain.chat_models.init_chat_model` | LLM 초기화 |
| `llm.with_structured_output()` | 구조화 출력 강제 |

---

## 4. 실습 과제

### 과제 1: 새로운 카테고리 추가 (난이도: 중)

이메일 분류에 `"inquiry"` (문의) 카테고리를 추가하라.

- `EmailState`의 `category` Literal에 `"inquiry"` 추가
- `EmailClassificationOuput`에도 반영
- 분류 프롬프트에 "inquiry: questions or information requests" 가이드라인 추가
- `PriorityScoreOutput`의 프롬프트에 "Inquiry emails: usually 5-7" 추가
- `RESPONSE_EXAMPLES`에 `"inquiry"` 카테고리 예시 3개 이상 추가
- 새로운 카테고리에 대한 테스트 케이스를 `test_full_graph`의 `parametrize`에 추가

### 과제 2: 조건 분기 그래프 (난이도: 중)

스팸 이메일은 `draft_response` 노드를 건너뛰고 바로 종료되도록 그래프를 수정하라.

- `add_conditional_edges`를 사용하여 `assing_priority` 이후 카테고리에 따라 분기
- 스팸이면 END로, 그 외에는 `draft_response`로 이동
- 이 변경에 맞게 테스트 코드를 업데이트하라. 스팸 이메일의 경우 `response` 필드가 없어야 한다.

### 과제 3: 심사 LLM 고도화 (난이도: 상)

현재의 `judge_response`를 개선하라.

- 단일 유사도 점수 대신, 여러 차원(톤, 전문성, 적절성, 길이)을 각각 평가하는 스키마를 만들어라.
- 각 차원별 점수를 평균내어 최종 점수를 산출하라.
- 차원별로 가중치를 다르게 적용할 수 있게 하라 (예: 적절성 40%, 톤 30%, 전문성 20%, 길이 10%).
- 테스트 실패 시 어떤 차원에서 점수가 낮았는지 출력되도록 하라.

### 과제 4: 테스트 안정성 분석 (난이도: 상)

동일한 테스트를 20번 반복 실행하여 AI 테스트의 안정성을 분석하라.

- `pytest-repeat` 플러그인을 설치한다.
- `pytest tests.py --count=20 -v`로 실행한다.
- 각 테스트의 통과/실패 비율을 집계한다.
- 실패한 케이스가 있다면 원인을 분석한다 (임계값 문제인지, 프롬프트 문제인지, 모델 문제인지).
- 테스트 안정성을 95% 이상으로 올리기 위해 어떤 조정이 필요한지 보고서를 작성하라.
