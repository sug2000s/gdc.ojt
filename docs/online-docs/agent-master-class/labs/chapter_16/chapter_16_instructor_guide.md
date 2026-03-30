# Chapter 16: 워크플로우 아키텍처 패턴 — 강사용 해설서

> 이 문서는 강사가 Chapter 16을 **미리 이해**하기 위한 해설서입니다.
> 강의 대본(lecture_script)과 별개로, 코드를 한 줄 한 줄 쉽게 풀어서 설명합니다.

---

## 전체 패턴 흐름 (먼저 큰 그림)

```
16.1  START → A → B → C → END                       "순차 체이닝 — 릴레이 경주"
       +
16.2  START → A → [검사] → Pass → B → C → END       "게이트 — 품질 검사원"
                    ↑        Fail ─┘
       +
16.3  START → 분류 → easy/medium/hard → END          "라우팅 — 접수 창구"
       +
16.4  START → A,B,C,D (동시) → 합침 → END            "병렬 — 팀 프로젝트"
       +
16.5  START → 기획자 → Send(N명) → 합침 → END        "오케스트레이터 — 회사 조직"
```

각 섹션이 **독립적인 패턴**이므로, 어느 것부터 가르쳐도 됩니다.
다만 16.2는 16.1의 확장이고, 16.5는 16.4의 심화이므로 순서대로가 자연스럽습니다.

---

## 비유로 이해하는 5가지 패턴

| 패턴 | 비유 |
|------|------|
| **16.1 Prompt Chaining** | 식당 주방: 재료 준비 → 조리 → 플레이팅. 앞 사람이 끝나야 뒷 사람이 시작 |
| **16.2 Chaining + Gate** | 식당 주방 + 품질 검사: 재료가 기준 미달이면 다시 준비 |
| **16.3 Routing** | 병원 접수: 증상 경중에 따라 일반의/전문의/응급실로 배정 |
| **16.4 Parallelization** | 팀 프로젝트: 4명이 동시에 각자 파트 작업 → 모이면 합침 |
| **16.5 Orchestrator-Workers** | 회사 조직: 팀장이 업무 분배 → 팀원들이 각자 수행 → 팀장이 취합 보고 |

---

## 16.1 Prompt Chaining — "순차적 릴레이"

### 전체 흐름 그림

```
START → list_ingredients → create_recipe → describe_plating → END
  │          │                  │                │
  │    요리명 받아서        재료 목록 받아서     레시피 받아서
  │    재료 목록 생성       조리법 생성         플레이팅 설명 생성
  │          │                  │                │
  │     [구조화 출력]       [일반 텍스트]      [일반 텍스트]
```

### import 정리

```python
import os
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
```

| import | 뭐하는 놈 |
|--------|----------|
| `os` | 환경변수(API 키 등)를 읽기 위한 파이썬 기본 모듈 |
| `List` | 타입 힌트용. `List[Ingredient]`처럼 "Ingredient의 리스트"를 표현 |
| `TypedDict` | 딕셔너리에 타입을 지정. 상태(State) 정의에 사용 |
| `StateGraph` | 그래프(워크플로우)를 만드는 설계도 |
| `START, END` | 시작점, 끝점 (특수 노드) |
| `init_chat_model` | LLM을 초기화하는 함수. `"openai:모델명"` 형식 |
| `BaseModel` | Pydantic 모델. LLM 응답을 **구조화된 형태**로 강제할 때 사용 |

### LLM 초기화

```python
llm = init_chat_model(f"openai:{os.getenv('OPENAI_MODEL_NAME')}")
```

`.env`에서 모델명(gpt-5.1) 읽어서 LLM 객체 생성. 이게 우리의 AI 두뇌.

### 상태 정의 — TypedDict

```python
class State(TypedDict):
    dish: str                    # 입력: 요리 이름 (예: "hummus")
    ingredients: list[dict]      # 1단계 출력: 재료 목록
    recipe_steps: str            # 2단계 출력: 조리법
    plating_instructions: str    # 3단계 출력: 플레이팅
```

Chapter 14에서는 `MessagesState`(채팅 전용)를 썼지만, 여기서는 `TypedDict`를 직접 정의한다.
왜? 챗봇이 아니라 **데이터 파이프라인**이기 때문. 각 단계가 다른 필드에 결과를 저장.

> **비유**: 식당 주문표. `dish`란에 요리명 쓰면, 각 단계가 자기 칸을 채워넣는 구조.

### 구조화된 출력 스키마 — Pydantic BaseModel

```python
class Ingredient(BaseModel):
    name: str        # 재료 이름 (예: "Chickpeas")
    quantity: str    # 양 (예: "1.5")
    unit: str        # 단위 (예: "cups")
```

Pydantic `BaseModel`은 "LLM 응답이 반드시 이 형태여야 해!"라고 강제하는 틀.
LLM이 자유 텍스트로 답하면 파싱이 어렵다. `BaseModel`로 정의하면 **JSON 형태로 강제**.

```python
class IngredientsOutput(BaseModel):
    ingredients: List[Ingredient]
```

`IngredientsOutput`은 `Ingredient`의 리스트를 감싸는 wrapper.
LLM이 `{"ingredients": [{"name": "...", "quantity": "...", "unit": "..."}, ...]}` 형태로 응답하게 됨.

### 노드 함수 1: list_ingredients — 재료 목록 생성

```python
def list_ingredients(state: State):
    structured_llm = llm.with_structured_output(IngredientsOutput)
    response = structured_llm.invoke(
        f"List 5-8 ingredients needed to make {state['dish']}"
    )
    return {"ingredients": [i.model_dump() for i in response.ingredients]}
```

한 줄씩:

1. `llm.with_structured_output(IngredientsOutput)` — LLM에게 "응답을 `IngredientsOutput` 형태로 줘"라고 지시. 이게 **구조화된 출력의 핵심**.
2. `structured_llm.invoke(...)` — 프롬프트를 보내면, LLM이 `IngredientsOutput` 객체로 응답.
3. `response.ingredients` — Pydantic 객체의 리스트. 각각 `Ingredient` 타입.
4. `i.model_dump()` — Pydantic 객체 → 딕셔너리 변환. `{"name": "Chickpeas", "quantity": "1.5", "unit": "cups"}`
5. `return {"ingredients": [...]}` — State의 `ingredients` 필드에 저장.

> **왜 `with_structured_output`을 쓰나?**
> 일반 `invoke()`는 자유 텍스트 반환. "Chickpeas 1.5 cups, Tahini 1/4 cup..." 이런 식.
> `with_structured_output`은 JSON으로 강제하니까 **프로그래밍으로 다루기 쉽다**.

### 노드 함수 2: create_recipe — 조리법 생성

```python
def create_recipe(state: State):
    response = llm.invoke(
        f"Write a step by step cooking instruction for {state['dish']}, "
        f"using these ingredients {state['ingredients']}"
    )
    return {"recipe_steps": response.content}
```

1. `state['ingredients']` — 이전 단계(list_ingredients)에서 생성한 재료 목록을 가져옴.
2. `llm.invoke(...)` — 일반 텍스트 응답. 여기서는 구조화 출력이 필요 없다 (사람이 읽을 텍스트니까).
3. `response.content` — AI 응답의 텍스트 부분.
4. `return {"recipe_steps": ...}` — State의 `recipe_steps` 필드에 저장.

> **포인트**: 이전 단계의 출력(`ingredients`)이 다음 단계의 입력이 된다. 이게 **체이닝**!

### 노드 함수 3: describe_plating — 플레이팅 설명

```python
def describe_plating(state: State):
    response = llm.invoke(
        f"Describe how to beautifully plate this dish {state['dish']} "
        f"based on this recipe {state['recipe_steps']}"
    )
    return {"plating_instructions": response.content}
```

create_recipe와 같은 구조. `recipe_steps`를 받아서 플레이팅 설명을 생성.

### 그래프 구성 & 실행

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_edge("list_ingredients", "create_recipe")
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

1. `StateGraph(State)` — "이 그래프는 State를 공유합니다"라는 선언.
2. `add_node("이름", 함수)` — 노드 등록. 이름은 문자열, 함수는 실제 로직.
3. `add_edge(A, B)` — A 끝나면 B 실행. 단방향 화살표.
4. `compile()` — 설계도 → 실행 가능한 그래프로 변환.

```python
result = graph.invoke({"dish": "hummus"})
```

초기 입력으로 `dish`만 넣으면, 나머지 3개 필드(`ingredients`, `recipe_steps`, `plating_instructions`)는 각 노드가 채워넣음.

> **비유**: 빈 주문표에 "hummus"만 적으면, 주방 라인에서 한 명씩 자기 칸을 채우는 것.

---

## 16.2 Prompt Chaining + Gate — "품질 검사관 추가"

### 전체 흐름 그림

```
START → list_ingredients → [gate: 3~8개?]
                               │
                          True (통과) → create_recipe → describe_plating → END
                               │
                          False (실패) → list_ingredients (재시도!)
                               ↑_______________|
```

16.1과 동일하되, **게이트(gate) 함수** 하나가 추가됨.
게이트는 "이전 단계 결과가 괜찮은가?" 검사하는 품질 관리자.

### 달라진 부분만 설명 (16.1과 동일한 코드는 생략)

### 게이트 함수 — 재료 개수 검증

```python
def gate(state: State):
    ingredients = state["ingredients"]
    count = len(ingredients)
    if count > 8 or count < 3:
        print(f"  GATE FAIL: {count} ingredients (need 3-8). Retrying...")
        return False
    print(f"  GATE PASS: {count} ingredients")
    return True
```

1. `state["ingredients"]` — 이전 노드(list_ingredients)가 생성한 재료 목록.
2. `len(ingredients)` — 재료 개수 세기.
3. 3~8개 범위 밖이면 → `False` 반환 (불합격).
4. 범위 안이면 → `True` 반환 (합격).

> **비유**: 식당 품질 검사관. "재료가 너무 적거나 너무 많으면 다시 준비해!"

### 조건부 엣지 — add_conditional_edges

```python
graph_builder.add_conditional_edges(
    "list_ingredients",
    gate,
    {
        True: "create_recipe",       # 통과 → 다음 단계
        False: "list_ingredients",   # 실패 → 재시도
    },
)
```

이게 16.2의 핵심!

1. 첫 번째 인자 `"list_ingredients"` — 이 노드가 끝나면...
2. 두 번째 인자 `gate` — 이 함수를 실행해서 결과를 봐.
3. 세 번째 인자 (딕셔너리) — 결과에 따라 어디로 갈지 매핑.
   - `gate()`가 `True` 반환 → `"create_recipe"` 노드로
   - `gate()`가 `False` 반환 → `"list_ingredients"` 노드로 (자기 자신! = 재시도)

> **주의**: `False → "list_ingredients"`이므로 **무한 루프 가능**!
> 실무에서는 반드시 `retry_count` 같은 카운터로 최대 재시도 횟수를 제한해야 한다.

### 실행 결과 예시

```
  Generated 8 ingredients
  GATE PASS: 8 ingredients       ← 8개니까 통과!

Final: 8 ingredients
```

만약 LLM이 10개를 생성했다면:

```
  Generated 10 ingredients
  GATE FAIL: 10 ingredients (need 3-8). Retrying...
  Generated 6 ingredients
  GATE PASS: 6 ingredients       ← 재시도 후 통과
```

---

## 16.3 Routing — "접수 창구에서 배정"

### 전체 흐름 그림

```
                              ┌→ dumb_node (쉬운 모델) → END
START → assess_difficulty → ─┤→ average_node (중간 모델) → END
                              └→ smart_node (어려운 모델) → END
```

질문의 난이도를 LLM이 판단해서, 난이도별 다른 모델(노드)로 보내는 패턴.

### 새로운 import

```python
from typing import Literal
from langgraph.types import Command
```

| import | 뭐하는 놈 |
|--------|----------|
| `Literal` | 값을 특정 선택지로 **제한**. `Literal["easy","medium","hard"]` = 이 3개 중 하나만 허용 |
| `Command` | LangGraph 명령 객체. `goto=`로 **다음에 갈 노드를 직접 지정** |

### 상태 정의

```python
class State(TypedDict):
    question: str      # 입력: 질문
    difficulty: str    # 판별 결과: easy/medium/hard
    answer: str        # 출력: 답변
    model_used: str    # 어떤 모델이 답했는지 기록
```

### Literal로 LLM 응답 제한 — DifficultyResponse

```python
class DifficultyResponse(BaseModel):
    difficulty_level: Literal["easy", "medium", "hard"]
```

`Literal["easy", "medium", "hard"]` — LLM이 이 3개 값 **중 하나만** 반환할 수 있다.
"kinda hard"나 "very easy" 같은 애매한 답이 불가능. 프로그래밍에서 다루기 깔끔.

> **비유**: 병원 접수 양식에 "경증/중증/응급" 3개 체크박스만 있는 것.

### 분류 노드 — assess_difficulty

```python
def assess_difficulty(state: State):
    structured_llm = llm.with_structured_output(DifficultyResponse)
    response = structured_llm.invoke(
        f"""
        Assess the difficulty of this question:
        Question: {state['question']}

        - EASY: Simple facts, basic definitions, yes/no answers
        - MEDIUM: Requires explanation, comparison, analysis
        - HARD: Complex reasoning, multiple steps, deep expertise
        """
    )
    level = response.difficulty_level
    goto_map = {"easy": "dumb_node", "medium": "average_node", "hard": "smart_node"}
    print(f"  Difficulty: {level} → {goto_map[level]}")
    return Command(goto=goto_map[level], update={"difficulty": level})
```

한 줄씩:

1. `with_structured_output(DifficultyResponse)` — LLM 응답을 `DifficultyResponse` 형태로 강제.
2. 프롬프트에 분류 기준(EASY/MEDIUM/HARD)을 명확히 제시 — LLM이 판단 근거로 사용.
3. `response.difficulty_level` — `"easy"`, `"medium"`, `"hard"` 중 하나.
4. `goto_map` — 난이도 → 노드 이름 매핑 딕셔너리.
5. `Command(goto=..., update=...)` — **이게 라우팅의 핵심!**
   - `goto=` — 다음에 실행할 노드를 **직접 지정**
   - `update=` — State에 값을 업데이트 (difficulty 필드 저장)

> **16.2의 `add_conditional_edges`와 뭐가 다른가?**
> - `add_conditional_edges`: 그래프 **설계 시** 분기를 정의 (엣지 기반)
> - `Command(goto=)`: 노드 **실행 중에** 다음 노드를 동적 결정 (노드 내부에서 결정)
> 둘 다 라우팅이지만, `Command`가 더 유연하다.

### 모델별 처리 노드

```python
def dumb_node(state: State):
    response = dumb_llm.invoke(state["question"])
    return {"answer": response.content, "model_used": "gpt-3.5 (simulated)"}

def average_node(state: State):
    response = average_llm.invoke(state["question"])
    return {"answer": response.content, "model_used": "gpt-4o (simulated)"}

def smart_node(state: State):
    response = smart_llm.invoke(state["question"])
    return {"answer": response.content, "model_used": "gpt-5 (simulated)"}
```

3개 노드 모두 같은 구조. 실무에서는 각각 **다른 모델**(저비용/중간/고성능)을 사용.
여기서는 학습 목적으로 같은 모델을 사용하되, `model_used`에 다른 이름을 기록.

### 그래프 구성 — destinations 주목

```python
graph_builder.add_node(
    "assess_difficulty", assess_difficulty,
    destinations=("dumb_node", "average_node", "smart_node"),
)
```

`destinations=` — `Command(goto=)`로 갈 수 있는 노드 목록을 **미리 선언**.
LangGraph가 그래프 시각화할 때 이 정보를 사용. 없어도 동작하지만, 있으면 더 명확.

```python
graph_builder.add_edge(START, "assess_difficulty")
graph_builder.add_edge("dumb_node", END)
graph_builder.add_edge("average_node", END)
graph_builder.add_edge("smart_node", END)
```

3개 노드 모두 → END. 어떤 경로든 한 번만 처리하고 끝.

### 실행 예시

```python
# 쉬운 질문
result = graph.invoke({"question": "What is the capital of France?"})
# → Difficulty: easy → dumb_node
# → Model: gpt-3.5 (simulated)

# 어려운 질문
result = graph.invoke({"question": "Explain the economic implications of quantum computing on global supply chains"})
# → Difficulty: hard → smart_node
# → Model: gpt-5 (simulated)
```

> **실무 활용**: 쉬운 질문은 저렴한 모델, 어려운 질문은 비싼 모델 → **비용 최적화**.

---

## 16.4 Parallelization — "팀 프로젝트, 동시 작업"

### 전체 흐름 그림

```
        ┌→ get_summary ────────────┐
        │→ get_sentiment ──────────┤
START → │→ get_key_points ─────────┤→ get_final_analysis → END
        └→ get_recommendation ─────┘
         (4개 동시 실행!)         (모두 끝나면 합침)
```

Fan-out / Fan-in 패턴:
- **Fan-out**: START에서 4개 노드로 **동시에** 퍼짐
- **Fan-in**: 4개 노드가 **모두 끝나면** 하나로 모임

### import 정리

```python
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
```

16.1과 동일. 새로운 import 없음! LangGraph의 기본 엣지만으로 병렬이 가능.

### 상태 정의 — 각 노드가 다른 필드에 쓴다

```python
class State(TypedDict):
    document: str          # 입력: 분석할 문서
    summary: str           # get_summary가 채움
    sentiment: str         # get_sentiment가 채움
    key_points: str        # get_key_points가 채움
    recommendation: str    # get_recommendation가 채움
    final_analysis: str    # get_final_analysis가 채움 (합산)
```

**핵심**: 4개 병렬 노드가 각각 **다른 필드**에 쓴다. 충돌 없음!

> **비유**: 4명이 같은 보고서의 다른 칸을 동시에 채우는 것. 서로 겹치지 않으니 문제 없다.

### 4개의 병렬 분석 노드

```python
def get_summary(state: State):
    print("  [parallel] get_summary started")
    response = llm.invoke(f"Write a 3-sentence summary of this document: {state['document'][:500]}")
    return {"summary": response.content}

def get_sentiment(state: State):
    print("  [parallel] get_sentiment started")
    response = llm.invoke(f"Analyse the sentiment and tone of this document: {state['document'][:500]}")
    return {"sentiment": response.content}

def get_key_points(state: State):
    print("  [parallel] get_key_points started")
    response = llm.invoke(f"List the 5 most important points of this document: {state['document'][:500]}")
    return {"key_points": response.content}

def get_recommendation(state: State):
    print("  [parallel] get_recommendation started")
    response = llm.invoke(f"Based on the document, list 3 recommended next steps: {state['document'][:500]}")
    return {"recommendation": response.content}
```

4개 함수 모두 같은 패턴:
1. `state['document'][:500]` — 같은 문서를 읽음 (입력 공유)
2. 각자 다른 프롬프트로 LLM 호출
3. 각자 **다른 필드**에 결과 저장 (`summary`, `sentiment`, `key_points`, `recommendation`)

> **왜 병렬이 가능한가?**
> 4개 노드가 같은 입력(`document`)을 **읽기만** 하고, 각자 **다른 필드에 쓴다**.
> 서로 의존하지 않으니 동시 실행 가능.

### 집계 노드 — get_final_analysis

```python
def get_final_analysis(state: State):
    print("  [join] get_final_analysis started")
    response = llm.invoke(
        f"""
    Give me a brief analysis combining these results:

    SUMMARY: {state['summary']}
    SENTIMENT: {state['sentiment']}
    KEY POINTS: {state['key_points']}
    RECOMMENDATIONS: {state['recommendation']}
    """
    )
    return {"final_analysis": response.content}
```

4개 노드가 **모두 완료된 후에만** 실행됨 (LangGraph가 자동으로 대기).
4개의 결과를 모아서 하나의 종합 분석을 생성.

### 그래프 구성 — 병렬의 비밀

```python
# START에서 4개 노드로 = 자동 병렬!
graph_builder.add_edge(START, "get_summary")
graph_builder.add_edge(START, "get_sentiment")
graph_builder.add_edge(START, "get_key_points")
graph_builder.add_edge(START, "get_recommendation")
```

**하나의 노드(START)에서 여러 노드로 엣지를 연결하면 → 자동 병렬 실행!**

특별한 API가 필요 없다. `add_edge`만으로 병렬이 된다.

```python
# 4개 모두 → 하나로 = join (모두 끝날 때까지 대기)
graph_builder.add_edge("get_summary", "get_final_analysis")
graph_builder.add_edge("get_sentiment", "get_final_analysis")
graph_builder.add_edge("get_key_points", "get_final_analysis")
graph_builder.add_edge("get_recommendation", "get_final_analysis")

graph_builder.add_edge("get_final_analysis", END)
```

**여러 노드에서 하나의 노드로 엣지를 연결하면 → 모두 끝날 때까지 대기(join)!**

> **핵심 정리**:
> - 1 → N 엣지 = **fan-out** (병렬 시작)
> - N → 1 엣지 = **fan-in** (모두 대기 후 합침)
> - 별도 API 필요 없음. 엣지 구조만으로 결정됨.

### 실행 결과

```
  [parallel] get_key_points started      ← 4개가 거의 동시에 시작!
  [parallel] get_recommendation started
  [parallel] get_sentiment started
  [parallel] get_summary started

  [join] get_final_analysis started      ← 4개 모두 끝난 후 시작
```

순차로 하면 4배 걸릴 작업이, 병렬로 하면 **가장 느린 노드 하나 시간**만 걸림.

---

## 16.5 Orchestrator-Workers — "팀장이 업무 분배"

### 전체 흐름 그림

```
START → orchestrator → [Send × N] → worker(1) ──┐
                                   → worker(2) ──┤→ synthesizer → END
                                   → worker(3) ──┤
                                   → worker(N) ──┘
                                   (동적 병렬!)
```

16.4와의 차이:
- 16.4: 병렬 노드 개수가 **코드에 고정** (4개)
- 16.5: 병렬 노드 개수가 **LLM이 동적으로 결정** (3개일 수도 5개일 수도)

### 새로운 import

```python
import operator
from typing import Annotated
from langgraph.types import Send
```

| import | 뭐하는 놈 |
|--------|----------|
| `operator` | `operator.add` — 리스트를 합치는 함수. 리듀서로 사용 |
| `Annotated` | 타입에 **추가 정보(리듀서)**를 붙이는 파이썬 문법 |
| `Send` | LangGraph의 **동적 디스패치 API**. 런타임에 노드를 생성하여 병렬 실행 |

### 상태 정의 — Annotated 리듀서 주목!

```python
class State(TypedDict):
    topic: str                                        # 입력: 연구 주제
    sections: list[str]                               # 오케스트레이터가 분할한 섹션 목록
    results: Annotated[list[dict], operator.add]      # 워커 결과 누적 ← 핵심!
    final_report: str                                 # 종합 보고서
```

**`Annotated[list[dict], operator.add]`가 핵심!**

- `list[dict]` — 타입은 딕셔너리의 리스트
- `operator.add` — **리듀서**. 여러 워커가 각자 결과를 반환하면 **덮어쓰기가 아니라 합침(append)**

> **비유**: 팀원들이 각자 보고서를 제출하면, 바인더에 계속 끼워넣는 것.
> 리듀서 없으면 마지막 팀원 보고서만 남는다!

이것은 Chapter 13에서 배운 개념의 실전 활용.

### 오케스트레이터 노드 — 주제를 섹션으로 분할

```python
def orchestrator(state: State):
    from pydantic import BaseModel
    from typing import List

    class Sections(BaseModel):
        sections: List[str]

    structured_llm = llm.with_structured_output(Sections)
    response = structured_llm.invoke(
        f"Break down this topic into 3-5 research sections: {state['topic']}"
    )
    print(f"  Orchestrator: {len(response.sections)} sections")
    for s in response.sections:
        print(f"    - {s}")
    return {"sections": response.sections}
```

한 줄씩:

1. `Sections(BaseModel)` — 함수 안에서 Pydantic 모델 정의. `sections: List[str]` (문자열 리스트).
2. `with_structured_output(Sections)` — LLM에게 "섹션 목록을 JSON으로 줘"라고 강제.
3. 프롬프트: "이 주제를 3~5개 연구 섹션으로 나눠줘"
4. `response.sections` — `["Clinical Applications", "Operations", ...]` 같은 리스트.
5. `return {"sections": response.sections}` — State에 저장.

> **비유**: 팀장이 "AI와 의료" 프로젝트를 받으면, "임상 적용", "운영 효율", "윤리/규제" 등으로 나누는 것.

### 워커 노드 — 개별 섹션 처리

```python
def worker(section: str):
    response = llm.invoke(f"Write a brief paragraph about: {section}")
    print(f"  Worker done: {section[:40]}...")
    return {"results": [{"section": section, "content": response.content}]}
```

**주의: 워커의 파라미터가 `state: State`가 아니라 `section: str`!**

Send API로 호출되는 워커는 **전체 State가 아니라, Send가 보낸 데이터만** 받는다.
`Send("worker", section)` → `worker(section)` 형태.

반환값: `{"results": [{"section": ..., "content": ...}]}` — 리듀서(`operator.add`)에 의해 누적.

### 디스패처 — Send API로 워커 병렬 실행

```python
def dispatch_workers(state: State):
    return [Send("worker", section) for section in state["sections"]]
```

**이 한 줄이 16.5의 가장 중요한 코드!**

1. `state["sections"]` — 오케스트레이터가 만든 섹션 목록 (예: 5개).
2. `Send("worker", section)` — "worker 노드를 section 데이터로 실행해줘"라는 명령.
3. 리스트 컴프리헨션으로 **섹션 개수만큼 Send 객체 생성**.
4. LangGraph가 이 Send들을 **병렬로 실행**.

> **비유**: 팀장이 "1번 주제는 김대리, 2번은 박대리, 3번은 이대리... 각자 알아서 해와" 하는 것.
> 몇 명한테 시킬지는 **팀장(오케스트레이터)이 결정**. 고정이 아님!

### 종합 노드 — synthesizer

```python
def synthesizer(state: State):
    sections_text = "\n\n".join(
        f"### {r['section']}\n{r['content']}" for r in state["results"]
    )
    response = llm.invoke(
        f"Combine these sections into a cohesive report:\n\n{sections_text}"
    )
    return {"final_report": response.content}
```

1. `state["results"]` — 모든 워커의 결과가 `operator.add`에 의해 합쳐진 리스트.
2. `"\n\n".join(...)` — 각 섹션을 마크다운 형태로 합침.
3. LLM에게 "이 섹션들을 하나의 일관된 보고서로 합쳐줘"라고 요청.
4. `return {"final_report": ...}` — 최종 보고서 저장.

### 그래프 구성 — conditional_edges + Send

```python
graph_builder.add_edge(START, "orchestrator")
graph_builder.add_conditional_edges("orchestrator", dispatch_workers, ["worker"])
graph_builder.add_edge("worker", "synthesizer")
graph_builder.add_edge("synthesizer", END)
```

한 줄씩:

1. `START → orchestrator` — 시작하면 오케스트레이터 실행.
2. `add_conditional_edges("orchestrator", dispatch_workers, ["worker"])` — 핵심!
   - `dispatch_workers` 함수가 `Send` 리스트를 반환
   - `["worker"]` — 가능한 목적지 노드 목록 (시각화용)
   - LangGraph가 Send 리스트를 받아서 **워커들을 병렬 실행**
3. `worker → synthesizer` — 모든 워커가 끝나면 synthesizer 실행.
4. `synthesizer → END` — 보고서 완성 후 종료.

### 실행 결과

```
  Orchestrator: 5 sections
    - Clinical Applications and Care Delivery
    - Health System Operations and Efficiency
    - Data, Algorithms, and Infrastructure
    - Ethical, Legal, and Regulatory Considerations
    - Public Health, Equity, and Global Implications

  Worker done: Clinical Applications and Care Delivery...
  Worker done: Health System Operations and Efficiency...
  Worker done: Data, Algorithms, and Infrastructure...
  Worker done: Public Health, Equity, and Global Implic...
  Worker done: Ethical, Legal, and Regulatory Considera...
```

오케스트레이터가 5개 섹션을 만들었으니, 워커 5개가 병렬 실행.
주제를 바꾸면 3개가 될 수도 4개가 될 수도 있다 — **동적**!

---

## 16.4 vs 16.5 비교

| 항목 | 16.4 Parallelization | 16.5 Orchestrator-Workers |
|------|---------------------|--------------------------|
| 병렬 노드 수 | **고정** (코드에 4개 하드코딩) | **동적** (LLM이 결정) |
| 병렬 방법 | `add_edge(START, 노드)` 반복 | `Send API` |
| 노드 종류 | 각각 다른 함수 | **같은 함수**를 다른 데이터로 |
| 결과 합산 | 각자 다른 필드에 저장 | `Annotated[list, operator.add]` 리듀서 |
| 비유 | 팀원 4명이 각자 다른 파트 | 팀장이 N명한테 같은 종류의 일 분배 |

---

## import 종합 정리표

| import | 사용 섹션 | 역할 |
|--------|----------|------|
| `os` | 전체 | 환경변수 읽기 |
| `TypedDict` | 전체 | 상태(State) 타입 정의 |
| `StateGraph` | 전체 | 그래프 설계도 생성 |
| `START, END` | 전체 | 시작/끝 특수 노드 |
| `init_chat_model` | 전체 | LLM 초기화 |
| `BaseModel` (Pydantic) | 16.1, 16.2, 16.3, 16.5 | 구조화된 출력 스키마 정의 |
| `List` (typing) | 16.1, 16.2, 16.5 | 타입 힌트 (`List[Ingredient]`) |
| `Literal` (typing) | 16.3 | 값을 특정 선택지로 제한 |
| `Command` | 16.3 | 노드 내부에서 다음 노드를 동적 지정 |
| `Send` | 16.5 | 런타임에 워커를 동적 병렬 디스패치 |
| `operator` | 16.5 | `operator.add` — 리스트 합치기 리듀서 |
| `Annotated` | 16.5 | 타입에 리듀서 정보 추가 |

---

## 핵심 개념 정리표

| 섹션 | 패턴 | 핵심 코드 | 비유 |
|------|------|----------|------|
| **16.1** | Prompt Chaining | `add_edge(A, B)` + `with_structured_output` | 식당 주방 라인 |
| **16.2** | Chaining + Gate | `add_conditional_edges` + `True/False` | 식당 + 품질 검사관 |
| **16.3** | Routing | `Literal` + `Command(goto=)` | 병원 접수 창구 |
| **16.4** | Parallelization | `START → 여러 노드` (fan-out/fan-in) | 팀 프로젝트 동시 작업 |
| **16.5** | Orchestrator-Workers | `Send API` + `operator.add` 리듀서 | 회사 조직 (팀장→팀원) |

---

## 강사를 위한 수업 팁

1. **16.1부터 시작**: 가장 단순. "이전 출력이 다음 입력"이라는 체이닝 개념만 이해시키면 됨.
2. **16.2는 16.1에 gate만 추가**: 코드 diff를 보여주면 효과적. "달라진 건 gate 함수 하나뿐".
3. **16.3은 독립적**: 16.1~16.2 없이도 설명 가능. `Command` vs `add_conditional_edges` 차이를 명확히.
4. **16.4는 시각적으로**: ASCII 그래프를 칠판에 그리고, "4개가 동시에!"를 강조.
5. **16.5는 16.4의 동적 버전**: "16.4에서 노드 수가 고정이었는데, 16.5는 LLM이 정한다"로 연결.
6. **비용 관점**: 16.3 라우팅은 실무에서 API 비용 최적화에 직결. 이 점을 강조하면 동기부여.
