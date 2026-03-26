# Chapter 16: 워크플로우 아키텍처 패턴 (Workflow Architecture Patterns)

---

## 1. 챕터 개요

이번 챕터에서는 AI 에이전트 시스템을 구축할 때 사용할 수 있는 핵심 **워크플로우 아키텍처 패턴**들을 학습한다. LangGraph를 활용하여 LLM 호출을 체계적으로 조합하는 다양한 방법을 실습하며, 각 패턴이 어떤 상황에서 적합한지 이해하는 것이 목표이다.

### 다루는 아키텍처 패턴

| 섹션 | 패턴 | 핵심 개념 |
|------|------|-----------|
| 16.0 | Introduction | 프로젝트 환경 구성 |
| 16.1 | Prompt Chaining | 순차적 LLM 호출 체인 |
| 16.2 | Prompt Chaining Gate | 조건부 분기(게이트) |
| 16.3 | Routing | 입력 기반 동적 라우팅 |
| 16.4 | Parallelization | 병렬 실행 및 결과 집계 |
| 16.5 | Orchestrator-Workers | 동적 작업 분배 (Map-Reduce) |

### 사용 기술 스택

- **Python 3.13**
- **LangGraph 0.6.6** -- 워크플로우 그래프 구성 프레임워크
- **LangChain 0.3.27** -- LLM 통합 레이어
- **OpenAI GPT-4o** -- 주요 LLM 모델
- **Pydantic** -- 구조화된 출력(Structured Output) 정의

---

## 2. 섹션별 상세 설명

---

### 16.0 Introduction -- 프로젝트 환경 구성

#### 주제 및 목표

새로운 프로젝트 `workflow-architectures`를 생성하고, LangGraph 기반 워크플로우 실험을 위한 개발 환경을 설정한다.

#### 핵심 개념 설명

이 섹션에서는 `uv` 패키지 매니저를 사용하여 Python 프로젝트를 초기화한다. `pyproject.toml`에 모든 의존성이 선언되어 있으며, Jupyter Notebook(`main.ipynb`)을 실행 환경으로 사용한다.

#### 코드 분석

**프로젝트 의존성 (`pyproject.toml`):**

```toml
[project]
name = "workflow-architectures"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "python-dotenv==1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

주요 의존성 설명:
- **`langgraph`**: 상태 기반 워크플로우 그래프를 구성하는 핵심 라이브러리. 노드(node)와 엣지(edge)로 LLM 호출 흐름을 정의한다.
- **`langchain[openai]`**: OpenAI 모델과의 통합을 제공한다. `init_chat_model()`로 다양한 모델을 초기화할 수 있다.
- **`grandalf`**: 그래프 시각화를 위한 라이브러리.
- **`ipykernel`**: Jupyter Notebook에서 가상 환경의 커널을 사용하기 위한 개발 의존성.

#### 실습 포인트

1. `uv init`으로 프로젝트를 생성한 후, `uv add`로 의존성을 추가하는 방법을 익힌다.
2. `.python-version` 파일로 Python 버전을 고정하는 패턴을 이해한다.
3. `.env` 파일에 `OPENAI_API_KEY`를 설정해야 LLM 호출이 정상 작동한다.

---

### 16.1 Prompt Chaining Architecture -- 순차적 프롬프트 체이닝

#### 주제 및 목표

가장 기본적인 워크플로우 패턴인 **Prompt Chaining**을 구현한다. 여러 LLM 호출을 **순차적으로 연결**하여, 이전 단계의 출력이 다음 단계의 입력이 되는 파이프라인을 만든다.

#### 핵심 개념 설명

**Prompt Chaining**은 복잡한 작업을 여러 개의 작은 단계로 분해하는 패턴이다. 각 단계는 하나의 LLM 호출로 처리되며, 결과가 상태(State)에 저장되어 다음 단계로 전달된다.

이 예제에서는 요리 레시피 생성 과정을 3단계로 분해한다:
1. **재료 나열** (list_ingredients) -- 요리에 필요한 재료 목록 생성
2. **레시피 작성** (create_recipe) -- 재료를 기반으로 조리법 생성
3. **플레이팅 설명** (describe_plating) -- 레시피를 기반으로 플레이팅 방법 설명

```
START --> list_ingredients --> create_recipe --> describe_plating --> END
```

이 패턴의 핵심은 **각 단계가 이전 단계의 결과에 의존**한다는 것이다. 재료를 모르면 레시피를 쓸 수 없고, 레시피를 모르면 플레이팅을 설명할 수 없다.

#### 코드 분석

**1단계: 상태(State) 및 데이터 모델 정의**

```python
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

llm = init_chat_model("openai:gpt-4o")
```

`init_chat_model()`은 LangChain에서 제공하는 유니버설 모델 초기화 함수이다. `"openai:gpt-4o"` 형식으로 프로바이더와 모델명을 지정한다.

```python
class State(TypedDict):
    dish: str
    ingredients: list[dict]
    recipe_steps: str
    plating_instructions: str
```

`State`는 워크플로우 전체에서 공유되는 **상태 객체**이다. `TypedDict`를 사용하여 각 필드의 타입을 명시한다. LangGraph에서 모든 노드는 이 State를 읽고 업데이트한다.

```python
class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: str

class IngredientsOutput(BaseModel):
    ingredients: List[Ingredient]
```

Pydantic `BaseModel`을 사용하여 **구조화된 출력(Structured Output)**을 정의한다. LLM이 자유 텍스트 대신 정해진 스키마에 맞는 JSON을 반환하도록 강제할 수 있다.

**2단계: 노드 함수 정의**

```python
def list_ingredients(state: State):
    structured_llm = llm.with_structured_output(IngredientsOutput)
    response = structured_llm.invoke(
        f"List 5-8 ingredients needed to make {state['dish']}"
    )
    return {
        "ingredients": response.ingredients,
    }
```

`with_structured_output(IngredientsOutput)`는 LLM의 응답을 `IngredientsOutput` Pydantic 모델로 자동 파싱하도록 설정한다. 이렇게 하면 LLM이 `{"ingredients": [{"name": "Chickpeas", "quantity": "1", "unit": "cup"}, ...]}` 형태의 구조화된 데이터를 반환한다.

각 노드 함수는 반드시 **딕셔너리를 반환**해야 한다. 반환된 딕셔너리의 키-값 쌍이 State에 업데이트된다.

```python
def create_recipe(state: State):
    response = llm.invoke(
        f"Write a step by step cooking instruction for {state['dish']}, "
        f"using these ingredients {state['ingredients']}",
    )
    return {
        "recipe_steps": response.content,
    }

def describe_plating(state: State):
    response = llm.invoke(
        f"Describe how to beautifully plate this dish {state['dish']} "
        f"based on this recipe {state['recipe_steps']}"
    )
    return {
        "plating_instructions": response.content,
    }
```

`create_recipe`는 `state['ingredients']`를 참조하고, `describe_plating`은 `state['recipe_steps']`를 참조한다. 이것이 바로 **체이닝의 핵심** -- 이전 단계의 출력이 다음 단계의 프롬프트에 포함된다.

**3단계: 그래프 구성 및 실행**

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

`StateGraph(State)`로 그래프 빌더를 생성하고, `add_node()`로 노드를 등록한 뒤, `add_edge()`로 노드 간 연결을 정의한다. `START`와 `END`는 LangGraph에서 제공하는 특수 노드로, 그래프의 시작과 끝을 나타낸다.

```python
graph.invoke({"dish": "hummus"})
```

`graph.invoke()`에 초기 State를 전달하면 그래프가 실행된다. `dish` 값만 제공하면 나머지 필드는 각 노드가 순차적으로 채워나간다.

#### 실습 포인트

1. `graph.invoke()`의 반환값을 확인하여 각 필드가 어떻게 채워지는지 관찰한다.
2. `dish` 값을 바꿔보며 다양한 요리에 대한 결과를 비교해본다.
3. `with_structured_output()`을 사용한 노드와 일반 `invoke()`를 사용한 노드의 차이를 이해한다.

---

### 16.2 Prompt Chaining Gate -- 조건부 게이트

#### 주제 및 목표

Prompt Chaining에 **조건부 분기(Gate)**를 추가하여, 특정 조건을 만족하지 않으면 이전 단계를 **재실행**하는 패턴을 구현한다.

#### 핵심 개념 설명

실제 애플리케이션에서는 LLM의 출력이 항상 기대에 부합하지 않을 수 있다. **Gate(게이트)**는 품질 검증 관문으로서, LLM 출력이 특정 기준을 충족하는지 검사한다. 기준을 충족하지 못하면 해당 단계를 다시 실행하여 더 나은 결과를 얻도록 한다.

이 예제에서는 재료 개수가 3~8개 범위 안에 들어야만 다음 단계로 진행할 수 있다:

```
START --> list_ingredients --[gate]--> create_recipe --> describe_plating --> END
                ^                          |
                |    (조건 불충족시)          |
                +----------<---------------+
```

#### 코드 분석

**게이트 함수 정의:**

```python
def gate(state: State):
    ingredients = state["ingredients"]

    if len(ingredients) > 8 or len(ingredients) < 3:
        return False

    return True
```

게이트 함수는 State를 받아서 `True` 또는 `False`를 반환한다. 이 반환값에 따라 그래프의 다음 경로가 결정된다.

- **True**: 재료 개수가 3~8개 범위 내 -- `create_recipe`로 진행
- **False**: 범위 밖 -- `list_ingredients`를 다시 실행 (재시도)

**조건부 엣지 설정:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_conditional_edges(
    "list_ingredients",
    gate,
    {
        True: "create_recipe",
        False: "list_ingredients",
    },
)
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

핵심은 `add_conditional_edges()` 메서드이다:
- **첫 번째 인자**: 출발 노드 (`"list_ingredients"`)
- **두 번째 인자**: 조건 판별 함수 (`gate`)
- **세 번째 인자**: 반환값과 목적지 노드의 매핑 딕셔너리

`gate`가 `False`를 반환하면 `"list_ingredients"`로 돌아가므로, 조건이 충족될 때까지 재시도 루프가 형성된다.

#### 게이트 패턴의 활용 시나리오

| 시나리오 | 게이트 조건 |
|---------|-----------|
| 코드 생성 | 생성된 코드가 구문 검사를 통과하는가? |
| 번역 | 번역 결과의 언어가 올바른가? |
| 데이터 추출 | 필수 필드가 모두 채워졌는가? |
| 요약 | 요약문의 길이가 적절한가? |

#### 실습 포인트

1. `gate` 함수의 조건을 변경해보며 재시도 횟수가 어떻게 달라지는지 확인한다.
2. 무한 루프를 방지하기 위해 최대 재시도 횟수를 추가하는 방법을 고민해본다 (예: State에 `retry_count` 필드 추가).
3. 게이트에서 단순 `True`/`False` 대신 여러 경로로 분기하는 것도 가능하다.

---

### 16.3 Routing Architecture -- 동적 라우팅

#### 주제 및 목표

입력의 특성에 따라 **서로 다른 처리 경로**로 분기하는 **Routing** 패턴을 구현한다. LLM이 입력을 분류하고, 분류 결과에 따라 적합한 모델이나 처리 로직을 선택한다.

#### 핵심 개념 설명

Routing 패턴은 "모든 작업에 같은 방식을 적용할 필요는 없다"는 아이디어에 기반한다. 쉬운 질문에 비싼 모델을 사용하는 것은 낭비이고, 어려운 질문에 약한 모델을 사용하면 품질이 떨어진다.

이 예제에서는 질문의 난이도를 자동으로 평가한 뒤, 난이도에 맞는 모델을 선택하여 응답을 생성한다:

```
                    +--> dumb_node (GPT-3.5) ---+
                    |                           |
START --> assess_difficulty --> average_node (GPT-4o) --> END
                    |                           |
                    +--> smart_node (GPT-5) ----+
```

#### 코드 분석

**모델 초기화:**

```python
llm = init_chat_model("openai:gpt-4o")

dumb_llm = init_chat_model("openai:gpt-3.5-turbo")
average_llm = init_chat_model("openai:gpt-4o")
smart_llm = init_chat_model("openai:gpt-5-2025-08-07")
```

세 가지 능력 수준의 LLM을 준비한다. 실제 프로덕션에서는 비용과 성능 사이의 균형을 맞추기 위해 이런 전략을 자주 사용한다.

**상태 및 스키마 정의:**

```python
class State(TypedDict):
    question: str
    difficulty: str
    answer: str
    model_used: str

class DifficultyResponse(BaseModel):
    difficulty_level: Literal["easy", "medium", "hard"]
```

`DifficultyResponse`는 `Literal` 타입을 사용하여 LLM이 반드시 `"easy"`, `"medium"`, `"hard"` 중 하나만 선택하도록 강제한다. 이것이 Structured Output의 강력한 장점이다 -- LLM의 응답을 프로그래밍적으로 제어 가능한 형태로 제한할 수 있다.

**난이도 평가 및 라우팅 노드:**

```python
def assess_difficulty(state: State):
    structured_llm = llm.with_structured_output(DifficultyResponse)

    response = structured_llm.invoke(
        f"""
        Assess the difficulty of this question
        Question: {state["question"]}

        - EASY: Simple facts, basic definitions, yes/no answers
        - MEDIUM: Requires explanation, comparison, analysis
        - HARD: Complex reasoning, multiple steps, deep expertise.
        """
    )

    difficulty_level = response.difficulty_level

    if difficulty_level == "easy":
        goto = "dumb_node"
    elif difficulty_level == "medium":
        goto = "average_node"
    elif difficulty_level == "hard":
        goto = "smart_node"

    return Command(
        goto=goto,
        update={
            "difficulty": difficulty_level,
        },
    )
```

이 함수에는 두 가지 중요한 개념이 있다:

1. **`Command` 객체**: LangGraph의 `Command`는 **상태 업데이트와 라우팅을 동시에** 수행한다. `goto`로 다음에 실행할 노드를 지정하고, `update`로 State를 업데이트한다. 이전의 `add_conditional_edges()`와 달리, 노드 함수 내부에서 직접 라우팅을 결정할 수 있다.

2. **난이도 평가 프롬프트**: 각 난이도의 기준을 명확히 제시하여 LLM이 일관된 분류를 하도록 유도한다.

**처리 노드들:**

```python
def dumb_node(state: State):
    response = dumb_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-3.5",
    }

def average_node(state: State):
    response = average_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-4o",
    }

def smart_node(state: State):
    response = smart_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-o3",
    }
```

각 노드는 자신에게 할당된 LLM으로 질문에 답한다. `model_used` 필드를 통해 어떤 모델이 사용되었는지 추적할 수 있다.

**그래프 구성:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("dumb_node", dumb_node)
graph_builder.add_node("average_node", average_node)
graph_builder.add_node("smart_node", smart_node)
graph_builder.add_node(
    "assess_difficulty",
    assess_difficulty,
    destinations=(
        "dumb_node",
        "average_node",
        "smart_node",
    ),
)

graph_builder.add_edge(START, "assess_difficulty")
graph_builder.add_edge("dumb_node", END)
graph_builder.add_edge("average_node", END)
graph_builder.add_edge("smart_node", END)

graph = graph_builder.compile()
```

`Command`를 반환하는 노드에는 `destinations` 매개변수를 추가해야 한다. 이것은 LangGraph에게 이 노드가 어떤 노드들로 라우팅할 수 있는지 알려준다. 그래프 시각화와 검증에 사용된다.

**실행:**

```python
graph.invoke({"question": "Investment potential of Uranium in 2026"})
```

이 질문은 복잡한 분석이 필요하므로 `"hard"`로 분류되어 `smart_node` (GPT-5)로 라우팅될 것이다.

#### 실습 포인트

1. 다양한 난이도의 질문을 입력하여 라우팅이 올바르게 작동하는지 확인한다.
2. `model_used` 필드를 확인하여 실제로 어떤 모델이 선택되었는지 검증한다.
3. 모델 선택 외에도, 다른 프롬프트 템플릿이나 도구(tool)로 라우팅하는 시나리오를 설계해본다.

---

### 16.4 Parallelization Architecture -- 병렬 실행

#### 주제 및 목표

여러 LLM 호출을 **동시에 병렬로 실행**한 뒤, 모든 결과가 모이면 **집계(aggregation)**하는 **Parallelization** 패턴을 구현한다.

#### 핵심 개념 설명

순차 실행은 간단하지만, 서로 독립적인 작업이 여러 개 있을 때는 비효율적이다. 예를 들어 문서에서 요약, 감정 분석, 핵심 포인트 추출, 추천 사항 도출을 순차적으로 하면 4배의 시간이 걸린다. 이 작업들은 서로 의존하지 않으므로 동시에 실행할 수 있다.

이 예제에서는 연준(Fed) 의장의 기자회견 발표문을 4가지 관점에서 동시에 분석한다:

```
            +--> get_summary --------+
            |                        |
            +--> get_sentiment ------+
START ----->|                        +--> get_final_analysis --> END
            +--> get_key_points -----+
            |                        |
            +--> get_recommendation -+
```

4개의 분석 노드가 **동시에** 실행되고, 모든 노드가 완료되면 `get_final_analysis`가 종합 분석을 수행한다.

#### 코드 분석

**상태 정의:**

```python
class State(TypedDict):
    document: str
    summary: str
    sentiment: str
    key_points: str
    recommendation: str
    final_analysis: str
```

각 병렬 노드가 채울 필드가 개별적으로 정의되어 있다. 이 필드들은 서로 독립적이므로 동시에 업데이트해도 충돌이 없다.

**병렬 노드 함수들:**

```python
def get_summary(state: State):
    response = llm.invoke(
        f"Write a 3-sentence summary of this document {state['document']}"
    )
    return {"summary": response.content}

def get_sentiment(state: State):
    response = llm.invoke(
        f"Analyse the sentiment and tone of this document {state['document']}"
    )
    return {"sentiment": response.content}

def get_key_points(state: State):
    response = llm.invoke(
        f"List the 5 most important points of this document {state['document']}"
    )
    return {"key_points": response.content}

def get_recommendation(state: State):
    response = llm.invoke(
        f"Based on the document, list 3 recommended next steps {state['document']}"
    )
    return {"recommendation": response.content}
```

4개의 함수가 모두 같은 `state["document"]`를 읽지만, 서로 다른 필드에 결과를 저장한다. 이것이 병렬 실행이 가능한 이유이다 -- 읽기는 공유하되, 쓰기는 분리된다.

**집계 노드:**

```python
def get_final_analysis(state: State):
    response = llm.invoke(
        f"""
    Give me an analysis of the following report

    DOCUMENT ANALYSIS REPORT
    ========================

    EXECUTIVE SUMMARY:
    {state['summary']}

    SENTIMENT ANALYSIS:
    {state['sentiment']}

    KEY POINTS:
    {state.get("key_points", "")}

    RECOMMENDATIONS:
    {state.get('recommendation', "N/A")}
    """
    )
    return {"final_analysis": response.content}
```

집계 노드는 병렬 노드들이 채운 모든 필드를 종합하여 최종 분석을 생성한다. 이 노드는 반드시 모든 병렬 노드가 **완료된 후**에만 실행된다.

**그래프 구성 -- 병렬 엣지의 핵심:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("get_summary", get_summary)
graph_builder.add_node("get_sentiment", get_sentiment)
graph_builder.add_node("get_key_points", get_key_points)
graph_builder.add_node("get_recommendation", get_recommendation)
graph_builder.add_node("get_final_analysis", get_final_analysis)

# START에서 4개 노드로 동시에 연결 = 병렬 실행!
graph_builder.add_edge(START, "get_summary")
graph_builder.add_edge(START, "get_sentiment")
graph_builder.add_edge(START, "get_key_points")
graph_builder.add_edge(START, "get_recommendation")

# 4개 노드 모두 get_final_analysis로 연결 = 모두 완료 후 실행
graph_builder.add_edge("get_summary", "get_final_analysis")
graph_builder.add_edge("get_sentiment", "get_final_analysis")
graph_builder.add_edge("get_key_points", "get_final_analysis")
graph_builder.add_edge("get_recommendation", "get_final_analysis")

graph_builder.add_edge("get_final_analysis", END)

graph = graph_builder.compile()
```

LangGraph에서 병렬 실행을 구현하는 방법은 매우 직관적이다:
- **`START`에서 여러 노드로 엣지를 연결**하면 해당 노드들이 동시에 실행된다.
- **여러 노드에서 하나의 노드로 엣지를 연결**하면, 모든 선행 노드가 완료될 때까지 대기한 후 실행된다 (join/barrier).

**스트리밍 실행:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

`graph.stream()`을 사용하면 각 노드의 실행이 완료될 때마다 결과를 받을 수 있다. `stream_mode="updates"`는 각 노드의 State 업데이트만 스트리밍한다. 병렬 노드의 결과가 완료되는 순서대로 출력되므로, 어떤 작업이 먼저 끝나는지 실시간으로 확인할 수 있다.

#### 실습 포인트

1. 순차 실행과 병렬 실행의 총 소요 시간을 비교해본다.
2. `stream_mode="updates"`로 결과가 도착하는 순서를 관찰한다 -- 어떤 분석이 가장 빨리 완료되는가?
3. 병렬 노드 수를 늘리거나 줄여보며 성능 변화를 측정한다.
4. 실제 문서(연준 발표문)를 사용하여 분석 결과의 품질을 평가해본다.

---

### 16.5 Orchestrator-Workers Architecture -- 오케스트레이터-워커

#### 주제 및 목표

**입력에 따라 동적으로 워커(worker)를 생성**하여 작업을 분배하고, 모든 워커의 결과를 수집하여 최종 결과를 만드는 **Orchestrator-Workers (Map-Reduce)** 패턴을 구현한다.

#### 핵심 개념 설명

16.4의 Parallelization은 **노드의 수가 고정**되어 있었다 (항상 4개의 분석 노드). 하지만 실제 상황에서는 입력에 따라 병렬 작업의 수가 달라져야 할 때가 많다. 예를 들어:
- 문서가 3개의 단락이면 3개의 요약 워커가 필요
- 문서가 20개의 단락이면 20개의 요약 워커가 필요

이 패턴에서는 **오케스트레이터(dispatcher)**가 입력을 분석하여 필요한 만큼의 워커를 동적으로 생성한다. LangGraph의 `Send` API를 사용하여 이를 구현한다.

```
            +--> summarize_p (단락 0) --+
            |                           |
            +--> summarize_p (단락 1) --+
START ----->|                           +--> final_summary --> END
            +--> summarize_p (단락 2) --+
            |                           |
            +--> summarize_p (단락 N) --+
```

워커의 수(N)는 실행 시점에 문서의 단락 수에 따라 결정된다.

#### 코드 분석

**새로운 임포트:**

```python
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.types import Send
from operator import add
```

- **`Send`**: 특정 노드에 특정 인자를 전달하며 실행을 지시하는 객체
- **`Annotated`와 `add`**: 리스트 필드에 대한 **리듀서(reducer)** 정의에 사용

**상태 정의 -- Annotated 리듀서:**

```python
class State(TypedDict):
    document: str
    final_summary: str
    summaries: Annotated[list[dict], add]
```

`Annotated[list[dict], add]`는 LangGraph의 핵심 개념인 **리듀서**이다. 여러 워커가 동시에 `summaries` 필드에 값을 반환하면, 기본 동작은 마지막 값으로 덮어쓰기이다. 하지만 `add` 리듀서를 지정하면 **모든 워커의 결과가 리스트에 누적**된다.

예를 들어 워커 A가 `{"summaries": [item_a]}`를, 워커 B가 `{"summaries": [item_b]}`를 반환하면, 최종 `summaries`는 `[item_a, item_b]`가 된다. `operator.add`는 리스트의 `+` 연산(연결)을 수행하기 때문이다.

**워커 노드:**

```python
def summarize_p(args):
    paragraph = args["paragraph"]
    index = args["index"]
    response = llm.invoke(
        f"Write a 3-sentence summary for this paragraph: {paragraph}",
    )
    return {
        "summaries": [
            {
                "summary": response.content,
                "index": index,
            }
        ],
    }
```

주의할 점:
- 이 함수의 매개변수는 `state`가 아니라 `args`이다. `Send`를 통해 전달되는 커스텀 인자를 받는다.
- `index`를 함께 저장하여 나중에 단락 순서를 복원할 수 있도록 한다.
- 반환 값의 `summaries`가 리스트로 감싸져 있다 -- `add` 리듀서와 함께 사용하기 위함이다.

**오케스트레이터 (디스패처) 함수:**

```python
def dispatch_summarizers(state: State):
    chunks = state["document"].split("\n\n")
    return [
        Send("summarize_p", {"paragraph": chunk, "index": index})
        for index, chunk in enumerate(chunks)
    ]
```

이것이 Orchestrator-Workers 패턴의 핵심이다:

1. 문서를 `"\n\n"` (빈 줄)으로 분할하여 단락 리스트를 만든다.
2. 각 단락에 대해 `Send("summarize_p", {...})` 객체를 생성한다.
3. `Send`의 첫 번째 인자는 실행할 노드 이름, 두 번째 인자는 해당 노드에 전달할 데이터이다.
4. `Send` 객체의 리스트를 반환하면 LangGraph가 해당 노드들을 **동시에 병렬 실행**한다.

문서에 단락이 15개 있으면 15개의 `summarize_p` 인스턴스가 동시에 실행된다. 이것이 16.4의 정적 병렬화와의 가장 큰 차이점이다.

**최종 집계 노드:**

```python
def final_summary(state: State):
    response = llm.invoke(
        f"Using the following summaries, give me a final one {state['summaries']}"
    )
    return {
        "final_summary": response.content,
    }
```

모든 워커의 요약이 `summaries` 리스트에 모이면, `final_summary` 노드가 이를 종합하여 최종 요약을 생성한다.

**그래프 구성:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("summarize_p", summarize_p)
graph_builder.add_node("final_summary", final_summary)

graph_builder.add_conditional_edges(
    START,
    dispatch_summarizers,
    ["summarize_p"],
)

graph_builder.add_edge("summarize_p", "final_summary")
graph_builder.add_edge("final_summary", END)

graph = graph_builder.compile()
```

`add_conditional_edges()`를 사용하지만, 여기서의 용도는 16.2의 게이트와 다르다:
- **세 번째 인자 `["summarize_p"]`**: 가능한 목적지 노드의 리스트이다. `dispatch_summarizers`가 `Send` 객체들을 반환하면 해당 노드들이 동적으로 생성된다.
- LangGraph는 모든 `Send`가 완료될 때까지 기다린 후 다음 엣지(`"summarize_p"` --> `"final_summary"`)를 진행한다.

**실행:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

스트리밍으로 실행하면 각 단락의 요약이 완료될 때마다 결과가 출력된다. 단락 수에 따라 워커 수가 자동으로 조절되는 것을 확인할 수 있다.

#### 실습 포인트

1. `dispatch_summarizers`가 생성하는 `Send` 객체의 수를 다양한 문서로 테스트해본다.
2. `Annotated[list[dict], add]` 리듀서를 제거하면 어떤 문제가 발생하는지 확인한다.
3. `index` 필드를 활용하여 결과를 원래 순서대로 정렬하는 후처리 로직을 추가해본다.
4. 요약 외에 다른 작업(번역, 키워드 추출 등)에 동일한 패턴을 적용해본다.

---

## 3. 챕터 핵심 정리

### 아키텍처 패턴 비교표

| 패턴 | 실행 방식 | 노드 수 | 적합한 상황 | LangGraph API |
|------|----------|---------|------------|---------------|
| **Prompt Chaining** | 순차적 | 고정 | 단계별 의존성이 있는 작업 | `add_edge()` |
| **Prompt Chaining + Gate** | 순차적 + 재시도 | 고정 | 품질 검증이 필요한 작업 | `add_conditional_edges()` |
| **Routing** | 분기 | 고정 | 입력 특성에 따라 다른 처리가 필요한 작업 | `Command(goto=...)` |
| **Parallelization** | 병렬 | 고정 | 독립적인 여러 분석을 동시에 수행 | 복수 `add_edge(START, ...)` |
| **Orchestrator-Workers** | 동적 병렬 | 가변 | 입력 크기에 따라 작업 수가 달라지는 경우 | `Send()` + `Annotated[..., add]` |

### 핵심 LangGraph 개념 요약

1. **StateGraph**: 상태 기반 그래프의 기본 구성 요소. `TypedDict`로 정의된 상태를 노드들이 공유한다.
2. **add_edge()**: 노드 간 고정 경로를 정의한다.
3. **add_conditional_edges()**: 함수의 반환값에 따라 다음 노드를 동적으로 결정한다.
4. **Command**: 노드 함수 내부에서 라우팅과 상태 업데이트를 동시에 수행한다.
5. **Send**: 특정 노드에 커스텀 인자를 전달하여 동적 병렬 실행을 구현한다.
6. **Annotated + Reducer**: 여러 노드가 같은 필드에 값을 추가할 때 병합 전략을 정의한다.
7. **with_structured_output()**: LLM의 출력을 Pydantic 모델로 구조화하여 프로그래밍적으로 활용 가능하게 한다.

---

## 4. 실습 과제

### 과제 1: 다국어 번역 체인 (Prompt Chaining)

요리 레시피 예제를 참고하여 다음 파이프라인을 구현하시오:
1. 사용자가 한국어 텍스트를 입력한다.
2. 첫 번째 노드가 영어로 번역한다.
3. 두 번째 노드가 영어 번역의 문법을 교정한다.
4. 세 번째 노드가 교정된 영어를 자연스러운 한국어로 역번역한다.

**보너스**: 역번역 결과가 원문과 유사한지 평가하는 게이트를 추가하시오.

### 과제 2: 고객 문의 라우팅 시스템 (Routing)

고객 문의를 분류하여 적절한 처리 경로로 라우팅하는 시스템을 구현하시오:
- **기술 지원**: 상세한 기술 해결책 제공
- **결제 문의**: 결제 관련 정보 안내
- **일반 문의**: 간단한 답변 제공
- **불만 처리**: 공감적 응답 + 해결 방안 제시

각 경로에서 다른 프롬프트 템플릿을 사용하도록 설계하시오.

### 과제 3: 뉴스 기사 종합 분석기 (Parallelization)

하나의 뉴스 기사를 입력받아 다음 분석을 **동시에** 수행하는 시스템을 구현하시오:
- 3줄 요약
- 감정/논조 분석
- 관련 키워드 추출 (Structured Output 사용)
- 팩트 체크 포인트 나열
- 독자 영향도 평가

모든 분석이 완료되면 종합 리포트를 생성하시오.

### 과제 4: 대용량 문서 처리기 (Orchestrator-Workers)

PDF 또는 긴 텍스트 문서를 입력받아 다음을 수행하는 시스템을 구현하시오:
1. 문서를 적절한 크기의 청크로 분할한다.
2. 각 청크에 대해 워커를 동적으로 생성하여 키워드 추출과 요약을 동시에 수행한다.
3. 모든 워커의 결과를 수집하여 최종 문서 요약과 키워드 클라우드 데이터를 생성한다.

**힌트**: `Annotated[list[dict], add]` 리듀서를 활용하여 워커 결과를 누적 수집하시오.

### 과제 5: 통합 아키텍처 설계 (종합)

위의 모든 패턴을 결합하여 "AI 에세이 작성 도우미"를 설계하시오:
1. **Routing**: 에세이 주제의 난이도를 평가하여 적절한 연구 깊이를 결정
2. **Orchestrator-Workers**: 주제를 여러 하위 주제로 분해하고 각각에 대해 연구 수행
3. **Parallelization**: 연구 결과를 바탕으로 서론/본론/결론을 동시에 초안 작성
4. **Prompt Chaining + Gate**: 초안을 교정하고, 품질 기준을 충족할 때까지 재작성

---

> **참고**: 이 챕터의 모든 코드는 `workflow-architectures/main.ipynb` 노트북에서 실행할 수 있다. 실행 전 `.env` 파일에 OpenAI API 키를 설정하고, `uv sync`로 의존성을 설치해야 한다.
