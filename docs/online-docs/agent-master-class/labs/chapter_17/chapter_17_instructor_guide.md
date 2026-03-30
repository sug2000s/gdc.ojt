# Chapter 17: LangGraph 워크플로우 테스팅 — 강사용 해설서

> 이 문서는 강사가 Chapter 17을 **미리 이해**하기 위한 해설서입니다.
> 강의 대본(lecture_script)과 별개로, 코드를 한 줄 한 줄 쉽게 풀어서 설명합니다.

---

## 전체 진화 흐름 (먼저 큰 그림)

```
17.1  규칙 기반 그래프          "if-elif-else, 결정적"
       +
17.2  Pytest 기초               "%%writefile + parametrize + !pytest"
       +
17.3  노드 단위 테스트           "graph.nodes + update_state 부분 실행"
       +
17.4  AI 노드 전환              "with_structured_output + Pydantic + Literal"
       +
17.5  AI 테스트 전략             "범위 검증, 일관성 검증"
       +
17.6  LLM-as-a-Judge            "Golden Examples + 유사도 점수 + Threshold"
```

핵심 흐름: **정확 일치 → 범위 검증 → LLM 판정**

각 섹션이 테스트 전략을 한 단계씩 진화시키는 구조입니다.

---

## 17.0 Setup

### import 정리

```python
import os
from dotenv import load_dotenv
```

| import | 뭐하는 놈 |
|--------|----------|
| `os` | 환경 변수 읽기 (`os.getenv`) |
| `load_dotenv` | `.env` 파일에서 환경 변수 로드 |

버전 체크:

```python
from importlib.metadata import version
print(f"pytest: {version('pytest')}")
```

`pytest`가 없으면 `pip install pytest`로 설치.

---

## 17.1 이메일 처리 그래프 — "테스트하기 쉬운 결정적 그래프"

### import 정리

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
```

| import | 역할 |
|--------|------|
| `TypedDict` | 상태 스키마를 딕셔너리처럼 정의 |
| `StateGraph` | 그래프(워크플로우) 설계도 |
| `START, END` | 시작점, 끝점 특수 노드 |

### 상태 정의

```python
class EmailState(TypedDict):
    email: str       # 입력 이메일 본문
    category: str    # 분류 결과 (urgent/spam/normal)
    priority: str    # 우선순위 (high/low/medium)
    response: str    # 생성된 응답
```

비유: **조립 라인의 부품 상자**. 이메일이 라인을 지나면서 카테고리, 우선순위, 응답이 채워짐.

### 노드 함수 — 3개 모두 규칙 기반

```python
def categorize_email(state: EmailState) -> dict:
    email = state["email"].lower()
    if "urgent" in email or "asap" in email:
        return {"category": "urgent"}
    elif "offer" in email or "discount" in email:
        return {"category": "spam"}
    return {"category": "normal"}
```

- `.lower()` — 대소문자 무시. "URGENT"도 "urgent"도 다 잡힘
- `if ... in ...` — 키워드가 포함되어 있는지 확인
- `return {"category": "urgent"}` — 상태의 `category` 필드만 업데이트

```python
def assign_priority(state: EmailState) -> dict:
    category = state["category"]
    if category == "urgent":
        return {"priority": "high"}
    elif category == "spam":
        return {"priority": "low"}
    return {"priority": "medium"}
```

카테고리 → 우선순위 **1:1 매핑**. urgent=high, spam=low, normal=medium.

```python
def generate_response(state: EmailState) -> dict:
    templates = {
        "urgent": "We received your urgent request and will respond within 1 hour.",
        "spam": "This email has been classified as spam.",
        "normal": "Thank you for your email. We will respond within 24 hours.",
    }
    return {"response": templates.get(state["category"], "Email received.")}
```

- `templates` 딕셔너리에서 카테고리에 맞는 템플릿을 꺼냄
- `.get(key, default)` — 키가 없으면 기본값 반환

### 그래프 구성

```python
def build_email_graph():
    builder = StateGraph(EmailState)
    builder.add_node("categorize_email", categorize_email)
    builder.add_node("assign_priority", assign_priority)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "categorize_email")
    builder.add_edge("categorize_email", "assign_priority")
    builder.add_edge("assign_priority", "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()
```

비유: **직선 컨베이어 벨트**. 분기 없이 순서대로 흐름.

```
START → categorize_email → assign_priority → generate_response → END
```

`build_email_graph()` 함수로 감싸는 이유: 테스트에서 매번 새 그래프를 만들기 위해.

---

## 17.2 Pytest 기초 — "노트북에서 pytest 돌리기"

### 핵심 패턴: %%writefile

```
%%writefile main.py       ← 셀 내용을 main.py 파일로 저장
%%writefile tests.py      ← 셀 내용을 tests.py 파일로 저장
!pytest tests.py -v       ← 터미널 명령 실행 (pytest)
```

비유: `%%writefile`은 **"셀 내용을 복사해서 파일에 붙여넣기"** 하는 마법 명령.

왜 필요한가?
- pytest는 `.py` 파일만 인식
- 노트북 셀은 `.py`가 아님
- 그래서 `%%writefile`로 코드를 파일로 빼는 것

### @pytest.mark.parametrize 상세 해설

```python
@pytest.mark.parametrize(
    "email, expected_category, expected_priority",   # ← 파라미터 이름
    [
        ("URGENT: Server down!", "urgent", "high"),   # ← 테스트 케이스 1
        ("Fix this ASAP please", "urgent", "high"),   # ← 테스트 케이스 2
        ("Special offer! Buy now!", "spam", "low"),   # ← 테스트 케이스 3
        # ... 총 6개
    ],
)
def test_email_pipeline(email, expected_category, expected_priority):
    graph = build_email_graph()
    result = graph.invoke({"email": email})
    assert result["category"] == expected_category
    assert result["priority"] == expected_priority
```

| 요소 | 설명 |
|------|------|
| 첫 번째 인자 | 쉼표로 구분된 파라미터 이름 문자열 |
| 두 번째 인자 | 테스트 케이스 리스트 (튜플의 리스트) |
| 함수 인자 | parametrize의 이름과 **동일해야** 함 |
| 실행 | 튜플 수만큼 테스트가 **자동 반복** 실행 |

비유: **엑셀 시트의 각 행이 하나의 테스트**. 함수는 하나지만 데이터만 바꿔서 반복.

### assert 문법

```python
assert result["category"] == expected_category   # 같지 않으면 FAIL
assert len(result["response"]) > 0               # 응답이 비어있으면 FAIL
```

`assert` 뒤의 조건이 `False`면 테스트 실패. 에러 메시지 없이도 실패.
커스텀 메시지를 넣으려면: `assert x == y, f"Expected {y}, got {x}"`

### !pytest -v 플래그

| 플래그 | 의미 |
|--------|------|
| `-v` | verbose, 각 테스트를 한 줄씩 표시 |
| `--tb=short` | traceback 짧게 |
| `-x` | 첫 실패에서 중단 |

---

## 17.3 노드 단위 테스트 — "부분만 테스트하기"

### 방법 1: graph.nodes["name"].invoke()

```python
cat_result = graph.nodes["categorize_email"].invoke(test_state)
```

| 요소 | 설명 |
|------|------|
| `graph.nodes` | 컴파일된 그래프의 노드 딕셔너리 |
| `["categorize_email"]` | 노드 이름으로 접근 |
| `.invoke(state)` | 그 노드만 단독 실행 |

비유: 컨베이어 벨트에서 **한 공정만 따로 떼서** 테스트하는 것.

주의: 전체 그래프가 아니라 노드 함수 하나만 실행됨. 그래프의 엣지는 무시.

### 방법 2: update_state + as_node

### 새로운 import

```python
from langgraph.checkpoint.memory import MemorySaver
```

| import | 역할 |
|--------|------|
| `MemorySaver` | **메모리 기반 체크포인터**. 상태를 메모리에 저장. 테스트용으로 가벼움 |

### 체크포인터가 있는 그래프

```python
graph_mem = builder.compile(checkpointer=MemorySaver())
```

`MemorySaver` vs `SqliteSaver`:
- `MemorySaver` — 프로세스 메모리에 저장. 빠르지만 종료하면 사라짐
- `SqliteSaver` — DB 파일에 저장. 영구 보존
- 테스트에는 `MemorySaver`가 적합 (가볍고 빠름)

### update_state 상세 해설

```python
graph_mem.update_state(
    config2,                        # 어떤 thread에?
    {"category": "urgent"},         # 어떤 값으로?
    as_node="categorize_email",     # 어떤 노드가 반환한 것처럼?
)
```

| 파라미터 | 설명 |
|----------|------|
| `config` | thread_id를 포함한 설정 (어떤 대화에 적용할지) |
| `values` | 주입할 상태 값 |
| `as_node` | **이 노드가 이 값을 반환한 것처럼** 처리 |

비유: **시험지 중간에 답을 강제로 적어 넣는 것**. "1번 문제 답은 무조건 A야" → 2번부터 채점.

### 부분 실행

```python
result_partial = graph_mem.invoke(None, config=config2)
```

`invoke(None)` — 새 입력 없이 **남은 노드만 실행**.
`as_node="categorize_email"` 이후의 `assign_priority → generate_response`만 실행.

이걸로:
- 특정 노드에 버그 의심 → 앞 노드 건너뛰고 해당 노드만 테스트
- "이 상태에서 시작하면 나머지가 제대로 동작하나?" 검증

---

## 17.4 AI 노드 전환 — "규칙을 AI로 교체"

### 새로운 import

```python
import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
```

| import | 역할 |
|--------|------|
| `Literal` | **허용 값을 제한**. `Literal["a", "b"]` → "a" 또는 "b"만 가능 |
| `BaseModel` | Pydantic 모델 기본 클래스. 데이터 검증 내장 |
| `Field` | 필드에 설명, 제약 조건 추가 |
| `init_chat_model` | LLM 초기화 (`"openai:모델명"` 형식) |

### Pydantic 출력 스키마

```python
class CategoryOutput(BaseModel):
    """이메일 카테고리 분류 결과"""
    category: Literal["urgent", "spam", "normal"] = Field(
        description="The email category"
    )
```

비유: **틀(금형)**. LLM이 뭘 말하든 이 틀에 맞춰서 나옴.

- `Literal["urgent", "spam", "normal"]` — 이 3개 값 외에는 **불가**
- LLM이 "critical"이라고 하고 싶어도 → "urgent"로 강제됨
- `Field(description=...)` — LLM에게 이 필드를 설명 (프롬프트에 포함됨)

같은 패턴으로 `PriorityOutput`, `ResponseOutput`도 정의.

### with_structured_output

```python
category_llm = llm.with_structured_output(CategoryOutput)
priority_llm = llm.with_structured_output(PriorityOutput)
response_llm = llm.with_structured_output(ResponseOutput)
```

| 코드 | 반환 타입 | 반환 예시 |
|------|----------|----------|
| `llm.invoke("...")` | 문자열 | `"This email is urgent"` |
| `category_llm.invoke("...")` | `CategoryOutput` 객체 | `CategoryOutput(category="urgent")` |

비유: 일반 LLM = **자유 작문**, structured LLM = **빈칸 채우기 시험**.

### AI 노드 함수 vs 규칙 기반

```python
# 규칙 기반 (17.1)
def categorize_email(state):
    if "urgent" in state["email"].lower():
        return {"category": "urgent"}

# AI 기반 (17.4)
def ai_categorize_email(state):
    result = category_llm.invoke(
        f"Classify this email: {state['email']}"
    )
    return {"category": result.category}
```

| 비교 항목 | 규칙 기반 | AI 기반 |
|-----------|----------|---------|
| 판단 방식 | 키워드 매칭 | 문맥 이해 |
| 결정성 | 결정적 (항상 같음) | 비결정적 (달라질 수 있음) |
| 유연성 | 키워드 목록 한정 | 새 표현도 이해 |
| 테스트 난이도 | 쉬움 (`assert ==`) | 어려움 (새 전략 필요) |

핵심: **그래프 구조는 동일**하고 노드 내부만 교체. LangGraph의 장점!

---

## 17.5 AI 테스트 전략 — "정확히 몰라도 검증할 수 있다"

### 4가지 전략

| 전략 | 코드 예시 | 비유 |
|------|----------|------|
| 유효값 범위 | `assert result in {"urgent", "spam", "normal"}` | 객관식 문제 (4개 중 하나면 OK) |
| 길이 범위 | `assert 20 <= len(resp) <= 1000` | 글자수 제한 (200자 이상 1000자 이하) |
| 최소 품질 | `assert score >= 70` | 합격 커트라인 |
| 일관성 | 3번 중 2번 동일 | 재현성 검사 |

### @pytest.fixture 상세

```python
@pytest.fixture
def ai_graph():
    return build_ai_email_graph()
```

비유: **시험 전에 교실을 세팅하는 것**. 매 테스트마다 깨끗한 그래프 제공.

- 여러 테스트 함수에서 `ai_graph`를 인자로 받으면 자동 주입
- 매번 새 그래프 생성 → 테스트 간 상태 오염 방지

### 일관성 테스트 상세

```python
from collections import Counter
most_common_count = Counter(categories).most_common(1)[0][1]
assert most_common_count >= 2
```

| 코드 | 설명 |
|------|------|
| `Counter(categories)` | 각 값의 출현 횟수 카운트 (예: `{"urgent": 3}`) |
| `.most_common(1)` | 가장 많은 것 1개 반환 (예: `[("urgent", 3)]`) |
| `[0][1]` | 그 횟수만 추출 (예: `3`) |
| `>= 2` | 3번 중 2번 이상이면 통과 (66% 일관성) |

---

## 17.6 LLM-as-a-Judge — "AI가 AI를 채점한다"

### 핵심 개념 3가지

| 요소 | 설명 | 비유 |
|------|------|------|
| **Golden Examples** | 카테고리별 이상적 응답 미리 작성 | 모범 답안지 |
| **Judge LLM** | 생성 응답 vs golden 비교 → 점수 | 채점관 |
| **Threshold** | 70점 이상이면 통과 | 합격 커트라인 |

### Golden Examples

```python
RESPONSE_EXAMPLES = {
    "urgent": "Thank you for alerting us. We have escalated this to our on-call team...",
    "spam": "This message has been identified as unsolicited commercial email...",
    "normal": "Thank you for your email. We have received your message...",
}
```

"이상적인 응답"을 미리 사람이 작성. AI 생성 응답이 이것과 얼마나 비슷한지가 점수.

### SimilarityScoreOutput

```python
class SimilarityScoreOutput(BaseModel):
    score: int = Field(gt=0, lt=100, description="Similarity score between 1 and 99")
    reasoning: str = Field(description="Brief explanation of the score")
```

| 필드 | 타입 | 제약 | 설명 |
|------|------|------|------|
| `score` | `int` | `gt=0, lt=100` (1~99) | 유사도 점수 |
| `reasoning` | `str` | 없음 | 점수 이유 설명 |

`gt=0` — greater than 0 (0 초과)
`lt=100` — less than 100 (100 미만)

비유: 시험지 채점 결과. 점수 + 코멘트.

### Judge 함수

```python
def judge_response(generated: str, golden: str) -> SimilarityScoreOutput:
    prompt = f"""You are an expert quality evaluator. Compare...

    Consider:
    - Tone and professionalism (30%)
    - Key information coverage (40%)
    - Appropriate length and format (30%)

    Golden Response:
    {golden}

    Generated Response:
    {generated}

    Score from 1 to 99..."""
    return judge_llm.invoke(prompt)
```

프롬프트의 평가 기준:
- **톤과 전문성 (30%)** — 비즈니스 톤이 맞는가?
- **핵심 정보 포함 (40%)** — 중요한 내용이 다 들어갔는가?
- **적절한 길이와 형식 (30%)** — 너무 짧거나 길지 않은가?

비유: **루브릭(채점표)**. 채점관이 이 기준에 따라 점수를 매김.

### 점수 기준

```
80-99: 우수 (Excellent match)
60-79: 양호 (Good match with minor differences)
40-59: 허용 (Acceptable but missing key elements)
1-39:  불량 (Poor match)
```

### Threshold = 70

```python
THRESHOLD = 70

assert score_result.score >= THRESHOLD
```

"70점 이상이면 합격". 이 숫자는 프로젝트마다 조정.
- 엄격한 프로젝트: 80
- 느슨한 프로젝트: 60

### Judge 자체를 검증하는 테스트

```python
def test_judge_perfect_match():
    golden = RESPONSE_EXAMPLES["urgent"]
    score_result = judge_response(golden, golden)
    assert score_result.score >= 90   # 자기 자신이니까 90+ 이어야 함

def test_judge_poor_match():
    poor_response = "lol ok whatever"
    score_result = judge_response(poor_response, golden)
    assert score_result.score < 40    # 엉터리니까 40 미만이어야 함
```

이 두 테스트가 하는 일: **"채점관이 제대로 채점하는지?"** 검증.
- 모범답안 vs 모범답안 → 높은 점수
- 모범답안 vs 쓰레기 → 낮은 점수

이게 안 되면 Judge 자체를 못 믿으니까, Judge 검증은 필수!

---

## 핵심 개념 정리표

| 섹션 | 테스트 대상 | 핵심 코드 | 테스트 전략 | 비유 |
|------|-----------|----------|-----------|------|
| **17.1** | 규칙 기반 그래프 | `if-elif-else` | - | 컨베이어 벨트 |
| **17.2** | 전체 파이프라인 | `%%writefile` + `@parametrize` | `assert ==` 정확 일치 | 엑셀 시트 채점 |
| **17.3** | 개별 노드 | `graph.nodes["name"].invoke()` | 단위 테스트 | 공정 하나만 테스트 |
| **17.4** | AI 노드 | `with_structured_output` + `Literal` | - (전환만) | 빈칸 채우기 시험 |
| **17.5** | AI 출력 범위 | `assert in` + `Counter` | 범위 + 일관성 | 객관식 + 재현성 |
| **17.6** | AI 응답 품질 | `SimilarityScoreOutput` + `judge_response` | LLM 판정 + Threshold | 채점관 + 커트라인 |

## import 전체 정리표

| import | 처음 등장 | 역할 |
|--------|----------|------|
| `TypedDict` | 17.1 | 상태 스키마 정의 |
| `StateGraph` | 17.1 | 그래프 설계도 |
| `START, END` | 17.1 | 시작/끝 특수 노드 |
| `MemorySaver` | 17.3 | 메모리 기반 체크포인터 (테스트용) |
| `Literal` | 17.4 | 허용 값 제한 |
| `BaseModel` | 17.4 | Pydantic 데이터 모델 |
| `Field` | 17.4 | 필드 설명/제약 조건 |
| `init_chat_model` | 17.4 | LLM 초기화 |
| `pytest` | 17.2 | 테스트 프레임워크 |
| `@pytest.mark.parametrize` | 17.2 | 여러 케이스 한 번에 테스트 |
| `@pytest.fixture` | 17.5 | 재사용 객체 생성 |
| `Counter` | 17.5 | 일관성 검증용 카운터 |
