# Chapter 15: LangGraph 프로젝트 파이프라인 — 강사용 해설서

> 이 문서는 강사가 Chapter 15를 **미리 이해**하기 위한 해설서입니다.
> 강의 대본(lecture_script)과 별개로, 코드를 한 줄 한 줄 쉽게 풀어서 설명합니다.

---

## 전체 진화 흐름 (먼저 큰 그림)

```
15.1  START → get_topic_info → write_draft → END         "2노드 선형 파이프라인"
       +
15.2  + Send API + Map-Reduce + with_structured_output    "병렬 섹션 작성"
       +
15.3  + interrupt() / Command(resume=) + SqliteSaver      "사람 리뷰"
       +
15.4  전부 통합                                            "완성된 파이프라인"
       +
15.5  langgraph.json / graph.py                           "프로덕션 배포"
```

각 섹션이 **이전 섹션에 패턴을 추가**하는 구조입니다.
Chapter 14(챗봇)에서 개별적으로 배운 패턴을, 여기서 **하나의 프로젝트로 합칩니다**.

---

## 15.1 기본 파이프라인 — "가장 단순한 파이프라인"

### import 정리

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
```

| import | 뭐하는 놈 |
|--------|----------|
| `TypedDict` | 파이프라인 상태를 정의하는 파이썬 타입. 딕셔너리에 타입 힌트 추가 |
| `StateGraph` | 그래프(워크플로우)를 만드는 설계도 |
| `START, END` | 시작점, 끝점 (특수 노드) |
| `init_chat_model` | LLM을 초기화하는 함수. `"openai:모델명"` 형식 |

### 챗봇 vs 파이프라인 — 상태 설계의 차이

```python
# 챗봇 (Chapter 14) — 대화 메시지가 쌓이는 구조
class State(MessagesState):
    pass

# 파이프라인 (Chapter 15) — 데이터가 단계별로 변환되는 구조
class PipelineState(TypedDict):
    topic: str
    background_info: str
    draft: str
```

**비유:**
- 챗봇 = 카카오톡 (메시지가 계속 쌓임)
- 파이프라인 = 공장 조립 라인 (재료 → 부품 → 완성품)

### 노드 함수 패턴

```python
def get_topic_info(state: PipelineState):
    topic = state["topic"]
    response = llm.invoke(f"...{topic}...")
    return {"background_info": response.content}
```

**모든 노드가 따르는 3단계:**
1. 상태에서 필요한 데이터를 꺼냄 (`state["topic"]`)
2. LLM 호출 (`llm.invoke(...)`)
3. 결과를 상태 필드에 반환 (`return {"background_info": ...}`)

`return`할 때 전체 상태를 반환하는 게 아니라, **변경할 필드만** 반환한다.
LangGraph가 알아서 기존 상태에 머지해준다.

### 그래프 조립

```python
graph_builder = StateGraph(PipelineState)
graph_builder.add_node("get_topic_info", get_topic_info)
graph_builder.add_node("write_draft", write_draft)
graph_builder.add_edge(START, "get_topic_info")
graph_builder.add_edge("get_topic_info", "write_draft")
graph_builder.add_edge("write_draft", END)
graph = graph_builder.compile()
```

`START → get_topic_info → write_draft → END`. 직선 구조.

### 실행

```python
result = graph.invoke({"topic": "LangGraph and AI Agent Orchestration"})
```

`invoke()`에 초기 상태(`topic`만 있는 딕셔너리)를 넘기면,
나머지 필드(`background_info`, `draft`)는 노드들이 채워준다.

---

## 15.2 병렬 작성 노드 — "이 챕터의 핵심!"

### 새로운 import 정리

```python
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.types import Send
```

| import | 뭐하는 놈 |
|--------|----------|
| `operator.add` | 리스트 합치기 리듀서. `[a] + [b] = [a, b]` |
| `Annotated` | 타입에 메타데이터(리듀서)를 붙이는 파이썬 기능 |
| `BaseModel, Field` | Pydantic 모델. LLM 구조화 출력의 스키마 정의 |
| `Send` | 동적으로 병렬 노드를 발송하는 객체 |

### with_structured_output — LLM을 프로그래밍 가능하게

```python
class SectionPlan(BaseModel):
    title: str = Field(description="Section title")
    key_points: list[str] = Field(description="Key points to cover")

class BlogOutline(BaseModel):
    sections: list[SectionPlan] = Field(description="List of sections")
```

**비유:** 레스토랑 주문서.
- 일반 LLM 호출 = "맛있는 거 아무거나 주세요" → 문자열 응답
- `with_structured_output` = "주문서 양식에 맞춰 주세요" → Pydantic 객체 응답

```python
planner = llm.with_structured_output(BlogOutline)
outline = planner.invoke("3개 섹션 아웃라인 만들어줘")
# outline.sections[0].title → "Introduction"
# outline.sections[0].key_points → ["point1", "point2"]
```

`Field(description=...)`이 중요한 이유:
LLM이 이 설명을 읽고 각 필드에 뭘 넣어야 하는지 판단한다.

### 상태 정의 — 리듀서의 역할

```python
class PipelineState(TypedDict):
    topic: str
    background_info: str
    sections: Annotated[list[str], operator.add]  # ← 핵심!
    combined_draft: str
```

`Annotated[list[str], operator.add]`가 하는 일:

```
write_section #1 반환: {"sections": ["섹션1 내용"]}
write_section #2 반환: {"sections": ["섹션2 내용"]}
write_section #3 반환: {"sections": ["섹션3 내용"]}

리듀서(operator.add)가 자동으로:
state["sections"] = ["섹션1 내용"] + ["섹션2 내용"] + ["섹션3 내용"]
                   = ["섹션1 내용", "섹션2 내용", "섹션3 내용"]
```

리듀서가 없으면? 마지막 반환값만 남는다 (덮어쓰기).
리듀서가 있으면? 모든 반환값이 합쳐진다 (누적).

### SectionWriteInput — Send가 전달하는 데이터

```python
class SectionWriteInput(TypedDict):
    topic: str
    background_info: str
    section_title: str
    section_key_points: list[str]
```

이건 **메인 상태(`PipelineState`)가 아니다!**
`Send`가 `write_section` 노드에 **개별적으로** 전달하는 입력 데이터의 형태.

**비유:** `PipelineState`는 공장 전체의 주문서, `SectionWriteInput`은 개별 작업자의 작업 지시서.

### dispatch_writers — 이것은 노드가 아닙니다!

```python
# 노드 등록 (add_node)
graph_builder.add_node("get_topic_info", get_topic_info)     # ← 노드
graph_builder.add_node("write_section", write_section)       # ← 노드
graph_builder.add_node("combine_sections", combine_sections) # ← 노드

# 엣지 등록 (add_conditional_edges)
graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
#                                    ↑ 출발 노드        ↑ 라우터 함수!        ↑ 가능한 목적지
```

**핵심 구분:**

| 항목 | 등록 방법 | 역할 |
|------|----------|------|
| `get_topic_info` | `add_node()` | 실행되는 노드 |
| `write_section` | `add_node()` | 실행되는 노드 |
| `dispatch_writers` | `add_conditional_edges()` | **라우팅만 하는 함수** |

`dispatch_writers`는 Chapter 14의 `tools_condition`과 같은 위치.
`tools_condition`은 "도구 호출 필요해? → tools / END" 결정.
`dispatch_writers`는 "몇 개 섹션을 어떻게 보낼까? → Send 리스트" 결정.

**비유:**
- 노드 = 실제 일하는 직원
- 라우터 = 업무 배분하는 팀장 (직접 일 안 함, 누가 뭘 할지만 결정)

### dispatch_writers 함수 상세

```python
def dispatch_writers(state: PipelineState):
    # 1. LLM에게 구조화된 아웃라인 생성 요청
    planner = llm.with_structured_output(BlogOutline)
    outline = planner.invoke("3개 섹션 아웃라인 만들어줘")

    # 2. 각 섹션을 Send 객체로 만들어서 병렬 발송
    return [
        Send("write_section", {
            "topic": state["topic"],
            "background_info": state["background_info"],
            "section_title": section.title,
            "section_key_points": [kp for kp in section.key_points],
        })
        for section in outline.sections
    ]
```

1단계: `with_structured_output`으로 아웃라인 생성 (3개 섹션)
2단계: 각 섹션을 `Send` 객체로 만들어서 반환

`Send("write_section", data)` = "write_section 노드를 이 data로 실행해줘"

리스트로 3개 반환하면 → 3개가 **동시에** 실행됨.
이것이 **Map** 단계.

### write_section 노드 — Map의 실행 단위

```python
def write_section(input_data: SectionWriteInput):
    key_points = "\n".join(f"- {kp}" for kp in input_data["section_key_points"])
    response = llm.invoke(f"Write a section about '{input_data['section_title']}'...")
    return {"sections": [response.content]}
```

주의: 파라미터가 `state`가 아니라 `input_data: SectionWriteInput`.
`Send`가 전달한 데이터를 받는다.

반환값: `{"sections": [response.content]}` — **리스트 하나**에 섹션 내용 하나.
3개 노드가 각각 이걸 반환하면, `operator.add` 리듀서가 3개를 합친다.

### combine_sections 노드 — Reduce

```python
def combine_sections(state: PipelineState):
    combined = f"# {state['topic']}\n\n"
    combined += "\n\n".join(state["sections"])  # 3개 섹션 합치기
    return {"combined_draft": combined}
```

`state["sections"]`는 이 시점에 3개 섹션이 모두 들어있다 (리듀서 덕분).
이걸 하나의 문자열로 합쳐서 `combined_draft`에 저장.

### 전체 흐름 정리

```
get_topic_info → dispatch_writers(라우터) → Send("write_section", 데이터1)
                                          → Send("write_section", 데이터2)
                                          → Send("write_section", 데이터3)
                                                       ↓ (병렬 실행)
                                               write_section x3
                                                       ↓
                                               sections: [3개 결과]  ← operator.add
                                                       ↓
                                               combine_sections
                                                       ↓
                                                      END
```

---

## 15.3 Human-in-the-loop — "사람이 초고를 리뷰"

### 새로운 import 정리

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
```

| import | 뭐하는 놈 |
|--------|----------|
| `sqlite3` | SQLite 데이터베이스 연결 |
| `SqliteSaver` | LangGraph 체크포인터. 매 노드 실행마다 상태를 DB에 저장 |
| `interrupt` | 그래프를 멈추는 함수. 사용자에게 값을 반환 |
| `Command` | 멈춘 그래프를 재개하는 객체. `resume=`로 값 전달 |

### 상태 정의

```python
class ReviewState(TypedDict):
    topic: str
    draft: str
    feedback: str      # ← 새로 추가! (사람의 피드백)
    final_post: str    # ← 새로 추가! (최종본)
```

### human_feedback 노드 — interrupt의 동작 원리

```python
def human_feedback(state: ReviewState):
    feedback = interrupt(
        f"DRAFT FOR REVIEW\n{state['draft'][:500]}...\n"
        f"Please provide your feedback:"
    )
    return {"feedback": feedback}
```

`interrupt()`가 호출되면 일어나는 일:

1. **그래프 실행 중단** — 현재 상태가 SQLite에 자동 저장
2. **인자 값이 사용자에게 반환** — "DRAFT FOR REVIEW\n..." 문자열
3. **대기** — 사용자가 `Command(resume=...)` 보낼 때까지

재개되면:
4. `interrupt()`가 `Command(resume=...)`의 값을 **반환** → `feedback` 변수에 저장
5. 노드 함수가 정상적으로 `return {"feedback": feedback}` 실행

**비유:** 이메일 결재 시스템.
- `interrupt()` = 결재 요청 이메일 보내기 (문서 첨부)
- `Command(resume=)` = 상사가 코멘트 달고 "승인/반려"

### 체크포인터 설정 — 왜 필수인가

```python
conn = sqlite3.connect("pipeline_review.db", check_same_thread=False)
graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
```

`interrupt()`로 멈추면 Python 프로세스의 메모리 상태가 사라질 수 있다.
체크포인터가 상태를 **DB에 저장**해서, 나중에 `Command(resume=)`로 재개할 때
정확히 그 시점의 상태를 **복원**한다.

체크포인터 없이 `interrupt` 쓰면? → 에러 발생.

### 실행 흐름

```
invoke({"topic": "..."})
  → write_draft 실행 → draft 생성
  → human_feedback 실행 → interrupt()! → 멈춤!

  snapshot.next = ('human_feedback',)  ← "여기서 멈춰있다"

invoke(Command(resume="피드백 내용"))
  → interrupt()가 "피드백 내용" 반환
  → human_feedback 완료 → feedback 저장
  → finalize_post 실행 → final_post 생성
  → END

  snapshot.next = ()  ← "완료!"
```

### config와 thread_id

```python
config = {"configurable": {"thread_id": "review_1"}}
```

`thread_id`가 대화 세션을 구분한다.
같은 `thread_id`로 `Command(resume=)`하면 → 그 세션의 멈춘 지점에서 재개.

---

## 15.4 완성된 파이프라인 — "전체 통합"

### 상태 — 모든 필드가 모인 최종 상태

```python
class FullPipelineState(TypedDict):
    topic: str                                      # 입력 (15.1)
    background_info: str                             # 배경 조사 (15.1)
    sections: Annotated[list[str], operator.add]     # 병렬 섹션 (15.2)
    combined_draft: str                              # 합친 초고 (15.2)
    feedback: str                                    # 사람 피드백 (15.3)
    final_post: str                                  # 최종본 (15.3)
```

### 그래프 조립 — 5개 노드 + 1개 라우터

```python
# 5개 노드 등록
graph_builder.add_node("get_topic_info", get_topic_info)
graph_builder.add_node("write_section", write_section)
graph_builder.add_node("combine_sections", combine_sections)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("finalize_post", finalize_post)

# 엣지 연결
graph_builder.add_edge(START, "get_topic_info")
graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
graph_builder.add_edge("write_section", "combine_sections")
graph_builder.add_edge("combine_sections", "human_feedback")
graph_builder.add_edge("human_feedback", "finalize_post")
graph_builder.add_edge("finalize_post", END)
```

`dispatch_writers`는 `add_node`에 **없다**. 라우터 함수이기 때문.

### add_conditional_edges의 세 번째 인자

```python
graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
#                                                                        ↑ 이것!
```

세 번째 인자 `["write_section"]`은 **가능한 목적지 리스트**.
LangGraph에게 "dispatch_writers가 보낼 수 있는 곳은 write_section뿐이야"라고 알려주는 것.
이게 있어야 그래프 시각화와 검증이 정상 동작한다.

### 전체 실행 흐름

```
invoke({"topic": "Building AI Agents with LangGraph"})
  │
  ▼ get_topic_info: LLM으로 배경 조사
  │
  ▼ dispatch_writers(라우터): 아웃라인 3개 섹션 생성 → Send x3
  │
  ├─▶ write_section("Section 1")  ─┐
  ├─▶ write_section("Section 2")  ─┼─ 병렬 실행!
  └─▶ write_section("Section 3")  ─┘
  │
  ▼ combine_sections: sections[3개] → combined_draft
  │
  ▼ human_feedback: interrupt()! → 멈춤
  │
  (사용자가 Command(resume="피드백") 전달)
  │
  ▼ finalize_post: 피드백 반영 → final_post
  │
  ▼ END
```

---

## 15.5 프로덕션 배포 구조 — "노트북에서 서비스로"

### 디렉토리 구조

```
my_pipeline/
├── langgraph.json       # 진입점 설정
├── graph.py             # 그래프 정의 (노드 등록, 엣지 연결, compile)
├── state.py             # 상태 스키마 (TypedDict, Pydantic 모델)
├── nodes.py             # 노드 함수들 (get_topic_info, write_section 등)
├── prompts.py           # LLM 프롬프트 템플릿
└── requirements.txt     # 의존성
```

**비유:** 노트북 = 프로토타입 공방, 프로덕션 = 자동화 공장.
같은 물건을 만들지만, 공장에서는 공정(파일)별로 분리되어야 한다.

### langgraph.json — 핵심 설정 파일

```json
{
    "dependencies": ["."],
    "graphs": {
        "blog_pipeline": "./graph.py:graph"
    },
    "env": ".env"
}
```

| 키 | 역할 |
|----|------|
| `dependencies` | Python 패키지 설치 경로 |
| `graphs` | 그래프 진입점 매핑. `"이름": "파일경로:변수명"` |
| `env` | 환경변수 파일 경로 |

`"./graph.py:graph"` = graph.py 파일에서 `graph`라는 변수를 찾아라.

### graph.py의 핵심

```python
# graph.py — 노트북 코드와 동일, import만 다름
from state import FullPipelineState
from nodes import get_topic_info, dispatch_writers, write_section, ...

graph_builder = StateGraph(FullPipelineState)
# ... 노드 등록, 엣지 연결 ...
graph = graph_builder.compile()  # ← 이 변수가 langgraph.json에 등록됨
```

### 배포 옵션

| 방식 | 설명 | 비유 |
|------|------|------|
| `langgraph dev` | 로컬 개발 + Studio UI | 개발자 PC에서 테스트 |
| `langgraph up` | Docker 컨테이너 | 서버에 올리기 |
| LangGraph Cloud | 관리형 배포 + LangSmith 통합 | AWS Lambda 같은 것 |

---

## 핵심 개념 정리표

| 섹션 | 추가된 것 | 핵심 코드 | 비유 |
|------|----------|----------|------|
| **15.1** | TypedDict 상태 + 선형 그래프 | `PipelineState` + `add_edge` | 공장 직선 라인 |
| **15.2** | Send + Map-Reduce + Structured Output | `dispatch_writers` → `Send()` | 택배 분류 → 병렬 배달 |
| **15.3** | interrupt + Command + SqliteSaver | `interrupt(draft)` → `Command(resume=fb)` | 이메일 결재 시스템 |
| **15.4** | 전체 통합 | 5노드 + 1라우터 | 자동화 공장 전체 라인 |
| **15.5** | 파일 분리 + langgraph.json | `"graph.py:graph"` | 프로토타입 → 공장 |

---

## 자주 묻는 질문 (FAQ)

### Q: dispatch_writers가 왜 노드가 아닌가요?

`add_node`으로 등록하면 노드입니다. `add_conditional_edges`의 인자로 들어가면 라우터입니다.
`dispatch_writers`는 "다음에 어디로 갈지 결정"하는 역할만 합니다.
실제 일(섹션 작성)은 `write_section` 노드가 합니다.

### Q: Send로 보내면 정말 동시에 실행되나요?

LangGraph가 Send 리스트를 받으면, 의존성이 없는 것들은 병렬로 실행합니다.
정확한 동시성은 실행 환경(async/sync)에 따라 다르지만,
논리적으로는 "독립적인 N개 실행"으로 취급됩니다.

### Q: operator.add 대신 다른 리듀서를 쓸 수 있나요?

네. 어떤 `(old_value, new_value) -> merged_value` 형태의 함수든 가능합니다.
예를 들어 중복 제거, 정렬, 최대값 선택 등의 커스텀 로직을 넣을 수 있습니다.

### Q: interrupt 없이 자동으로 끝까지 돌리면 안 되나요?

가능합니다. `human_feedback` 노드를 빼고 `combine_sections → finalize_post`로 직접 연결하면 됩니다.
하지만 실무에서는 사람이 리뷰하는 게 거의 필수입니다.
AI가 완벽하지 않으니까요.
