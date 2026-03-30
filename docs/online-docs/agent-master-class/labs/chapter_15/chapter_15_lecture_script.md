# Chapter 15: LangGraph 프로젝트 파이프라인 — 강의 대본

---

## 오프닝 (2분)

> 자, Chapter 14에서 챗봇을 만들면서 LangGraph의 핵심 패턴을 익혔습니다.
>
> 오늘은 그 패턴들을 **전부 조합**해서 하나의 완성된 파이프라인을 만듭니다.
>
> 프로젝트 주제는 **Blog Post Generator** — 블로그 글 자동 생성기입니다.
>
> ```
> 15.0 Setup
> 15.1 기본 파이프라인       (선형 2-노드 그래프)
> 15.2 병렬 작성 노드        (Send API + Map-Reduce)
> 15.3 Human-in-the-loop    (interrupt + Command resume)
> 15.4 완성된 파이프라인     (전체 패턴 통합)
> 15.5 프로덕션 배포 구조    (langgraph.json, graph.py)
> ```
>
> Chapter 14와 마찬가지로, 각 섹션이 이전 위에 쌓이는 구조입니다.
> 마지막에는 주제 조사, 병렬 섹션 작성, 사람 리뷰, 최종본 생성까지 전부 자동화된 파이프라인이 나옵니다.
>
> 시작합시다.

---

## 15.0 Setup & Environment (2분)

### 셀 실행 전

> 환경 확인입니다.
> `.env`에서 API 키를 불러오고, `langgraph`와 `langchain` 버전을 체크합니다.
>
> Chapter 14와 동일한 환경입니다. 추가 패키지 없습니다.

### 셀 실행 후

> API 키와 버전이 정상 출력되면 준비 완료입니다.
> 에러가 나면 `uv sync` 또는 `pip install` 확인해주세요.

---

## 15.1 기본 파이프라인 — 2노드 선형 그래프 (8분)

### 개념 설명

> 가장 단순한 파이프라인부터 시작합니다.
>
> 구조는 이겁니다:
> ```
> START → get_topic_info → write_draft → END
> ```
>
> **챗봇과 뭐가 다르냐?**
> Chapter 14의 챗봇은 `MessagesState`를 사용했습니다 — 대화 메시지가 쌓이는 구조였죠.
> 파이프라인은 **`TypedDict`로 직접 상태를 설계**합니다.
> 왜냐하면, 파이프라인은 대화가 아니라 **데이터가 단계별로 변환**되는 구조이기 때문입니다.
>
> 상태를 봅시다:
> ```python
> class PipelineState(TypedDict):
>     topic: str             # 입력: 블로그 주제
>     background_info: str   # 1단계 출력: 배경 조사
>     draft: str             # 2단계 출력: 초고
> ```
>
> 각 노드가 상태의 한 필드를 채웁니다.
> `get_topic_info`가 `background_info`를 채우고, `write_draft`가 `draft`를 채웁니다.
> 데이터가 파이프처럼 흘러갑니다.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> ```python
> def get_topic_info(state: PipelineState):
>     topic = state["topic"]
>     response = llm.invoke(f"Provide a concise background summary about: {topic}...")
>     return {"background_info": response.content}
> ```
>
> 노드 함수의 패턴은 Chapter 13, 14와 동일합니다:
> 1. 상태에서 필요한 데이터를 꺼내고
> 2. LLM을 호출하고
> 3. 결과를 상태 필드에 반환
>
> `write_draft`도 같은 패턴입니다. `background_info`를 받아서 `draft`를 만듭니다.
>
> 그래프 조립도 익숙합니다:
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_draft", write_draft)
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_edge("get_topic_info", "write_draft")
> graph_builder.add_edge("write_draft", END)
> ```
>
> 선형 엣지로 순서대로 연결. 가장 기본적인 파이프라인입니다.
>
> 실행해봅시다.

### 셀 실행 후

> 결과를 보세요.
> `background_info`에 주제에 대한 배경 정보가 들어있고,
> `draft`에 그것을 바탕으로 쓴 블로그 초고가 들어있습니다.
>
> 단 2개 노드로 "조사 → 작성" 파이프라인이 완성됐습니다.
>
> 하지만 문제가 있죠? 블로그 글이 **한 덩어리**입니다.
> 실제로는 여러 섹션을 나눠서 쓰고 싶죠.
> 그래서 다음에 **병렬 작성**을 추가합니다.

---

## 15.2 병렬 작성 노드 — Send API + Map-Reduce (15분)

### 개념 설명

> 이번 섹션이 Chapter 15의 **하이라이트**입니다.
>
> 블로그 글을 N개 섹션으로 나누고, **각 섹션을 동시에 작성**하고, 합칩니다.
>
> ```
> get_topic_info → dispatch_writers ──→ write_section (x3) → combine_sections
>                                  ├─→ write_section
>                                  └─→ write_section
> ```
>
> 여기서 3가지 새로운 개념이 등장합니다:
>
> 1. **`Send` API** — 동적으로 병렬 노드를 발송하는 것. Chapter 13에서 배웠죠.
> 2. **`Annotated[list[str], operator.add]`** — 병렬 결과를 자동으로 합치는 리듀서
> 3. **`with_structured_output()`** — LLM이 Pydantic 모델 형태로 구조화된 출력을 생성
>
> 하나씩 봅시다.

### with_structured_output 설명

> 먼저 `with_structured_output`부터.
>
> 지금까지 LLM 응답은 항상 문자열이었죠? `response.content`로 텍스트를 받았습니다.
>
> 하지만 우리는 블로그 글의 **섹션 제목과 핵심 포인트**를 구조화된 데이터로 받고 싶습니다.
>
> 그래서 Pydantic 모델로 출력 형태를 정의합니다:
>
> ```python
> class SectionPlan(BaseModel):
>     title: str = Field(description="Section title")
>     key_points: list[str] = Field(description="Key points to cover")
>
> class BlogOutline(BaseModel):
>     sections: list[SectionPlan] = Field(description="List of sections")
> ```
>
> 그리고 LLM에게 이 형태로 출력하라고 지시합니다:
>
> ```python
> planner = llm.with_structured_output(BlogOutline)
> outline = planner.invoke("3개 섹션으로 아웃라인 만들어줘")
> ```
>
> 이러면 `outline.sections`가 `SectionPlan` 리스트로 나옵니다.
> 문자열 파싱 없이 바로 `.title`, `.key_points`로 접근 가능!
>
> 이것이 LLM을 **프로그래밍 가능한 도구**로 만드는 핵심입니다.

### dispatch_writers 설명 — 이것은 노드가 아닙니다!

> 자, 여기가 정말 중요합니다.
>
> **`dispatch_writers`는 노드가 아닙니다!**
>
> 코드를 보세요:
>
> ```python
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> ```
>
> `add_conditional_edges`의 두 번째 인자로 들어갑니다.
> 이건 Chapter 14의 `tools_condition`과 같은 위치입니다.
> 즉, **엣지의 라우팅 함수**입니다.
>
> `add_node`로 등록되지 않았습니다. 그래프 시각화에서도 노드로 나타나지 않습니다.
>
> **무슨 역할이냐?**
>
> `get_topic_info` 노드가 끝나면, LangGraph가 "다음에 어디로 갈까?"를 결정할 때
> `dispatch_writers` 함수를 호출합니다.
>
> 이 함수는 `Send` 객체의 리스트를 반환합니다:
>
> ```python
> def dispatch_writers(state: PipelineState):
>     planner = llm.with_structured_output(BlogOutline)
>     outline = planner.invoke("아웃라인 만들어줘")
>     return [
>         Send("write_section", {
>             "topic": state["topic"],
>             "section_title": section.title,
>             "section_key_points": section.key_points,
>         })
>         for section in outline.sections
>     ]
> ```
>
> 각 `Send` 객체가 `write_section` 노드의 **독립적인 실행 인스턴스**를 생성합니다.
> 섹션이 3개면 3개의 `write_section`이 **동시에** 실행됩니다.
>
> 비유하면:
> - `dispatch_writers`는 **택배 분류 센터**입니다
> - 소포(섹션 데이터)를 분류해서 각각 다른 배달원(`write_section`)에게 보냅니다
> - 분류 센터 자체는 배달(노드 실행)을 하지 않습니다

### Map-Reduce 패턴 설명

> 병렬 실행 결과는 어떻게 합쳐질까요?
>
> 상태 정의를 보세요:
>
> ```python
> class PipelineState(TypedDict):
>     sections: Annotated[list[str], operator.add]  # 리듀서!
> ```
>
> `operator.add`가 리듀서입니다.
> 각 `write_section`이 `{"sections": ["섹션 내용"]}`을 반환하면,
> LangGraph가 자동으로 리스트를 **합칩니다(concatenate)**.
>
> 3개 노드가 각각 1개씩 반환 → `sections`에 3개가 모임.
>
> 이것이 **Map-Reduce** 패턴입니다:
> - **Map** = `Send`로 병렬 발송하여 각각 처리
> - **Reduce** = `operator.add` 리듀서로 결과 병합
>
> `combine_sections` 노드가 `state["sections"]` (3개 다 모인 리스트)를 받아서
> 하나의 `combined_draft`로 합칩니다.

### 코드 실행

> 실행해봅시다.

### 셀 실행 후

> 결과를 보면:
> - "3 sections written in parallel!" — 3개 섹션이 병렬로 작성됨
> - 각 섹션이 `## 제목`으로 시작하는 구조
> - `combined_draft`에 전체 초고가 합쳐져 있음
>
> 한 노드가 순차적으로 3번 쓴 게 아닙니다.
> `Send`로 3개를 **동시에** 발송해서 병렬로 실행한 겁니다.
>
> 실무에서 섹션이 10개, 20개가 되어도 같은 패턴으로 확장됩니다.

---

## 15.3 Human-in-the-loop — interrupt + Command resume (10분)

### 개념 설명

> 15.2에서 초고가 자동으로 만들어졌습니다.
> 하지만 바로 발행하면 안 되겠죠? **사람이 리뷰**해야 합니다.
>
> Chapter 14.3에서 배운 `interrupt`와 `Command`를 여기에 적용합니다.
>
> ```
> write_draft → human_feedback(interrupt!) → [사람 피드백] → finalize_post → END
> ```
>
> 핵심은 3가지:
> - `interrupt(value)` — 그래프를 멈추고 사용자에게 초고를 보여줌
> - `Command(resume=feedback)` — 피드백을 전달하며 재개
> - `SqliteSaver` — 멈춘 사이 상태를 보존하는 체크포인터 (필수!)

### 코드 설명 (셀 실행 전)

> 상태를 봅시다:
>
> ```python
> class ReviewState(TypedDict):
>     topic: str
>     draft: str
>     feedback: str
>     final_post: str
> ```
>
> `feedback`와 `final_post` 필드가 추가됐습니다.
>
> `human_feedback` 노드가 핵심입니다:
>
> ```python
> def human_feedback(state: ReviewState):
>     feedback = interrupt(
>         f"DRAFT FOR REVIEW\n{state['draft'][:500]}...\n"
>         f"Please provide your feedback:"
>     )
>     return {"feedback": feedback}
> ```
>
> `interrupt()`가 호출되면:
> 1. 그래프가 **멈춤** — 상태는 SQLite에 자동 저장
> 2. `interrupt()`의 인자가 사용자에게 반환됨 (초고 미리보기)
> 3. 사용자가 피드백을 줄 때까지 대기
>
> 재개할 때:
> ```python
> result = graph.invoke(
>     Command(resume="Make it more concise. Add code examples."),
>     config=config,
> )
> ```
>
> `Command(resume=...)`의 값이 `interrupt()`의 **반환값**이 됩니다.
> 즉 `feedback = "Make it more concise..."` 가 되고, 이게 상태에 저장됩니다.
>
> `finalize_post` 노드가 초고 + 피드백을 LLM에 넘겨서 최종본을 생성합니다.
>
> 체크포인터가 **반드시** 필요합니다:
> ```python
> conn = sqlite3.connect("pipeline_review.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> 체크포인터 없이 `interrupt`를 쓰면 에러납니다.
> 멈춘 사이 상태를 어딘가에 저장해야 하니까요.

### 셀 실행 — 1단계: 초고 작성

> 첫 번째 셀을 실행합니다.

### 셀 실행 후 (1단계)

> `snapshot.next`가 `('human_feedback',)`입니다.
> `human_feedback` 노드에서 `interrupt` 걸려서 멈춰있다는 뜻입니다.
>
> 초고 미리보기가 출력되고, 피드백을 기다리고 있습니다.

### 셀 실행 — 2단계: 피드백 전달

> `Command(resume="Make it more concise. Add a code example.")`로 재개합니다.

### 셀 실행 후 (2단계)

> `Status: COMPLETE` — 파이프라인이 완료됐습니다.
> `snapshot.next`가 빈 튜플 `()`.
>
> `final_post`에 피드백이 반영된 최종본이 들어있습니다.
> 원래 초고보다 간결해지고, 코드 예제가 추가됐을 겁니다.
>
> 이것이 AI + 사람 협업의 기본 구조입니다.
> AI가 초안을 만들고, 사람이 방향을 잡아주고, AI가 다듬는 것.

---

## 15.4 완성된 파이프라인 — 모든 패턴 통합 (10분)

### 개념 설명

> 이제 15.1~15.3의 모든 패턴을 **하나로 합칩니다**.
>
> ```
> [START]
>    │
> [get_topic_info] ───── LLM 배경 조사
>    │
> [dispatch_writers] ──── with_structured_output + Send API (라우터!)
>    │
> [write_section] x N ─── 병렬 작성 (Map)
>    │
> [combine_sections] ──── 초고 병합 (Reduce)
>    │
> [human_feedback] ───── interrupt() 리뷰
>    │
> [finalize_post] ────── 피드백 반영 최종본
>    │
> [END]
> ```
>
> 코드가 길어 보이지만, 각 부분은 이미 다 배운 것입니다.
> 새로운 건 없습니다. **조합**만 하는 겁니다.

### 상태 설명

> 상태가 제일 큰 단서입니다:
>
> ```python
> class FullPipelineState(TypedDict):
>     topic: str                                      # 입력
>     background_info: str                             # 15.1에서 옴
>     sections: Annotated[list[str], operator.add]     # 15.2에서 옴 (Map-Reduce)
>     combined_draft: str                              # 15.2에서 옴
>     feedback: str                                    # 15.3에서 옴 (HITL)
>     final_post: str                                  # 15.3에서 옴
> ```
>
> 6개 필드가 파이프라인의 전체 데이터 흐름을 보여줍니다.
> 각 노드가 하나씩 채워나갑니다.

### 그래프 조립 설명

> 그래프 조립 코드를 봅시다:
>
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_section", write_section)
> graph_builder.add_node("combine_sections", combine_sections)
> graph_builder.add_node("human_feedback", human_feedback)
> graph_builder.add_node("finalize_post", finalize_post)
> ```
>
> **5개 노드**를 등록합니다.
> `dispatch_writers`는 여기에 **없습니다**! 노드가 아니니까요.
>
> ```python
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> graph_builder.add_edge("write_section", "combine_sections")
> graph_builder.add_edge("combine_sections", "human_feedback")
> graph_builder.add_edge("human_feedback", "finalize_post")
> graph_builder.add_edge("finalize_post", END)
> ```
>
> `dispatch_writers`는 `add_conditional_edges`의 **라우터 함수**로 등록됩니다.
> `get_topic_info` 실행 후, 이 라우터가 `Send` 객체들을 반환해서 병렬 노드를 발송합니다.
>
> 체크포인터도 빠지면 안 됩니다:
> ```python
> conn = sqlite3.connect("pipeline_full.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```

### 셀 실행 — 1단계: 파이프라인 실행

> 실행합니다. 주제는 "Building AI Agents with LangGraph".

### 셀 실행 후 (1단계)

> 출력을 봅시다:
> - `Dispatching 3 parallel writers...` — 3개 섹션 병렬 발송
> - `Paused at: ('human_feedback',)` — 리뷰 단계에서 멈춤
> - `Sections written: 3` — 3개 섹션 완료
> - 초고 미리보기가 출력됨
>
> 여기까지 자동으로: 주제 조사 → 아웃라인 생성 → 3개 섹션 병렬 작성 → 초고 병합
> 그리고 사람 리뷰를 기다리고 있습니다.

### 셀 실행 — 2단계: 피드백 전달

> 피드백을 줍니다:
> ```python
> Command(resume="Add a practical code example in each section. Make the tone more engaging.")
> ```

### 셀 실행 후 (2단계)

> `Status: COMPLETE` — 완료!
>
> 최종본에 피드백이 반영되어 있습니다.
> 코드 예제가 추가되고 톤이 바뀌었을 겁니다.
>
> 이것이 **완전한 AI 글쓰기 파이프라인**입니다:
> 1. AI가 주제를 조사하고
> 2. 구조를 계획하고
> 3. 섹션을 병렬로 작성하고
> 4. 사람이 리뷰하고
> 5. AI가 피드백을 반영해서 최종본 생성
>
> 6개 노드(정확히는 5개 노드 + 1개 라우터), 3가지 LangGraph 패턴의 조합입니다.

---

## 15.5 프로덕션 배포 구조 (5분)

### 개념 설명

> 노트북에서 실습하는 건 좋은데, 실제 서비스로 배포하려면?
>
> LangGraph는 프로덕션 배포를 위한 표준 구조를 제공합니다.
>
> ```
> my_pipeline/
> ├── langgraph.json       # 진입점 설정
> ├── graph.py             # 그래프 정의
> ├── state.py             # 상태 스키마
> ├── nodes.py             # 노드 함수들
> ├── prompts.py           # LLM 프롬프트 템플릿
> └── requirements.txt     # 의존성
> ```
>
> **노트북의 코드를 파일별로 분리**하는 겁니다.
>
> 핵심은 `langgraph.json`:
> ```json
> {
>     "dependencies": ["."],
>     "graphs": {
>         "blog_pipeline": "./graph.py:graph"
>     },
>     "env": ".env"
> }
> ```
>
> `"blog_pipeline": "./graph.py:graph"` — `graph.py` 파일의 `graph` 변수가 진입점이라는 선언.
>
> `graph.py`에는 우리가 노트북에서 만든 것과 동일한 코드가 들어갑니다.
> `StateGraph` 생성, 노드 등록, 엣지 연결, `compile()`.
>
> 다른 건 import 경로만 바뀝니다.

### 배포 옵션

> 배포 방식은 3가지입니다:
>
> | 방식 | 설명 |
> |------|------|
> | `langgraph dev` | 로컬 개발 + Studio UI (http://localhost:8123) |
> | `langgraph up` | Docker 컨테이너로 실행 |
> | LangGraph Cloud | LangSmith 통합 관리형 배포 |
>
> `langgraph dev`를 실행하면 Studio UI에서 그래프를 **시각적으로** 테스트할 수 있습니다.
> 노드 클릭하면 입출력 확인 가능하고, 인터럽트 지점에서 직접 피드백을 줄 수도 있습니다.
>
> 실제 프로덕션에서는 `langgraph up`으로 Docker화하거나,
> LangGraph Cloud에 배포해서 API 엔드포인트로 노출합니다.

---

## 마무리 (3분)

> 오늘 만든 것을 정리합시다.
>
> **Blog Post Generator 파이프라인:**
> 1. `get_topic_info` — LLM으로 주제 배경 조사
> 2. `dispatch_writers` — 라우터 함수, `with_structured_output`으로 아웃라인 생성, `Send`로 병렬 발송
> 3. `write_section` x N — 병렬 섹션 작성 (Map)
> 4. `combine_sections` — 결과 병합 (Reduce)
> 5. `human_feedback` — `interrupt()`로 사람 리뷰
> 6. `finalize_post` — 피드백 반영 최종본
>
> **사용된 LangGraph 패턴:**
>
> | 패턴 | 어디에 사용 |
> |------|-----------|
> | `TypedDict` 상태 | 파이프라인 데이터 흐름 설계 |
> | `Send` API | 병렬 섹션 발송 (Map) |
> | `operator.add` 리듀서 | 병렬 결과 자동 병합 (Reduce) |
> | `with_structured_output` | LLM 출력 구조화 (아웃라인) |
> | `conditional_edges` | 라우터 함수로 동적 분기 |
> | `interrupt` + `Command` | 사람 리뷰 워크플로우 |
> | `SqliteSaver` | 인터럽트 사이 상태 보존 |
>
> Chapter 13에서 기초를 배우고, Chapter 14에서 챗봇에 적용하고,
> 오늘 Chapter 15에서 **전체를 통합한 프로덕션급 파이프라인**을 완성했습니다.
>
> 핵심을 하나만 기억하세요:
> **`dispatch_writers`는 노드가 아니라 라우터 함수**입니다.
> `add_conditional_edges`에 등록되고, `Send` 객체를 반환해서 병렬 노드를 발송합니다.
>
> 다음 챕터에서 뵙겠습니다.
