# Chapter 16: 워크플로우 아키텍처 패턴 — 강의 대본

---

## 오프닝 (2분)

> 자, 오늘은 **워크플로우 아키텍처 패턴** 5가지를 배워봅니다.
>
> 지난 챕터에서 LangGraph의 기초 빌딩 블록을 배웠잖아요?
> 오늘은 그걸 조합해서 **실무에서 자주 쓰는 패턴**을 만들어봅니다.
>
> 오늘 할 내용:
>
> ```
> 16.1 Prompt Chaining        (순차적 LLM 호출)
> 16.2 Prompt Chaining + Gate (품질 검증 후 재시도)
> 16.3 Routing                (난이도별 동적 라우팅)
> 16.4 Parallelization        (병렬 실행 + 집계)
> 16.5 Orchestrator-Workers   (동적 작업 분배)
> ```
>
> 각 패턴은 이전 패턴 위에 쌓이는 구조입니다.
> Chaining이 기본이고, 거기에 Gate, Routing, Parallelization을 더하고,
> 마지막에 전부 합친 게 Orchestrator-Workers입니다.
>
> 시작합시다.

---

## 16.0 Setup & Environment (2분)

### 셀 실행 전

> 먼저 환경 확인합니다.
> `.env` 파일에서 API 키와 모델 이름을 불러옵니다.
>
> 오늘 필요한 패키지:
> - `langgraph >= 1.1` — 핵심 프레임워크
> - `langchain >= 1.2` — LLM 연동
> - `pydantic` — 구조화된 출력 스키마

### 셀 실행 후

> `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`이 정상 출력되면 준비 완료입니다.
> 다음 셀에서 `langgraph`와 `langchain` 버전도 확인합니다.

---

## 16.1 Prompt Chaining — 순차적 체이닝 (10분)

### 개념 설명

> 가장 기본적인 패턴입니다. **여러 LLM 호출을 순차적으로 연결**합니다.
>
> 이전 단계의 출력이 다음 단계의 입력이 됩니다.
>
> ```
> START → list_ingredients → create_recipe → describe_plating → END
> ```
>
> 요리 레시피를 예로 들겠습니다:
> 1. 요리 이름 → **재료 목록** 생성
> 2. 재료 목록 → **조리법** 생성
> 3. 조리법 → **플레이팅 설명** 생성
>
> 핵심은 `with_structured_output()`입니다.
> 첫 번째 노드에서 LLM 응답을 **Pydantic 모델로 강제**해서
> 다음 노드가 확실한 구조를 받도록 합니다.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> 먼저 State 정의:
> ```python
> class State(TypedDict):
>     dish: str                    # 입력: 요리 이름
>     ingredients: list[dict]      # 1단계 출력
>     recipe_steps: str            # 2단계 출력
>     plating_instructions: str    # 3단계 출력
> ```
>
> 그리고 Pydantic 모델로 구조화된 출력 스키마를 정의합니다:
> ```python
> class Ingredient(BaseModel):
>     name: str
>     quantity: str
>     unit: str
> ```
>
> `list_ingredients` 노드에서 핵심 부분:
> ```python
> structured_llm = llm.with_structured_output(IngredientsOutput)
> ```
>
> 이렇게 하면 LLM이 자유 텍스트가 아니라 **정확한 JSON 구조**로 응답합니다.
> `name`, `quantity`, `unit`이 빠질 수가 없어요.
>
> 나머지 두 노드(`create_recipe`, `describe_plating`)는 일반 `invoke()`를 씁니다.
> 이전 단계의 결과를 프롬프트에 넣어서 호출하는 거죠.
>
> 그래프 구성은 단순합니다. `add_edge`로 순서대로 연결.
> 자, 실행해봅시다.

### 셀 실행 후

> 결과를 봅시다.
>
> `=== Ingredients ===` 부분에 재료가 구조적으로 출력됐죠?
> `chickpeas: 1 can`, `tahini: 1/4 cup` 같은 형태.
> `with_structured_output()` 덕분에 깔끔한 구조를 받았습니다.
>
> 그 아래 Recipe와 Plating은 자유 텍스트입니다.
> 각 단계가 이전 단계의 결과를 입력으로 사용한 게 핵심입니다.
>
> **Chaining의 장점**: 복잡한 작업을 단계별로 나누면
> 각 단계의 프롬프트가 단순해지고, 결과의 품질이 올라갑니다.

### Exercise 16.1 (3분)

> **Exercise 1**: `dish` 값을 `"bibimbap"`이나 `"sushi"`로 바꿔보세요.
>
> **Exercise 2**: `with_structured_output()`을 쓴 노드와 일반 `invoke()`를 쓴 노드의 차이를 비교해보세요.
>
> **Exercise 3**: 4번째 단계(예: 와인 페어링 추천)를 추가해보세요.

---

## 16.2 Prompt Chaining + Gate — 조건부 게이트 (8분)

### 개념 설명

> 16.1의 Chaining에 **품질 검증**을 추가합니다.
>
> LLM 출력이 기준을 충족하지 않으면? **이전 단계를 재실행**합니다.
>
> ```
> START → list_ingredients → [gate: 3~8개?] → Yes → create_recipe → ...
>                              ↑               → No ─┘ (재시도)
> ```
>
> 게이트 함수는 단순합니다. `True`면 통과, `False`면 재시도.
>
> 이건 13.7에서 배운 **조건부 엣지**의 실전 활용입니다.
> `add_conditional_edges`에서 `True`는 다음 단계, `False`는 자기 자신으로 돌아갑니다.

### 코드 설명 (셀 실행 전)

> 16.1과 거의 같은 코드인데, 게이트 함수가 추가됐습니다:
>
> ```python
> def gate(state: State):
>     count = len(state["ingredients"])
>     if count > 8 or count < 3:
>         print(f"  GATE FAIL: {count} ingredients (need 3-8). Retrying...")
>         return False
>     print(f"  GATE PASS: {count} ingredients")
>     return True
> ```
>
> 재료가 3개 미만이거나 8개 초과면 실패. 다시 `list_ingredients`를 호출합니다.
>
> 그래프에서 핵심 부분:
> ```python
> graph_builder.add_conditional_edges(
>     "list_ingredients",
>     gate,
>     {True: "create_recipe", False: "list_ingredients"},
> )
> ```
>
> `False`일 때 자기 자신으로 돌아가는 **루프**가 생깁니다.
> 실행해봅시다.

### 셀 실행 후

> `Generated 8 ingredients` → `GATE PASS: 8 ingredients`
>
> 이번에는 한 번에 통과했습니다. LLM이 "5~8개"라고 요청했으니까 대부분 통과해요.
>
> 하지만 게이트 조건을 `len == 5`로 엄격하게 바꾸면?
> 재시도가 여러 번 발생하는 걸 볼 수 있습니다.
>
> **주의점**: 무한 루프 가능성이 있습니다.
> 실무에서는 반드시 `retry_count`를 State에 넣어서 최대 재시도 횟수를 제한해야 합니다.

### Exercise 16.2 (3분)

> **Exercise 1**: 게이트 조건을 `len(ingredients) == 5`로 바꿔서 재시도 횟수를 관찰하세요.
>
> **Exercise 2**: `retry_count` 필드를 State에 추가하고 최대 3회로 제한해보세요.
>
> **Exercise 3**: 게이트 패턴이 유용한 시나리오를 생각해보세요 — 코드 구문 검사, 번역 언어 확인, 필수 필드 검증 등.

---

## 16.3 Routing — 동적 라우팅 (10분)

### 개념 설명

> 세 번째 패턴입니다. **입력에 따라 다른 경로로 라우팅**합니다.
>
> 질문의 난이도를 LLM이 평가하고, 난이도별로 다른 모델 노드로 보냅니다:
>
> ```
> START → assess_difficulty → easy   → dumb_node   (GPT-3.5)
>                           → medium → average_node (GPT-4o)
>                           → hard   → smart_node   (GPT-5)
> ```
>
> 핵심이 두 가지입니다:
>
> 1. **Structured Output + Literal**: LLM 응답을 `"easy"`, `"medium"`, `"hard"` 세 값으로 제한
> 2. **Command**: 13.9에서 배운 Command로 라우팅 + 상태 업데이트를 한 번에
>
> 실무에서는 비용 절감에 아주 유용합니다.
> 쉬운 질문에 비싼 모델을 쓸 필요가 없으니까요.

### 코드 설명 (셀 실행 전)

> `DifficultyResponse`를 봅시다:
> ```python
> class DifficultyResponse(BaseModel):
>     difficulty_level: Literal["easy", "medium", "hard"]
> ```
>
> `Literal`로 LLM이 이 세 값 중 하나만 반환하도록 강제합니다.
>
> `assess_difficulty` 노드:
> ```python
> def assess_difficulty(state: State):
>     structured_llm = llm.with_structured_output(DifficultyResponse)
>     response = structured_llm.invoke(...)
>     level = response.difficulty_level
>     goto_map = {"easy": "dumb_node", "medium": "average_node", "hard": "smart_node"}
>     return Command(goto=goto_map[level], update={"difficulty": level})
> ```
>
> `Command`로 **어디로 갈지**와 **상태 업데이트**를 동시에 처리합니다.
> `add_conditional_edges`가 필요 없어요.
>
> 그래프 구성에서 `destinations` 파라미터가 새로 나옵니다:
> ```python
> graph_builder.add_node(
>     "assess_difficulty", assess_difficulty,
>     destinations=("dumb_node", "average_node", "smart_node"),
> )
> ```
>
> Command를 쓸 때 가능한 목적지를 알려주는 겁니다.
>
> 자, 쉬운 질문과 어려운 질문 두 개를 테스트합니다.

### 셀 실행 — 쉬운 질문

> `"What is the capital of France?"` — 프랑스의 수도.
>
> ```
> Difficulty: easy → dumb_node
> Model: gpt-3.5 (simulated)
> Answer: The capital of France is Paris.
> ```
>
> 간단한 질문이니까 `easy`로 판단하고 `dumb_node`로 라우팅됐습니다.

### 셀 실행 — 어려운 질문

> `"Explain the economic implications of quantum computing on global supply chains"`
>
> ```
> Difficulty: hard → smart_node
> Model: gpt-5 (simulated)
> ```
>
> 복잡한 질문이니까 `hard`로 판단하고 `smart_node`로 갔습니다.
>
> 지금은 같은 모델로 시뮬레이션하지만, 실무에서는 실제로 다른 모델을 쓰면
> **비용을 크게 절감**할 수 있습니다. 쉬운 건 저렴한 모델, 어려운 건 고급 모델.

### Exercise 16.3 (3분)

> **Exercise 1**: 다양한 난이도의 질문을 넣어보고 라우팅이 올바른지 확인하세요.
>
> **Exercise 2**: `model_used` 필드로 실제 경로를 검증하세요.
>
> **Exercise 3**: 모델 선택 외에 다른 프롬프트 템플릿으로 라우팅하는 시나리오를 설계해보세요.

---

## 16.4 Parallelization — 병렬 실행 (10분)

### 개념 설명

> 네 번째 패턴입니다. **독립적인 LLM 호출을 동시에 실행**합니다.
>
> ```
>         → get_summary ────────┐
>         → get_sentiment ──────┤
> START → → get_key_points ─────┤→ get_final_analysis → END
>         → get_recommendation ─┘
> ```
>
> **Fan-out / Fan-in** 패턴이라고도 합니다:
> - Fan-out: START에서 4개 노드로 동시 출발
> - Fan-in: 4개 노드가 모두 끝나면 하나로 합류
>
> LangGraph에서 이걸 구현하는 방법이 놀랍게도 간단합니다.
> `START`에서 여러 노드로 엣지를 연결하면 **자동으로 병렬 실행**됩니다.
> 여러 노드에서 하나로 연결하면 **자동으로 join** (모두 완료될 때까지 대기).
>
> 13.8의 Send API와 다른 점: 여기는 **서로 다른 함수**를 동시에 실행합니다.
> Send API는 **같은 함수를 다른 입력**으로 동시에 실행하죠.

### 코드 설명 (셀 실행 전)

> State에 6개 필드가 있습니다:
> ```python
> class State(TypedDict):
>     document: str          # 입력 문서
>     summary: str           # 병렬 노드 1
>     sentiment: str         # 병렬 노드 2
>     key_points: str        # 병렬 노드 3
>     recommendation: str    # 병렬 노드 4
>     final_analysis: str    # 집계 결과
> ```
>
> 4개의 병렬 노드는 모두 같은 `document`를 읽지만, **다른 필드에 씁니다**.
> 서로 겹치지 않으니까 동시에 실행해도 충돌이 없습니다.
>
> 그래프 구성의 핵심:
> ```python
> # Fan-out: START에서 4개로
> graph_builder.add_edge(START, "get_summary")
> graph_builder.add_edge(START, "get_sentiment")
> graph_builder.add_edge(START, "get_key_points")
> graph_builder.add_edge(START, "get_recommendation")
>
> # Fan-in: 4개에서 1개로
> graph_builder.add_edge("get_summary", "get_final_analysis")
> graph_builder.add_edge("get_sentiment", "get_final_analysis")
> graph_builder.add_edge("get_key_points", "get_final_analysis")
> graph_builder.add_edge("get_recommendation", "get_final_analysis")
> ```
>
> 이것만으로 병렬 + join이 완성됩니다. 실행해봅시다.

### 셀 실행 후

> 출력을 보세요:
> ```
> [parallel] get_key_points started
> [parallel] get_recommendation started
> [parallel] get_sentiment started
> [parallel] get_summary started
> ```
>
> 4개가 **거의 동시에** 시작됐죠? 순서가 랜덤인 것도 병렬 실행의 증거입니다.
>
> 그 다음:
> ```
> [join] get_final_analysis started
> ```
>
> 4개가 모두 끝난 후에야 `get_final_analysis`가 실행됩니다.
>
> Final Analysis를 보면 summary, sentiment, key_points, recommendation을
> 모두 종합한 분석 결과가 나옵니다.
>
> **성능 관점**: 순차 실행이면 4번의 LLM 호출 시간이 합산됩니다.
> 병렬 실행이면 **가장 느린 하나의 시간**만 걸립니다. 3~4배 빨라질 수 있어요.

### Exercise 16.4 (3분)

> **Exercise 1**: 5번째 병렬 노드(예: `get_risks`)를 추가하고 결과가 집계에 포함되는지 확인하세요.
>
> **Exercise 2**: `graph.stream()`으로 실행해서 병렬 시작을 직접 관찰하세요.
>
> **Exercise 3**: 순차 실행과 병렬 실행의 총 시간을 비교해보세요.

---

## 16.5 Orchestrator-Workers — 동적 작업 분배 (10분)

### 개념 설명

> 마지막 패턴입니다. 가장 강력하고, 이전 패턴들을 모두 결합합니다.
>
> ```
> START → orchestrator → [Send x N] → worker x N → synthesizer → END
> ```
>
> 핵심 아이디어:
> 1. **Orchestrator**: 주제를 분석해서 섹션 목록을 생성 (몇 개가 될지 미리 모름)
> 2. **Send API**: 섹션 수만큼 Worker를 동적으로 병렬 실행
> 3. **Synthesizer**: 모든 Worker 결과를 합쳐서 최종 보고서 생성
>
> 16.4의 Parallelization과 다른 점:
> - 16.4는 **노드 수가 고정** (코드에서 4개를 정의)
> - 16.5는 **노드 수가 동적** (LLM이 결정, Send API로 실행)
>
> 이건 13.8 Send API + 16.4 Parallelization의 **실전 결합**입니다.

### 코드 설명 (셀 실행 전)

> State를 봅시다:
> ```python
> class State(TypedDict):
>     topic: str
>     sections: list[str]
>     results: Annotated[list[dict], operator.add]  # 리듀서!
>     final_report: str
> ```
>
> `results`에 `operator.add` 리듀서가 있습니다.
> Worker들이 병렬로 결과를 반환하면 자동으로 누적됩니다.
>
> **Orchestrator**:
> ```python
> def orchestrator(state: State):
>     structured_llm = llm.with_structured_output(Sections)
>     response = structured_llm.invoke(
>         f"Break down this topic into 3-5 research sections: {state['topic']}"
>     )
>     return {"sections": response.sections}
> ```
>
> LLM이 주제를 3~5개 섹션으로 나눕니다. `with_structured_output()`으로 구조화.
>
> **Dispatcher** (Send API):
> ```python
> def dispatch_workers(state: State):
>     return [Send("worker", section) for section in state["sections"]]
> ```
>
> 섹션 수만큼 `Send`를 반환. 5개면 5개의 Worker가 병렬 실행됩니다.
>
> **Worker**:
> ```python
> def worker(section: str):  # State가 아니라 str을 받는다!
>     response = llm.invoke(f"Write a brief paragraph about: {section}")
>     return {"results": [{"section": section, "content": response.content}]}
> ```
>
> 13.8에서 배운 것처럼, Send API로 전달되는 값은 **State가 아닌 커스텀 입력**입니다.
>
> **Synthesizer**: 모든 Worker 결과를 합쳐서 최종 보고서.
>
> 실행해봅시다.

### 셀 실행 후

> ```
> Orchestrator: 5 sections
>   - Introduction to AI in Healthcare
>   - Applications of AI in Clinical Settings
>   - Ethical Considerations and Challenges
>   - Impact on Patient Outcomes
>   - Future Trends and Predictions
> ```
>
> Orchestrator가 5개 섹션으로 나눴습니다.
>
> 그 다음 Worker 5개가 병렬로 실행되고:
> ```
> Worker done: Introduction to AI in Healthcare...
> Worker done: Future Trends and Predictions...
> Worker done: Impact on Patient Outcomes...
> ```
>
> 순서가 랜덤인 거 보이시죠? 병렬 실행이니까요.
>
> 마지막에 Final Report가 5개 섹션을 합친 완성된 보고서입니다.
>
> **이게 바로 Orchestrator-Workers 패턴의 힘입니다.**
> 주제만 바꾸면 자동으로 다른 구조의 보고서가 생성됩니다.
> 섹션 수도 내용도 LLM이 결정하니까요.

### Exercise 16.5 (3분)

> **Exercise 1**: `topic`을 바꿔서 다른 섹션 분할이 나오는지 확인하세요.
>
> **Exercise 2**: Worker에 `time.sleep(1)`을 추가해서 병렬 실행의 효과를 체감하세요.
>
> **Exercise 3**: Worker가 `BaseModel`로 구조화된 출력을 반환하도록 수정해보세요.

---

## 종합 실습 안내 (3분)

> 노트북 마지막에 4개의 종합 과제가 있습니다.
>
> **과제 1** (★★☆): 번역 체인 + 게이트 — 한국어→영어→일본어 순차 번역, 50자 미만이면 재시도
> **과제 2** (★★☆): 감정 기반 라우팅 — 사용자 메시지 감정 분석 후 positive/negative/neutral 경로
> **과제 3** (★★★): 코드 리뷰 병렬 분석 — 보안/성능/가독성/테스트 4개 관점 동시 분석 후 종합
> **과제 4** (★★★): Orchestrator-Workers 블로그 — 주제→목차 생성→섹션별 워커→합성
>
> 과제 1~2는 기본, 3~4는 도전 과제입니다.
> 시간 배분: 쉬운 것 10분, 어려운 것 15분씩.

---

## 마무리 (2분)

> 오늘 배운 5가지 패턴을 정리합니다.
>
> | 패턴 | 구조 | 핵심 |
> |------|------|------|
> | Prompt Chaining | A → B → C | 순차 연결, structured output으로 안정성 확보 |
> | Chaining + Gate | A → [검증] → B / ↩ A | 품질 검증 후 재시도, 무한 루프 주의 |
> | Routing | A → 분기 → B or C or D | Command + Literal로 안전한 라우팅 |
> | Parallelization | Fan-out → Fan-in | 엣지만 연결하면 자동 병렬, 성능 3~4배 향상 |
> | Orchestrator-Workers | O → Send x N → S | 동적 병렬, Send API + Reducer로 결과 누적 |
>
> 이 5가지는 **실무에서 가장 자주 쓰는 패턴**입니다.
> 대부분의 AI 워크플로우는 이 패턴들의 조합으로 만들 수 있습니다.
>
> 다음 챕터에서 더 고급 패턴을 배워봅니다.
> 수고하셨습니다.
