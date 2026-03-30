# Chapter 17: LangGraph 워크플로우 테스팅 — 강의 대본

---

## 오프닝 (2분)

> 자, 지금까지 LangGraph로 워크플로우를 만들어봤습니다.
>
> 그런데 만든 그래프가 **제대로 동작하는지** 어떻게 확인하죠?
>
> 오늘은 LangGraph 워크플로우를 **pytest로 체계적으로 테스트**하는 방법을 배웁니다.
>
> 순서는 이렇습니다:
>
> ```
> 17.0 Setup
> 17.1 이메일 처리 그래프    (규칙 기반, 결정적)
> 17.2 Pytest 기초          (%%writefile, parametrize)
> 17.3 노드 단위 테스트      (graph.nodes, update_state)
> 17.4 AI 노드 전환          (with_structured_output)
> 17.5 AI 테스트 전략        (Range-based assertions)
> 17.6 LLM-as-a-Judge       (Golden examples, 유사도 점수)
> ```
>
> 핵심 흐름을 미리 말씀드리면:
> **정확히 일치 → 범위로 검증 → LLM이 판정**
>
> 규칙 기반은 `assert ==`로 끝나지만,
> AI 기반은 출력이 매번 달라서 새로운 전략이 필요합니다.
>
> 이 진화를 하나씩 체험해봅시다. 시작합니다.

---

## 17.0 Setup & Environment (2분)

### 셀 실행 전

> 먼저 환경을 확인합니다.
> `.env`에서 API 키를 불러오고, `langgraph`, `langchain`, `pytest` 버전을 체크합니다.
>
> 이전 챕터와 동일한 환경이고, 추가로 `pytest`가 필요합니다.

### 셀 실행 후

> API 키와 버전이 정상 출력되면 준비 완료입니다.
> `pytest`가 없으면 `pip install pytest`로 설치해주세요.

---

## 17.1 이메일 처리 그래프 — Rule-based Email Classifier (8분)

### 개념 설명

> 먼저 **테스트하기 쉬운 그래프**부터 만들겠습니다.
>
> 왜 규칙 기반부터 시작하느냐?
> 규칙 기반은 **결정적(deterministic)**이기 때문입니다.
> 같은 입력 → 항상 같은 출력. 테스트가 간단합니다.
>
> 그래프 구조는 직선형입니다:
>
> ```
> START → categorize_email → assign_priority → generate_response → END
> ```
>
> 세 개 노드가 하는 일:
> - `categorize_email` — 키워드로 카테고리 분류 (urgent / spam / normal)
> - `assign_priority` — 카테고리 기반 우선순위 (high / low / medium)
> - `generate_response` — 카테고리별 정형 응답 생성
>
> 모두 LLM 없이 `if-elif-else`로 동작합니다.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> 먼저 상태 정의:
>
> ```python
> class EmailState(TypedDict):
>     email: str
>     category: str
>     priority: str
>     response: str
> ```
>
> 이메일이 그래프를 흐르면서 `category`, `priority`, `response`가 채워지는 구조입니다.
>
> `categorize_email` 함수를 보면:
>
> ```python
> def categorize_email(state: EmailState) -> dict:
>     email = state["email"].lower()
>     if "urgent" in email or "asap" in email:
>         return {"category": "urgent"}
>     elif "offer" in email or "discount" in email:
>         return {"category": "spam"}
>     return {"category": "normal"}
> ```
>
> 소문자로 바꾼 다음 키워드를 찾습니다.
> "urgent"나 "asap"이 있으면 urgent, "offer"나 "discount"가 있으면 spam, 나머지는 normal.
>
> `assign_priority`는 카테고리 → 우선순위 매핑:
> urgent → high, spam → low, 나머지 → medium.
>
> `generate_response`는 카테고리별 템플릿 응답을 반환합니다.
>
> 그래프 구성은:
>
> ```python
> builder.add_edge(START, "categorize_email")
> builder.add_edge("categorize_email", "assign_priority")
> builder.add_edge("assign_priority", "generate_response")
> builder.add_edge("generate_response", END)
> ```
>
> 직선형 파이프라인. 분기 없이 순서대로 실행됩니다.

### 셀 실행 후

> 실행해봅시다.
>
> ```
> "URGENT: Server is down, fix ASAP!" → category: urgent, priority: high
> "Special offer! 50% discount today!" → category: spam, priority: low
> "Hi, I have a question about my order." → category: normal, priority: medium
> ```
>
> 예상대로 동작하죠?
> 이제 이걸 **수동 확인 대신 자동 테스트**로 바꿀 겁니다.

---

## 17.2 Pytest 기초 — parametrize + %%writefile (8분)

### 개념 설명

> 노트북에서 pytest를 실행하는 패턴이 있습니다. 3단계:
>
> 1. `%%writefile main.py` — 테스트 대상 코드를 `.py` 파일로 저장
> 2. `%%writefile tests.py` — 테스트 코드를 `.py` 파일로 저장
> 3. `!pytest tests.py -v` — 테스트 실행
>
> **`%%writefile`이 뭐냐?**
> Jupyter 매직 커맨드입니다. 셀 내용을 그대로 파일로 저장합니다.
> 셀 첫 줄에 `%%writefile 파일명`을 쓰면 됩니다.
>
> 왜 이게 필요하냐?
> pytest는 `.py` 파일만 실행합니다. 노트북 셀은 직접 못 읽어요.
> 그래서 `%%writefile`로 코드를 파일로 빼는 겁니다.
>
> 그리고 `@pytest.mark.parametrize` — 여러 입력/기대값을 한 번에 테스트하는 데코레이터입니다.
> 하나의 테스트 함수로 6개, 10개 케이스를 돌릴 수 있습니다.

### 코드 설명 (셀 실행 전)

> 먼저 `%%writefile main.py`로 이메일 그래프 코드를 파일로 저장합니다.
> 17.1에서 만든 코드 그대로예요. 차이가 없습니다.
>
> 다음 `%%writefile tests.py`를 봅시다.
>
> ```python
> @pytest.mark.parametrize(
>     "email, expected_category, expected_priority",
>     [
>         ("URGENT: Server down!", "urgent", "high"),
>         ("Fix this ASAP please", "urgent", "high"),
>         ("Special offer! Buy now!", "spam", "low"),
>         ("Get 50% discount today", "spam", "low"),
>         ("Hi, I have a question", "normal", "medium"),
>         ("Meeting tomorrow at 3pm", "normal", "medium"),
>     ],
> )
> def test_email_pipeline(email, expected_category, expected_priority):
>     graph = build_email_graph()
>     result = graph.invoke({"email": email})
>     assert result["category"] == expected_category
>     assert result["priority"] == expected_priority
>     assert len(result["response"]) > 0
> ```
>
> `@pytest.mark.parametrize`가 하는 일:
> - 첫 번째 인자: 파라미터 이름들 (쉼표로 구분)
> - 두 번째 인자: 테스트 케이스 리스트 (튜플의 리스트)
> - 6개 튜플 → 테스트가 6번 실행됩니다
>
> `assert`는 조건이 False이면 테스트 실패.
> `result["category"] == expected_category` — 정확히 일치해야 통과.
>
> 그 아래 개별 노드 테스트도 있습니다:
>
> ```python
> def test_categorize_urgent():
>     result = categorize_email({"email": "This is URGENT!", ...})
>     assert result["category"] == "urgent"
> ```
>
> 노드 함수를 직접 호출해서 테스트. 이게 **단위 테스트의 기본**입니다.

### 셀 실행 후

> `!pytest tests.py -v`를 실행하면:
>
> ```
> tests.py::test_email_pipeline[URGENT: Server down!-urgent-high]   PASSED
> tests.py::test_email_pipeline[Fix this ASAP please-urgent-high]   PASSED
> ...
> tests.py::test_priority_mapping                                    PASSED
> tests.py::test_response_templates                                  PASSED
> ```
>
> 전부 PASSED. 초록색!
>
> `-v`는 verbose. 각 테스트 케이스를 한 줄씩 보여줍니다.
> parametrize 덕분에 테스트 이름에 입력값이 표시되어서 어떤 케이스인지 바로 알 수 있습니다.
>
> 이게 규칙 기반 테스트의 장점입니다: **`assert ==`로 정확히 검증 가능.**

---

## 17.3 노드 단위 테스트 — graph.nodes + update_state (8분)

### 개념 설명

> 17.2에서는 전체 그래프를 통으로 테스트했습니다.
> 하지만 실무에서는 **특정 노드만 따로** 테스트해야 할 때가 많습니다.
>
> 두 가지 방법이 있습니다:
>
> **방법 1: `graph.nodes["name"].invoke(state)`**
> - 컴파일된 그래프에서 특정 노드만 직접 실행
> - 그래프를 통과하지 않고 노드 함수를 직접 호출하는 것과 같음
>
> **방법 2: `graph.update_state(config, values, as_node="name")`**
> - 특정 노드가 실행된 것처럼 상태를 강제 주입
> - 그 다음 노드부터 **부분 실행** 가능
> - `MemorySaver` 체크포인터가 필요

### 코드 설명 — 방법 1 (셀 실행 전)

> ```python
> test_state = {"email": "URGENT: Need help ASAP", "category": "", "priority": "", "response": ""}
>
> cat_result = graph.nodes["categorize_email"].invoke(test_state)
> print(f"categorize_email result: {cat_result}")
> ```
>
> `graph.nodes`는 딕셔너리입니다. 키가 노드 이름, 값이 노드 객체.
> `.invoke(state)`로 그 노드만 실행합니다.
>
> 중요한 점: **그래프 전체를 돌리는 게 아닙니다.**
> `categorize_email` 하나만 실행하고 결과를 받습니다.
>
> 이렇게 하면 "내 노드 함수가 올바른 값을 반환하는가?"를 독립적으로 검증할 수 있습니다.

### 셀 실행 후 — 방법 1

> ```
> categorize_email result: {'category': 'urgent'}
> assign_priority result: {'priority': 'high'}
> generate_response result: {'response': 'This email has been classified as spam.'}
> ```
>
> 각 노드가 독립적으로 잘 동작합니다.

### 코드 설명 — 방법 2: update_state (셀 실행 전)

> 이번엔 `update_state` 방법입니다.
>
> 먼저 체크포인터가 있는 그래프를 만듭니다:
>
> ```python
> from langgraph.checkpoint.memory import MemorySaver
> graph_mem = builder.compile(checkpointer=MemorySaver())
> ```
>
> `MemorySaver`는 메모리 기반 체크포인터. SQLite 대신 메모리에 상태를 저장합니다.
> 테스트할 때 가벼워서 좋습니다.
>
> 그 다음:
>
> ```python
> config2 = {"configurable": {"thread_id": "test_partial_2"}}
>
> # 먼저 전체 실행
> graph_mem.invoke({"email": "Hello, normal email"}, config=config2)
>
> # categorize_email이 "urgent"를 반환한 것처럼 상태 강제 주입
> graph_mem.update_state(
>     config2,
>     {"category": "urgent"},
>     as_node="categorize_email",
> )
> ```
>
> `as_node="categorize_email"` — 이게 핵심입니다!
> "categorize_email 노드가 이 값을 반환한 것처럼 처리해줘"라는 뜻입니다.
>
> 원래 "normal"이었던 카테고리를 "urgent"로 강제로 바꿨습니다.
>
> ```python
> result_partial = graph_mem.invoke(None, config=config2)
> ```
>
> `invoke(None)` — 새 입력 없이 **나머지 노드만 실행**.
> `assign_priority → generate_response`가 실행됩니다.
> category가 "urgent"이므로 priority는 "high"가 됩니다.

### 셀 실행 후 — 방법 2

> ```
> 주입 후 상태: category=urgent
> 부분 실행 결과: category=urgent, priority=high
> assert 통과! update_state로 부분 실행 성공
> ```
>
> 이게 왜 유용하냐?
>
> 실무에서 3번째 노드에 버그가 의심될 때,
> 앞의 두 노드를 굳이 실행할 필요 없이
> **원하는 상태를 주입하고 3번째만 테스트**할 수 있습니다.
>
> 테스트 시간도 줄고, 원인 파악도 빨라집니다.

---

## 17.4 AI 노드 전환 — LLM + with_structured_output (8분)

### 개념 설명

> 자, 여기서 패러다임이 바뀝니다.
>
> 지금까지는 규칙 기반이라 `if-elif-else`로 결정적이었습니다.
> 이제 **같은 그래프 구조에서 노드 함수만 AI로 교체**합니다.
>
> 그래프 구조는 동일:
> ```
> START → categorize_email → assign_priority → generate_response → END
> ```
>
> 바뀌는 건 노드 내부의 로직뿐입니다.
>
> 핵심 기술: **`with_structured_output`**
>
> LLM은 기본적으로 자유 텍스트를 반환합니다.
> 하지만 우리 그래프는 `"urgent"`, `"spam"`, `"normal"` 중 하나가 필요합니다.
>
> `with_structured_output(Pydantic모델)` — LLM 출력을 Pydantic 모델로 강제합니다.
> LLM이 뭘 말하든 정해진 필드와 타입으로 파싱됩니다.

### 코드 설명 (셀 실행 전)

> Pydantic 출력 스키마를 봅시다:
>
> ```python
> class CategoryOutput(BaseModel):
>     category: Literal["urgent", "spam", "normal"] = Field(
>         description="The email category"
>     )
> ```
>
> `Literal["urgent", "spam", "normal"]` — 이 3개 값만 허용!
> LLM이 "critical"이라고 하고 싶어도 이 3개 중 하나로 강제됩니다.
>
> `Field(description=...)` — LLM에게 이 필드가 뭔지 설명합니다.
>
> 같은 패턴으로 `PriorityOutput`, `ResponseOutput`도 정의합니다.
>
> Structured LLM 생성:
>
> ```python
> category_llm = llm.with_structured_output(CategoryOutput)
> priority_llm = llm.with_structured_output(PriorityOutput)
> response_llm = llm.with_structured_output(ResponseOutput)
> ```
>
> 이제 `category_llm.invoke("...")` 하면 `CategoryOutput` 객체가 반환됩니다.
> 문자열이 아니라 `.category` 속성이 있는 객체.
>
> AI 노드 함수:
>
> ```python
> def ai_categorize_email(state: EmailState) -> dict:
>     result = category_llm.invoke(
>         f"Classify this email into one of: urgent, spam, normal.\n\nEmail: {state['email']}"
>     )
>     return {"category": result.category}
> ```
>
> 규칙 기반과 비교하면:
> - 규칙 기반: `if "urgent" in email` → 키워드 매칭
> - AI 기반: LLM이 문맥을 이해하고 판단
>
> "서버가 다운됐다"는 "urgent"라는 키워드가 없어도 AI는 urgent로 분류할 수 있습니다.

### 셀 실행 후

> ```
> "URGENT: Production database is corrupted" → Category: urgent, Priority: high
> "Congratulations! You won $1,000,000!" → Category: spam, Priority: low
> ```
>
> AI가 잘 분류합니다.
>
> 그런데 문제! **같은 이메일을 다시 실행하면 결과가 달라질 수 있습니다.**
> AI 출력은 **비결정적(non-deterministic)**이기 때문입니다.
>
> 이건 `assert ==`로 테스트할 수 없다는 뜻입니다.
> 그래서 새로운 테스트 전략이 필요합니다.

---

## 17.5 AI 테스트 전략 — Range-based Assertions (8분)

### 개념 설명

> AI 출력은 매번 달라집니다.
> 그러면 어떻게 테스트하죠?
>
> **범위(range)로 검증합니다!**
>
> 4가지 전략:
>
> 1. **유효값 범위**: `assert result in ["urgent", "spam", "normal"]`
>    - 값이 뭐든 이 3개 중 하나면 OK
>
> 2. **길이 범위**: `assert 20 <= len(response) <= 1000`
>    - 너무 짧거나 너무 길지 않으면 OK
>
> 3. **최소 품질 기준**: `assert score >= threshold`
>    - 점수가 기준 이상이면 OK
>
> 4. **일관성**: 같은 입력 N번 실행 → 과반수 일치
>    - 3번 중 2번 이상 같은 결과면 OK
>
> 정확한 값을 모르니까, "적어도 이 범위 안에는 있어야 한다"를 검증하는 겁니다.

### 코드 설명 (셀 실행 전)

> `%%writefile tests_ai.py`로 테스트 파일을 만듭니다.
>
> ```python
> VALID_CATEGORIES = {"urgent", "spam", "normal"}
> VALID_PRIORITIES = {"high", "medium", "low"}
> ```
>
> 유효값 집합을 미리 정의합니다.
>
> ```python
> @pytest.fixture
> def ai_graph():
>     return build_ai_email_graph()
> ```
>
> `@pytest.fixture` — 테스트마다 재사용되는 객체를 만드는 함수.
> 여러 테스트 함수에서 `ai_graph`를 인자로 받으면 자동으로 주입됩니다.
>
> ```python
> def test_output_in_valid_range(ai_graph, email):
>     result = ai_graph.invoke({"email": email})
>     assert result["category"] in VALID_CATEGORIES
>     assert result["priority"] in VALID_PRIORITIES
>     assert len(result["response"]) > 10
>     assert len(result["response"]) < 2000
> ```
>
> `assert ==`가 아니라 `assert in`!
> "urgent인지 확인"이 아니라 "urgent, spam, normal 중 하나인지 확인".
>
> 일관성 테스트도 봅시다:
>
> ```python
> def test_consistency_over_runs(ai_graph):
>     email = "URGENT: Production is completely down!"
>     categories = []
>     for _ in range(3):
>         result = ai_graph.invoke({"email": email})
>         categories.append(result["category"])
>
>     from collections import Counter
>     most_common_count = Counter(categories).most_common(1)[0][1]
>     assert most_common_count >= 2
> ```
>
> 3번 실행해서 가장 많이 나온 값이 2 이상이면 통과.
> 즉 66% 일관성. "3번 중 2번은 같아야 한다."

### 셀 실행 후

> ```
> tests_ai.py::test_output_in_valid_range[URGENT: Server...]     PASSED
> tests_ai.py::test_clear_cases_match_expected[CRITICAL...]       PASSED
> tests_ai.py::test_response_length_range                          PASSED
> tests_ai.py::test_consistency_over_runs                          PASSED
> ```
>
> 전부 PASSED!
>
> 여기서 중요한 포인트:
> 규칙 기반 테스트와 AI 테스트의 차이를 정리하면:
>
> | 규칙 기반 | AI 기반 |
> |-----------|---------|
> | `assert ==` 정확 일치 | `assert in` 범위 검증 |
> | 1번 실행이면 충분 | N번 실행 → 일관성 확인 |
> | 항상 같은 결과 | 매번 다를 수 있음 |
>
> 테스트 전략이 완전히 다릅니다.

---

## 17.6 LLM-as-a-Judge — Golden Examples + Similarity Scoring (10분)

### 개념 설명

> 자, 마지막 단계입니다. 가장 강력한 테스트 방법.
>
> 17.5에서 범위 검증은 "값이 유효한가?"를 확인했습니다.
> 하지만 **"응답의 품질이 좋은가?"**는 어떻게 확인하죠?
>
> 사람이 일일이 읽고 판단? 비효율적.
> 그래서 **LLM이 판정자(Judge) 역할**을 합니다.
>
> 패턴은 3단계:
>
> 1. **Golden Examples** — 카테고리별 이상적인 응답을 미리 준비
> 2. **Judge LLM** — 생성된 응답과 golden example을 비교해서 유사도 점수 부여
> 3. **Threshold** — 점수가 70 이상이면 통과
>
> 비유하자면:
> - Golden Example = 모범 답안지
> - Judge LLM = 채점관
> - Threshold = 합격 커트라인

### 코드 설명 — Golden Examples (셀 실행 전)

> ```python
> RESPONSE_EXAMPLES = {
>     "urgent": (
>         "Thank you for alerting us. We have escalated this to our on-call team "
>         "and will provide an update within 1 hour..."
>     ),
>     "spam": (
>         "This message has been identified as unsolicited commercial email..."
>     ),
>     "normal": (
>         "Thank you for your email. We have received your message and a team "
>         "member will respond within 24 hours..."
>     ),
> }
> ```
>
> 카테고리별로 "이상적인 응답"을 미리 작성해둡니다.
> AI가 생성한 응답이 이것과 **얼마나 비슷한지**를 평가할 겁니다.

### 코드 설명 — SimilarityScoreOutput + Judge 함수

> ```python
> class SimilarityScoreOutput(BaseModel):
>     score: int = Field(gt=0, lt=100, description="Similarity score between 1 and 99")
>     reasoning: str = Field(description="Brief explanation of the score")
> ```
>
> Judge LLM이 반환하는 구조:
> - `score` — 1~99 사이의 유사도 점수
> - `reasoning` — 왜 그 점수를 줬는지 설명
>
> `gt=0, lt=100` — Pydantic의 제약 조건. 0보다 크고 100보다 작아야 합니다.
>
> ```python
> judge_llm = llm.with_structured_output(SimilarityScoreOutput)
> ```
>
> 또 `with_structured_output`! 이번에는 점수와 이유를 구조화합니다.
>
> Judge 함수:
>
> ```python
> def judge_response(generated: str, golden: str) -> SimilarityScoreOutput:
>     prompt = f"""You are an expert quality evaluator. Compare the generated response
> with the golden (ideal) response and score their similarity.
>
> Consider:
> - Tone and professionalism (30%)
> - Key information coverage (40%)
> - Appropriate length and format (30%)
>
> Golden Response:
> {golden}
>
> Generated Response:
> {generated}
>
> Score from 1 to 99..."""
>     return judge_llm.invoke(prompt)
> ```
>
> 프롬프트에 평가 기준을 명시합니다:
> - 톤과 전문성 30%
> - 핵심 정보 포함 40%
> - 적절한 길이와 형식 30%
>
> Judge LLM이 이 기준에 따라 점수를 매깁니다.

### 셀 실행 후 — Judge 테스트

> ```
> Category: urgent
> AI Response: "We have received your critical alert..."
> Similarity Score: 82
> Reasoning: "Both responses acknowledge urgency and promise timely action..."
> ```
>
> 점수 82! Threshold 70을 넘으니 PASS.

### 코드 설명 — pytest에서 Judge 사용

> `%%writefile tests_judge.py`를 봅시다.
>
> ```python
> THRESHOLD = 70
>
> def test_response_quality_above_threshold(ai_graph, email, expected_category):
>     result = ai_graph.invoke({"email": email})
>     golden = RESPONSE_EXAMPLES[expected_category]
>     score_result = judge_response(result["response"], golden)
>
>     assert score_result.score >= THRESHOLD
> ```
>
> AI가 생성한 응답을 golden example과 비교.
> Judge가 매긴 점수가 70 이상이면 통과.
>
> 두 가지 극단 케이스 테스트도 있습니다:
>
> ```python
> def test_judge_perfect_match():
>     golden = RESPONSE_EXAMPLES["urgent"]
>     score_result = judge_response(golden, golden)
>     assert score_result.score >= 90
> ```
>
> Golden example 자체를 넣으면 90점 이상이어야 합니다. (자기 자신과 비교니까)
>
> ```python
> def test_judge_poor_match():
>     poor_response = "lol ok whatever"
>     score_result = judge_response(poor_response, golden)
>     assert score_result.score < 40
> ```
>
> 엉터리 응답은 40점 미만이어야 합니다.
>
> 이렇게 Judge 자체도 검증합니다. "Judge가 제대로 판단하는가?"

### 셀 실행 후 — pytest

> ```
> tests_judge.py::test_response_quality_above_threshold[EMERGENCY...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[FREE GIFT...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[Hi, could...]   PASSED
> tests_judge.py::test_judge_perfect_match                               PASSED
> tests_judge.py::test_judge_poor_match                                  PASSED
> ```
>
> 전부 통과!

---

## 클로징 — 테스트 전략의 진화 (4분)

> 오늘 배운 내용을 정리합니다.
>
> **테스트 전략의 진화:**
>
> ```
> 17.1-17.2  규칙 기반  →  assert == (정확 일치)
> 17.4-17.5  AI 기반    →  assert in (범위 검증)
> 17.6       LLM Judge  →  assert score >= threshold (품질 검증)
> ```
>
> 결정적 시스템에서 비결정적 시스템으로 갈수록
> 테스트 방법이 **느슨해지지만 더 현실적**이 됩니다.
>
> 핵심 도구들 정리:
>
> | 도구 | 용도 |
> |------|------|
> | `%%writefile` | 노트북에서 .py 파일 생성 |
> | `@pytest.mark.parametrize` | 여러 케이스 한 번에 테스트 |
> | `graph.nodes["name"].invoke()` | 개별 노드 단위 테스트 |
> | `update_state(as_node="...")` | 상태 주입 후 부분 실행 |
> | `with_structured_output` | LLM 출력 구조화 |
> | `SimilarityScoreOutput` | Judge 점수 스키마 |
>
> 다음 챕터로 넘어가기 전에 Final Exercises를 풀어보세요.
> 특히 **과제 2 (AI 분류 정확도 벤치마크)**가 실무에서 가장 많이 쓰이는 패턴입니다.
>
> 수고하셨습니다!
