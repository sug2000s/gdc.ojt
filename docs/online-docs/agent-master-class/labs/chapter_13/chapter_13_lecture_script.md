# Chapter 13: LangGraph 기초 — 강의 대본

---

## 오프닝 (2분)

> 자, 오늘은 **LangGraph**를 처음부터 배워봅니다.
>
> 지금까지 우리가 만든 에이전트들은 코드로 직접 루프를 돌렸잖아요?
> LangGraph는 이걸 **그래프(Graph)** 구조로 설계하게 해줍니다.
>
> 노드(Node)는 "할 일", 엣지(Edge)는 "다음에 뭐 할지".
> 이걸 조합하면 복잡한 AI 워크플로우도 깔끔하게 만들 수 있습니다.
>
> 오늘 할 내용:
>
> ```
> 13.1 첫 번째 그래프       (Node, Edge, START, END)
> 13.2 상태 관리             (State 읽기/수정)
> 13.4 다중 스키마           (입력/출력/내부 분리)
> 13.5 리듀서               (상태 누적)
> 13.6 노드 캐싱            (CachePolicy)
> 13.7 조건부 엣지          (동적 분기)
> 13.8 Send API            (동적 병렬 처리)
> 13.9 Command             (노드 내부 라우팅)
> ```
>
> 많아 보이지만, 하나씩 붙여가면 자연스럽게 이해됩니다. 시작합시다.

---

## 13.0 Setup & Environment (3분)

### 셀 실행 전

> 먼저 환경부터 확인합니다.
> `.env` 파일의 API 키를 불러오고, `langgraph`와 `langchain` 버전을 체크합니다.
>
> 오늘 필요한 패키지:
> - `langgraph >= 0.6.6` — 핵심 프레임워크
> - `langchain[openai] >= 0.3.27` — OpenAI 연동
> - `grandalf` — 그래프 시각화

### 셀 실행 후

> 버전이 정상적으로 출력되면 준비 완료입니다.
> 에러가 나면 `uv sync` 또는 `pip install` 확인해주세요.

---

## 13.1 Your First Graph (10분)

### 개념 설명

> LangGraph의 핵심은 세 가지입니다:
>
> 1. **State** — 그래프 전체에서 공유하는 데이터. `TypedDict`로 정의.
> 2. **Node** — 실행할 함수. 상태를 입력으로 받습니다.
> 3. **Edge** — 노드 간의 연결. 실행 순서를 결정합니다.
>
> 그리고 두 가지 특수 노드가 있습니다:
> - `START` — 그래프 시작점
> - `END` — 그래프 종료점
>
> 이 다섯 가지만 알면 어떤 그래프든 만들 수 있습니다.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> ```python
> class State(TypedDict):
>     hello: str
> ```
>
> 상태에 `hello`라는 문자열 필드 하나.
>
> ```python
> graph_builder = StateGraph(State)
> ```
>
> `StateGraph`에 상태 스키마를 넘겨서 그래프 빌더를 만듭니다.
>
> 노드 함수 세 개는 단순히 `print`만 합니다. 아직 상태를 수정하지 않아요.
>
> 그리고 `add_node`로 노드를 등록하고, `add_edge`로 순서를 연결합니다:
>
> ```
> START -> node_one -> node_two -> node_three -> END
> ```
>
> 마지막에 `compile()` → `invoke()`로 실행합니다.
> 자, 실행해봅시다.

### 셀 실행 후

> `node_one`, `node_two`, `node_three` 순서대로 출력됐죠?
> 이게 가장 기본적인 **선형 그래프**입니다.
>
> 다음 셀에서 `draw_mermaid_png()`로 그래프를 시각화해볼 수 있습니다.
> 화살표가 순서대로 연결된 게 보이죠? 이게 우리가 만든 그래프입니다.

### Exercise 13.1 (5분)

> **Exercise 1**: `START` 엣지를 빼보세요. 어떤 에러가 나는지 확인.
>
> **Exercise 2**: 노드 순서를 바꿔보세요. `add_edge`의 순서가 실행 흐름을 결정합니다.
>
> **Exercise 3**: `add_node("node_one", node_one)` 대신 `add_node(node_one)`만 쓰면?
> 함수 이름이 자동으로 노드 이름이 되는지 확인해보세요.

---

## 13.2 Graph State (10분)

### 개념 설명

> 이제 핵심입니다. **노드가 상태를 읽고 수정하는 방법.**
>
> 규칙은 간단합니다:
> 1. 노드는 `state`를 입력으로 받는다
> 2. 딕셔너리를 반환하면 상태가 업데이트된다
> 3. **기본 동작은 덮어쓰기**
>
> 반환하지 않은 키는 이전 값이 유지됩니다.

### 코드 설명

> 상태에 `hello`(문자열)와 `a`(불리언) 두 필드가 있습니다.
>
> `node_one`은 둘 다 업데이트합니다:
> ```python
> return {"hello": "from node one.", "a": True}
> ```
>
> `node_two`는 `hello`만 업데이트합니다. `a`는 안 건드리죠.
> 그러면 `a`는? **이전 값 `True`가 그대로 유지됩니다.**
>
> 실행해봅시다.

### 셀 실행 후

> 출력을 보세요:
> ```
> node_one {'hello': 'world'}           ← 초기 입력 받음
> node_two {'hello': 'from node one.', 'a': True}  ← node_one이 업데이트한 값
> node_three {'hello': 'from node two.', 'a': True} ← a는 여전히 True
> ```
>
> `a`가 계속 `True`인 거 보이시죠? `node_two`, `node_three`가 `a`를 반환하지 않았으니까.
> **반환한 키만 덮어쓰고, 나머지는 유지.** 이게 기본 전략입니다.
>
> 최종 결과: `{'hello': 'from node three.', 'a': True}`

### Exercise 13.2 (5분)

> **Exercise 1**: 초기 입력에 `"a": False`를 넣어보세요. `node_one`에서 어떻게 보이나요?
>
> **Exercise 2**: 노드에서 `State`에 없는 키를 반환하면? 예: `return {"unknown": 123}`
>
> **Exercise 3**: `node_two`에서 `a`를 `False`로 바꿔보세요. `node_three`에서 확인.

---

## 13.4 Multiple Schemas (10분)

### 개념 설명

> 실제 앱에서는 이런 상황이 자주 생깁니다:
>
> - 사용자한테 받는 입력과 내부 처리 데이터가 다르다
> - 최종 출력은 내부 상태의 일부만 보여줘야 한다
> - 일부 노드만 접근하는 비공개 상태가 필요하다
>
> LangGraph는 **세 가지 스키마를 분리**할 수 있습니다:
>
> | 매개변수 | 역할 |
> |----------|------|
> | 첫 번째 인자 | 내부 전체 상태 (Private) |
> | `input_schema` | 외부 입력 형태 |
> | `output_schema` | 외부 출력 형태 |

### 코드 설명

> `PrivateState`는 내부용 (`a`, `b`).
> `InputState`는 외부 입력 (`hello`).
> `OutputState`는 외부 출력 (`bye`).
>
> 각 노드가 **다른 스키마를 타입 힌트**로 사용하고 있습니다.
> `node_one`은 `InputState`만 보고, `node_two`는 `PrivateState`를 봅니다.
>
> 실행해봅시다.

### 셀 실행 후

> 출력 확인:
> ```
> node_one -> {'hello': 'world'}       ← InputState만 봄
> node_two -> {}                        ← PrivateState인데 아직 a, b 없음
> node_three -> {'a': 1}                ← node_two가 a를 설정
> node_four -> {'a': 1, 'b': 1}        ← 전체 PrivateState
> {'secret': True}                      ← MegaPrivate
> ```
>
> **최종 결과: `{'bye': 'world'}`**
>
> `a`, `b`, `secret`은 출력에 안 나옵니다.
> `OutputState`에 `bye`만 정의했으니까, 그것만 반환됩니다.
>
> 이게 API 설계에서 중요합니다. 내부 처리 데이터가 밖으로 새나가지 않습니다.

### Exercise 13.4 (5분)

> **Exercise 1**: `output_schema`를 빼보세요. 반환값이 어떻게 달라지나요?
>
> **Exercise 2**: `invoke({"hello": "world", "extra": 123})` — 존재하지 않는 필드를 넣으면?
>
> **Exercise 3**: 보안 관점에서 왜 이게 중요한지 생각해보세요.

---

## 13.5 Reducer Functions (10분)

### 개념 설명

> 13.2에서 기본 전략이 "덮어쓰기"라고 했죠?
>
> 그런데 **채팅 메시지**를 생각해보세요.
> 새 메시지가 올 때마다 이전 메시지가 사라지면 안 되잖아요?
> 메시지를 **누적**해야 합니다.
>
> 이걸 해결하는 게 **Reducer 함수**입니다.
>
> ```python
> messages: Annotated[list[str], operator.add]
> ```
>
> 이 한 줄이 의미하는 건:
> "messages가 업데이트되면 덮어쓰지 말고, 기존 리스트에 **이어 붙여라**."
>
> `operator.add`가 리스트의 `+` 연산을 해줍니다.

### 셀 실행 후

> 결과: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`
>
> 초기 입력 `["Hello!"]`에 `node_one`이 반환한 `["Hello, nice to meet you!"]`이 **결합**됐습니다.
>
> **리듀서가 없었다면?** `["Hello, nice to meet you!"]`만 남고 `["Hello!"]`는 사라졌을 겁니다.
>
> 채팅 앱에서 리듀서는 필수입니다. 대화 히스토리를 쌓아야 하니까요.

### Exercise 13.5 (5분)

> **Exercise 1**: `node_two`에서도 메시지를 추가해보세요. 세 개가 쌓이나요?
>
> **Exercise 2**: `Annotated`를 빼고 그냥 `messages: list[str]`로 하면? 결과 비교.
>
> **Exercise 3**: 커스텀 리듀서를 만들어보세요. 예: 중복 제거, 최근 5개만 유지.

---

## 13.6 Node Caching (7분)

### 개념 설명

> 어떤 노드는 실행 비용이 높습니다. LLM 호출이나 외부 API 호출 같은.
> 같은 입력에 같은 결과가 나온다면? **캐시하면 됩니다.**
>
> ```python
> cache_policy=CachePolicy(ttl=20)  # 20초 동안 캐시
> ```
>
> `ttl`은 Time-To-Live. 이 시간이 지나면 캐시가 만료되고 다시 실행합니다.

### 셀 실행

> `node_two`는 현재 시간을 반환합니다.
> 5초 간격으로 6번 실행하면 총 30초.
> `ttl=20`이니까:
>
> - **Run 1~4** (0~20초): 같은 시간이 출력됨 — 캐시 히트!
> - **Run 5~6** (20초 이후): 새로운 시간 — 캐시 만료, 재실행
>
> 실제로 확인해봅시다. (약 30초 소요)

### 셀 실행 후

> 보이시죠? 처음 몇 번은 같은 시간, 그 다음에 바뀝니다.
> API 호출 비용 절감에 아주 유용합니다.

---

## 13.7 Conditional Edges (15분)

### 개념 설명

> 지금까지는 **선형 그래프**만 만들었습니다. A → B → C.
>
> 하지만 실제로는 "조건에 따라 다른 경로"가 필요하죠.
>
> `add_conditional_edges`는 **분기 함수**의 반환값에 따라 다음 노드를 결정합니다.
>
> ```python
> add_conditional_edges(
>     출발_노드,
>     분기_함수,        # 상태를 보고 값을 반환
>     {값: 목적지_노드}  # 반환값 → 노드 매핑
> )
> ```

### 코드 설명

> `decide_path` 함수를 봅시다:
>
> ```python
> def decide_path(state: State):
>     return state["seed"] % 2 == 0  # True 또는 False
> ```
>
> `seed`가 짝수면 `True`, 홀수면 `False`.
>
> 매핑:
> ```python
> {True: "node_one", False: "node_two"}
> ```
>
> 그러니까:
> - seed=42 (짝수) → `True` → `node_one` → `node_two` → ...
> - seed=7 (홀수) → `False` → 바로 `node_two` → ...
>
> `node_two` 이후에도 또 조건부 분기가 있습니다. 동일한 `decide_path`를 재사용하고 있어요.

### 셀 실행 — seed=42

> 짝수니까:
> ```
> node_one -> {'seed': 42}
> node_two -> {'seed': 42}
> node_three -> {'seed': 42}
> ```
> `START → node_one → node_two → node_three → END` 경로.

### 셀 실행 — seed=7

> 홀수니까:
> ```
> node_two -> {'seed': 7}
> node_four -> {'seed': 7}
> ```
> `START → node_two → node_four → END` 경로.
>
> 같은 그래프인데 입력에 따라 완전히 다른 경로를 탑니다!
> 그래프 시각화를 보면 분기가 명확히 보입니다.

### Exercise 13.7 (5분)

> **Exercise 1**: `seed` 값을 여러 개 넣어보면서 경로 변화를 관찰.
>
> **Exercise 2**: 노드 이름을 직접 반환하는 방식으로 바꿔보세요:
> ```python
> def decide_path(state) -> Literal["node_three", "node_four"]:
>     if state["seed"] % 2 == 0:
>         return "node_three"
>     else:
>         return "node_four"
> ```
>
> **Exercise 3**: 3개 이상 분기하는 조건부 엣지를 만들어보세요.

---

## 13.8 Send API (15분)

### 개념 설명

> 조건부 엣지는 "어디로 갈 것인가"를 결정했죠.
> **Send API**는 한 단계 더 나아갑니다:
>
> **같은 노드를 다른 입력으로 여러 번 동시에 실행.**
>
> ```python
> Send("node_two", word)  # node_two를 word 입력으로 실행
> ```
>
> `dispatcher` 함수가 리스트로 여러 `Send`를 반환하면,
> 그만큼 `node_two`가 **병렬 실행**됩니다.
>
> 이건 **Map-Reduce** 패턴입니다:
> - Map: 데이터를 쪼개서 각각 처리
> - Reduce: 결과를 모아서 합침 (Reducer 함수!)

### 코드 설명

> 중요한 포인트:
>
> ```python
> def node_two(word: str):  # State가 아니라 str을 받는다!
> ```
>
> Send API로 전달되는 값은 **State가 아닌 커스텀 입력**입니다.
> 각 단어가 개별적으로 `node_two`에 전달됩니다.
>
> `output` 필드에는 `Annotated[..., operator.add]` 리듀서가 있어서
> 병렬 실행된 모든 결과가 자동으로 합쳐집니다.
>
> `dispatcher`:
> ```python
> def dispatcher(state):
>     return [Send("node_two", word) for word in state["words"]]
> ```
> 6개 단어면 6개의 `Send` → 6개의 `node_two` 병렬 실행.

### 셀 실행 후

> 결과:
> ```
> hello -> 5 letters
> world -> 5 letters
> how   -> 3 letters
> are   -> 3 letters
> you   -> 3 letters
> doing -> 5 letters
> ```
>
> 6개의 단어가 각각 처리되고, 결과가 `output` 리스트로 합쳐졌습니다.
>
> 실제로 이걸 어디 쓰냐?
> - 여러 문서를 동시에 요약
> - 여러 API에서 동시에 데이터 수집
> - 여러 사용자 요청을 병렬 처리

### Exercise 13.8 (5분)

> **Exercise 1**: 단어를 20개로 늘려보세요.
>
> **Exercise 2**: `node_two`에 `time.sleep(1)` 추가. 전체 실행 시간이 1초인지 6초인지 확인.
>
> **Exercise 3**: 실제 활용 사례를 하나 직접 구현해보세요.

---

## 13.9 Command (10분)

### 개념 설명

> 지금까지:
> - 상태 업데이트 = 노드가 딕셔너리 반환
> - 라우팅 = `add_conditional_edges` + 분기 함수
>
> 이 둘이 **분리**되어 있었습니다.
>
> **Command**는 이걸 **하나로 합칩니다**:
>
> ```python
> Command(
>     goto="account_support",       # 어디로 갈지
>     update={"reason": "..."},     # 상태 업데이트
> )
> ```
>
> 노드 안에서 "상태 바꾸고 + 다음 노드 결정"을 한 번에.
> `add_conditional_edges`도 분기 함수도 필요 없습니다.

### 코드 설명

> `triage_node`를 봅시다:
>
> ```python
> def triage_node(state) -> Command[Literal["account_support", "tech_support"]]:
>     return Command(
>         goto="account_support",
>         update={"transfer_reason": "The user wants to change password."},
>     )
> ```
>
> **반환 타입의 `Command[Literal[...]]`가 핵심입니다.**
> 이 타입 힌트 덕분에 LangGraph가 가능한 경로를 알 수 있고,
> `add_edge`를 안 써도 그래프가 구성됩니다.
>
> 그래프 구성에서 `triage_node` 이후 엣지가 **없는** 거 보이시죠?
> Command가 런타임에 다음 노드를 결정하니까요.

### 셀 실행 후

> ```
> account_support running
> Result: {'transfer_reason': 'The user wants to change password.'}
> ```
>
> `triage_node`가 Command로:
> 1. `transfer_reason`을 업데이트하고
> 2. `account_support`로 라우팅했습니다.
>
> 그래프 시각화를 보면 `triage_node`에서 두 노드로 분기하는 게 보입니다.
> 타입 힌트만으로 이게 가능한 겁니다.

### Exercise 13.9 (5분)

> **Exercise 1**: 조건을 넣어서 `tech_support`로도 라우팅되게 만들어보세요.
>
> **Exercise 2**: `Command` vs `add_conditional_edges` — 각각의 장단점은?
>
> **Exercise 3**: 실제 고객 상담 시스템처럼 여러 단계 라우팅을 구현해보세요.

---

## 종합 실습 안내 (3분)

> 노트북 마지막에 5개의 종합 과제가 있습니다.
>
> **과제 1** (★☆☆): 카운터 그래프 — 각 노드가 counter를 +1
> **과제 2** (★★☆): 채팅 시뮬레이터 — 리듀서로 메시지 누적
> **과제 3** (★★☆): 나이별 라우팅 — 조건부 엣지
> **과제 4** (★★★): 대문자 변환 — Send API
> **과제 5** (★★★): 상담 라우터 — Command
>
> 과제 1~2는 기본, 3~5는 도전 과제입니다.
> 시간 배분: 쉬운 것 10분, 어려운 것 15분씩.

---

## 마무리 (3분)

> 오늘 배운 걸 정리합니다.
>
> | 개념 | 핵심 |
> |------|------|
> | StateGraph | 상태 기반 그래프의 뼈대 |
> | Node / Edge | 할 일 / 순서 |
> | State | 기본은 덮어쓰기, Reducer로 누적 가능 |
> | Multiple Schemas | 입력/출력/내부 분리 |
> | CachePolicy | 노드별 캐싱으로 성능 최적화 |
> | Conditional Edges | 상태에 따른 동적 분기 |
> | Send API | 동적 병렬 실행 (Map-Reduce) |
> | Command | 라우팅 + 상태 업데이트를 한 번에 |
>
> 이것들이 LangGraph의 기초 빌딩 블록입니다.
> 다음 챕터에서는 이걸 활용해서 실제 **챗봇**을 만들어봅니다.
>
> 수고하셨습니다.
