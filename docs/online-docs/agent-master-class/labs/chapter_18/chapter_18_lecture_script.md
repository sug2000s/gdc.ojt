# Chapter 18: 멀티 에이전트 아키텍처 — 강의 대본

---

## 오프닝 (2분)

> 자, 오늘은 **멀티 에이전트 아키텍처**를 배웁니다.
>
> 지금까지는 에이전트 하나가 모든 걸 처리했죠?
> 하지만 현실에서는 **여러 에이전트가 협력**해야 할 때가 많습니다.
>
> 예를 들어 고객이 한국어로 물어보면 한국어 에이전트가,
> 스페인어로 물어보면 스페인어 에이전트가 대응하는 거죠.
>
> 오늘 할 내용:
>
> ```
> 18.1 Network Architecture    (P2P — 에이전트끼리 자율 전환)
> 18.2 Supervisor Architecture  (중앙 조정자가 라우팅)
> 18.3 Supervisor as Tools      (에이전트를 도구로 캡슐화)
> ```
>
> 같은 문제를 세 가지 다른 아키텍처로 풀어봅니다.
> 각각의 장단점을 직접 체험하면서 비교합시다. 시작합니다.

---

## 18.0 Setup & Environment (3분)

### 셀 실행 전

> 먼저 환경을 확인합니다.
> `.env` 파일에서 API 키를 불러오고, `langgraph`와 `langchain` 버전을 체크합니다.
>
> 오늘 필요한 핵심 패키지:
> - `langgraph >= 1.1` — 멀티 에이전트 지원
> - `langchain >= 1.2` — LLM 연동
> - `langchain-openai` — OpenAI 모델

### 셀 실행 후

> 버전이 정상 출력되면 준비 완료입니다.
> 에러가 나면 `uv sync` 또는 `pip install` 확인해주세요.

---

## 18.1 Network Architecture — P2P 에이전트 전환 (15분)

### 개념 설명

> 첫 번째 아키텍처는 **네트워크(P2P)** 방식입니다.
>
> 중앙 조정자가 없습니다. 각 에이전트가 **자율적으로** 판단해서
> 다른 에이전트에게 대화를 넘깁니다.
>
> ```
> korean_agent ◄──► greek_agent
>       ▲               ▲
>       └──► spanish_agent ◄──┘
> ```
>
> 핵심 개념 세 가지:
>
> 1. **`handoff_tool`** — 에이전트 전환 도구. `Command(goto=..., graph=Command.PARENT)`로 부모 그래프에서 전환.
> 2. **`make_agent()` 팩토리** — 동일 구조의 에이전트를 매개변수만 바꿔서 생성.
> 3. **서브그래프** — 각 에이전트는 독립적인 ReAct 루프를 가짐.
>
> `Command.PARENT`가 핵심입니다. 서브그래프 안에서 부모 그래프의 다른 노드로 점프하는 거예요.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다. 4단계로 나뉩니다.
>
> **1단계 — 상태 정의:**
> ```python
> class AgentsState(MessagesState):
>     current_agent: str
>     transfered_by: str
> ```
> `MessagesState`를 확장해서 현재 에이전트와 전환한 에이전트를 추적합니다.
>
> **2단계 — `make_agent()` 팩토리:**
> 에이전트마다 프롬프트와 도구만 다르고 구조는 동일합니다.
> `agent → tools_condition → tools → agent` 루프.
> 이 패턴이 바로 ReAct 패턴이죠.
>
> **3단계 — `handoff_tool`:**
> ```python
> return Command(
>     update={"current_agent": transfer_to},
>     goto=transfer_to,
>     graph=Command.PARENT,  # 부모 그래프에서 전환!
> )
> ```
> `graph=Command.PARENT`가 없으면 서브그래프 안에서만 돌게 됩니다.
> 이게 있어야 부모 그래프의 다른 에이전트 노드로 점프합니다.
>
> 자기 자신에게 전환하는 무한 루프 방어도 들어있습니다.
>
> **4단계 — 최상위 그래프:**
> 각 에이전트를 노드로 등록하고 `destinations`로 가능한 전환 대상을 명시합니다.
> `START → korean_agent`로 기본 시작점을 설정합니다.
>
> 자, 실행해봅시다.

### 셀 실행 후 — 한국어 메시지

> 한국어로 "안녕하세요! 계정 문제가 있어요"를 보냈습니다.
>
> `korean_agent`가 직접 처리했죠? 전환 없이 바로 응답.
> 한국어 에이전트한테 한국어가 왔으니 당연히 자기가 처리합니다.

### 셀 실행 후 — 스페인어 메시지

> 이번엔 스페인어 "Hola! Necesito ayuda con mi cuenta."
>
> 출력을 보세요:
> ```
> [korean_agent] current_agent=spanish_agent    ← 감지 후 전환!
> [spanish_agent] current_agent=spanish_agent    ← 스페인어로 응답
> ```
>
> `korean_agent`가 스페인어를 감지하고 `handoff_tool`을 호출해서
> `spanish_agent`로 넘겼습니다. 자율적 판단입니다.
>
> 이게 네트워크 아키텍처의 특징이에요.
> 각 에이전트가 "내가 처리할 수 없으면 넘긴다"를 스스로 결정합니다.

### Exercise 18.1 (5분)

> **Exercise 1**: 그리스어 메시지를 보내서 전환 흐름을 확인해보세요.
>
> **Exercise 2**: 일본어 에이전트를 추가해보세요. `handoff_tool`의 docstring도 수정해야 합니다.
>
> **Exercise 3**: `Command.PARENT`를 제거하면 어떤 에러가 나는지 실험해보세요.

---

## 18.2 Supervisor Architecture — 중앙 조정자 (15분)

### 개념 설명

> 두 번째 아키텍처는 **슈퍼바이저** 방식입니다.
>
> 네트워크에서는 모든 에이전트가 라우팅 로직을 가졌죠?
> 슈퍼바이저 방식은 다릅니다. **하나의 중앙 조정자**가 모든 라우팅을 담당합니다.
>
> ```
>          Supervisor
>         /    |     \
>    korean  greek  spanish
>         \    |     /
>          Supervisor  ← 다시 돌아옴
> ```
>
> 에이전트는 자기 역할에만 집중합니다. 라우팅은 몰라도 됩니다.
>
> 핵심:
> - **Structured Output** — `SupervisorOutput(next_agent, reasoning)`으로 안전한 라우팅
> - **`Literal` 타입** — 가능한 값을 제한해서 잘못된 라우팅 방지
> - **순환 그래프** — agent → supervisor → agent → ... → `__end__`
> - **`reasoning` 필드** — 왜 그 에이전트를 선택했는지 추적 가능

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> **슈퍼바이저 출력 스키마:**
> ```python
> class SupervisorOutput(BaseModel):
>     next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
>     reasoning: str
> ```
> `Literal`로 가능한 값을 제한합니다. LLM이 엉뚱한 값을 반환할 수 없어요.
> `__end__`가 있어서 대화 종료도 가능합니다.
>
> **에이전트 팩토리 — `make_agent()`:**
> 18.1과 비교해보세요. `tools` 매개변수가 없습니다!
> 에이전트는 라우팅 도구가 필요 없어요. 단순히 응답만 합니다.
> 구조가 훨씬 단순해졌죠.
>
> **슈퍼바이저 노드:**
> ```python
> structured_llm = llm.with_structured_output(SupervisorOutput)
> ```
> `with_structured_output`으로 LLM이 반드시 `SupervisorOutput` 형식으로 응답하게 합니다.
> 그리고 `Command(goto=response.next_agent)`로 라우팅.
>
> **그래프 구조 — 순환:**
> ```
> START → supervisor → agent → supervisor → agent → ... → END
> ```
> 에이전트가 응답하면 다시 슈퍼바이저로 돌아옵니다.
> 슈퍼바이저가 `__end__`를 반환하면 종료.
>
> 실행해봅시다.

### 셀 실행 후 — 한국어 메시지

> 출력을 보세요:
> ```
> Supervisor → korean_agent (reason: ...)
> ```
>
> 슈퍼바이저가 한국어를 감지하고 `korean_agent`로 라우팅했습니다.
> `reasoning` 필드에 이유가 나옵니다. 디버깅에 아주 유용하죠.
>
> 에이전트가 응답 후 다시 슈퍼바이저로 돌아가고,
> 슈퍼바이저가 `__end__`를 반환하면 종료됩니다.
>
> 네트워크와 비교하면: 에이전트 코드가 훨씬 단순합니다.
> 대신 슈퍼바이저가 모든 판단을 담당하죠.

### 셀 실행 후 — 스페인어 메시지

> 이번엔 스페인어. 슈퍼바이저가 `spanish_agent`로 라우팅합니다.
>
> 에이전트는 자기가 스페인어 담당인지 아닌지 판단할 필요가 없어요.
> 슈퍼바이저가 다 해주니까. 이게 역할 분리의 장점입니다.

### Exercise 18.2 (5분)

> **Exercise 1**: `reasoning` 필드를 출력해서 슈퍼바이저의 판단 근거를 분석해보세요.
>
> **Exercise 2**: 에이전트를 추가할 때 네트워크 vs 슈퍼바이저에서 각각 어떤 코드를 수정해야 하는지 비교해보세요.
>
> **Exercise 3**: `SupervisorOutput`에서 `__end__` 옵션을 제거하면 어떻게 되는지 실험해보세요.

---

## 18.3 Supervisor as Tools — 에이전트를 도구로 캡슐화 (10분)

### 개념 설명

> 세 번째 아키텍처. **가장 깔끔한 구조**입니다.
>
> 핵심 아이디어: 에이전트를 `@tool` 함수로 만든다.
> 슈퍼바이저는 `bind_tools`로 에이전트 도구를 바인딩하고,
> LLM의 tool calling으로 자연스럽게 호출합니다.
>
> ```
> Supervisor ──tools_condition──► ToolNode
>                                ├ korean_agent
>                                ├ greek_agent
>                                └ spanish_agent
> ```
>
> 별도 라우팅 로직이 필요 없습니다. LLM이 알아서 도구를 선택하니까요.
> Structured Output도, Command도, handoff_tool도 없습니다.
>
> 에이전트 추가 = `@tool` 함수 하나 추가. 끝.

### 코드 설명 (셀 실행 전)

> 코드를 봅시다. 놀라울 정도로 단순합니다.
>
> **에이전트 = @tool 함수:**
> ```python
> @tool
> def korean_agent(message: str) -> str:
>     """Transfer to Korean customer support agent."""
>     response = llm.invoke(...)
>     return response.content
> ```
>
> docstring이 중요합니다. LLM이 이 설명을 보고 언제 이 도구를 호출할지 결정하니까요.
>
> **슈퍼바이저:**
> ```python
> llm_with_tools = llm.bind_tools(agent_tools)
> ```
> LLM에 에이전트 도구들을 바인딩. 시스템 프롬프트에서 "적절한 언어 에이전트로 라우팅하라"고 지시.
>
> **그래프 — ReAct 구조:**
> ```
> START → supervisor → tools_condition → ToolNode → supervisor → ... → END
> ```
>
> 이건 사실 Chapter 15에서 배운 ReAct 패턴과 동일합니다.
> 다만 도구가 일반 함수가 아니라 **에이전트**인 것뿐이죠.
>
> 실행해봅시다.

### 셀 실행 후 — 한국어 메시지

> 결과를 보세요:
> ```
> 고객님, 비밀번호 변경을 도와드리겠습니다...
> ```
>
> 슈퍼바이저가 `korean_agent` 도구를 호출했고,
> 한국어로 응답이 돌아왔습니다.
>
> 18.2와 비교하면 코드가 절반도 안 됩니다.
> Structured Output 스키마도 없고, Command도 없고, 순환 구조 설정도 없어요.

### 셀 실행 후 — 스페인어 메시지

> 스페인어도 마찬가지. `spanish_agent` 도구가 호출됩니다.
>
> LLM이 "이 고객은 스페인어니까 spanish_agent를 호출하자"를
> tool calling으로 자연스럽게 결정합니다.
>
> 이 아키텍처의 장점이 느껴지시죠? 가장 적은 코드, 가장 깔끔한 구조.

### Exercise 18.3 (5분)

> **Exercise 1**: 일본어 에이전트를 `@tool`로 추가해보세요. 코드 수정이 얼마나 간단한지 느껴보세요.
>
> **Exercise 2**: 세 가지 아키텍처(Network, Supervisor, Supervisor+Tools)를 비교 정리해보세요:
> - 라우팅 방식, 에이전트 복잡도, 확장성, 디버깅 용이성
>
> **Exercise 3**: 언어 라우팅 외에 실무 시나리오(부서별 라우팅, 기술 스택별 분류)를 설계해보세요.

---

## 아키텍처 비교 정리 (3분)

> 세 가지 아키텍처를 비교합니다.
>
> | | Network (18.1) | Supervisor (18.2) | Supervisor+Tools (18.3) |
> |--|---------|------------|------------------|
> | **라우팅** | 각 에이전트가 자율 | 중앙 슈퍼바이저 | LLM tool calling |
> | **에이전트 복잡도** | 높음 (handoff 로직) | 낮음 (응답만) | 최소 (@tool 함수) |
> | **확장성** | 모든 에이전트 수정 | 슈퍼바이저만 수정 | 도구 추가만 |
> | **디버깅** | 어려움 | reasoning 추적 | tool call 추적 |
> | **적합한 경우** | 에이전트 소수, 자율성 필요 | 중규모, 제어 필요 | 대규모, 깔끔한 구조 |
>
> 실무에서는 **Supervisor+Tools(18.3)**가 가장 많이 쓰입니다.
> 하지만 에이전트 간 자율적 협업이 필요하면 Network도 좋고,
> 라우팅 로직을 세밀하게 제어해야 하면 Supervisor가 적합합니다.
>
> 상황에 맞는 아키텍처를 선택하는 게 중요합니다.

---

## 종합 실습 안내 (2분)

> 노트북 마지막에 3개의 종합 과제가 있습니다.
>
> **과제 1** (★★☆): 4개 언어 Network 아키텍처 — 일본어 에이전트 추가
> **과제 2** (★★★): 부서별 Supervisor 라우팅 — billing, tech, general
> **과제 3** (★★★): Supervisor+Tools + 전문 도구 — weather, calculator, search
>
> 과제 1은 15분, 과제 2~3은 20분씩 잡으세요.

---

## 마무리 (2분)

> 오늘 배운 걸 정리합니다.
>
> | 개념 | 핵심 |
> |------|------|
> | Network (P2P) | `handoff_tool` + `Command.PARENT`로 자율 전환 |
> | Supervisor | Structured Output으로 중앙 라우팅, 순환 그래프 |
> | Supervisor+Tools | 에이전트를 `@tool`로 캡슐화, ReAct 패턴 재활용 |
> | `make_agent()` | 에이전트 팩토리 — 동일 구조, 다른 매개변수 |
> | `Command.PARENT` | 서브그래프에서 부모 그래프로 점프 |
> | `destinations` | 가능한 전환 대상을 그래프에 명시 |
>
> 멀티 에이전트는 LangGraph의 꽃입니다.
> 실제 프로덕션에서 에이전트 시스템을 만들 때 이 세 패턴 중 하나를 쓰게 됩니다.
>
> 수고하셨습니다.
