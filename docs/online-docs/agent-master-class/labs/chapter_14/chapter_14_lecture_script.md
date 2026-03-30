# Chapter 14: LangGraph 챗봇 구축 — 강의 대본

---

## 오프닝 (2분)

> 자, 지난 챕터에서 LangGraph의 기초 빌딩 블록을 배웠습니다.
>
> 오늘은 그걸 활용해서 **실용적인 챗봇**을 단계적으로 만들어봅니다.
>
> 단순한 챗봇에서 시작해서, 하나씩 기능을 붙여갈 겁니다:
>
> ```
> 14.0 기본 챗봇        (MessagesState + init_chat_model)
> 14.1 도구 노드         (Tool Calling + ReAct 패턴)
> 14.2 메모리            (SQLite 체크포인터)
> 14.3 Human-in-the-loop (interrupt + Command resume)
> 14.4 타임 트래블       (State Fork)
> ```
>
> 각 단계가 이전 단계 위에 쌓이는 구조입니다.
> 마지막에는 메모리도 있고, 사람 개입도 되고, 과거로 돌아갈 수도 있는 챗봇이 완성됩니다.
>
> 시작합시다.

---

## 14.0 Setup & Environment (2분)

### 셀 실행 전

> 먼저 환경을 확인합니다.
> `.env` 파일에서 API 키를 불러오고, `langgraph`와 `langchain` 버전을 체크합니다.
>
> Chapter 13과 동일한 환경입니다. 추가 패키지는 없습니다.

### 셀 실행 후

> API 키와 버전이 정상적으로 출력되면 준비 완료입니다.
> 에러가 나면 `uv sync` 또는 `pip install` 확인해주세요.

---

## 14.0 기본 챗봇 — MessagesState (8분)

### 개념 설명

> Chapter 13에서는 `TypedDict`로 직접 상태를 정의했습니다.
> 챗봇에서는 이것보다 편한 게 있습니다: **`MessagesState`**.
>
> `MessagesState`가 뭐냐?
> - `messages: list` 필드가 기본 제공됩니다
> - **리듀서가 내장**되어 있어서 메시지가 자동으로 누적됩니다
> - Chapter 13에서 `Annotated[list, operator.add]` 했던 거, 이미 포함입니다
>
> LLM 초기화도 간단합니다:
> ```python
> llm = init_chat_model("openai:gpt-4o-mini")
> ```
> `init_chat_model`은 프로바이더:모델명 형식으로 어떤 LLM이든 한 줄로 초기화합니다.
>
> 그래프 구조는 가장 단순한 형태:
> ```
> START → chatbot → END
> ```

### 코드 설명 (셀 실행 전)

> 코드를 봅시다.
>
> ```python
> class State(MessagesState):
>     pass
> ```
>
> `MessagesState`를 상속만 하면 됩니다. `messages` 필드가 자동으로 들어옵니다.
>
> ```python
> def chatbot(state: State):
>     response = llm.invoke(state["messages"])
>     return {"messages": [response]}
> ```
>
> `chatbot` 노드는:
> 1. 상태에서 `messages`를 꺼내서 LLM에 전달
> 2. LLM 응답을 `messages`에 추가
>
> 리듀서 덕분에 기존 메시지가 사라지지 않고 응답이 뒤에 붙습니다.
>
> 엣지는 `START → chatbot → END`. 가장 단순한 선형 그래프.
> 실행해봅시다.

### 셀 실행 후

> 출력을 보세요:
> ```
> human: how are you?
> ai: I'm just a computer program, so I don't have feelings...
> ```
>
> `human` 메시지와 `ai` 응답이 `messages` 리스트에 쌓여있습니다.
>
> 이게 가장 기본적인 챗봇입니다.
> 하지만 문제가 있죠? **대화 기록이 유지되지 않습니다.**
> `invoke()`가 끝나면 상태가 사라집니다. 이건 나중에 14.2에서 해결합니다.

### Exercise 14.0 (3분)

> **Exercise 1**: `MessagesState` 대신 `TypedDict`로 직접 `messages: list`를 정의하면 어떤 차이가 생기나요? 리듀서가 없으면 메시지가 어떻게 되는지 확인해보세요.
>
> **Exercise 2**: `init_chat_model`의 프로바이더를 바꿔보세요. `"anthropic:claude-sonnet-4-20250514"` 같은 식으로.
>
> **Exercise 3**: `State`에 `system_prompt: str` 필드를 추가해서 동적 시스템 프롬프트를 구현해보세요.

---

## 14.1 도구 노드 — Tool Calling + ReAct 패턴 (10분)

### 개념 설명

> 기본 챗봇은 LLM 지식으로만 대답합니다.
> 실시간 날씨나 DB 조회처럼 **외부 도구**가 필요할 때는?
>
> **ReAct 패턴**입니다:
> 1. LLM이 "이 도구를 호출해야겠다"고 판단 → tool_calls 생성
> 2. 도구 실행 → 결과 반환
> 3. LLM이 결과를 보고 최종 응답 생성
>
> 그래프 구조:
> ```
> START → chatbot → [tool_calls 있으면?] → tools → chatbot → ... → END
> ```
>
> 핵심 컴포넌트 세 가지:
> - `@tool` — Python 함수를 LLM 도구로 변환
> - `ToolNode` — 도구 실행을 담당하는 노드
> - `tools_condition` — tool_calls가 있으면 `tools`로, 없으면 `END`로

### 코드 설명 (셀 실행 전)

> 코드를 단계별로 봅시다.
>
> **1단계: 도구 정의**
> ```python
> @tool
> def get_weather(city: str):
>     """Gets weather in city"""
>     return f"The weather in {city} is sunny."
> ```
> `@tool` 데코레이터가 이 함수를 LLM이 호출 가능한 도구로 만듭니다.
> **docstring이 중요합니다** — LLM이 이걸 보고 언제 이 도구를 쓸지 결정합니다.
>
> **2단계: LLM에 도구 바인딩**
> ```python
> llm_with_tools = llm.bind_tools(tools=[get_weather])
> ```
> LLM에게 "이 도구를 쓸 수 있어"라고 알려주는 겁니다.
>
> **3단계: 그래프 구성**
> ```python
> graph_builder.add_conditional_edges("chatbot", tools_condition)
> graph_builder.add_edge("tools", "chatbot")
> ```
>
> `tools_condition`은 LangGraph가 제공하는 내장 함수입니다.
> LLM 응답에 `tool_calls`가 있으면 `"tools"` 반환, 없으면 `"__end__"` 반환.
>
> 도구 결과는 다시 `chatbot`으로 돌아갑니다. LLM이 결과를 보고 최종 답변을 만드니까요.
>
> 자, 실행해봅시다.

### 셀 실행 후 — 도구가 필요한 질문

> ```
> human: what is the weather in machupichu
> ai:                          ← tool_calls만 있고 content는 비어있음
> tool: The weather in Machupichu is sunny.
> ai: The weather in Machupicchu is sunny.
> ```
>
> 흐름이 보이시나요?
> 1. 사용자가 날씨를 물어봄
> 2. LLM이 `get_weather` 호출이 필요하다고 판단 → tool_calls 생성
> 3. `tools_condition`이 tool_calls 감지 → `tools` 노드로 라우팅
> 4. `ToolNode`가 `get_weather("Machupichu")` 실행
> 5. 결과가 다시 `chatbot`으로 → LLM이 최종 응답 생성
>
> 이게 **ReAct 루프**입니다. Reasoning(추론) + Acting(행동).

### 셀 실행 후 — 도구가 필요 없는 질문

> ```
> human: hello, how are you?
> ai: Hello! I'm just a computer program...
> ```
>
> 이번에는 `tool` 메시지가 없습니다.
> LLM이 도구 없이 답할 수 있다고 판단했으니까요.
> `tools_condition`이 `END`로 보낸 겁니다.
>
> **같은 그래프인데 입력에 따라 경로가 달라집니다.** Chapter 13의 조건부 엣지와 같은 원리.

### Exercise 14.1 (3분)

> **Exercise 1**: `get_time` 같은 도구를 하나 더 추가해보세요. LLM이 상황에 맞게 적절한 도구를 선택하나요?
>
> **Exercise 2**: 도구의 docstring을 변경하면 LLM의 선택이 어떻게 달라지나요?
>
> **Exercise 3**: `tools_condition`을 커스텀 조건 함수로 교체해보세요.

---

## 14.2 메모리 — SQLite 체크포인터 (10분)

### 개념 설명

> 14.0에서 말한 문제, 기억하시죠? `invoke()` 끝나면 상태가 사라진다.
>
> 챗봇이 이전 대화를 기억하려면 **체크포인터**가 필요합니다.
>
> 체크포인터가 하는 일:
> - 매 노드 실행 후 상태를 DB에 저장
> - 같은 세션(thread_id)이면 이전 상태를 불러와서 이어감
>
> ```python
> conn = sqlite3.connect("memory.db")
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> **`thread_id`**가 핵심입니다:
> - 같은 thread_id → 대화 이어감 (메모리 유지)
> - 다른 thread_id → 완전히 새로운 대화
>
> 실제 채팅 앱에서 "대화방"이 thread_id라고 생각하면 됩니다.

### 코드 설명 (셀 실행 전)

> 코드는 14.1의 도구 챗봇과 거의 동일합니다.
> 달라진 건 딱 두 줄:
>
> ```python
> conn = sqlite3.connect("memory.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> `compile()`에 `checkpointer`를 넘기는 것만으로 메모리가 활성화됩니다.
>
> 그리고 `invoke()` 호출 시:
> ```python
> config = {"configurable": {"thread_id": "1"}}
> result = graph.invoke({"messages": [...]}, config=config)
> ```
>
> `config`에 `thread_id`를 넣어서 세션을 구분합니다.
> 실행해봅시다.

### 셀 실행 후 — thread_id="1", 첫 대화

> ```
> Hello Alice! How can I assist you today?
> ```
>
> 이름을 알려줬습니다. 다음 셀에서 같은 thread_id로 물어봅시다.

### 셀 실행 후 — thread_id="1", 이어서

> ```
> Your name is Alice.
> ```
>
> **기억하고 있습니다!** 같은 thread_id니까 이전 대화가 메모리에 남아있는 겁니다.

### 셀 실행 후 — thread_id="2", 새 대화

> ```
> I'm sorry, but I don't have access to personal information about you...
> ```
>
> thread_id가 다르니까 **완전히 새로운 대화**입니다. 이름을 모릅니다.
>
> 이게 체크포인터의 핵심입니다:
> - 같은 thread → 기억 유지
> - 다른 thread → 완전 분리

### 셀 실행 후 — get_state_history

> ```
> next: (), messages: 4
> next: ('chatbot',), messages: 3
> next: ('__start__',), messages: 2
> ...
> ```
>
> `get_state_history()`는 해당 thread의 모든 스냅샷을 보여줍니다.
> 각 노드 실행 전후의 상태가 기록되어 있습니다.
>
> `next`가 비어있으면 그래프가 완료된 시점이고,
> `('chatbot',)`이면 chatbot 노드 실행 직전 시점입니다.
>
> 이 히스토리가 나중에 14.4 타임 트래블에서 중요해집니다.

### Exercise 14.2 (3분)

> **Exercise 1**: 같은 thread_id로 여러 번 대화해보세요. 메모리가 계속 유지되나요?
>
> **Exercise 2**: `stream_mode="updates"`로 스트리밍 실행을 해보세요.
>
> **Exercise 3**: `get_state_history()`의 각 스냅샷에서 상태 변화를 추적해보세요.

---

## 14.3 Human-in-the-loop (10분)

### 개념 설명

> 지금까지 만든 챗봇은 **완전 자동**입니다.
> 사용자가 요청하면 LLM이 알아서 처리하고 끝.
>
> 하지만 실무에서는 **사람이 중간에 개입**해야 할 때가 많습니다:
> - AI가 생성한 코드를 사람이 리뷰
> - 중요한 결정 전에 사람의 승인
> - AI 결과물에 대한 피드백
>
> LangGraph의 해결책: **`interrupt()`와 `Command(resume=...)`**
>
> 흐름:
> 1. 그래프 실행 중 `interrupt()` 호출 → 실행 일시 중단
> 2. 사용자가 피드백 제공
> 3. `Command(resume=피드백)` → 피드백을 전달하며 실행 재개
>
> 체크포인터가 있어서 중단된 상태가 저장되고, 이어서 실행 가능합니다.

### 코드 설명 (셀 실행 전)

> 이 예제는 **시 작성 챗봇**입니다. LLM이 시를 쓰고, 사람의 피드백을 받습니다.
>
> 핵심은 `get_human_feedback` 도구:
> ```python
> @tool
> def get_human_feedback(poem: str):
>     """Asks the user for feedback on the poem."""
>     feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
>     return feedback
> ```
>
> `interrupt()`가 호출되면:
> 1. 그래프 실행이 **멈춥니다**
> 2. 인자로 전달한 값이 사용자에게 표시됩니다
> 3. `Command(resume=...)`로 재개하면 그 값이 `interrupt()`의 반환값이 됩니다
>
> LLM의 시스템 프롬프트를 보세요:
> ```
> ALWAYS ASK FOR FEEDBACK FIRST.
> Only after you receive positive feedback you can return the final poem.
> ```
>
> LLM이 시를 쓰면 반드시 피드백을 요청하고, 긍정적 피드백을 받아야 최종 결과를 반환합니다.
> 실행해봅시다.

### 셀 실행 후 — 1단계: 시 작성 요청

> ```
> Next: ('tools',)
> ```
>
> `next`가 `('tools',)`입니다. 즉, tools 노드에서 **멈춰있는** 상태.
> LLM이 `get_human_feedback`을 호출했고, `interrupt()`에서 중단된 겁니다.
>
> 이제 사용자가 피드백을 제공할 차례입니다.

### 셀 실행 후 — 2단계: 부정적 피드백

> ```python
> Command(resume="It is too long! Make it shorter, 4 lines max.")
> ```
>
> 부정적 피드백을 줬더니 LLM이 다시 시를 수정하고 또 피드백을 요청합니다.
> `next`가 여전히 `('tools',)` — 다시 인터럽트 대기 중.

### 셀 실행 후 — 3단계: 긍정적 피드백

> ```python
> Command(resume="It looks great!")
> ```
>
> 이번에는 긍정적 피드백을 줬으니까, LLM이 최종 시를 반환합니다.
> `next`가 `()` — 그래프 완료.
>
> 전체 대화 흐름을 보세요:
> ```
> human: Please make a poem about Python code.
> ai:                          ← 시를 쓰고 피드백 도구 호출
> tool: It is too long!...     ← 사람 피드백 (부정적)
> ai:                          ← 수정하고 다시 피드백 요청
> tool: It looks great!        ← 사람 피드백 (긍정적)
> ai: Here's the final poem... ← 최종 결과
> ```
>
> **사람이 루프 안에 들어가 있는** 겁니다. 그래서 Human-in-the-loop.

### Exercise 14.3 (3분)

> **Exercise 1**: 부정적 피드백을 여러 번 연속으로 주면 LLM이 어떻게 반응하나요?
>
> **Exercise 2**: `Command(resume=...)`에 딕셔너리(구조화된 데이터)를 전달해보세요.
>
> **Exercise 3**: 검토 → 승인 → 배포 파이프라인처럼 여러 인터럽트 포인트를 설계해보세요.

---

## 14.4 타임 트래블 — State Fork (10분)

### 개념 설명

> 체크포인터가 모든 상태를 저장한다고 했죠?
> 그럼 **과거 시점으로 돌아갈 수 있지 않을까?**
>
> 맞습니다. 이게 **타임 트래블**입니다.
>
> 핵심 API:
> - `get_state_history()` — 모든 스냅샷을 시간순으로 조회
> - `update_state()` — 과거 체크포인트를 수정해서 새 분기 생성
>
> 이게 왜 유용하냐?
> - **디버깅**: 에러가 난 시점으로 돌아가서 원인 분석
> - **A/B 테스팅**: 같은 시점에서 다른 입력으로 분기
> - **롤백**: 잘못된 실행을 취소하고 다시 시작

### 코드 설명 (셀 실행 전)

> 간단한 챗봇을 만들고, 대화를 두 번 합니다.
>
> 1. "I live in Europe. My city is Valencia." → Valencia 관련 응답
> 2. "What are some good restaurants near me?" → Valencia 기준 레스토랑 추천
>
> 그 다음에 `get_state_history()`로 모든 스냅샷을 확인하고,
> 과거 시점에서 "Valencia"를 "Zagreb"로 바꿔서 **포크(fork)**합니다.
>
> 실행해봅시다.

### 셀 실행 후 — 대화 시작

> ```
> Valencia is a beautiful city located on the eastern coast of Spain...
> ```
>
> 정상적인 대화입니다.

### 셀 실행 후 — 이어서 질문

> ```
> Valencia 기준 응답:
> La Pepica — Famous for its paella...
> ```
>
> Valencia 기준으로 레스토랑을 추천합니다.

### 셀 실행 후 — 상태 히스토리

> ```
> Snapshot 0: next=(), messages=4
> Snapshot 1: next=('chatbot',), messages=3
> ...
> Snapshot 5: next=('__start__',), messages=0
> ```
>
> 총 6개 스냅샷. 각각이 그래프 실행의 특정 시점입니다.
> 우리는 이 중 **사용자가 도시를 말한 직후 시점**을 찾을 겁니다.

### 셀 실행 후 — 포크 (Valencia → Zagreb)

> ```python
> graph.update_state(
>     fork_config,
>     {"messages": [HumanMessage(content="I live in Europe. My city is Zagreb.")]},
> )
> result_fork = graph.invoke(None, config=fork_config)
> ```
>
> 과거 스냅샷의 설정(`fork_config`)으로 상태를 업데이트하고,
> `invoke(None)`으로 그 시점부터 다시 실행합니다.
>
> 결과를 보면 Zagreb 기준 응답이 나옵니다.
> **같은 대화의 과거 시점에서 분기한** 겁니다.
>
> 원래 대화는 그대로 유지됩니다. 포크는 새로운 분기를 만드는 것이지, 기존 대화를 덮어쓰는 게 아닙니다.

### Exercise 14.4 (3분)

> **Exercise 1**: 더 긴 대화를 나눈 후 중간 시점으로 포크해보세요.
>
> **Exercise 2**: 포크한 분기에서 원래 분기와 다른 질문을 하여 결과를 비교해보세요.
>
> **Exercise 3**: 타임 트래블이 유용한 실무 시나리오를 생각해보세요 (A/B 테스팅, 디버깅, 롤백).

---

## 종합 실습 안내 (3분)

> 노트북 마지막에 4개의 종합 과제가 있습니다.
>
> **과제 1** (★★☆): 다중 도구 챗봇 — `get_weather`, `get_time`, `get_news` 3개 도구
> **과제 2** (★★☆): 대화 이력 유지 — thread_id로 세션 관리
> **과제 3** (★★★): 코드 리뷰 HITL — interrupt()로 사람 리뷰
> **과제 4** (★★★): 타임 트래블 A/B 테스트 — 같은 시점에서 분기 비교
>
> 과제 1~2는 기본, 3~4는 도전 과제입니다.
> 시간 배분: 쉬운 것 10분, 어려운 것 15분씩.

---

## 마무리 (2분)

> 오늘 배운 걸 정리합니다.
>
> | 개념 | 핵심 |
> |------|------|
> | MessagesState | 메시지 리듀서 내장, 챗봇 상태의 표준 |
> | init_chat_model | 프로바이더:모델명으로 한 줄 초기화 |
> | @tool + ToolNode | 외부 도구를 그래프에 통합 |
> | tools_condition | tool_calls 유무로 자동 분기 |
> | SqliteSaver | DB 기반 상태 저장, thread_id로 세션 분리 |
> | interrupt() | 그래프 실행 일시 중단, 사람 개입 |
> | Command(resume) | 피드백을 전달하며 실행 재개 |
> | get_state_history | 전체 스냅샷 히스토리 조회 |
> | update_state | 과거 시점에서 새 분기(fork) 생성 |
>
> 기본 챗봇에서 시작해서 도구, 메모리, 사람 개입, 타임 트래블까지.
> 이 패턴들이 실제 프로덕션 AI 에이전트의 핵심 빌딩 블록입니다.
>
> 수고하셨습니다.
