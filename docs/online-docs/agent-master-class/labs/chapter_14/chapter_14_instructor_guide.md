# Chapter 14: LangGraph 챗봇 — 강사용 해설서

> 이 문서는 강사가 Chapter 14를 **미리 이해**하기 위한 해설서입니다.
> 강의 대본(lecture_script)과 별개로, 코드를 한 줄 한 줄 쉽게 풀어서 설명합니다.

---

## 전체 진화 흐름 (먼저 큰 그림)

```
14.0  START → chatbot → END                          "대화 1회, 기억 없음"
       +
14.1  START → chatbot ↔ tools → END                  "도구 사용 가능"
       +
14.2  + checkpointer (SqliteSaver)                    "대화 기억함"
       +
14.3  + interrupt() / Command(resume=)                "사람이 중간에 개입"
       +
14.4  + get_state_history / update_state              "과거로 포크"
```

각 섹션이 **이전 섹션에 한 가지만 추가**하는 구조입니다.
그래서 순서대로 가르치면 자연스럽게 이해됩니다.

---

## 14.0 기본 챗봇 — "가장 단순한 챗봇"

### import 정리

```python
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph.message import MessagesState
```

| import | 뭐하는 놈 |
|--------|----------|
| `StateGraph` | 그래프(워크플로우)를 만드는 설계도 |
| `START, END` | 시작점, 끝점 (특수 노드) |
| `init_chat_model` | LLM을 초기화하는 함수. `"openai:모델명"` 형식 |
| `MessagesState` | **채팅 전용 상태**. `messages` 리스트가 내장되어 있음 |

### LLM 초기화

```python
llm = init_chat_model(f"openai:{os.getenv('OPENAI_MODEL_NAME')}")
```

`.env`에서 모델명(gpt-5.1) 읽어서 LLM 객체 생성. 이게 우리의 AI 두뇌.

### 상태 정의 — MessagesState

```python
class State(MessagesState):
    pass
```

Chapter 13에서는 `TypedDict`로 직접 상태를 만들었다.
챗봇에서는 `MessagesState`를 상속하면 끝.

**이것만으로 `messages: list` 필드가 자동 생성**되고,
메시지가 **덮어쓰기가 아니라 누적(append)**된다.

> Chapter 13의 `Annotated[list, operator.add]` 리듀서가 내장된 것이라고 이해하면 됨.

### 그래프 빌더 생성

```python
graph_builder = StateGraph(State)
```

설계도(빌더) 생성. "이 그래프는 State를 공유합니다"라는 선언.

### 챗봇 노드 함수 — 이게 핵심!

```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

**이게 챗봇의 핵심이자 전부.**

1. `state["messages"]` — 지금까지의 대화 기록 전체를 꺼냄
2. `llm.invoke(...)` — AI에게 대화 기록을 통째로 넘겨서 응답 생성
3. `return {"messages": [response]}` — 응답을 messages에 추가

`MessagesState` 덕분에 return하면 **기존 메시지에 append** 된다. 덮어쓰기 아님!

### 그래프 구성 & 실행

```python
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

그래프 구조: `START → chatbot → END`. 이게 가장 단순한 챗봇.

```python
result = graph.invoke(
    {"messages": [{"role": "user", "content": "how are you?"}]}
)
```

- `invoke()`에 초기 메시지를 딕셔너리로 넘김
- `{"role": "user", "content": "..."}` → LangChain이 자동으로 `HumanMessage` 객체로 변환
- **결과에는 [HumanMessage, AIMessage] 두 개가 들어있음** — 원래 질문 + AI 응답

---

## 14.1 도구 노드 — "AI가 외부 기능을 호출"

14.0에서 추가되는 것: **AI가 직접 답 못하는 질문에 도구를 사용**

### 새로운 import

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
```

| import | 역할 |
|--------|------|
| `@tool` | 일반 함수를 "AI가 호출할 수 있는 도구"로 변환 |
| `ToolNode` | 도구 실행을 담당하는 미리 만들어진 노드 |
| `tools_condition` | "AI가 도구를 부르고 싶어해?" 판단하는 라우팅 함수 |

### 도구 정의

```python
@tool
def get_weather(city: str):
    """Gets weather in city"""
    return f"The weather in {city} is sunny."
```

`@tool` 데코레이터가 하는 일:
- 함수의 **docstring** → 도구 설명 (AI가 이걸 읽고 "이 도구 쓸까?" 판단)
- 함수의 **파라미터 타입** → 입력 스키마 (AI가 인자를 자동 구성)
- AI가 직접 실행하는 게 아님! "이 도구를 이렇게 호출해줘"라고 **요청**하는 것

### LLM에 도구 바인딩

```python
llm_with_tools = llm.bind_tools(tools=[get_weather])
```

LLM에게 "너 이 도구 쓸 수 있어"라고 알려주는 것.

### 그래프 구조 변경 — 루프!

```python
tool_node = ToolNode(tools=[get_weather])

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # 도구 결과 → 다시 챗봇으로
```

이전과 비교:

```
이전: START → chatbot → END

지금: START → chatbot → [도구 필요?] → Yes → tools → chatbot → ...
                                     → No  → END
```

**`tools_condition`이 핵심!**
- AI 응답에 `tool_calls`가 있으면 → `"tools"` 노드로
- 없으면 → `END`로

`tools → chatbot` 엣지가 있으니 **루프**. AI가 도구 결과를 받고 다시 판단.

### 실행 흐름 예시 (날씨 질문)

```
1. User: "마추픽추 날씨 어때?"
2. AI: "get_weather(city='machupichu') 호출해줘" (tool_calls 발생)
3. tools_condition: tool_calls 있네 → tools 노드로
4. ToolNode: get_weather("machupichu") 실행 → "sunny" 반환
5. 다시 chatbot으로: AI가 도구 결과를 받아서 자연어로 정리
6. AI: "마추픽추는 맑은 날씨입니다" (tool_calls 없음 → END)
```

### 도구 필요 없는 질문은?

```
1. User: "hello, how are you?"
2. AI: "I'm fine!" (tool_calls 없음)
3. tools_condition: tool_calls 없네 → END
```

도구 노드를 거치지 않고 바로 끝남.

---

## 14.2 메모리 — "대화를 기억하는 챗봇"

14.1에 추가되는 것: `invoke()` 끝나도 **대화가 DB에 저장**됨

### 체크포인터 설정 — 딱 2줄!

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("memory.db", check_same_thread=False)
graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
```

1. SQLite DB 연결
2. `compile()` 할 때 `checkpointer=` 넘기면 끝

이제 그래프가 **매 노드 실행마다 자동으로 상태를 DB에 저장**.

> `check_same_thread=False`는 Jupyter의 비동기 환경에서 SQLite를 안전하게 쓰기 위한 옵션.

### thread_id — 대화 세션 구분

```python
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [...]}, config=config)
```

**`thread_id`가 대화 세션 ID!**
- 같은 `thread_id` → 이전 대화 이어감 (이름 기억함)
- 다른 `thread_id` → 완전 새 대화 (이름 모름)

카카오톡 채팅방 번호라고 생각하면 됨.

### 실행 예시

```python
# thread_id="1"로 "내 이름은 Alice"
# → AI: "안녕하세요 Alice!"

# 같은 thread_id="1"로 "내 이름 뭐야?"
# → AI: "Alice입니다" (기억!)

# 다른 thread_id="2"로 "내 이름 뭐야?"
# → AI: "모르겠습니다" (새 대화!)
```

### 상태 히스토리

```python
for state in graph.get_state_history(config):
    print(f"next: {state.next}, messages: {len(state.values.get('messages', []))}")
```

`get_state_history()` — 그래프 실행의 **모든 중간 스냅샷** 조회.
각 노드 실행 후 상태가 어떻게 변했는지 시간순으로 볼 수 있음.
이게 14.4 타임 트래블의 기초가 된다.

---

## 14.3 Human-in-the-loop — "사람이 중간에 개입"

AI가 자동으로 끝까지 가는 게 아니라, **중간에 멈추고 사람 피드백을 받음**

### 핵심 2개

| 함수 | 역할 |
|------|------|
| `interrupt("메시지")` | 여기서 **멈춰!** 사용자한테 물어봐 |
| `Command(resume="답변")` | 사용자가 답 줬어, **다시 시작해** |

### 피드백 도구

```python
from langgraph.types import interrupt, Command

@tool
def get_human_feedback(poem: str):
    """Asks the user for feedback on the poem."""
    feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
    return feedback
```

이 도구가 호출되면:
1. `interrupt()` 실행 → **그래프 멈춤** (상태는 DB에 저장)
2. 사용자가 피드백 제공할 때까지 대기
3. `Command(resume="피드백 내용")` → `interrupt()`가 그 값을 **반환**
4. 반환값이 `ToolMessage`로 AI에게 전달 → AI가 다음 행동 결정

### 시스템 프롬프트

```python
def chatbot(state: State):
    response = llm_with_tools.invoke(
        f"""
        You are an expert in making poems.
        Use the `get_human_feedback` tool to get feedback on your poem.
        Only after you receive positive feedback you can return the final poem.
        ALWAYS ASK FOR FEEDBACK FIRST.
        ...
    """
    )
```

프롬프트에서 AI에게 명확한 지시:
- 시를 만드는 전문가 역할
- 반드시 `get_human_feedback` 도구를 사용할 것
- 긍정적 피드백 받은 후에만 최종 시를 반환할 것

### 실행 흐름 (3단계)

**1단계: 시 작성 요청 → interrupt에서 멈춤**

```python
config = {"configurable": {"thread_id": "poem_1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Please make a poem about Python code."}]},
    config=config,
)
snapshot = graph.get_state(config)
print("Next:", snapshot.next)  # ('tools',) ← 도구 노드에서 멈춤!
```

AI가 시를 쓰고 `get_human_feedback` 도구 호출 → `interrupt()` → **멈춤**

`snapshot.next`가 `('tools',)`면 = "도구 노드에서 사람 기다리는 중"

**2단계: 부정적 피드백 → 수정 요청**

```python
result = graph.invoke(
    Command(resume="It is too long! Make it shorter, 4 lines max."),
    config=config,
)
```

`Command(resume=...)` → 멈춘 지점에서 재개.
`interrupt()`가 `"It is too long!..."`을 반환 → AI가 받아서 짧게 수정 → 다시 피드백 요청 → 또 멈춤

**3단계: 긍정적 피드백 → 완료**

```python
result = graph.invoke(
    Command(resume="It looks great!"),
    config=config,
)
snapshot = graph.get_state(config)
print("Next:", snapshot.next)  # () ← 빈 튜플 = 완료!
```

### 전체 흐름도

```
invoke("시 써줘") → AI가 시 생성 → interrupt! (멈춤)
                  ↓
Command(resume="너무 길어") → AI가 수정 → interrupt! (또 멈춤)
                  ↓
Command(resume="좋아!") → AI가 최종 시 반환 → END (완료)
```

### snapshot.next 읽는 법

| `snapshot.next` 값 | 의미 |
|---------------------|------|
| `('tools',)` | 도구 노드에서 인터럽트 대기중 |
| `('chatbot',)` | 챗봇 노드 실행 대기중 |
| `()` (빈 튜플) | **그래프 실행 완료** |

---

## 14.4 타임 트래블 — "과거로 돌아가서 다른 선택"

체크포인터가 저장한 **스냅샷을 이용해 과거 시점으로 포크(fork)**

### 기본 대화

```python
config = {"configurable": {"thread_id": "tt_1"}}

# 1번째 대화: "나는 Valencia에 살아"
result = graph.invoke(
    {"messages": [{"role": "user", "content": "I live in Europe. My city is Valencia."}]},
    config=config,
)

# 2번째 대화: "근처 맛집 추천해줘"
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What are some good restaurants near me?"}]},
    config=config,
)
# → AI가 Valencia 맛집을 추천
```

### 상태 히스토리 탐색

```python
state_history = list(graph.get_state_history(config))
for i, snap in enumerate(state_history):
    print(f"Snapshot {i}: next={snap.next}, messages={len(snap.values.get('messages', []))}")
```

모든 스냅샷 조회. 시간 **역순**으로 나옴.
각 스냅샷에는:
- `values` — 그 시점의 전체 상태
- `next` — 다음에 실행될 노드
- `config` — 이 스냅샷의 고유 ID (checkpoint_id)

### 포크 — 과거 시점에서 다른 선택

```python
from langchain_core.messages import HumanMessage

# "Valencia"라고 말한 시점의 스냅샷 찾기
fork_point = None
for snap in state_history:
    if snap.next == ("chatbot",) and len(snap.values.get("messages", [])) == 1:
        fork_point = snap
        break

# 그 시점을 "Zagreb"로 바꿔서 새 분기 생성
graph.update_state(
    fork_point.config,
    {"messages": [HumanMessage(content="I live in Europe. My city is Zagreb.")]},
)

# 새 분기에서 실행
result_fork = graph.invoke(None, config=fork_point.config)
# → AI가 Zagreb 맛집을 추천!
```

**핵심: `update_state()`로 과거 체크포인트의 상태를 수정 → 새 분기 생성**

```
원래 타임라인:  Valencia → Valencia 맛집 추천
포크 타임라인:  Zagreb → Zagreb 맛집 추천

원래 대화는 그대로! 새 분기만 추가된 것.
```

### 실무 활용

| 활용 | 설명 |
|------|------|
| **디버깅** | AI가 이상한 답 했을 때 그 시점으로 돌아가서 원인 분석 |
| **A/B 테스팅** | 같은 시점에서 다른 입력으로 결과 비교 |
| **롤백** | 사용자가 "아까 그 답이 더 좋았어" 하면 되돌리기 |

---

## 핵심 개념 정리표

| 섹션 | 추가된 것 | 핵심 코드 | 비유 |
|------|----------|----------|------|
| **14.0** | MessagesState | `llm.invoke(state["messages"])` | 1회성 전화 |
| **14.1** | Tool Calling | `@tool` + `ToolNode` + `tools_condition` | 전화 중 검색 |
| **14.2** | Memory | `SqliteSaver` + `thread_id` | 채팅앱 (기록 남음) |
| **14.3** | HITL | `interrupt()` + `Command(resume=)` | 결재 시스템 |
| **14.4** | Time Travel | `get_state_history` + `update_state` | Git branch |
