# Chapter 19: 교육용 AI 에이전트 (Tutor Agent)

---

## 1. 챕터 개요

Chapter 19에서는 **교육용 AI 튜터 에이전트 시스템**을 구축한다. 이 시스템은 LangGraph의 멀티 에이전트 아키텍처를 활용하여 학습자의 수준을 평가하고, 그에 맞는 최적의 학습 방식으로 안내하는 지능형 교육 플랫폼이다.

### 시스템 구성 요소

| 에이전트 | 역할 | 학습 방법론 |
|---------|------|------------|
| **Classification Agent** | 학습자 수준 평가 및 라우팅 | 교육 평가 전문가 |
| **Teacher Agent** | 체계적 단계별 교육 | 구조화된 강의식 교육 |
| **Feynman Agent** | 파인만 기법 기반 이해도 검증 | "쉽게 설명하지 못하면 이해한 것이 아니다" |
| **Quiz Agent** | 퀴즈 기반 능동적 학습 평가 | 연구 기반 객관식 퀴즈 생성 |

### 핵심 학습 목표

- LangGraph의 `create_react_agent`를 사용한 멀티 에이전트 시스템 설계
- 에이전트 간 전환(Transfer) 패턴 구현
- `Command` 객체를 활용한 그래프 내 에이전트 라우팅
- 조건부 엣지(Conditional Edges)를 활용한 동적 워크플로우
- Pydantic 기반 구조화된 출력(Structured Output) 활용
- Firecrawl을 이용한 웹 검색 도구 구현

### 프로젝트 아키텍처

```
tutor-agent/
├── main.py                          # 메인 그래프 정의
├── langgraph.json                   # LangGraph 설정
├── pyproject.toml                   # 프로젝트 의존성
├── agents/
│   ├── classification_agent.py      # 학습자 분류 에이전트
│   ├── teacher_agent.py             # 교사 에이전트
│   ├── feynman_agent.py             # 파인만 기법 에이전트
│   └── quiz_agent.py                # 퀴즈 에이전트
└── tools/
    ├── shared_tools.py              # 공유 도구 (전환, 웹 검색)
    └── quiz_tools.py                # 퀴즈 생성 도구
```

### 에이전트 플로우 다이어그램

```
[START] → [router_check] ─→ [classification_agent] → [END]
                │
                ├─→ [teacher_agent]
                ├─→ [feynman_agent]
                └─→ [quiz_agent]
```

`classification_agent`가 학습자를 평가한 후, `transfer_to_agent` 도구를 통해 적절한 에이전트로 전환한다. 이후 대화 재개 시 `router_check`가 `current_agent` 상태를 확인하여 올바른 에이전트로 라우팅한다.

---

## 2. 섹션별 상세 설명

---

### 2.1 섹션 19.0 — Introduction (프로젝트 초기 설정)

**커밋**: `0516cd0` "19.0 Introduction"

#### 주제 및 목표

새로운 `tutor-agent` 프로젝트의 기반을 구축한다. Python 프로젝트 구조를 설정하고, 필요한 모든 의존성 패키지를 정의한다.

#### 핵심 개념 설명

##### 프로젝트 구조 초기화

`uv`(Python 패키지 매니저)를 사용하여 새 프로젝트를 생성한다. `uv`는 `pip`보다 훨씬 빠른 현대적 Python 패키지 관리 도구이다.

##### Python 버전 관리

```
3.13
```

`.python-version` 파일은 프로젝트에서 사용할 Python 버전을 명시한다. 이 파일은 `pyenv`, `uv` 등의 도구가 자동으로 인식하여 올바른 Python 버전을 사용하도록 한다.

##### 의존성 정의 (pyproject.toml)

```toml
[project]
name = "tutor-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "firecrawl-py==2.16",
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "langgraph-supervisor==0.0.29",
    "langgraph-swarm==0.0.14",
    "pytest==8.4.2",
    "python-dotenv==1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel==6.30.1",
]
```

**주요 패키지 설명:**

| 패키지 | 버전 | 역할 |
|--------|------|------|
| `firecrawl-py` | 2.16 | 웹 검색 및 스크래핑 API 클라이언트 |
| `grandalf` | 0.8 | 그래프 시각화 (LangGraph 그래프 렌더링용) |
| `langchain[openai]` | 0.3.27 | LangChain 프레임워크 + OpenAI 통합 |
| `langgraph` | 0.6.6 | 상태 기반 에이전트 그래프 프레임워크 |
| `langgraph-checkpoint-sqlite` | 2.0.11 | SQLite 기반 체크포인트 저장소 |
| `langgraph-cli[inmem]` | 0.4.0 | LangGraph CLI (인메모리 모드 포함) |
| `langgraph-supervisor` | 0.0.29 | 슈퍼바이저 에이전트 패턴 |
| `langgraph-swarm` | 0.0.14 | 스웜(Swarm) 에이전트 패턴 |
| `pytest` | 8.4.2 | 테스트 프레임워크 |
| `python-dotenv` | 1.1.1 | `.env` 파일에서 환경 변수 로드 |

##### langgraph-supervisor vs langgraph-swarm

이 프로젝트에서는 두 가지 멀티 에이전트 패턴 라이브러리를 모두 설치한다:
- **Supervisor 패턴**: 중앙 관리자가 에이전트를 지휘하는 방식
- **Swarm 패턴**: 에이전트들이 자율적으로 서로에게 작업을 전달하는 방식

이 프로젝트는 **Swarm에 가까운 패턴**을 사용하는데, 각 에이전트가 `transfer_to_agent` 도구를 통해 직접 다른 에이전트로 전환하기 때문이다.

#### 실습 포인트

1. `uv init tutor-agent`로 프로젝트 생성 후 `pyproject.toml`을 수정하여 의존성을 추가해 본다.
2. `uv sync`를 실행하여 모든 의존성을 설치한다.
3. `.env` 파일을 생성하고 `OPENAI_API_KEY`와 `FIRECRAWL_API_KEY`를 설정한다.

---

### 2.2 섹션 19.1 — Classification Agent (학습자 분류 에이전트)

**커밋**: `269599b` "19.1 Classification Agent"

#### 주제 및 목표

학습자의 수준, 학습 스타일, 학습 목표를 파악하여 최적의 학습 에이전트로 연결하는 **분류 에이전트**를 구현한다. 이 에이전트는 전체 시스템의 진입점(Entry Point) 역할을 한다.

#### 핵심 개념 설명

##### create_react_agent 이해

LangGraph의 `create_react_agent`는 **ReAct(Reasoning + Acting) 패턴**의 에이전트를 간편하게 생성하는 함수이다.

```python
from langgraph.prebuilt import create_react_agent

classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="...",       # 시스템 프롬프트
    tools=[...],        # 사용 가능한 도구 목록
)
```

**ReAct 패턴이란?**
- **Reasoning**: LLM이 현재 상황을 분석하고 다음 행동을 결정
- **Acting**: 결정에 따라 도구를 호출하거나 응답을 생성
- 이 두 단계를 반복하며 작업을 완수한다

##### Classification Agent의 평가 프로세스

```python
classification_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are an Educational Assessment Specialist. Your role is to understand
    each learner's knowledge level, learning style, and educational needs
    through conversation.

    ## Your Assessment Process:

    ### Phase 1: Topic & Current Knowledge
    - Ask what topic they want to learn about
    - Probe their current understanding with 2-3 targeted questions
    - Gauge their experience level: complete beginner, some knowledge, or intermediate

    ### Phase 2: Learning Preference Identification
    Ask strategic questions to identify their preferred learning approach:
    - **Examples vs Theory**: "Do you prefer learning through concrete examples
      or understanding the theory first?"
    - **Detail Level**: "Do you like simple, straightforward explanations
      or detailed technical depth?"
    - **Learning Pace**: "Do you prefer step-by-step breakdowns
      or big-picture overviews?"
    - **Interaction Style**: "Do you learn better by practicing with questions
      or by reading explanations?"

    ### Phase 3: Learning Goals & Preferences
    - What's their learning goal? (understand basics, pass test, apply in work, etc.)
    - How much time do they have?
    - Do they prefer structured lessons or flexible exploration?
    ...
    """,
    tools=[transfer_to_agent],
)
```

이 프롬프트의 핵심 설계 원칙:

1. **3단계 평가 구조**: 주제 파악 → 학습 선호도 → 학습 목표 순서로 체계적 평가
2. **과부하 방지**: "Don't overwhelm - max 2 questions at a time" — 한 번에 최대 2개 질문만
3. **암시적 단서 활용**: 사용자가 기술 용어를 올바르게 사용하면 어느 정도 기반이 있다고 판단

##### 에이전트 추천 로직

```python
    ## Your Recommendations & Transfer:
    After completing your assessment, choose the best learning approach
    and USE the transfer_to_agent tool:

    - **"quiz_agent"**: If they want to test knowledge, prefer active recall,
      or learn through practice
    - **"teacher_agent"**: If they need structured, step-by-step explanations
      or are beginners
    - **"feynman_agent"**: If they claim to understand concepts
      but may need validation
```

각 에이전트로의 전환 기준:
- **quiz_agent**: 능동적 회상(Active Recall)을 선호하는 학습자
- **teacher_agent**: 초보자 또는 체계적 설명이 필요한 학습자
- **feynman_agent**: 이해했다고 주장하지만 검증이 필요한 학습자

##### 개발자 치트 코드

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to a random agent (quiz_agent, teacher_agent, or feynman_agent)
    for testing purposes using the transfer_to_agent tool.
```

테스트 편의를 위해 "GODMODE"를 입력하면 평가 과정을 건너뛰고 바로 에이전트로 전환하는 기능을 넣었다. 이는 개발 중 빠른 테스트를 위한 실용적인 패턴이다.

##### transfer_to_agent 도구 (초기 버전)

```python
from langgraph.types import Command
from langchain_core.tools import tool


@tool
def transfer_to_agent(agent_name: str):
    """
    Transfer to the given agent

    Args:
        agent_name: Name of the agent to transfer to, one of:
                    'quiz_agent', 'teacher_agent' or 'feynman_agent'
    """
    return f"Transfer to {agent_name} completed."
    # return Command(
    #     goto=agent_name,
    #     graph=Command.PARENT,
    # )
```

**중요 포인트:** 이 초기 버전에서는 `Command` 기반의 실제 전환 로직이 **주석 처리**되어 있다. 아직 다른 에이전트 노드가 그래프에 등록되지 않았기 때문에, 단순히 문자열을 반환하는 스텁(Stub)으로 구현하였다. 이는 점진적 개발(Incremental Development) 전략의 좋은 예시이다.

##### 메인 그래프 구성

```python
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import START, END, StateGraph, MessagesState
from agents.classification_agent import classification_agent


class TutorState(MessagesState):
    pass


graph_builder = StateGraph(TutorState)

graph_builder.add_node("classification_agent", classification_agent)

graph_builder.add_edge(START, "classification_agent")
graph_builder.add_edge("classification_agent", END)

graph = graph_builder.compile()
```

**코드 분석:**

1. **`load_dotenv()`**: `.env` 파일에서 API 키 등의 환경 변수를 로드. **반드시 import 전에 호출**해야 한다 — 다른 모듈의 import 시점에 환경 변수가 필요할 수 있기 때문이다.

2. **`TutorState(MessagesState)`**: LangGraph의 `MessagesState`를 상속받아 대화 메시지를 자동으로 관리한다. 이 시점에서는 `pass`로 추가 상태 없이 사용한다.

3. **그래프 구조**: `START → classification_agent → END`라는 단순한 선형 구조이다.

##### LangGraph 설정 파일

```json
{
    "dependencies": [
        "agents/classification_agent.py",
        "tools/shared_tools.py",
        "main.py"
    ],
    "graphs": {
        "tutor": "./main.py:graph"
    },
    "env": "./env"
}
```

`langgraph.json`은 LangGraph CLI(`langgraph dev`)가 참조하는 설정 파일이다:
- **dependencies**: 의존 파일 목록 (변경 감지용)
- **graphs**: 노출할 그래프와 그 진입점
- **env**: 환경 변수 파일 경로

#### 실습 포인트

1. `langgraph dev`로 개발 서버를 실행하고 LangGraph Studio에서 그래프를 확인한다.
2. Classification Agent와 대화하며 평가 프로세스가 자연스럽게 진행되는지 테스트한다.
3. 프롬프트를 수정하여 다른 평가 기준을 추가해 본다 (예: "시각적 학습자 vs 청각적 학습자").

---

### 2.3 섹션 19.2 — Feynman Agent & Teacher Agent

**커밋**: `5c2dfa9` "19.2 Feynman Agent"

#### 주제 및 목표

이 섹션에서는 두 개의 핵심 학습 에이전트를 구현하고, 에이전트 간 전환 메커니즘을 완성한다:
- **Teacher Agent**: 체계적 단계별 교육
- **Feynman Agent**: 파인만 기법으로 이해도 검증
- **웹 검색 도구**: Firecrawl 기반 실시간 정보 검색
- **실제 에이전트 전환**: `Command` 객체를 활용한 라우팅

#### 핵심 개념 설명

##### Feynman Agent — 파인만 학습 기법

리처드 파인만(Richard Feynman)의 학습 철학을 AI 에이전트로 구현하였다. 핵심 원칙은 **"쉽게 설명하지 못하면 이해한 것이 아니다"**이다.

```python
from langgraph.prebuilt import create_react_agent
from tools.shared_tools import transfer_to_agent, web_search_tool


feynman_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Feynman Technique Master. Your approach follows the systematic
    Feynman Method: Research → Request Simple Explanation → Evaluate Complexity
    → Ask Clarifying Questions → Complete or Repeat.

    ## The Feynman Philosophy:
    "If you can't explain it simply, you don't understand it well enough."
    Your job is to reveal gaps in understanding through the power of
    simple explanation.
    ...
    """,
    tools=[
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**파인만 기법의 6단계 프로세스:**

| 단계 | 이름 | 설명 |
|------|------|------|
| Step 1 | Research Phase | 웹 검색으로 개념에 대한 정확한 정보 확보 |
| Step 2 | Request Simple Explanation | "8살 어린이에게 설명하듯" 요청 |
| Step 3 | Get User Explanation | 사용자 설명을 경청하고 분석 |
| Step 4 | Evaluate Complexity | 전문 용어, 논리적 빈틈, 모호한 설명 평가 |
| Step 5 | Ask Clarifying Questions | 복잡한 부분에 대해 구체적 질문 |
| Step 6 | Complete | 충분히 간결하면 마스터리 인정 |

**핵심 평가 기준:**

```
    ## Your Evaluation Criteria:
    - No unexplained technical terms
    - Clear cause-and-effect relationships
    - Uses analogies or examples a child would understand
    - Logical flow without gaps
    - Their own words, not memorized definitions
```

이 기준은 학습자가 단순히 정의를 암기했는지, 아니면 진정으로 이해했는지를 구분하는 데 사용된다. 특히 "자신만의 말(Their own words)"을 강조하는 것이 중요하다.

##### Teacher Agent — 체계적 교육 에이전트

```python
from langgraph.prebuilt import create_react_agent
from tools.shared_tools import transfer_to_agent, web_search_tool


teacher_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Master Teacher who builds understanding through structured,
    step-by-step learning. Your approach follows a proven teaching methodology:
    Research → Break Down → Explain → Confirm → Progress.
    ...
    """,
    tools=[
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**Teacher Agent의 교육 방법론:**

```
    ### Step 1: Research Phase
    - Use web_search_tool to get current, accurate information

    ### Step 2: Concept Breakdown
    - Divide complex topics into smaller, logical chunks
    - Arrange concepts from foundational to advanced

    ### Step 3: Explain One Concept at a Time
    - Use simple, clear language
    - Provide concrete examples and analogies
    - Present just ONE concept - don't overwhelm

    ### Step 4: Confirmation Check (Critical!)
    - Ask directly: "Does this make sense so far?"
    - Wait for their response and evaluate it carefully

    ### Step 5: Re-explain or Progress
    - If "No" or confused: Re-explain using different approach
    - If "Yes" and demonstrate understanding: Move to Step 6

    ### Step 6: Next Concept or Complete
    - More concepts: Move to next (back to Step 3)
    - Topic complete: Summarize connections
```

**Teacher Agent의 핵심 교육 규칙:**

```
    ## Critical Teaching Rules:
    1. Always confirm understanding before moving to the next concept
    2. If they don't understand, explain differently (not just repeat)
    3. Break complex topics into the smallest possible pieces
    4. Use examples from their world and experience
    5. Be patient - true understanding takes time
```

규칙 2번이 특히 중요하다 — 이해하지 못했을 때 같은 설명을 반복하는 것이 아니라, **다른 방식으로 설명**해야 한다. 이것은 실제 우수한 교사의 핵심 역량이기도 하다.

##### 웹 검색 도구 (web_search_tool)

```python
import re
import os
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain_core.tools import tool


@tool
def web_search_tool(query: str):
    """
    Web Search Tool.
    Args:
        query: str
            The query to search the web for.
    Returns
        A list of search results with the website content in Markdown format.
    """
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    response = app.search(
        query=query,
        limit=5,
        scrape_options=ScrapeOptions(
            formats=["markdown"],
        ),
    )

    if not response.success:
        return "Error using tool."

    cleaned_chunks = []

    for result in response.data:
        title = result["title"]
        url = result["url"]
        markdown = result["markdown"]

        cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
        cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)

        cleaned_result = {
            "title": title,
            "url": url,
            "markdown": cleaned,
        }

        cleaned_chunks.append(cleaned_result)

    return cleaned_chunks
```

**코드 분석:**

1. **FirecrawlApp**: Firecrawl API를 사용하여 웹 검색을 수행한다. Google 검색 + 페이지 스크래핑을 결합한 서비스이다.
2. **`limit=5`**: 검색 결과를 5개로 제한하여 토큰 비용을 절약한다.
3. **`ScrapeOptions(formats=["markdown"])`**: 결과를 마크다운 형식으로 받아 LLM이 처리하기 쉽게 한다.
4. **텍스트 정제(Cleaning)**:
   - `re.sub(r"\\+|\n+", "", markdown)`: 불필요한 이스케이프 문자와 과도한 줄바꿈 제거
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: 마크다운 링크와 URL 제거

이러한 정제 과정은 **토큰 절약**과 **노이즈 감소**를 위해 매우 중요하다. 웹 페이지의 원본 마크다운에는 네비게이션 링크, 광고 링크 등 불필요한 정보가 많기 때문이다.

##### transfer_to_agent 도구 완성 (Command 활용)

```python
@tool
def transfer_to_agent(agent_name: str):
    """
    Transfer to the given agent

    Args:
        agent_name: Name of the agent to transfer to, one of:
                    'teacher_agent' or 'feynman_agent'
    """
    return Command(
        goto=agent_name,
        graph=Command.PARENT,
        update={
            "current_agent": agent_name,
        },
    )
```

이전 섹션의 스텁 구현이 **실제 `Command` 기반 전환 로직**으로 변경되었다.

**`Command` 객체의 각 파라미터:**

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `goto` | `agent_name` | 이동할 대상 노드 이름 |
| `graph` | `Command.PARENT` | 부모 그래프에서 노드를 찾겠다는 의미 |
| `update` | `{"current_agent": agent_name}` | 그래프 상태를 업데이트 |

**`Command.PARENT`가 필요한 이유:**
`transfer_to_agent`는 도구(Tool) 내부에서 호출된다. 도구는 에이전트의 하위 컨텍스트에서 실행되므로, 그래프의 최상위 레벨에 있는 다른 노드로 이동하려면 `Command.PARENT`를 지정하여 부모 그래프 레벨에서의 전환임을 명시해야 한다.

##### 메인 그래프 진화 — 라우터 패턴

```python
from agents.classification_agent import classification_agent
from agents.teacher_agent import teacher_agent
from agents.feynman_agent import feynman_agent


class TutorState(MessagesState):
    current_agent: str


def router_check(state: TutorState):
    current_agent = state.get("current_agent", "classification_agent")
    return current_agent


graph_builder = StateGraph(TutorState)

graph_builder.add_node(
    "classification_agent",
    classification_agent,
    destinations=(
        "quiz_agent",
        "teacher_agent",
        "feynman_agent",
    ),
)
graph_builder.add_node("teacher_agent", teacher_agent)
graph_builder.add_node("feynman_agent", feynman_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
    ],
)
graph_builder.add_edge("classification_agent", END)

graph = graph_builder.compile()
```

**코드 분석:**

1. **`TutorState`에 `current_agent` 필드 추가**: 어떤 에이전트가 현재 활성 상태인지 추적한다.

2. **`router_check` 함수**: 대화가 재개될 때(새 메시지가 들어올 때) 현재 상태의 `current_agent` 값을 확인하여 적절한 노드로 라우팅한다. 기본값은 `"classification_agent"`이다.

3. **`destinations` 파라미터**: `classification_agent` 노드에 `destinations`를 지정하여 이 노드에서 도달 가능한 다른 노드를 LangGraph에 알려준다. 이는 `Command`를 사용한 에이전트 전환이 정상적으로 작동하기 위해 필요하다.

4. **`add_conditional_edges`**: `START`에서 조건부 분기를 추가한다. `router_check` 함수의 반환값에 따라 세 에이전트 중 하나로 진입한다.

**라우팅 흐름:**
```
새 대화 시작:
  current_agent 미설정 → router_check 기본값 → "classification_agent"

classification_agent가 transfer_to_agent("teacher_agent") 호출:
  Command(goto="teacher_agent", update={"current_agent": "teacher_agent"})
  → teacher_agent로 전환

다음 메시지:
  current_agent = "teacher_agent" → router_check → teacher_agent로 직행
```

#### 실습 포인트

1. Feynman Agent에게 자신이 잘 아는 개념을 설명해 보고, 어떤 피드백을 주는지 확인한다.
2. Teacher Agent에게 새로운 주제를 학습 요청하여 단계별 교육 과정을 경험한다.
3. `web_search_tool`의 정규식을 수정하여 다른 정제 전략을 실험해 본다.
4. `router_check` 함수에 로깅을 추가하여 라우팅 과정을 추적해 본다.

---

### 2.4 섹션 19.3 — Quiz Agent (퀴즈 에이전트)

**커밋**: `e188909` "19.3 Quiz Agent"

#### 주제 및 목표

웹 검색 결과를 기반으로 구조화된 객관식 퀴즈를 동적 생성하는 **Quiz Agent**를 구현한다. Pydantic의 **Structured Output**을 활용하여 LLM이 정해진 형식의 퀴즈를 생성하도록 한다.

#### 핵심 개념 설명

##### Pydantic 기반 구조화된 출력 (Structured Output)

이 섹션의 가장 중요한 기술적 개념은 **Structured Output**이다. LLM에게 자유 형식의 텍스트가 아닌, 미리 정의된 스키마에 맞는 데이터를 생성하도록 요청한다.

```python
from pydantic import BaseModel, Field
from typing import Literal, List


class Question(BaseModel):

    question: str = Field(description="The quiz question text")
    options: List[str] = Field(
        description="Exactly 4 multiple choice options, labeled A, B, C, D."
    )
    correct_answer: str = Field(
        description="The correct answer (MUST MATCH ONE OF 'options')"
    )
    explanation: str = Field(
        description="Exaplanation of why the answer is correct "
                    "and the other ones are wrong."
    )


class Quiz(BaseModel):
    topic: str = Field(description="The main topic being tested")
    questions: List[Question] = Field(
        description="List of the quiz questions"
    )
```

**스키마 설계의 핵심:**

1. **`Question` 모델**: 각 문제의 구조를 엄격하게 정의
   - `question`: 질문 텍스트
   - `options`: 정확히 4개의 보기 (A, B, C, D)
   - `correct_answer`: 정답 (반드시 options 중 하나와 일치)
   - `explanation`: 정답의 이유와 오답의 이유 설명

2. **`Quiz` 모델**: 퀴즈 전체 구조
   - `topic`: 퀴즈 주제
   - `questions`: Question 객체의 리스트

3. **`Field(description=...)`**: 각 필드의 설명을 LLM에게 전달하여 더 정확한 출력을 유도한다. 특히 `"MUST MATCH ONE OF 'options'"`처럼 제약 조건을 명시하는 것이 중요하다.

##### generate_quiz 도구

```python
from langchain.chat_models import init_chat_model


@tool
def generate_quiz(
    research_text: str,
    topic: str,
    difficulty: Literal[
        "easy",
        "medium",
        "hard",
    ],
    num_questions: int,
):
    """
    Generate a structured quiz with multiple choice questions
    based on research information.

    Args:
        research_text: str - Research information about the topic.
        topic: str - The main topic/subject for the quiz
        difficulty: Literal["easy", "medium", "hard"] - The difficulty level
        num_questions: int - Number of questions to generate (between 1-30)

    Returns:
        Quiz object with structured questions
    """
    model = init_chat_model("openai:gpt-4o")
    structured_model = model.with_structured_output(Quiz)

    prompt = f"""
    Create a {difficulty} quiz, about {topic} with {num_questions}
    using the following research information.

    <RESEARCH_INFORMATION>
    {research_text}
    </RESEARCH_INFORMATION>

    Make sure to use the RESEARCH_INFORMATION to create
    the most accurate questions.
    """

    quiz = structured_model.invoke(prompt)

    return quiz
```

**코드 분석:**

1. **`init_chat_model("openai:gpt-4o")`**: LangChain의 범용 모델 초기화 함수. 문자열 형식으로 프로바이더와 모델을 지정한다.

2. **`model.with_structured_output(Quiz)`**: 이것이 핵심이다. 일반 채팅 모델을 **구조화된 출력 모델**로 변환한다. OpenAI의 JSON mode / function calling을 내부적으로 활용하여 `Quiz` Pydantic 모델에 맞는 출력을 보장한다.

3. **`Literal["easy", "medium", "hard"]`**: Python의 타입 힌트를 사용하여 difficulty 파라미터를 3가지 값으로 제한한다. LLM이 도구를 호출할 때 이 제약 조건을 인식한다.

4. **`<RESEARCH_INFORMATION>` XML 태그**: 프롬프트 내에서 연구 데이터를 명확히 구분하기 위해 XML 태그를 사용한다. LLM이 프롬프트 지시사항과 데이터를 혼동하지 않도록 한다.

**도구 내부에서 별도의 LLM을 호출하는 패턴:**
`generate_quiz`는 도구(Tool)이지만, 내부에서 다시 LLM을 호출한다. 이것은 **"도구로서의 에이전트(Agent as Tool)"** 패턴의 변형이다. 외부 에이전트(Quiz Agent)가 이 도구를 호출하면, 도구 내부의 LLM이 구조화된 퀴즈를 생성하여 반환한다. 이렇게 하면 퀴즈 생성 로직을 깔끔하게 캡슐화할 수 있다.

##### Quiz Agent 프롬프트 — 엄격한 워크플로우

```python
quiz_agent = create_react_agent(
    model="openai:gpt-4o",
    prompt="""
    You are a Quiz Master and Learning Assessment Specialist.
    Your role is to create engaging, research-based quizzes
    and provide detailed educational feedback.

    ## Your Tools:
    - **web_search_tool**: Research current information on any topic
    - **generate_quiz**: Create structured multiple-choice quizzes
      based on research data
    - **transfer_to_agent**: Switch to other learning agents when appropriate

    ## Your Systematic Quiz Process:

    ### Step 1: Research the Topic
    - Use web_search_tool to gather current, accurate information

    ### Step 2: Ask About Quiz Length
    - **"short"**: 3-5 questions
    - **"medium"**: 6-10 questions
    - **"long"**: 11-15 questions

    ### Step 3: Generate Structured Quiz
    Use the generate_quiz tool with research_text, topic, difficulty,
    num_questions

    ### Step 4: Present Questions One by One
    - Wait for their answer before revealing the correct answer

    ### Step 5: Provide Detailed Feedback
    - If Correct: celebration + explanation
    - If Incorrect: correct answer + detailed explanation

    ### Step 6: Continue Through Quiz
    - Keep track of score, provide final summary
    ...
    """,
    tools=[
        generate_quiz,
        transfer_to_agent,
        web_search_tool,
    ],
)
```

**프롬프트의 워크플로우 강제 패턴:**

```
    ## CRITICAL WORKFLOW - MUST FOLLOW IN ORDER:
    1. STEP 1: RESEARCH FIRST - You MUST use web_search_tool before anything else
    2. STEP 2: ASK LENGTH - Ask student how many questions they want
    3. STEP 3: CALL generate_quiz - Pass the research_text from step 1
    4. STEP 4: PRESENT ONE BY ONE - Show questions individually
    5. STEP 5: USE EXPLANATIONS - Use the explanations provided by the quiz tool

    NEVER call generate_quiz without research_text from web_search_tool first!
```

이 부분은 **LLM의 행동을 엄격하게 제어**하기 위한 프롬프트 엔지니어링이다. "CRITICAL", "MUST", "NEVER" 같은 강조 표현과 순서 번호를 사용하여 LLM이 정해진 순서를 따르도록 한다. 특히 웹 검색 없이 퀴즈를 생성하면 부정확한 정보가 포함될 수 있으므로, 반드시 리서치를 먼저 수행하도록 강제한다.

##### 메인 그래프에 Quiz Agent 통합

```python
from agents.quiz_agent import quiz_agent

# ... 기존 코드에 추가 ...
graph_builder.add_node("quiz_agent", quiz_agent)

graph_builder.add_conditional_edges(
    START,
    router_check,
    [
        "teacher_agent",
        "feynman_agent",
        "classification_agent",
        "quiz_agent",       # 추가됨
    ],
)
```

quiz_agent 노드를 그래프에 등록하고, `router_check`의 조건부 엣지 목록에 추가한다.

##### Classification Agent 업데이트

```python
    ## Developer Cheat Code:
    If the user says "GODMODE", skip all assessment and immediately transfer
    to quiz_agent for testing purposes using the transfer_to_agent tool.
```

GODMODE 치트 코드가 "랜덤 에이전트"에서 **`quiz_agent` 고정**으로 변경되었다. 새로 추가된 Quiz Agent를 집중 테스트하기 위한 변경이다.

##### shared_tools.py 업데이트

```python
    agent_name: Name of the agent to transfer to, one of:
                'quiz_agent', 'teacher_agent' or 'feynman_agent'
```

`transfer_to_agent`의 docstring에 `quiz_agent`가 다시 추가되었다. LLM은 docstring을 읽고 도구의 파라미터 값을 결정하므로, 여기에 가능한 에이전트 이름을 정확하게 나열하는 것이 중요하다.

#### 실습 포인트

1. Quiz Agent에게 특정 주제로 퀴즈를 요청하고, 생성되는 퀴즈의 품질을 평가해 본다.
2. `generate_quiz`의 `difficulty` 파라미터를 바꿔가며 난이도별 문제 차이를 확인한다.
3. `Question` 모델에 `hint` 필드를 추가하여 힌트 기능을 구현해 본다.
4. 퀴즈 결과를 상태에 저장하여 학습 이력을 추적하는 기능을 설계해 본다.

---

## 3. 챕터 핵심 정리

### 아키텍처 패턴

1. **멀티 에이전트 Swarm 패턴**: 에이전트들이 `Command` 객체를 사용하여 자율적으로 서로에게 작업을 전달한다. 중앙 관리자 없이 각 에이전트가 판단하여 적절한 에이전트로 전환한다.

2. **조건부 라우팅 패턴**: `router_check` 함수와 `add_conditional_edges`를 결합하여 대화 상태에 따라 적절한 에이전트로 자동 라우팅한다.

3. **점진적 개발 전략**: 스텁 → 실제 구현 순서로 개발한다 (19.1에서 `transfer_to_agent`를 스텁으로 만들고 19.2에서 완성).

### 프롬프트 엔지니어링

4. **단계별 프로세스 프롬프트**: 모든 에이전트가 명확한 단계(Step 1, Step 2...)를 가진 체계적 프롬프트를 사용한다. 이는 LLM의 행동을 예측 가능하게 만든다.

5. **역할 기반 페르소나**: 각 에이전트에 "Educational Assessment Specialist", "Master Teacher", "Feynman Technique Master", "Quiz Master"와 같은 구체적 역할을 부여하여 행동 범위를 제한한다.

6. **강제 워크플로우 패턴**: "CRITICAL", "MUST", "NEVER" 같은 강조 표현으로 LLM이 정해진 순서를 따르도록 프롬프트를 설계한다.

### 도구 설계

7. **Structured Output**: Pydantic 모델 + `with_structured_output()`을 사용하여 LLM 출력을 프로그래밍적으로 처리 가능한 구조화된 데이터로 변환한다.

8. **도구 내 LLM 호출 패턴**: `generate_quiz` 도구가 내부에서 별도의 LLM을 호출하여 구조화된 퀴즈를 생성한다. 이는 로직 캡슐화의 좋은 예시이다.

9. **웹 검색 결과 정제**: 정규식을 사용하여 불필요한 링크, URL, 이스케이프 문자를 제거함으로써 토큰을 절약하고 LLM 입력 품질을 높인다.

### LangGraph 핵심 API

| API | 용도 |
|-----|------|
| `create_react_agent()` | ReAct 패턴 에이전트 생성 |
| `StateGraph` | 상태 기반 그래프 정의 |
| `MessagesState` | 메시지 기반 상태 관리 |
| `Command(goto, graph, update)` | 그래프 내 에이전트 전환 |
| `Command.PARENT` | 부모 그래프 레벨에서의 전환 |
| `add_conditional_edges()` | 조건부 분기 엣지 추가 |
| `add_node(destinations=...)` | 노드에서 도달 가능한 대상 명시 |

---

## 4. 실습 과제

### 과제 1: 새로운 에이전트 추가 (기본)

**Flashcard Agent**를 만들어 시스템에 추가하라.

- `agents/flashcard_agent.py` 생성
- 학습 주제에 대해 웹 검색 후 플래시카드(앞면: 질문, 뒷면: 답변)를 생성
- Pydantic 모델로 `Flashcard`와 `FlashcardDeck` 스키마 정의
- `main.py`의 그래프에 노드 추가 및 라우팅 설정
- `classification_agent`의 프롬프트에 flashcard_agent 전환 조건 추가

### 과제 2: 학습 이력 추적 (중급)

`TutorState`를 확장하여 학습 이력을 추적하는 기능을 구현하라.

- `TutorState`에 `quiz_scores: list[dict]`, `topics_learned: list[str]`, `current_topic: str` 필드 추가
- Quiz Agent가 퀴즈 결과(점수, 주제, 날짜)를 상태에 저장하도록 수정
- Teacher Agent가 이전에 학습한 주제를 참조하여 연관 개념을 제안하도록 수정
- 학습 이력을 요약하는 `progress_report` 도구 구현

### 과제 3: 적응형 난이도 조절 (고급)

학습자의 퀴즈 성적에 따라 자동으로 난이도를 조절하는 시스템을 구현하라.

- 연속 정답률이 80% 이상이면 난이도를 한 단계 올림
- 연속 정답률이 50% 미만이면 Teacher Agent로 자동 전환
- 난이도 변경 이력을 상태에 기록
- 최종 학습 보고서에 난이도 변화 곡선을 텍스트로 시각화

### 과제 4: 에이전트 간 컨텍스트 공유 (고급)

현재 시스템에서는 에이전트 간 전환 시 이전 에이전트의 평가 결과가 새 에이전트에 명시적으로 전달되지 않는다.

- `TutorState`에 `learner_profile: dict` 필드를 추가하여 Classification Agent의 평가 결과를 구조화하여 저장
- 각 에이전트가 `learner_profile`을 참조하여 학습자 수준에 맞는 응답을 생성하도록 프롬프트 수정
- `transfer_to_agent` 도구에 전환 사유(`transfer_reason`)를 추가하여 다음 에이전트가 맥락을 이해하도록 구현

### 과제 5: 커스텀 도구 개발 (심화)

`generate_quiz` 패턴을 참고하여 다음 도구들을 구현하라.

- **`generate_summary`**: 웹 검색 결과를 기반으로 학습 요약본을 생성하는 도구. Pydantic 모델로 `Section`, `Summary` 스키마 정의.
- **`evaluate_explanation`**: Feynman Agent에서 사용할 수 있는 자동 평가 도구. 학습자의 설명을 분석하여 전문 용어 사용 여부, 논리적 일관성, 간결성을 0-10 점수로 평가.

---

> **참고**: 이 챕터의 코드를 실행하려면 `OPENAI_API_KEY`와 `FIRECRAWL_API_KEY` 환경 변수가 필요합니다. `.env` 파일에 설정하거나 터미널에서 `export`로 지정하세요. LangGraph Studio에서 시각적으로 그래프를 확인하려면 `langgraph dev` 명령어를 사용하세요.
