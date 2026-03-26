# Chapter 6: AutoGen 고급 멀티 에이전트 프로젝트

---

## 챕터 개요

이번 챕터에서는 Microsoft의 **AutoGen** 프레임워크를 활용하여 실전 수준의 멀티 에이전트 시스템을 구축한다. 이전 챕터들에서 배운 에이전트 기초 개념을 넘어, 여러 에이전트가 **팀(Team)**을 이루어 협업하는 고급 패턴을 학습한다.

본 챕터는 두 가지 핵심 프로젝트를 통해 진행된다:

1. **이메일 최적화 팀 (Email Optimizer Team)**: `RoundRobinGroupChat`을 사용하여 에이전트들이 순서대로 돌아가며 이메일을 개선하는 파이프라인 구축
2. **딥 리서치 클론 (Deep Research Clone)**: `SelectorGroupChat`을 사용하여 AI가 자동으로 적절한 에이전트를 선택하며 웹 리서치를 수행하는 지능형 연구 시스템 구축

이 두 프로젝트를 통해 **에이전트 오케스트레이션의 두 가지 핵심 패턴** -- 순차적 파이프라인과 동적 선택 -- 을 깊이 이해할 수 있다.

### 학습 목표

- AutoGen 프레임워크의 팀(Team) 개념과 그룹 채팅 패턴 이해
- `RoundRobinGroupChat`을 활용한 순차적 멀티 에이전트 파이프라인 설계
- `SelectorGroupChat`을 활용한 동적 에이전트 선택 시스템 구현
- 외부 도구(Tool)를 에이전트에 연결하는 방법 습득
- 종료 조건(Termination Condition)을 활용한 워크플로우 제어
- 실제 웹 검색 API(Firecrawl)를 에이전트에 통합하는 실습

### 사용 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.13+ | 런타임 |
| AutoGen (autogen) | >= 0.9.7 | 에이전트 프레임워크 코어 |
| autogen-agentchat | >= 0.7.2 | 팀/그룹 채팅 기능 |
| autogen-ext[openai] | >= 0.7.2 | OpenAI 모델 연동 |
| firecrawl-py | >= 2.16.5 | 웹 검색 및 스크래핑 |
| python-dotenv | >= 1.1.1 | 환경변수 관리 |
| ipykernel | >= 6.30.1 | Jupyter 노트북 실행 |
| gpt-4o-mini | - | LLM 모델 |

---

## 6.0 프로젝트 소개 및 환경 설정

### 주제 및 목표

이 섹션에서는 "Deep Research Clone" 프로젝트의 기본 구조를 설정한다. `pyproject.toml` 파일을 통해 프로젝트의 의존성을 정의하고, **uv** 패키지 관리자를 사용한 Python 프로젝트 초기화 방법을 학습한다.

### 핵심 개념 설명

#### pyproject.toml이란?

`pyproject.toml`은 현대 Python 프로젝트의 표준 설정 파일이다. 과거 `setup.py`나 `requirements.txt` 방식을 대체하며, 프로젝트 메타데이터와 의존성을 하나의 파일에서 관리할 수 있게 해준다.

#### uv 패키지 관리자

이 프로젝트는 **uv**라는 차세대 Python 패키지 관리자를 사용한다. uv는 Rust로 작성되어 pip보다 10-100배 빠른 속도를 제공하며, 가상환경 생성과 의존성 관리를 통합적으로 처리한다.

### 코드 분석

```toml
[project]
name = "deep-research-clone"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "autogen>=0.9.7",
    "autogen-agentchat>=0.7.2",
    "autogen-ext[openai]>=0.7.2",
    "firecrawl-py>=2.16.5",
    "python-dotenv>=1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

**의존성 분석:**

| 패키지 | 역할 |
|--------|------|
| `autogen` | AutoGen 프레임워크 코어. 에이전트 생성 및 관리의 기반 |
| `autogen-agentchat` | 그룹 채팅(팀) 기능 제공. `RoundRobinGroupChat`, `SelectorGroupChat` 등 |
| `autogen-ext[openai]` | OpenAI 모델(GPT-4o-mini 등)과의 연동 확장 |
| `firecrawl-py` | 웹 검색 및 웹페이지 콘텐츠 추출 API 클라이언트 |
| `python-dotenv` | `.env` 파일에서 API 키 등 환경변수를 로드 |
| `ipykernel` | Jupyter 노트북에서 Python 커널 사용 (개발 의존성) |

**주목할 점:**
- `requires-python = ">=3.13"`으로 최신 Python 버전을 요구한다
- `[dependency-groups]`의 `dev` 그룹은 개발 환경에서만 필요한 패키지를 분리한다
- `autogen-ext[openai]`에서 대괄호 `[openai]`는 "extras"라 불리는 선택적 의존성을 의미한다

### 실습 포인트

1. **프로젝트 초기화**: 터미널에서 다음 명령어로 프로젝트를 시작할 수 있다:
   ```bash
   mkdir deep-research-clone
   cd deep-research-clone
   uv init
   ```

2. **의존성 설치**: uv를 사용하여 의존성을 설치한다:
   ```bash
   uv add autogen autogen-agentchat "autogen-ext[openai]" firecrawl-py python-dotenv
   uv add --dev ipykernel
   ```

3. **환경변수 설정**: `.env` 파일을 생성하여 API 키를 관리한다:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   echo "FIRECRAWL_API_KEY=fc-your-key-here" >> .env
   ```

---

## 6.1 이메일 최적화 팀 (Email Optimizer Team)

### 주제 및 목표

이 섹션에서는 **RoundRobinGroupChat**을 사용하여 5개의 전문 에이전트가 순서대로 이메일을 개선하는 파이프라인을 구축한다. 각 에이전트는 고유한 전문 분야(명확성, 톤, 설득력, 종합, 비평)를 담당하며, 라운드 로빈 방식으로 순차적으로 작업을 수행한다.

### 핵심 개념 설명

#### RoundRobinGroupChat

`RoundRobinGroupChat`은 AutoGen에서 제공하는 가장 단순하면서도 강력한 팀 패턴이다. 참여 에이전트들이 **정해진 순서대로 돌아가며** 발언하는 방식으로 동작한다.

```
사용자 입력 → ClarityAgent → ToneAgent → PersuasionAgent → SynthesizerAgent → CriticAgent
                  ↑                                                                    |
                  └────────────────── (기준 미달 시 다시 순환) ──────────────────────────┘
```

이 패턴은 다음과 같은 상황에 적합하다:
- 각 에이전트의 역할이 명확히 구분되어 있을 때
- 순차적 개선(iterative refinement)이 필요할 때
- 워크플로우의 순서가 고정되어 있을 때

#### 종료 조건 (Termination Conditions)

AutoGen은 팀의 실행을 제어하기 위해 종료 조건을 제공한다. 이 프로젝트에서는 두 가지 종료 조건을 조합하여 사용한다:

- **TextMentionTermination**: 특정 텍스트(예: "TERMINATE")가 에이전트 응답에 포함되면 종료
- **MaxMessageTermination**: 최대 메시지 수에 도달하면 종료

이 두 조건을 `|` (OR) 연산자로 결합하면, **둘 중 하나라도 충족되면** 팀이 중단된다.

#### 에이전트 전문화 (Agent Specialization)

멀티 에이전트 시스템의 핵심은 각 에이전트에게 **하나의 명확한 역할**을 부여하는 것이다. 하나의 "만능" 에이전트보다 여러 전문 에이전트가 협업하는 것이 더 나은 결과를 만들어낸다. 이는 소프트웨어 공학의 **단일 책임 원칙(Single Responsibility Principle)**과 유사하다.

### 코드 분석

#### 1단계: 임포트 및 모델 설정

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
```

각 임포트의 역할:
- `RoundRobinGroupChat`: 순차적 그룹 채팅 팀 클래스
- `AssistantAgent`: AI 모델 기반 에이전트 클래스
- `OpenAIChatCompletionClient`: OpenAI API와의 통신 클라이언트
- `MaxMessageTermination`, `TextMentionTermination`: 종료 조건 클래스들
- `Console`: 팀 실행 결과를 콘솔에 실시간 출력하는 UI 유틸리티

#### 2단계: 전문 에이전트 정의

이 프로젝트의 핵심은 5개의 전문 에이전트를 설계하는 것이다. 각 에이전트의 `system_message`가 어떻게 역할을 제한하고 집중시키는지 주목하자.

**ClarityAgent (명확성 전문가)**

```python
clarity_agent = AssistantAgent(
    "ClarityAgent",
    model_client=model,
    system_message="""You are an expert editor focused on clarity and simplicity.
            Your job is to eliminate ambiguity, redundancy, and make every sentence
            crisp and clear. Don't worry about persuasion or tone — just make the
            message easy to read and understand.""",
)
```

> **설계 포인트**: "Don't worry about persuasion or tone"라는 명시적 범위 제한에 주목하라. 이렇게 역할 범위를 명확히 제한해야 다른 에이전트와의 역할 충돌을 방지할 수 있다.

**ToneAgent (톤 전문가)**

```python
tone_agent = AssistantAgent(
    "ToneAgent",
    model_client=model,
    system_message="""You are a communication coach focused on emotional tone and
            professionalism. Your job is to make the email sound warm, confident,
            and human — while staying professional and appropriate for the audience.
            Improve the emotional resonance, polish the phrasing, and adjust any
            words that may come off as stiff, cold, or overly casual.""",
)
```

> **설계 포인트**: 감정적 톤과 전문성 사이의 균형을 명시적으로 요구한다. "warm, confident, and human" vs "professional and appropriate"이라는 상반될 수 있는 요소를 동시에 지시하여 균형 잡힌 결과를 유도한다.

**PersuasionAgent (설득력 전문가)**

```python
persuasion_agent = AssistantAgent(
    "PersuasionAgent",
    model_client=model,
    system_message="""You are a persuasion expert trained in marketing, behavioral
            psychology, and copywriting. Your job is to enhance the email's persuasive
            power: improve call to action, structure arguments, and emphasize benefits.
            Remove weak or passive language.""",
)
```

> **설계 포인트**: 마케팅, 행동심리학, 카피라이팅이라는 구체적인 전문 분야를 명시함으로써 에이전트의 응답 품질을 높인다. "Remove weak or passive language"처럼 구체적인 행동 지시가 포함되어 있다.

**SynthesizerAgent (종합 전문가)**

```python
synthesizer_agent = AssistantAgent(
    "SynthesizerAgent",
    model_client=model,
    system_message="""You are an advanced email-writing specialist. Your role is to
            read all prior agent responses and revisions, and then **synthesize the
            best ideas** into a unified, polished draft of the email. Focus on:
            Integrating clarity, tone, and persuasion improvements; Ensuring coherence,
            fluency, and a natural voice; Creating a version that feels professional,
            effective, and readable.""",
)
```

> **설계 포인트**: 이 에이전트는 앞선 세 에이전트의 결과를 **종합**하는 메타 역할을 수행한다. "read all prior agent responses"라는 지시를 통해 이전 대화 맥락을 적극 활용하도록 유도한다. 이것이 RoundRobin 패턴에서 Synthesizer 에이전트의 핵심 가치이다.

**CriticAgent (비평 전문가)**

```python
critic_agent = AssistantAgent(
    "CriticAgent",
    model_client=model,
    system_message="""You are an email quality evaluator. Your job is to perform a
            final review of the synthesized email and determine if it meets professional
            standards. Review the email for: Clarity and flow, appropriate professional
            tone, effective call-to-action, and overall coherence. Be constructive but
            decisive. If the email has major flaws (unclear message, unprofessional tone,
            or missing key elements), provide ONE specific improvement suggestion.
            If the email meets professional standards and communicates effectively,
            respond with 'The email meets professional standards.' followed by
            `TERMINATE` on a new line. You should only approve emails that are perfect
            enough for professional use, dont settle.""",
)
```

> **설계 포인트**: CriticAgent는 "게이트키퍼" 역할을 한다. 품질 기준을 통과하면 "TERMINATE"를 출력하여 팀을 종료시키고, 미달이면 개선 제안을 하여 다음 라운드를 유발한다. "dont settle"이라는 지시로 높은 품질 기준을 유지하도록 한다.

#### 3단계: 종료 조건 설정

```python
text_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=30)

termination_condition = text_termination | max_messages_termination
```

이 설정의 의미:
- CriticAgent가 "TERMINATE"를 출력하면 **즉시 종료** (품질 승인)
- 최대 30개 메시지에 도달하면 **강제 종료** (무한 루프 방지)
- `|` 연산자는 OR 조건을 의미한다. 둘 중 하나만 충족되면 종료

> **설계 원칙**: 항상 `MaxMessageTermination`을 안전장치로 포함해야 한다. 에이전트가 합의에 도달하지 못하고 무한히 순환하는 상황을 방지하기 위함이다.

#### 4단계: 팀 생성 및 실행

```python
team = RoundRobinGroupChat(
    participants=[
        clarity_agent,
        tone_agent,
        persuasion_agent,
        synthesizer_agent,
        critic_agent,
    ],
    termination_condition=termination_condition,
)

await Console(
    team.run_stream(
        task="Hi! Im hungry, buy me lunch and invest in my business. Thanks."
    )
)
```

**핵심 분석:**

- `participants` 리스트의 **순서가 곧 실행 순서**이다. ClarityAgent가 먼저, CriticAgent가 마지막으로 동작한다.
- `run_stream()`은 비동기 스트리밍 방식으로 팀을 실행한다. `Console`로 감싸면 실시간으로 각 에이전트의 응답을 확인할 수 있다.
- `await` 키워드는 이 코드가 비동기(async) 환경에서 실행됨을 나타낸다. Jupyter 노트북에서는 최상위 `await`이 자동으로 지원된다.

#### 실행 결과 분석

실제 실행 결과를 단계별로 살펴보면:

**1) 입력 (User)**:
```
Hi! Im hungry, buy me lunch and invest in my business. Thanks.
```
비격식적이고 직접적인, 전문적이지 않은 이메일이다.

**2) ClarityAgent 출력**:
```
Hi! I'm hungry. Please buy me lunch and invest in my business. Thank you.
```
문법 오류 수정("Im" → "I'm"), 문장 분리, 공손한 표현 추가에 집중했다.

**3) ToneAgent 출력**:
```
Subject: A Quick Favor

Hi there!
I hope you're doing well! I find myself feeling a bit peckish today...
Warm regards,
[Your Name]
```
따뜻하고 전문적인 톤으로 완전히 재구성했다. 제목과 인사말, 서명을 추가했다.

**4) PersuasionAgent 출력**:
```
Subject: Let's Make Delicious Opportunities Happen!

Hi [Recipient's Name],
...I promise to make it worth your while...
Together, we can turn potential into profit!
```
설득력 있는 언어("turn potential into profit"), 행동 유도("I promise to make it worth your while"), 혜택 강조를 추가했다.

**5) SynthesizerAgent 출력**:
앞선 세 에이전트의 최선의 요소를 통합한 최종 이메일을 작성했다.

**6) CriticAgent 출력**:
```
The email meets professional standards.
TERMINATE
```
품질 기준을 통과하여 "TERMINATE"를 출력, 팀 실행이 종료되었다.

### 도구 파일: tools.py

이메일 최적화 팀에서는 도구가 필요하지 않지만, 다음 섹션을 위해 `tools.py` 파일이 함께 생성되었다.

```python
import os, re
from firecrawl import FirecrawlApp, ScrapeOptions


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

이 웹 검색 도구는 Firecrawl API를 활용한다:

1. **FirecrawlApp 초기화**: 환경변수에서 API 키를 로드하여 클라이언트를 생성
2. **검색 실행**: 쿼리에 대해 최대 5개의 검색 결과를 마크다운 형식으로 가져옴
3. **결과 정제**: 두 단계의 정규표현식 처리로 불필요한 요소를 제거
   - `re.sub(r"\\+|\n+", "", markdown)`: 과도한 줄바꿈과 역슬래시 제거
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: 마크다운 링크와 URL 제거

> **왜 텍스트를 정제하는가?** 웹 스크래핑 결과에는 네비게이션 링크, 광고 URL 등 불필요한 콘텐츠가 많이 포함된다. 이를 제거하면 LLM에 전달되는 토큰 수를 줄이고, 핵심 정보에 집중할 수 있게 된다.

### 실습 포인트

1. **에이전트 순서 실험**: `participants` 리스트의 순서를 변경하여 결과가 어떻게 달라지는지 확인해보자. 예를 들어, PersuasionAgent를 ClarityAgent보다 먼저 배치하면 어떤 결과가 나올까?

2. **에이전트 추가/제거**: 새로운 전문 에이전트(예: "BrevityAgent" -- 간결함 전문가)를 추가하거나, 기존 에이전트를 제거하여 결과의 차이를 관찰해보자.

3. **종료 조건 변경**: `MaxMessageTermination`의 값을 10으로 줄이거나 50으로 늘려보자. CriticAgent의 기준을 더 엄격하게 만들어 여러 라운드가 실행되도록 해보자.

4. **다양한 입력 테스트**: 격식적인 이메일, 사과 이메일, 영업 이메일 등 다양한 유형의 입력으로 테스트해보자.

---

## 6.2 딥 리서치 (Deep Research)

### 주제 및 목표

이 섹션에서는 **SelectorGroupChat**을 사용하여 OpenAI의 "Deep Research" 기능을 클론한 지능형 연구 시스템을 구축한다. 이전 섹션의 고정된 순서(RoundRobin) 방식과 달리, AI가 대화 맥락을 분석하여 **다음에 어떤 에이전트가 동작해야 하는지 자동으로 결정**하는 동적 오케스트레이션을 구현한다.

### 핵심 개념 설명

#### SelectorGroupChat vs RoundRobinGroupChat

두 팀 패턴의 핵심 차이를 비교하면:

| 특성 | RoundRobinGroupChat | SelectorGroupChat |
|------|-------------------|-------------------|
| 에이전트 선택 방식 | 고정 순서 (순차적) | AI가 동적으로 선택 |
| 유연성 | 낮음 | 높음 |
| 예측 가능성 | 높음 | 상대적으로 낮음 |
| 적합한 상황 | 파이프라인, 검토 체인 | 복잡한 워크플로우, 분기가 있는 작업 |
| 추가 비용 | 없음 | 선택을 위한 LLM 호출 추가 발생 |

#### SelectorGroupChat의 동작 원리

```
사용자 질문 입력
       ↓
  ┌─────────────────┐
  │  Selector LLM   │ ← selector_prompt + 대화 이력 참조
  │  (에이전트 선택)  │
  └────────┬────────┘
           ↓
  ┌────────┴────────────────────────────────────────────┐
  │                                                      │
  ▼              ▼              ▼            ▼           ▼
research    research     research     research     quality
_planner    _agent       _enhancer    _analyst     _reviewer
  │              │              │            │           │
  └──────────────┴──────────────┴────────────┴───────────┘
                          ↓
                  다시 Selector LLM이
                  다음 에이전트 선택
                          ↓
                      (반복...)
```

`SelectorGroupChat`은 매 턴마다 **별도의 LLM 호출**을 통해 다음 에이전트를 선택한다. 이때 `selector_prompt`에 정의된 워크플로우 규칙과 현재 대화 이력을 참조한다.

#### UserProxyAgent

`UserProxyAgent`는 **사람(Human)**을 팀의 구성원으로 포함시키는 에이전트이다. 에이전트 팀이 사람의 피드백을 요청하거나 승인을 받아야 할 때 사용한다. `input_func=input`을 통해 표준 입력으로 사람의 응답을 받는다.

이것은 **Human-in-the-Loop (HITL)** 패턴의 구현이다. 완전 자동화와 완전 수동 사이의 균형점으로, AI가 대부분의 작업을 수행하되 핵심 의사결정은 사람이 내리도록 한다.

### 코드 분석

#### 1단계: 임포트 및 모델 설정

```python
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from tools import web_search_tool, save_report_to_md
```

6.1과 비교했을 때의 주요 변경점:
- `RoundRobinGroupChat` 대신 `SelectorGroupChat` 사용
- `UserProxyAgent` 추가 (Human-in-the-Loop)
- 커스텀 도구(`web_search_tool`, `save_report_to_md`) 임포트

#### 2단계: 6개의 전문 에이전트 정의

이 프로젝트에서는 연구 프로세스의 각 단계를 담당하는 6개의 에이전트를 정의한다.

**research_planner (연구 기획자)**

```python
research_planner = AssistantAgent(
    "research_planner",
    description="A strategic research coordinator that breaks down complex questions into research subtasks",
    model_client=model_client,
    system_message="""You are a research planning specialist. Your job is to create a focused research plan.

For each research question, create a FOCUSED research plan with:

1. **Core Topics**: 2-3 main areas to investigate
2. **Search Queries**: Create 3-5 specific search queries covering:
   - Latest developments and news
   - Key statistics or data
   - Expert analysis or studies
   - Future outlook

Keep the plan focused and achievable. Quality over quantity.""",
)
```

> **설계 포인트**: `description` 파라미터에 주목하라. `SelectorGroupChat`에서는 이 `description`이 Selector LLM에게 에이전트의 역할을 설명하는 데 사용된다. RoundRobin 방식에서는 불필요했지만, 동적 선택 방식에서는 **필수적**이다.

**research_agent (웹 리서치 실행자)**

```python
research_agent = AssistantAgent(
    "research_agent",
    description="A web research specialist that searches and extracts content",
    tools=[web_search_tool],
    model_client=model_client,
    system_message="""You are a web research specialist. Your job is to conduct focused searches based on the research plan.

RESEARCH STRATEGY:
1. **Execute 3-5 searches** from the research plan
2. **Extract key information** from the results:
   - Main facts and statistics
   - Recent developments
   - Expert opinions
   - Important context

3. **Quality focus**:
   - Prioritize authoritative sources
   - Look for recent information (within 2 years)
   - Note diverse perspectives

After completing the searches from the plan, summarize what you found. Your goal is to gather 5-10 quality sources.""",
)
```

> **핵심**: `tools=[web_search_tool]`으로 웹 검색 도구가 연결되어 있다. 이 에이전트만이 실제로 외부 웹에 접근할 수 있다. 도구를 특정 에이전트에만 할당함으로써, 불필요한 검색 호출을 방지하고 역할을 명확히 한다.

**research_analyst (연구 분석가)**

```python
research_analyst = AssistantAgent(
    "research_analyst",
    description="An expert analyst that creates research reports",
    model_client=model_client,
    system_message="""You are a research analyst. Create a comprehensive report from the gathered research.

CREATE A RESEARCH REPORT with:

## Executive Summary
- Key findings and conclusions
- Main insights

## Background & Current State
- Current landscape
- Recent developments
- Key statistics and data

## Analysis & Insights
- Main trends
- Different perspectives
- Expert opinions

## Future Outlook
- Emerging trends
- Predictions
- Implications

## Sources
- List all sources used

Write a clear, well-structured report based on the research gathered. End with "REPORT_COMPLETE" when finished.""",
)
```

> **설계 포인트**: 시스템 메시지에 보고서의 **정확한 구조**(섹션 헤더)를 제시한다. 이렇게 구조화된 출력 형식을 지정하면 일관성 있는 결과를 얻을 수 있다. "REPORT_COMPLETE"라는 시그널 단어로 작업 완료를 표시하는 것은 다른 에이전트(quality_reviewer)와의 **프로토콜**로 작동한다.

**quality_reviewer (품질 검토자)**

```python
quality_reviewer = AssistantAgent(
    "quality_reviewer",
    description="A quality assurance specialist that evaluates research completeness and accuracy",
    tools=[save_report_to_md],
    model_client=model_client,
    system_message="""You are a quality reviewer. Your job is to check if the research analyst has produced a complete research report.

Look for:
- A comprehensive research report from the research analyst that ends with "REPORT_COMPLETE"
- The research question is fully answered
- Sources are cited and reliable
- The report includes summary, key information, analysis, and sources

When you see a complete research report that ends with "REPORT_COMPLETE":
1. First, use the save_report_to_md tool to save the report to report.md
2. Then say: "The research is complete. The report has been saved to report.md. Please review the report and let me know if you approve it or need additional research."

If the research analyst has NOT yet created a complete report, tell them to create one now.""",
)
```

> **핵심**: 이 에이전트에는 `save_report_to_md` 도구가 연결되어 있다. 파일 저장이라는 **부작용(side effect)**을 가진 도구를 품질 검토 에이전트에게만 부여함으로써, 보고서가 품질 검토를 통과한 후에만 저장되도록 워크플로우를 제어한다.

**research_enhancer (연구 보완 전문가)**

```python
research_enhancer = AssistantAgent(
    "research_enhancer",
    description="A specialist that identifies critical gaps only",
    model_client=model_client,
    system_message="""You are a research enhancement specialist. Your job is to identify ONLY CRITICAL gaps.

Review the research and ONLY suggest additional searches if there are MAJOR gaps like:
- Completely missing recent developments (last 6 months)
- No statistics or data at all
- Missing a crucial perspective that was specifically asked for

If the research covers the basics reasonably well, say: "The research is sufficient to proceed with the report."

Only suggest 1-2 additional searches if absolutely necessary. We prioritize getting a good report done rather than perfect coverage.""",
)
```

> **설계 포인트**: "ONLY CRITICAL gaps"와 "We prioritize getting a good report done rather than perfect coverage"라는 지시는 **과도한 연구 루프를 방지**하기 위한 것이다. 완벽주의적 에이전트가 끊임없이 추가 검색을 요구하는 것은 멀티 에이전트 시스템에서 흔한 문제이다.

**user_proxy (사용자 대리)**

```python
user_proxy = UserProxyAgent(
    "user_proxy",
    description="Human reviewer who can request additional research or approve final results",
    input_func=input,
)
```

> 최종 보고서에 대한 사람의 승인을 받는 역할이다. 사용자가 "APPROVED"를 입력하면 전체 워크플로우가 종료된다.

#### 3단계: Selector Prompt (에이전트 선택 프롬프트)

이 프로젝트에서 가장 중요한 부분이다. Selector Prompt는 SelectorGroupChat이 다음에 어떤 에이전트를 선택할지 결정하는 데 사용되는 지시문이다.

```python
selector_prompt = """
Choose the best agent for the current task based on the conversation history:

{roles}

Current conversation:
{history}

Available agents:
- research_planner: Plan the research approach (ONLY at the start)
- research_agent: Search for and extract content from web sources (after planning)
- research_enhancer: Identify CRITICAL gaps only (use sparingly)
- research_analyst: Write the final research report
- quality_reviewer: Check if a complete report exists
- user_proxy: Ask the human for feedback

WORKFLOW:
1. If no planning done yet → select research_planner
2. If planning done but no research → select research_agent
3. After research_agent completes initial searches → select research_enhancer ONCE
4. If enhancer says "sufficient to proceed" → select research_analyst
5. If enhancer suggests critical searches → select research_agent ONCE more then research_analyst
6. If research_analyst said "REPORT_COMPLETE" → select quality_reviewer
7. If quality_reviewer asked for user feedback → select user_proxy

IMPORTANT: After research_agent has searched 2 times maximum, proceed to research_analyst regardless.

Pick the agent that should work next based on this workflow."""
```

**상세 분석:**

1. **`{roles}`와 `{history}` 템플릿 변수**: AutoGen이 자동으로 에이전트 설명과 대화 이력을 주입한다. 이를 통해 Selector LLM은 현재 상태를 파악할 수 있다.

2. **명시적 워크플로우 규칙**: 7단계의 워크플로우가 조건문 형태로 명확히 정의되어 있다. 이것은 **상태 머신(State Machine)**과 유사한 패턴이다:
   ```
   시작 → [계획] → [검색] → [보완 검토] → [보고서 작성] → [품질 검토] → [사용자 승인] → 종료
                      ↑           |
                      └───────────┘ (보완 필요 시)
   ```

3. **안전장치**: "After research_agent has searched 2 times maximum, proceed to research_analyst regardless." -- 이 규칙은 무한 검색 루프를 방지한다.

4. **"use sparingly"와 같은 수식어**: Selector LLM에게 특정 에이전트의 사용 빈도에 대한 힌트를 제공한다.

#### 4단계: 팀 생성 및 실행

```python
text_termination = TextMentionTermination("APPROVED")
max_message_termination = MaxMessageTermination(max_messages=50)
termination_condition = text_termination | max_message_termination

team = SelectorGroupChat(
    participants=[
        research_agent,
        research_analyst,
        research_enhancer,
        research_planner,
        quality_reviewer,
        user_proxy,
    ],
    selector_prompt=selector_prompt,
    model_client=model_client,
    termination_condition=termination_condition,
)
```

**핵심 비교 (6.1 vs 6.2):**

| 요소 | 6.1 Email Optimizer | 6.2 Deep Research |
|------|-------------------|-------------------|
| 팀 클래스 | `RoundRobinGroupChat` | `SelectorGroupChat` |
| 종료 키워드 | "TERMINATE" | "APPROVED" |
| 최대 메시지 | 30 | 50 |
| `selector_prompt` | 없음 | 상세한 워크플로우 정의 |
| `model_client` (팀 레벨) | 없음 | 에이전트 선택용 LLM 필요 |
| `participants` 순서 | 실행 순서 결정 | 순서 무관 |

> **주목**: `SelectorGroupChat`에서는 `participants` 리스트의 순서가 실행 순서에 영향을 주지 않는다. 대신 `selector_prompt`에 정의된 규칙이 순서를 결정한다.

#### 5단계: 실행

```python
await Console(
    team.run_stream(task="Research about the new development in Nuclear Energy"),
)
```

이 한 줄로 전체 연구 파이프라인이 시작된다. 실행 흐름은 다음과 같다:

1. Selector LLM이 `research_planner`를 선택
2. research_planner가 핵심 주제와 검색 쿼리를 생성
3. Selector LLM이 `research_agent`를 선택
4. research_agent가 웹 검색 도구로 실제 검색 수행
5. Selector LLM이 `research_enhancer`를 선택
6. research_enhancer가 연구 충분성을 평가
7. Selector LLM이 `research_analyst`를 선택
8. research_analyst가 종합 보고서를 작성하고 "REPORT_COMPLETE" 출력
9. Selector LLM이 `quality_reviewer`를 선택
10. quality_reviewer가 보고서를 `report.md`에 저장
11. Selector LLM이 `user_proxy`를 선택
12. 사용자가 "APPROVED" 입력 시 종료

#### tools.py의 변경사항 (6.1 → 6.2)

```python
# 변경 1: 검색 결과 수 축소 (5 → 2)
response = app.search(
    query=query,
    limit=2,  # 이전: limit=5
    scrape_options=ScrapeOptions(
        formats=["markdown"],
    ),
)

# 변경 2: 새로운 도구 함수 추가
def save_report_to_md(content: str) -> str:
    """Save report content to report.md file."""
    with open("report.md", "w") as f:
        f.write(content)
    return "report.md"
```

**변경 이유 분석:**

1. **`limit=5` → `limit=2`**: 검색 결과를 줄인 것은 **토큰 절약**과 **비용 최적화**를 위한 것이다. 딥 리서치에서는 여러 번의 검색이 이루어지므로, 각 검색에서 너무 많은 결과를 가져오면 컨텍스트 윈도우가 빠르게 소진된다.

2. **`save_report_to_md` 추가**: 연구 결과를 파일로 저장하는 도구이다. `quality_reviewer` 에이전트에 연결되어, 품질 검토를 통과한 보고서만 저장되도록 한다.

#### 생성된 보고서 예시 (report.md)

실행 결과로 생성된 보고서의 구조:

```markdown
# Comprehensive Report on New Developments in Nuclear Energy

## Executive Summary
The nuclear energy sector is witnessing a renaissance as nations prioritize
decarbonization and seek reliable energy sources...

## Background & Current State
In 2023, global electricity production from nuclear energy increased by 2.6%...

## Analysis & Insights
1. **Small Modular Reactors (SMRs)**: These offer cheaper and quicker deployment...
2. **Nuclear Fusion**: Significant progress is being made in fusion research...
3. **Policy Evolution**: Governments are increasingly recognizing nuclear energy's potential...

## Future Outlook
Emerging trends in nuclear energy indicate a potential shift towards more
integrated energy systems...

## Sources
1. International Atomic Energy Agency (IAEA) Report on Nuclear Power for 2023.
2. U.S. Department of Energy Blog: "10 Big Wins for Nuclear Energy in 2023."
...
```

이 보고서는 에이전트가 웹에서 실제로 검색한 데이터를 기반으로 자동 생성된 것이다. 구조화된 시스템 메시지 덕분에 일관된 형식의 전문적인 보고서가 만들어졌다.

### 실습 포인트

1. **다른 연구 주제 테스트**: "Research about the impact of AI on healthcare" 등 다양한 주제로 실행해보자. 에이전트들이 주제에 맞게 검색 쿼리를 조정하는 것을 관찰하라.

2. **Selector Prompt 수정**: 워크플로우 규칙을 변경하여 동작이 어떻게 달라지는지 확인해보자. 예를 들어, research_enhancer를 제거하고 research_agent에서 바로 research_analyst로 넘어가는 규칙을 만들어보자.

3. **에이전트 추가**: "fact_checker" 에이전트를 추가하여 보고서의 사실 관계를 검증하는 단계를 삽입해보자.

4. **도구 확장**: `web_search_tool` 외에 추가 도구(예: 학술 논문 검색, 뉴스 전용 검색)를 만들어 research_agent에 제공해보자.

5. **Human-in-the-Loop 변형**: `user_proxy`의 개입 시점을 변경해보자. 예를 들어, 연구 계획 단계에서도 사용자 승인을 받도록 수정해보자.

---

## 챕터 핵심 정리

### 1. 두 가지 팀 오케스트레이션 패턴

| 패턴 | 클래스 | 선택 방식 | 사용 시점 |
|------|--------|-----------|-----------|
| **순차적 파이프라인** | `RoundRobinGroupChat` | 고정 순서 | 역할이 명확하고 순서가 정해진 작업 |
| **동적 선택** | `SelectorGroupChat` | AI 기반 선택 | 복잡하고 분기가 있는 워크플로우 |

### 2. 에이전트 설계 원칙

- **단일 책임**: 각 에이전트는 하나의 명확한 역할만 수행해야 한다
- **범위 제한**: system_message에서 "하지 말아야 할 것"도 명시하라 (예: "Don't worry about persuasion or tone")
- **description 활용**: SelectorGroupChat에서는 description이 에이전트 선택에 직접 사용된다
- **시그널 단어 프로토콜**: "TERMINATE", "REPORT_COMPLETE", "APPROVED" 같은 키워드로 에이전트 간 통신 프로토콜을 구축하라

### 3. 종료 조건 설계

- 항상 **의미적 종료**(TextMentionTermination)와 **안전장치**(MaxMessageTermination)를 **조합**하라
- `|` (OR) 연산자로 결합하여 유연한 종료 조건을 만들어라
- 안전장치 없이 의미적 종료만 사용하면 무한 루프 위험이 있다

### 4. 도구(Tool) 할당 전략

- 도구는 **필요한 에이전트에게만** 선택적으로 할당하라
- 파일 저장 같은 부작용이 있는 도구는 **검증 에이전트**에게 부여하여 품질 게이트를 구현하라
- 웹 검색 도구는 토큰 비용을 고려하여 결과 수를 적절히 제한하라

### 5. Selector Prompt 설계 요령

- **상태 머신** 관점으로 워크플로우를 정의하라
- 각 상태(조건)에서 어떤 에이전트를 선택해야 하는지 **명시적 규칙**으로 서술하라
- **안전장치 규칙**을 포함하라 (예: "2 times maximum, proceed regardless")
- 에이전트 사용 빈도에 대한 힌트를 제공하라 (예: "use sparingly", "ONLY at the start")

---

## 실습 과제

### 과제 1: 코드 리뷰 팀 만들기 (기초)

**목표**: `RoundRobinGroupChat`을 사용하여 코드 리뷰 팀을 구축하시오.

**요구사항**:
- SecurityAgent: 보안 취약점 검토
- PerformanceAgent: 성능 문제 분석
- ReadabilityAgent: 코드 가독성 평가
- SummaryAgent: 모든 리뷰를 종합하여 최종 피드백 작성
- ApprovalAgent: 최종 승인 또는 수정 요청 (종료 제어)

**힌트**: 이메일 최적화 팀의 구조를 참고하되, system_message를 코드 리뷰에 맞게 수정하라.

### 과제 2: Selector Prompt 최적화 (중급)

**목표**: 6.2의 딥 리서치 시스템에서 `selector_prompt`를 수정하여 다음 기능을 추가하시오.

**요구사항**:
- `fact_checker` 에이전트를 추가하고, research_analyst가 보고서를 작성한 후 quality_reviewer 전에 fact_checker가 동작하도록 워크플로우를 수정하라
- fact_checker는 보고서의 핵심 주장을 웹 검색으로 교차 검증한다
- selector_prompt의 워크플로우 규칙에 새로운 단계를 추가하라

### 과제 3: 나만의 Deep Research 시스템 확장 (고급)

**목표**: 6.2의 딥 리서치 시스템을 확장하여 다음 기능을 구현하시오.

**요구사항**:
1. 검색 도구를 2개로 분리: `academic_search_tool` (학술 논문 전용)과 `news_search_tool` (뉴스 전용)
2. 각 도구를 별도의 에이전트에 할당
3. 보고서 형식을 사용자가 선택할 수 있도록 `user_proxy`의 초기 입력에서 형식을 지정
4. 최종 보고서를 마크다운과 PDF 두 가지 형식으로 저장

**힌트**:
- Firecrawl의 `search()` 메서드에 도메인 필터를 추가하여 학술/뉴스를 분리할 수 있다
- `save_report_to_md`를 확장하여 `save_report_to_pdf` 도구를 만들어보라

### 과제 4: 에이전트 협업 디버깅 (분석)

**목표**: 다음 상황을 분석하고 해결 방안을 제시하시오.

**시나리오**: 딥 리서치 시스템에서 research_agent가 검색을 5번 이상 반복하며 research_analyst로 넘어가지 않는 문제가 발생했다.

**질문**:
1. 이 문제의 가능한 원인 3가지를 제시하라
2. selector_prompt를 어떻게 수정하면 이 문제를 해결할 수 있는가?
3. 코드 레벨에서 추가할 수 있는 안전장치는 무엇인가?

---

## 참고 자료

- [AutoGen 공식 문서](https://microsoft.github.io/autogen/)
- [Firecrawl API 문서](https://docs.firecrawl.dev/)
- 프로젝트 소스 코드: `deep-research-clone/` 디렉토리
  - `email-optimizer-team.ipynb`: 이메일 최적화 팀 노트북
  - `deep-research-team.ipynb`: 딥 리서치 팀 노트북
  - `tools.py`: 웹 검색 및 파일 저장 도구
  - `report.md`: 생성된 연구 보고서 예시
  - `pyproject.toml`: 프로젝트 의존성 정의
