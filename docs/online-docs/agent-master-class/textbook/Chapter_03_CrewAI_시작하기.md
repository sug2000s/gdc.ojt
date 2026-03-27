# Chapter 3: CrewAI 시작하기

---

## 챕터 개요

이번 챕터에서는 **CrewAI** 프레임워크를 사용하여 AI 에이전트를 구축하는 방법을 처음부터 단계별로 학습한다. CrewAI는 여러 AI 에이전트가 협력하여 복잡한 작업을 수행할 수 있도록 설계된 Python 프레임워크로, "Crew(팀)"라는 개념을 중심으로 에이전트와 태스크를 조직화한다.

이 챕터는 4개의 섹션으로 구성되며, 간단한 번역 에이전트에서 출발하여 점차 실전 수준의 뉴스 리더 에이전트 시스템으로 발전시켜 나간다.

| 섹션 | 주제 | 핵심 학습 내용 |
|------|------|----------------|
| 3.1 | Your First CrewAI Agent | 프로젝트 구조, Agent/Task/Crew 기본 개념, YAML 설정 |
| 3.2 | Custom Tools | 커스텀 도구 제작, `@tool` 데코레이터, 에이전트에 도구 연결 |
| 3.3 | News Reader Tasks and Agents | 실전 에이전트 설계, 상세 프롬프트 작성, 다중 에이전트 구성 |
| 3.4 | News Reader Crew | 실제 도구(검색/스크래핑) 통합, LLM 모델 지정, Crew 실행 및 결과 확인 |

### 학습 목표

이 챕터를 완료하면 다음을 할 수 있다:

1. CrewAI 프로젝트를 처음부터 설정하고 구조화할 수 있다
2. YAML 설정 파일을 사용하여 에이전트와 태스크를 정의할 수 있다
3. 커스텀 도구를 만들어 에이전트에 연결할 수 있다
4. 여러 에이전트가 순차적으로 협업하는 Crew를 구성하고 실행할 수 있다
5. 실전 수준의 뉴스 수집/요약/편집 파이프라인을 구축할 수 있다

### 사전 준비 사항

- Python 3.13 이상
- `uv` 패키지 매니저 (Python 프로젝트 관리용)
- OpenAI API 키 (`.env` 파일에 설정)
- 기본적인 Python 문법 이해

---

## 3.1 Your First CrewAI Agent

### 주제 및 목표

첫 번째 섹션에서는 CrewAI의 기본 구조를 이해하고, 간단한 **번역 에이전트(Translator Agent)**를 만들어본다. 이 과정에서 CrewAI의 핵심 3요소인 **Agent**, **Task**, **Crew**의 관계를 파악하고, YAML 기반 설정 파일의 역할을 학습한다.

### 핵심 개념 설명

#### CrewAI의 3대 핵심 개념

CrewAI는 세 가지 핵심 구성요소로 이루어져 있다:

1. **Agent (에이전트)**: 특정 역할(role), 목표(goal), 배경(backstory)을 가진 AI 작업자. 사람으로 비유하면 팀의 한 구성원이다.
2. **Task (태스크)**: 에이전트가 수행해야 하는 구체적인 작업. 작업 설명(description)과 기대 결과(expected_output)를 포함한다.
3. **Crew (크루)**: 에이전트와 태스크를 묶어서 실행하는 팀 단위. 에이전트들이 태스크를 순서대로 처리한다.

```
┌─────────────────────────────────────────┐
│                  Crew                   │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Agent   │───>│     Task 1       │   │
│  │(번역가)  │    │(영→이탈리아어)   │   │
│  └──────────┘    └──────────────────┘   │
│       │                   │             │
│       │          (결과가 다음 태스크로)  │
│       │                   ▼             │
│       │          ┌──────────────────┐   │
│       └─────────>│     Task 2       │   │
│                  │(이탈리아어→그리스어)│  │
│                  └──────────────────┘   │
└─────────────────────────────────────────┘
```

#### `@CrewBase` 데코레이터와 프로젝트 구조

CrewAI는 **데코레이터 기반 클래스 패턴**을 사용한다. `@CrewBase` 데코레이터를 클래스에 적용하면, CrewAI가 자동으로 `config/agents.yaml`과 `config/tasks.yaml` 파일을 읽어들여 `self.agents_config`와 `self.tasks_config` 딕셔너리로 제공한다.

#### 프로젝트 디렉토리 구조

```
news-reader-agent/
├── .python-version          # Python 버전 지정 (3.13)
├── .gitignore               # Git 제외 파일 목록
├── pyproject.toml           # 프로젝트 설정 및 의존성
├── config/
│   ├── agents.yaml          # 에이전트 정의
│   └── tasks.yaml           # 태스크 정의
├── main.py                  # 메인 실행 파일
└── uv.lock                  # 의존성 잠금 파일
```

이 구조는 CrewAI의 관례(convention)를 따른다. `config/` 디렉토리에 YAML 파일을 두면 `@CrewBase`가 자동으로 인식한다.

### 코드 분석

#### 프로젝트 의존성 (`pyproject.toml`)

```toml
[project]
name = "news-reader-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "python-dotenv>=1.1.1",
]
```

- `crewai[tools]`: CrewAI 프레임워크와 도구 확장 패키지를 함께 설치한다. `[tools]`는 extras로, 추가 도구 관련 의존성을 포함한다.
- `python-dotenv`: `.env` 파일에서 환경 변수를 로드하기 위한 라이브러리. API 키를 안전하게 관리하는 데 사용한다.

#### 에이전트 설정 (`config/agents.yaml`)

```yaml
translator_agent:
  role: >
    Translator to translate from English to Italian
  goal: >
    To be a good and useful translator to avoid misunderstandings.
  backstory: >
    You grew up between New York and Palermo, you can speak two languages
    fluently, and you can detect the cultural differences.
```

**각 필드의 역할:**

| 필드 | 설명 | 예시에서의 역할 |
|------|------|-----------------|
| `role` | 에이전트의 직무/역할 정의 | 영어-이탈리아어 번역가 |
| `goal` | 에이전트가 달성해야 할 목표 | 오해 없는 정확한 번역 |
| `backstory` | 에이전트의 배경 설정 (성격과 전문성 부여) | 뉴욕과 팔레르모에서 자란 이중 언어 사용자 |

> **왜 `backstory`가 중요한가?**
> `backstory`는 단순한 장식이 아니다. LLM이 응답을 생성할 때 이 배경 정보를 컨텍스트로 사용하여, 더 일관되고 전문적인 결과를 만들어낸다. 예를 들어 "문화적 차이를 감지할 수 있다"라는 정보는 번역 시 문화적 뉘앙스를 반영하도록 유도한다.

#### 태스크 설정 (`config/tasks.yaml`)

```yaml
translate_task:
  description: >
    Translate {sentence} from English to Italian without making mistakes.
  expected_output: >
    A well formatted translation from English to Italian using proper
    capitalization of names and places.
  agent: translator_agent

retranslate_task:
  description: >
    Translate {sentence} from Italian to Greek without making mistakes.
  expected_output: >
    A well formatted translation from Italian to Greek using proper
    capitalization of names and places.
  agent: translator_agent
```

**핵심 포인트:**

- `{sentence}`: 중괄호로 감싼 **변수 플레이스홀더**. 실행 시 `kickoff(inputs={"sentence": "..."})`로 전달된 값으로 치환된다.
- `expected_output`: 에이전트에게 결과물의 형태를 명확히 알려준다. 이것이 있어야 에이전트가 무엇을 반환해야 하는지 정확히 이해한다.
- `agent`: 이 태스크를 수행할 에이전트의 이름. `agents.yaml`에 정의된 키와 일치해야 한다.
- 두 태스크 모두 같은 `translator_agent`를 사용한다. 하나의 에이전트가 여러 태스크를 수행할 수 있다.

#### 메인 실행 파일 (`main.py`)

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, agent, task, crew


@CrewBase
class TranslatorCrew:

    @agent
    def translator_agent(self):
        return Agent(
            config=self.agents_config["translator_agent"],
        )

    @task
    def translate_task(self):
        return Task(
            config=self.tasks_config["translate_task"],
        )

    @task
    def retranslate_task(self):
        return Task(
            config=self.tasks_config["retranslate_task"],
        )

    @crew
    def assemble_crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )


TranslatorCrew().assemble_crew().kickoff(
    inputs={
        "sentence": "I'm Nico and I like to ride my bicicle in Napoli",
    }
)
```

**코드 동작 흐름 상세 분석:**

1. **`dotenv.load_dotenv()`**: `.env` 파일에서 `OPENAI_API_KEY` 등의 환경 변수를 로드한다. CrewAI는 내부적으로 이 키를 사용해 LLM API를 호출한다.

2. **`@CrewBase` 데코레이터**: 클래스에 적용되면 `config/agents.yaml`과 `config/tasks.yaml`을 자동으로 읽어들인다. 이를 통해 `self.agents_config`와 `self.tasks_config`에 접근할 수 있다.

3. **`@agent` 데코레이터**: 메서드가 Agent 객체를 반환함을 CrewAI에 알린다. 메서드 이름(`translator_agent`)은 YAML의 키와 일치해야 한다.

4. **`@task` 데코레이터**: 메서드가 Task 객체를 반환함을 CrewAI에 알린다. 태스크는 **정의된 순서대로** 실행된다. `translate_task` → `retranslate_task` 순서로 실행된다.

5. **`@crew` 데코레이터**: Crew 객체를 조합하는 메서드에 적용한다.
   - `self.agents`: `@agent`로 표시된 모든 에이전트의 리스트 (자동 수집)
   - `self.tasks`: `@task`로 표시된 모든 태스크의 리스트 (자동 수집)
   - `verbose=True`: 실행 과정을 콘솔에 상세히 출력한다.

6. **`kickoff(inputs={...})`**: Crew를 실행한다. `inputs` 딕셔너리의 값이 YAML의 `{sentence}` 플레이스홀더를 대체한다.

> **실행 결과 흐름:**
> 1. `translate_task`: "I'm Nico and I like to ride my bicicle in Napoli" → 이탈리아어로 번역
> 2. `retranslate_task`: 이탈리아어 번역 결과 → 그리스어로 번역
>
> 두 번째 태스크는 첫 번째 태스크의 결과를 **자동으로 컨텍스트로 받는다**. 이것이 CrewAI에서 태스크 체이닝이 작동하는 방식이다.

### 실습 포인트

- `.env` 파일에 `OPENAI_API_KEY=sk-...`를 설정한 후, `uv run python main.py`로 실행해보자.
- `verbose=True`를 통해 에이전트의 사고 과정(Chain of Thought)을 관찰하자.
- `backstory`를 변경해보며 결과물의 품질이 어떻게 달라지는지 비교해보자.
- 세 번째 태스크(예: 그리스어→한국어 번역)를 추가해보자.

---

## 3.2 Custom Tools

### 주제 및 목표

이 섹션에서는 CrewAI 에이전트에 **커스텀 도구(Custom Tool)**를 연결하는 방법을 학습한다. LLM은 기본적으로 텍스트 생성만 할 수 있지만, 도구를 통해 외부 기능(계산, API 호출, 파일 읽기 등)을 수행할 수 있게 된다.

### 핵심 개념 설명

#### Tool (도구)란?

AI 에이전트의 **도구(Tool)**는 에이전트가 LLM의 텍스트 생성 능력 외에 추가로 사용할 수 있는 함수이다. 에이전트는 태스크를 수행하면서 "이 작업에는 도구가 필요하다"고 판단하면, 자동으로 적절한 도구를 호출한다.

```
┌──────────────────────────────────────┐
│           에이전트 (Agent)            │
│                                      │
│  "문장의 글자 수를 세야 하는데...    │
│   count_letters 도구를 사용하자!"    │
│                                      │
│   ┌─────────────────────────────┐    │
│   │  Tool: count_letters        │    │
│   │  Input: "Hello World"       │    │
│   │  Output: 11                 │    │
│   └─────────────────────────────┘    │
└──────────────────────────────────────┘
```

도구가 중요한 이유는 LLM이 수학적 계산이나 정확한 데이터 조회 등에서 **환각(hallucination)**을 일으킬 수 있기 때문이다. 글자 수를 세는 단순한 작업도 LLM이 직접 하면 틀릴 수 있지만, `len()` 함수를 도구로 제공하면 정확한 결과를 보장한다.

#### `@tool` 데코레이터

CrewAI는 `@tool` 데코레이터를 제공하여 일반 Python 함수를 에이전트가 사용할 수 있는 도구로 변환한다. 함수의 **docstring**이 에이전트에게 도구의 용도를 설명하는 역할을 한다.

### 코드 분석

#### 커스텀 도구 정의 (`tools.py`)

```python
from crewai.tools import tool


@tool
def count_letters(sentence: str):
    """
    This function is to count the amount of letters in a sentence.
    The input is a `sentence` string.
    The output is a number.
    """
    print("tool called with input:", sentence)
    return len(sentence)
```

**핵심 분석:**

- **`@tool` 데코레이터**: 이 데코레이터가 함수를 CrewAI 도구로 변환한다. 내부적으로 함수의 시그니처와 docstring을 분석하여 LLM이 이해할 수 있는 도구 스키마를 생성한다.
- **타입 힌트 `sentence: str`**: 필수적이다. CrewAI는 이 타입 힌트를 사용하여 LLM에게 입력 파라미터의 형식을 알려준다.
- **docstring**: 에이전트가 "이 도구를 언제 사용해야 하는가?"를 판단하는 데 사용된다. 명확하고 상세하게 작성해야 한다. 입력과 출력의 형태를 설명하는 것이 좋다.
- **`print()` 문**: 디버깅용. 도구가 실제로 호출되는지, 어떤 입력을 받는지 확인할 수 있다.
- **`return len(sentence)`**: 실제 로직. LLM이 직접 글자를 세는 대신 Python의 `len()` 함수로 정확한 결과를 반환한다.

#### 새로운 에이전트와 태스크 추가 (`config/agents.yaml`)

```yaml
counter_agent:
  role: >
    To count the lenght of things.
  goal: >
    To be a good counter that never lies or makes things up.
  backstory: >
    You are a genius counter.
```

`goal`에서 "never lies or makes things up"이라는 표현에 주목하자. 이것은 에이전트가 추측하지 않고 반드시 도구를 사용하도록 유도하는 프롬프트 기법이다.

#### 태스크 추가 (`config/tasks.yaml`)

```yaml
count_task:
  description: >
    Count the amount of letters in a sentence.
  expected_output: >
    The number of letters in a sentence.
  agent: counter_agent
```

#### main.py에서 도구 연결

```python
from tools import count_letters

# ... 클래스 내부 ...

@agent
def counter_agent(self):
    return Agent(
        config=self.agents_config["counter_agent"],
        tools=[count_letters],  # 도구를 에이전트에 연결
    )

@task
def count_task(self):
    return Task(
        config=self.tasks_config["count_task"],
    )
```

**핵심 포인트:**

- `tools=[count_letters]`: Agent 생성 시 `tools` 파라미터에 도구 리스트를 전달한다. 하나의 에이전트에 여러 도구를 연결할 수 있다.
- 도구는 **에이전트 수준**에서 연결된다 (태스크가 아닌). 에이전트가 어떤 태스크를 수행하든 자신에게 할당된 도구를 사용할 수 있다.
- `count_task`에는 도구를 직접 명시하지 않는다. 태스크를 담당하는 에이전트(`counter_agent`)가 이미 도구를 가지고 있기 때문이다.

### 실습 포인트

- `print()` 출력을 관찰하여 도구가 실제로 호출되는 시점을 확인하자.
- docstring을 모호하게 변경해보고, 에이전트가 도구를 올바르게 호출하는지 테스트하자.
- 새로운 도구(예: 단어 수 세기, 대문자 변환)를 만들어 에이전트에 추가해보자.
- 하나의 에이전트에 여러 도구를 연결하고, 에이전트가 상황에 맞게 적절한 도구를 선택하는지 관찰하자.

---

## 3.3 News Reader Tasks and Agents

### 주제 및 목표

이 섹션에서는 앞서 만든 간단한 번역/카운터 예제를 **실전 뉴스 리더 시스템**으로 완전히 재구성한다. 3개의 전문화된 에이전트와 3개의 상세한 태스크를 설계하며, 프로덕션 수준의 프롬프트 엔지니어링 기법을 학습한다.

### 핵심 개념 설명

#### 다중 에이전트 아키텍처

뉴스 리더 시스템은 **3단계 파이프라인** 구조를 따른다:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ News Hunter  │────>│  Summarizer  │────>│   Curator    │
│   Agent      │     │    Agent     │     │    Agent     │
│              │     │              │     │              │
│ 뉴스 수집    │     │ 기사 요약    │     │ 최종 보고서  │
│ & 필터링     │     │ (3단계)      │     │ 편집 & 큐레  │
│              │     │              │     │ 이션         │
└──────────────┘     └──────────────┘     └──────────────┘
  Task 1:              Task 2:              Task 3:
  content_harvesting   summarization        final_report_assembly

  output:              output:              output:
  content_harvest.md   summary.md           final_report.md
```

각 에이전트는 **전문 분야가 다르다**. 이것이 다중 에이전트 시스템의 핵심 장점이다. 하나의 에이전트가 모든 것을 하는 것보다, 각자의 전문성에 집중하는 것이 더 좋은 결과를 만든다.

#### 프롬프트 엔지니어링: 상세한 `backstory` 작성

이 섹션에서 가장 중요한 변화는 에이전트 설정의 **깊이와 상세함**이다. 3.1에서의 단순한 2줄짜리 backstory가 10줄 이상의 상세한 프로필로 변화한다.

#### 태스크의 `output_file` 설정

각 태스크가 결과를 **마크다운 파일로 자동 저장**하도록 설정한다. 이를 통해 각 단계의 중간 결과물을 확인하고 디버깅할 수 있다.

### 코드 분석

#### 에이전트 설정 (`config/agents.yaml`)

**1. News Hunter Agent - 뉴스 수집 전문가**

```yaml
news_hunter_agent:
  role: >
    Senior News Intelligence Specialist
  goal: >
    Discover and collect the most relevant, credible, and up-to-date news
    articles from diverse sources across specified topics, ensuring
    comprehensive coverage while filtering out misinformation and
    low-quality content
  backstory: >
    You are a seasoned digital journalist with 15 years of experience in
    news aggregation and fact-checking. You have an exceptional ability to
    identify credible sources, spot trending stories before they break
    mainstream, and navigate the complex landscape of digital media. Your
    network spans traditional media outlets, independent journalists, and
    expert sources across multiple industries. You pride yourself on your
    ability to separate signal from noise in the overwhelming flow of daily
    news, and you have a keen sense for detecting bias and misinformation.
    You understand the importance of source diversity and always
    cross-reference information from multiple outlets before considering
    it reliable.
  verbose: true
  inject_date: true
```

**새로운 설정 옵션:**

| 옵션 | 설명 |
|------|------|
| `verbose: true` | 에이전트의 사고 과정을 상세히 출력 |
| `inject_date: true` | 현재 날짜를 에이전트의 컨텍스트에 자동 주입. 뉴스 시의성 판단에 필수적 |

**`backstory` 분석 - 왜 이렇게 상세하게 작성하는가:**

- "15 years of experience": 전문성 수준을 설정하여 LLM이 고품질 판단을 내리도록 유도
- "separate signal from noise": 필터링 능력을 강조하여 관련 없는 기사를 걸러내도록 유도
- "detecting bias and misinformation": 신뢰성 평가 기능을 활성화
- "source diversity": 다양한 출처에서 정보를 수집하도록 유도

**2. Summarizer Agent - 요약 전문가**

```yaml
summarizer_agent:
  role: >
    Expert News Analyst and Content Synthesizer
  goal: >
    Transform raw news articles into clear, concise, and comprehensive
    summaries that capture essential information, context, and implications
    while maintaining objectivity and highlighting key insights for busy
    readers
  backstory: >
    You are a skilled news analyst with a background in journalism and
    information science. You've worked as an editor for major news
    publications and have a talent for distilling complex stories into
    digestible summaries without losing critical nuance. Your expertise
    spans multiple domains including politics, technology, economics, and
    international affairs. ...
  verbose: true
  inject_date: true
  llm: openai/o3
```

**새로운 설정: `llm: openai/o3`**

특정 에이전트에 **다른 LLM 모델을 지정**할 수 있다. 요약 작업은 높은 수준의 이해력과 표현력이 필요하므로, 더 강력한 모델(o3)을 사용한다. 이렇게 에이전트별로 모델을 다르게 설정하면 비용과 성능을 최적화할 수 있다.

**3. Curator Agent - 편집 전문가**

```yaml
curator_agent:
  role: >
    Senior News Editor and Editorial Curator
  goal: >
    Curate and editorialize summarized news content into a cohesive,
    engaging narrative that provides context, identifies the most important
    stories, and creates a meaningful reading experience that helps users
    understand not just what happened, but why it matters
  backstory: >
    You are a veteran news editor with 20+ years of experience at top-tier
    publications like The New York Times, The Economist, and Reuters. ...
  verbose: true
  inject_date: true
```

#### 태스크 설정 (`config/tasks.yaml`)

**1. Content Harvesting Task - 뉴스 수집 태스크**

이 태스크는 가장 상세한 지시사항을 포함한다. 핵심 부분을 살펴보자:

```yaml
content_harvesting_task:
  description: >
    Collect recent news articles based on {topic}.

    Steps include:
    1. Use the search tool to search for recent news articles about {topic}
    2. From the search results, identify URLs from credible sources.

    3. **IMPORTANT: Only select actual article pages, not topic hubs or
       tag listings**
      You must filter out any URLs that are likely to be:
      - Topic/tag/section index pages (e.g., URLs containing "/tag/",
        "/topic/", "/hub/", "/section/", "/category/")
      - Pages with no unique headline or timestamp
      - Pages that only contain a list of other stories or links
```

**프롬프트 엔지니어링 기법 분석:**

1. **단계별 지시 (Step-by-step)**: 번호를 매겨 순서대로 수행할 작업을 명시한다.
2. **명시적 필터링 규칙**: "IMPORTANT"를 대문자로 강조하고, 허용/거부 URL 패턴을 구체적 예시와 함께 제공한다.
3. **체크리스트 형식**: 허용(`✅`)과 거부(`❌`) 기호로 시각적으로 구분한다.
4. **수치 기준 제공**: "200단어 미만 기사 제거", "48시간 이상 된 기사 제거" 등 구체적 숫자를 제시한다.
5. **점수 체계**: 신뢰도(1-10)와 관련성(1-10) 점수를 매기도록 요구한다.

```yaml
  expected_output: >
    A well-structured markdown document containing the collected news
    articles with this exact format:

    # News Articles Collection: {topic}

    **Collection Summary**
    - Total articles found:
    - Articles after filtering:
    - Duplicates removed:
    ...
  agent: news_hunter_agent
  markdown: true
  output_file: output/content_harvest.md
  create_directory: true
```

**태스크 출력 설정:**

| 옵션 | 설명 |
|------|------|
| `markdown: true` | 출력을 마크다운 형식으로 처리 |
| `output_file: output/content_harvest.md` | 결과를 지정된 파일에 자동 저장 |
| `create_directory: true` | `output/` 디렉토리가 없으면 자동 생성 |

**2. Summarization Task - 요약 태스크**

```yaml
summarization_task:
  description: >
    Take each of the URLs from the previous task and generate a summary
    for each article.

    Use the scrape tool to extract the full article content from the URL.

    For each article found in the file, create:
    1. **Headline Summary** (≤280 characters, tweet-style)
    2. **Executive Summary** (150-200 words, concise briefing)
    3. **Comprehensive Summary** (500-700 words with full context)
```

이 태스크는 **3단계 요약 체계**를 요구한다. 이는 다양한 독자층의 요구를 충족시키는 실전적인 설계 패턴이다:
- 트윗 수준의 속보 → SNS 공유용
- 임원 요약 → 바쁜 전문가용
- 상세 요약 → 심층 이해가 필요한 독자용

**3. Final Report Assembly Task - 최종 보고서 태스크**

```yaml
final_report_assembly_task:
  description: >
    Create the final, publication-ready markdown news briefing by combining
    all previous work into a professional, cohesive report suitable for
    daily publication.

    Assembly process:
    1. **Follow the editorial plan** from the curation task for structure
    2. **Apply appropriate summary levels** for each story
    3. **Include editorial transitions** and section introductions
    4. **Add professional opening** that summarizes the day's key
       developments
    5. **Create closing section** that ties together themes
    6. **Ensure consistent formatting** and professional presentation
    7. **Include proper attribution** and source references
```

이 태스크는 앞선 두 태스크의 결과물을 종합하여 **출판 가능한 수준**의 뉴스 브리핑을 생성한다.

#### 메인 파일의 변화 (`main.py`)

```python
@CrewBase
class NewsReaderAgent:

    @agent
    def news_hunter_agent(self):
        return Agent(
            config=self.agents_config["news_hunter_agent"],
        )

    @agent
    def summarizer_agent(self):
        return Agent(
            config=self.agents_config["summarizer_agent"],
        )

    @agent
    def curator_agent(self):
        return Agent(
            config=self.agents_config["curator_agent"],
        )

    @task
    def content_harvesting_task(self):
        return Task(
            config=self.tasks_config["content_harvesting_task"],
        )

    @task
    def summarization_task(self):
        return Task(
            config=self.tasks_config["summarization_task"],
        )

    @task
    def final_report_assembly_task(self):
        return Task(
            config=self.tasks_config["final_report_assembly_task"],
        )

    @crew
    def crew(self):
        return Crew(
            tasks=self.tasks,
            agents=self.agents,
            verbose=True,
        )


NewsReaderAgent().crew().kickoff()
```

**주요 변화 사항:**

1. 클래스명이 `TranslatorCrew` → `NewsReaderAgent`로 변경
2. Crew 메서드명이 `assemble_crew` → `crew`로 단순화
3. `kickoff()`에 `inputs`가 아직 없다 (다음 섹션에서 추가)
4. 이 시점에서는 에이전트에 도구가 아직 연결되지 않았다 (설계 단계)

### 실습 포인트

- 에이전트의 `backstory`를 더 상세하게 또는 더 간략하게 변경하고 결과물의 품질 차이를 비교하자.
- `expected_output`의 형식을 변경하여 다른 출력 구조를 실험하자.
- `tasks.yaml`에서 `output_file`의 경로를 변경하고, 파일이 올바르게 생성되는지 확인하자.
- 네 번째 에이전트(예: 번역가)를 추가하여 최종 보고서를 한국어로 번역하는 파이프라인을 설계해보자.

---

## 3.4 News Reader Crew

### 주제 및 목표

마지막 섹션에서는 설계된 뉴스 리더 시스템에 **실제 도구를 연결**하여 완전히 작동하는 Crew를 완성한다. 웹 검색 도구와 웹 스크래핑 도구를 구현하고, 각 에이전트에 적절한 LLM 모델을 지정하며, 실제 주제("Cambodia Thailand War")로 시스템을 실행하여 결과물을 확인한다.

### 핵심 개념 설명

#### 실전 도구 (Production Tools)

3.2에서는 `len()` 함수를 감싼 간단한 도구를 만들었지만, 이번 섹션에서는 실제 웹 서비스와 상호작용하는 도구를 구현한다:

1. **검색 도구 (Search Tool)**: Serper API를 사용한 구글 검색
2. **스크래핑 도구 (Scrape Tool)**: Playwright + BeautifulSoup를 사용한 웹 페이지 콘텐츠 추출

#### CrewAI 내장 도구 vs 커스텀 도구

| 구분 | 내장 도구 | 커스텀 도구 |
|------|-----------|-------------|
| 예시 | `SerperDevTool` | `scrape_tool` |
| 장점 | 설정이 간단, 즉시 사용 가능 | 완전한 제어, 특수 요구 대응 |
| 단점 | 커스터마이징 제한적 | 직접 구현 필요 |

#### 에이전트별 LLM 모델 지정

모든 에이전트에 같은 모델을 사용하면 비효율적이다. 작업 복잡도에 따라 모델을 다르게 지정하면 비용을 절약하면서도 필요한 곳에 높은 품질을 보장할 수 있다.

### 코드 분석

#### 도구 구현 (`tools.py`)

**1. 검색 도구 - SerperDevTool**

```python
import time
from crewai.tools import tool
from crewai_tools import SerperDevTool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

search_tool = SerperDevTool(
    n_results=30,
)
```

- `SerperDevTool`: CrewAI가 제공하는 내장 검색 도구. Serper API를 통해 Google 검색 결과를 가져온다.
- `n_results=30`: 검색 결과를 최대 30개까지 가져온다. 뉴스 수집의 포괄성을 위해 넉넉하게 설정한다.
- 사용하려면 `.env` 파일에 `SERPER_API_KEY`를 설정해야 한다.

**2. 스크래핑 도구 - 커스텀 구현**

```python
@tool
def scrape_tool(url: str):
    """
    Use this when you need to read the content of a website.
    Returns the content of a website, in case the website is not
    available, it returns 'No content'.
    Input should be a `url` string. for example
    (https://www.reuters.com/world/asia-pacific/...)
    """

    print(f"Scrapping URL: {url}")

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(url)

        time.sleep(5)

        html = page.content()

        browser.close()

        soup = BeautifulSoup(html, "html.parser")

        unwanted_tags = [
            "header",
            "footer",
            "nav",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "button",
            "input",
            "select",
            "textarea",
            "img",
            "svg",
            "canvas",
            "audio",
            "video",
            "embed",
            "object",
        ]

        for tag in soup.find_all(unwanted_tags):
            tag.decompose()

        content = soup.get_text(separator=" ")

        return content if content != "" else "No content"
```

**코드 동작 흐름 분석:**

1. **Playwright 브라우저 실행**: `sync_playwright()`로 Chromium 브라우저를 헤드리스(화면 없이) 모드로 실행한다. 이는 JavaScript로 렌더링되는 동적 웹페이지도 처리할 수 있게 해준다.

2. **페이지 로드 및 대기**: `page.goto(url)` 후 `time.sleep(5)`로 5초간 대기한다. 이는 JavaScript 렌더링과 동적 콘텐츠 로딩이 완료될 시간을 확보한다.

3. **HTML 파싱**: `BeautifulSoup`로 HTML을 파싱한다.

4. **불필요한 태그 제거**: `unwanted_tags` 리스트에 정의된 태그들을 모두 제거한다. 이를 통해 네비게이션, 광고, 스크립트 등의 노이즈를 걸러내고 순수 기사 텍스트만 추출한다.

5. **텍스트 추출**: `soup.get_text(separator=" ")`로 모든 텍스트를 공백으로 연결하여 추출한다.

6. **안전한 반환**: 콘텐츠가 비어있으면 "No content"를 반환하여 에이전트가 실패를 인지할 수 있게 한다.

> **왜 `requests` 대신 Playwright를 사용하는가?**
> 현대 뉴스 사이트는 대부분 JavaScript로 콘텐츠를 동적 렌더링한다. `requests` 라이브러리는 정적 HTML만 가져오므로 기사 본문이 누락될 수 있다. Playwright는 실제 브라우저를 구동하므로 JavaScript 실행 후의 완전한 DOM을 얻을 수 있다.

#### 에이전트에 LLM 모델 지정 (`config/agents.yaml`)

```yaml
news_hunter_agent:
  # ... (기존 설정)
  llm: openai/o4-mini-2025-04-16

summarizer_agent:
  # ... (기존 설정)
  llm: openai/o4-mini-2025-04-16   # o3에서 o4-mini로 변경

curator_agent:
  # ... (기존 설정)
  llm: openai/o4-mini-2025-04-16
```

3.3에서 summarizer_agent에 `openai/o3`를 지정했던 것이 `openai/o4-mini-2025-04-16`으로 변경되었다. 모든 에이전트가 동일한 모델을 사용하게 되었는데, 이는 비용 효율성과 충분한 성능 사이의 균형을 반영한다. `llm` 필드의 형식은 `provider/model-name`이다.

#### 메인 파일 완성 (`main.py`)

```python
from tools import search_tool, scrape_tool


@CrewBase
class NewsReaderAgent:

    @agent
    def news_hunter_agent(self):
        return Agent(
            config=self.agents_config["news_hunter_agent"],
            tools=[search_tool, scrape_tool],
        )

    @agent
    def summarizer_agent(self):
        return Agent(
            config=self.agents_config["summarizer_agent"],
            tools=[
                scrape_tool,
            ],
        )

    @agent
    def curator_agent(self):
        return Agent(
            config=self.agents_config["curator_agent"],
        )

    # ... 태스크 정의는 동일 ...

    @crew
    def crew(self):
        return Crew(
            tasks=self.tasks,
            agents=self.agents,
            verbose=True,
        )


result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

**에이전트별 도구 할당 전략:**

| 에이전트 | 도구 | 이유 |
|----------|------|------|
| `news_hunter_agent` | `search_tool`, `scrape_tool` | 검색으로 기사를 찾고, 스크래핑으로 내용을 읽어야 한다 |
| `summarizer_agent` | `scrape_tool` | 이전 태스크의 URL로 기사를 다시 읽어 상세히 요약한다 |
| `curator_agent` | (없음) | 이전 태스크의 요약 결과만 편집하므로 외부 도구 불필요 |

**`kickoff()` 실행 및 결과 처리:**

```python
result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

- `inputs={"topic": "Cambodia Thailand War."}`: YAML의 `{topic}` 플레이스홀더를 대체한다.
- `result.tasks_output`: 각 태스크의 실행 결과를 리스트로 반환한다. 3개 태스크이므로 3개의 결과가 포함된다.

#### 실행 결과물

Crew 실행 후 `output/` 디렉토리에 3개의 마크다운 파일이 생성된다:

**1. `output/content_harvest.md` - 수집된 기사 목록**

```markdown
# News Articles Collection: Cambodia Thailand War.
**Collection Summary**
- Total articles found: 4
- Articles after filtering: 3
- Duplicates removed: 0
- Sources accessed: Reuters, AP News, BBC
- Search queries used: "Cambodia Thailand War recent news August 2025"...
- Search timestamp: 2025-08-05

---
## Article 1: Cambodia and Thailand begin talks in Malaysia...
**Source:** Reuters
**Date:** 2025-08-04 06:19 UTC
**URL:** https://www.reuters.com/world/asia-pacific/...
**Category:** International
**Credibility Score:** 9
**Relevance Score:** 10
```

news_hunter_agent가 3개의 신뢰할 수 있는 기사(Reuters, AP News, BBC)를 수집하고, 각각에 신뢰도와 관련성 점수를 매긴 것을 확인할 수 있다.

**2. `output/summary.md` - 3단계 요약**

각 기사에 대해 트윗 수준 요약(280자 이내), 임원 요약(150-200단어), 상세 요약(500-700단어)이 생성된다. `expected_output`에서 지정한 형식을 충실히 따른다.

**3. `output/final_report.md` - 최종 뉴스 브리핑**

모든 정보를 종합한 출판 수준의 최종 보고서. Executive Summary, Lead Story, Breaking News, Editor's Analysis 등의 섹션으로 구성된 전문적인 뉴스 브리핑이다.

### 실습 포인트

- `inputs`의 `topic`을 다른 주제로 변경하여 실행해보자 (예: "AI regulation 2025", "climate change policy").
- `scrape_tool`의 `time.sleep(5)` 값을 조절하며 속도와 안정성의 트레이드오프를 실험하자.
- `unwanted_tags` 리스트를 수정하여 추출 품질을 개선해보자.
- `n_results=30`을 더 작거나 큰 값으로 변경하여 검색 범위의 영향을 관찰하자.
- 새로운 에이전트(예: 팩트체커)를 추가하여 파이프라인을 확장해보자.

---

## 챕터 핵심 정리

### 1. CrewAI의 핵심 아키텍처

- **Agent**: 역할, 목표, 배경을 가진 AI 작업자. YAML로 설정하고 Python에서 인스턴스화한다.
- **Task**: 구체적인 작업 지시. `description`, `expected_output`, `agent` 지정이 핵심이다.
- **Crew**: Agent와 Task를 묶어 실행하는 팀 단위. `kickoff()`으로 실행한다.
- **Tool**: 에이전트에 외부 기능을 제공하는 함수. `@tool` 데코레이터로 생성한다.

### 2. 프로젝트 구조 관례

```
project/
├── config/
│   ├── agents.yaml    # 에이전트 정의 (role, goal, backstory)
│   └── tasks.yaml     # 태스크 정의 (description, expected_output)
├── main.py            # @CrewBase 클래스와 실행 코드
├── tools.py           # 커스텀 도구 정의
├── output/            # 태스크 결과 파일 저장
└── pyproject.toml     # 의존성 관리
```

### 3. 프롬프트 엔지니어링 원칙

- **상세한 backstory**: 에이전트의 전문성과 성격을 구체적으로 설정하면 결과물 품질이 향상된다.
- **단계별 지시**: `description`에서 번호를 매겨 순서대로 수행할 작업을 명시한다.
- **구체적 기준**: 숫자, 예시, 허용/거부 패턴으로 모호함을 제거한다.
- **출력 형식 템플릿**: `expected_output`에 마크다운 형식의 템플릿을 제공하면 일관된 결과물을 얻는다.

### 4. 도구 설계 원칙

- docstring이 도구의 사용 시점을 결정한다. 명확하고 상세하게 작성해야 한다.
- 타입 힌트는 필수이다. LLM이 올바른 인자를 전달하는 데 사용된다.
- 에이전트에 필요한 도구만 할당한다. 불필요한 도구는 혼란을 야기한다.

### 5. 다중 에이전트 설계 패턴

- **전문화 원칙**: 각 에이전트는 하나의 전문 분야에 집중한다.
- **파이프라인 패턴**: 태스크가 순차적으로 실행되며, 이전 태스크의 결과가 다음 태스크의 입력이 된다.
- **모델 최적화**: 작업 복잡도에 따라 에이전트별로 다른 LLM 모델을 지정할 수 있다.

---

## 실습 과제

### 과제 1: 기본 - 나만의 첫 Crew 만들기

**목표**: CrewAI의 기본 구조를 직접 구현해본다.

**요구사항**:
1. 2개의 에이전트를 가진 Crew를 만들어라:
   - `writer_agent`: 주어진 주제에 대해 짧은 글을 작성하는 에이전트
   - `reviewer_agent`: 작성된 글을 검토하고 피드백을 주는 에이전트
2. 각 에이전트에 적절한 `role`, `goal`, `backstory`를 작성하라
3. 2개의 태스크를 정의하라:
   - `writing_task`: `{topic}`에 대한 300단어 글 작성
   - `review_task`: 작성된 글의 문법, 논리, 가독성 검토
4. `verbose=True`로 실행하고 에이전트의 사고 과정을 관찰하라

### 과제 2: 중급 - 커스텀 도구 활용

**목표**: 실용적인 커스텀 도구를 제작하고 에이전트에 연결한다.

**요구사항**:
1. 다음 커스텀 도구를 구현하라:
   - `get_weather(city: str)`: 날씨 API를 호출하여 현재 날씨를 반환 (무료 API 사용)
   - `calculate(expression: str)`: 수학 표현식을 계산하여 결과 반환
2. `travel_planner_agent`를 만들고 두 도구를 모두 연결하라
3. `plan_trip_task`를 정의하여 특정 도시의 날씨를 확인하고 여행 계획을 수립하게 하라
4. docstring을 변경해가며 도구 호출 패턴의 변화를 관찰하라

### 과제 3: 심화 - 뉴스 리더 확장

**목표**: 챕터에서 만든 뉴스 리더 시스템을 확장한다.

**요구사항**:
1. 기존 뉴스 리더에 다음 에이전트를 추가하라:
   - `translator_agent`: 최종 보고서를 한국어로 번역
   - `fact_checker_agent`: 기사 간 사실 교차 검증
2. `translator_agent`에는 별도의 LLM 모델을 지정하라 (예: `openai/gpt-4o`)
3. `fact_checker_agent`에 적절한 도구를 설계하고 연결하라
4. 5개 단계의 파이프라인이 순차적으로 동작하도록 태스크를 구성하라
5. 다양한 주제로 실행하고, 각 `output_file`의 결과물을 비교 분석하라

### 과제 4: 도전 - 자율 에이전트 팀 설계

**목표**: 실무에 적용 가능한 다중 에이전트 시스템을 처음부터 설계한다.

**요구사항**:
1. 자신이 관심 있는 도메인(금융, 교육, 건강 등)을 선택하라
2. 최소 3개의 전문화된 에이전트를 설계하라
3. 각 에이전트에 최소 1개의 커스텀 도구를 제작하여 연결하라
4. 상세한 `backstory`와 구체적인 `expected_output` 형식을 작성하라
5. 전체 시스템을 실행하고, 결과물의 품질을 평가하는 기준을 수립하라
6. 결과물을 마크다운 파일로 저장하고 발표 자료를 준비하라

---

> **다음 챕터 예고**: Chapter 4에서는 CrewAI의 고급 기능인 에이전트 간 통신, 조건부 태스크 실행, 메모리 시스템 등을 학습하여 더욱 정교한 에이전트 시스템을 구축한다.
