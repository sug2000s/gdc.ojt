# Chapter 5: CrewAI Flows - 콘텐츠 파이프라인 에이전트 구축

---

## 1. 챕터 개요

이번 챕터에서는 **CrewAI Flows**를 활용하여 AI 기반 콘텐츠 생성 파이프라인을 처음부터 끝까지 구축하는 방법을 학습합니다. Flow는 CrewAI에서 제공하는 **워크플로우 오케스트레이션 시스템**으로, 여러 단계의 작업을 순서대로 또는 조건에 따라 실행할 수 있게 해주는 핵심 기능입니다.

### 학습 목표

- CrewAI Flow의 기본 구조와 데코레이터(`@start`, `@listen`, `@router`) 이해
- Pydantic 모델을 활용한 Flow 상태(State) 관리
- 조건부 라우팅과 반복 루프(Refinement Loop) 구현
- Flow 내에서 LLM과 Agent를 직접 호출하는 방법
- Crew를 Flow에 통합하여 복잡한 AI 파이프라인 완성

### 프로젝트 구조

```
content-pipeline-agent/
├── main.py              # Flow 메인 로직
├── tools.py             # 웹 검색 도구 (Firecrawl)
├── seo_crew.py          # SEO 분석 Crew
├── virality_crew.py     # 바이럴 분석 Crew
├── pyproject.toml       # 프로젝트 의존성
├── crewai_flow.html     # Flow 시각화 파일
└── .gitignore
```

### 핵심 의존성

```toml
[project]
name = "content-pipeline-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "firecrawl-py>=2.16.3",
    "python-dotenv>=1.1.1",
]
```

- **crewai[tools]**: CrewAI 프레임워크와 도구 확장팩
- **firecrawl-py**: 웹 검색 및 스크래핑 API 클라이언트
- **python-dotenv**: 환경 변수(.env) 관리

---

## 2. 섹션별 상세 설명

---

### 2.1 Your First Flow (첫 번째 Flow 만들기)

**커밋:** `c47fd95`

#### 주제 및 목표

CrewAI Flow의 가장 기본적인 구조를 이해합니다. Flow가 무엇인지, 어떤 데코레이터들이 있는지, 상태(State)는 어떻게 관리되는지를 학습합니다.

#### 핵심 개념 설명

**Flow란?**

Flow는 CrewAI에서 제공하는 **워크플로우 엔진**입니다. 여러 함수(step)를 정의하고, 각 함수가 어떤 순서로 실행될지를 데코레이터로 선언합니다. 이를 통해 복잡한 AI 파이프라인을 직관적으로 구성할 수 있습니다.

**주요 데코레이터:**

| 데코레이터 | 역할 | 설명 |
|------------|------|------|
| `@start()` | 시작점 | Flow가 시작될 때 가장 먼저 실행되는 함수 |
| `@listen(fn)` | 리스너 | 지정한 함수가 완료되면 실행 |
| `@router(fn)` | 라우터 | 반환값에 따라 다른 경로로 분기 |
| `and_(a, b)` | AND 조건 | 두 함수가 **모두** 완료되어야 실행 |
| `or_(a, b)` | OR 조건 | 두 함수 중 **하나라도** 완료되면 실행 |

**Flow State (상태 관리):**

Flow는 Pydantic `BaseModel`을 상태 객체로 사용합니다. 모든 step에서 `self.state`를 통해 공유 상태에 접근하고 수정할 수 있습니다.

#### 코드 분석

```python
from crewai.flow.flow import Flow, listen, start, router, and_, or_
from pydantic import BaseModel


class MyFirstFlowState(BaseModel):
    user_id: int = 1
    is_admin: bool = False


class MyFirstFlow(Flow[MyFirstFlowState]):

    @start()
    def first(self):
        print(self.state.user_id)
        print("Hello")

    @listen(first)
    def second(self):
        self.state.user_id = 2
        print("world")

    @listen(first)
    def third(self):
        print("!")

    @listen(and_(second, third))
    def final(self):
        print(":)")

    @router(final)
    def route(self):
        if self.state.is_admin:
            return "even"
        else:
            return "odd"

    @listen("even")
    def handle_even(self):
        print("even")

    @listen("odd")
    def handle_odd(self):
        print("odd")


flow = MyFirstFlow()

flow.plot()
flow.kickoff()
```

**코드 흐름 상세 분석:**

1. **`MyFirstFlowState`**: Pydantic 모델로 Flow의 상태를 정의합니다. `user_id`와 `is_admin` 두 가지 필드를 가집니다.

2. **`MyFirstFlow(Flow[MyFirstFlowState])`**: 제네릭 타입으로 상태 클래스를 지정합니다. 이렇게 하면 `self.state`가 `MyFirstFlowState` 타입이 됩니다.

3. **`@start()` - `first()`**: Flow의 진입점입니다. `self.state.user_id`를 출력하고 "Hello"를 출력합니다.

4. **`@listen(first)` - `second()`와 `third()`**: `first()`가 완료되면 **동시에(병렬로)** 실행됩니다. `second()`는 상태를 변경(`user_id = 2`)하고, `third()`는 "!"를 출력합니다. 같은 함수를 listen하는 여러 함수가 있으면 병렬로 실행된다는 점이 중요합니다.

5. **`@listen(and_(second, third))` - `final()`**: `second()`와 `third()`가 **모두** 완료된 후에야 실행됩니다. `and_()` 조건은 동기화 지점(synchronization point) 역할을 합니다.

6. **`@router(final)` - `route()`**: `final()` 이후 조건에 따라 분기합니다. 반환하는 문자열(`"even"` 또는 `"odd"`)에 따라 해당 문자열을 listen하는 함수가 실행됩니다.

7. **`@listen("even")` / `@listen("odd")`**: 라우터가 반환한 문자열 값을 listen합니다. 함수 참조가 아닌 **문자열**을 listen할 수 있다는 점이 핵심입니다.

8. **`flow.plot()`**: Flow의 실행 흐름을 HTML 파일(`crewai_flow.html`)로 시각화합니다.

9. **`flow.kickoff()`**: Flow를 실행합니다.

#### 함께 생성된 도구 파일: `tools.py`

```python
import os, re
from crewai.tools import tool
from firecrawl import FirecrawlApp, ScrapeOptions


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

**도구 분석:**

- `@tool` 데코레이터로 CrewAI 도구를 정의합니다.
- Firecrawl API를 사용하여 웹 검색을 수행합니다.
- 검색 결과에서 불필요한 링크와 특수문자를 정규식으로 제거(클리닝)합니다.
- 깔끔한 마크다운 형태의 결과를 반환합니다.

#### 실습 포인트

- `is_admin`을 `True`로 바꿔서 라우팅 경로가 변경되는 것을 확인해보세요.
- `@listen` 데코레이터에서 함수 참조(`first`)와 문자열(`"even"`)을 사용하는 차이를 이해하세요.
- `flow.plot()`으로 생성된 HTML 파일을 브라우저에서 열어 시각적으로 Flow 구조를 확인하세요.
- `and_`와 `or_`의 동작 차이를 실험해보세요.

---

### 2.2 Content Pipeline Flow (콘텐츠 파이프라인 플로우)

**커밋:** `1e78354`

#### 주제 및 목표

첫 번째 예제에서 배운 Flow 개념을 활용하여, 실제 사용 가능한 **콘텐츠 생성 파이프라인**의 골격을 설계합니다. 트윗, 블로그 포스트, LinkedIn 게시물 등 다양한 콘텐츠 유형에 대응하는 파이프라인 구조를 만듭니다.

#### 핵심 개념 설명

**실전 파이프라인 설계 원칙:**

1. **입력 검증(Validation)**: Flow 시작 시 잘못된 입력을 조기에 차단
2. **조건부 라우팅**: 콘텐츠 유형에 따라 다른 처리 경로 선택
3. **품질 검사 분기**: 블로그는 SEO 검사, 소셜 미디어는 바이럴 검사
4. **통합 마무리**: 모든 경로가 최종적으로 하나의 완료 단계로 수렴

#### 코드 분석

**상태 모델 설계:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""    # "tweet", "blog", "linkedin" 중 하나
    topic: str = ""           # 콘텐츠 주제

    # Internal
    max_length: int = 0       # 콘텐츠 유형별 최대 길이
```

상태를 **Inputs**(외부 입력)과 **Internal**(내부 처리용)로 구분하여 관리합니다. 이는 깔끔한 코드 구조를 위한 좋은 관행입니다.

**파이프라인 초기화 및 검증:**

```python
class ContentPipelineFlow(Flow[ContentPipelineState]):

    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The content type is wrong.")

        if self.topic == "":
            raise ValueError("The topic can't be blank.")

        if self.state.content_type == "tweet":
            self.state.max_length = 150
        elif self.state.content_type == "blog":
            self.state.max_length = 800
        elif self.state.content_type == "linkedin":
            self.state.max_length = 500
```

- 시작 단계에서 입력값을 검증합니다 (Fail Fast 패턴).
- 콘텐츠 유형에 따라 `max_length`를 설정하여 이후 단계에서 활용할 수 있도록 합니다.

**리서치 및 라우팅:**

```python
    @listen(init_content_pipeline)
    def conduct_research(self):
        print("Researching....")
        return True

    @router(conduct_research)
    def router(self):
        content_type = self.state.content_type
        if content_type == "blog":
            return "make_blog"
        elif content_type == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"
```

`@router` 데코레이터가 반환하는 문자열에 따라 실행 경로가 결정됩니다. 이것이 Flow에서 **동적 분기**를 구현하는 핵심 패턴입니다.

**콘텐츠 유형별 처리 및 품질 검사:**

```python
    @listen("make_blog")
    def handle_make_blog(self):
        print("Making blog post...")

    @listen("make_tweet")
    def handle_make_tweet(self):
        print("Making tweet...")

    @listen("make_linkedin_post")
    def handle_make_linkedin_post(self):
        print("Making linkedin post...")

    @listen(handle_make_blog)
    def check_seo(self):
        print("Checking Blog SEO")

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        print("Checking virality...")

    @listen(or_(check_virality, check_seo))
    def finalize_content(self):
        print("Finalizing content")
```

**실행 흐름 다이어그램:**

```
init_content_pipeline
        |
  conduct_research
        |
      router -----> "make_blog" -----> handle_make_blog -----> check_seo ------\
        |                                                                        |
        +-------> "make_tweet" -----> handle_make_tweet ------\                  |
        |                                                      +--> check_virality --> finalize_content
        +-------> "make_linkedin_post" -> handle_make_linkedin_post --/
```

**핵심 설계 포인트:**
- 블로그는 **SEO 검사** 경로로, 트윗과 LinkedIn 게시물은 **바이럴 검사** 경로로 분기합니다.
- `or_(check_virality, check_seo)`를 사용하여 어느 검사가 완료되든 `finalize_content`가 실행됩니다.
- 콘텐츠 유형에 따라 다른 품질 기준을 적용하는 것은 실무에서도 매우 일반적인 패턴입니다.

**Flow 실행 (inputs 전달):**

```python
flow = ContentPipelineFlow()

flow.kickoff(
    inputs={
        "content_type": "tweet",
        "topic": "AI Dog Training",
    },
)
```

`kickoff()`에 `inputs` 딕셔너리를 전달하면 상태의 해당 필드가 자동으로 설정됩니다.

#### 실습 포인트

- `content_type`을 `"blog"`, `"tweet"`, `"linkedin"`으로 각각 바꿔가며 실행 경로가 어떻게 달라지는지 확인하세요.
- `flow.plot()`으로 파이프라인의 시각적 구조를 확인하세요.
- `or_`와 `and_`의 차이를 이 맥락에서 생각해보세요: 왜 `finalize_content`에서 `or_`을 사용하는지 이해하세요.

---

### 2.3 Refinement Loop (개선 반복 루프)

**커밋:** `482e52c`

#### 주제 및 목표

AI가 생성한 콘텐츠의 품질이 기준에 미달할 경우, 자동으로 콘텐츠를 다시 생성하도록 하는 **반복 루프(Refinement Loop)** 패턴을 구현합니다. 이는 AI 에이전트 시스템에서 품질 보장을 위한 핵심 패턴입니다.

#### 핵심 개념 설명

**Refinement Loop란?**

Refinement Loop는 "생성 -> 평가 -> 재생성"의 순환 구조입니다. 평가 점수가 기준을 충족할 때까지 콘텐츠를 반복적으로 개선합니다. 이 패턴은 다음과 같은 상황에서 필수적입니다:

- LLM이 처음부터 완벽한 결과를 내지 못할 때
- 품질 기준이 높아서 여러 번의 시도가 필요할 때
- 자동화된 품질 보증(QA) 프로세스가 필요할 때

**라우터를 활용한 루프 구현:**

Flow의 `@router`는 단순한 분기뿐 아니라 **이전 단계로 되돌아가는 루프**를 만드는 데도 사용할 수 있습니다. `@listen`에 문자열 조건을 추가하면 라우터의 반환값에 따라 이전 단계가 다시 실행됩니다.

#### 코드 분석

**상태에 점수와 콘텐츠 필드 추가:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: int = 0                # 품질 점수 추가

    # Content
    blog_post: str = ""           # 생성된 블로그 포스트
    tweet: str = ""               # 생성된 트윗
    linkedin_post: str = ""       # 생성된 LinkedIn 게시물
```

점수(`score`)와 각 콘텐츠 유형별 결과를 저장하는 필드가 추가되었습니다. 이 상태 필드들이 루프의 핵심 역할을 합니다.

**`or_` 문자열을 활용한 재진입 포인트:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        # 블로그가 이미 만들어졌으면 기존 것을 AI에게 보여주고 개선 요청,
        # 아직 없으면 새로 생성
        print("Making blog post...")

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        print("Making tweet...")

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        print("Making linkedin post...")
```

**핵심 변경사항**: `@listen("make_blog")`가 `@listen(or_("make_blog", "remake_blog"))`로 변경되었습니다. 이렇게 하면:
- 최초 생성 시: 라우터가 `"make_blog"`를 반환하여 실행
- 재생성 시: `score_router`가 `"remake_blog"`를 반환하여 같은 함수가 다시 실행

이것이 바로 **루프의 재진입 포인트**입니다.

**점수 기반 라우터 (루프 제어):**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score >= 8:
            return "check_passed"       # 통과 -> finalize_content로
        else:
            if content_type == "blog":
                return "remake_blog"     # 루프백 -> handle_make_blog로
            elif content_type == "linkedin":
                return "remake_linkedin_post"  # 루프백
            else:
                return "remake_tweet"          # 루프백

    @listen("check_passed")
    def finalize_content(self):
        print("Finalizing content")
```

**개선된 실행 흐름:**

```
init_content_pipeline
        |
  conduct_research
        |
  conduct_research_router
        |
   +---------+-----------+
   |         |           |
make_blog  make_tweet  make_linkedin_post
   |         |           |
check_seo  check_virality (or_)
   |         |
   +----+----+
        |
   score_router
    /        \
score >= 8   score < 8
    |            |
"check_passed"  "remake_blog" / "remake_tweet" / "remake_linkedin_post"
    |                    |
finalize_content    (루프: 해당 콘텐츠 재생성 단계로 돌아감)
```

**라우터 이름 변경 주의:**

```python
    @router(conduct_research)
    def conduct_research_router(self):   # 기존 "router"에서 이름 변경
```

기존의 `router`라는 메서드명이 `conduct_research_router`로 변경되었습니다. Flow에서 메서드명은 시각화와 디버깅에 중요하므로, 역할을 명확히 드러내는 이름을 사용하는 것이 좋습니다.

#### 실습 포인트

- `score >= 8` 기준을 변경하면서 루프가 몇 번 반복되는지 관찰해보세요.
- 무한 루프 방지를 위한 최대 반복 횟수(max iteration) 로직을 추가해보세요.
- `remake_*` 경로에서 기존 콘텐츠를 참조하여 개선하는 로직을 설계해보세요 (다음 섹션에서 구현됩니다).

---

### 2.4 LLMs and Agents (LLM과 에이전트 통합)

**커밋:** `c341770`

#### 주제 및 목표

지금까지 `print()` 문으로 대체했던 자리표시자(placeholder)에 **실제 LLM 호출**과 **Agent**를 연결합니다. CrewAI Flow 안에서 LLM을 직접 호출하는 방법과, Agent를 독립적으로 사용하는 방법을 모두 학습합니다.

#### 핵심 개념 설명

**Flow 안에서 AI를 사용하는 두 가지 방법:**

1. **`LLM.call()`**: LLM을 직접 호출합니다. 빠르고 간단하며, 구조화된 출력(Structured Output)이 필요할 때 유용합니다.
2. **`Agent.kickoff()`**: Agent를 생성하여 실행합니다. 도구(Tool) 사용이 필요하거나, 더 복잡한 추론이 필요할 때 사용합니다.

**Pydantic 모델을 활용한 구조화된 출력(Structured Output):**

LLM의 응답을 자유 텍스트가 아닌 **정해진 구조**로 받을 수 있습니다. 이는 후속 처리 단계에서 데이터를 안정적으로 다룰 수 있게 해줍니다.

#### 코드 분석

**구조화된 출력을 위한 Pydantic 모델 정의:**

```python
from typing import List
from pydantic import BaseModel


class BlogPost(BaseModel):
    title: str
    subtitle: str
    sections: List[str]


class Tweet(BaseModel):
    content: str
    hashtags: str


class LinkedInPost(BaseModel):
    hook: str
    content: str
    call_to_action: str


class Score(BaseModel):
    score: int = 0
    reason: str = ""
```

각 콘텐츠 유형에 맞는 **출력 스키마**를 정의합니다:
- **BlogPost**: 제목, 부제, 여러 섹션으로 구성
- **Tweet**: 내용과 해시태그
- **LinkedInPost**: 훅(주의를 끄는 첫 문장), 본문, CTA(행동 유도 문구)
- **Score**: 점수와 그 이유 (품질 평가용)

**상태 모델 업데이트:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    research: str = ""              # 리서치 결과 저장
    score: Score | None = None      # Score 객체로 변경

    # Content
    blog_post: BlogPost | None = None   # Pydantic 모델로 변경
    tweet: str = ""
    linkedin_post: str = ""
```

`str` 타입이었던 `blog_post`가 `BlogPost | None`으로 변경되었습니다. `None`은 아직 생성되지 않은 상태를 나타냅니다.

**Agent를 활용한 리서치 단계:**

```python
from crewai.agent import Agent
from tools import web_search_tool

    @listen(init_content_pipeline)
    def conduct_research(self):

        researcher = Agent(
            role="Head Researcher",
            backstory="You're like a digital detective who loves digging up "
                      "fascinating facts and insights. You have a knack for "
                      "finding the good stuff that others miss.",
            goal=f"Find the most interesting and useful info about "
                 f"{self.state.topic}",
            tools=[web_search_tool],
        )

        self.state.research = researcher.kickoff(
            f"Find the most interesting and useful info about "
            f"{self.state.topic}"
        )
```

**Agent vs LLM 직접 호출의 차이:**

| 특성 | `Agent.kickoff()` | `LLM.call()` |
|------|-------------------|---------------|
| 도구 사용 | 가능 (웹 검색 등) | 불가 |
| 추론 단계 | 다단계 추론 | 단일 호출 |
| 속도 | 상대적으로 느림 | 빠름 |
| 적합한 용도 | 리서치, 복잡한 작업 | 콘텐츠 생성, 변환 |

리서치 단계에서는 **웹 검색 도구**가 필요하므로 Agent를 사용하고, 콘텐츠 생성에서는 LLM을 직접 호출합니다.

**LLM을 활용한 블로그 생성 (구조화된 출력):**

```python
from crewai import LLM

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):

        blog_post = self.state.blog_post

        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            # 최초 생성
            self.state.blog_post = llm.call(
                f"""
            Make a blog post on the topic {self.state.topic}
            using the following research:

            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
        else:
            # 재생성 (기존 콘텐츠 + 점수 피드백 기반 개선)
            self.state.blog_post = llm.call(
                f"""
            You wrote this blog post on {self.state.topic},
            but it does not have a good SEO score because of
            {self.state.score.reason}

            Improve it.

            <blog post>
            {self.state.blog_post.model_dump_json()}
            </blog post>

            Use the following research.

            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
```

**핵심 분석:**

1. **`LLM(model="openai/o4-mini", response_format=BlogPost)`**: `response_format`에 Pydantic 모델을 전달하면, LLM이 해당 모델의 JSON 스키마에 맞는 응답을 생성합니다.

2. **최초 생성 vs 재생성 분기**: `blog_post is None`으로 판단합니다.
   - 최초: 리서치 결과만 전달
   - 재생성: 기존 콘텐츠 + 점수가 낮은 이유(`self.state.score.reason`)를 함께 전달하여 개선 유도

3. **`model_dump_json()`**: Pydantic 모델을 JSON 문자열로 변환하여 LLM에게 전달합니다.

4. **프롬프트에서 XML 태그 활용**: `<research>`, `<blog post>` 등 XML 태그로 프롬프트의 각 섹션을 구분합니다. 이는 LLM이 프롬프트 구조를 더 잘 이해하도록 돕는 효과적인 기법입니다.

#### 실습 포인트

- `response_format`을 제거하고 실행해서 구조화된 출력과 자유 텍스트 출력의 차이를 비교해보세요.
- 다른 LLM 모델(`"openai/gpt-4o"` 등)로 교체하여 결과 품질을 비교해보세요.
- 프롬프트를 수정하여 블로그 포스트의 톤이나 스타일을 변경해보세요.
- Agent의 `backstory`를 수정하면 리서치 결과가 어떻게 달라지는지 실험해보세요.

---

### 2.5 Adding Crews To Flows (Flow에 Crew 통합하기)

**커밋:** `8e039ec`

#### 주제 및 목표

개별 Agent나 LLM 호출을 넘어서, **Crew**(에이전트 팀)를 Flow에 통합합니다. SEO 분석 Crew와 바이럴 분석 Crew를 만들어 콘텐츠 품질 평가를 수행합니다. 이 섹션이 이 챕터의 하이라이트이자 가장 중요한 부분입니다.

#### 핵심 개념 설명

**Flow + Crew 통합의 가치:**

Flow는 **전체 워크플로우의 흐름을 제어**하고, Crew는 **특정 단계에서 복잡한 작업을 수행**합니다. 이 둘을 결합하면:

- Flow가 전체 파이프라인의 오케스트레이션을 담당
- 각 단계에서 필요할 때 전문화된 Crew를 호출
- Crew의 출력 결과를 Flow 상태로 받아 다음 단계에서 활용

**`@CrewBase` 데코레이터:**

CrewAI에서 Crew를 클래스로 정의할 때 사용하는 데코레이터입니다. `@agent`, `@task`, `@crew` 데코레이터와 함께 사용하여 Crew의 구성원과 작업을 선언적으로 정의합니다.

#### 코드 분석

**SEO 분석 Crew (`seo_crew.py`):**

```python
from crewai.project import CrewBase, agent, task, crew
from crewai import Agent, Task, Crew
from pydantic import BaseModel


class Score(BaseModel):
    score: int
    reason: str


@CrewBase
class SeoCrew:

    @agent
    def seo_expert(self):
        return Agent(
            role="SEO Specialist",
            goal="Analyze blog posts for SEO optimization and provide a score "
                 "with detailed reasoning. Be very very very demanding, "
                 "don't give underserved good scores.",
            backstory="""You are an experienced SEO specialist with expertise
            in content optimization. You analyze blog posts for keyword usage,
            meta descriptions, content structure, readability, and search
            intent alignment to help content rank better in search engines.""",
            verbose=True,
        )

    @task
    def seo_audit(self):
        return Task(
            description="""Analyze the blog post for SEO effectiveness
            and provide:

            1. An SEO score from 0-10 based on:
               - Keyword optimization
               - Title effectiveness
               - Content structure (headers, paragraphs)
               - Content length and quality
               - Readability
               - Search intent alignment

            2. A clear reason explaining the score, focusing on:
               - Main strengths (if score is high)
               - Critical weaknesses that need improvement (if score is low)
               - The most important factor affecting the score

            Blog post to analyze: {blog_post}
            Target topic: {topic}
            """,
            expected_output="""A Score object with:
            - score: integer from 0-10 rating the SEO quality
            - reason: string explaining the main factors affecting the score""",
            agent=self.seo_expert(),
            output_pydantic=Score,
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
```

**Crew 구조 분석:**

1. **`@CrewBase`**: 클래스를 Crew 기반 클래스로 선언합니다. `self.agents`와 `self.tasks` 프로퍼티를 자동으로 제공합니다.

2. **`@agent`**: Agent를 정의합니다. SEO 전문가로서 "매우 엄격하게 점수를 매기라"는 지침이 포함되어 있습니다. 이는 Refinement Loop가 제대로 작동하도록(쉽게 통과하지 않도록) 하기 위한 설계입니다.

3. **`@task`**: Task를 정의합니다. `{blog_post}`와 `{topic}`은 `kickoff(inputs={})`에서 전달되는 변수입니다. `output_pydantic=Score`로 구조화된 출력을 지정합니다.

4. **`@crew`**: 최종 Crew 객체를 반환합니다. `self.agents`와 `self.tasks`는 `@CrewBase`가 자동으로 수집한 목록입니다.

**바이럴 분석 Crew (`virality_crew.py`):**

```python
@CrewBase
class ViralityCrew:

    @agent
    def virality_expert(self):
        return Agent(
            role="Social Media Virality Expert",
            goal="Analyze social media content for viral potential and "
                 "provide a score with actionable feedback",
            backstory="""You are a social media strategist with deep
            expertise in viral content creation. You've analyzed thousands
            of viral posts across Twitter and LinkedIn, understanding the
            psychology of engagement, shareability, and what makes content
            spread. You know the specific mechanics that drive virality on
            each platform - from hook writing to emotional triggers.""",
            verbose=True,
        )

    @task
    def virality_audit(self):
        return Task(
            description="""Analyze the social media content for viral
            potential and provide:

            1. A virality score from 0-10 based on:
               - Hook strength and attention-grabbing potential
               - Emotional resonance and relatability
               - Shareability factor
               - Call-to-action effectiveness
               - Platform-specific best practices
               - Trending topic alignment
               - Content format optimization

            2. A clear reason explaining the score

            Content to analyze: {content}
            Content type: {content_type}
            Target topic: {topic}
            """,
            expected_output="""A Score object with:
            - score: integer from 0-10 rating the viral potential
            - reason: string explaining the main factors affecting virality""",
            agent=self.virality_expert(),
            output_pydantic=Score,
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )
```

바이럴 Crew는 SEO Crew와 구조는 동일하지만, 평가 기준이 다릅니다:
- SEO Crew: 키워드 최적화, 제목 효과, 콘텐츠 구조 등
- 바이럴 Crew: 훅 강도, 감정적 공감, 공유 가능성, 플랫폼별 모범 사례 등

**Flow에서 Crew 호출:**

```python
from seo_crew import SeoCrew
from virality_crew import ViralityCrew

    @listen(handle_make_blog)
    def check_seo(self):

        result = (
            SeoCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "blog_post": self.state.blog_post.model_dump_json(),
                }
            )
        )
        self.state.score = result.pydantic

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        result = (
            ViralityCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "content_type": self.state.content_type,
                    "content": (
                        self.state.tweet
                        if self.state.contenty_type == "tweet"
                        else self.state.linkedin_post
                    ),
                }
            )
        )
        self.state.score = result.pydantic
```

**Crew 호출 패턴 분석:**

```python
SeoCrew()           # 1. Crew 클래스 인스턴스 생성
    .crew()         # 2. Crew 객체 획득
    .kickoff(       # 3. Crew 실행
        inputs={    # 4. Task의 {변수}에 매핑될 입력값 전달
            "topic": self.state.topic,
            "blog_post": self.state.blog_post.model_dump_json(),
        }
    )
```

- `result.pydantic`: Crew의 실행 결과에서 Pydantic 모델을 추출합니다. Task에서 `output_pydantic=Score`를 지정했으므로, `result.pydantic`는 `Score` 객체입니다.

**트윗과 LinkedIn 콘텐츠 생성 로직 완성:**

이 커밋에서는 블로그뿐 아니라 트윗과 LinkedIn 게시물 생성 로직도 LLM을 활용하여 완전히 구현되었습니다:

```python
    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        tweet = self.state.tweet
        llm = LLM(model="openai/o4-mini", response_format=Tweet)

        if tweet is None:
            result = llm.call(
                f"""
            Make a tweet that can go viral on the topic
            {self.state.topic} using the following research:
            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )
        else:
            result = llm.call(
                f"""
            You wrote this tweet on {self.state.topic}, but it does
            not have a good virality score because of
            {self.state.score.reason}

            Improve it.
            <tweet>
            {self.state.tweet.model_dump_json()}
            </tweet>
            Use the following research.
            <research>
            ================
            {self.state.research}
            ================
            </research>
            """
            )

        self.state.tweet = Tweet.model_validate_json(result)
```

**`model_validate_json(result)`**: LLM이 반환한 JSON 문자열을 Pydantic 모델로 파싱합니다. `response_format=Tweet`을 지정했으므로 LLM은 `Tweet` 스키마에 맞는 JSON을 반환하고, 이를 다시 Pydantic 객체로 변환하는 것입니다.

**상태 모델 최종 업데이트:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    research: str = ""
    score: Score | None = None

    # Content
    blog_post: BlogPost | None = None
    tweet: Tweet | None = None           # str에서 Tweet으로 변경
    linkedin_post: LinkedInPost | None = None  # str에서 LinkedInPost으로 변경
```

#### 실습 포인트

- `SeoCrew`의 `goal`에서 "Be very very very demanding"을 제거하면 점수가 어떻게 변하는지 확인해보세요.
- 새로운 Crew(예: 문법 검사 Crew, 팩트 체크 Crew)를 만들어 파이프라인에 추가해보세요.
- `virality_crew.py`에 Agent를 추가하여 멀티 에이전트 Crew로 확장해보세요.
- Crew의 `verbose=True`를 통해 실행 과정을 관찰하세요.

---

### 2.6 Conclusions (마무리 및 최종 완성)

**커밋:** `be0cf85`

#### 주제 및 목표

파이프라인의 최종 단계를 완성합니다. 품질 기준을 조정하고, 최종 콘텐츠 출력을 정리하며, 전체 파이프라인이 end-to-end로 작동하도록 마무리합니다.

#### 핵심 개념 설명

**최종 조정 사항:**

1. **점수 기준 완화**: `score >= 8`에서 `score >= 7`로 변경하여 실용적인 수준으로 조정
2. **재생성 로깅**: 디버깅을 위한 로그 메시지 추가
3. **최종 출력 포맷팅**: 콘텐츠 유형에 따른 결과 출력
4. **반환값 구현**: Flow의 최종 결과를 반환

#### 코드 분석

**점수 기준 조정:**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score.score >= 7:        # 8에서 7로 완화
            return "check_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"
```

점수 기준을 7로 낮춘 것은 실용적인 판단입니다. 너무 높은 기준은 무한에 가까운 루프를 유발할 수 있고, 너무 낮으면 품질이 떨어집니다. 실제 프로덕션 환경에서는 이 기준을 설정 파일이나 환경 변수로 관리하는 것이 좋습니다.

**최종 콘텐츠 출력:**

```python
    @listen("check_passed")
    def finalize_content(self):
        """Finalize the content"""
        print("Finalizing content...")

        if self.state.content_type == "blog":
            print(f"Blog Post: {self.state.blog_post.title}")
            print(f"SEO Score: {self.state.score.score}/100")
        elif self.state.content_type == "tweet":
            print(f"Tweet: {self.state.tweet}")
            print(f"Virality Score: {self.state.score.score}/100")
        elif self.state.content_type == "linkedin":
            print(f"LinkedIn: {self.state.linkedin_post.title}")
            print(f"Virality Score: {self.state.score.score}/100")

        print("Content ready for publication!")
        return (
            self.state.linkedin_post
            if self.state.content_type == "linkedin"
            else (
                self.state.tweet
                if self.state.content_type == "tweet"
                else self.state.blog_post
            )
        )
```

**반환값 패턴 분석:**

`finalize_content()`는 콘텐츠 유형에 따라 해당 Pydantic 모델을 반환합니다. Flow의 마지막 step이 반환하는 값은 `flow.kickoff()`의 반환값이 됩니다. 이를 통해 Flow를 호출한 외부 코드에서 결과를 받아 활용할 수 있습니다.

**재생성 시 로그 추가:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        blog_post = self.state.blog_post
        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            result = llm.call(...)
        else:
            print("Remaking blog.")   # 디버깅용 로그 추가
            result = llm.call(...)
```

**바이럴 체크 수정:**

```python
    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        result = (
            ViralityCrew()
            .crew()
            .kickoff(
                inputs={
                    "topic": self.state.topic,
                    "content_type": self.state.content_type,
                    "content": (
                        self.state.tweet.model_dump_json()       # 수정됨
                        if self.state.contenty_type == "tweet"
                        else self.state.linkedin_post.model_dump_json()  # 수정됨
                    ),
                }
            )
        )
        self.state.score = result.pydantic
```

`.model_dump_json()`이 추가되어 Pydantic 모델을 JSON 문자열로 올바르게 직렬화합니다.

#### 실습 포인트

- 전체 파이프라인을 `content_type`별로 실행하여 결과를 비교해보세요.
- `flow.kickoff()`의 반환값을 변수에 저장하고 활용하는 코드를 작성해보세요.
- 최대 반복 횟수 제한 로직을 `score_router`에 추가해보세요.

---

## 3. 챕터 핵심 정리

### Flow 기본 구조

| 개념 | 설명 | 데코레이터 |
|------|------|------------|
| 시작점 | Flow가 시작되는 첫 함수 | `@start()` |
| 리스너 | 특정 함수 완료 후 실행 | `@listen(fn)` |
| 라우터 | 조건에 따라 분기 | `@router(fn)` |
| AND 조건 | 모든 선행 함수 완료 필요 | `and_(a, b)` |
| OR 조건 | 하나라도 완료되면 실행 | `or_(a, b)` |
| 문자열 리스닝 | 라우터 반환값에 반응 | `@listen("문자열")` |

### 상태 관리 패턴

```python
class MyState(BaseModel):
    # 입력값
    input_field: str = ""

    # 내부 처리용
    intermediate_data: str = ""

    # 결과물 (Pydantic 모델 활용)
    output: MyOutputModel | None = None
```

### AI 호출 방식 비교

| 방식 | 용도 | 도구 사용 | 구조화 출력 |
|------|------|----------|------------|
| `LLM.call()` | 단순 생성/변환 | 불가 | `response_format` |
| `Agent.kickoff()` | 리서치, 복합 추론 | 가능 | 제한적 |
| `Crew.kickoff()` | 팀 기반 복합 작업 | 가능 | `output_pydantic` |

### Refinement Loop 패턴

```
생성 --> 평가 --> 점수 확인 --[통과]--> 완료
                    |
              [미달] --> 재생성 (루프백)
```

핵심: `@listen(or_("make_x", "remake_x"))`로 최초 생성과 재생성을 같은 함수에서 처리

### Crew를 Flow에 통합하는 패턴

```python
result = MyCrewClass().crew().kickoff(inputs={...})
self.state.score = result.pydantic
```

---

## 4. 실습 과제

### 과제 1: 기본 Flow 만들기 (난이도: 초급)

다음 요구사항에 맞는 Flow를 작성하세요:
- 사용자의 이름과 언어(한국어/영어)를 입력받는다
- 언어에 따라 인사말을 다르게 생성한다 (라우터 활용)
- 최종적으로 인사말을 출력한다

**힌트:** `@start()`, `@router()`, `@listen("문자열")` 데코레이터를 사용하세요.

### 과제 2: LLM 통합 Flow (난이도: 중급)

레시피 생성 Flow를 만드세요:
- 입력: 재료 목록과 요리 스타일 (한식/양식/중식)
- `conduct_research` 단계에서 Agent를 사용해 해당 재료로 만들 수 있는 요리를 검색
- `generate_recipe` 단계에서 LLM을 직접 호출하여 구조화된 레시피 생성
- Pydantic 모델: `Recipe(title: str, ingredients: List[str], steps: List[str], cooking_time: int)`

### 과제 3: Refinement Loop가 포함된 파이프라인 (난이도: 고급)

이메일 마케팅 콘텐츠 생성 파이프라인을 구축하세요:
- 입력: 제품명, 타겟 고객층, 이메일 유형(프로모션/뉴스레터/환영 이메일)
- 리서치 Agent가 제품과 타겟 고객에 대해 조사
- LLM이 이메일 콘텐츠 생성
- 품질 평가 Crew가 이메일의 효과성을 평가 (제목 매력도, CTA 효과, 톤 적절성)
- 점수가 7점 미만이면 재생성하는 Refinement Loop 구현
- 최대 3회까지만 반복하고 그 이후에는 최선의 결과를 반환

**힌트:**
- 상태에 `iteration_count: int = 0` 필드를 추가하세요
- `score_router`에서 `iteration_count >= 3`이면 점수와 관계없이 `"check_passed"`를 반환하세요

### 과제 4: 멀티 Crew 파이프라인 (난이도: 고급)

본 챕터의 Content Pipeline을 확장하세요:
- 새로운 Crew 추가: `GrammarCrew` (문법 및 가독성 검사)
- 블로그 포스트의 경우 SEO 검사와 문법 검사를 **병렬로** 실행 (`and_` 활용)
- 두 검사의 점수를 가중 평균으로 합산하여 최종 점수 산출
- 모든 검사를 통과해야 `finalize_content`로 진행

---

## 부록: 전체 최종 코드 구조 요약

### main.py 최종 구조

```
ContentPipelineState (Pydantic BaseModel)
├── content_type, topic         # 입력
├── max_length, research, score # 내부
└── blog_post, tweet, linkedin_post  # 출력 (Pydantic 모델)

ContentPipelineFlow (Flow)
├── init_content_pipeline()          @start       - 입력 검증 및 초기화
├── conduct_research()               @listen      - Agent로 웹 리서치
├── conduct_research_router()        @router      - 콘텐츠 유형별 분기
├── handle_make_blog()               @listen(or_) - LLM으로 블로그 생성/재생성
├── handle_make_tweet()              @listen(or_) - LLM으로 트윗 생성/재생성
├── handle_make_linkedin_post()      @listen(or_) - LLM으로 LinkedIn 생성/재생성
├── check_seo()                      @listen      - SeoCrew로 SEO 평가
├── check_virality()                 @listen(or_) - ViralityCrew로 바이럴 평가
├── score_router()                   @router(or_) - 점수 기반 통과/루프백
└── finalize_content()               @listen      - 최종 결과 출력 및 반환
```

### 핵심 파일 관계

```
main.py ──imports──> tools.py (web_search_tool)
   |
   ├──imports──> seo_crew.py (SeoCrew) ──> Score 모델 반환
   └──imports──> virality_crew.py (ViralityCrew) ──> Score 모델 반환
```

이 챕터를 통해 CrewAI Flow의 모든 핵심 기능을 학습하고, 실제 프로덕션에서 활용 가능한 콘텐츠 생성 파이프라인을 완성했습니다. Flow는 단순한 워크플로우 도구가 아니라, LLM, Agent, Crew를 유기적으로 연결하는 **오케스트레이션 프레임워크**라는 점을 기억하세요.
