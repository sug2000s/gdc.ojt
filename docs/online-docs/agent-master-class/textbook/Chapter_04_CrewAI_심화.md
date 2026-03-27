# Chapter 4: CrewAI 심화 - Job Hunter Agent 구축

---

## 챕터 개요

이번 챕터에서는 CrewAI 프레임워크를 활용하여 **실제 실무에서 사용할 수 있는 복합 AI 에이전트 시스템**을 구축한다. 단일 에이전트가 아닌, 여러 에이전트가 **역할을 분담**하고 **작업 결과를 서로 전달**하며 협업하는 구조를 설계하고 구현하는 과정을 단계별로 학습한다.

프로젝트의 주제는 **"Job Hunter Agent"** 로, 다음과 같은 워크플로우를 자동화한다:

1. 웹에서 채용 공고를 검색하고 추출
2. 사용자의 이력서와 채용 공고를 매칭하여 점수 부여
3. 최적의 채용 공고를 선택
4. 선택된 공고에 맞춰 이력서를 재작성
5. 해당 회사를 조사
6. 면접 준비 자료를 생성

이 챕터를 통해 학습하는 핵심 기술:

| 기술 | 설명 |
|------|------|
| **Agent 정의 (YAML)** | 역할(role), 목표(goal), 배경스토리(backstory)를 YAML로 선언적 관리 |
| **Task 정의 (YAML)** | 작업 설명, 기대 출력, 담당 에이전트를 YAML로 구성 |
| **Structured Output (Pydantic)** | 에이전트의 출력을 Pydantic 모델로 강제하여 구조화 |
| **Context 전달** | 선행 Task의 결과를 후속 Task에 자동 전달 |
| **커스텀 Tool (Firecrawl)** | 외부 API를 활용한 웹 검색/스크래핑 도구 제작 |
| **Knowledge Source** | 텍스트 파일(이력서)을 에이전트의 지식으로 주입 |

---

## 4.1 Agents와 Tasks 정의

### 주제 및 목표

CrewAI의 핵심 구성 요소인 **Agent**와 **Task**를 YAML 설정 파일로 정의하는 방법을 학습한다. 이 단계에서는 코드를 작성하기 전에 먼저 **"누가(Agent) 무엇을(Task) 할 것인가"** 를 설계한다.

### 핵심 개념 설명

#### 프로젝트 구조

```
job-hunter-agent/
├── config/
│   ├── agents.yaml      # 에이전트 정의
│   └── tasks.yaml       # 작업 정의
├── knowledge/
│   └── resume.txt       # 사용자 이력서 (지식 소스)
├── main.py              # 메인 실행 파일 (이 단계에서는 비어있음)
├── pyproject.toml       # 프로젝트 의존성
└── output/              # 결과물 출력 디렉토리
```

CrewAI는 **선언적 접근 방식**을 채택한다. 에이전트와 작업을 코드가 아닌 YAML 파일로 정의함으로써:
- 비개발자도 에이전트의 행동을 이해하고 수정할 수 있다
- 코드와 설정을 분리하여 유지보수성을 높인다
- 에이전트의 역할과 목표를 명확하게 문서화한다

#### Agent 정의의 핵심 요소

CrewAI에서 Agent는 세 가지 핵심 속성으로 정의된다:

| 속성 | 역할 | 예시 |
|------|------|------|
| `role` | 에이전트의 직책/역할 | "Senior Job Market Research Specialist" |
| `goal` | 에이전트가 달성해야 할 목표 | "Discover and analyze relevant job opportunities..." |
| `backstory` | 에이전트의 전문성과 배경 | "You are an experienced talent acquisition specialist with 12+ years..." |

`backstory`는 단순한 설명이 아니라, LLM이 해당 역할에 맞는 **전문적인 판단**을 내리도록 유도하는 **프롬프트 엔지니어링**의 핵심이다.

### 코드 분석

#### agents.yaml - 5개 에이전트 정의

이 프로젝트에서는 5개의 전문 에이전트를 정의한다:

**1) 채용 공고 검색 에이전트 (job_search_agent)**

```yaml
job_search_agent:
  role: >
    Senior Job Market Research Specialist
  goal: >
    Discover and analyze relevant job opportunities from major job platforms that
    match the user's skills, experience level, and career preferences, providing
    detailed job information and market insights for optimal application strategy
  backstory: >
    You are an experienced talent acquisition specialist with 12+ years in recruitment
    and job market analysis. You have deep expertise in navigating job boards,
    understanding hiring trends, and identifying the best opportunities for candidates
    across various industries. You excel at reading between the lines of job descriptions
    to understand what employers really want, and you have a keen eye for spotting red
    flags in job postings. Your background includes working with both startups and
    Fortune 500 companies, giving you insight into different hiring cultures and
    expectations.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

> **포인트:** `backstory`에 "12+ years in recruitment"라는 구체적인 경력과 "reading between the lines of job descriptions"같은 세부 능력을 명시함으로써, LLM이 단순 검색이 아닌 **전문가적 분석**을 수행하도록 유도한다.

**2) 직무 매칭 에이전트 (job_matching_agent)**

```yaml
job_matching_agent:
  role: >
    Job Matching Expert
  goal: >
    Evaluate a list of extracted jobs and the user's resume to determine how well
    each opportunity aligns with the candidate's skills, preferences, and career goals.
    Provide match scores and rationales to guide the user toward the best-fit roles.
  backstory: >
    You are an intelligent job match evaluator trained on thousands of hiring decisions
    and successful placements. You analyze roles based on hard skills, soft skills, work
    preferences, and red flags mentioned in resumes. You understand that not all job
    titles are created equal, and that the fit depends on nuanced alignment between a
    candidate's profile and the opportunity's true requirements. Your job is to score
    each opportunity from 1 to 5 and justify that score clearly.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

**3) 이력서 최적화 에이전트 (resume_optimization_agent)**

```yaml
resume_optimization_agent:
  role: >
    Resume Optimization Specialist
  goal: >
    Rewrite and tailor the user's resume to closely match the selected job opportunity,
    increasing their chances of landing an interview.
  backstory: >
    You are a seasoned resume expert and former recruiter who has reviewed thousands of
    applications across tech, finance, and creative industries. You know exactly how to
    align a candidate's background to what employers are looking for. You understand how
    to optimize resumes with ATS-friendly keywords, clear summaries, and industry-relevant
    framing.
  verbose: true
  respect_context_window: true
  llm: openai/o4-mini-2025-04-16
```

> **포인트:** `respect_context_window: true` 옵션은 에이전트가 LLM의 컨텍스트 윈도우 크기를 존중하도록 하여, 너무 긴 입력으로 인한 오류를 방지한다. 이력서 재작성처럼 많은 텍스트를 다루는 에이전트에 특히 유용하다.

**4) 회사 조사 에이전트 (company_research_agent)**

```yaml
company_research_agent:
  role: >
    Company Research and Interview Strategist
  goal: >
    Help candidates deeply understand the company they are applying to and anticipate
    key interview themes.
  backstory: >
    You are a hybrid of a recruiter, career coach, and market analyst. You've advised
    thousands of job seekers on how to position themselves based on company signals,
    mission alignment, and role structure.
  verbose: true
  respect_context_window: true
  llm: openai/o4-mini-2025-04-16
```

**5) 면접 준비 에이전트 (interview_prep_agent)**

```yaml
interview_prep_agent:
  role: >
    Interview Strategist and Preparation Coach
  goal: >
    Generate a sharp, confident, and well-informed briefing for a job interview using
    all available assets.
  backstory: >
    You are a former head of talent, now an elite interview coach for engineers and
    product teams. You specialize in converting raw candidate and company data into
    clear interview strategies.
  verbose: true
  llm: openai/o4-mini-2025-04-16
```

#### tasks.yaml - 6개 작업 정의

각 Task는 `description`, `expected_output`, `agent` 세 가지 핵심 필드로 구성된다.

**1) 채용 공고 추출 작업 (job_extraction_task)**

```yaml
job_extraction_task:
  description: >
    Find and extract {level} level {position} jobs in {location}.

    Steps include:
    1. Use Web Search Tool to search for {level} level {position} jobs in {location}.
    2. Extract the job listings from the search results.
    3. Filter out job listings that are not {level} level {position} jobs in {location}.
  expected_output: >
    A JSON object matching the `JobList` schema.
  agent: job_search_agent
```

> **포인트:** `{level}`, `{position}`, `{location}`은 **템플릿 변수**다. 실행 시 `kickoff(inputs={...})`를 통해 실제 값으로 치환된다. 이를 통해 하나의 Task 정의로 다양한 검색 조건을 처리할 수 있다.

**2) 직무 매칭 작업 (job_matching_task)**

```yaml
job_matching_task:
  description: >
    Given a list of extracted jobs (JobList) and the user's resume, evaluate how well
    each job aligns with the user's:
    - Tech stack
    - Role level
    - Industry and company size preferences
    - Remote/work flexibility
    - Contract type
    - Salary expectations
    - Keywords and disqualifiers in the resume

    For each job, assign a `match_score` from 1 (poor fit) to 5 (perfect fit), and
    explain your reasoning.
  expected_output: >
    A JSON object matching the original `Job` schema, with two additional fields per job:
    - match_score: integer from 1 to 5
    - reason: a short explanation for the score
  agent: job_matching_agent
```

**3) 직무 선택 작업 (job_selection_task)**

```yaml
job_selection_task:
  description: >
    Given a list of jobs that each contain a `match_score` and a `reason` field
    (RankedJobList), your task is to:
    1. Analyze the `match_score` and reasons to determine the best-fit job.
    2. Select the single best job.
    3. Justify your choice.
    4. Set the `selected` field to `true` for the top job and `false` for all others.
  expected_output: >
    A JSON object matching the `ChosenJob` schema of the selected job.
  agent: job_matching_agent
```

**4) 이력서 재작성 작업 (resume_rewriting_task)**

```yaml
resume_rewriting_task:
  description: >
    Given the user's real resume and the selected job (ChosenJob), rewrite the existing
    resume to emphasize alignment with the job, without fabricating or inflating any facts.
  expected_output: >
    A Markdown-formatted version of the real user's resume, rewritten and optimized for
    the selected job.
  agent: resume_optimization_agent
  output_file: output/rewritten_resume.md
  create_directory: true
  markdown: true
```

> **포인트:** `output_file`을 지정하면 Task의 결과가 자동으로 파일에 저장된다. `create_directory: true`는 출력 디렉토리가 없을 경우 자동 생성한다.

**5) 회사 조사 작업 (company_research_task)**

```yaml
company_research_task:
  description: >
    Given the selected job (ChosenJob), research the hiring company using public web
    resources.
  expected_output: >
    A Markdown file with the following sections:
    - ## Company Overview
    - ## Mission and Values
    - ## Recent News or Changes
    - ## Role Context and Product Involvement
    - ## Likely Interview Topics
    - ## Suggested Questions to Ask
  agent: company_research_agent
  markdown: true
  output_file: output/company_research.md
```

**6) 면접 준비 작업 (interview_prep_task)**

```yaml
interview_prep_task:
  description: >
    Combine the following information:
    1. The selected job (ChosenJob)
    2. The tailored resume (RewrittenResume)
    3. The company research summary (CompanyResearch)

    Create a detailed interview preparation document.
  expected_output: >
    A Markdown document titled "Interview Prep: $CompanyName - $JobTitle" with sections:
    - ## Job Overview
    - ## Why This Job Is a Fit
    - ## Resume Highlights for This Role
    - ## Company Summary
    - ## Predicted Interview Questions
    - ## Questions to Ask Them
    - ## Concepts To Know/Review
    - ## Strategic Advice
  agent: interview_prep_agent
  output_file: output/interview_prep.md
```

#### 의존성 설정 (pyproject.toml)

```toml
[project]
name = "job-hunter-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "crewai[tools]>=0.152.0",
    "firecrawl-py>=2.16.3",
    "python-dotenv>=1.1.1",
]
```

- `crewai[tools]`: CrewAI 프레임워크와 내장 도구 모음
- `firecrawl-py`: 웹 검색 및 스크래핑 API 클라이언트
- `python-dotenv`: `.env` 파일에서 환경 변수(API 키 등)를 로드

#### Knowledge Source (resume.txt)

사용자의 이력서를 텍스트 파일로 저장하여 에이전트의 **지식 소스**로 활용한다:

```text
# Juan Srisaiwong
**Full Stack Developer**

## Professional Summary
Passionate Full Stack Developer with 3 years of experience building scalable web
applications and modern user interfaces. Proficient in JavaScript ecosystem with
expertise in React, Node.js, and cloud technologies.

## Technical Skills
Frontend: React, Vue.js, HTML5, CSS3, Sass, TypeScript, JavaScript (ES6+)
Backend: Node.js, Express.js, Python, Django, RESTful APIs, GraphQL
Databases: PostgreSQL, MySQL, MongoDB, Redis
Cloud & DevOps: AWS (EC2, S3, Lambda), Docker, CI/CD, Git, GitHub Actions
...
```

### 실습 포인트

1. **에이전트 설계 원칙**: 각 에이전트는 하나의 명확한 전문 분야를 가져야 한다. "모든 것을 다 하는 에이전트"보다 "한 가지를 잘 하는 에이전트 여러 개"가 더 좋은 결과를 낸다.
2. **backstory 작성법**: 구체적인 경력 연수, 특정 전문 분야, 업무 스타일을 포함하면 LLM의 출력 품질이 크게 향상된다.
3. **Task 설명의 구체성**: Task의 `description`은 에이전트에게 주어지는 **실질적인 프롬프트**다. 단계별 지침, 평가 기준, 제약 조건을 명확히 기술해야 한다.
4. **템플릿 변수 활용**: `{level}`, `{position}` 같은 변수를 사용하면 동일한 설정으로 다양한 시나리오를 처리할 수 있다.

---

## 4.2 Context와 Structured Outputs

### 주제 및 목표

이 섹션에서는 두 가지 핵심 기능을 구현한다:
1. **Structured Output**: Pydantic 모델을 사용하여 에이전트의 출력을 **구조화된 데이터**로 강제
2. **Context 전달**: 선행 Task의 결과를 후속 Task에 자동으로 전달하는 **의존성 체인** 구성

### 핵심 개념 설명

#### Structured Output이란?

LLM은 기본적으로 자유 형식의 텍스트를 생성한다. 하지만 프로그래밍 워크플로우에서는 **예측 가능한 구조**의 데이터가 필요하다. CrewAI는 Pydantic 모델을 통해 에이전트의 출력을 특정 스키마에 맞추도록 강제할 수 있다.

```
LLM 자유 출력:  "I found 3 jobs. The first one is..."  (파싱 불가)
구조화된 출력:   {"jobs": [{"job_title": "...", "company_name": "..."}]}  (프로그래밍 가능)
```

#### Context란?

CrewAI에서 `context`는 **Task 간의 데이터 흐름**을 정의한다. Task A의 결과를 Task B의 입력으로 전달할 때 사용한다.

```
job_extraction_task ─> job_matching_task ─> job_selection_task ─┬─> resume_rewriting_task ──┐
                                                                 ├─> company_research_task ──┤
                                                                 └───────────────────────────┴─> interview_prep_task
```

위 다이어그램에서:
- `resume_rewriting_task`는 `job_selection_task`의 결과(선택된 직무)를 context로 받는다
- `interview_prep_task`는 세 개 Task(job_selection, resume_rewriting, company_research)의 결과를 모두 context로 받는다

### 코드 분석

#### models.py - Pydantic 모델 정의

```python
from typing import List
from pydantic import BaseModel
from datetime import date


class Job(BaseModel):

    job_title: str
    company_name: str
    job_location: str
    is_remote_friendly: bool | None = None
    employment_type: str | None = None
    compensation: str | None = None
    job_posting_url: str
    job_summary: str

    key_qualifications: List[str] | None = None
    job_responsibilities: List[str] | None = None
    date_listed: date | None = None
    required_technologies: List[str] | None = None
    core_keywords: List[str] | None = None

    role_seniority_level: str | None = None
    years_of_experience_required: str | None = None
    minimum_education: str | None = None
    job_benefits: List[str] | None = None
    includes_equity: bool | None = None
    offers_visa_sponsorship: bool | None = None
    hiring_company_size: str | None = None
    hiring_industry: str | None = None
    source_listing_url: str | None = None
    full_raw_job_description: str | None = None


class JobList(BaseModel):
    jobs: List[Job]


class RankedJob(BaseModel):
    job: Job
    match_score: int
    reason: str


class RankedJobList(BaseModel):
    ranked_jobs: List[RankedJob]


class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
```

**모델 구조의 핵심 설계 원칙:**

1. **필수 필드와 선택 필드의 구분**: `job_title`, `company_name`은 항상 필요하지만, `includes_equity`, `offers_visa_sponsorship` 같은 정보는 모든 채용 공고에 있지 않다. `| None = None`으로 선택적 필드를 정의하여 유연성을 확보한다.

2. **점진적 데이터 확장 패턴**: 데이터가 파이프라인을 통과하면서 점점 풍부해진다:
   - `Job`: 기본 채용 정보
   - `RankedJob`: Job + 매칭 점수 + 이유 (평가 결과 추가)
   - `ChosenJob`: Job + 선택 여부 + 이유 (의사결정 결과 추가)

3. **컴포지션 패턴**: `RankedJob`은 `Job`을 상속하지 않고 **포함(composition)** 한다. 이렇게 하면 원본 Job 데이터를 변경하지 않으면서 추가 정보를 덧붙일 수 있다.

#### main.py - Crew 구성 및 실행

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, task, agent, crew
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool


@CrewBase
class JobHunterCrew:

    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config["job_search_agent"],
            tools=[web_search_tool],
        )

    @agent
    def job_matching_agent(self):
        return Agent(config=self.agents_config["job_matching_agent"])

    @agent
    def resume_optimization_agent(self):
        return Agent(config=self.agents_config["resume_optimization_agent"])

    @agent
    def company_research_agent(self):
        return Agent(config=self.agents_config["company_research_agent"])

    @agent
    def interview_prep_agent(self):
        return Agent(config=self.agents_config["interview_prep_agent"])
```

**`@CrewBase` 데코레이터의 역할:**

`@CrewBase`는 클래스에 다음 기능을 자동으로 부여한다:
- `self.agents_config`: `config/agents.yaml` 파일을 자동으로 로드
- `self.tasks_config`: `config/tasks.yaml` 파일을 자동으로 로드
- `self.agents`: `@agent` 데코레이터가 붙은 모든 메서드의 반환값을 리스트로 수집
- `self.tasks`: `@task` 데코레이터가 붙은 모든 메서드의 반환값을 리스트로 수집

**Task 정의와 Structured Output 연결:**

```python
    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,  # 출력을 JobList 스키마로 강제
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,  # 출력을 RankedJobList 스키마로 강제
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,  # 출력을 ChosenJob 스키마로 강제
        )
```

> **핵심:** `output_pydantic=JobList`를 지정하면, CrewAI가 LLM의 출력을 자동으로 `JobList` 스키마에 맞게 파싱한다. 스키마에 맞지 않는 출력이 생성되면 자동으로 재시도한다.

**Context를 통한 Task 간 데이터 전달:**

```python
    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),  # 선택된 직무 정보를 전달
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),  # 선택된 직무 정보를 전달
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),      # 선택된 직무
                self.resume_rewriting_task(),   # 재작성된 이력서
                self.company_research_task(),   # 회사 조사 결과
            ],
        )
```

> **핵심:** `context` 파라미터는 **리스트**로 여러 Task를 받을 수 있다. `interview_prep_task`는 3개 Task의 결과를 모두 참조하여 종합적인 면접 준비 자료를 생성한다. CrewAI는 context에 지정된 Task들이 완료된 후에야 해당 Task를 실행한다.

**Crew 조립 및 실행:**

```python
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,  # @agent 메서드들이 자동 수집됨
            tasks=self.tasks,    # @task 메서드들이 자동 수집됨
            verbose=True,        # 실행 과정을 상세히 출력
        )


JobHunterCrew().crew().kickoff()
```

`self.agents`와 `self.tasks`는 `@CrewBase`가 자동으로 생성하는 프로퍼티로, 각각 `@agent`와 `@task` 데코레이터가 붙은 모든 메서드의 반환값을 순서대로 수집한 리스트다.

### 실습 포인트

1. **Pydantic 모델 설계**: 필수 필드는 최소한으로, 선택 필드는 `| None = None`으로 설정하여 다양한 데이터 소스에 대응한다.
2. **Context 의존성 그래프**: Task 간의 의존성을 그래프로 그려보면 병렬 실행 가능한 Task를 식별할 수 있다. 예를 들어 `resume_rewriting_task`와 `company_research_task`는 동일한 context만 필요하므로 병렬 실행이 가능하다.
3. **`output_pydantic` vs `output_file`**: 구조화된 데이터가 필요하면 `output_pydantic`, 사람이 읽을 문서가 필요하면 `output_file`을 사용한다. 두 가지를 동시에 사용할 수도 있다.

---

## 4.3 Firecrawl Tool - 커스텀 웹 검색 도구

### 주제 및 목표

CrewAI 에이전트가 외부 세계와 상호작용할 수 있도록 **커스텀 Tool**을 제작한다. Firecrawl API를 활용하여 웹 검색을 수행하고, 결과를 정제하여 에이전트에게 전달하는 도구를 구현한다.

### 핵심 개념 설명

#### Tool이란?

CrewAI에서 Tool은 에이전트가 **외부 리소스에 접근**하거나 **특정 작업을 수행**할 수 있게 해주는 함수다. 에이전트는 LLM의 판단에 따라 적절한 시점에 Tool을 호출한다.

```
에이전트의 사고 과정:
1. "Senior level Golang Developer 채용 공고를 찾아야 해"
2. "web_search_tool을 사용하자"
3. web_search_tool("Senior Golang Developer jobs Netherlands") 호출
4. 결과를 분석하고 JobList 스키마에 맞게 정리
```

#### Firecrawl이란?

Firecrawl은 웹 페이지를 크롤링하고 내용을 **마크다운 형식**으로 변환하는 API 서비스다. 일반적인 웹 스크래핑과 달리:
- JavaScript 렌더링 지원 (SPA 페이지도 크롤링 가능)
- 자동으로 불필요한 요소(광고, 네비게이션 등) 제거
- 깨끗한 마크다운 텍스트로 변환

### 코드 분석

#### tools.py - 웹 검색 도구 구현

```python
import os, re

from crewai.tools import tool
from firecrawl import FirecrawlApp, ScrapeOptions


@tool
def web_search_tool(query: str):
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

**코드 상세 분석:**

1. **`@tool` 데코레이터**: CrewAI의 도구 데코레이터로, 일반 함수를 에이전트가 사용할 수 있는 Tool로 변환한다. 함수의 이름과 docstring이 에이전트에게 도구 설명으로 전달된다.

2. **FirecrawlApp 초기화**: `.env` 파일에서 `FIRECRAWL_API_KEY`를 읽어 인증한다.

3. **검색 실행**:
   ```python
   response = app.search(
       query=query,       # 검색어
       limit=5,           # 최대 5개 결과
       scrape_options=ScrapeOptions(
           formats=["markdown"],  # 마크다운 형식으로 반환
       ),
   )
   ```
   `limit=5`로 제한하는 이유는 LLM의 컨텍스트 윈도우를 절약하고, 너무 많은 정보로 인한 혼란을 방지하기 위함이다.

4. **결과 정제 (Data Cleaning)**:
   ```python
   # 불필요한 백슬래시와 연속 줄바꿈 제거
   cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
   # 마크다운 링크와 URL 제거 (토큰 절약)
   cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)
   ```
   웹 크롤링 결과에는 불필요한 링크, 특수 문자 등이 많다. 이를 제거하면:
   - LLM에 전달되는 **토큰 수를 절약**할 수 있다
   - 에이전트가 **핵심 내용에 집중**할 수 있다
   - API 비용을 절감할 수 있다

5. **반환값**: 제목, URL, 정제된 마크다운 내용을 포함하는 딕셔너리 리스트를 반환한다.

### 실습 포인트

1. **`@tool` 데코레이터 활용**: 어떤 Python 함수든 `@tool`을 붙이면 에이전트 도구로 사용할 수 있다. 데이터베이스 조회, 외부 API 호출, 파일 처리 등 다양한 용도로 활용 가능하다.
2. **데이터 정제의 중요성**: LLM에 전달되는 데이터는 깨끗할수록 좋다. 불필요한 HTML 태그, URL, 특수 문자를 제거하면 출력 품질이 향상된다.
3. **에러 핸들링**: `if not response.success`로 API 호출 실패를 처리한다. 프로덕션 환경에서는 더 정교한 에러 핸들링과 재시도 로직이 필요하다.
4. **결과 제한**: `limit=5`처럼 검색 결과 수를 제한하는 것은 비용과 품질의 균형을 맞추는 중요한 설계 결정이다.

---

## 4.5 Conclusions - Knowledge Source와 최종 실행

### 주제 및 목표

마지막 섹션에서는 다음을 완성한다:
1. **Knowledge Source** 연결: 이력서 텍스트 파일을 에이전트의 지식으로 주입
2. **Tool docstring 추가**: 에이전트가 도구를 올바르게 사용할 수 있도록 설명 추가
3. **실행 입력값 전달**: `kickoff(inputs={...})`로 템플릿 변수에 실제 값 전달
4. **결과 확인**: 전체 파이프라인 실행 및 출력 검증

### 핵심 개념 설명

#### Knowledge Source란?

CrewAI의 Knowledge Source는 에이전트에게 **사전 지식**을 제공하는 메커니즘이다. 텍스트 파일, PDF, CSV 등 다양한 형식의 데이터를 에이전트의 컨텍스트에 자동으로 포함시킨다.

일반적인 프롬프트에 직접 텍스트를 넣는 것과 다르게, Knowledge Source는:
- **벡터 데이터베이스(ChromaDB)** 를 사용하여 관련 정보를 검색
- 대용량 문서도 효율적으로 처리 가능
- 여러 에이전트 간에 동일한 지식을 공유 가능

#### Tool Docstring의 중요성

에이전트는 Tool의 **docstring을 읽고** 해당 도구를 언제, 어떻게 사용할지 판단한다. 명확한 docstring이 없으면 에이전트가 도구를 잘못 사용하거나 아예 사용하지 않을 수 있다.

### 코드 분석

#### main.py - Knowledge Source 추가 및 최종 구성

**Knowledge Source 설정:**

```python
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt",
    ]
)
```

`TextFileKnowledgeSource`는 텍스트 파일을 읽어 내부적으로 ChromaDB 벡터 데이터베이스에 임베딩으로 저장한다. 이후 에이전트가 관련 질문을 할 때 유사도 검색을 통해 적절한 정보를 자동으로 제공한다.

**에이전트에 Knowledge Source 연결:**

```python
@agent
def job_matching_agent(self):
    return Agent(
        config=self.agents_config["job_matching_agent"],
        knowledge_sources=[resume_knowledge],  # 이력서 지식 추가
    )

@agent
def resume_optimization_agent(self):
    return Agent(
        config=self.agents_config["resume_optimization_agent"],
        knowledge_sources=[resume_knowledge],  # 이력서 지식 추가
    )

@agent
def company_research_agent(self):
    return Agent(
        config=self.agents_config["company_research_agent"],
        knowledge_sources=[resume_knowledge],  # 이력서 지식 추가
        tools=[web_search_tool],               # 웹 검색 도구 추가
    )

@agent
def interview_prep_agent(self):
    return Agent(
        config=self.agents_config["interview_prep_agent"],
        knowledge_sources=[resume_knowledge],  # 이력서 지식 추가
    )
```

> **포인트:** 모든 에이전트가 동일한 `resume_knowledge` 인스턴스를 공유한다. 이를 통해:
> - `job_matching_agent`: 이력서 기반으로 채용 공고와의 매칭 점수 산출
> - `resume_optimization_agent`: 원본 이력서를 참조하여 재작성
> - `company_research_agent`: 이력서의 기술 스택을 고려한 회사 조사 + 웹 검색
> - `interview_prep_agent`: 이력서 내용을 반영한 면접 준비 자료 생성

**`company_research_agent`에만 `tools=[web_search_tool]`이 추가**된 점에 주목하자. 이 에이전트는 회사 정보를 웹에서 조사해야 하므로 검색 도구가 필요하다. 반면 `resume_optimization_agent`나 `interview_prep_agent`는 이미 context로 전달받은 정보만으로 충분하다.

#### tools.py - Docstring 추가

```python
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
    # ... (이하 동일)
```

docstring은 에이전트의 **도구 사용 가이드** 역할을 한다. 에이전트는 이 설명을 읽고:
- 어떤 인자를 전달해야 하는지 (`query: str`)
- 어떤 결과를 기대할 수 있는지 (마크다운 형식의 검색 결과 리스트)

를 이해하고 적절히 도구를 호출한다.

#### 실행 입력값 전달 및 결과 출력

```python
result = (
    JobHunterCrew()
    .crew()
    .kickoff(
        inputs={
            "level": "Senior",
            "position": "Golang Developer",
            "location": "Netherlands",
        }
    )
)

for task_output in result.tasks_output:
    print(task_output.pydantic)
```

`kickoff(inputs={...})`를 통해 YAML에 정의된 템플릿 변수가 치환된다:

```yaml
# tasks.yaml의 원본
description: >
  Find and extract {level} level {position} jobs in {location}.

# 실행 시 치환 결과
description: >
  Find and extract Senior level Golang Developer jobs in Netherlands.
```

`result.tasks_output`은 각 Task의 실행 결과를 순서대로 담고 있으며, `task_output.pydantic`으로 Pydantic 모델 인스턴스에 직접 접근할 수 있다.

#### 실행 결과 예시

전체 파이프라인이 실행되면 `output/` 디렉토리에 세 개의 마크다운 파일이 생성된다:

**1) output/rewritten_resume.md** - 선택된 직무(Senior Golang Developer)에 맞춰 재작성된 이력서:
- 직책을 "Full Stack Developer"에서 "Senior Backend Developer (API Design | Microservices | Cloud-Native)"로 변경
- Go 기반 FinTech 환경에 적합하도록 기술 스택 재구성
- 경력 기술을 API, 마이크로서비스, 성능 최적화 중심으로 재작성

**2) output/company_research.md** - 선택된 회사(FinTech Innovators)에 대한 조사 보고서:
- 회사 개요, 미션과 가치관, 최근 뉴스
- 역할의 기술 스택 분석 (Go, Kafka, Kubernetes, Terraform 등)
- 예상 면접 주제와 지원자가 물어볼 질문 목록

**3) output/interview_prep.md** - 종합 면접 준비 문서:
- 직무 개요와 적합성 분석
- 이력서 하이라이트
- 예상 면접 질문 (Golang, API 설계, 이벤트 기반 아키텍처 등)
- 전략적 조언 (자신감 있는 학습자 태도, 해결 지향적 접근 등)

### 실습 포인트

1. **Knowledge Source 활용**: 이력서, 회사 정보, 제품 문서 등 에이전트에게 필요한 배경 지식을 파일로 제공할 수 있다. 벡터 데이터베이스를 통해 대용량 문서도 효율적으로 처리된다.
2. **Tool docstring**: 에이전트가 도구를 올바르게 사용하려면 명확한 docstring이 필수다. 인자 설명, 반환값 형식, 사용 시나리오를 포함하자.
3. **입력값 템플릿**: `kickoff(inputs={...})`를 통해 동일한 에이전트 시스템을 다양한 조건으로 재사용할 수 있다.
4. **결과 접근**: `result.tasks_output`으로 각 Task의 결과에 프로그래밍적으로 접근하여 후처리할 수 있다.

---

## 챕터 핵심 정리

### 1. CrewAI의 선언적 구조

| 구성 요소 | 정의 위치 | 역할 |
|-----------|-----------|------|
| Agent | `config/agents.yaml` | 역할, 목표, 배경 스토리 정의 |
| Task | `config/tasks.yaml` | 작업 설명, 기대 출력, 담당 에이전트 지정 |
| Crew | `main.py` (`@CrewBase`) | 에이전트와 작업을 조합하여 실행 |

### 2. 데이터 흐름 제어

- **Structured Output (`output_pydantic`)**: Pydantic 모델로 에이전트 출력을 구조화
- **Context**: `context=[task_a(), task_b()]`로 Task 간 데이터 의존성 정의
- **Knowledge Source**: 텍스트 파일 등을 벡터 DB에 저장하여 에이전트의 사전 지식으로 활용

### 3. 커스텀 Tool 제작

- `@tool` 데코레이터로 Python 함수를 에이전트 도구로 변환
- 명확한 docstring으로 에이전트의 올바른 도구 사용 유도
- 외부 API 결과는 반드시 정제(cleaning)하여 토큰 효율성 확보

### 4. 멀티 에이전트 협업 패턴

```
검색 Agent ──> 매칭 Agent ──> 선택 Agent ──┬──> 이력서 Agent (context: 선택 결과)
                                            ├──> 회사 조사 Agent (context: 선택 결과)
                                            └──> 면접 준비 Agent (context: 선택+이력서+조사)
```

- 각 에이전트는 **단일 책임 원칙**을 따른다
- 순차 실행이 필요한 Task와 병렬 실행 가능한 Task를 구분한다
- context를 통해 정보가 자연스럽게 흐르도록 설계한다

### 5. 실전 설계 팁

- **backstory는 길수록 좋다**: 구체적인 경력, 전문 분야, 업무 스타일을 상세히 기술하면 LLM이 더 전문적인 출력을 생성한다
- **필드는 관대하게, 출력은 엄격하게**: Pydantic 모델에서 선택적 필드(`| None = None`)를 충분히 두되, 핵심 필드는 필수로 지정한다
- **토큰 절약**: 웹 크롤링 결과의 불필요한 링크, 특수문자를 제거하여 API 비용을 절감한다
- **`respect_context_window: true`**: 많은 텍스트를 다루는 에이전트에 설정하여 컨텍스트 윈도우 초과 오류를 방지한다

---

## 실습 과제

### 과제 1: 에이전트 역할 커스터마이징 (난이도: 하)

`agents.yaml`의 에이전트 설정을 수정하여 **다른 직군(예: 디자이너, 마케터)** 에 적합한 Job Hunter Agent를 만들어보자.

**요구사항:**
- `job_search_agent`의 backstory를 해당 직군의 채용 시장에 맞게 수정
- `resume_optimization_agent`의 backstory를 해당 직군의 이력서 작성 관행에 맞게 수정
- `knowledge/resume.txt`를 새로운 이력서로 교체
- `kickoff(inputs={...})`의 입력값을 변경하여 실행

### 과제 2: 새로운 Pydantic 모델 추가 (난이도: 중)

현재 `ChosenJob` 모델에 다음 필드를 추가하고, 관련 Task의 description과 expected_output을 수정해보자:

```python
class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
    # 새로 추가할 필드
    salary_competitiveness: str       # "above_market", "at_market", "below_market"
    career_growth_potential: int      # 1-5 점수
    work_life_balance_score: int      # 1-5 점수
    recommended_negotiation_points: List[str]  # 협상 포인트 목록
```

### 과제 3: 새로운 Tool 제작 (난이도: 중)

Firecrawl 대신 다른 API를 사용하는 커스텀 Tool을 만들어보자. 예시:

```python
@tool
def glassdoor_review_tool(company_name: str):
    """
    Glassdoor Review Tool.
    Args:
        company_name: str
            The company name to search reviews for.
    Returns:
        A summary of employee reviews for the company.
    """
    # 구현해보기
    pass
```

**힌트:** SerpAPI, Google Custom Search API 등을 활용하여 회사 리뷰를 검색하고 정제하는 로직을 구현한다.

### 과제 4: 병렬 실행 최적화 (난이도: 상)

현재 파이프라인에서 `resume_rewriting_task`와 `company_research_task`는 동일한 context(`job_selection_task`)만 필요하므로 이론적으로 병렬 실행이 가능하다. CrewAI의 `Process.hierarchical` 또는 비동기 실행 기능을 연구하여, 이 두 Task를 병렬로 실행하도록 수정해보자.

```python
@crew
def crew(self):
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        verbose=True,
        process=Process.hierarchical,  # 계층적 실행 모드
        manager_llm="openai/gpt-4o",   # 관리자 LLM
    )
```

### 과제 5: 전체 파이프라인 확장 (난이도: 상)

새로운 에이전트와 Task를 추가하여 파이프라인을 확장해보자:

1. **salary_negotiation_agent**: 선택된 직무의 급여 범위를 분석하고 협상 전략을 제시하는 에이전트
2. **cover_letter_agent**: 이력서와 회사 조사 결과를 바탕으로 커버레터를 자동 생성하는 에이전트

각 에이전트에 대해:
- `agents.yaml`에 role, goal, backstory를 정의
- `tasks.yaml`에 description, expected_output을 정의
- `models.py`에 필요한 Pydantic 모델을 추가
- `main.py`에 `@agent`, `@task` 메서드를 추가하고 적절한 context를 설정

---

## 참고: 전체 최종 코드

아래는 이 챕터에서 완성한 모든 핵심 파일의 최종 상태다.

### main.py (최종)

```python
import dotenv

dotenv.load_dotenv()

from crewai import Crew, Agent, Task
from crewai.project import CrewBase, task, agent, crew
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from models import JobList, RankedJobList, ChosenJob
from tools import web_search_tool

resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt",
    ]
)


@CrewBase
class JobHunterCrew:

    @agent
    def job_search_agent(self):
        return Agent(
            config=self.agents_config["job_search_agent"],
            tools=[web_search_tool],
        )

    @agent
    def job_matching_agent(self):
        return Agent(
            config=self.agents_config["job_matching_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @agent
    def resume_optimization_agent(self):
        return Agent(
            config=self.agents_config["resume_optimization_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @agent
    def company_research_agent(self):
        return Agent(
            config=self.agents_config["company_research_agent"],
            knowledge_sources=[resume_knowledge],
            tools=[web_search_tool],
        )

    @agent
    def interview_prep_agent(self):
        return Agent(
            config=self.agents_config["interview_prep_agent"],
            knowledge_sources=[resume_knowledge],
        )

    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,
        )

    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),
                self.resume_rewriting_task(),
                self.company_research_task(),
            ],
        )

    @crew
    def crew(self):
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
        )


result = (
    JobHunterCrew()
    .crew()
    .kickoff(
        inputs={
            "level": "Senior",
            "position": "Golang Developer",
            "location": "Netherlands",
        }
    )
)

for task_output in result.tasks_output:
    print(task_output.pydantic)
```

### models.py (최종)

```python
from typing import List
from pydantic import BaseModel
from datetime import date


class Job(BaseModel):
    job_title: str
    company_name: str
    job_location: str
    is_remote_friendly: bool | None = None
    employment_type: str | None = None
    compensation: str | None = None
    job_posting_url: str
    job_summary: str
    key_qualifications: List[str] | None = None
    job_responsibilities: List[str] | None = None
    date_listed: date | None = None
    required_technologies: List[str] | None = None
    core_keywords: List[str] | None = None
    role_seniority_level: str | None = None
    years_of_experience_required: str | None = None
    minimum_education: str | None = None
    job_benefits: List[str] | None = None
    includes_equity: bool | None = None
    offers_visa_sponsorship: bool | None = None
    hiring_company_size: str | None = None
    hiring_industry: str | None = None
    source_listing_url: str | None = None
    full_raw_job_description: str | None = None


class JobList(BaseModel):
    jobs: List[Job]


class RankedJob(BaseModel):
    job: Job
    match_score: int
    reason: str


class RankedJobList(BaseModel):
    ranked_jobs: List[RankedJob]


class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
```

### tools.py (최종)

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
