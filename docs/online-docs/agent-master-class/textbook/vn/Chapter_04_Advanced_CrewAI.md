# Chapter 4: CrewAI Nâng cao - Xây dựng Job Hunter Agent

---

## Tổng quan chương

Trong chương này, chúng ta sử dụng framework CrewAI để xây dựng **hệ thống AI agent phức hợp có thể áp dụng trong thực tế**. Không chỉ là một agent đơn lẻ, chúng ta thiết kế và triển khai cấu trúc nơi nhiều agent **phân chia vai trò** và **truyền kết quả công việc cho nhau** để hợp tác, học từng bước quá trình này.

Chủ đề dự án là **"Job Hunter Agent"**, tự động hóa quy trình sau:

1. Tìm kiếm và trích xuất tin tuyển dụng từ web
2. Đối chiếu hồ sơ người dùng với tin tuyển dụng và chấm điểm
3. Chọn tin tuyển dụng tối ưu
4. Viết lại hồ sơ phù hợp với tin tuyển dụng đã chọn
5. Nghiên cứu công ty
6. Tạo tài liệu chuẩn bị phỏng vấn

Kỹ năng chính được học trong chương này:

| Kỹ năng | Mô tả |
|---------|-------|
| **Định nghĩa Agent (YAML)** | Quản lý khai báo role, goal, backstory trong YAML |
| **Định nghĩa Task (YAML)** | Cấu hình mô tả tác vụ, đầu ra mong đợi, agent phụ trách trong YAML |
| **Structured Output (Pydantic)** | Ép buộc đầu ra agent tuân theo mô hình Pydantic để cấu trúc hóa |
| **Truyền Context** | Tự động truyền kết quả Task trước sang Task sau |
| **Tool tùy chỉnh (Firecrawl)** | Tạo công cụ tìm kiếm/thu thập web sử dụng API bên ngoài |
| **Knowledge Source** | Đưa file văn bản (hồ sơ) vào làm kiến thức cho agent |

---

## 4.1 Định nghĩa Agents và Tasks

### Chủ đề và mục tiêu

Học cách định nghĩa các thành phần cốt lõi của CrewAI -- **Agent** và **Task** -- bằng file cấu hình YAML. Ở giai đoạn này, trước khi viết code, chúng ta thiết kế trước **"ai (Agent) sẽ làm gì (Task)."**

### Giải thích khái niệm cốt lõi

#### Cấu trúc dự án

```
job-hunter-agent/
├── config/
│   ├── agents.yaml      # Định nghĩa agent
│   └── tasks.yaml       # Định nghĩa task
├── knowledge/
│   └── resume.txt       # Hồ sơ người dùng (nguồn kiến thức)
├── main.py              # File thực thi chính (trống ở giai đoạn này)
├── pyproject.toml       # Phụ thuộc dự án
└── output/              # Thư mục đầu ra kết quả
```

CrewAI áp dụng **phương pháp khai báo**. Bằng cách định nghĩa agent và task trong file YAML thay vì code:
- Người không phải developer cũng có thể hiểu và sửa hành vi agent
- Tách biệt code và cấu hình cải thiện khả năng bảo trì
- Vai trò và mục tiêu agent được tài liệu hóa rõ ràng

#### Yếu tố cốt lõi của định nghĩa Agent

Trong CrewAI, Agent được định nghĩa bởi ba thuộc tính cốt lõi:

| Thuộc tính | Vai trò | Ví dụ |
|-----------|--------|-------|
| `role` | Chức danh/vai trò của agent | "Senior Job Market Research Specialist" |
| `goal` | Mục tiêu agent phải đạt được | "Discover and analyze relevant job opportunities..." |
| `backstory` | Chuyên môn và nền tảng của agent | "You are an experienced talent acquisition specialist with 12+ years..." |

`backstory` không phải mô tả đơn thuần -- đây là cốt lõi của **prompt engineering** hướng dẫn LLM đưa ra **phán đoán cấp chuyên gia** phù hợp với vai trò.

### Phân tích code

#### agents.yaml - Định nghĩa 5 Agent

Dự án này định nghĩa 5 agent chuyên biệt:

**1) Agent tìm kiếm việc làm (job_search_agent)**

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

> **Điểm chính:** Bằng cách chỉ định kinh nghiệm cụ thể như "12+ years in recruitment" và khả năng chi tiết như "reading between the lines of job descriptions" trong `backstory`, LLM được hướng dẫn thực hiện **phân tích cấp chuyên gia** thay vì tìm kiếm đơn giản.

**2) Agent đối sánh việc làm (job_matching_agent)**

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

**3) Agent tối ưu hồ sơ (resume_optimization_agent)**

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

> **Điểm chính:** Tùy chọn `respect_context_window: true` đảm bảo agent tôn trọng kích thước cửa sổ ngữ cảnh LLM, ngăn lỗi từ đầu vào quá dài. Đặc biệt hữu ích cho agent xử lý nhiều văn bản như viết lại hồ sơ.

**4) Agent nghiên cứu công ty (company_research_agent)**

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

**5) Agent chuẩn bị phỏng vấn (interview_prep_agent)**

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

#### tasks.yaml - Định nghĩa 6 Task

Mỗi Task gồm ba trường cốt lõi: `description`, `expected_output`, và `agent`.

**1) Task trích xuất việc làm (job_extraction_task)**

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

> **Điểm chính:** `{level}`, `{position}`, và `{location}` là **biến mẫu**. Chúng được thay thế bằng giá trị thực qua `kickoff(inputs={...})` khi chạy. Điều này cho phép một định nghĩa Task xử lý nhiều điều kiện tìm kiếm khác nhau.

**2) Task đối sánh việc làm (job_matching_task)**

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

**3) Task chọn việc làm (job_selection_task)**

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

**4) Task viết lại hồ sơ (resume_rewriting_task)**

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

> **Điểm chính:** Khi chỉ định `output_file`, kết quả Task tự động được lưu vào file. `create_directory: true` tự động tạo thư mục đầu ra nếu chưa tồn tại.

**5) Task nghiên cứu công ty (company_research_task)**

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

**6) Task chuẩn bị phỏng vấn (interview_prep_task)**

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

#### Cấu hình phụ thuộc (pyproject.toml)

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

- `crewai[tools]`: Framework CrewAI và bộ công cụ tích hợp
- `firecrawl-py`: Client API tìm kiếm và thu thập web
- `python-dotenv`: Tải biến môi trường (API key, v.v.) từ file `.env`

#### Knowledge Source (resume.txt)

Hồ sơ người dùng được lưu dưới dạng file văn bản và sử dụng làm **nguồn kiến thức** cho agent:

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

### Điểm thực hành

1. **Nguyên tắc thiết kế agent**: Mỗi agent nên có một lĩnh vực chuyên môn rõ ràng. "Nhiều agent mỗi cái làm tốt một việc" cho kết quả tốt hơn "một agent làm mọi thứ."
2. **Cách viết backstory**: Bao gồm số năm kinh nghiệm cụ thể, lĩnh vực chuyên môn riêng, phong cách làm việc cải thiện đáng kể chất lượng đầu ra LLM.
3. **Tính cụ thể của mô tả Task**: `description` của Task là **prompt thực tế** được giao cho agent. Hướng dẫn từng bước, tiêu chí đánh giá, ràng buộc phải được nêu rõ ràng.
4. **Sử dụng biến mẫu**: Sử dụng biến như `{level}`, `{position}` cho phép cùng cấu hình xử lý nhiều kịch bản khác nhau.

---

## 4.2 Context và Structured Outputs

### Chủ đề và mục tiêu

Trong phần này, chúng ta triển khai hai tính năng chính:
1. **Structured Output**: Sử dụng mô hình Pydantic để ép buộc đầu ra agent thành **dữ liệu có cấu trúc**
2. **Truyền Context**: Cấu hình **chuỗi phụ thuộc** tự động truyền kết quả Task trước sang Task sau

### Giải thích khái niệm cốt lõi

#### Structured Output là gì?

LLM về cơ bản tạo văn bản dạng tự do. Tuy nhiên, quy trình lập trình cần dữ liệu có **cấu trúc dự đoán được**. CrewAI có thể ép buộc đầu ra agent tuân theo schema cụ thể thông qua mô hình Pydantic.

```
Đầu ra tự do LLM:    "I found 3 jobs. The first one is..."  (không thể phân tích)
Đầu ra có cấu trúc:   {"jobs": [{"job_title": "...", "company_name": "..."}]}  (lập trình được)
```

#### Context là gì?

Trong CrewAI, `context` định nghĩa **luồng dữ liệu giữa các Task**. Được sử dụng khi truyền kết quả Task A làm đầu vào cho Task B.

```
job_extraction_task -> job_matching_task -> job_selection_task -+-> resume_rewriting_task --+
                                                                +-> company_research_task --+
                                                                +---------------------------+-> interview_prep_task
```

Trong sơ đồ trên:
- `resume_rewriting_task` nhận kết quả `job_selection_task` (việc làm đã chọn) làm context
- `interview_prep_task` nhận kết quả cả ba Task (job_selection, resume_rewriting, company_research) làm context

### Phân tích code

#### models.py - Định nghĩa mô hình Pydantic

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

**Nguyên tắc thiết kế cốt lõi của cấu trúc mô hình:**

1. **Phân biệt trường bắt buộc và tùy chọn**: `job_title`, `company_name` luôn cần thiết, nhưng thông tin như `includes_equity`, `offers_visa_sponsorship` không có trong mọi tin tuyển dụng. Định nghĩa trường tùy chọn bằng `| None = None` đảm bảo tính linh hoạt.

2. **Mẫu làm giàu dữ liệu dần dần**: Dữ liệu trở nên phong phú hơn khi đi qua pipeline:
   - `Job`: Thông tin tuyển dụng cơ bản
   - `RankedJob`: Job + điểm đối sánh + lý do (thêm kết quả đánh giá)
   - `ChosenJob`: Job + trạng thái đã chọn + lý do (thêm kết quả quyết định)

3. **Mẫu composition**: `RankedJob` không kế thừa từ `Job` mà **chứa (composition)** nó. Cách này cho phép thêm thông tin bổ sung mà không sửa đổi dữ liệu Job gốc.

#### main.py - Cấu hình và thực thi Crew

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

**Vai trò của decorator `@CrewBase`:**

`@CrewBase` tự động cung cấp các khả năng sau cho lớp:
- `self.agents_config`: Tự động tải file `config/agents.yaml`
- `self.tasks_config`: Tự động tải file `config/tasks.yaml`
- `self.agents`: Thu thập giá trị trả về của tất cả phương thức có decorator `@agent` thành danh sách
- `self.tasks`: Thu thập giá trị trả về của tất cả phương thức có decorator `@task` thành danh sách

**Định nghĩa Task và kết nối Structured Output:**

```python
    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,  # Ép đầu ra theo schema JobList
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,  # Ép đầu ra theo schema RankedJobList
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,  # Ép đầu ra theo schema ChosenJob
        )
```

> **Điểm chính:** Khi chỉ định `output_pydantic=JobList`, CrewAI tự động phân tích đầu ra LLM để tuân theo schema `JobList`. Nếu đầu ra không khớp schema, tự động thử lại.

**Truyền dữ liệu giữa các Task qua Context:**

```python
    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),  # Truyền thông tin việc làm đã chọn
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),  # Truyền thông tin việc làm đã chọn
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),      # Việc làm đã chọn
                self.resume_rewriting_task(),   # Hồ sơ đã viết lại
                self.company_research_task(),   # Kết quả nghiên cứu công ty
            ],
        )
```

> **Điểm chính:** Tham số `context` nhận **danh sách** nhiều Task. `interview_prep_task` tham chiếu kết quả cả 3 Task để tạo tài liệu chuẩn bị phỏng vấn toàn diện. CrewAI đợi tất cả Task được chỉ định trong context hoàn thành rồi mới thực thi Task đó.

**Tổ hợp và thực thi Crew:**

```python
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,  # Các phương thức @agent tự động thu thập
            tasks=self.tasks,    # Các phương thức @task tự động thu thập
            verbose=True,        # In chi tiết tiến trình thực thi
        )


JobHunterCrew().crew().kickoff()
```

`self.agents` và `self.tasks` là thuộc tính tự động được tạo bởi `@CrewBase`, thu thập giá trị trả về của tất cả phương thức có decorator `@agent` và `@task` theo thứ tự thành danh sách.

### Điểm thực hành

1. **Thiết kế mô hình Pydantic**: Giữ trường bắt buộc tối thiểu, thiết lập trường tùy chọn bằng `| None = None` để đối phó với các nguồn dữ liệu khác nhau.
2. **Đồ thị phụ thuộc Context**: Vẽ đồ thị phụ thuộc giữa các Task giúp xác định Task có thể chạy song song. Ví dụ, `resume_rewriting_task` và `company_research_task` chỉ cần cùng context nên có thể chạy song song.
3. **`output_pydantic` vs `output_file`**: Sử dụng `output_pydantic` khi cần dữ liệu có cấu trúc, `output_file` khi cần tài liệu con người đọc được. Có thể sử dụng cả hai cùng lúc.

---

## 4.3 Firecrawl Tool - Công cụ tìm kiếm web tùy chỉnh

### Chủ đề và mục tiêu

Tạo **Tool tùy chỉnh** cho phép agent CrewAI tương tác với thế giới bên ngoài. Triển khai công cụ thực hiện tìm kiếm web bằng Firecrawl API, làm sạch kết quả và chuyển cho agent.

### Giải thích khái niệm cốt lõi

#### Tool là gì?

Trong CrewAI, Tool là hàm cho phép agent **truy cập tài nguyên bên ngoài** hoặc **thực hiện thao tác cụ thể**. Agent gọi Tool vào thời điểm thích hợp dựa trên phán đoán của LLM.

```
Quá trình suy nghĩ của agent:
1. "Tôi cần tìm tin tuyển dụng Senior level Golang Developer"
2. "Hãy dùng web_search_tool"
3. web_search_tool("Senior Golang Developer jobs Netherlands") được gọi
4. Phân tích kết quả và sắp xếp theo schema JobList
```

#### Firecrawl là gì?

Firecrawl là dịch vụ API thu thập trang web và chuyển đổi nội dung thành **định dạng markdown**. Khác với thu thập web thông thường:
- Hỗ trợ render JavaScript (có thể thu thập trang SPA)
- Tự động loại bỏ phần tử không cần thiết (quảng cáo, navigation, v.v.)
- Chuyển đổi thành văn bản markdown sạch

### Phân tích code

#### tools.py - Triển khai công cụ tìm kiếm web

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

**Phân tích code chi tiết:**

1. **Decorator `@tool`**: Decorator công cụ của CrewAI chuyển đổi hàm thường thành Tool mà agent sử dụng được. Tên và docstring hàm được truyền cho agent làm mô tả công cụ.

2. **Khởi tạo FirecrawlApp**: Xác thực bằng cách đọc `FIRECRAWL_API_KEY` từ file `.env`.

3. **Thực thi tìm kiếm**:
   ```python
   response = app.search(
       query=query,       # Truy vấn tìm kiếm
       limit=5,           # Tối đa 5 kết quả
       scrape_options=ScrapeOptions(
           formats=["markdown"],  # Trả về dạng markdown
       ),
   )
   ```
   Lý do giới hạn bằng `limit=5` là để tiết kiệm cửa sổ ngữ cảnh LLM và ngăn nhầm lẫn từ quá nhiều thông tin.

4. **Làm sạch dữ liệu (Data Cleaning)**:
   ```python
   # Loại bỏ backslash thừa và xuống dòng liên tục
   cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
   # Loại bỏ link markdown và URL (tiết kiệm token)
   cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)
   ```
   Kết quả thu thập web chứa nhiều link và ký tự đặc biệt không cần thiết. Loại bỏ chúng:
   - **Tiết kiệm token** gửi cho LLM
   - Giúp agent **tập trung vào nội dung cốt lõi**
   - Giảm chi phí API

5. **Giá trị trả về**: Trả về danh sách từ điển chứa tiêu đề, URL và nội dung markdown đã làm sạch.

### Điểm thực hành

1. **Sử dụng decorator `@tool`**: Bất kỳ hàm Python nào cũng có thể dùng làm công cụ agent bằng cách thêm `@tool`. Có thể dùng cho nhiều mục đích như truy vấn database, gọi API bên ngoài, xử lý file.
2. **Tầm quan trọng của làm sạch dữ liệu**: Dữ liệu truyền cho LLM càng sạch càng tốt. Loại bỏ thẻ HTML, URL, ký tự đặc biệt thừa cải thiện chất lượng đầu ra.
3. **Xử lý lỗi**: `if not response.success` xử lý lỗi gọi API. Trong môi trường production cần logic xử lý lỗi và thử lại tinh vi hơn.
4. **Giới hạn kết quả**: Giới hạn kết quả tìm kiếm bằng `limit=5` là quyết định thiết kế quan trọng cân bằng chi phí và chất lượng.

---

## 4.5 Kết luận - Knowledge Source và thực thi cuối cùng

### Chủ đề và mục tiêu

Trong phần cuối, chúng ta hoàn thành:
1. **Knowledge Source** kết nối: Đưa file văn bản hồ sơ làm kiến thức cho agent
2. **Thêm docstring cho Tool**: Thêm mô tả để agent sử dụng công cụ đúng cách
3. **Truyền giá trị đầu vào**: Truyền giá trị thực cho biến mẫu qua `kickoff(inputs={...})`
4. **Xác minh kết quả**: Thực thi toàn bộ pipeline và kiểm tra đầu ra

### Giải thích khái niệm cốt lõi

#### Knowledge Source là gì?

Knowledge Source của CrewAI là cơ chế cung cấp **kiến thức sẵn có** cho agent. Tự động bao gồm dữ liệu từ nhiều định dạng như file văn bản, PDF, CSV, v.v. vào ngữ cảnh agent.

Khác với việc đưa văn bản trực tiếp vào prompt, Knowledge Source:
- Sử dụng **cơ sở dữ liệu vector (ChromaDB)** để tìm kiếm thông tin liên quan
- Xử lý hiệu quả tài liệu dung lượng lớn
- Cho phép chia sẻ cùng kiến thức giữa nhiều agent

#### Tầm quan trọng của Docstring cho Tool

Agent **đọc docstring của Tool** để xác định khi nào và cách sử dụng công cụ đó. Không có docstring rõ ràng, agent có thể sử dụng sai công cụ hoặc không sử dụng luôn.

### Phân tích code

#### main.py - Thêm Knowledge Source và cấu hình cuối cùng

**Thiết lập Knowledge Source:**

```python
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

resume_knowledge = TextFileKnowledgeSource(
    file_paths=[
        "resume.txt",
    ]
)
```

`TextFileKnowledgeSource` đọc file văn bản và nội bộ lưu trữ dưới dạng embedding trong cơ sở dữ liệu vector ChromaDB. Khi agent sau đó hỏi câu hỏi liên quan, thông tin phù hợp được tự động cung cấp qua tìm kiếm tương đồng.

**Kết nối Knowledge Source với Agent:**

```python
@agent
def job_matching_agent(self):
    return Agent(
        config=self.agents_config["job_matching_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức hồ sơ
    )

@agent
def resume_optimization_agent(self):
    return Agent(
        config=self.agents_config["resume_optimization_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức hồ sơ
    )

@agent
def company_research_agent(self):
    return Agent(
        config=self.agents_config["company_research_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức hồ sơ
        tools=[web_search_tool],               # Thêm công cụ tìm kiếm web
    )

@agent
def interview_prep_agent(self):
    return Agent(
        config=self.agents_config["interview_prep_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức hồ sơ
    )
```

> **Điểm chính:** Tất cả agent chia sẻ cùng instance `resume_knowledge`. Qua đó:
> - `job_matching_agent`: Tính điểm đối sánh giữa tin tuyển dụng và hồ sơ
> - `resume_optimization_agent`: Tham chiếu hồ sơ gốc để viết lại
> - `company_research_agent`: Xem xét tech stack hồ sơ trong nghiên cứu công ty + tìm kiếm web
> - `interview_prep_agent`: Tạo tài liệu chuẩn bị phỏng vấn phản ánh nội dung hồ sơ

**Lưu ý chỉ `company_research_agent` được thêm `tools=[web_search_tool]`.** Agent này cần công cụ tìm kiếm để nghiên cứu thông tin công ty từ web. Ngược lại, `resume_optimization_agent` và `interview_prep_agent` có đủ thông tin từ context đã được truyền.

#### tools.py - Thêm Docstring

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
    # ... (phần còn lại tương tự)
```

Docstring đóng vai trò **hướng dẫn sử dụng công cụ** cho agent. Agent đọc mô tả này và hiểu:
- Cần truyền đối số nào (`query: str`)
- Kết quả mong đợi là gì (danh sách kết quả tìm kiếm dạng markdown)

và gọi công cụ phù hợp.

#### Truyền giá trị đầu vào và xuất kết quả

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

Biến mẫu được định nghĩa trong YAML được thay thế qua `kickoff(inputs={...})`:

```yaml
# Bản gốc trong tasks.yaml
description: >
  Find and extract {level} level {position} jobs in {location}.

# Kết quả sau khi thay thế khi chạy
description: >
  Find and extract Senior level Golang Developer jobs in Netherlands.
```

`result.tasks_output` chứa kết quả thực thi mỗi Task theo thứ tự, và bạn có thể truy cập trực tiếp instance mô hình Pydantic qua `task_output.pydantic`.

#### Ví dụ kết quả thực thi

Khi toàn bộ pipeline chạy, ba file markdown được tạo trong thư mục `output/`:

**1) output/rewritten_resume.md** - Hồ sơ viết lại phù hợp việc làm đã chọn (Senior Golang Developer):
- Chức danh thay đổi từ "Full Stack Developer" thành "Senior Backend Developer (API Design | Microservices | Cloud-Native)"
- Tech stack tái cấu trúc phù hợp môi trường FinTech dựa trên Go
- Mô tả kinh nghiệm viết lại tập trung vào API, microservices và tối ưu hiệu suất

**2) output/company_research.md** - Báo cáo nghiên cứu công ty đã chọn (FinTech Innovators):
- Tổng quan công ty, sứ mệnh và giá trị, tin tức gần đây
- Phân tích tech stack của vai trò (Go, Kafka, Kubernetes, Terraform, v.v.)
- Chủ đề phỏng vấn dự kiến và câu hỏi ứng viên nên hỏi

**3) output/interview_prep.md** - Tài liệu chuẩn bị phỏng vấn toàn diện:
- Tổng quan việc làm và phân tích sự phù hợp
- Điểm nổi bật hồ sơ
- Câu hỏi phỏng vấn dự kiến (Golang, thiết kế API, kiến trúc hướng sự kiện, v.v.)
- Lời khuyên chiến lược (thái độ học hỏi tự tin, tiếp cận hướng giải pháp, v.v.)

### Điểm thực hành

1. **Sử dụng Knowledge Source**: Có thể cung cấp kiến thức nền cần thiết cho agent dưới dạng file -- hồ sơ, thông tin công ty, tài liệu sản phẩm, v.v. Tài liệu dung lượng lớn được xử lý hiệu quả qua cơ sở dữ liệu vector.
2. **Docstring cho Tool**: Docstring rõ ràng là bắt buộc để agent sử dụng công cụ đúng. Bao gồm mô tả đối số, định dạng giá trị trả về và kịch bản sử dụng.
3. **Mẫu đầu vào**: Qua `kickoff(inputs={...})`, có thể tái sử dụng cùng hệ thống agent dưới nhiều điều kiện khác nhau.
4. **Truy cập kết quả**: Sử dụng `result.tasks_output` để truy cập kết quả mỗi Task theo chương trình để xử lý hậu kỳ.

---

## Tóm tắt điểm chính của chương

### 1. Cấu trúc khai báo của CrewAI

| Thành phần | Vị trí định nghĩa | Vai trò |
|-----------|-------------------|--------|
| Agent | `config/agents.yaml` | Định nghĩa role, goal, backstory |
| Task | `config/tasks.yaml` | Định nghĩa mô tả tác vụ, đầu ra mong đợi, agent phụ trách |
| Crew | `main.py` (`@CrewBase`) | Kết hợp agent và task để thực thi |

### 2. Kiểm soát luồng dữ liệu

- **Structured Output (`output_pydantic`)**: Cấu trúc hóa đầu ra agent bằng mô hình Pydantic
- **Context**: Định nghĩa phụ thuộc dữ liệu giữa Task bằng `context=[task_a(), task_b()]`
- **Knowledge Source**: Lưu file văn bản v.v. trong vector DB và sử dụng làm kiến thức sẵn có cho agent

### 3. Tạo Tool tùy chỉnh

- Chuyển đổi hàm Python thành công cụ agent bằng decorator `@tool`
- Hướng dẫn sử dụng công cụ đúng bằng docstring rõ ràng
- Luôn làm sạch kết quả API bên ngoài để đảm bảo hiệu quả token

### 4. Mẫu hợp tác đa Agent

```
Agent Tìm kiếm --> Agent Đối sánh --> Agent Chọn lọc --+-> Agent Hồ sơ (context: kết quả chọn)
                                                         +-> Agent Nghiên cứu công ty (context: kết quả chọn)
                                                         +-> Agent Chuẩn bị PV (context: chọn+hồ sơ+nghiên cứu)
```

- Mỗi agent tuân theo **nguyên tắc trách nhiệm đơn lẻ**
- Phân biệt Task cần thực thi tuần tự và Task có thể chạy song song
- Thiết kế thông tin chảy tự nhiên qua context

### 5. Mẹo thiết kế thực tế

- **Backstory càng dài càng tốt**: Chi tiết kinh nghiệm cụ thể, lĩnh vực chuyên môn, phong cách làm việc dẫn dắt LLM tạo đầu ra chuyên nghiệp hơn
- **Rộng rãi với trường, nghiêm ngặt với đầu ra**: Đặt đủ trường tùy chọn (`| None = None`) trong mô hình Pydantic nhưng trường cốt lõi phải bắt buộc
- **Tiết kiệm token**: Loại bỏ link và ký tự đặc biệt thừa từ kết quả thu thập web để giảm chi phí API
- **`respect_context_window: true`**: Thiết lập cho agent xử lý nhiều văn bản để ngăn lỗi tràn cửa sổ ngữ cảnh

---

## Bài tập thực hành

### Bài tập 1: Tùy chỉnh vai trò Agent (Độ khó: Dễ)

Sửa cấu hình agent trong `agents.yaml` để tạo Job Hunter Agent phù hợp với **nghề khác (ví dụ: designer, marketer)**.

**Yêu cầu:**
- Sửa backstory `job_search_agent` phù hợp thị trường tuyển dụng nghề đó
- Sửa backstory `resume_optimization_agent` phù hợp quy cách viết hồ sơ nghề đó
- Thay `knowledge/resume.txt` bằng hồ sơ mới
- Thay đổi giá trị đầu vào trong `kickoff(inputs={...})` và chạy

### Bài tập 2: Thêm mô hình Pydantic mới (Độ khó: Trung bình)

Thêm các trường sau vào mô hình `ChosenJob` hiện tại và sửa description và expected_output của Task liên quan:

```python
class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
    # Trường mới cần thêm
    salary_competitiveness: str       # "above_market", "at_market", "below_market"
    career_growth_potential: int      # Điểm 1-5
    work_life_balance_score: int      # Điểm 1-5
    recommended_negotiation_points: List[str]  # Danh sách điểm đàm phán
```

### Bài tập 3: Tạo Tool mới (Độ khó: Trung bình)

Tạo Tool tùy chỉnh sử dụng API khác thay vì Firecrawl. Ví dụ:

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
    # Thử triển khai
    pass
```

**Gợi ý:** Sử dụng SerpAPI, Google Custom Search API, v.v. để triển khai logic tìm kiếm và làm sạch đánh giá công ty.

### Bài tập 4: Tối ưu thực thi song song (Độ khó: Khó)

Trong pipeline hiện tại, `resume_rewriting_task` và `company_research_task` chỉ cần cùng context (`job_selection_task`), nên về lý thuyết có thể chạy song song. Nghiên cứu `Process.hierarchical` hoặc tính năng thực thi bất đồng bộ của CrewAI và sửa hai Task này để chạy song song.

```python
@crew
def crew(self):
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        verbose=True,
        process=Process.hierarchical,  # Chế độ thực thi phân cấp
        manager_llm="openai/gpt-4o",   # LLM quản lý
    )
```

### Bài tập 5: Mở rộng toàn bộ Pipeline (Độ khó: Khó)

Thêm agent và Task mới để mở rộng pipeline:

1. **salary_negotiation_agent**: Agent phân tích khoảng lương cho việc làm đã chọn và đề xuất chiến lược đàm phán
2. **cover_letter_agent**: Agent tự động tạo thư xin việc dựa trên hồ sơ và kết quả nghiên cứu công ty

Cho mỗi agent:
- Định nghĩa role, goal, backstory trong `agents.yaml`
- Định nghĩa description, expected_output trong `tasks.yaml`
- Thêm mô hình Pydantic cần thiết trong `models.py`
- Thêm phương thức `@agent`, `@task` trong `main.py` với cấu hình context phù hợp

---

## Tham khảo: Toàn bộ code cuối cùng

Dưới đây là trạng thái cuối cùng của tất cả file cốt lõi hoàn thành trong chương này.

### main.py (Cuối cùng)

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

### models.py (Cuối cùng)

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

### tools.py (Cuối cùng)

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
