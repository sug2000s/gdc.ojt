# Chương 4: CrewAI Nâng cao - Xây dựng Job Hunter Agent

---

## Tổng quan chương

Trong chương này, chúng ta sẽ sử dụng framework CrewAI để xây dựng **hệ thống AI Agent phức hợp có thể áp dụng trong thực tế**. Không phải agent đơn lẻ, mà nhiều agent **phân chia vai trò** và **truyền kết quả công việc cho nhau** để cộng tác, thiết kế và triển khai cấu trúc này theo từng bước.

Chủ đề dự án là **"Job Hunter Agent"**, tự động hóa workflow sau:

1. Tìm kiếm và trích xuất tin tuyển dụng trên web
2. Đối chiếu CV người dùng với tin tuyển dụng và chấm điểm
3. Chọn tin tuyển dụng tối ưu
4. Viết lại CV phù hợp với tin đã chọn
5. Nghiên cứu công ty
6. Tạo tài liệu chuẩn bị phỏng vấn

Kỹ thuật cốt lõi học được qua chương này:

| Kỹ thuật | Mô tả |
|----------|-------|
| **Định nghĩa Agent (YAML)** | Quản lý khai báo role, goal, backstory bằng YAML |
| **Định nghĩa Task (YAML)** | Cấu hình mô tả công việc, kết quả mong đợi, agent phụ trách bằng YAML |
| **Structured Output (Pydantic)** | Cưỡng chế đầu ra agent thành dữ liệu có cấu trúc bằng Pydantic model |
| **Truyền Context** | Tự động truyền kết quả Task trước cho Task sau |
| **Tool tùy chỉnh (Firecrawl)** | Tạo công cụ tìm kiếm/scraping web sử dụng API bên ngoài |
| **Knowledge Source** | Đưa file văn bản (CV) vào làm kiến thức của agent |

---

## 4.1 Định nghĩa Agents và Tasks

### Chủ đề và mục tiêu

Học cách định nghĩa **Agent** và **Task** - các thành phần cốt lõi của CrewAI bằng file cấu hình YAML. Ở bước này, trước khi viết mã, chúng ta thiết kế **"Ai (Agent) sẽ làm gì (Task)"**.

### Giải thích khái niệm chính

#### Cấu trúc dự án

```
job-hunter-agent/
├── config/
│   ├── agents.yaml      # Định nghĩa agent
│   └── tasks.yaml       # Định nghĩa task
├── knowledge/
│   └── resume.txt       # CV người dùng (nguồn kiến thức)
├── main.py              # File thực thi chính (trống ở bước này)
├── pyproject.toml       # Phụ thuộc dự án
└── output/              # Thư mục xuất kết quả
```

CrewAI áp dụng **phương pháp khai báo (declarative)**. Định nghĩa agent và task bằng file YAML thay vì mã:
- Người không phải developer cũng có thể hiểu và sửa đổi hành vi agent
- Tách biệt mã và cấu hình để tăng khả năng bảo trì
- Tài liệu hóa rõ ràng vai trò và mục tiêu của agent

#### Yếu tố cốt lõi trong định nghĩa Agent

Trong CrewAI, Agent được định nghĩa bằng ba thuộc tính cốt lõi:

| Thuộc tính | Vai trò | Ví dụ |
|------------|---------|-------|
| `role` | Chức danh/vai trò của agent | "Senior Job Market Research Specialist" |
| `goal` | Mục tiêu agent cần đạt | "Discover and analyze relevant job opportunities..." |
| `backstory` | Chuyên môn và nền tảng | "You are an experienced talent acquisition specialist with 12+ years..." |

`backstory` không chỉ là mô tả đơn thuần mà là cốt lõi của **prompt engineering** hướng dẫn LLM đưa ra **phán đoán chuyên gia** phù hợp với vai trò.

### Phân tích mã

#### agents.yaml - Định nghĩa 5 agent

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

> **Điểm lưu ý:** Bằng cách chỉ rõ kinh nghiệm cụ thể "12+ years in recruitment" và năng lực chi tiết "reading between the lines of job descriptions" trong `backstory`, hướng dẫn LLM thực hiện **phân tích chuyên gia** thay vì tìm kiếm đơn thuần.

**2) Agent đối chiếu việc làm (job_matching_agent)**

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

**3) Agent tối ưu hóa CV (resume_optimization_agent)**

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

> **Điểm lưu ý:** Tùy chọn `respect_context_window: true` giúp agent tôn trọng kích thước cửa sổ ngữ cảnh của LLM, ngăn lỗi do đầu vào quá dài. Đặc biệt hữu ích cho agent xử lý nhiều văn bản như viết lại CV.

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

#### tasks.yaml - Định nghĩa 6 task

Mỗi Task gồm ba trường cốt lõi: `description`, `expected_output`, `agent`.

**1) Task trích xuất tin tuyển dụng (job_extraction_task)**

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

> **Điểm lưu ý:** `{level}`, `{position}`, `{location}` là **biến template**. Khi thực thi, giá trị thực được thay thế qua `kickoff(inputs={...})`. Cho phép xử lý nhiều điều kiện tìm kiếm khác nhau với cùng một định nghĩa Task.

**2) Task đối chiếu việc làm (job_matching_task)**

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

**4) Task viết lại CV (resume_rewriting_task)**

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

> **Điểm lưu ý:** Chỉ định `output_file` sẽ tự động lưu kết quả Task vào file. `create_directory: true` tự động tạo thư mục đầu ra nếu không tồn tại.

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

#### Phụ thuộc (pyproject.toml)

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
- `firecrawl-py`: Client API tìm kiếm và scraping web
- `python-dotenv`: Tải biến môi trường (API key, v.v.) từ file `.env`

#### Knowledge Source (resume.txt)

Lưu CV người dùng dưới dạng file văn bản để sử dụng làm **nguồn kiến thức** của agent:

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

1. **Nguyên tắc thiết kế agent**: Mỗi agent phải có một lĩnh vực chuyên môn rõ ràng. "Nhiều agent mỗi cái giỏi một thứ" tốt hơn "một agent làm tất cả".
2. **Cách viết backstory**: Bao gồm số năm kinh nghiệm cụ thể, lĩnh vực chuyên môn, phong cách làm việc sẽ cải thiện đáng kể chất lượng đầu ra của LLM.
3. **Tính cụ thể của mô tả Task**: `description` của Task là **prompt thực tế** cho agent. Cần mô tả rõ ràng chỉ dẫn từng bước, tiêu chí đánh giá, ràng buộc.
4. **Sử dụng biến template**: Dùng biến như `{level}`, `{position}` để xử lý nhiều kịch bản với cùng cấu hình.

---

## 4.2 Context và Structured Outputs

### Chủ đề và mục tiêu

Phần này triển khai hai tính năng cốt lõi:
1. **Structured Output**: Cưỡng chế đầu ra agent thành **dữ liệu có cấu trúc** bằng Pydantic model
2. **Truyền Context**: Xây dựng **chuỗi phụ thuộc** tự động truyền kết quả Task trước cho Task sau

### Giải thích khái niệm chính

#### Structured Output là gì?

LLM về cơ bản tạo văn bản tự do. Nhưng trong workflow lập trình cần dữ liệu có **cấu trúc dự đoán được**. CrewAI có thể cưỡng chế đầu ra agent khớp với schema cụ thể qua Pydantic model.

```
Đầu ra tự do LLM:  "I found 3 jobs. The first one is..."  (không thể phân tích)
Đầu ra có cấu trúc: {"jobs": [{"job_title": "...", "company_name": "..."}]}  (lập trình được)
```

#### Context là gì?

Trong CrewAI, `context` định nghĩa **luồng dữ liệu giữa các Task**. Dùng khi truyền kết quả Task A làm đầu vào Task B.

```
job_extraction_task ─> job_matching_task ─> job_selection_task ─┬─> resume_rewriting_task ──┐
                                                                 ├─> company_research_task ──┤
                                                                 └───────────────────────────┴─> interview_prep_task
```

Trong sơ đồ trên:
- `resume_rewriting_task` nhận kết quả (việc làm đã chọn) của `job_selection_task` làm context
- `interview_prep_task` nhận kết quả của cả ba Task (job_selection, resume_rewriting, company_research) làm context

### Phân tích mã

#### models.py - Định nghĩa Pydantic model

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

**Nguyên tắc thiết kế cốt lõi của cấu trúc model:**

1. **Phân biệt trường bắt buộc và tùy chọn**: `job_title`, `company_name` luôn cần thiết, nhưng thông tin như `includes_equity`, `offers_visa_sponsorship` không phải tin tuyển dụng nào cũng có. Định nghĩa trường tùy chọn bằng `| None = None` để đảm bảo tính linh hoạt.

2. **Mẫu mở rộng dữ liệu dần dần**: Dữ liệu ngày càng phong phú khi đi qua pipeline:
   - `Job`: Thông tin tuyển dụng cơ bản
   - `RankedJob`: Job + điểm đối chiếu + lý do (thêm kết quả đánh giá)
   - `ChosenJob`: Job + trạng thái chọn + lý do (thêm kết quả quyết định)

3. **Mẫu Composition**: `RankedJob` không kế thừa `Job` mà **chứa (composition)** nó. Như vậy có thể thêm thông tin bổ sung mà không thay đổi dữ liệu Job gốc.

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

**Định nghĩa Task và kết nối Structured Output:**

```python
    @task
    def job_extraction_task(self):
        return Task(
            config=self.tasks_config["job_extraction_task"],
            output_pydantic=JobList,  # Cưỡng chế đầu ra theo schema JobList
        )

    @task
    def job_matching_task(self):
        return Task(
            config=self.tasks_config["job_matching_task"],
            output_pydantic=RankedJobList,  # Cưỡng chế đầu ra theo schema RankedJobList
        )

    @task
    def job_selection_task(self):
        return Task(
            config=self.tasks_config["job_selection_task"],
            output_pydantic=ChosenJob,  # Cưỡng chế đầu ra theo schema ChosenJob
        )
```

> **Cốt lõi:** Chỉ định `output_pydantic=JobList` thì CrewAI tự động phân tích đầu ra LLM khớp với schema `JobList`. Nếu đầu ra không khớp schema sẽ tự động thử lại.

**Truyền dữ liệu giữa Task qua Context:**

```python
    @task
    def resume_rewriting_task(self):
        return Task(
            config=self.tasks_config["resume_rewriting_task"],
            context=[
                self.job_selection_task(),  # Truyền thông tin việc đã chọn
            ],
        )

    @task
    def company_research_task(self):
        return Task(
            config=self.tasks_config["company_research_task"],
            context=[
                self.job_selection_task(),  # Truyền thông tin việc đã chọn
            ],
        )

    @task
    def interview_prep_task(self):
        return Task(
            config=self.tasks_config["interview_prep_task"],
            context=[
                self.job_selection_task(),      # Việc đã chọn
                self.resume_rewriting_task(),   # CV đã viết lại
                self.company_research_task(),   # Kết quả nghiên cứu công ty
            ],
        )
```

> **Cốt lõi:** Tham số `context` nhận **danh sách** nhiều Task. `interview_prep_task` tham chiếu kết quả của 3 Task để tạo tài liệu chuẩn bị phỏng vấn tổng hợp. CrewAI chỉ thực thi Task khi các Task trong context đã hoàn thành.

**Lắp ráp và thực thi Crew:**

```python
    @crew
    def crew(self):
        return Crew(
            agents=self.agents,  # Tự động thu thập từ phương thức @agent
            tasks=self.tasks,    # Tự động thu thập từ phương thức @task
            verbose=True,        # Xuất chi tiết quá trình thực thi
        )


JobHunterCrew().crew().kickoff()
```

### Điểm thực hành

1. **Thiết kế Pydantic model**: Trường bắt buộc tối thiểu, trường tùy chọn dùng `| None = None` để đáp ứng nhiều nguồn dữ liệu khác nhau.
2. **Đồ thị phụ thuộc Context**: Vẽ sơ đồ phụ thuộc giữa Task giúp xác định Task có thể chạy song song. Ví dụ, `resume_rewriting_task` và `company_research_task` chỉ cần cùng context nên có thể chạy song song.
3. **`output_pydantic` vs `output_file`**: Dùng `output_pydantic` khi cần dữ liệu có cấu trúc, dùng `output_file` khi cần tài liệu cho người đọc. Có thể dùng cả hai đồng thời.

---

## 4.3 Firecrawl Tool - Công cụ tìm kiếm web tùy chỉnh

### Chủ đề và mục tiêu

Tạo **Tool tùy chỉnh** để agent CrewAI tương tác với thế giới bên ngoài. Triển khai công cụ tìm kiếm web sử dụng Firecrawl API, tinh chế kết quả và truyền cho agent.

### Giải thích khái niệm chính

#### Tool là gì?

Trong CrewAI, Tool là hàm cho phép agent **truy cập tài nguyên bên ngoài** hoặc **thực hiện tác vụ cụ thể**. Agent gọi Tool vào thời điểm phù hợp theo phán đoán của LLM.

```
Quá trình suy nghĩ của agent:
1. "Cần tìm tin tuyển dụng Senior level Golang Developer"
2. "Sử dụng web_search_tool"
3. web_search_tool("Senior Golang Developer jobs Netherlands") gọi
4. Phân tích kết quả và sắp xếp theo schema JobList
```

#### Firecrawl là gì?

Firecrawl là dịch vụ API crawl trang web và chuyển đổi nội dung sang **định dạng markdown**. Khác với web scraping thông thường:
- Hỗ trợ JavaScript rendering (crawl được cả trang SPA)
- Tự động loại bỏ thành phần không cần thiết (quảng cáo, navigation, v.v.)
- Chuyển đổi thành văn bản markdown sạch

### Phân tích mã

#### tools.py - Triển khai công cụ tìm kiếm web

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

**Phân tích chi tiết mã:**

1. **Decorator `@tool`**: Chuyển đổi hàm thông thường thành Tool mà agent có thể sử dụng. Tên hàm và docstring được truyền cho agent làm mô tả công cụ.

2. **Khởi tạo FirecrawlApp**: Đọc `FIRECRAWL_API_KEY` từ file `.env` để xác thực.

3. **Thực hiện tìm kiếm**:
   ```python
   response = app.search(
       query=query,       # Từ khóa tìm kiếm
       limit=5,           # Tối đa 5 kết quả
       scrape_options=ScrapeOptions(
           formats=["markdown"],  # Trả về ở định dạng markdown
       ),
   )
   ```
   Giới hạn `limit=5` để tiết kiệm cửa sổ ngữ cảnh LLM và tránh nhầm lẫn do quá nhiều thông tin.

4. **Tinh chế kết quả (Data Cleaning)**:
   ```python
   # Loại bỏ backslash và xuống dòng liên tục không cần thiết
   cleaned = re.sub(r"\\+|\n+", "", markdown).strip()
   # Loại bỏ liên kết markdown và URL (tiết kiệm token)
   cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)
   ```
   Kết quả web crawling chứa nhiều liên kết, ký tự đặc biệt không cần thiết. Loại bỏ chúng giúp:
   - **Tiết kiệm số token** truyền cho LLM
   - Agent **tập trung vào nội dung cốt lõi**
   - Giảm chi phí API

5. **Giá trị trả về**: Trả về danh sách dictionary chứa tiêu đề, URL, nội dung markdown đã tinh chế.

### Điểm thực hành

1. **Sử dụng decorator `@tool`**: Bất kỳ hàm Python nào gắn `@tool` đều có thể dùng làm công cụ agent. Có thể ứng dụng cho nhiều mục đích: truy vấn database, gọi API bên ngoài, xử lý file, v.v.
2. **Tầm quan trọng của tinh chế dữ liệu**: Dữ liệu truyền cho LLM càng sạch càng tốt. Loại bỏ thẻ HTML, URL, ký tự đặc biệt không cần thiết giúp cải thiện chất lượng đầu ra.
3. **Xử lý lỗi**: Xử lý thất bại gọi API bằng `if not response.success`. Trong môi trường production cần logic xử lý lỗi và retry tinh vi hơn.
4. **Giới hạn kết quả**: Giới hạn số kết quả tìm kiếm như `limit=5` là quyết định thiết kế quan trọng cân bằng chi phí và chất lượng.

---

## 4.5 Kết luận - Knowledge Source và thực thi cuối cùng

### Chủ đề và mục tiêu

Phần cuối cùng hoàn thành:
1. **Kết nối Knowledge Source**: Đưa file văn bản CV vào làm kiến thức của agent
2. **Thêm Tool docstring**: Thêm mô tả để agent sử dụng công cụ đúng cách
3. **Truyền giá trị đầu vào thực thi**: Truyền giá trị thực cho biến template qua `kickoff(inputs={...})`
4. **Xác nhận kết quả**: Thực thi toàn bộ pipeline và xác minh đầu ra

### Giải thích khái niệm chính

#### Knowledge Source là gì?

Knowledge Source của CrewAI là cơ chế cung cấp **kiến thức sẵn có** cho agent. Tự động đưa dữ liệu từ nhiều định dạng như file văn bản, PDF, CSV vào ngữ cảnh agent.

Khác với việc đặt văn bản trực tiếp vào prompt thông thường, Knowledge Source:
- Sử dụng **vector database (ChromaDB)** để tìm kiếm thông tin liên quan
- Xử lý hiệu quả cả tài liệu lớn
- Chia sẻ cùng kiến thức giữa nhiều agent

#### Tầm quan trọng của Tool Docstring

Agent **đọc docstring** của Tool để phán đoán khi nào và cách sử dụng công cụ. Không có docstring rõ ràng, agent có thể sử dụng sai hoặc không sử dụng công cụ.

### Phân tích mã

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

`TextFileKnowledgeSource` đọc file văn bản và lưu nội bộ dưới dạng embedding trong vector database ChromaDB. Sau đó khi agent đặt câu hỏi liên quan, thông tin phù hợp được cung cấp tự động qua tìm kiếm tương đồng.

**Kết nối Knowledge Source với agent:**

```python
@agent
def job_matching_agent(self):
    return Agent(
        config=self.agents_config["job_matching_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức CV
    )

@agent
def resume_optimization_agent(self):
    return Agent(
        config=self.agents_config["resume_optimization_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức CV
    )

@agent
def company_research_agent(self):
    return Agent(
        config=self.agents_config["company_research_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức CV
        tools=[web_search_tool],               # Thêm công cụ tìm kiếm web
    )

@agent
def interview_prep_agent(self):
    return Agent(
        config=self.agents_config["interview_prep_agent"],
        knowledge_sources=[resume_knowledge],  # Thêm kiến thức CV
    )
```

> **Điểm lưu ý:** Tất cả agent chia sẻ cùng instance `resume_knowledge`. Qua đó:
> - `job_matching_agent`: Tính điểm đối chiếu tin tuyển dụng dựa trên CV
> - `resume_optimization_agent`: Tham chiếu CV gốc để viết lại
> - `company_research_agent`: Nghiên cứu công ty có tính đến tech stack trong CV + tìm kiếm web
> - `interview_prep_agent`: Tạo tài liệu chuẩn bị phỏng vấn phản ánh nội dung CV

Chú ý chỉ `company_research_agent` được thêm `tools=[web_search_tool]`. Agent này cần công cụ tìm kiếm vì phải nghiên cứu thông tin công ty trên web. Ngược lại, `resume_optimization_agent` hay `interview_prep_agent` chỉ cần thông tin đã nhận qua context.

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

Biến template trong YAML được thay thế qua `kickoff(inputs={...})`:

```yaml
# Bản gốc tasks.yaml
description: >
  Find and extract {level} level {position} jobs in {location}.

# Kết quả thay thế khi thực thi
description: >
  Find and extract Senior level Golang Developer jobs in Netherlands.
```

`result.tasks_output` chứa kết quả thực thi từng Task theo thứ tự, có thể truy cập trực tiếp instance Pydantic model qua `task_output.pydantic`.

#### Ví dụ kết quả thực thi

Khi toàn bộ pipeline thực thi, 3 file markdown được tạo trong thư mục `output/`:

**1) output/rewritten_resume.md** - CV viết lại phù hợp với Senior Golang Developer:
- Đổi chức danh từ "Full Stack Developer" thành "Senior Backend Developer (API Design | Microservices | Cloud-Native)"
- Tái cấu trúc tech stack phù hợp với môi trường FinTech dựa trên Go
- Viết lại kinh nghiệm tập trung vào API, microservices, tối ưu hiệu suất

**2) output/company_research.md** - Báo cáo nghiên cứu công ty (FinTech Innovators):
- Tổng quan công ty, sứ mệnh và giá trị, tin tức gần đây
- Phân tích tech stack của vị trí (Go, Kafka, Kubernetes, Terraform, v.v.)
- Chủ đề phỏng vấn dự kiến và danh sách câu hỏi ứng viên nên hỏi

**3) output/interview_prep.md** - Tài liệu chuẩn bị phỏng vấn tổng hợp:
- Tổng quan vị trí và phân tích phù hợp
- Điểm nổi bật CV
- Câu hỏi phỏng vấn dự kiến (Golang, thiết kế API, kiến trúc event-driven, v.v.)
- Lời khuyên chiến lược (thái độ người học tự tin, tiếp cận hướng giải pháp, v.v.)

### Điểm thực hành

1. **Sử dụng Knowledge Source**: Có thể cung cấp CV, thông tin công ty, tài liệu sản phẩm, v.v. dưới dạng file cho agent. Xử lý hiệu quả tài liệu lớn qua vector database.
2. **Tool docstring**: Docstring rõ ràng là bắt buộc để agent sử dụng công cụ đúng. Bao gồm mô tả đối số, định dạng giá trị trả về, kịch bản sử dụng.
3. **Template giá trị đầu vào**: Tái sử dụng cùng hệ thống agent cho nhiều điều kiện khác nhau qua `kickoff(inputs={...})`.
4. **Truy cập kết quả**: Truy cập kết quả từng Task bằng lập trình qua `result.tasks_output` để xử lý hậu kỳ.

---

## Tổng kết chương

### 1. Cấu trúc khai báo của CrewAI

| Thành phần | Vị trí định nghĩa | Vai trò |
|------------|-------------------|---------|
| Agent | `config/agents.yaml` | Định nghĩa vai trò, mục tiêu, câu chuyện nền |
| Task | `config/tasks.yaml` | Chỉ định mô tả công việc, kết quả mong đợi, agent phụ trách |
| Crew | `main.py` (`@CrewBase`) | Tổ hợp agent và task để thực thi |

### 2. Kiểm soát luồng dữ liệu

- **Structured Output (`output_pydantic`)**: Cấu trúc hóa đầu ra agent bằng Pydantic model
- **Context**: Định nghĩa phụ thuộc dữ liệu giữa Task bằng `context=[task_a(), task_b()]`
- **Knowledge Source**: Lưu file văn bản, v.v. vào vector DB để sử dụng làm kiến thức sẵn có của agent

### 3. Tạo Tool tùy chỉnh

- Chuyển đổi hàm Python thành công cụ agent bằng decorator `@tool`
- Hướng dẫn agent sử dụng đúng công cụ qua docstring rõ ràng
- Tinh chế (cleaning) kết quả API bên ngoài để đảm bảo hiệu quả token

### 4. Mẫu cộng tác đa agent

```
Agent tìm kiếm ──> Agent đối chiếu ──> Agent chọn ──┬──> Agent CV (context: kết quả chọn)
                                                       ├──> Agent nghiên cứu (context: kết quả chọn)
                                                       └──> Agent phỏng vấn (context: chọn+CV+nghiên cứu)
```

- Mỗi agent tuân theo **nguyên tắc trách nhiệm đơn lẻ**
- Phân biệt Task cần thực thi tuần tự và Task có thể chạy song song
- Thiết kế để thông tin chảy tự nhiên qua context

### 5. Mẹo thiết kế thực chiến

- **Backstory càng dài càng tốt**: Mô tả chi tiết kinh nghiệm cụ thể, lĩnh vực chuyên môn, phong cách làm việc giúp LLM tạo đầu ra chuyên nghiệp hơn
- **Trường linh hoạt, đầu ra nghiêm ngặt**: Đặt đủ trường tùy chọn (`| None = None`) trong Pydantic model nhưng chỉ định trường cốt lõi là bắt buộc
- **Tiết kiệm token**: Loại bỏ liên kết, ký tự đặc biệt không cần thiết từ kết quả web crawling để giảm chi phí API
- **`respect_context_window: true`**: Thiết lập cho agent xử lý nhiều văn bản để ngăn lỗi vượt cửa sổ ngữ cảnh

---

## Bài tập thực hành

### Bài tập 1: Tùy chỉnh vai trò agent (Độ khó: Thấp)

Sửa cấu hình agent trong `agents.yaml` để tạo Job Hunter Agent phù hợp cho **ngành nghề khác (ví dụ: designer, marketer)**.

**Yêu cầu:**
- Sửa backstory của `job_search_agent` phù hợp với thị trường tuyển dụng ngành đó
- Sửa backstory của `resume_optimization_agent` phù hợp với thực hành viết CV ngành đó
- Thay `knowledge/resume.txt` bằng CV mới
- Thay đổi giá trị đầu vào `kickoff(inputs={...})` và chạy

### Bài tập 2: Thêm Pydantic model mới (Độ khó: Trung bình)

Thêm trường sau vào model `ChosenJob` hiện tại và sửa description, expected_output của Task liên quan:

```python
class ChosenJob(BaseModel):
    job: Job
    selected: bool
    reason: str
    # Trường mới thêm
    salary_competitiveness: str       # "above_market", "at_market", "below_market"
    career_growth_potential: int      # Điểm 1-5
    work_life_balance_score: int      # Điểm 1-5
    recommended_negotiation_points: List[str]  # Danh sách điểm thương lượng
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
    # Triển khai thử
    pass
```

**Gợi ý:** Sử dụng SerpAPI, Google Custom Search API, v.v. để tìm kiếm review công ty và triển khai logic tinh chế.

### Bài tập 4: Tối ưu hóa thực thi song song (Độ khó: Cao)

Trong pipeline hiện tại, `resume_rewriting_task` và `company_research_task` chỉ cần cùng context (`job_selection_task`) nên về lý thuyết có thể chạy song song. Nghiên cứu `Process.hierarchical` hoặc tính năng thực thi bất đồng bộ của CrewAI để sửa hai Task này chạy song song.

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

### Bài tập 5: Mở rộng toàn bộ pipeline (Độ khó: Cao)

Thêm agent và Task mới để mở rộng pipeline:

1. **salary_negotiation_agent**: Agent phân tích phạm vi lương của vị trí đã chọn và đề xuất chiến lược thương lượng
2. **cover_letter_agent**: Agent tự động tạo cover letter dựa trên CV và kết quả nghiên cứu công ty

Cho mỗi agent:
- Định nghĩa role, goal, backstory trong `agents.yaml`
- Định nghĩa description, expected_output trong `tasks.yaml`
- Thêm Pydantic model cần thiết trong `models.py`
- Thêm phương thức `@agent`, `@task` trong `main.py` và thiết lập context phù hợp

---

## Tham khảo: Mã nguồn cuối cùng hoàn chỉnh

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
