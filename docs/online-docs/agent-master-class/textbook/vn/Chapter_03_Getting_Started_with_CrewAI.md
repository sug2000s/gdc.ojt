# Chapter 3: Bắt đầu với CrewAI

---

## Tổng quan chương

Trong chương này, chúng ta sẽ học từng bước cách xây dựng AI agent bằng framework **CrewAI** từ đầu. CrewAI là một framework Python được thiết kế để nhiều AI agent có thể hợp tác thực hiện các tác vụ phức tạp, được tổ chức xung quanh khái niệm "Crew (đội nhóm)" với các agent và task.

Chương này gồm 4 phần, bắt đầu từ một agent dịch thuật đơn giản và dần phát triển thành hệ thống agent đọc tin tức cấp độ production.

| Phần | Chủ đề | Nội dung học tập chính |
|------|--------|----------------------|
| 3.1 | Your First CrewAI Agent | Cấu trúc dự án, khái niệm cơ bản Agent/Task/Crew, cấu hình YAML |
| 3.2 | Custom Tools | Tạo công cụ tùy chỉnh, decorator `@tool`, kết nối công cụ với agent |
| 3.3 | News Reader Tasks and Agents | Thiết kế agent thực tế, viết prompt chi tiết, cấu hình đa agent |
| 3.4 | News Reader Crew | Tích hợp công cụ thực (tìm kiếm/thu thập), chỉ định mô hình LLM, thực thi Crew và kết quả |

### Mục tiêu học tập

Sau khi hoàn thành chương này, bạn sẽ có thể:

1. Thiết lập và cấu trúc dự án CrewAI từ đầu
2. Định nghĩa agent và task bằng file cấu hình YAML
3. Tạo công cụ tùy chỉnh và kết nối chúng với agent
4. Cấu hình và thực thi Crew nơi nhiều agent hợp tác tuần tự
5. Xây dựng pipeline thu thập/tóm tắt/biên tập tin tức cấp production

### Điều kiện tiên quyết

- Python 3.13 trở lên
- Trình quản lý gói `uv` (để quản lý dự án Python)
- OpenAI API key (cấu hình trong file `.env`)
- Hiểu biết cơ bản về cú pháp Python

---

## 3.1 Your First CrewAI Agent

### Chủ đề và mục tiêu

Trong phần đầu tiên này, chúng ta tìm hiểu cấu trúc cơ bản của CrewAI và tạo một **Agent dịch thuật (Translator Agent)** đơn giản. Qua quá trình này, chúng ta nắm bắt mối quan hệ giữa ba yếu tố cốt lõi của CrewAI -- **Agent**, **Task**, và **Crew** -- và học vai trò của các file cấu hình dựa trên YAML.

### Giải thích khái niệm cốt lõi

#### Ba khái niệm cốt lõi của CrewAI

CrewAI được xây dựng trên ba thành phần cốt lõi:

1. **Agent (Tác nhân)**: Một nhân viên AI với vai trò (role), mục tiêu (goal) và câu chuyện nền (backstory) cụ thể. Hãy nghĩ như một thành viên trong đội.
2. **Task (Nhiệm vụ)**: Một công việc cụ thể mà agent phải thực hiện. Bao gồm mô tả (description) và kết quả mong đợi (expected_output).
3. **Crew (Đội)**: Đơn vị đội nhóm gộp agent và task lại để thực thi. Các agent xử lý task theo thứ tự.

```
+------------------------------------------+
|                  Crew                    |
|                                          |
|  +----------+    +------------------+    |
|  |  Agent   |--->|     Task 1       |    |
|  |(Dịch giả)|    |(Anh->Italia)     |    |
|  +----------+    +------------------+    |
|       |                   |              |
|       |     (kết quả chuyển sang task kế)|
|       |                   v              |
|       |          +------------------+    |
|       +--------->|     Task 2       |    |
|                  |(Italia->Hy Lạp)  |    |
|                  +------------------+    |
+------------------------------------------+
```

#### Decorator `@CrewBase` và cấu trúc dự án

CrewAI sử dụng **mẫu lớp dựa trên decorator**. Khi decorator `@CrewBase` được áp dụng cho một lớp, CrewAI tự động đọc các file `config/agents.yaml` và `config/tasks.yaml` và cung cấp chúng dưới dạng từ điển `self.agents_config` và `self.tasks_config`.

#### Cấu trúc thư mục dự án

```
news-reader-agent/
├── .python-version          # Chỉ định phiên bản Python (3.13)
├── .gitignore               # Danh sách file Git loại trừ
├── pyproject.toml           # Cấu hình dự án và phụ thuộc
├── config/
│   ├── agents.yaml          # Định nghĩa agent
│   └── tasks.yaml           # Định nghĩa task
├── main.py                  # File thực thi chính
└── uv.lock                  # File khóa phụ thuộc
```

Cấu trúc này tuân theo quy ước của CrewAI. Khi đặt file YAML trong thư mục `config/`, `@CrewBase` tự động nhận diện chúng.

### Phân tích code

#### Phụ thuộc dự án (`pyproject.toml`)

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

- `crewai[tools]`: Cài đặt framework CrewAI cùng với gói mở rộng công cụ. `[tools]` là extras chỉ định, bao gồm các phụ thuộc liên quan đến công cụ bổ sung.
- `python-dotenv`: Thư viện tải biến môi trường từ file `.env`. Dùng để quản lý API key an toàn.

#### Cấu hình Agent (`config/agents.yaml`)

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

**Vai trò của mỗi trường:**

| Trường | Mô tả | Vai trò trong ví dụ |
|--------|-------|-------------------|
| `role` | Định nghĩa công việc/vai trò của agent | Dịch giả Anh-Italia |
| `goal` | Mục tiêu agent phải đạt được | Dịch chính xác không gây hiểu lầm |
| `backstory` | Thiết lập nền tảng cho agent (trao tính cách và chuyên môn) | Người song ngữ lớn lên ở New York và Palermo |

> **Tại sao `backstory` quan trọng?**
> `backstory` không phải trang trí đơn thuần. LLM sử dụng thông tin nền này làm ngữ cảnh khi tạo phản hồi, tạo ra kết quả nhất quán và chuyên nghiệp hơn. Ví dụ, câu "có thể phát hiện sự khác biệt văn hóa" khuyến khích bản dịch phản ánh sắc thái văn hóa.

#### Cấu hình Task (`config/tasks.yaml`)

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

**Điểm chính:**

- `{sentence}`: **Placeholder biến** được bọc trong dấu ngoặc nhọn. Được thay thế khi chạy bằng giá trị truyền qua `kickoff(inputs={"sentence": "..."})`.
- `expected_output`: Cho agent biết rõ kết quả nên ở dạng nào. Cần thiết để agent hiểu chính xác phải trả về gì.
- `agent`: Tên agent sẽ thực hiện task này. Phải khớp với khóa được định nghĩa trong `agents.yaml`.
- Cả hai task đều sử dụng cùng `translator_agent`. Một agent có thể thực hiện nhiều task.

#### File thực thi chính (`main.py`)

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

**Phân tích chi tiết luồng hoạt động code:**

1. **`dotenv.load_dotenv()`**: Tải các biến môi trường như `OPENAI_API_KEY` từ file `.env`. CrewAI nội bộ sử dụng khóa này để gọi LLM API.

2. **Decorator `@CrewBase`**: Khi áp dụng cho lớp, tự động đọc `config/agents.yaml` và `config/tasks.yaml`. Cho phép truy cập `self.agents_config` và `self.tasks_config`.

3. **Decorator `@agent`**: Báo cho CrewAI rằng phương thức trả về đối tượng Agent. Tên phương thức (`translator_agent`) phải khớp với khóa trong YAML.

4. **Decorator `@task`**: Báo cho CrewAI rằng phương thức trả về đối tượng Task. Các task được thực thi **theo thứ tự định nghĩa**. `translate_task` -> `retranslate_task` là thứ tự thực thi.

5. **Decorator `@crew`**: Áp dụng cho phương thức tổ hợp đối tượng Crew.
   - `self.agents`: Danh sách tất cả agent được đánh dấu `@agent` (tự động thu thập)
   - `self.tasks`: Danh sách tất cả task được đánh dấu `@task` (tự động thu thập)
   - `verbose=True`: In chi tiết tiến trình thực thi ra console.

6. **`kickoff(inputs={...})`**: Thực thi Crew. Các giá trị trong từ điển `inputs` thay thế placeholder `{sentence}` trong YAML.

> **Luồng kết quả thực thi:**
> 1. `translate_task`: "I'm Nico and I like to ride my bicicle in Napoli" -> Dịch sang tiếng Italia
> 2. `retranslate_task`: Kết quả dịch tiếng Italia -> Dịch sang tiếng Hy Lạp
>
> Task thứ hai **tự động nhận kết quả task đầu tiên làm ngữ cảnh**. Đây là cách task chaining hoạt động trong CrewAI.

### Điểm thực hành

- Sau khi thiết lập `OPENAI_API_KEY=sk-...` trong file `.env`, thử chạy bằng `uv run python main.py`.
- Sử dụng `verbose=True` để quan sát quá trình suy nghĩ của agent (Chain of Thought).
- Thử thay đổi `backstory` và so sánh chất lượng kết quả khác nhau như thế nào.
- Thử thêm task thứ ba (ví dụ: dịch Hy Lạp -> Hàn Quốc).

---

## 3.2 Custom Tools

### Chủ đề và mục tiêu

Trong phần này, chúng ta học cách kết nối **Công cụ tùy chỉnh (Custom Tool)** với agent CrewAI. Mặc dù LLM chỉ có thể tạo văn bản theo mặc định, công cụ cho phép chúng thực hiện các chức năng bên ngoài (tính toán, gọi API, đọc file, v.v.).

### Giải thích khái niệm cốt lõi

#### Tool (Công cụ) là gì?

**Công cụ (Tool)** của AI agent là hàm mà agent có thể sử dụng ngoài khả năng tạo văn bản của LLM. Khi agent xác định "tác vụ này cần công cụ" trong quá trình thực hiện task, nó tự động gọi công cụ phù hợp.

```
+--------------------------------------+
|           Agent                      |
|                                      |
|  "Tôi cần đếm số chữ cái trong     |
|   câu này...                         |
|   Hãy dùng công cụ count_letters!"  |
|                                      |
|   +-----------------------------+    |
|   |  Tool: count_letters        |    |
|   |  Input: "Hello World"       |    |
|   |  Output: 11                 |    |
|   +-----------------------------+    |
+--------------------------------------+
```

Công cụ quan trọng vì LLM có thể tạo ra **ảo giác (hallucination)** trong tính toán toán học hoặc tra cứu dữ liệu chính xác. Ngay cả tác vụ đơn giản như đếm chữ cái cũng có thể sai nếu LLM tự làm, nhưng cung cấp `len()` dưới dạng công cụ đảm bảo kết quả chính xác.

#### Decorator `@tool`

CrewAI cung cấp decorator `@tool` để chuyển đổi hàm Python thông thường thành công cụ mà agent có thể sử dụng. **Docstring** của hàm đóng vai trò giải thích mục đích công cụ cho agent.

### Phân tích code

#### Định nghĩa công cụ tùy chỉnh (`tools.py`)

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

**Phân tích chính:**

- **Decorator `@tool`**: Decorator này chuyển đổi hàm thành công cụ CrewAI. Nội bộ, nó phân tích chữ ký hàm và docstring để tạo schema công cụ mà LLM có thể hiểu.
- **Type hint `sentence: str`**: Bắt buộc. CrewAI sử dụng type hint này để thông báo cho LLM về định dạng tham số đầu vào.
- **docstring**: Được agent sử dụng để xác định "khi nào nên dùng công cụ này?" Cần viết rõ ràng và chi tiết. Tốt nhất nên mô tả định dạng đầu vào và đầu ra.
- **Câu lệnh `print()`**: Để debug. Cho phép xác nhận công cụ có thực sự được gọi không và nhận đầu vào gì.
- **`return len(sentence)`**: Logic thực tế. Thay vì LLM tự đếm chữ cái, hàm `len()` của Python trả về kết quả chính xác.

#### Thêm Agent và Task mới (`config/agents.yaml`)

```yaml
counter_agent:
  role: >
    To count the lenght of things.
  goal: >
    To be a good counter that never lies or makes things up.
  backstory: >
    You are a genius counter.
```

Lưu ý biểu đạt "never lies or makes things up" trong `goal`. Đây là kỹ thuật prompt khuyến khích agent luôn sử dụng công cụ thay vì đoán.

#### Thêm Task (`config/tasks.yaml`)

```yaml
count_task:
  description: >
    Count the amount of letters in a sentence.
  expected_output: >
    The number of letters in a sentence.
  agent: counter_agent
```

#### Kết nối công cụ trong main.py

```python
from tools import count_letters

# ... bên trong lớp ...

@agent
def counter_agent(self):
    return Agent(
        config=self.agents_config["counter_agent"],
        tools=[count_letters],  # Kết nối công cụ với agent
    )

@task
def count_task(self):
    return Task(
        config=self.tasks_config["count_task"],
    )
```

**Điểm chính:**

- `tools=[count_letters]`: Truyền danh sách công cụ qua tham số `tools` khi tạo Agent. Có thể kết nối nhiều công cụ với một agent.
- Công cụ được kết nối ở **cấp agent** (không phải cấp task). Dù agent thực hiện task nào, nó đều có thể sử dụng các công cụ được gán cho mình.
- Không có công cụ nào được chỉ định trực tiếp trong `count_task`. Vì agent phụ trách task (`counter_agent`) đã có sẵn công cụ.

### Điểm thực hành

- Quan sát đầu ra `print()` để xác nhận thời điểm công cụ thực sự được gọi.
- Thử làm docstring mơ hồ và kiểm tra agent có gọi công cụ đúng không.
- Tạo công cụ mới (ví dụ: đếm từ, chuyển đổi chữ hoa) và thêm vào agent.
- Kết nối nhiều công cụ với một agent và quan sát agent có chọn công cụ phù hợp cho từng tình huống không.

---

## 3.3 News Reader Tasks and Agents

### Chủ đề và mục tiêu

Trong phần này, chúng ta hoàn toàn tái cấu trúc các ví dụ dịch thuật/đếm đơn giản trước đó thành **hệ thống đọc tin tức production**. Chúng ta thiết kế 3 agent chuyên biệt và 3 task chi tiết, học các kỹ thuật prompt engineering cấp production.

### Giải thích khái niệm cốt lõi

#### Kiến trúc đa Agent

Hệ thống đọc tin tức tuân theo cấu trúc **pipeline 3 giai đoạn**:

```
+--------------+     +--------------+     +--------------+
| News Hunter  |---->|  Summarizer  |---->|   Curator    |
|   Agent      |     |    Agent     |     |    Agent     |
|              |     |              |     |              |
| Thu thập     |     | Tóm tắt      |     | Biên tập &   |
| tin tức &    |     | bài viết     |     | Báo cáo      |
| lọc          |     | (3 cấp độ)  |     | cuối cùng    |
+--------------+     +--------------+     +--------------+
  Task 1:              Task 2:              Task 3:
  content_harvesting   summarization        final_report_assembly

  output:              output:              output:
  content_harvest.md   summary.md           final_report.md
```

Mỗi agent có **chuyên môn khác nhau**. Đây là ưu điểm chính của hệ thống đa agent. Để mỗi agent tập trung vào chuyên môn riêng cho kết quả tốt hơn so với một agent làm mọi thứ.

#### Prompt Engineering: Viết `backstory` chi tiết

Thay đổi quan trọng nhất trong phần này là **độ sâu và chi tiết** của cấu hình agent. Backstory đơn giản 2 dòng từ 3.1 phát triển thành hồ sơ chi tiết hơn 10 dòng.

#### Cấu hình `output_file` của Task

Mỗi task được cấu hình để **tự động lưu kết quả dưới dạng file markdown**. Điều này cho phép kiểm tra và debug kết quả trung gian ở mỗi giai đoạn.

### Phân tích code

#### Cấu hình Agent (`config/agents.yaml`)

**1. News Hunter Agent - Chuyên gia thu thập tin tức**

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

**Tùy chọn cấu hình mới:**

| Tùy chọn | Mô tả |
|----------|-------|
| `verbose: true` | In chi tiết quá trình suy nghĩ của agent |
| `inject_date: true` | Tự động đưa ngày hiện tại vào ngữ cảnh agent. Cần thiết để đánh giá tính thời sự tin tức |

**Phân tích `backstory` - Tại sao viết chi tiết đến vậy:**

- "15 years of experience": Thiết lập mức chuyên môn để khuyến khích LLM đưa ra phán đoán chất lượng cao
- "separate signal from noise": Nhấn mạnh khả năng lọc để khuyến khích lọc bỏ bài viết không liên quan
- "detecting bias and misinformation": Kích hoạt khả năng đánh giá độ tin cậy
- "source diversity": Khuyến khích thu thập thông tin từ các nguồn đa dạng

**2. Summarizer Agent - Chuyên gia tóm tắt**

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

**Cài đặt mới: `llm: openai/o3`**

Có thể **chỉ định mô hình LLM khác nhau** cho các agent cụ thể. Vì tóm tắt đòi hỏi mức độ hiểu biết và diễn đạt cao, nên sử dụng mô hình mạnh hơn (o3). Bằng cách cấu hình mô hình khác nhau cho mỗi agent, bạn có thể tối ưu chi phí và hiệu suất.

**3. Curator Agent - Chuyên gia biên tập**

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

#### Cấu hình Task (`config/tasks.yaml`)

**1. Content Harvesting Task - Task thu thập tin tức**

Task này chứa hướng dẫn chi tiết nhất. Hãy xem các phần chính:

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

**Phân tích kỹ thuật Prompt Engineering:**

1. **Hướng dẫn từng bước**: Các tác vụ cần thực hiện được chỉ định theo thứ tự đánh số.
2. **Quy tắc lọc rõ ràng**: "IMPORTANT" được nhấn mạnh bằng chữ hoa, với các mẫu URL cho phép/từ chối kèm ví dụ cụ thể.
3. **Định dạng danh sách kiểm tra**: Phân biệt trực quan bằng ký hiệu cho phép và từ chối.
4. **Tiêu chí số liệu**: Cung cấp con số cụ thể như "loại bỏ bài viết dưới 200 từ", "loại bỏ bài viết cũ hơn 48 giờ."
5. **Hệ thống chấm điểm**: Yêu cầu chấm điểm độ tin cậy (1-10) và mức liên quan (1-10).

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

**Cài đặt đầu ra Task:**

| Tùy chọn | Mô tả |
|----------|-------|
| `markdown: true` | Xử lý đầu ra ở định dạng markdown |
| `output_file: output/content_harvest.md` | Tự động lưu kết quả vào file chỉ định |
| `create_directory: true` | Tự động tạo thư mục `output/` nếu chưa tồn tại |

**2. Summarization Task - Task tóm tắt**

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

Task này yêu cầu **hệ thống tóm tắt 3 cấp**. Đây là mẫu thiết kế thực tế đáp ứng nhu cầu của các đối tượng độc giả khác nhau:
- Tóm tắt cấp tweet -> Để chia sẻ mạng xã hội
- Tóm tắt cho lãnh đạo -> Cho chuyên gia bận rộn
- Tóm tắt chi tiết -> Cho độc giả cần hiểu sâu

**3. Final Report Assembly Task - Task lắp ráp báo cáo cuối cùng**

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

Task này tổng hợp kết quả của hai task trước đó để tạo bản tin tức **sẵn sàng xuất bản**.

#### Thay đổi trong file chính (`main.py`)

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

**Thay đổi chính:**

1. Tên lớp thay đổi từ `TranslatorCrew` thành `NewsReaderAgent`
2. Tên phương thức Crew đơn giản hóa từ `assemble_crew` thành `crew`
3. `kickoff()` chưa có `inputs` (sẽ thêm trong phần tiếp theo)
4. Tại thời điểm này, agent chưa được kết nối công cụ (giai đoạn thiết kế)

### Điểm thực hành

- Thay đổi `backstory` của agent chi tiết hơn hoặc ngắn gọn hơn và so sánh sự khác biệt chất lượng kết quả.
- Thay đổi định dạng `expected_output` để thử nghiệm cấu trúc đầu ra khác nhau.
- Thay đổi đường dẫn `output_file` trong `tasks.yaml` và xác nhận file được tạo đúng.
- Thêm agent thứ tư (ví dụ: dịch giả) và thiết kế pipeline dịch báo cáo cuối cùng sang tiếng Hàn.

---

## 3.4 News Reader Crew

### Chủ đề và mục tiêu

Trong phần cuối cùng này, chúng ta kết nối **công cụ thực tế** với hệ thống đọc tin tức đã thiết kế để hoàn thành Crew hoạt động đầy đủ. Chúng ta triển khai công cụ tìm kiếm web và thu thập web, chỉ định mô hình LLM phù hợp cho mỗi agent, và chạy hệ thống với chủ đề thực ("Cambodia Thailand War") để xác minh kết quả.

### Giải thích khái niệm cốt lõi

#### Công cụ Production

Trong 3.2, chúng ta tạo công cụ đơn giản bọc `len()`, nhưng trong phần này, chúng ta triển khai công cụ tương tác với dịch vụ web thực:

1. **Công cụ tìm kiếm (Search Tool)**: Tìm kiếm Google bằng Serper API
2. **Công cụ thu thập (Scrape Tool)**: Trích xuất nội dung trang web bằng Playwright + BeautifulSoup

#### Công cụ tích hợp sẵn vs Công cụ tùy chỉnh

| Phân loại | Công cụ tích hợp | Công cụ tùy chỉnh |
|-----------|-----------------|-------------------|
| Ví dụ | `SerperDevTool` | `scrape_tool` |
| Ưu điểm | Thiết lập đơn giản, sẵn sàng sử dụng ngay | Kiểm soát hoàn toàn, đáp ứng yêu cầu đặc biệt |
| Nhược điểm | Giới hạn tùy chỉnh | Phải tự triển khai |

#### Chỉ định mô hình LLM theo Agent

Sử dụng cùng mô hình cho tất cả agent là không hiệu quả. Chỉ định mô hình khác nhau theo độ phức tạp tác vụ giúp tiết kiệm chi phí đồng thời đảm bảo chất lượng cao ở nơi cần thiết.

### Phân tích code

#### Triển khai công cụ (`tools.py`)

**1. Công cụ tìm kiếm - SerperDevTool**

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

- `SerperDevTool`: Công cụ tìm kiếm tích hợp do CrewAI cung cấp. Lấy kết quả tìm kiếm Google qua Serper API.
- `n_results=30`: Lấy tối đa 30 kết quả tìm kiếm. Thiết lập rộng rãi cho thu thập tin tức toàn diện.
- Sử dụng yêu cầu thiết lập `SERPER_API_KEY` trong file `.env`.

**2. Công cụ thu thập - Triển khai tùy chỉnh**

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

**Phân tích luồng hoạt động code:**

1. **Khởi chạy trình duyệt Playwright**: Khởi chạy trình duyệt Chromium ở chế độ headless (không hiển thị) bằng `sync_playwright()`. Cho phép xử lý trang web động render bằng JavaScript.

2. **Tải trang và đợi**: Sau `page.goto(url)`, đợi 5 giây bằng `time.sleep(5)`. Cung cấp thời gian cho JavaScript render và nội dung động tải xong.

3. **Phân tích HTML**: Phân tích HTML bằng `BeautifulSoup`.

4. **Loại bỏ thẻ không mong muốn**: Loại bỏ tất cả thẻ được định nghĩa trong danh sách `unwanted_tags`. Lọc nhiễu như navigation, quảng cáo và script, chỉ trích xuất văn bản bài viết thuần túy.

5. **Trích xuất văn bản**: Trích xuất tất cả văn bản kết nối bằng khoảng trắng sử dụng `soup.get_text(separator=" ")`.

6. **Trả về an toàn**: Trả về "No content" nếu nội dung rỗng để agent nhận biết lỗi.

> **Tại sao dùng Playwright thay vì `requests`?**
> Hầu hết trang tin tức hiện đại render nội dung động bằng JavaScript. Thư viện `requests` chỉ lấy HTML tĩnh, nên nội dung bài viết có thể bị thiếu. Playwright chạy trình duyệt thực, nên có thể lấy DOM hoàn chỉnh sau khi JavaScript thực thi.

#### Chỉ định mô hình LLM cho Agent (`config/agents.yaml`)

```yaml
news_hunter_agent:
  # ... (cài đặt hiện có)
  llm: openai/o4-mini-2025-04-16

summarizer_agent:
  # ... (cài đặt hiện có)
  llm: openai/o4-mini-2025-04-16   # Đổi từ o3 sang o4-mini

curator_agent:
  # ... (cài đặt hiện có)
  llm: openai/o4-mini-2025-04-16
```

`openai/o3` được chỉ định cho summarizer_agent trong 3.3 đã được thay đổi thành `openai/o4-mini-2025-04-16`. Tất cả agent hiện sử dụng cùng mô hình, phản ánh sự cân bằng giữa hiệu quả chi phí và hiệu suất đủ tốt. Định dạng trường `llm` là `provider/model-name`.

#### Hoàn thành file chính (`main.py`)

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

    # ... định nghĩa task tương tự ...

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

**Chiến lược phân bổ công cụ theo Agent:**

| Agent | Công cụ | Lý do |
|-------|---------|-------|
| `news_hunter_agent` | `search_tool`, `scrape_tool` | Cần tìm bài viết qua tìm kiếm và đọc nội dung qua thu thập |
| `summarizer_agent` | `scrape_tool` | Đọc lại bài viết từ URL task trước để tóm tắt chi tiết |
| `curator_agent` | (không có) | Chỉ biên tập kết quả tóm tắt từ task trước, không cần công cụ bên ngoài |

**Thực thi `kickoff()` và xử lý kết quả:**

```python
result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

- `inputs={"topic": "Cambodia Thailand War."}`: Thay thế placeholder `{topic}` trong YAML.
- `result.tasks_output`: Trả về kết quả thực thi mỗi task dưới dạng danh sách. Vì có 3 task nên có 3 kết quả.

#### Đầu ra thực thi

Sau khi Crew thực thi, 3 file markdown được tạo trong thư mục `output/`:

**1. `output/content_harvest.md` - Danh sách bài viết thu thập**

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

Có thể thấy news_hunter_agent đã thu thập 3 bài viết đáng tin cậy (Reuters, AP News, BBC) và chấm điểm độ tin cậy và mức liên quan cho mỗi bài.

**2. `output/summary.md` - Tóm tắt 3 cấp**

Cho mỗi bài viết, tóm tắt cấp tweet (dưới 280 ký tự), tóm tắt cho lãnh đạo (150-200 từ), và tóm tắt chi tiết (500-700 từ) được tạo ra. Định dạng tuân thủ trung thành những gì được chỉ định trong `expected_output`.

**3. `output/final_report.md` - Bản tin cuối cùng**

Báo cáo cuối cùng tổng hợp tất cả thông tin thành bản tin chất lượng xuất bản. Bản tin chuyên nghiệp có cấu trúc với các phần như Executive Summary, Lead Story, Breaking News và Editor's Analysis.

### Điểm thực hành

- Thử thay đổi `topic` trong `inputs` sang chủ đề khác (ví dụ: "AI regulation 2025", "climate change policy").
- Thử nghiệm đánh đổi tốc độ-ổn định bằng cách điều chỉnh giá trị `time.sleep(5)` trong `scrape_tool`.
- Thử sửa danh sách `unwanted_tags` để cải thiện chất lượng trích xuất.
- Thay đổi `n_results=30` thành giá trị nhỏ hơn hoặc lớn hơn để quan sát ảnh hưởng của phạm vi tìm kiếm.
- Thử thêm agent mới (ví dụ: kiểm tra sự thật) để mở rộng pipeline.

---

## Tóm tắt điểm chính của chương

### 1. Kiến trúc cốt lõi của CrewAI

- **Agent**: Nhân viên AI có vai trò, mục tiêu và câu chuyện nền. Cấu hình trong YAML và khởi tạo trong Python.
- **Task**: Hướng dẫn công việc cụ thể. Các yếu tố chính là `description`, `expected_output` và phân công `agent`.
- **Crew**: Đơn vị đội nhóm gộp Agent và Task để thực thi. Khởi chạy bằng `kickoff()`.
- **Tool**: Hàm cung cấp khả năng bên ngoài cho agent. Tạo bằng decorator `@tool`.

### 2. Quy ước cấu trúc dự án

```
project/
├── config/
│   ├── agents.yaml    # Định nghĩa agent (role, goal, backstory)
│   └── tasks.yaml     # Định nghĩa task (description, expected_output)
├── main.py            # Lớp @CrewBase và code thực thi
├── tools.py           # Định nghĩa công cụ tùy chỉnh
├── output/            # Lưu trữ file kết quả task
└── pyproject.toml     # Quản lý phụ thuộc
```

### 3. Nguyên tắc Prompt Engineering

- **Backstory chi tiết**: Thiết lập chuyên môn và tính cách cụ thể cho agent cải thiện chất lượng đầu ra.
- **Hướng dẫn từng bước**: Đánh số các tác vụ cần thực hiện theo thứ tự trong `description`.
- **Tiêu chí cụ thể**: Loại bỏ sự mơ hồ bằng số liệu, ví dụ, và mẫu cho phép/từ chối.
- **Mẫu định dạng đầu ra**: Cung cấp mẫu định dạng markdown trong `expected_output` cho kết quả nhất quán.

### 4. Nguyên tắc thiết kế công cụ

- Docstring quyết định thời điểm công cụ được sử dụng. Phải viết rõ ràng và chi tiết.
- Type hint là bắt buộc. LLM sử dụng chúng để truyền đối số đúng.
- Chỉ gán công cụ agent cần thiết. Công cụ không cần thiết gây nhầm lẫn.

### 5. Mẫu thiết kế đa Agent

- **Nguyên tắc chuyên biệt hóa**: Mỗi agent tập trung vào một lĩnh vực chuyên môn.
- **Mẫu pipeline**: Task thực thi tuần tự, kết quả mỗi task trở thành đầu vào task tiếp theo.
- **Tối ưu mô hình**: Có thể chỉ định mô hình LLM khác nhau cho mỗi agent dựa trên độ phức tạp tác vụ.

---

## Bài tập thực hành

### Bài tập 1: Cơ bản - Tạo Crew đầu tiên của bạn

**Mục tiêu**: Tự triển khai cấu trúc cơ bản của CrewAI.

**Yêu cầu**:
1. Tạo Crew với 2 agent:
   - `writer_agent`: Agent viết bài ngắn về chủ đề cho trước
   - `reviewer_agent`: Agent đánh giá bài viết và cung cấp phản hồi
2. Viết `role`, `goal`, và `backstory` phù hợp cho mỗi agent
3. Định nghĩa 2 task:
   - `writing_task`: Viết bài 300 từ về `{topic}`
   - `review_task`: Đánh giá bài viết về ngữ pháp, logic và khả năng đọc
4. Chạy với `verbose=True` và quan sát quá trình suy nghĩ của agent

### Bài tập 2: Trung cấp - Sử dụng công cụ tùy chỉnh

**Mục tiêu**: Tạo công cụ tùy chỉnh thực tế và kết nối với agent.

**Yêu cầu**:
1. Triển khai các công cụ tùy chỉnh sau:
   - `get_weather(city: str)`: Gọi API thời tiết để trả về thời tiết hiện tại (dùng API miễn phí)
   - `calculate(expression: str)`: Tính biểu thức toán học và trả về kết quả
2. Tạo `travel_planner_agent` và kết nối cả hai công cụ
3. Định nghĩa `plan_trip_task` kiểm tra thời tiết thành phố cụ thể và lập kế hoạch du lịch
4. Thay đổi docstring và quan sát thay đổi trong mẫu gọi công cụ

### Bài tập 3: Nâng cao - Mở rộng News Reader

**Mục tiêu**: Mở rộng hệ thống đọc tin tức đã xây dựng trong chương.

**Yêu cầu**:
1. Thêm các agent sau vào hệ thống đọc tin tức hiện có:
   - `translator_agent`: Dịch báo cáo cuối cùng sang tiếng Hàn
   - `fact_checker_agent`: Xác minh chéo sự thật giữa các bài viết
2. Chỉ định mô hình LLM khác cho `translator_agent` (ví dụ: `openai/gpt-4o`)
3. Thiết kế và kết nối công cụ phù hợp cho `fact_checker_agent`
4. Cấu hình task sao cho pipeline 5 giai đoạn hoạt động tuần tự
5. Chạy với nhiều chủ đề và so sánh/phân tích kết quả trong mỗi `output_file`

### Bài tập 4: Thử thách - Thiết kế đội Agent tự trị

**Mục tiêu**: Thiết kế hệ thống đa agent áp dụng được vào công việc thực tế từ đầu.

**Yêu cầu**:
1. Chọn lĩnh vực bạn quan tâm (tài chính, giáo dục, sức khỏe, v.v.)
2. Thiết kế ít nhất 3 agent chuyên biệt
3. Tạo và kết nối ít nhất 1 công cụ tùy chỉnh cho mỗi agent
4. Viết `backstory` chi tiết và định dạng `expected_output` cụ thể
5. Chạy toàn bộ hệ thống và thiết lập tiêu chí đánh giá chất lượng kết quả
6. Lưu kết quả dưới dạng file markdown và chuẩn bị tài liệu trình bày

---

> **Xem trước chương tiếp theo**: Trong Chapter 4, chúng ta học các tính năng nâng cao của CrewAI như giao tiếp giữa agent, thực thi task có điều kiện và hệ thống bộ nhớ để xây dựng hệ thống agent tinh vi hơn.
