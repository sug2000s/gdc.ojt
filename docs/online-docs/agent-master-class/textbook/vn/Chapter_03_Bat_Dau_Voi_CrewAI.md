# Chương 3: Bắt đầu với CrewAI

---

## Tổng quan chương

Trong chương này, chúng ta sẽ học cách xây dựng AI Agent bằng framework **CrewAI** từ đầu theo từng bước. CrewAI là framework Python được thiết kế để nhiều AI Agent cộng tác thực hiện các tác vụ phức tạp, tổ chức agent và task xung quanh khái niệm "Crew (Đội)".

Chương này gồm 4 phần, bắt đầu từ agent dịch thuật đơn giản và dần phát triển thành hệ thống agent đọc tin tức ở mức thực chiến.

| Phần | Chủ đề | Nội dung học tập chính |
|------|--------|------------------------|
| 3.1 | Your First CrewAI Agent | Cấu trúc dự án, khái niệm cơ bản Agent/Task/Crew, cấu hình YAML |
| 3.2 | Custom Tools | Tạo công cụ tùy chỉnh, decorator `@tool`, kết nối công cụ với agent |
| 3.3 | News Reader Tasks and Agents | Thiết kế agent thực chiến, viết prompt chi tiết, cấu hình đa agent |
| 3.4 | News Reader Crew | Tích hợp công cụ thực tế (tìm kiếm/scraping), chỉ định model LLM, chạy Crew và xác nhận kết quả |

### Mục tiêu học tập

Sau khi hoàn thành chương này, bạn có thể:

1. Thiết lập và cấu trúc dự án CrewAI từ đầu
2. Sử dụng file cấu hình YAML để định nghĩa agent và task
3. Tạo công cụ tùy chỉnh và kết nối với agent
4. Xây dựng và chạy Crew với nhiều agent cộng tác tuần tự
5. Xây dựng pipeline thu thập/tóm tắt/biên tập tin tức ở mức thực chiến

### Yêu cầu trước

- Python 3.13 trở lên
- Trình quản lý gói `uv` (quản lý dự án Python)
- OpenAI API key (thiết lập trong file `.env`)
- Hiểu biết cơ bản về cú pháp Python

---

## 3.1 Your First CrewAI Agent

### Chủ đề và mục tiêu

Trong phần đầu tiên, chúng ta tìm hiểu cấu trúc cơ bản của CrewAI và tạo một **Agent dịch thuật (Translator Agent)** đơn giản. Qua quá trình này, nắm bắt mối quan hệ giữa 3 yếu tố cốt lõi của CrewAI: **Agent**, **Task**, **Crew**, và học vai trò của file cấu hình YAML.

### Giải thích khái niệm chính

#### 3 khái niệm cốt lõi của CrewAI

CrewAI gồm ba thành phần cốt lõi:

1. **Agent (Tác nhân)**: Nhân viên AI có vai trò (role), mục tiêu (goal), câu chuyện nền (backstory) cụ thể. Ví von như một thành viên trong đội.
2. **Task (Tác vụ)**: Công việc cụ thể mà agent cần thực hiện. Bao gồm mô tả công việc (description) và kết quả mong đợi (expected_output).
3. **Crew (Đội)**: Đơn vị nhóm gom agent và task lại để thực thi. Các agent xử lý task theo thứ tự.

```
┌─────────────────────────────────────────┐
│                  Crew                   │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Agent   │───>│     Task 1       │   │
│  │(Dịch giả)│    │(Anh→Ý)          │   │
│  └──────────┘    └──────────────────┘   │
│       │                   │             │
│       │          (Kết quả sang task kế) │
│       │                   ▼             │
│       │          ┌──────────────────┐   │
│       └─────────>│     Task 2       │   │
│                  │(Ý→Hy Lạp)       │   │
│                  └──────────────────┘   │
└─────────────────────────────────────────┘
```

#### Decorator `@CrewBase` và cấu trúc dự án

CrewAI sử dụng **mẫu lớp dựa trên decorator**. Khi áp dụng decorator `@CrewBase` vào lớp, CrewAI tự động đọc file `config/agents.yaml` và `config/tasks.yaml`, cung cấp dưới dạng dictionary `self.agents_config` và `self.tasks_config`.

#### Cấu trúc thư mục dự án

```
news-reader-agent/
├── .python-version          # Chỉ định phiên bản Python (3.13)
├── .gitignore               # Danh sách file Git bỏ qua
├── pyproject.toml           # Cấu hình dự án và phụ thuộc
├── config/
│   ├── agents.yaml          # Định nghĩa agent
│   └── tasks.yaml           # Định nghĩa task
├── main.py                  # File thực thi chính
└── uv.lock                  # File khóa phụ thuộc
```

Cấu trúc này tuân theo quy ước (convention) của CrewAI. Đặt file YAML trong thư mục `config/` sẽ được `@CrewBase` tự động nhận diện.

### Phân tích mã

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

- `crewai[tools]`: Cài đặt framework CrewAI cùng gói mở rộng công cụ. `[tools]` là extras, bao gồm các phụ thuộc liên quan đến công cụ bổ sung.
- `python-dotenv`: Thư viện tải biến môi trường từ file `.env`. Dùng để quản lý API key an toàn.

#### Cấu hình agent (`config/agents.yaml`)

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

**Vai trò của từng trường:**

| Trường | Mô tả | Vai trò trong ví dụ |
|--------|--------|---------------------|
| `role` | Định nghĩa chức vụ/vai trò của agent | Dịch giả Anh-Ý |
| `goal` | Mục tiêu agent cần đạt được | Dịch chính xác, tránh hiểu lầm |
| `backstory` | Thiết lập nền tảng (trao tính cách và chuyên môn) | Người song ngữ lớn lên ở New York và Palermo |

> **Tại sao `backstory` quan trọng?**
> `backstory` không chỉ là trang trí. LLM sử dụng thông tin nền này làm ngữ cảnh khi tạo phản hồi, tạo ra kết quả nhất quán và chuyên nghiệp hơn. Ví dụ, thông tin "có thể phát hiện sự khác biệt văn hóa" hướng dẫn phản ánh sắc thái văn hóa khi dịch.

#### Cấu hình task (`config/tasks.yaml`)

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

**Điểm quan trọng:**

- `{sentence}`: **Placeholder biến** bao bọc bằng dấu ngoặc nhọn. Khi thực thi, giá trị được truyền qua `kickoff(inputs={"sentence": "..."})` sẽ thay thế.
- `expected_output`: Thông báo rõ ràng cho agent về hình thức kết quả. Có điều này agent mới hiểu chính xác cần trả về gì.
- `agent`: Tên agent thực hiện task này. Phải khớp với key đã định nghĩa trong `agents.yaml`.
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

**Phân tích chi tiết luồng hoạt động mã:**

1. **`dotenv.load_dotenv()`**: Tải biến môi trường như `OPENAI_API_KEY` từ file `.env`. CrewAI sử dụng key này nội bộ để gọi LLM API.

2. **Decorator `@CrewBase`**: Khi áp dụng vào lớp, tự động đọc `config/agents.yaml` và `config/tasks.yaml`. Qua đó truy cập được `self.agents_config` và `self.tasks_config`.

3. **Decorator `@agent`**: Thông báo cho CrewAI rằng phương thức trả về đối tượng Agent. Tên phương thức (`translator_agent`) phải khớp với key trong YAML.

4. **Decorator `@task`**: Thông báo cho CrewAI rằng phương thức trả về đối tượng Task. Task được thực thi **theo thứ tự định nghĩa**. `translate_task` → `retranslate_task` theo thứ tự.

5. **Decorator `@crew`**: Áp dụng vào phương thức tổ hợp đối tượng Crew.
   - `self.agents`: Danh sách tất cả agent được đánh dấu `@agent` (tự động thu thập)
   - `self.tasks`: Danh sách tất cả task được đánh dấu `@task` (tự động thu thập)
   - `verbose=True`: Xuất chi tiết quá trình thực thi ra console.

6. **`kickoff(inputs={...})`**: Chạy Crew. Giá trị trong dictionary `inputs` thay thế placeholder `{sentence}` trong YAML.

> **Luồng kết quả thực thi:**
> 1. `translate_task`: "I'm Nico and I like to ride my bicicle in Napoli" → Dịch sang tiếng Ý
> 2. `retranslate_task`: Kết quả dịch tiếng Ý → Dịch sang tiếng Hy Lạp
>
> Task thứ hai **tự động nhận kết quả của task đầu tiên làm ngữ cảnh**. Đây là cách task chaining hoạt động trong CrewAI.

### Điểm thực hành

- Thiết lập `OPENAI_API_KEY=sk-...` trong file `.env`, sau đó chạy `uv run python main.py`.
- Quan sát quá trình suy nghĩ (Chain of Thought) của agent qua `verbose=True`.
- Thay đổi `backstory` và so sánh sự khác biệt về chất lượng kết quả.
- Thử thêm task thứ ba (ví dụ: dịch Hy Lạp→Tiếng Việt).

---

## 3.2 Custom Tools

### Chủ đề và mục tiêu

Trong phần này, chúng ta học cách kết nối **công cụ tùy chỉnh (Custom Tool)** với agent CrewAI. LLM về cơ bản chỉ có thể tạo văn bản, nhưng thông qua công cụ có thể thực hiện các chức năng bên ngoài (tính toán, gọi API, đọc file, v.v.).

### Giải thích khái niệm chính

#### Tool (Công cụ) là gì?

**Công cụ (Tool)** của AI Agent là hàm mà agent có thể sử dụng ngoài khả năng tạo văn bản của LLM. Agent khi thực hiện task sẽ tự động gọi công cụ phù hợp nếu phán đoán rằng "cần công cụ cho tác vụ này".

```
┌──────────────────────────────────────┐
│           Agent                      │
│                                      │
│  "Cần đếm số chữ cái trong câu...  │
│   Sử dụng công cụ count_letters!"   │
│                                      │
│   ┌─────────────────────────────┐    │
│   │  Tool: count_letters        │    │
│   │  Input: "Hello World"       │    │
│   │  Output: 11                 │    │
│   └─────────────────────────────┘    │
└──────────────────────────────────────┘
```

Lý do công cụ quan trọng là LLM có thể gây **ảo giác (hallucination)** trong tính toán toán học hay truy vấn dữ liệu chính xác. Ngay cả tác vụ đơn giản như đếm chữ cái cũng có thể sai nếu LLM tự làm, nhưng cung cấp hàm `len()` như công cụ sẽ đảm bảo kết quả chính xác.

#### Decorator `@tool`

CrewAI cung cấp decorator `@tool` để chuyển đổi hàm Python thông thường thành công cụ mà agent có thể sử dụng. **Docstring** của hàm đóng vai trò giải thích mục đích công cụ cho agent.

### Phân tích mã

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

**Phân tích cốt lõi:**

- **Decorator `@tool`**: Chuyển đổi hàm thành công cụ CrewAI. Nội bộ phân tích signature và docstring của hàm để tạo schema công cụ mà LLM hiểu được.
- **Type hint `sentence: str`**: Bắt buộc. CrewAI sử dụng type hint này để thông báo cho LLM về định dạng tham số đầu vào.
- **Docstring**: Được agent sử dụng để phán đoán "khi nào nên dùng công cụ này?". Phải viết rõ ràng và chi tiết. Nên mô tả hình thức đầu vào và đầu ra.
- **Câu lệnh `print()`**: Dùng cho debug. Xác nhận công cụ có thực sự được gọi không và nhận đầu vào gì.
- **`return len(sentence)`**: Logic thực tế. Thay vì LLM tự đếm chữ cái, sử dụng hàm `len()` của Python để trả về kết quả chính xác.

#### Thêm agent và task mới (`config/agents.yaml`)

```yaml
counter_agent:
  role: >
    To count the lenght of things.
  goal: >
    To be a good counter that never lies or makes things up.
  backstory: >
    You are a genius counter.
```

Chú ý biểu đạt "never lies or makes things up" trong `goal`. Đây là kỹ thuật prompt hướng dẫn agent không đoán mà bắt buộc phải sử dụng công cụ.

#### Thêm task (`config/tasks.yaml`)

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

**Điểm quan trọng:**

- `tools=[count_letters]`: Truyền danh sách công cụ vào tham số `tools` khi tạo Agent. Có thể kết nối nhiều công cụ cho một agent.
- Công cụ được kết nối ở **cấp agent** (không phải cấp task). Agent có thể sử dụng công cụ đã gán cho bất kỳ task nào.
- `count_task` không chỉ định công cụ trực tiếp. Vì agent phụ trách task (`counter_agent`) đã có công cụ.

### Điểm thực hành

- Quan sát đầu ra `print()` để xác nhận thời điểm công cụ thực sự được gọi.
- Thay đổi docstring thành mơ hồ và kiểm tra agent có gọi công cụ đúng không.
- Tạo công cụ mới (ví dụ: đếm từ, chuyển đổi chữ hoa) và thêm vào agent.
- Kết nối nhiều công cụ vào một agent và quan sát agent chọn công cụ phù hợp theo tình huống.

---

## 3.3 News Reader Tasks and Agents

### Chủ đề và mục tiêu

Trong phần này, chúng ta tái cấu trúc hoàn toàn ví dụ dịch thuật/đếm đơn giản trước đó thành **hệ thống đọc tin tức thực chiến**. Thiết kế 3 agent chuyên biệt và 3 task chi tiết, đồng thời học kỹ thuật prompt engineering cấp production.

### Giải thích khái niệm chính

#### Kiến trúc đa agent

Hệ thống đọc tin tức tuân theo cấu trúc **pipeline 3 giai đoạn**:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ News Hunter  │────>│  Summarizer  │────>│   Curator    │
│   Agent      │     │    Agent     │     │    Agent     │
│              │     │              │     │              │
│ Thu thập     │     │ Tóm tắt     │     │ Biên tập &   │
│ tin tức &    │     │ bài viết    │     │ báo cáo      │
│ lọc          │     │ (3 cấp)     │     │ cuối cùng    │
└──────────────┘     └──────────────┘     └──────────────┘
  Task 1:              Task 2:              Task 3:
  content_harvesting   summarization        final_report_assembly

  output:              output:              output:
  content_harvest.md   summary.md           final_report.md
```

Mỗi agent có **lĩnh vực chuyên môn khác nhau**. Đây là ưu điểm cốt lõi của hệ thống đa agent. Thay vì một agent làm tất cả, mỗi agent tập trung vào chuyên môn riêng sẽ tạo ra kết quả tốt hơn.

#### Prompt Engineering: Viết `backstory` chi tiết

Thay đổi quan trọng nhất trong phần này là **độ sâu và chi tiết** của cấu hình agent. Backstory 2 dòng đơn giản ở 3.1 trở thành profile chi tiết hơn 10 dòng.

#### Thiết lập `output_file` của task

Cấu hình mỗi task tự động **lưu kết quả vào file markdown**. Qua đó có thể xác nhận và debug kết quả trung gian ở mỗi bước.

### Phân tích mã

#### Cấu hình agent (`config/agents.yaml`)

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
|----------|--------|
| `verbose: true` | Xuất chi tiết quá trình suy nghĩ của agent |
| `inject_date: true` | Tự động đưa ngày hiện tại vào ngữ cảnh agent. Cần thiết để đánh giá tính kịp thời của tin tức |

**Phân tích `backstory` - Tại sao viết chi tiết như vậy:**

- "15 years of experience": Đặt mức độ chuyên môn để hướng dẫn LLM đưa ra phán đoán chất lượng cao
- "separate signal from noise": Nhấn mạnh khả năng lọc để hướng dẫn loại bỏ bài viết không liên quan
- "detecting bias and misinformation": Kích hoạt chức năng đánh giá độ tin cậy
- "source diversity": Hướng dẫn thu thập thông tin từ nhiều nguồn đa dạng

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

**Cấu hình mới: `llm: openai/o3`**

Có thể **chỉ định model LLM khác** cho agent cụ thể. Công việc tóm tắt cần mức hiểu biết và diễn đạt cao nên sử dụng model mạnh hơn (o3). Cấu hình model khác nhau cho từng agent giúp tối ưu hóa chi phí và hiệu suất.

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

#### Cấu hình task (`config/tasks.yaml`)

**1. Content Harvesting Task - Task thu thập tin tức**

Task này chứa chỉ dẫn chi tiết nhất. Hãy xem phần cốt lõi:

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

1. **Chỉ dẫn theo từng bước (Step-by-step)**: Đánh số để chỉ rõ công việc thực hiện theo thứ tự.
2. **Quy tắc lọc rõ ràng**: Nhấn mạnh "IMPORTANT" bằng chữ hoa, cung cấp mẫu URL cho phép/từ chối kèm ví dụ cụ thể.
3. **Định dạng checklist**: Phân biệt trực quan bằng ký hiệu cho phép (✅) và từ chối (❌).
4. **Cung cấp tiêu chí số**: Đưa ra con số cụ thể như "loại bỏ bài viết dưới 200 từ", "loại bỏ bài viết quá 48 giờ".
5. **Hệ thống chấm điểm**: Yêu cầu chấm điểm tin cậy (1-10) và liên quan (1-10).

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

**Cấu hình đầu ra task:**

| Tùy chọn | Mô tả |
|----------|--------|
| `markdown: true` | Xử lý đầu ra ở định dạng markdown |
| `output_file: output/content_harvest.md` | Tự động lưu kết quả vào file chỉ định |
| `create_directory: true` | Tự động tạo thư mục `output/` nếu không tồn tại |

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

Task này yêu cầu **hệ thống tóm tắt 3 cấp**. Đây là mẫu thiết kế thực chiến đáp ứng nhu cầu của nhiều đối tượng độc giả:
- Tóm tắt cấp tweet → Chia sẻ trên mạng xã hội
- Tóm tắt điều hành → Cho chuyên gia bận rộn
- Tóm tắt chi tiết → Cho độc giả cần hiểu sâu

**3. Final Report Assembly Task - Task báo cáo cuối cùng**

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

Task này tổng hợp kết quả của hai task trước để tạo bản tin tức **ở mức có thể xuất bản**.

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

1. Tên lớp đổi từ `TranslatorCrew` → `NewsReaderAgent`
2. Tên phương thức Crew đơn giản hóa từ `assemble_crew` → `crew`
3. `kickoff()` chưa có `inputs` (sẽ thêm ở phần tiếp theo)
4. Tại thời điểm này agent chưa được kết nối công cụ (giai đoạn thiết kế)

### Điểm thực hành

- Thay đổi `backstory` của agent chi tiết hơn hoặc ngắn gọn hơn và so sánh sự khác biệt chất lượng kết quả.
- Thay đổi định dạng `expected_output` để thử nghiệm cấu trúc đầu ra khác.
- Thay đổi đường dẫn `output_file` trong `tasks.yaml` và xác nhận file được tạo đúng.
- Thêm agent thứ tư (ví dụ: dịch giả) để thiết kế pipeline dịch báo cáo cuối cùng sang tiếng Việt.

---

## 3.4 News Reader Crew

### Chủ đề và mục tiêu

Trong phần cuối cùng, chúng ta **kết nối công cụ thực tế** với hệ thống đọc tin tức đã thiết kế để hoàn thành Crew hoạt động đầy đủ. Triển khai công cụ tìm kiếm web và scraping web, chỉ định model LLM phù hợp cho từng agent, và chạy hệ thống với chủ đề thực tế ("Cambodia Thailand War") để xác nhận kết quả.

### Giải thích khái niệm chính

#### Công cụ thực chiến (Production Tools)

Ở phần 3.2, chúng ta đã tạo công cụ đơn giản bao bọc hàm `len()`, nhưng phần này triển khai công cụ tương tác với dịch vụ web thực tế:

1. **Công cụ tìm kiếm (Search Tool)**: Tìm kiếm Google sử dụng Serper API
2. **Công cụ scraping (Scrape Tool)**: Trích xuất nội dung trang web sử dụng Playwright + BeautifulSoup

#### Công cụ tích hợp CrewAI vs Công cụ tùy chỉnh

| Phân loại | Công cụ tích hợp | Công cụ tùy chỉnh |
|-----------|-------------------|---------------------|
| Ví dụ | `SerperDevTool` | `scrape_tool` |
| Ưu điểm | Cấu hình đơn giản, sử dụng ngay | Kiểm soát hoàn toàn, đáp ứng yêu cầu đặc biệt |
| Nhược điểm | Hạn chế tùy chỉnh | Cần tự triển khai |

#### Chỉ định model LLM theo agent

Sử dụng cùng model cho tất cả agent là không hiệu quả. Chỉ định model khác nhau theo độ phức tạp công việc giúp tiết kiệm chi phí đồng thời đảm bảo chất lượng cao nơi cần thiết.

### Phân tích mã

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

- `SerperDevTool`: Công cụ tìm kiếm tích hợp của CrewAI. Lấy kết quả tìm kiếm Google qua Serper API.
- `n_results=30`: Lấy tối đa 30 kết quả tìm kiếm. Thiết lập rộng rãi để đảm bảo tính toàn diện của việc thu thập tin tức.
- Cần thiết lập `SERPER_API_KEY` trong file `.env`.

**2. Công cụ scraping - Triển khai tùy chỉnh**

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

**Phân tích luồng hoạt động mã:**

1. **Khởi động trình duyệt Playwright**: Chạy trình duyệt Chromium ở chế độ headless (không hiển thị) với `sync_playwright()`. Cho phép xử lý cả trang web động được render bằng JavaScript.

2. **Tải trang và chờ**: Sau `page.goto(url)`, chờ 5 giây với `time.sleep(5)`. Đảm bảo thời gian cho JavaScript render và tải nội dung động hoàn thành.

3. **Phân tích HTML**: Phân tích HTML bằng `BeautifulSoup`.

4. **Loại bỏ thẻ không cần thiết**: Loại bỏ tất cả thẻ trong danh sách `unwanted_tags`. Lọc bỏ nhiễu từ navigation, quảng cáo, script và chỉ trích xuất văn bản bài viết thuần túy.

5. **Trích xuất văn bản**: Trích xuất tất cả văn bản nối bằng khoảng trắng với `soup.get_text(separator=" ")`.

6. **Trả về an toàn**: Nếu nội dung rỗng, trả về "No content" để agent nhận biết thất bại.

> **Tại sao dùng Playwright thay vì `requests`?**
> Hầu hết trang tin tức hiện đại render nội dung động bằng JavaScript. Thư viện `requests` chỉ lấy HTML tĩnh nên có thể bỏ sót nội dung bài viết. Playwright chạy trình duyệt thực tế nên có thể lấy DOM hoàn chỉnh sau khi JavaScript thực thi.

#### Chỉ định model LLM cho agent (`config/agents.yaml`)

```yaml
news_hunter_agent:
  # ... (cấu hình hiện có)
  llm: openai/o4-mini-2025-04-16

summarizer_agent:
  # ... (cấu hình hiện có)
  llm: openai/o4-mini-2025-04-16   # Đổi từ o3 sang o4-mini

curator_agent:
  # ... (cấu hình hiện có)
  llm: openai/o4-mini-2025-04-16
```

Từ `openai/o3` đã chỉ định cho summarizer_agent ở 3.3 đổi thành `openai/o4-mini-2025-04-16`. Tất cả agent sử dụng cùng model, phản ánh sự cân bằng giữa hiệu quả chi phí và hiệu suất đủ. Định dạng trường `llm` là `provider/model-name`.

#### Hoàn thiện file chính (`main.py`)

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

    # ... định nghĩa task giống như trước ...

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

**Chiến lược gán công cụ theo agent:**

| Agent | Công cụ | Lý do |
|-------|---------|-------|
| `news_hunter_agent` | `search_tool`, `scrape_tool` | Cần tìm kiếm bài viết và đọc nội dung bằng scraping |
| `summarizer_agent` | `scrape_tool` | Đọc lại bài viết từ URL của task trước để tóm tắt chi tiết |
| `curator_agent` | (không có) | Chỉ biên tập kết quả tóm tắt từ task trước nên không cần công cụ bên ngoài |

**Thực thi `kickoff()` và xử lý kết quả:**

```python
result = NewsReaderAgent().crew().kickoff(
    inputs={"topic": "Cambodia Thailand War."}
)

for task_output in result.tasks_output:
    print(task_output)
```

- `inputs={"topic": "Cambodia Thailand War."}`: Thay thế placeholder `{topic}` trong YAML.
- `result.tasks_output`: Trả về danh sách kết quả thực thi từng task. Có 3 task nên bao gồm 3 kết quả.

#### Kết quả đầu ra

Sau khi chạy Crew, 3 file markdown được tạo trong thư mục `output/`:

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

Có thể thấy news_hunter_agent đã thu thập 3 bài viết đáng tin cậy (Reuters, AP News, BBC) và chấm điểm tin cậy và liên quan cho từng bài.

**2. `output/summary.md` - Tóm tắt 3 cấp**

Mỗi bài viết được tạo tóm tắt cấp tweet (dưới 280 ký tự), tóm tắt điều hành (150-200 từ), tóm tắt chi tiết (500-700 từ). Tuân thủ định dạng đã chỉ định trong `expected_output`.

**3. `output/final_report.md` - Bản tin cuối cùng**

Báo cáo cuối cùng tổng hợp tất cả thông tin ở mức xuất bản. Cấu trúc chuyên nghiệp gồm Executive Summary, Lead Story, Breaking News, Editor's Analysis, v.v.

### Điểm thực hành

- Thay đổi `topic` trong `inputs` sang chủ đề khác (ví dụ: "AI regulation 2025", "climate change policy") và chạy thử.
- Điều chỉnh giá trị `time.sleep(5)` trong `scrape_tool` để thử nghiệm trade-off giữa tốc độ và ổn định.
- Sửa danh sách `unwanted_tags` để cải thiện chất lượng trích xuất.
- Thay đổi `n_results=30` thành giá trị nhỏ hơn hoặc lớn hơn để quan sát ảnh hưởng của phạm vi tìm kiếm.
- Thêm agent mới (ví dụ: fact-checker) để mở rộng pipeline.

---

## Tổng kết chương

### 1. Kiến trúc cốt lõi của CrewAI

- **Agent**: Nhân viên AI có vai trò, mục tiêu, câu chuyện nền. Cấu hình bằng YAML và khởi tạo trong Python.
- **Task**: Chỉ dẫn công việc cụ thể. Chỉ định `description`, `expected_output`, `agent` là cốt lõi.
- **Crew**: Đơn vị nhóm gom Agent và Task để thực thi. Chạy bằng `kickoff()`.
- **Tool**: Hàm cung cấp chức năng bên ngoài cho agent. Tạo bằng decorator `@tool`.

### 2. Quy ước cấu trúc dự án

```
project/
├── config/
│   ├── agents.yaml    # Định nghĩa agent (role, goal, backstory)
│   └── tasks.yaml     # Định nghĩa task (description, expected_output)
├── main.py            # Lớp @CrewBase và mã thực thi
├── tools.py           # Định nghĩa công cụ tùy chỉnh
├── output/            # Lưu file kết quả task
└── pyproject.toml     # Quản lý phụ thuộc
```

### 3. Nguyên tắc Prompt Engineering

- **Backstory chi tiết**: Thiết lập cụ thể chuyên môn và tính cách agent cải thiện chất lượng kết quả.
- **Chỉ dẫn theo bước**: Đánh số trong `description` để chỉ rõ công việc thực hiện theo thứ tự.
- **Tiêu chí cụ thể**: Loại bỏ sự mơ hồ bằng con số, ví dụ, mẫu cho phép/từ chối.
- **Template định dạng đầu ra**: Cung cấp template markdown trong `expected_output` để nhận kết quả nhất quán.

### 4. Nguyên tắc thiết kế công cụ

- Docstring quyết định thời điểm sử dụng công cụ. Phải viết rõ ràng và chi tiết.
- Type hint là bắt buộc. Được LLM sử dụng để truyền đối số đúng.
- Chỉ gán công cụ cần thiết cho agent. Công cụ không cần thiết gây nhầm lẫn.

### 5. Mẫu thiết kế đa agent

- **Nguyên tắc chuyên biệt hóa**: Mỗi agent tập trung vào một lĩnh vực chuyên môn.
- **Mẫu pipeline**: Task thực thi tuần tự, kết quả task trước trở thành đầu vào task sau.
- **Tối ưu hóa model**: Có thể chỉ định model LLM khác nhau cho từng agent theo độ phức tạp công việc.

---

## Bài tập thực hành

### Bài tập 1: Cơ bản - Tạo Crew đầu tiên của bạn

**Mục tiêu**: Tự triển khai cấu trúc cơ bản của CrewAI.

**Yêu cầu**:
1. Tạo Crew với 2 agent:
   - `writer_agent`: Agent viết bài ngắn về chủ đề cho trước
   - `reviewer_agent`: Agent đánh giá và phản hồi bài viết
2. Viết `role`, `goal`, `backstory` phù hợp cho từng agent
3. Định nghĩa 2 task:
   - `writing_task`: Viết bài 300 từ về `{topic}`
   - `review_task`: Đánh giá ngữ pháp, logic, khả năng đọc của bài viết
4. Chạy với `verbose=True` và quan sát quá trình suy nghĩ của agent

### Bài tập 2: Trung cấp - Sử dụng công cụ tùy chỉnh

**Mục tiêu**: Tạo công cụ tùy chỉnh thực dụng và kết nối với agent.

**Yêu cầu**:
1. Triển khai công cụ tùy chỉnh sau:
   - `get_weather(city: str)`: Gọi API thời tiết và trả về thời tiết hiện tại (sử dụng API miễn phí)
   - `calculate(expression: str)`: Tính toán biểu thức toán học và trả về kết quả
2. Tạo `travel_planner_agent` và kết nối cả hai công cụ
3. Định nghĩa `plan_trip_task` để kiểm tra thời tiết thành phố cụ thể và lập kế hoạch du lịch
4. Thay đổi docstring và quan sát sự thay đổi mẫu gọi công cụ

### Bài tập 3: Nâng cao - Mở rộng News Reader

**Mục tiêu**: Mở rộng hệ thống đọc tin tức đã xây dựng trong chương.

**Yêu cầu**:
1. Thêm agent sau vào news reader hiện có:
   - `translator_agent`: Dịch báo cáo cuối cùng sang tiếng Việt
   - `fact_checker_agent`: Xác minh chéo sự thật giữa các bài viết
2. Chỉ định model LLM riêng cho `translator_agent` (ví dụ: `openai/gpt-4o`)
3. Thiết kế và kết nối công cụ phù hợp cho `fact_checker_agent`
4. Cấu hình task để pipeline 5 bước hoạt động tuần tự
5. Chạy với nhiều chủ đề khác nhau và phân tích so sánh kết quả từng `output_file`

### Bài tập 4: Thử thách - Thiết kế đội agent tự trị

**Mục tiêu**: Thiết kế hệ thống đa agent có thể áp dụng thực tế từ đầu.

**Yêu cầu**:
1. Chọn lĩnh vực bạn quan tâm (tài chính, giáo dục, sức khỏe, v.v.)
2. Thiết kế tối thiểu 3 agent chuyên biệt
3. Tạo tối thiểu 1 công cụ tùy chỉnh cho mỗi agent
4. Viết `backstory` chi tiết và định dạng `expected_output` cụ thể
5. Chạy toàn bộ hệ thống và xây dựng tiêu chí đánh giá chất lượng kết quả
6. Lưu kết quả vào file markdown và chuẩn bị tài liệu thuyết trình

---

> **Giới thiệu chương tiếp theo**: Chapter 4 sẽ học các tính năng nâng cao của CrewAI như giao tiếp giữa các agent, thực thi task có điều kiện, hệ thống bộ nhớ, v.v. để xây dựng hệ thống agent tinh vi hơn.
