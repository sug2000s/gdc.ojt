# Chương 6: Dự án Multi-Agent Nâng cao với AutoGen

---

## Tổng quan chương

Trong chương này, chúng ta sẽ sử dụng framework **AutoGen** của Microsoft để xây dựng hệ thống multi-agent ở mức production. Vượt ra ngoài các khái niệm agent cơ bản đã học ở các chương trước, chúng ta sẽ học các pattern nâng cao nơi nhiều agent hợp tác như các **Đội (Team)**.

Chương này tiến hành thông qua hai dự án cốt lõi:

1. **Đội Tối ưu hóa Email (Email Optimizer Team)**: Sử dụng `RoundRobinGroupChat` để xây dựng pipeline nơi các agent lần lượt cải thiện email theo thứ tự
2. **Bản sao Deep Research (Deep Research Clone)**: Sử dụng `SelectorGroupChat` để xây dựng hệ thống nghiên cứu thông minh nơi AI tự động chọn agent phù hợp để thực hiện nghiên cứu web

Thông qua hai dự án này, bạn có thể hiểu sâu **hai pattern cốt lõi của điều phối agent** -- pipeline tuần tự và lựa chọn động.

### Mục tiêu học tập

- Hiểu khái niệm Đội (Team) và pattern group chat của framework AutoGen
- Thiết kế pipeline multi-agent tuần tự sử dụng `RoundRobinGroupChat`
- Triển khai hệ thống lựa chọn agent động sử dụng `SelectorGroupChat`
- Học cách kết nối công cụ bên ngoài (Tool) với agent
- Điều khiển workflow sử dụng điều kiện kết thúc (Termination Condition)
- Thực hành tích hợp API tìm kiếm web thực tế (Firecrawl) vào agent

### Stack công nghệ

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.13+ | Runtime |
| AutoGen (autogen) | >= 0.9.7 | Core framework agent |
| autogen-agentchat | >= 0.7.2 | Tính năng team/group chat |
| autogen-ext[openai] | >= 0.7.2 | Tích hợp mô hình OpenAI |
| firecrawl-py | >= 2.16.5 | Tìm kiếm và scraping web |
| python-dotenv | >= 1.1.1 | Quản lý biến môi trường |
| ipykernel | >= 6.30.1 | Thực thi Jupyter notebook |
| gpt-4o-mini | - | Mô hình LLM |

---

## 6.0 Giới thiệu dự án và thiết lập môi trường

### Chủ đề và mục tiêu

Trong phần này, chúng ta thiết lập cấu trúc cơ bản của dự án "Deep Research Clone". Chúng ta định nghĩa phụ thuộc dự án thông qua file `pyproject.toml` và học cách khởi tạo dự án Python sử dụng trình quản lý gói **uv**.

### Khái niệm cốt lõi

#### pyproject.toml là gì?

`pyproject.toml` là file cấu hình tiêu chuẩn cho các dự án Python hiện đại. Nó thay thế các cách tiếp cận cũ `setup.py` và `requirements.txt`, cho phép bạn quản lý metadata dự án và phụ thuộc trong một file duy nhất.

#### Trình quản lý gói uv

Dự án này sử dụng **uv**, trình quản lý gói Python thế hệ tiếp theo. Được viết bằng Rust, uv cung cấp tốc độ nhanh hơn pip 10-100 lần và xử lý tạo môi trường ảo và quản lý phụ thuộc một cách tích hợp.

### Phân tích mã nguồn

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

**Phân tích phụ thuộc:**

| Gói | Vai trò |
|-----|---------|
| `autogen` | Core framework AutoGen. Nền tảng cho việc tạo và quản lý agent |
| `autogen-agentchat` | Cung cấp tính năng group chat (team). `RoundRobinGroupChat`, `SelectorGroupChat`, v.v. |
| `autogen-ext[openai]` | Mở rộng tích hợp mô hình OpenAI (GPT-4o-mini, v.v.) |
| `firecrawl-py` | Client API tìm kiếm web và trích xuất nội dung trang web |
| `python-dotenv` | Tải biến môi trường như API key từ file `.env` |
| `ipykernel` | Python kernel cho Jupyter notebook (phụ thuộc phát triển) |

**Điểm đáng chú ý:**
- `requires-python = ">=3.13"` yêu cầu phiên bản Python mới nhất
- Nhóm `dev` trong `[dependency-groups]` tách riêng các gói chỉ cần trong môi trường phát triển
- Dấu ngoặc vuông `[openai]` trong `autogen-ext[openai]` chỉ phụ thuộc tùy chọn gọi là "extras"

### Điểm thực hành

1. **Khởi tạo dự án**: Bạn có thể bắt đầu dự án với các lệnh terminal sau:
   ```bash
   mkdir deep-research-clone
   cd deep-research-clone
   uv init
   ```

2. **Cài đặt phụ thuộc**: Cài đặt phụ thuộc sử dụng uv:
   ```bash
   uv add autogen autogen-agentchat "autogen-ext[openai]" firecrawl-py python-dotenv
   uv add --dev ipykernel
   ```

3. **Thiết lập biến môi trường**: Tạo file `.env` để quản lý API key:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   echo "FIRECRAWL_API_KEY=fc-your-key-here" >> .env
   ```

---

## 6.1 Đội Tối ưu hóa Email (Email Optimizer Team)

### Chủ đề và mục tiêu

Trong phần này, chúng ta xây dựng pipeline sử dụng **RoundRobinGroupChat** nơi 5 agent chuyên biệt lần lượt cải thiện email. Mỗi agent chịu trách nhiệm cho một lĩnh vực chuyên môn riêng (rõ ràng, giọng điệu, thuyết phục, tổng hợp, phê bình) và làm việc tuần tự theo kiểu round-robin.

### Khái niệm cốt lõi

#### RoundRobinGroupChat

`RoundRobinGroupChat` là pattern team đơn giản nhất nhưng mạnh mẽ được cung cấp bởi AutoGen. Nó hoạt động bằng cách các agent tham gia **lần lượt theo thứ tự cố định**.

```
Đầu vào người dùng → ClarityAgent → ToneAgent → PersuasionAgent → SynthesizerAgent → CriticAgent
                  ↑                                                                    |
                  └────────────────── (quay lại nếu không đạt tiêu chuẩn) ─────────────┘
```

Pattern này phù hợp cho các tình huống sau:
- Khi vai trò của mỗi agent được định nghĩa rõ ràng
- Khi cần cải thiện lặp đi lặp lại (iterative refinement)
- Khi thứ tự workflow cố định

#### Điều kiện kết thúc (Termination Conditions)

AutoGen cung cấp điều kiện kết thúc để điều khiển thực thi team. Trong dự án này, hai điều kiện kết thúc được kết hợp:

- **TextMentionTermination**: Kết thúc khi văn bản cụ thể (ví dụ: "TERMINATE") xuất hiện trong phản hồi của agent
- **MaxMessageTermination**: Kết thúc khi đạt số tin nhắn tối đa

Khi hai điều kiện này được kết hợp bằng toán tử `|` (OR), team dừng khi **bất kỳ điều kiện nào được thỏa mãn**.

#### Chuyên biệt hóa Agent (Agent Specialization)

Cốt lõi của hệ thống multi-agent là gán **một vai trò rõ ràng** cho mỗi agent. Nhiều agent chuyên biệt hợp tác tạo ra kết quả tốt hơn một agent "làm tất cả" duy nhất. Điều này tương tự **Nguyên tắc Trách nhiệm Đơn lẻ (Single Responsibility Principle)** trong kỹ thuật phần mềm.

### Phân tích mã nguồn

#### Bước 1: Import và thiết lập mô hình

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
```

Vai trò của mỗi import:
- `RoundRobinGroupChat`: Class team group chat tuần tự
- `AssistantAgent`: Class agent dựa trên mô hình AI
- `OpenAIChatCompletionClient`: Client giao tiếp với OpenAI API
- `MaxMessageTermination`, `TextMentionTermination`: Các class điều kiện kết thúc
- `Console`: Tiện ích UI để xuất kết quả thực thi team theo thời gian thực trên console

#### Bước 2: Định nghĩa các Agent chuyên biệt

Cốt lõi của dự án này là thiết kế 5 agent chuyên biệt. Chú ý cách `system_message` của mỗi agent hạn chế và tập trung vai trò của họ.

**ClarityAgent (Chuyên gia Rõ ràng)**

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

> **Điểm thiết kế**: Lưu ý hạn chế phạm vi rõ ràng "Don't worry about persuasion or tone." Việc giới hạn rõ phạm vi vai trò như thế này ngăn xung đột vai trò với các agent khác.

**ToneAgent (Chuyên gia Giọng điệu)**

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

> **Điểm thiết kế**: Yêu cầu rõ ràng sự cân bằng giữa giọng điệu cảm xúc và tính chuyên nghiệp. Bằng cách đồng thời hướng dẫn các yếu tố có thể mâu thuẫn -- "warm, confident, and human" vs "professional and appropriate" -- nó hướng dẫn kết quả cân bằng.

**PersuasionAgent (Chuyên gia Thuyết phục)**

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

> **Điểm thiết kế**: Bằng cách chỉ định các lĩnh vực chuyên môn cụ thể -- marketing, tâm lý hành vi và copywriting -- chất lượng phản hồi của agent được nâng cao. Các chỉ dẫn hành động cụ thể như "Remove weak or passive language" được bao gồm.

**SynthesizerAgent (Chuyên gia Tổng hợp)**

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

> **Điểm thiết kế**: Agent này thực hiện vai trò meta **tổng hợp** kết quả của ba agent trước đó. Hướng dẫn "read all prior agent responses" khuyến khích sử dụng tích cực ngữ cảnh cuộc hội thoại trước đó. Đây là giá trị cốt lõi của agent Synthesizer trong pattern RoundRobin.

**CriticAgent (Chuyên gia Phê bình)**

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

> **Điểm thiết kế**: CriticAgent đóng vai trò "người gác cổng". Nếu tiêu chuẩn chất lượng được đáp ứng, nó xuất "TERMINATE" để kết thúc team; nếu không, nó đưa ra gợi ý cải thiện để kích hoạt vòng tiếp theo. Hướng dẫn "dont settle" duy trì tiêu chuẩn chất lượng cao.

#### Bước 3: Thiết lập điều kiện kết thúc

```python
text_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=30)

termination_condition = text_termination | max_messages_termination
```

Ý nghĩa cấu hình này:
- Nếu CriticAgent xuất "TERMINATE" -> **kết thúc ngay lập tức** (chất lượng được chấp nhận)
- Nếu đạt tối đa 30 tin nhắn -> **kết thúc bắt buộc** (ngăn vòng lặp vô hạn)
- Toán tử `|` có nghĩa là điều kiện OR. Kết thúc xảy ra khi bất kỳ điều kiện nào được thỏa mãn

> **Nguyên tắc thiết kế**: Luôn bao gồm `MaxMessageTermination` như một mạng an toàn. Điều này ngăn các tình huống agent không đạt được đồng thuận và quay vòng vô hạn.

#### Bước 4: Tạo và thực thi Team

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

**Phân tích chính:**

- **Thứ tự danh sách `participants` chính là thứ tự thực thi**. ClarityAgent đi trước, CriticAgent đi cuối.
- `run_stream()` chạy team theo cách streaming bất đồng bộ. Bọc bằng `Console` cho phép kiểm tra phản hồi của mỗi agent theo thời gian thực.
- Từ khóa `await` cho biết mã này chạy trong môi trường bất đồng bộ (async). Trong Jupyter notebook, `await` cấp cao nhất được hỗ trợ tự động.

#### Phân tích kết quả thực thi

Hãy xem xét kết quả thực thi thực tế từng bước:

**1) Đầu vào (User)**:
```
Hi! Im hungry, buy me lunch and invest in my business. Thanks.
```
Email không trang trọng, trực tiếp, không chuyên nghiệp.

**2) Đầu ra ClarityAgent**:
```
Hi! I'm hungry. Please buy me lunch and invest in my business. Thank you.
```
Tập trung vào sửa lỗi ngữ pháp ("Im" -> "I'm"), tách câu, và thêm biểu đạt lịch sự.

**3) Đầu ra ToneAgent**:
```
Subject: A Quick Favor

Hi there!
I hope you're doing well! I find myself feeling a bit peckish today...
Warm regards,
[Your Name]
```
Tái cấu trúc hoàn toàn với giọng ấm áp và chuyên nghiệp. Thêm dòng tiêu đề, lời chào và chữ ký.

**4) Đầu ra PersuasionAgent**:
```
Subject: Let's Make Delicious Opportunities Happen!

Hi [Recipient's Name],
...I promise to make it worth your while...
Together, we can turn potential into profit!
```
Thêm ngôn ngữ thuyết phục ("turn potential into profit"), lời kêu gọi hành động ("I promise to make it worth your while"), và nhấn mạnh lợi ích.

**5) Đầu ra SynthesizerAgent**:
Viết email cuối cùng tích hợp các yếu tố tốt nhất từ ba agent trước.

**6) Đầu ra CriticAgent**:
```
The email meets professional standards.
TERMINATE
```
Đạt tiêu chuẩn chất lượng và xuất "TERMINATE," kết thúc thực thi team.

### File công cụ: tools.py

Đội Tối ưu hóa Email không cần công cụ, nhưng file `tools.py` được tạo cùng cho phần tiếp theo.

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

**Phân tích mã:**

Công cụ tìm kiếm web này sử dụng Firecrawl API:

1. **Khởi tạo FirecrawlApp**: Tạo client bằng cách tải API key từ biến môi trường
2. **Thực thi tìm kiếm**: Lấy tối đa 5 kết quả tìm kiếm ở định dạng markdown cho truy vấn
3. **Làm sạch kết quả**: Hai giai đoạn xử lý regex để loại bỏ các yếu tố không cần thiết
   - `re.sub(r"\\+|\n+", "", markdown)`: Loại bỏ xuống dòng và dấu gạch chéo ngược thừa
   - `re.sub(r"\[[^\]]+\]\([^\)]+\)|https?://[^\s]+", "", cleaned)`: Loại bỏ liên kết markdown và URL

> **Tại sao làm sạch văn bản?** Kết quả web scraping chứa nhiều yếu tố không cần thiết như liên kết điều hướng và URL quảng cáo. Loại bỏ chúng giảm số token gửi đến LLM và cho phép tập trung vào thông tin thiết yếu.

### Điểm thực hành

1. **Thí nghiệm thứ tự Agent**: Thay đổi thứ tự danh sách `participants` và xem kết quả khác nhau như thế nào. Ví dụ, điều gì xảy ra nếu PersuasionAgent được đặt trước ClarityAgent?

2. **Thêm/Xóa Agent**: Thêm agent chuyên biệt mới (ví dụ: "BrevityAgent" -- chuyên gia ngắn gọn) hoặc xóa agent hiện có và quan sát sự khác biệt trong kết quả.

3. **Thay đổi điều kiện kết thúc**: Giảm `MaxMessageTermination` xuống 10 hoặc tăng lên 50. Làm tiêu chuẩn của CriticAgent nghiêm ngặt hơn để nhiều vòng được thực thi.

4. **Kiểm thử đầu vào đa dạng**: Kiểm thử với nhiều loại đầu vào khác nhau như email trang trọng, email xin lỗi, email bán hàng, v.v.

---

## 6.2 Deep Research (Nghiên cứu Chuyên sâu)

### Chủ đề và mục tiêu

Trong phần này, chúng ta xây dựng hệ thống nghiên cứu thông minh clone tính năng "Deep Research" của OpenAI sử dụng **SelectorGroupChat**. Khác với cách tiếp cận thứ tự cố định (RoundRobin) ở phần trước, đây triển khai điều phối động nơi AI phân tích ngữ cảnh cuộc hội thoại và **tự động quyết định agent nào nên hành động tiếp theo**.

### Khái niệm cốt lõi

#### SelectorGroupChat vs RoundRobinGroupChat

So sánh sự khác biệt chính giữa hai pattern team:

| Đặc điểm | RoundRobinGroupChat | SelectorGroupChat |
|-----------|-------------------|-------------------|
| Phương thức chọn Agent | Thứ tự cố định (tuần tự) | AI chọn động |
| Tính linh hoạt | Thấp | Cao |
| Khả năng dự đoán | Cao | Tương đối thấp |
| Tình huống phù hợp | Pipeline, chuỗi kiểm tra | Workflow phức tạp, tác vụ có phân nhánh |
| Chi phí thêm | Không | Thêm lệnh gọi LLM để chọn |

#### Cách SelectorGroupChat hoạt động

```
Đầu vào câu hỏi người dùng
       ↓
  ┌─────────────────┐
  │  Selector LLM   │ ← tham chiếu selector_prompt + lịch sử hội thoại
  │  (chọn agent)   │
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
                  Selector LLM chọn
                  agent tiếp theo
                          ↓
                      (lặp lại...)
```

`SelectorGroupChat` sử dụng **lệnh gọi LLM riêng biệt** mỗi lượt để chọn agent tiếp theo. Nó tham chiếu các quy tắc workflow được định nghĩa trong `selector_prompt` và lịch sử hội thoại hiện tại.

#### UserProxyAgent

`UserProxyAgent` là agent bao gồm **con người** như một thành viên của team. Nó được sử dụng khi team agent cần yêu cầu phản hồi của con người hoặc nhận phê duyệt. Phản hồi của con người được nhận qua đầu vào tiêu chuẩn thông qua `input_func=input`.

Đây là triển khai pattern **Human-in-the-Loop (HITL)**. Là điểm cân bằng giữa tự động hóa hoàn toàn và thủ công hoàn toàn, AI thực hiện phần lớn công việc trong khi quyết định quan trọng do con người đưa ra.

### Phân tích mã nguồn

#### Bước 1: Import và thiết lập mô hình

```python
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from tools import web_search_tool, save_report_to_md
```

Thay đổi chính so với 6.1:
- Sử dụng `SelectorGroupChat` thay vì `RoundRobinGroupChat`
- Thêm `UserProxyAgent` (Human-in-the-Loop)
- Import công cụ tùy chỉnh (`web_search_tool`, `save_report_to_md`)

#### Bước 2: Định nghĩa 6 Agent chuyên biệt

Dự án này định nghĩa 6 agent, mỗi agent chịu trách nhiệm cho một giai đoạn của quy trình nghiên cứu.

**research_planner (Người lập kế hoạch nghiên cứu)**

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

> **Điểm thiết kế**: Lưu ý tham số `description`. Trong `SelectorGroupChat`, `description` này được sử dụng để giải thích vai trò của agent cho Selector LLM. Trong khi không cần thiết trong cách tiếp cận RoundRobin, nó là **bắt buộc** trong cách tiếp cận lựa chọn động.

**research_agent (Người thực thi nghiên cứu web)**

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

> **Điểm chính**: `tools=[web_search_tool]` kết nối công cụ tìm kiếm web. Chỉ agent này có thể thực sự truy cập web bên ngoài. Bằng cách chỉ gán công cụ cho agent cụ thể, các lệnh tìm kiếm không cần thiết được ngăn chặn và vai trò được xác định rõ ràng.

**research_analyst (Nhà phân tích nghiên cứu)**

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

> **Điểm thiết kế**: System message cung cấp **cấu trúc chính xác** (tiêu đề phần) của báo cáo. Chỉ định định dạng đầu ra có cấu trúc như vậy mang lại kết quả nhất quán. Sử dụng từ tín hiệu "REPORT_COMPLETE" để chỉ hoàn thành tác vụ hoạt động như một **giao thức** với các agent khác (quality_reviewer).

**quality_reviewer (Người kiểm tra chất lượng)**

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

> **Điểm chính**: Agent này có công cụ `save_report_to_md` được kết nối. Bằng cách chỉ cung cấp cho agent kiểm tra chất lượng một công cụ có **tác dụng phụ (side effect)** (lưu file), workflow được điều khiển sao cho báo cáo chỉ được lưu sau khi vượt qua kiểm tra chất lượng.

**research_enhancer (Chuyên gia bổ sung nghiên cứu)**

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

> **Điểm thiết kế**: Các hướng dẫn "ONLY CRITICAL gaps" và "We prioritize getting a good report done rather than perfect coverage" nhằm **ngăn vòng lặp nghiên cứu quá mức**. Agent cầu toàn liên tục yêu cầu tìm kiếm thêm là vấn đề phổ biến trong hệ thống multi-agent.

**user_proxy (Đại diện người dùng)**

```python
user_proxy = UserProxyAgent(
    "user_proxy",
    description="Human reviewer who can request additional research or approve final results",
    input_func=input,
)
```

> Đóng vai trò nhận phê duyệt của con người cho báo cáo cuối cùng. Khi người dùng nhập "APPROVED," toàn bộ workflow kết thúc.

#### Bước 3: Selector Prompt (Prompt chọn Agent)

Đây là phần quan trọng nhất của dự án. Selector Prompt là chỉ dẫn được SelectorGroupChat sử dụng để xác định agent nào sẽ được chọn tiếp theo.

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

**Phân tích chi tiết:**

1. **Biến template `{roles}` và `{history}`**: AutoGen tự động chèn mô tả agent và lịch sử hội thoại. Điều này cho phép Selector LLM nắm bắt trạng thái hiện tại.

2. **Quy tắc workflow rõ ràng**: Workflow 7 bước được định nghĩa rõ ràng dưới dạng điều kiện. Đây là pattern tương tự **State Machine (Máy trạng thái)**:
   ```
   Bắt đầu → [Lập kế hoạch] → [Tìm kiếm] → [Đánh giá bổ sung] → [Viết báo cáo] → [Kiểm tra chất lượng] → [Phê duyệt người dùng] → Kết thúc
                      ↑           |
                      └───────────┘ (nếu cần bổ sung)
   ```

3. **Bảo vệ an toàn**: "After research_agent has searched 2 times maximum, proceed to research_analyst regardless." -- Quy tắc này ngăn vòng lặp tìm kiếm vô hạn.

4. **Bổ ngữ như "use sparingly"**: Cung cấp gợi ý cho Selector LLM về tần suất sử dụng agent cụ thể.

#### Bước 4: Tạo và thực thi Team

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

**So sánh chính (6.1 vs 6.2):**

| Yếu tố | 6.1 Email Optimizer | 6.2 Deep Research |
|---------|-------------------|-------------------|
| Class Team | `RoundRobinGroupChat` | `SelectorGroupChat` |
| Từ khóa kết thúc | "TERMINATE" | "APPROVED" |
| Tin nhắn tối đa | 30 | 50 |
| `selector_prompt` | Không có | Định nghĩa workflow chi tiết |
| `model_client` (cấp team) | Không có | Cần LLM để chọn agent |
| Thứ tự `participants` | Quyết định thứ tự thực thi | Thứ tự không liên quan |

> **Lưu ý**: Trong `SelectorGroupChat`, thứ tự danh sách `participants` không ảnh hưởng đến thứ tự thực thi. Thay vào đó, các quy tắc được định nghĩa trong `selector_prompt` quyết định thứ tự.

#### Bước 5: Thực thi

```python
await Console(
    team.run_stream(task="Research about the new development in Nuclear Energy"),
)
```

Một dòng này bắt đầu toàn bộ pipeline nghiên cứu. Luồng thực thi như sau:

1. Selector LLM chọn `research_planner`
2. research_planner tạo chủ đề cốt lõi và truy vấn tìm kiếm
3. Selector LLM chọn `research_agent`
4. research_agent thực hiện tìm kiếm web thực tế bằng công cụ tìm kiếm web
5. Selector LLM chọn `research_enhancer`
6. research_enhancer đánh giá mức đủ của nghiên cứu
7. Selector LLM chọn `research_analyst`
8. research_analyst viết báo cáo tổng hợp và xuất "REPORT_COMPLETE"
9. Selector LLM chọn `quality_reviewer`
10. quality_reviewer lưu báo cáo vào `report.md`
11. Selector LLM chọn `user_proxy`
12. Kết thúc khi người dùng nhập "APPROVED"

#### Thay đổi trong tools.py (6.1 -> 6.2)

```python
# Thay đổi 1: Giảm kết quả tìm kiếm (5 → 2)
response = app.search(
    query=query,
    limit=2,  # Trước đó: limit=5
    scrape_options=ScrapeOptions(
        formats=["markdown"],
    ),
)

# Thay đổi 2: Thêm hàm công cụ mới
def save_report_to_md(content: str) -> str:
    """Save report content to report.md file."""
    with open("report.md", "w") as f:
        f.write(content)
    return "report.md"
```

**Phân tích thay đổi:**

1. **`limit=5` -> `limit=2`**: Giảm kết quả tìm kiếm nhằm **tiết kiệm token** và **tối ưu chi phí**. Trong nghiên cứu chuyên sâu, nhiều lần tìm kiếm được thực hiện, nên lấy quá nhiều kết quả mỗi lần tìm kiếm nhanh chóng cạn kiệt context window.

2. **Thêm `save_report_to_md`**: Công cụ lưu kết quả nghiên cứu vào file. Được kết nối với agent `quality_reviewer`, đảm bảo chỉ báo cáo vượt qua kiểm tra chất lượng mới được lưu.

#### Ví dụ báo cáo được tạo (report.md)

Cấu trúc báo cáo được tạo từ việc thực thi:

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

Báo cáo này được tạo tự động dựa trên dữ liệu thực tế tìm kiếm từ web bởi các agent. Nhờ system message có cấu trúc, báo cáo chuyên nghiệp với định dạng nhất quán được tạo ra.

### Điểm thực hành

1. **Kiểm thử chủ đề nghiên cứu khác**: Thử chạy với nhiều chủ đề khác nhau như "Research about the impact of AI on healthcare." Quan sát cách agent điều chỉnh truy vấn tìm kiếm phù hợp với chủ đề.

2. **Sửa đổi Selector Prompt**: Thay đổi quy tắc workflow và xem hành vi khác nhau như thế nào. Ví dụ, thử tạo quy tắc bỏ qua research_enhancer và đi thẳng từ research_agent sang research_analyst.

3. **Thêm Agent**: Thêm agent "fact_checker" để chèn bước xác minh tính chính xác thực tế của báo cáo.

4. **Mở rộng công cụ**: Tạo công cụ bổ sung ngoài `web_search_tool` (ví dụ: tìm kiếm bài báo học thuật, tìm kiếm chỉ tin tức) và cung cấp cho research_agent.

5. **Biến thể Human-in-the-Loop**: Thay đổi thời điểm can thiệp của `user_proxy`. Ví dụ, sửa đổi để phê duyệt người dùng cũng được yêu cầu ở giai đoạn lập kế hoạch nghiên cứu.

---

## Tóm tắt trọng tâm chương

### 1. Hai Pattern điều phối Team

| Pattern | Class | Phương thức chọn | Khi nào sử dụng |
|---------|-------|------------------|-----------------|
| **Pipeline tuần tự** | `RoundRobinGroupChat` | Thứ tự cố định | Tác vụ có vai trò rõ ràng và thứ tự cố định |
| **Lựa chọn động** | `SelectorGroupChat` | Chọn dựa trên AI | Workflow phức tạp có phân nhánh |

### 2. Nguyên tắc thiết kế Agent

- **Trách nhiệm đơn lẻ**: Mỗi agent chỉ nên thực hiện một vai trò rõ ràng
- **Hạn chế phạm vi**: Cũng chỉ định "không nên làm gì" trong system_message (ví dụ: "Don't worry about persuasion or tone")
- **Sử dụng description**: Trong SelectorGroupChat, description được sử dụng trực tiếp để chọn agent
- **Giao thức từ tín hiệu**: Xây dựng giao thức giao tiếp giữa các agent bằng từ khóa như "TERMINATE", "REPORT_COMPLETE", "APPROVED"

### 3. Thiết kế điều kiện kết thúc

- Luôn **kết hợp** kết thúc ngữ nghĩa (TextMentionTermination) với **mạng an toàn** (MaxMessageTermination)
- Kết hợp bằng toán tử `|` (OR) để có điều kiện kết thúc linh hoạt
- Chỉ sử dụng kết thúc ngữ nghĩa mà không có mạng an toàn có nguy cơ vòng lặp vô hạn

### 4. Chiến lược phân bổ công cụ

- Phân bổ công cụ **chọn lọc chỉ cho agent cần chúng**
- Gán công cụ có tác dụng phụ (như lưu file) cho **agent xác minh** để triển khai quality gate
- Giới hạn phù hợp số kết quả cho công cụ tìm kiếm web xem xét chi phí token

### 5. Mẹo thiết kế Selector Prompt

- Định nghĩa workflow từ góc nhìn **state machine**
- Mô tả agent nào cần chọn ở mỗi trạng thái (điều kiện) bằng **quy tắc rõ ràng**
- Bao gồm **quy tắc bảo vệ an toàn** (ví dụ: "2 times maximum, proceed regardless")
- Cung cấp gợi ý về tần suất sử dụng agent (ví dụ: "use sparingly", "ONLY at the start")

---

## Bài tập thực hành

### Bài tập 1: Xây dựng Đội Review Code (Cơ bản)

**Mục tiêu**: Xây dựng đội review code sử dụng `RoundRobinGroupChat`.

**Yêu cầu**:
- SecurityAgent: Kiểm tra lỗ hổng bảo mật
- PerformanceAgent: Phân tích vấn đề hiệu suất
- ReadabilityAgent: Đánh giá khả năng đọc code
- SummaryAgent: Tổng hợp tất cả review thành phản hồi cuối
- ApprovalAgent: Phê duyệt cuối hoặc yêu cầu sửa đổi (điều khiển kết thúc)

**Gợi ý**: Tham khảo cấu trúc đội tối ưu email, nhưng sửa system_messages cho phù hợp review code.

### Bài tập 2: Tối ưu Selector Prompt (Trung cấp)

**Mục tiêu**: Sửa đổi `selector_prompt` trong hệ thống deep research 6.2 để thêm các tính năng sau.

**Yêu cầu**:
- Thêm agent `fact_checker` và sửa workflow để fact_checker chạy sau khi research_analyst viết báo cáo nhưng trước quality_reviewer
- fact_checker xác minh chéo các khẳng định chính của báo cáo thông qua tìm kiếm web
- Thêm bước mới vào quy tắc workflow của selector_prompt

### Bài tập 3: Mở rộng Hệ thống Deep Research của riêng bạn (Nâng cao)

**Mục tiêu**: Mở rộng hệ thống deep research 6.2 để triển khai các tính năng sau.

**Yêu cầu**:
1. Chia công cụ tìm kiếm thành 2: `academic_search_tool` (chỉ bài báo học thuật) và `news_search_tool` (chỉ tin tức)
2. Gán mỗi công cụ cho agent riêng biệt
3. Cho phép người dùng chọn định dạng báo cáo từ đầu vào ban đầu của `user_proxy`
4. Lưu báo cáo cuối cùng ở cả hai định dạng markdown và PDF

**Gợi ý**:
- Bạn có thể thêm bộ lọc domain vào phương thức `search()` của Firecrawl để tách kết quả học thuật/tin tức
- Mở rộng `save_report_to_md` để tạo công cụ `save_report_to_pdf`

### Bài tập 4: Gỡ lỗi hợp tác Agent (Phân tích)

**Mục tiêu**: Phân tích tình huống sau và đề xuất giải pháp.

**Kịch bản**: Trong hệ thống deep research, research_agent lặp lại tìm kiếm hơn 5 lần và không chuyển sang research_analyst.

**Câu hỏi**:
1. Đề xuất 3 nguyên nhân có thể cho vấn đề này
2. Nên sửa selector_prompt như thế nào để giải quyết vấn đề?
3. Những bảo vệ an toàn nào có thể thêm ở cấp độ mã?

---

## Tài liệu tham khảo

- [Tài liệu chính thức AutoGen](https://microsoft.github.io/autogen/)
- [Tài liệu API Firecrawl](https://docs.firecrawl.dev/)
- Mã nguồn dự án: thư mục `deep-research-clone/`
  - `email-optimizer-team.ipynb`: Notebook đội tối ưu email
  - `deep-research-team.ipynb`: Notebook đội deep research
  - `tools.py`: Công cụ tìm kiếm web và lưu file
  - `report.md`: Ví dụ báo cáo nghiên cứu được tạo
  - `pyproject.toml`: Định nghĩa phụ thuộc dự án
