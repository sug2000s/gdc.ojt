# Chương 5: CrewAI Flows - Xây dựng Agent Pipeline Nội dung

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ học cách xây dựng pipeline tạo nội dung dựa trên AI từ đầu đến cuối bằng **CrewAI Flows**. Flow là một **hệ thống điều phối workflow** được cung cấp bởi CrewAI, là tính năng cốt lõi cho phép bạn thực thi các tác vụ nhiều bước theo thứ tự hoặc theo điều kiện.

### Mục tiêu học tập

- Hiểu cấu trúc cơ bản của CrewAI Flow và các decorator (`@start`, `@listen`, `@router`)
- Quản lý trạng thái (State) của Flow bằng Pydantic model
- Triển khai định tuyến có điều kiện và vòng lặp cải thiện (Refinement Loop)
- Gọi LLM và Agent trực tiếp trong Flow
- Tích hợp Crew vào Flow để hoàn thành pipeline AI phức tạp

### Cấu trúc dự án

```
content-pipeline-agent/
├── main.py              # Logic chính của Flow
├── tools.py             # Công cụ tìm kiếm web (Firecrawl)
├── seo_crew.py          # Crew phân tích SEO
├── virality_crew.py     # Crew phân tích độ viral
├── pyproject.toml       # Phụ thuộc dự án
├── crewai_flow.html     # File trực quan hóa Flow
└── .gitignore
```

### Phụ thuộc chính

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

- **crewai[tools]**: Framework CrewAI và gói mở rộng công cụ
- **firecrawl-py**: Client API tìm kiếm và scraping web
- **python-dotenv**: Quản lý biến môi trường (.env)

---

## 2. Giải thích chi tiết từng phần

---

### 2.1 Flow đầu tiên của bạn

**Commit:** `c47fd95`

#### Chủ đề và mục tiêu

Hiểu cấu trúc cơ bản nhất của CrewAI Flow. Tìm hiểu Flow là gì, có những decorator nào, và trạng thái (State) được quản lý như thế nào.

#### Khái niệm cốt lõi

**Flow là gì?**

Flow là một **engine workflow** được cung cấp bởi CrewAI. Bạn định nghĩa nhiều hàm (step) và khai báo thứ tự thực thi của mỗi hàm bằng decorator. Điều này cho phép bạn cấu thành các pipeline AI phức tạp một cách trực quan.

**Các Decorator chính:**

| Decorator | Vai trò | Mô tả |
|-----------|---------|-------|
| `@start()` | Điểm bắt đầu | Hàm đầu tiên được thực thi khi Flow bắt đầu |
| `@listen(fn)` | Listener | Thực thi khi hàm được chỉ định hoàn thành |
| `@router(fn)` | Router | Phân nhánh sang các đường dẫn khác nhau dựa trên giá trị trả về |
| `and_(a, b)` | Điều kiện AND | Chỉ thực thi khi **cả hai** hàm đều hoàn thành |
| `or_(a, b)` | Điều kiện OR | Thực thi khi **bất kỳ một** trong các hàm hoàn thành |

**Flow State (Quản lý trạng thái):**

Flow sử dụng Pydantic `BaseModel` làm đối tượng trạng thái. Tất cả các step có thể truy cập và sửa đổi trạng thái chia sẻ thông qua `self.state`.

#### Phân tích mã nguồn

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

**Phân tích chi tiết luồng mã:**

1. **`MyFirstFlowState`**: Định nghĩa trạng thái của Flow dưới dạng Pydantic model. Có hai trường: `user_id` và `is_admin`.

2. **`MyFirstFlow(Flow[MyFirstFlowState])`**: Chỉ định class trạng thái dưới dạng generic type. Điều này làm cho `self.state` có kiểu `MyFirstFlowState`.

3. **`@start()` - `first()`**: Điểm vào của Flow. In ra `self.state.user_id` và "Hello".

4. **`@listen(first)` - `second()` và `third()`**: Chạy **đồng thời (song song)** khi `first()` hoàn thành. `second()` thay đổi trạng thái (`user_id = 2`), và `third()` in "!". Điểm quan trọng là nhiều hàm listen cùng một hàm sẽ thực thi song song.

5. **`@listen(and_(second, third))` - `final()`**: Chỉ thực thi sau khi **cả hai** `second()` và `third()` hoàn thành. Điều kiện `and_()` đóng vai trò như một điểm đồng bộ hóa (synchronization point).

6. **`@router(final)` - `route()`**: Phân nhánh dựa trên điều kiện sau `final()`. Hàm nào listen chuỗi được trả về (`"even"` hoặc `"odd"`) sẽ được thực thi.

7. **`@listen("even")` / `@listen("odd")`**: Listen giá trị chuỗi được trả về bởi router. Điểm mấu chốt là bạn có thể listen **chuỗi**, không chỉ tham chiếu hàm.

8. **`flow.plot()`**: Trực quan hóa đường dẫn thực thi của Flow dưới dạng file HTML (`crewai_flow.html`).

9. **`flow.kickoff()`**: Chạy Flow.

#### File công cụ đi kèm: `tools.py`

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

**Phân tích công cụ:**

- Định nghĩa công cụ CrewAI bằng decorator `@tool`.
- Thực hiện tìm kiếm web bằng Firecrawl API.
- Loại bỏ liên kết và ký tự đặc biệt không cần thiết từ kết quả tìm kiếm bằng biểu thức chính quy (làm sạch).
- Trả về kết quả ở định dạng markdown sạch.

#### Điểm thực hành

- Thay đổi `is_admin` thành `True` và xác nhận rằng đường dẫn định tuyến thay đổi.
- Hiểu sự khác biệt giữa việc sử dụng tham chiếu hàm (`first`) và chuỗi (`"even"`) trong decorator `@listen`.
- Mở file HTML được tạo bởi `flow.plot()` trong trình duyệt để kiểm tra trực quan cấu trúc Flow.
- Thử nghiệm sự khác biệt hành vi giữa `and_` và `or_`.

---

### 2.2 Content Pipeline Flow (Flow Pipeline Nội dung)

**Commit:** `1e78354`

#### Chủ đề và mục tiêu

Sử dụng các khái niệm Flow đã học từ ví dụ đầu tiên, thiết kế khung sườn của một **pipeline tạo nội dung** thực tế. Xây dựng cấu trúc pipeline xử lý các loại nội dung khác nhau như tweet, bài blog, và bài đăng LinkedIn.

#### Khái niệm cốt lõi

**Nguyên tắc thiết kế Pipeline thực tế:**

1. **Xác thực đầu vào (Validation)**: Chặn đầu vào không hợp lệ sớm khi Flow bắt đầu
2. **Định tuyến có điều kiện**: Chọn đường xử lý khác nhau dựa trên loại nội dung
3. **Phân nhánh kiểm tra chất lượng**: Kiểm tra SEO cho blog, kiểm tra viral cho mạng xã hội
4. **Hoàn thành thống nhất**: Tất cả đường dẫn cuối cùng hội tụ vào một bước hoàn thành duy nhất

#### Phân tích mã nguồn

**Thiết kế State Model:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""    # Một trong "tweet", "blog", "linkedin"
    topic: str = ""           # Chủ đề nội dung

    # Internal
    max_length: int = 0       # Độ dài tối đa theo loại nội dung
```

Trạng thái được tổ chức thành **Inputs** (đầu vào bên ngoài) và **Internal** (để xử lý nội bộ). Đây là thực hành tốt cho cấu trúc mã sạch.

**Khởi tạo và xác thực Pipeline:**

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

- Xác thực đầu vào ở bước bắt đầu (pattern Fail Fast).
- Thiết lập `max_length` dựa trên loại nội dung để sử dụng trong các bước tiếp theo.

**Nghiên cứu và định tuyến:**

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

Chuỗi được trả về bởi decorator `@router` quyết định đường thực thi. Đây là pattern cốt lõi để triển khai **phân nhánh động** trong Flow.

**Xử lý theo loại nội dung và kiểm tra chất lượng:**

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

**Sơ đồ luồng thực thi:**

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

**Điểm thiết kế chính:**
- Blog phân nhánh sang đường **kiểm tra SEO**, trong khi tweet và bài đăng LinkedIn phân nhánh sang đường **kiểm tra viral**.
- Sử dụng `or_(check_virality, check_seo)` đảm bảo `finalize_content` thực thi bất kể kiểm tra nào hoàn thành.
- Áp dụng tiêu chí chất lượng khác nhau dựa trên loại nội dung là pattern rất phổ biến trong thực tế.

**Thực thi Flow (truyền inputs):**

```python
flow = ContentPipelineFlow()

flow.kickoff(
    inputs={
        "content_type": "tweet",
        "topic": "AI Dog Training",
    },
)
```

Truyền dictionary `inputs` cho `kickoff()` sẽ tự động thiết lập các trường trạng thái tương ứng.

#### Điểm thực hành

- Thay đổi `content_type` thành `"blog"`, `"tweet"`, và `"linkedin"` lần lượt và quan sát đường thực thi thay đổi như thế nào.
- Kiểm tra cấu trúc trực quan của pipeline với `flow.plot()`.
- Suy nghĩ về sự khác biệt giữa `or_` và `and_` trong ngữ cảnh này: hiểu tại sao `or_` được sử dụng trong `finalize_content`.

---

### 2.3 Refinement Loop (Vòng lặp cải thiện)

**Commit:** `482e52c`

#### Chủ đề và mục tiêu

Triển khai pattern **Refinement Loop** tự động tạo lại nội dung khi chất lượng nội dung do AI tạo ra thấp hơn ngưỡng. Đây là pattern cốt lõi để đảm bảo chất lượng trong hệ thống agent AI.

#### Khái niệm cốt lõi

**Refinement Loop là gì?**

Refinement Loop là cấu trúc tuần hoàn "tạo -> đánh giá -> tạo lại". Nó cải thiện nội dung lặp đi lặp lại cho đến khi điểm đánh giá đạt ngưỡng. Pattern này cần thiết trong các trường hợp sau:

- Khi LLM không thể tạo kết quả hoàn hảo ngay lần đầu
- Khi tiêu chuẩn chất lượng cao và cần nhiều lần thử
- Khi cần quy trình đảm bảo chất lượng (QA) tự động

**Triển khai vòng lặp với Router:**

`@router` của Flow không chỉ dùng cho phân nhánh đơn giản mà còn để tạo **vòng lặp quay về bước trước**. Bằng cách thêm điều kiện chuỗi vào `@listen`, bước trước có thể thực thi lại dựa trên giá trị trả về của router.

#### Phân tích mã nguồn

**Thêm trường điểm và nội dung vào State:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: int = 0                # Thêm điểm chất lượng

    # Content
    blog_post: str = ""           # Bài blog đã tạo
    tweet: str = ""               # Tweet đã tạo
    linkedin_post: str = ""       # Bài đăng LinkedIn đã tạo
```

Điểm (`score`) và các trường lưu kết quả cho từng loại nội dung đã được thêm. Các trường trạng thái này đóng vai trò chính trong vòng lặp.

**Điểm tái nhập sử dụng `or_` với chuỗi:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        # Nếu blog đã tồn tại, hiển thị cái hiện có cho AI và yêu cầu cải thiện,
        # nếu chưa thì tạo mới
        print("Making blog post...")

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        print("Making tweet...")

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        print("Making linkedin post...")
```

**Thay đổi chính**: `@listen("make_blog")` đã được đổi thành `@listen(or_("make_blog", "remake_blog"))`. Điều này có nghĩa:
- Khi tạo lần đầu: Router trả về `"make_blog"` để thực thi
- Khi tạo lại: `score_router` trả về `"remake_blog"` để thực thi lại cùng hàm

Đây chính là **điểm tái nhập vòng lặp**.

**Router dựa trên điểm (Điều khiển vòng lặp):**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score >= 8:
            return "check_passed"       # Đạt -> đến finalize_content
        else:
            if content_type == "blog":
                return "remake_blog"     # Quay lại -> đến handle_make_blog
            elif content_type == "linkedin":
                return "remake_linkedin_post"  # Quay lại
            else:
                return "remake_tweet"          # Quay lại

    @listen("check_passed")
    def finalize_content(self):
        print("Finalizing content")
```

**Luồng thực thi cải tiến:**

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
finalize_content    (Vòng lặp: quay về bước tạo lại nội dung tương ứng)
```

**Lưu ý đổi tên Router:**

```python
    @router(conduct_research)
    def conduct_research_router(self):   # Đổi tên từ "router"
```

Tên phương thức đã được đổi từ `router` thành `conduct_research_router`. Vì tên phương thức quan trọng cho trực quan hóa và gỡ lỗi trong Flow, nên sử dụng tên mô tả rõ vai trò là thực hành tốt.

#### Điểm thực hành

- Thay đổi ngưỡng `score >= 8` và quan sát vòng lặp lặp lại bao nhiêu lần.
- Thêm logic số lần lặp tối đa (max iteration) để ngăn vòng lặp vô hạn.
- Thiết kế logic trong đường `remake_*` tham chiếu nội dung hiện có để cải thiện (được triển khai trong phần tiếp theo).

---

### 2.4 LLMs and Agents (LLM và Agent)

**Commit:** `c341770`

#### Chủ đề và mục tiêu

Kết nối **lệnh gọi LLM thực tế** và **Agent** vào các placeholder trước đó đã được thay thế bằng câu lệnh `print()`. Học cả cách gọi LLM trực tiếp trong CrewAI Flow và cách sử dụng Agent độc lập.

#### Khái niệm cốt lõi

**Hai cách sử dụng AI trong Flow:**

1. **`LLM.call()`**: Gọi LLM trực tiếp. Nhanh và đơn giản, hữu ích khi cần đầu ra có cấu trúc (Structured Output).
2. **`Agent.kickoff()`**: Tạo và chạy Agent. Sử dụng khi cần dùng công cụ (Tool) hoặc suy luận phức tạp hơn.

**Đầu ra có cấu trúc sử dụng Pydantic Model:**

Phản hồi của LLM có thể nhận ở **cấu trúc được định trước** thay vì văn bản tự do. Điều này cho phép xử lý dữ liệu ổn định trong các bước xử lý tiếp theo.

#### Phân tích mã nguồn

**Định nghĩa Pydantic Model cho đầu ra có cấu trúc:**

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

Định nghĩa **schema đầu ra** cho từng loại nội dung:
- **BlogPost**: Gồm tiêu đề, phụ đề và nhiều phần
- **Tweet**: Nội dung và hashtag
- **LinkedInPost**: Hook (câu đầu thu hút sự chú ý), nội dung và CTA (lời kêu gọi hành động)
- **Score**: Điểm và lý do (để đánh giá chất lượng)

**Cập nhật State Model:**

```python
class ContentPipelineState(BaseModel):

    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    research: str = ""              # Lưu kết quả nghiên cứu
    score: Score | None = None      # Đổi thành đối tượng Score

    # Content
    blog_post: BlogPost | None = None   # Đổi thành Pydantic model
    tweet: str = ""
    linkedin_post: str = ""
```

`blog_post` trước đó có kiểu `str` đã được đổi thành `BlogPost | None`. `None` biểu thị trạng thái chưa được tạo.

**Bước nghiên cứu sử dụng Agent:**

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

**Sự khác biệt giữa Agent và gọi LLM trực tiếp:**

| Thuộc tính | `Agent.kickoff()` | `LLM.call()` |
|------------|-------------------|---------------|
| Sử dụng công cụ | Có thể (tìm kiếm web, v.v.) | Không thể |
| Bước suy luận | Suy luận nhiều bước | Gọi đơn |
| Tốc độ | Tương đối chậm | Nhanh |
| Trường hợp phù hợp | Nghiên cứu, tác vụ phức tạp | Tạo nội dung, chuyển đổi |

Bước nghiên cứu sử dụng Agent vì cần **công cụ tìm kiếm web**, còn việc tạo nội dung gọi LLM trực tiếp.

**Tạo blog sử dụng LLM (Đầu ra có cấu trúc):**

```python
from crewai import LLM

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):

        blog_post = self.state.blog_post

        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            # Tạo lần đầu
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
            # Tạo lại (cải thiện dựa trên nội dung hiện có + phản hồi điểm)
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

**Phân tích chính:**

1. **`LLM(model="openai/o4-mini", response_format=BlogPost)`**: Truyền Pydantic model cho `response_format` khiến LLM tạo phản hồi khớp với JSON schema của model đó.

2. **Phân nhánh tạo lần đầu vs tạo lại**: Xác định bởi `blog_post is None`.
   - Lần đầu: Chỉ truyền kết quả nghiên cứu
   - Tạo lại: Truyền cả nội dung hiện có + lý do điểm thấp (`self.state.score.reason`) để hướng dẫn cải thiện

3. **`model_dump_json()`**: Chuyển đổi Pydantic model thành chuỗi JSON để truyền cho LLM.

4. **Sử dụng thẻ XML trong prompt**: Thẻ XML như `<research>`, `<blog post>` được dùng để phân tách các phần của prompt. Đây là kỹ thuật hiệu quả giúp LLM hiểu cấu trúc prompt tốt hơn.

#### Điểm thực hành

- Xóa `response_format` và chạy mã để so sánh sự khác biệt giữa đầu ra có cấu trúc và đầu ra văn bản tự do.
- Chuyển sang mô hình LLM khác (ví dụ: `"openai/gpt-4o"`) và so sánh chất lượng kết quả.
- Sửa đổi prompt để thay đổi giọng điệu hoặc phong cách của bài blog.
- Thử nghiệm sửa đổi `backstory` của Agent để xem kết quả nghiên cứu thay đổi như thế nào.

---

### 2.5 Thêm Crew vào Flow

**Commit:** `8e039ec`

#### Chủ đề và mục tiêu

Vượt ra ngoài các lệnh gọi Agent hoặc LLM riêng lẻ, tích hợp **Crew** (đội agent) vào Flow. Tạo Crew phân tích SEO và Crew phân tích viral để đánh giá chất lượng nội dung. Phần này là điểm nổi bật và quan trọng nhất của chương.

#### Khái niệm cốt lõi

**Giá trị của tích hợp Flow + Crew:**

Flow **điều khiển toàn bộ workflow**, và Crew **thực hiện tác vụ phức tạp tại các bước cụ thể**. Kết hợp hai thành phần này:

- Flow xử lý điều phối toàn bộ pipeline
- Các Crew chuyên biệt được gọi khi cần ở mỗi bước
- Kết quả đầu ra của Crew được nắm bắt dưới dạng trạng thái Flow để sử dụng trong các bước tiếp theo

**Decorator `@CrewBase`:**

Decorator được sử dụng khi định nghĩa Crew dưới dạng class trong CrewAI. Được sử dụng cùng với các decorator `@agent`, `@task` và `@crew` để định nghĩa các thành viên và tác vụ của Crew một cách khai báo.

#### Phân tích mã nguồn

**Crew phân tích SEO (`seo_crew.py`):**

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

**Phân tích cấu trúc Crew:**

1. **`@CrewBase`**: Khai báo class là class cơ sở Crew. Tự động cung cấp thuộc tính `self.agents` và `self.tasks`.

2. **`@agent`**: Định nghĩa Agent. Chuyên gia SEO bao gồm hướng dẫn "chấm điểm rất nghiêm ngặt." Đây là quyết định thiết kế để đảm bảo Refinement Loop hoạt động đúng (không đạt quá dễ).

3. **`@task`**: Định nghĩa Task. `{blog_post}` và `{topic}` là biến được truyền từ `kickoff(inputs={})`. `output_pydantic=Score` chỉ định đầu ra có cấu trúc.

4. **`@crew`**: Trả về đối tượng Crew cuối cùng. `self.agents` và `self.tasks` là danh sách được `@CrewBase` tự động thu thập.

**Crew phân tích Viral (`virality_crew.py`):**

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

Crew Viral có cấu trúc giống với Crew SEO, nhưng với tiêu chí đánh giá khác nhau:
- Crew SEO: Tối ưu từ khóa, hiệu quả tiêu đề, cấu trúc nội dung, v.v.
- Crew Viral: Độ mạnh của hook, cộng hưởng cảm xúc, khả năng chia sẻ, thực hành tốt nhất theo nền tảng, v.v.

**Gọi Crew từ Flow:**

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

**Phân tích pattern gọi Crew:**

```python
SeoCrew()           # 1. Tạo instance class Crew
    .crew()         # 2. Lấy đối tượng Crew
    .kickoff(       # 3. Thực thi Crew
        inputs={    # 4. Truyền giá trị đầu vào ánh xạ đến {biến} của Task
            "topic": self.state.topic,
            "blog_post": self.state.blog_post.model_dump_json(),
        }
    )
```

- `result.pydantic`: Trích xuất Pydantic model từ kết quả thực thi của Crew. Vì `output_pydantic=Score` được chỉ định trong Task, `result.pydantic` là đối tượng `Score`.

**Hoàn thành logic tạo nội dung Tweet và LinkedIn:**

Trong commit này, không chỉ blog mà cả logic tạo tweet và bài đăng LinkedIn cũng đã được triển khai đầy đủ bằng LLM:

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

**`model_validate_json(result)`**: Parse chuỗi JSON được LLM trả về thành Pydantic model. Vì `response_format=Tweet` được chỉ định, LLM trả về JSON khớp với schema `Tweet`, sau đó được chuyển đổi lại thành đối tượng Pydantic.

**Cập nhật State Model cuối cùng:**

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
    tweet: Tweet | None = None           # Đổi từ str thành Tweet
    linkedin_post: LinkedInPost | None = None  # Đổi từ str thành LinkedInPost
```

#### Điểm thực hành

- Xóa "Be very very very demanding" từ `goal` của `SeoCrew` và xem điểm thay đổi như thế nào.
- Tạo Crew mới (ví dụ: Crew kiểm tra ngữ pháp, Crew kiểm tra sự thật) và thêm vào pipeline.
- Mở rộng `virality_crew.py` bằng cách thêm Agent để tạo Crew đa agent.
- Quan sát quá trình thực thi thông qua `verbose=True` trên Crew.

---

### 2.6 Kết luận

**Commit:** `be0cf85`

#### Chủ đề và mục tiêu

Hoàn thành giai đoạn cuối cùng của pipeline. Điều chỉnh ngưỡng chất lượng, tổ chức đầu ra nội dung cuối cùng, và hoàn thiện để toàn bộ pipeline hoạt động end-to-end.

#### Khái niệm cốt lõi

**Điều chỉnh cuối cùng:**

1. **Hạ ngưỡng điểm**: Thay đổi từ `score >= 8` thành `score >= 7` ở mức thực tế
2. **Ghi log tạo lại**: Thêm thông báo log để gỡ lỗi
3. **Định dạng đầu ra cuối**: Xuất kết quả dựa trên loại nội dung
4. **Triển khai giá trị trả về**: Trả về kết quả cuối cùng của Flow

#### Phân tích mã nguồn

**Điều chỉnh ngưỡng điểm:**

```python
    @router(or_(check_seo, check_virality))
    def score_router(self):

        content_type = self.state.content_type
        score = self.state.score

        if score.score >= 7:        # Hạ từ 8 xuống 7
            return "check_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"
```

Hạ ngưỡng điểm xuống 7 là quyết định thực tế. Ngưỡng quá cao có thể gây ra vòng lặp gần như vô hạn, trong khi quá thấp làm giảm chất lượng. Trong môi trường production, nên quản lý ngưỡng này thông qua file cấu hình hoặc biến môi trường.

**Đầu ra nội dung cuối cùng:**

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

**Phân tích pattern giá trị trả về:**

`finalize_content()` trả về Pydantic model tương ứng dựa trên loại nội dung. Giá trị được trả về bởi step cuối cùng của Flow trở thành giá trị trả về của `flow.kickoff()`. Điều này cho phép mã bên ngoài gọi Flow nhận và sử dụng kết quả.

**Thêm log tạo lại:**

```python
    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        blog_post = self.state.blog_post
        llm = LLM(model="openai/o4-mini", response_format=BlogPost)

        if blog_post is None:
            result = llm.call(...)
        else:
            print("Remaking blog.")   # Thêm log gỡ lỗi
            result = llm.call(...)
```

**Sửa kiểm tra Viral:**

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
                        self.state.tweet.model_dump_json()       # Đã sửa
                        if self.state.contenty_type == "tweet"
                        else self.state.linkedin_post.model_dump_json()  # Đã sửa
                    ),
                }
            )
        )
        self.state.score = result.pydantic
```

`.model_dump_json()` đã được thêm để serialize Pydantic model thành chuỗi JSON đúng cách.

#### Điểm thực hành

- Chạy toàn bộ pipeline cho từng `content_type` và so sánh kết quả.
- Viết mã để lưu giá trị trả về của `flow.kickoff()` vào biến và sử dụng nó.
- Thêm logic giới hạn số lần lặp tối đa vào `score_router`.

---

## 3. Tóm tắt trọng tâm chương

### Cấu trúc cơ bản của Flow

| Khái niệm | Mô tả | Decorator |
|-----------|-------|-----------|
| Điểm bắt đầu | Hàm đầu tiên nơi Flow bắt đầu | `@start()` |
| Listener | Thực thi sau khi một hàm cụ thể hoàn thành | `@listen(fn)` |
| Router | Phân nhánh dựa trên điều kiện | `@router(fn)` |
| Điều kiện AND | Yêu cầu tất cả hàm trước đó hoàn thành | `and_(a, b)` |
| Điều kiện OR | Thực thi khi bất kỳ hàm nào hoàn thành | `or_(a, b)` |
| Nghe chuỗi | Phản hồi giá trị trả về của router | `@listen("chuỗi")` |

### Pattern quản lý trạng thái

```python
class MyState(BaseModel):
    # Giá trị đầu vào
    input_field: str = ""

    # Cho xử lý nội bộ
    intermediate_data: str = ""

    # Kết quả (sử dụng Pydantic model)
    output: MyOutputModel | None = None
```

### So sánh phương thức gọi AI

| Phương thức | Trường hợp sử dụng | Sử dụng công cụ | Đầu ra có cấu trúc |
|------------|---------------------|-----------------|---------------------|
| `LLM.call()` | Tạo/chuyển đổi đơn giản | Không thể | `response_format` |
| `Agent.kickoff()` | Nghiên cứu, suy luận phức tạp | Có thể | Hạn chế |
| `Crew.kickoff()` | Tác vụ phức tạp theo đội | Có thể | `output_pydantic` |

### Pattern Refinement Loop

```
Tạo --> Đánh giá --> Kiểm tra điểm --[Đạt]--> Hoàn thành
                    |
              [Không đạt] --> Tạo lại (quay lại)
```

Chìa khóa: Xử lý cả tạo lần đầu và tạo lại trong cùng hàm với `@listen(or_("make_x", "remake_x"))`

### Pattern tích hợp Crew vào Flow

```python
result = MyCrewClass().crew().kickoff(inputs={...})
self.state.score = result.pydantic
```

---

## 4. Bài tập thực hành

### Bài tập 1: Xây dựng Flow cơ bản (Độ khó: Sơ cấp)

Viết Flow đáp ứng các yêu cầu sau:
- Nhận tên người dùng và ngôn ngữ (Tiếng Hàn/Tiếng Anh) làm đầu vào
- Tạo lời chào khác nhau dựa trên ngôn ngữ (sử dụng router)
- Xuất lời chào cuối cùng

**Gợi ý:** Sử dụng các decorator `@start()`, `@router()` và `@listen("chuỗi")`.

### Bài tập 2: Flow tích hợp LLM (Độ khó: Trung cấp)

Tạo Flow sinh công thức nấu ăn:
- Đầu vào: Danh sách nguyên liệu và phong cách nấu (Hàn/Âu/Trung)
- Sử dụng Agent trong bước `conduct_research` để tìm kiếm các món ăn có thể làm với nguyên liệu
- Gọi LLM trực tiếp trong bước `generate_recipe` để tạo công thức có cấu trúc
- Pydantic model: `Recipe(title: str, ingredients: List[str], steps: List[str], cooking_time: int)`

### Bài tập 3: Pipeline với Refinement Loop (Độ khó: Cao cấp)

Xây dựng pipeline tạo nội dung email marketing:
- Đầu vào: Tên sản phẩm, đối tượng mục tiêu, loại email (khuyến mãi/bản tin/email chào mừng)
- Agent nghiên cứu điều tra sản phẩm và đối tượng mục tiêu
- LLM tạo nội dung email
- Crew đánh giá chất lượng đánh giá hiệu quả email (sức hấp dẫn tiêu đề, hiệu quả CTA, sự phù hợp giọng điệu)
- Triển khai Refinement Loop tạo lại nếu điểm dưới 7
- Giới hạn tối đa 3 lần lặp, sau đó trả về kết quả tốt nhất

**Gợi ý:**
- Thêm trường `iteration_count: int = 0` vào state
- Trong `score_router`, trả về `"check_passed"` bất kể điểm khi `iteration_count >= 3`

### Bài tập 4: Pipeline đa Crew (Độ khó: Cao cấp)

Mở rộng Content Pipeline từ chương này:
- Thêm Crew mới: `GrammarCrew` (kiểm tra ngữ pháp và khả năng đọc)
- Với bài blog, chạy kiểm tra SEO và kiểm tra ngữ pháp **song song** (sử dụng `and_`)
- Tính điểm cuối cùng bằng trung bình có trọng số của cả hai kiểm tra
- Chỉ tiến đến `finalize_content` khi tất cả kiểm tra đạt

---

## Phụ lục: Tóm tắt cấu trúc mã cuối cùng hoàn chỉnh

### Cấu trúc cuối cùng main.py

```
ContentPipelineState (Pydantic BaseModel)
├── content_type, topic         # Đầu vào
├── max_length, research, score # Nội bộ
└── blog_post, tweet, linkedin_post  # Đầu ra (Pydantic model)

ContentPipelineFlow (Flow)
├── init_content_pipeline()          @start       - Xác thực và khởi tạo đầu vào
├── conduct_research()               @listen      - Nghiên cứu web với Agent
├── conduct_research_router()        @router      - Phân nhánh theo loại nội dung
├── handle_make_blog()               @listen(or_) - Tạo/tạo lại blog với LLM
├── handle_make_tweet()              @listen(or_) - Tạo/tạo lại tweet với LLM
├── handle_make_linkedin_post()      @listen(or_) - Tạo/tạo lại LinkedIn với LLM
├── check_seo()                      @listen      - Đánh giá SEO với SeoCrew
├── check_virality()                 @listen(or_) - Đánh giá viral với ViralityCrew
├── score_router()                   @router(or_) - Đạt/quay lại dựa trên điểm
└── finalize_content()               @listen      - Xuất và trả về kết quả cuối
```

### Mối quan hệ file chính

```
main.py ──imports──> tools.py (web_search_tool)
   |
   ├──imports──> seo_crew.py (SeoCrew) ──> Trả về model Score
   └──imports──> virality_crew.py (ViralityCrew) ──> Trả về model Score
```

Qua chương này, chúng ta đã học tất cả tính năng cốt lõi của CrewAI Flow và hoàn thành một pipeline tạo nội dung có thể sử dụng trong production. Hãy nhớ rằng Flow không chỉ là công cụ workflow đơn giản, mà là một **framework điều phối** kết nối hữu cơ LLM, Agent và Crew.
