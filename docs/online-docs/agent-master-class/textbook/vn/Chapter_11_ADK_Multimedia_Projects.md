# Chapter 11: Dự án Đa phương tiện ADK - Trình tạo YouTube Shorts tự động

---

## 1. Tổng quan chương

Trong chương này, chúng ta xây dựng **hệ thống đa tác tử tự động tạo video YouTube Shorts** sử dụng Google ADK (Agent Development Kit). Khi người dùng nhập một chủ đề, các tác tử AI sẽ tự động xử lý toàn bộ quy trình từ lập kế hoạch nội dung, tạo hình ảnh, tạo lời thoại giọng nói, đến lắp ráp video.

### Tên dự án
**youtube-shorts-maker** - Trình tạo video YouTube Shorts định dạng dọc (9:16) tự động

### Mục tiêu học tập
- Hiểu và sử dụng các loại tác tử ADK đa dạng (Agent, ParallelAgent, SequentialAgent)
- Thiết kế các mẫu điều phối đa tác tử
- Cấu trúc hóa đầu ra tác tử sử dụng Pydantic schema
- Tích hợp OpenAI API (GPT-Image-1, TTS) làm công cụ (Tool)
- Quản lý tệp đa phương tiện sử dụng hệ thống Artifact của ADK
- Lắp ráp video sử dụng FFmpeg
- Kiểm soát hành vi tác tử thông qua Callback

### Kiến trúc tổng thể

```
ShortsProducerAgent (Trình điều phối gốc)
    |
    +-- ContentPlannerAgent (Lập kế hoạch nội dung)
    |
    +-- AssetGeneratorAgent (ParallelAgent - Tạo tài nguyên song song)
    |       |
    |       +-- ImageGeneratorAgent (SequentialAgent - Tạo hình ảnh tuần tự)
    |       |       |
    |       |       +-- PromptBuilderAgent (Tối ưu hóa prompt)
    |       |       +-- ImageBuilderAgent (Tạo hình ảnh)
    |       |
    |       +-- VoiceGeneratorAgent (Tạo lời thoại giọng nói)
    |
    +-- VideoAssemblerAgent (Lắp ráp video cuối cùng)
```

### Luồng công việc
1. **Giai đoạn 1**: Thu thập đầu vào của người dùng và xác nhận yêu cầu
2. **Giai đoạn 2**: ContentPlannerAgent tạo kịch bản có cấu trúc
3. **Giai đoạn 3**: AssetGeneratorAgent tạo hình ảnh và giọng nói **song song**
4. **Giai đoạn 4**: VideoAssemblerAgent lắp ráp video MP4 cuối cùng bằng FFmpeg
5. **Giai đoạn 5**: Giao sản phẩm cuối cùng

---

## 2. Chi tiết từng phần

---

### 11.0 Introduction - Thiết lập ban đầu dự án

#### Chủ đề và mục tiêu
Thiết lập cấu trúc dự án cơ bản và định nghĩa các gói phụ thuộc cần thiết. Hiểu cấu trúc thư mục tiêu chuẩn cho các dự án ADK.

#### Khái niệm chính

**Cấu trúc dự án ADK**: Google ADK yêu cầu một cấu trúc thư mục cụ thể. Bạn phải đặt `agent.py` và `__init__.py` bên trong thư mục có tên trùng với tên gói.

**Các phụ thuộc dự án**:

```toml
[project]
name = "youtube-shorts-maker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "google-adk>=1.12.0",
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "openai>=1.101.0",
]
```

Vai trò của từng phụ thuộc:
- **google-adk**: Google Agent Development Kit - cốt lõi của framework tác tử
- **google-genai**: Thư viện Generative AI của Google (dùng cho định nghĩa kiểu và quản lý Artifact)
- **litellm**: Thư viện tích hợp các nhà cung cấp LLM khác nhau (cho phép sử dụng mô hình OpenAI trong ADK)
- **openai**: Client API OpenAI (dùng trực tiếp cho tạo hình ảnh và TTS)

**Tệp `__init__.py`**:

```python
from . import agent
```

Dòng duy nhất này khiến ADK tự động import module `agent.py` khi tải gói. ADK tìm `root_agent` trong module này và thực thi nó.

**Cấu trúc thư mục ban đầu**:

```
youtube-shorts-maker/
    .python-version          # Chỉ định Python 3.13
    pyproject.toml           # Cấu hình dự án và phụ thuộc
    README.md
    uv.lock                  # Tệp khóa trình quản lý gói uv
    youtube_shorts_maker/
        __init__.py          # Khởi tạo gói
        agent.py             # Tác tử gốc (tệp trống)
        sub_agents/
            content_planner/
                agent.py     # Tác tử lập kế hoạch nội dung (tệp trống)
                prompt.py    # Định nghĩa prompt (tệp trống)
```

#### Điểm thực hành
- Tìm hiểu cách khởi tạo dự án sử dụng `uv`.
- Làm quen với cấu trúc thư mục mà ADK yêu cầu (import module `agent` từ `__init__.py`, biến `root_agent`).
- Ghi nhớ mẫu module hóa tác tử sử dụng thư mục `sub_agents/`.

---

### 11.1 Content Planner Agent - Tác tử lập kế hoạch nội dung

#### Chủ đề và mục tiêu
Triển khai tác tử xuất toàn bộ kế hoạch nội dung cho YouTube Shorts dưới dạng JSON có cấu trúc. Học cách sử dụng `output_schema` và `output_key` với mô hình Pydantic.

#### Khái niệm chính

**1) Tác tử gốc (ShortsProducerAgent) - Mẫu điều phối**

Trình điều phối là tác tử cấp cao phối hợp nhiều tác tử con để thực hiện các tác vụ phức tạp. Nó sử dụng `AgentTool` để gọi các tác tử con như thể chúng là công cụ.

```python
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.content_planner.agent import content_planner_agent
from .prompt import SHORTS_PRODUCER_DESCRIPTION, SHORTS_PRODUCER_PROMPT

MODEL = LiteLlm(model="openai/gpt-4o")

shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),
    ],
)

root_agent = shorts_producer_agent
```

Điểm chính:
- **`LiteLlm(model="openai/gpt-4o")`**: ADK mặc định sử dụng mô hình Google, nhưng thông qua wrapper `LiteLlm`, bạn cũng có thể sử dụng các mô hình khác như GPT-4o của OpenAI.
- **`AgentTool(agent=...)`**: Bọc tác tử con như một Tool để tác tử gốc có thể gọi khi cần. Đây là một trong những mẫu cốt lõi của ADK.
- **`root_agent`**: Biến điểm vào mà ADK tìm kiếm khi chạy.

**2) Thiết kế prompt điều phối**

```python
SHORTS_PRODUCER_PROMPT = """
You are the ShortsProducerAgent, the primary orchestrator for creating
vertical YouTube Shorts videos (9:16 portrait format). Your role is to
guide users through the entire video creation process and coordinate
specialized sub-agents.

## Your Workflow:

### Phase 1: User Input & Planning
1. **Greet the user** and ask for details about their desired YouTube Short:
   - What topic/subject do they want to cover?
   - What style or tone should the video have?
   - Any specific requirements or preferences?
   - Target audience considerations?

2. **Clarify and confirm** the requirements before proceeding.

### Phase 2: Content Planning
3. **Use ContentPlannerAgent** to create the structured script...

### Phase 3: Asset Generation (Parallel)
4. **Use AssetGeneratorAgent** to create multimedia assets...

### Phase 4: Video Assembly
5. **Use VideoAssemblerAgent** to create the final video...

### Phase 5: Delivery
6. **Present the final result** to the user...
"""
```

Nguyên tắc thiết kế prompt điều phối:
- **Chỉ rõ luồng công việc từng bước**: Hướng dẫn rõ ràng tác tử về thứ tự thực hiện tác vụ
- **Chỉ định tác tử con nào sử dụng ở mỗi bước**: ContentPlanner -> AssetGenerator -> VideoAssembler
- **Bao gồm hướng dẫn** xử lý lỗi và giao tiếp với người dùng

**3) ContentPlannerAgent - Đầu ra có cấu trúc (output_schema)**

```python
from pydantic import BaseModel, Field
from typing import List

class SceneOutput(BaseModel):
    id: int = Field(description="Scene ID number")
    narration: str = Field(description="Narration text for the scene")
    visual_description: str = Field(
        description="Detailed description for image generation"
    )
    embedded_text: str = Field(
        description="Text overlay for the image (can be any case/style)"
    )
    embedded_text_location: str = Field(
        description="Where to position the text on the image"
    )
    duration: int = Field(description="Duration in seconds for this scene")


class ContentPlanOutput(BaseModel):
    topic: str = Field(description="The topic of the YouTube Short")
    total_duration: int = Field(
        description="Total video duration in seconds (max 20)"
    )
    scenes: List[SceneOutput] = Field(
        description="List of scenes (agent decides how many)"
    )


content_planner_agent = Agent(
    name="ContentPlannerAgent",
    description=CONTENT_PLANNER_DESCRIPTION,
    instruction=CONTENT_PLANNER_PROMPT,
    model=MODEL,
    output_schema=ContentPlanOutput,
    output_key="content_planner_output",
)
```

Điểm chính:
- **`output_schema`**: Khi chỉ định mô hình Pydantic, đầu ra của tác tử buộc phải tuân theo cấu trúc schema đó. Nó buộc LLM phản hồi theo cấu trúc JSON đã định thay vì văn bản tự do.
- **`output_key`**: Tên khóa để lưu đầu ra của tác tử vào trạng thái phiên. Khi đặt thành `"content_planner_output"`, các tác tử khác có thể truy cập sau qua `tool_context.state.get("content_planner_output")`.
- **`Field(description=...)`**: Mô tả của mỗi trường đóng vai trò hướng dẫn cho LLM tạo giá trị đúng.

**4) Chiến lược chính trong prompt Content Planner**

Các điểm đáng chú ý của prompt:
- **Giới hạn tối đa 20 giây**: Ràng buộc phù hợp với đặc tính YouTube Shorts
- **Số lượng cảnh linh hoạt**: Tác tử quyết định cấu hình tối ưu với 3-6 cảnh
- **Chiến lược thời gian**: Giới thiệu nhanh (2-3 giây), nội dung chính (3-5 giây), kết thúc mạnh (2-4 giây)
- **Yêu cầu xác thực**: Hướng dẫn kiểm tra tổng thời gian không vượt quá 20 giây trước khi xuất
- **Cung cấp ví dụ cụ thể**: Ví dụ "Perfect Scrambled Eggs" cho thấy rõ định dạng đầu ra mong đợi

#### Phân tích mã - Luồng dữ liệu giữa các tác tử

```
Đầu vào người dùng ("Tạo Shorts về nấu ăn")
    |
    v
ShortsProducerAgent (Trình điều phối)
    |  Gọi qua AgentTool
    v
ContentPlannerAgent
    |  output_schema: ContentPlanOutput
    |  output_key: "content_planner_output"
    v
Lưu vào trạng thái phiên: state["content_planner_output"] = {
    "topic": "...",
    "total_duration": 18,
    "scenes": [
        {"id": 1, "narration": "...", "visual_description": "...", ...},
        {"id": 2, ...},
        ...
    ]
}
```

#### Điểm thực hành
- Thử nghiệm cách `Field(description=...)` trong mô hình Pydantic ảnh hưởng đến đầu ra LLM.
- Thử thay đổi `output_key` và truy cập từ các tác tử tiếp theo.
- Xem chất lượng đầu ra thay đổi như thế nào khi bạn sửa đổi ví dụ trong prompt.

---

### 11.2 Prompt Builder Agent - Tác tử tối ưu hóa prompt hình ảnh

#### Chủ đề và mục tiêu
Triển khai tác tử chuyển đổi mô tả trực quan từ kế hoạch nội dung thành prompt được tối ưu hóa cho GPT-Image-1. Hiểu sự khác biệt giữa ParallelAgent và SequentialAgent và thiết kế phân cấp tác tử.

#### Khái niệm chính

**1) Sử dụng ba loại tác tử**

Phần này giới thiệu cả ba loại tác tử ADK chính:

```python
# 1. ParallelAgent - Thực thi các tác tử con đồng thời
from google.adk.agents import ParallelAgent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,   # Pipeline tạo hình ảnh
        # voice_generator_agent, # (sẽ thêm ở 11.4)
    ],
)
```

**ParallelAgent** thực thi các tác tử trong danh sách `sub_agents` **đồng thời (song song)**. Tạo hình ảnh và tạo giọng nói không phụ thuộc lẫn nhau, nên có thể xử lý song song.

```python
# 2. SequentialAgent - Thực thi các tác tử con theo thứ tự
from google.adk.agents import SequentialAgent

image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,   # Trước: Tối ưu hóa prompt
        # image_builder_agent,  # Sau: Tạo hình ảnh (sẽ thêm ở 11.3)
    ],
)
```

**SequentialAgent** thực thi `sub_agents` **theo thứ tự**. Vì phải tối ưu hóa prompt trước rồi mới tạo hình ảnh, xử lý tuần tự là bắt buộc.

```python
# 3. Agent - Tác tử LLM tổng quát (có thể dùng công cụ và suy luận)
from google.adk.agents import Agent

prompt_builder_agent = Agent(
    name="PromptBuilderAgent",
    description=PROMPT_BUILDER_DESCRIPTION,
    instruction=PROMPT_BUILDER_PROMPT,
    model=MODEL,
    output_schema=PromptBuilderOutput,
    output_key="prompt_builder_output",
)
```

**Agent** là tác tử cơ bản nhất sử dụng LLM để suy luận, gọi công cụ và tạo đầu ra có cấu trúc.

**2) Mẫu thiết kế phân cấp tác tử**

```
AssetGeneratorAgent (ParallelAgent)
    |
    +-- ImageGeneratorAgent (SequentialAgent)
    |       |
    |       +-- PromptBuilderAgent (Agent) -- Tối ưu hóa prompt
    |       +-- ImageBuilderAgent (Agent)  -- Tạo hình ảnh
    |
    +-- VoiceGeneratorAgent (Agent)         -- Tạo giọng nói
```

Nguyên lý cốt lõi của thiết kế này:
- **Tạo hình ảnh** phải **tuần tự**: tối ưu prompt -> tạo hình ảnh (SequentialAgent)
- **Tạo hình ảnh** và **tạo giọng nói** có thể thực hiện **song song** (ParallelAgent)
- Mỗi tác vụ riêng lẻ cần suy luận LLM (Agent)

**3) Schema đầu ra của Prompt Builder**

```python
class OptimizedPrompt(BaseModel):
    scene_id: int = Field(
        description="Scene ID from the original content plan"
    )
    enhanced_prompt: str = Field(
        description="Detailed prompt with technical specs and text overlay "
                    "instructions for vertical YouTube Shorts"
    )

class PromptBuilderOutput(BaseModel):
    optimized_prompts: List[OptimizedPrompt] = Field(
        description="Array of optimized image generation prompts "
                    "for vertical YouTube Shorts"
    )
```

**4) Truyền dữ liệu giữa các tác tử qua State**

Prompt tham chiếu đầu ra của tác tử trước bằng `{content_planner_output}`:

```python
PROMPT_BUILDER_PROMPT = """
...
## Your Task:
Take the structured content plan: {content_planner_output} and create
optimized vertical image generation prompts for each scene...
"""
```

Khi sử dụng cú pháp `{tên_biến}` trong `instruction` của ADK, giá trị tương ứng từ trạng thái phiên sẽ được tự động chèn vào. Đây là cách nó kết nối với `output_key`:
1. ContentPlannerAgent lưu kết quả với `output_key="content_planner_output"`
2. `instruction` của PromptBuilderAgent tham chiếu dữ liệu đó qua `{content_planner_output}`

**5) Chiến lược tối ưu hóa prompt**

Các tác vụ nâng cao mà prompt builder thực hiện:
- **Thêm thông số kỹ thuật**: Tỷ lệ dọc 9:16, độ phân giải 1080x1920
- **Chi tiết trực quan**: Ánh sáng, góc máy, bố cục, v.v.
- **Hướng dẫn chữ phủ**: Vị trí, khoảng đệm, dễ đọc
- **Nhất quán phong cách**: Duy trì cùng phong cách trực quan qua tất cả các cảnh

Ví dụ chuyển đổi:
```
Gốc: "Stovetop dial on low"
Tối ưu: "Close-up shot of modern stovetop control dial set to low heat
setting, 9:16 portrait aspect ratio, 1080x1920 resolution, vertical
composition, warm kitchen lighting, shallow depth of field, photorealistic,
sharp focus, with bold white text 'Secret #1: Low Heat' positioned at
top center of image with generous padding from borders..."
```

#### Điểm thực hành
- Hoán đổi `ParallelAgent` và `SequentialAgent` để quan sát thứ tự thực thi thay đổi ra sao.
- Kiểm tra log để xem giá trị thực tế thay thế biến mẫu `{content_planner_output}`.
- Sửa đổi hướng dẫn tối ưu hóa prompt để thay đổi phong cách hình ảnh được tạo.

---

### 11.3 Image Builder Agent - Tác tử tạo hình ảnh

#### Chủ đề và mục tiêu
Triển khai tác tử tạo hình ảnh thực tế sử dụng prompt đã tối ưu với API OpenAI GPT-Image-1 và lưu chúng bằng hệ thống Artifact của ADK.

#### Khái niệm chính

**1) Định nghĩa tác tử - Agent có công cụ**

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .prompt import IMAGE_BUILDER_DESCRIPTION, IMAGE_BUILDER_PROMPT
from .tools import generate_images

MODEL = LiteLlm(model="openai/gpt-4o")

image_builder_agent = Agent(
    name="ImageBuilder",
    description=IMAGE_BUILDER_DESCRIPTION,
    instruction=IMAGE_BUILDER_PROMPT,
    model=MODEL,
    output_key="image_builder_output",
    tools=[
        generate_images,
    ],
)
```

Tác tử này sử dụng hàm Python `generate_images` làm công cụ. Trong ADK, khi bạn truyền hàm Python thông thường vào danh sách `tools`, nó tự động được đăng ký như một công cụ.

**2) Thêm ImageBuilder vào SequentialAgent**

```python
image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,    # Bước 1: Tối ưu hóa prompt
        image_builder_agent,     # Bước 2: Tạo hình ảnh
    ],
)
```

Pipeline tạo hình ảnh giờ đã hoàn thành. PromptBuilder chạy trước và lưu prompt đã tối ưu vào state, sau đó ImageBuilder đọc prompt đó và tạo hình ảnh.

**3) Công cụ generate_images - Cốt lõi của hệ thống Artifact ADK**

```python
import base64
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext

client = OpenAI()


async def generate_images(tool_context: ToolContext):

    # 1. Lấy đầu ra của tác tử trước từ trạng thái phiên
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")

    # 2. Kiểm tra danh sách artifact đã tồn tại (ngăn tạo trùng)
    existing_artifacts = await tool_context.list_artifacts()

    generated_images = []

    for prompt in optimized_prompts:
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        filename = f"scene_{scene_id}_image.jpeg"

        # 3. Bỏ qua nếu đã tồn tại (caching)
        if filename in existing_artifacts:
            generated_images.append({
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100],
                "filename": filename,
            })
            continue

        # 4. Tạo hình ảnh với API OpenAI GPT-Image-1
        image = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",    # Định dạng dọc (tỷ lệ 2:3)
        )

        # 5. Giải mã Base64
        image_bytes = base64.b64decode(image.data[0].b64_json)

        # 6. Lưu dưới dạng ADK Artifact
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=image_bytes,
            )
        )

        await tool_context.save_artifact(
            filename=filename,
            artifact=artifact,
        )

        generated_images.append({
            "scene_id": scene_id,
            "prompt": enhanced_prompt[:100],
            "filename": filename,
        })

        return {
            "total_images": len(generated_images),
            "generated_images": generated_images,
            "status": "complete",
        }
```

**Giải thích chi tiết các khái niệm chính:**

**(a) `ToolContext` - Ngữ cảnh cốt lõi cho công cụ**

`tool_context: ToolContext` là tham số đặc biệt mà ADK tự động tiêm vào hàm công cụ. Thông qua nó, bạn có thể:
- `tool_context.state`: Truy cập trạng thái chia sẻ của phiên (chia sẻ dữ liệu giữa các tác tử)
- `tool_context.list_artifacts()`: Truy vấn danh sách artifact đã lưu
- `tool_context.save_artifact()`: Lưu artifact mới
- `tool_context.load_artifact()`: Tải artifact

ADK tự động tiêm ngữ cảnh khi tìm thấy `tool_context: ToolContext` trong chữ ký hàm. Bạn không cần truyền thủ công.

**(b) Hệ thống Artifact**

Artifact là cơ chế quản lý dữ liệu nhị phân (hình ảnh, âm thanh, video, v.v.) trong ADK. Nó sử dụng `google.genai.types.Part` và `types.Blob` để lưu dữ liệu cùng với kiểu MIME.

```python
artifact = types.Part(
    inline_data=types.Blob(
        mime_type="image/jpeg",
        data=image_bytes,      # raw bytes
    )
)
await tool_context.save_artifact(filename=filename, artifact=artifact)
```

Thay vì hệ thống tệp, điều này lưu tệp trong hệ thống quản lý phiên của ADK, cho phép dữ liệu tồn tại qua các phiên và có thể truy cập bởi các tác tử khác.

**(c) Mẫu ngăn tạo trùng**

```python
existing_artifacts = await tool_context.list_artifacts()
if filename in existing_artifacts:
    # Bỏ qua nếu đã tồn tại
    continue
```

Nếu hình ảnh đã được tạo, nó sẽ không bị tạo lại. Đây là mẫu quan trọng tiết kiệm chi phí gọi API và thời gian.

**(d) Tham số API tạo hình ảnh OpenAI**

```python
image = client.images.generate(
    model="gpt-image-1",       # Mô hình tạo hình ảnh của OpenAI
    prompt=enhanced_prompt,     # Prompt đã tối ưu
    n=1,                        # Tạo 1 hình ảnh
    quality="low",              # Chất lượng (low/medium/high)
    moderation="low",           # Mức lọc nội dung
    output_format="jpeg",       # Định dạng đầu ra
    background="opaque",        # Nền (opaque: không trong suốt)
    size="1024x1536",           # Tỷ lệ dọc (YouTube Shorts)
)
```

#### Điểm thực hành
- Thay đổi tham số `quality` thành `"high"` và so sánh sự khác biệt chất lượng hình ảnh.
- Xem kết quả khi thay đổi `size` thành `"1024x1024"`.
- Kiểm tra cơ chế caching qua `tool_context.list_artifacts()` hoạt động ra sao khi chạy lại.

---

### 11.4 Audio Narration Agent - Tác tử tường thuật giọng nói

#### Chủ đề và mục tiêu
Triển khai tác tử tạo âm thanh tường thuật cho mỗi cảnh sử dụng API OpenAI TTS (Text-to-Speech). Hoàn thiện cấu trúc tạo hình ảnh và giọng nói đồng thời bằng cách thêm trình tạo giọng nói vào ParallelAgent.

#### Khái niệm chính

**1) Thêm VoiceGenerator vào ParallelAgent**

```python
from google.adk.agents import ParallelAgent
from .prompt import ASSET_GENERATOR_DESCRIPTION
from .image_generator.agent import image_generator_agent
from .voice_generator.agent import voice_generator_agent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,     # Pipeline tạo hình ảnh
        voice_generator_agent,     # Tác tử tạo giọng nói (đã thêm!)
    ],
)
```

Giờ khi `AssetGeneratorAgent` được gọi, tạo hình ảnh (`ImageGeneratorAgent`) và tạo giọng nói (`VoiceGeneratorAgent`) thực thi **đồng thời**. Vì hai tác vụ độc lập, xử lý song song có thể giảm đáng kể tổng thời gian thực thi.

**2) Định nghĩa VoiceGeneratorAgent**

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .prompt import VOICE_GENERATOR_PROMPT, VOICE_GENERATOR_DESCRIPTION
from .tools import generate_narrations

MODEL = LiteLlm(model="openai/gpt-4o")

voice_generator_agent = Agent(
    name="VoiceGeneratorAgent",
    description=VOICE_GENERATOR_DESCRIPTION,
    instruction=VOICE_GENERATOR_PROMPT,
    model=MODEL,
    tools=[
        generate_narrations,
    ],
)
```

**3) Chiến lược chọn giọng nói - Thiết kế prompt**

Prompt của tác tử này được thiết kế để LLM **tự chọn giọng nói phù hợp** dựa trên tâm trạng nội dung:

```python
VOICE_GENERATOR_PROMPT = """
...
## Content Plan:
{content_planner_output}

## Voice Selection Guidelines:
- **Cooking/Food content**: Use "fable" for warm, engaging instruction
- **Fitness/Energy content**: Use "nova" for energetic, motivating tone
- **Educational content**: Use "alloy" for clear, neutral delivery
- **Relaxation/Wellness**: Use "echo" for calm, soothing voice
- **Professional/Business**: Use "onyx" for authoritative tone
- **Creative/Artistic**: Use "shimmer" for soft, inspiring delivery
...
"""
```

Nguyên tắc thiết kế chính:
- Tham chiếu toàn bộ kế hoạch nội dung từ bước trước qua `{content_planner_output}`
- Cung cấp hướng dẫn chọn giọng nhưng **ủy quyền quyết định cuối cùng cho LLM**
- Tách `input` (văn bản đọc) và `instructions` (tông giọng, tốc độ, v.v.) cho mỗi cảnh để kiểm soát chi tiết

**4) Công cụ generate_narrations - Sử dụng API TTS**

```python
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext
from typing import List, Dict, Any

client = OpenAI()


async def generate_narrations(
    tool_context: ToolContext,
    voice: str,
    voice_instructions: List[Dict[str, Any]]
):
    """
    Generate narration audio for each scene using OpenAI TTS API

    Args:
        tool_context: Tool context to access artifacts and save files
        voice: Selected voice for TTS (alloy, echo, fable, onyx, nova, shimmer)
        voice_instructions: List of dictionaries containing narration
                           instructions for each scene
    """

    existing_artifacts = await tool_context.list_artifacts()
    generated_narrations = []

    for instruction in voice_instructions:
        text_input = instruction.get("input")
        instructions = instruction.get("instructions")
        scene_id = instruction.get("scene_id")
        filename = f"scene_{scene_id}_narration.mp3"

        # Caching: bỏ qua nếu đã tồn tại
        if filename in existing_artifacts:
            generated_narrations.append({
                "scene_id": scene_id,
                "filename": filename,
                "input": text_input,
                "instructions": instructions[:50],
            })
            continue

        # Gọi API OpenAI TTS (phản hồi dạng streaming)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text_input,
            instructions=instructions,
        ) as response:
            audio_data = response.read()

        # Lưu dưới dạng ADK Artifact
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="audio/mpeg",
                data=audio_data
            )
        )

        await tool_context.save_artifact(
            filename=filename,
            artifact=artifact,
        )

        generated_narrations.append({
            "scene_id": scene_id,
            "filename": filename,
            "input": text_input,
            "instructions": instructions[:50],
        })

        return {
            "success": True,
            "narrations": generated_narrations,
            "total_narrations": len(generated_narrations),
        }
```

**Giải thích chi tiết các khái niệm chính:**

**(a) Thiết kế tham số hàm công cụ**

```python
async def generate_narrations(
    tool_context: ToolContext,    # ADK tự động tiêm
    voice: str,                   # LLM chọn và truyền
    voice_instructions: List[Dict[str, Any]]  # LLM tạo và truyền
):
```

`tool_context` được ADK tự động tiêm, trong khi `voice` và `voice_instructions` được LLM tạo với giá trị phù hợp theo hướng dẫn prompt. Đây là sức mạnh của công cụ ADK -- **LLM xác định các đối số một cách thông minh khi gọi công cụ**.

**(b) Nhận dữ liệu TTS qua phản hồi streaming**

```python
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice=voice,
    input=text_input,
    instructions=instructions,
) as response:
    audio_data = response.read()
```

Sử dụng `with_streaming_response` cho phép nhận dữ liệu âm thanh theo cách streaming, hiệu quả về bộ nhớ. Mô hình `gpt-4o-mini-tts` cho phép kiểm soát chi tiết tốc độ nói, tông giọng, cảm xúc, v.v. thông qua tham số `instructions`.

**(c) Quy ước đặt tên tệp**

```python
filename = f"scene_{scene_id}_narration.mp3"   # Âm thanh
filename = f"scene_{scene_id}_image.jpeg"       # Hình ảnh (từ 11.3)
```

Sử dụng quy ước đặt tên nhất quán cho phép VideoAssembler dễ dàng tìm và sắp xếp tệp theo số cảnh sau này.

#### Điểm thực hành
- Thử các tham số `voice` khác nhau để so sánh sự khác biệt giọng cho cùng văn bản.
- Thử các hướng dẫn khác như "Speak very slowly and dramatically".
- Chuyển từ ParallelAgent sang SequentialAgent và đo sự khác biệt thời gian thực thi.

---

### 11.5 Video Assembly - Tác tử lắp ráp video

#### Chủ đề và mục tiêu
Triển khai tác tử lắp ráp tất cả artifact hình ảnh và âm thanh đã tạo thành video MP4 cuối cùng sử dụng FFmpeg. Tìm hiểu về tải Artifact, quản lý tệp tạm và xây dựng đồ thị bộ lọc FFmpeg.

#### Khái niệm chính

**1) Định nghĩa VideoAssemblerAgent**

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from .prompt import VIDEO_ASSEMBLER_DESCRIPTION, VIDEO_ASSEMBLER_PROMPT
from .tools import assemble_video

MODEL = LiteLlm(model="openai/gpt-4o")

video_assembler_agent = Agent(
    name="VideoAssemblerAgent",
    model=MODEL,
    description=VIDEO_ASSEMBLER_DESCRIPTION,
    instruction=VIDEO_ASSEMBLER_PROMPT,
    output_key="video_assembler_output",
    tools=[
        assemble_video,
    ],
)
```

**2) Kết nối toàn bộ pipeline vào tác tử gốc**

```python
shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),     # Bước 1
        AgentTool(agent=asset_generator_agent),      # Bước 2
        AgentTool(agent=video_assembler_agent),      # Bước 3 (đã thêm!)
    ],
)
```

Toàn bộ pipeline giờ đã hoàn thành.

**3) Công cụ assemble_video - Lắp ráp video với FFmpeg**

Công cụ này là phần phức tạp nhất của dự án. Hãy phân tích từng bước.

**(a) Tải Artifact và tạo tệp tạm**

```python
async def assemble_video(tool_context: ToolContext) -> str:
    temp_files = []  # Theo dõi tệp tạm để dọn dẹp

    try:
        # Lấy thông tin cảnh từ kế hoạch nội dung
        content_planner_output = tool_context.state.get(
            "content_planner_output", {}
        )
        scenes = content_planner_output.get("scenes", [])

        # Truy vấn danh sách artifact đã lưu
        existing_artifacts = await tool_context.list_artifacts()

        # Phân loại và sắp xếp tệp hình ảnh/âm thanh
        image_files = []
        audio_files = []
        for artifact_name in existing_artifacts:
            if artifact_name.endswith("_image.jpeg"):
                image_files.append(artifact_name)
            elif artifact_name.endswith("_narration.mp3"):
                audio_files.append(artifact_name)

        # Sắp xếp theo số cảnh
        def extract_scene_number(filename):
            match = re.search(r"scene_(\d+)_", filename)
            return int(match.group(1)) if match else 0

        image_files.sort(key=extract_scene_number)
        audio_files.sort(key=extract_scene_number)
```

Vì ADK Artifact được lưu trong bộ nhớ/phiên, chúng phải được trích xuất thành **tệp tạm** để sử dụng với FFmpeg:

```python
        temp_image_paths = []
        temp_audio_paths = []

        for i, (image_name, audio_name) in enumerate(
            zip(image_files, audio_files)
        ):
            # Lưu artifact hình ảnh vào tệp tạm
            image_artifact = await tool_context.load_artifact(
                filename=image_name
            )
            if image_artifact and image_artifact.inline_data:
                temp_image = tempfile.NamedTemporaryFile(
                    suffix=".jpeg", delete=False
                )
                temp_image.write(image_artifact.inline_data.data)
                temp_image.close()
                temp_image_paths.append(temp_image.name)
                temp_files.append(temp_image.name)  # Thêm vào danh sách dọn dẹp

            # Xử lý artifact âm thanh tương tự
            audio_artifact = await tool_context.load_artifact(
                filename=audio_name
            )
            if audio_artifact and audio_artifact.inline_data:
                temp_audio = tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                )
                temp_audio.write(audio_artifact.inline_data.data)
                temp_audio.close()
                temp_audio_paths.append(temp_audio.name)
                temp_files.append(temp_audio.name)
```

**(b) Xây dựng đồ thị bộ lọc FFmpeg**

Sử dụng `filter_complex` của FFmpeg để kết hợp nhiều hình ảnh và âm thanh thành một video:

```python
        input_args = []
        filter_parts = []

        for i, (temp_image, temp_audio) in enumerate(
            zip(temp_image_paths, temp_audio_paths)
        ):
            # Thêm hình ảnh và âm thanh của mỗi cảnh làm đầu vào
            input_args.extend(["-i", temp_image, "-i", temp_audio])

            scene_duration = scenes[i].get("duration", 4)

            # Tạo luồng video: lặp hình ảnh tĩnh trong thời gian chỉ định
            total_frames = int(30 * scene_duration)  # 30fps * giây
            filter_parts.append(
                f"[{i*2}:v]scale=1080:1920:"
                f"force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps=30,"
                f"loop={total_frames-1}:size=1:start=0[v{i}]"
            )

            # Luồng âm thanh (đi qua không chuyển đổi)
            filter_parts.append(f"[{i*2+1}:a]anull[a{i}]")

        # Nối tất cả các luồng
        video_inputs = "".join(
            [f"[v{i}]" for i in range(len(scenes))]
        )
        audio_inputs = "".join(
            [f"[a{i}]" for i in range(len(scenes))]
        )
        filter_parts.append(
            f"{video_inputs}concat=n={len(scenes)}:v=1:a=0[outv]"
        )
        filter_parts.append(
            f"{audio_inputs}concat=n={len(scenes)}:v=0:a=1[outa]"
        )
```

**Giải thích đồ thị bộ lọc FFmpeg:**

Cho mỗi cảnh:
1. `scale=1080:1920:force_original_aspect_ratio=decrease` - Co giãn thành 1080x1920 (dọc) giữ nguyên tỷ lệ
2. `pad=1080:1920:(ow-iw)/2:(oh-ih)/2` - Đệm phần thiếu (letterbox)
3. `setsar=1` - Đặt tỷ lệ pixel thành 1:1
4. `fps=30` - Đặt 30fps
5. `loop={total_frames-1}:size=1:start=0` - Lặp hình ảnh tĩnh theo số khung hình chỉ định

Bộ lọc `concat` nối tất cả luồng video và âm thanh theo thứ tự.

**(c) Thực thi FFmpeg và lưu video cuối cùng**

```python
        ffmpeg_cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",    # Codec video H.264
                "-c:a", "aac",        # Codec âm thanh AAC
                "-pix_fmt", "yuv420p", # Định dạng pixel tương thích cao
                "-r", "30",           # 30fps
                output_path,
            ]
        )

        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        # Lưu video cuối cùng dưới dạng artifact
        with open(output_path, "rb") as f:
            video_data = f.read()

        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="video/mp4", data=video_data
            )
        )

        await tool_context.save_artifact(
            filename="youtube_short_final.mp4", artifact=artifact
        )
```

**(d) Dọn dẹp tệp tạm - Mẫu finally**

```python
    finally:
        # Dọn dẹp tệp tạm
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Failed to cleanup {temp_file}: {e}")
```

Mẫu `try/finally` đảm bảo tệp tạm luôn được dọn dẹp ngay cả khi xảy ra lỗi. Đây là mẫu quản lý tài nguyên quan trọng trong mã production.

#### Điểm thực hành
- Thay đổi tùy chọn `scale` và `pad` của bộ lọc FFmpeg để thử các độ phân giải khác nhau.
- Điều chỉnh giá trị `loop` để thay đổi thời lượng cảnh.
- Kiểm tra phần xử lý lỗi và quan sát lỗi nào xảy ra khi FFmpeg chưa được cài đặt.
- Nghiên cứu cách sử dụng pipe thay vì tệp tạm.

---

### 11.6 Callbacks - Kiểm soát hành vi tác tử thông qua Callback

#### Chủ đề và mục tiêu
Hiểu hệ thống callback của ADK và học cách sử dụng `before_model_callback` để kiểm tra và lọc đầu vào người dùng trước khi gọi LLM.

#### Khái niệm chính

**1) Callback là gì?**

Callback là các hàm có thể can thiệp tại các thời điểm cụ thể trong hoạt động của tác tử. ADK cung cấp nhiều điểm callback:
- `before_model_callback`: Thực thi **trước** khi gọi LLM
- `after_model_callback`: Thực thi **sau** khi gọi LLM
- `before_tool_callback`: Thực thi **trước** khi gọi công cụ
- `after_tool_callback`: Thực thi **sau** khi gọi công cụ

**2) Triển khai before_model_callback**

```python
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    # Kiểm tra tin nhắn cuối cùng trong lịch sử hội thoại
    history = llm_request.contents
    last_message = history[-1]

    # Chỉ kiểm tra nếu là tin nhắn người dùng
    if last_message.role == "user":
        text = last_message.parts[0].text

        # Kiểm tra từ khóa bị cấm
        if "hummus" in text:
            # Trả về phản hồi từ chối ngay lập tức mà không gọi LLM
            return LlmResponse(
                content=types.Content(
                    parts=[
                        types.Part(
                            text="Sorry I can't help with that."
                        ),
                    ],
                    role="model",
                )
            )

    # Trả về None để tiếp tục gọi LLM bình thường
    return None
```

**Cách thức hoạt động của Callback:**

```
Đầu vào người dùng
    |
    v
before_model_callback thực thi
    |
    +-- Trả về LlmResponse --> Bỏ qua gọi LLM và trả phản hồi ngay
    |
    +-- Trả về None ---------> Tiếp tục gọi LLM bình thường
                                |
                                v
                            LLM tạo phản hồi
```

- **Nếu trả về `LlmResponse`**: Lệnh gọi LLM bị chặn hoàn toàn, và phản hồi được trả về được sử dụng nguyên trạng. Không phát sinh chi phí API.
- **Nếu trả về `None`**: Lệnh gọi LLM bình thường tiếp tục.

**3) Đăng ký Callback**

```python
shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),
        AgentTool(agent=asset_generator_agent),
        AgentTool(agent=video_assembler_agent),
    ],
    before_model_callback=before_model_callback,  # Đăng ký callback!
)
```

Khi truyền hàm vào tham số `before_model_callback`, callback được thực thi trước mỗi lần gọi LLM từ tác tử này.

**4) Các trường hợp sử dụng Callback**

Ví dụ này cho thấy lọc từ khóa đơn giản, nhưng trong dự án thực tế, callback có thể được sử dụng đa dạng:

| Trường hợp sử dụng | Mô tả |
|---------------------|-------|
| **Lọc nội dung** | Chặn yêu cầu không phù hợp trước khi gửi đến LLM |
| **Kiểm soát chi phí** | Giới hạn gọi LLM khi vượt quá số token nhất định |
| **Ghi log/Giám sát** | Ghi log tất cả các lần gọi LLM |
| **Caching** | Trả về phản hồi đã cache cho các yêu cầu giống nhau |
| **Sửa đổi prompt** | Sửa đổi prompt gửi đến LLM một cách động |
| **Xác thực/Phân quyền** | Hạn chế tính năng theo quyền người dùng |

**5) CallbackContext và LlmRequest**

```python
def before_model_callback(
    callback_context: CallbackContext,  # Truy cập tên tác tử, trạng thái, v.v.
    llm_request: LlmRequest,           # Nội dung yêu cầu gửi đến LLM
):
```

- `callback_context.agent_name`: Tên tác tử đang thực thi
- `llm_request.contents`: Lịch sử hội thoại (gồm role, parts)
- Sửa đổi `llm_request` có thể thay đổi chính yêu cầu gửi đến LLM

#### Điểm thực hành
- Thêm bộ lọc từ khóa khác ngoài "hummus".
- Triển khai `after_model_callback` để hậu xử lý phản hồi LLM.
- In `callback_context.agent_name` để quan sát tác tử nào kích hoạt callback.
- Thêm `before_tool_callback` để triển khai ghi log trước khi gọi công cụ.
- Sửa đổi `llm_request.contents` trong callback để thêm tin nhắn hệ thống động.

---

## 3. Tóm tắt chương

### Tóm tắt các loại tác tử ADK

| Loại tác tử | Lớp | Đặc điểm | Sử dụng trong dự án này |
|-------------|------|----------|------------------------|
| **Agent** | `google.adk.agents.Agent` | Suy luận dựa trên LLM, có thể sử dụng công cụ | ShortsProducer, ContentPlanner, PromptBuilder, ImageBuilder, VoiceGenerator, VideoAssembler |
| **ParallelAgent** | `google.adk.agents.ParallelAgent` | Thực thi tác tử con song song | AssetGenerator (tạo hình ảnh + giọng nói đồng thời) |
| **SequentialAgent** | `google.adk.agents.SequentialAgent` | Thực thi tác tử con tuần tự | ImageGenerator (tối ưu prompt -> tạo hình ảnh) |

### Tóm tắt các khái niệm ADK chính

| Khái niệm | Mô tả | Mã liên quan |
|-----------|-------|-------------|
| **AgentTool** | Bọc tác tử như công cụ để tác tử khác gọi | `AgentTool(agent=content_planner_agent)` |
| **output_schema** | Bắt buộc cấu trúc đầu ra tác tử với mô hình Pydantic | `output_schema=ContentPlanOutput` |
| **output_key** | Khóa lưu đầu ra tác tử vào trạng thái phiên | `output_key="content_planner_output"` |
| **ToolContext** | Truy cập state/artifact từ hàm công cụ | `tool_context.state.get(...)` |
| **Artifact** | Lưu trữ/quản lý dữ liệu nhị phân (hình ảnh, âm thanh, v.v.) | `tool_context.save_artifact(...)` |
| **Callback** | Hàm can thiệp tại điểm hoạt động cụ thể của tác tử | `before_model_callback=...` |
| **Biến mẫu prompt** | Tự động chèn giá trị state qua `{tên_biến}` | `{content_planner_output}` |

### Tóm tắt mẫu thiết kế

1. **Mẫu điều phối**: Tác tử gốc phối hợp các tác tử con theo thứ tự
2. **Mẫu pipeline**: SequentialAgent kết nối các bước xử lý dữ liệu tuần tự
3. **Mẫu xử lý song song**: ParallelAgent chạy các tác vụ độc lập đồng thời
4. **Mẫu caching Artifact**: Kiểm tra artifact tồn tại để ngăn tạo trùng
5. **Mẫu quản lý tệp tạm**: Đảm bảo dọn dẹp tài nguyên với try/finally
6. **Mẫu bảo vệ Callback**: Xác thực và lọc đầu vào với before_model_callback

---

## 4. Bài tập thực hành

### Bài tập 1: Cơ bản - Thêm loại nội dung mới
Sửa đổi prompt của ContentPlannerAgent để hỗ trợ YouTube Shorts kiểu "tóm tắt tin tức". Bao gồm tông giọng khẩn cấp đặc trưng của tin tức, tường thuật tập trung vào sự kiện, và mô tả trực quan kiểu đồ họa tin tức.

### Bài tập 2: Trung cấp - Thêm tác tử phụ đề
Tạo `SubtitleGeneratorAgent` mới tạo tệp phụ đề SRT dựa trên văn bản tường thuật mỗi cảnh, và thêm vào ParallelAgent của AssetGeneratorAgent. Sửa đổi lệnh FFmpeg của VideoAssemblerAgent để ghi phụ đề vào video (burn-in).

### Bài tập 3: Trung cấp - Triển khai nhiều Callback
Triển khai các callback sau:
- `before_model_callback`: Lọc sử dụng danh sách từ khóa bị cấm trên đầu vào người dùng
- `after_model_callback`: Đếm và ghi log số token của phản hồi LLM
- `before_tool_callback`: Ghi lại thời gian bắt đầu gọi công cụ
- `after_tool_callback`: Tính thời gian thực thi công cụ và xuất log hiệu suất

### Bài tập 4: Nâng cao - Thêm tác tử nhạc nền
Triển khai `BGMAgent` tạo hoặc chọn nhạc nền và thêm vào AssetGeneratorAgent. Sửa đổi lệnh FFmpeg của VideoAssemblerAgent để trộn tường thuật và BGM (có điều chỉnh âm lượng) vào video cuối cùng.

### Bài tập 5: Nâng cao - Cơ chế phục hồi lỗi
Triển khai logic thử lại tự động khi lỗi API xảy ra trong quá trình tạo hình ảnh hoặc giọng nói. Bao gồm tối đa 3 lần thử lại, exponential backoff, và bảo tồn artifact đã tạo thành công khi thành công một phần.

---

## Phụ lục: Cách chạy dự án

### Thiết lập môi trường

```bash
# Di chuyển đến thư mục dự án
cd youtube-shorts-maker

# Cài đặt phụ thuộc (sử dụng uv)
uv sync

# Thiết lập biến môi trường
export OPENAI_API_KEY="your-openai-api-key"

# Xác nhận cài đặt FFmpeg
ffmpeg -version
```

### Chạy ADK

```bash
# Chạy với server phát triển ADK
adk web youtube_shorts_maker

# Hoặc chạy với CLI
adk run youtube_shorts_maker
```

### Cấu trúc thư mục cuối cùng

```
youtube-shorts-maker/
    youtube_shorts_maker/
        __init__.py
        agent.py                          # Tác tử gốc + callback
        prompt.py                         # Prompt điều phối
        sub_agents/
            content_planner/
                agent.py                  # ContentPlannerAgent
                prompt.py                 # Prompt lập kế hoạch nội dung
            asset_generator/
                agent.py                  # AssetGeneratorAgent (ParallelAgent)
                prompt.py                 # Mô tả trình tạo tài nguyên
                image_generator/
                    agent.py              # ImageGeneratorAgent (SequentialAgent)
                    prompt_builder/
                        agent.py          # PromptBuilderAgent
                        prompt.py         # Prompt của prompt builder
                    image_builder/
                        agent.py          # ImageBuilderAgent
                        prompt.py         # Prompt của image builder
                        tools.py          # Công cụ generate_images
                voice_generator/
                    agent.py              # VoiceGeneratorAgent
                    prompt.py             # Prompt trình tạo giọng nói
                    tools.py              # Công cụ generate_narrations
            video_assembler/
                agent.py                  # VideoAssemblerAgent
                prompt.py                 # Prompt lắp ráp video
                tools.py                  # Công cụ assemble_video (FFmpeg)
```
