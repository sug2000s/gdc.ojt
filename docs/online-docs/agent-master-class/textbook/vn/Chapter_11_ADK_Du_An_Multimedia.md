# Chapter 11: Du an Multimedia ADK - Trinh tu dong tao YouTube Shorts

---

## 1. Tong quan chuong

Trong chuong nay, chung ta se su dung Google ADK (Agent Development Kit) de xay dung **he thong multi-agent tu dong tao video YouTube Shorts**. Khi nguoi dung nhap chu de, cac AI agent se tu dong xu ly toan bo quy trinh tu len ke hoach noi dung, tao hinh anh, tao giong doc narration, den lap rap video.

### Ten du an
**youtube-shorts-maker** - Trinh tu dong san xuat video YouTube Shorts dang doc (9:16)

### Muc tieu hoc tap
- Hieu va su dung cac loai agent da dang cua Google ADK (Agent, ParallelAgent, SequentialAgent)
- Thiet ke pattern dieu phoi multi-agent (orchestration)
- Cau truc hoa dau ra cua agent bang Pydantic schema
- Tich hop OpenAI API (GPT-Image-1, TTS) lam cong cu (Tool)
- Quan ly file multimedia bang he thong Artifact cua ADK
- Lap rap video bang FFmpeg
- Dieu khien hanh vi agent thong qua Callback

### Kien truc tong the

```
ShortsProducerAgent (root orchestrator)
    |
    +-- ContentPlannerAgent (len ke hoach noi dung)
    |
    +-- AssetGeneratorAgent (ParallelAgent - tao asset song song)
    |       |
    |       +-- ImageGeneratorAgent (SequentialAgent - tao hinh anh tuan tu)
    |       |       |
    |       |       +-- PromptBuilderAgent (toi uu hoa prompt)
    |       |       +-- ImageBuilderAgent (tao hinh anh)
    |       |
    |       +-- VoiceGeneratorAgent (tao giong doc narration)
    |
    +-- VideoAssemblerAgent (lap rap video cuoi cung)
```

### Luong workflow
1. **Phase 1**: Thu thap dau vao nguoi dung va xac nhan yeu cau
2. **Phase 2**: ContentPlannerAgent tao script co cau truc
3. **Phase 3**: AssetGeneratorAgent tao hinh anh va giong doc **song song**
4. **Phase 4**: VideoAssemblerAgent lap rap video MP4 cuoi cung bang FFmpeg
5. **Phase 5**: Giao san pham cuoi cung

---

## 2. Mo ta chi tiet tung section

---

### 11.0 Introduction - Thiet lap ban dau du an

#### Chu de va Muc tieu
Thiet lap cau truc co ban cua du an va dinh nghia cac goi phu thuoc can thiet. Hieu cau truc thu muc chuan cua du an ADK.

#### Giai thich khai niem cot loi

**Cau truc du an ADK**: Google ADK yeu cau mot cau truc thu muc cu the. Can dat `agent.py` va `__init__.py` trong thu muc cung ten voi ten goi.

**Phu thuoc du an**:

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

Vai tro cua tung phu thuoc:
- **google-adk**: Google Agent Development Kit - cot loi cua framework agent
- **google-genai**: Thu vien Generative AI cua Google (su dung cho dinh nghia kieu va quan ly Artifact)
- **litellm**: Thu vien tich hop nhieu nha cung cap LLM (cho phep su dung model OpenAI trong ADK)
- **openai**: OpenAI API client (su dung truc tiep cho tao hinh anh, TTS)

**File `__init__.py`**:

```python
from . import agent
```

Dong nay dam bao rang ADK tu dong import module `agent.py` khi tai goi. ADK se tim va thuc thi `root_agent` tu module nay.

**Cau truc thu muc ban dau**:

```
youtube-shorts-maker/
    .python-version          # Chi dinh Python 3.13
    pyproject.toml           # Cau hinh du an va phu thuoc
    README.md
    uv.lock                  # File khoa cua uv package manager
    youtube_shorts_maker/
        __init__.py          # Khoi tao goi
        agent.py             # Root agent (file trong)
        sub_agents/
            content_planner/
                agent.py     # Agent len ke hoach noi dung (file trong)
                prompt.py    # Dinh nghia prompt (file trong)
```

#### Diem thuc hanh
- Lam quen voi cach khoi tao du an su dung `uv`.
- Nam vung cau truc thu muc ma ADK yeu cau (import module `agent` trong `__init__.py`, bien `root_agent`).
- Ghi nho pattern module hoa agent bang thu muc `sub_agents/`.

---

### 11.1 Content Planner Agent - Agent len ke hoach noi dung

#### Chu de va Muc tieu
Xay dung agent xuat ke hoach noi dung toan bo YouTube Shorts duoi dang JSON co cau truc. Hoc cach su dung `output_schema` voi Pydantic model va `output_key`.

#### Giai thich khai niem cot loi

**1) Root Agent (ShortsProducerAgent) - Pattern Orchestrator**

Orchestrator la agent cap tren dieu phoi nhieu agent con de thuc hien tac vu phuc tap. Su dung `AgentTool` de goi agent con nhu mot cong cu.

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

Diem cot loi:
- **`LiteLlm(model="openai/gpt-4o")`**: ADK mac dinh su dung model Google, nhung thong qua wrapper `LiteLlm` co the su dung cac model khac nhu GPT-4o cua OpenAI.
- **`AgentTool(agent=...)`**: Boc agent con thanh cong cu (Tool) de root agent co the goi khi can. Day la mot trong nhung pattern cot loi cua ADK.
- **`root_agent`**: Bien diem vao ma ADK tim kiem khi thuc thi.

**2) Thiet ke prompt Orchestrator**

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

Nguyen tac thiet ke prompt orchestrator:
- **Chi ro workflow tung buoc**: Chi dinh ro agent phai lam gi theo thu tu nao
- **Chi dinh agent con cho tung buoc**: ContentPlanner -> AssetGenerator -> VideoAssembler
- **Bao gom huong dan xu ly loi va giao tiep voi nguoi dung**

**3) ContentPlannerAgent - Dau ra co cau truc (output_schema)**

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

Diem cot loi:
- **`output_schema`**: Khi chi dinh Pydantic model, dau ra cua agent bat buoc phai tuong ung voi schema do. Buoc LLM tra loi theo cau truc JSON da dinh thay vi van ban tu do.
- **`output_key`**: Ten khoa luu dau ra cua agent vao trang thai phien (state). Khi dat la `"content_planner_output"`, cac agent khac co the truy cap qua `tool_context.state.get("content_planner_output")`.
- **`Field(description=...)`**: Mo ta cua tung truong dong vai tro huong dan de LLM tao ra gia tri dung.

**4) Chien luoc cot loi cua prompt Content Planner**

Cac diem dang chu y trong prompt:
- **Gioi han toi da 20 giay**: Rang buoc phu hop voi dac diem YouTube Shorts
- **Linh hoat so canh**: Agent tu quyet dinh cau hinh toi uu tu 3-6 canh
- **Chien luoc thoi gian**: Intro nhanh (2-3 giay), noi dung chinh (3-5 giay), ket manh (2-4 giay)
- **Yeu cau kiem tra**: Chi thi xac nhan tong thoi gian khong vuot qua 20 giay truoc khi xuat
- **Cung cap vi du cu the**: Vi du "Perfect Scrambled Eggs" cho thay ro dang dau ra mong doi

#### Phan tich code - Luong du lieu giua cac agent

```
Dau vao nguoi dung ("Tao Shorts ve chu de nau an")
    |
    v
ShortsProducerAgent (orchestrator)
    |  Goi qua AgentTool
    v
ContentPlannerAgent
    |  output_schema: ContentPlanOutput
    |  output_key: "content_planner_output"
    v
Luu vao trang thai phien: state["content_planner_output"] = {
    "topic": "...",
    "total_duration": 18,
    "scenes": [
        {"id": 1, "narration": "...", "visual_description": "...", ...},
        {"id": 2, ...},
        ...
    ]
}
```

#### Diem thuc hanh
- Thu nghiem anh huong cua `Field(description=...)` trong Pydantic model len dau ra LLM.
- Thay doi `output_key` va test cach truy cap tu agent tiep theo.
- Thay doi vi du (Example) trong prompt va quan sat su thay doi chat luong dau ra.

---

### 11.2 Prompt Builder Agent - Agent toi uu hoa prompt hinh anh

#### Chu de va Muc tieu
Xay dung agent chuyen doi mo ta hinh anh tu ke hoach noi dung thanh prompt toi uu cho GPT-Image-1. Hieu su khac biet giua ParallelAgent va SequentialAgent, va thiet ke cau truc phan cap agent.

#### Giai thich khai niem cot loi

**1) Su dung ba loai agent**

Trong section nay, ca ba loai agent chinh cua ADK deu xuat hien:

```python
# 1. ParallelAgent - Thuc thi cac agent con dong thoi
from google.adk.agents import ParallelAgent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,   # Pipeline tao hinh anh
        # voice_generator_agent, # (Se them o 11.4)
    ],
)
```

**ParallelAgent** thuc thi cac agent trong danh sach `sub_agents` **dong thoi (song song)**. Tao hinh anh va tao giong doc khong phu thuoc nhau nen co the xu ly song song.

```python
# 2. SequentialAgent - Thuc thi cac agent con theo thu tu
from google.adk.agents import SequentialAgent

image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,   # Truoc: toi uu hoa prompt
        # image_builder_agent,  # Sau: tao hinh anh (se them o 11.3)
    ],
)
```

**SequentialAgent** thuc thi `sub_agents` **theo thu tu**. Phai toi uu hoa prompt truoc roi moi tao hinh anh duoc, nen xu ly tuan tu la bat buoc.

```python
# 3. Agent - Agent chung dua tren LLM (co the su dung cong cu, suy luan)
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

**Agent** la loai agent co ban nhat, su dung LLM de suy luan, goi cong cu va tao dau ra co cau truc.

**2) Pattern thiet ke cau truc phan cap agent**

```
AssetGeneratorAgent (ParallelAgent)
    |
    +-- ImageGeneratorAgent (SequentialAgent)
    |       |
    |       +-- PromptBuilderAgent (Agent) -- Toi uu hoa prompt
    |       +-- ImageBuilderAgent (Agent)  -- Tao hinh anh
    |
    +-- VoiceGeneratorAgent (Agent)         -- Tao giong doc
```

Nguyen ly cot loi cua thiet ke nay:
- **Tao hinh anh** phai toi uu hoa prompt -> tao hinh anh theo **thu tu** (SequentialAgent)
- **Tao hinh anh** va **tao giong doc** co the chay **song song** (ParallelAgent)
- Moi tac vu rieng le can suy luan LLM (Agent)

**3) Schema dau ra cua Prompt Builder**

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

**4) Truyen du lieu giua cac agent qua State**

Trong prompt su dung `{content_planner_output}` de tham chieu dau ra cua agent truoc:

```python
PROMPT_BUILDER_PROMPT = """
...
## Your Task:
Take the structured content plan: {content_planner_output} and create
optimized vertical image generation prompts for each scene...
"""
```

Cu phap `{ten_bien}` trong `instruction` cua ADK tu dong inject gia tri tuong ung tu trang thai phien. Day la cach ket noi voi `output_key`:
1. ContentPlannerAgent luu ket qua voi `output_key="content_planner_output"`
2. PromptBuilderAgent tham chieu du lieu do qua `{content_planner_output}` trong `instruction`

**5) Chien luoc toi uu hoa prompt**

Cac cong viec nang cao (Enhancement) ma prompt builder thuc hien:
- **Them thong so ky thuat**: Ti le doc 9:16, do phan giai 1080x1920
- **Chi tiet hinh anh**: Anh sang, goc camera, bo cuc
- **Chi thi text overlay**: Vi tri, padding, do doc
- **Nhat quan phong cach**: Duy tri cung phong cach hinh anh cho tat ca canh

Vi du chuyen doi:
```
Goc: "Stovetop dial on low"
Toi uu: "Close-up shot of modern stovetop control dial set to low heat
setting, 9:16 portrait aspect ratio, 1080x1920 resolution, vertical
composition, warm kitchen lighting, shallow depth of field, photorealistic,
sharp focus, with bold white text 'Secret #1: Low Heat' positioned at
top center of image with generous padding from borders..."
```

#### Diem thuc hanh
- Hoan doi `ParallelAgent` va `SequentialAgent` va quan sat su khac biet ve thu tu thuc thi.
- Kiem tra gia tri thuc te duoc thay the cho bien template `{content_planner_output}` qua log.
- Sua doi chi thi toi uu hoa prompt de thay doi phong cach hinh anh duoc tao.

---

### 11.3 Image Builder Agent - Agent tao hinh anh

#### Chu de va Muc tieu
Xay dung agent su dung prompt da toi uu de tao hinh anh thuc te bang OpenAI GPT-Image-1 API va luu tru bang he thong Artifact cua ADK.

#### Giai thich khai niem cot loi

**1) Dinh nghia Agent - Agent co cong cu**

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

Agent nay su dung ham Python `generate_images` lam cong cu. Trong ADK, khi truyen ham Python thong thuong vao danh sach `tools`, no se tu dong duoc dang ky lam cong cu.

**2) Them ImageBuilder vao SequentialAgent**

```python
image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,    # Buoc 1: Toi uu hoa prompt
        image_builder_agent,     # Buoc 2: Tao hinh anh
    ],
)
```

Pipeline tao hinh anh da hoan thanh. PromptBuilder chay truoc de luu prompt da toi uu vao state, sau do ImageBuilder doc prompt do de tao hinh anh.

**3) Cong cu generate_images - Cot loi cua he thong Artifact ADK**

```python
import base64
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext

client = OpenAI()


async def generate_images(tool_context: ToolContext):

    # 1. Lay dau ra cua agent truoc tu trang thai phien
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")

    # 2. Kiem tra danh sach artifact da tao (tranh tao trung lap)
    existing_artifacts = await tool_context.list_artifacts()

    generated_images = []

    for prompt in optimized_prompts:
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        filename = f"scene_{scene_id}_image.jpeg"

        # 3. Neu da ton tai thi bo qua (caching)
        if filename in existing_artifacts:
            generated_images.append({
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100],
                "filename": filename,
            })
            continue

        # 4. Tao hinh anh bang OpenAI GPT-Image-1 API
        image = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",    # Dang doc (ti le 2:3)
        )

        # 5. Giai ma Base64
        image_bytes = base64.b64decode(image.data[0].b64_json)

        # 6. Luu lam ADK Artifact
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

**Giai thich chi tiet khai niem cot loi:**

**(a) `ToolContext` - Context cot loi cua cong cu**

`tool_context: ToolContext` la tham so dac biet ma ADK tu dong inject vao ham cong cu. Thong qua do:
- `tool_context.state`: Truy cap trang thai chia se cua phien (chia se du lieu giua cac agent)
- `tool_context.list_artifacts()`: Truy van danh sach artifact da luu
- `tool_context.save_artifact()`: Luu artifact moi
- `tool_context.load_artifact()`: Tai artifact

ADK tu dong inject khi ham co `tool_context: ToolContext` trong signature. Nguoi dung khong can truyen truc tiep.

**(b) He thong Artifact**

Artifact la co che quan ly du lieu nhi phan (hinh anh, audio, video...) trong ADK. Su dung `google.genai.types.Part` va `types.Blob` de luu du lieu kem MIME type.

```python
artifact = types.Part(
    inline_data=types.Blob(
        mime_type="image/jpeg",
        data=image_bytes,      # raw bytes
    )
)
await tool_context.save_artifact(filename=filename, artifact=artifact)
```

Thay vi luu vao file system, luu vao he thong quan ly phien cua ADK giup du lieu duoc duy tri giua cac phien va cac agent khac co the truy cap.

**(c) Pattern chong tao trung lap**

```python
existing_artifacts = await tool_context.list_artifacts()
if filename in existing_artifacts:
    # Da ton tai thi bo qua
    continue
```

Neu hinh anh da duoc tao thi khong tao lai. Day la pattern quan trong giup tiet kiem chi phi va thoi gian goi API.

**(d) Tham so OpenAI Image Generation API**

```python
image = client.images.generate(
    model="gpt-image-1",       # Model tao hinh anh cua OpenAI
    prompt=enhanced_prompt,     # Prompt da toi uu
    n=1,                        # Tao 1 hinh
    quality="low",              # Chat luong (low/medium/high)
    moderation="low",           # Muc loc noi dung
    output_format="jpeg",       # Dinh dang dau ra
    background="opaque",        # Nen (opaque: khong trong suot)
    size="1024x1536",           # Ti le doc (YouTube Shorts)
)
```

#### Diem thuc hanh
- Thay doi tham so `quality` thanh `"high"` va so sanh su khac biet chat luong hinh anh.
- Thay doi `size` thanh `"1024x1024"` va xem ket qua.
- Test co che caching qua `tool_context.list_artifacts()` khi chay lai.

---

### 11.4 Audio Narration Agent - Agent tao giong doc narration

#### Chu de va Muc tieu
Xay dung agent su dung OpenAI TTS (Text-to-Speech) API de tao audio narration cho tung canh. Them voice generator vao ParallelAgent de hoan thanh cau truc tao hinh anh va giong doc dong thoi.

#### Giai thich khai niem cot loi

**1) Them VoiceGenerator vao ParallelAgent**

```python
from google.adk.agents import ParallelAgent
from .prompt import ASSET_GENERATOR_DESCRIPTION
from .image_generator.agent import image_generator_agent
from .voice_generator.agent import voice_generator_agent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,     # Pipeline tao hinh anh
        voice_generator_agent,     # Agent tao giong doc (them moi!)
    ],
)
```

Bay gio khi `AssetGeneratorAgent` duoc goi, tao hinh anh (`ImageGeneratorAgent`) va tao giong doc (`VoiceGeneratorAgent`) se chay **dong thoi**. Hai tac vu nay doc lap voi nhau nen xu ly song song giup giam dang ke tong thoi gian thuc thi.

**2) Dinh nghia VoiceGeneratorAgent**

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

**3) Chien luoc chon giong noi - Thiet ke prompt**

Prompt cua agent nay duoc thiet ke de LLM **tu chon** giong noi phu hop voi khong khi noi dung:

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

Nguyen tac thiet ke cot loi:
- Tham chieu toan bo ke hoach noi dung tu buoc truoc qua `{content_planner_output}`
- Cung cap huong dan chon giong noi nhung **uy quyen quyet dinh cuoi cung cho LLM**
- Tach `input` (van ban doc) va `instructions` (giong dieu, toc do) cho tung canh de dieu khien chi tiet

**4) Cong cu generate_narrations - Su dung TTS API**

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

        # Caching: bo qua neu da ton tai
        if filename in existing_artifacts:
            generated_narrations.append({
                "scene_id": scene_id,
                "filename": filename,
                "input": text_input,
                "instructions": instructions[:50],
            })
            continue

        # Goi OpenAI TTS API (phan hoi streaming)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text_input,
            instructions=instructions,
        ) as response:
            audio_data = response.read()

        # Luu lam ADK Artifact
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

**Giai thich chi tiet khai niem cot loi:**

**(a) Thiet ke tham so ham cong cu**

```python
async def generate_narrations(
    tool_context: ToolContext,    # ADK tu dong inject
    voice: str,                   # LLM chon va truyen
    voice_instructions: List[Dict[str, Any]]  # LLM xay dung va truyen
):
```

`tool_context` duoc ADK tu dong inject, con `voice` va `voice_instructions` do LLM tao ra gia tri phu hop theo chi dan trong prompt. Day la diem manh cua cong cu ADK -- **LLM quyet dinh thong minh cac tham so khi goi cong cu**.

**(b) Nhan du lieu TTS qua streaming response**

```python
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice=voice,
    input=text_input,
    instructions=instructions,
) as response:
    audio_data = response.read()
```

Su dung `with_streaming_response` giup nhan du lieu audio theo kieu streaming, hieu qua hon ve bo nho. Model `gpt-4o-mini-tts` cho phep dieu khien chi tiet toc do noi, giong dieu, cam xuc qua tham so `instructions`.

**(c) Quy tac dat ten file**

```python
filename = f"scene_{scene_id}_narration.mp3"   # Audio
filename = f"scene_{scene_id}_image.jpeg"       # Hinh anh (tu 11.3)
```

Su dung quy tac dat ten nhat quan de VideoAssembler sau nay co the de dang tim va sap xep file theo so canh.

#### Diem thuc hanh
- Thay doi tham so `voice` de so sanh su khac biet giong noi voi cung van ban.
- Thu thay doi `instructions` thanh "Speak very slowly and dramatically".
- Doi ParallelAgent thanh SequentialAgent va do su khac biet thoi gian thuc thi.

---

### 11.5 Video Assembly - Agent lap rap video

#### Chu de va Muc tieu
Xay dung agent lap rap tat ca hinh anh va audio artifact da tao thanh video MP4 cuoi cung bang FFmpeg. Hoc cach tai Artifact, quan ly file tam va cau hinh filter graph cua FFmpeg.

#### Giai thich khai niem cot loi

**1) Dinh nghia VideoAssemblerAgent**

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

**2) Ket noi toan bo pipeline vao root agent**

```python
shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),     # Buoc 1
        AgentTool(agent=asset_generator_agent),      # Buoc 2
        AgentTool(agent=video_assembler_agent),      # Buoc 3 (them moi!)
    ],
)
```

Toan bo pipeline da hoan thanh.

**3) Cong cu assemble_video - Lap rap video bang FFmpeg**

Cong cu nay la phan phuc tap nhat cua du an. Phan tich tung buoc:

**(a) Tai Artifact va tao file tam**

```python
async def assemble_video(tool_context: ToolContext) -> str:
    temp_files = []  # Theo doi file tam de don dep

    try:
        # Lay thong tin canh tu ke hoach noi dung
        content_planner_output = tool_context.state.get(
            "content_planner_output", {}
        )
        scenes = content_planner_output.get("scenes", [])

        # Truy van danh sach artifact da luu
        existing_artifacts = await tool_context.list_artifacts()

        # Phan loai va sap xep file hinh anh/audio
        image_files = []
        audio_files = []
        for artifact_name in existing_artifacts:
            if artifact_name.endswith("_image.jpeg"):
                image_files.append(artifact_name)
            elif artifact_name.endswith("_narration.mp3"):
                audio_files.append(artifact_name)

        # Sap xep theo so canh
        def extract_scene_number(filename):
            match = re.search(r"scene_(\d+)_", filename)
            return int(match.group(1)) if match else 0

        image_files.sort(key=extract_scene_number)
        audio_files.sort(key=extract_scene_number)
```

Artifact cua ADK duoc luu trong bo nho/phien, nen de su dung voi FFmpeg can trich xuat ra **file tam**:

```python
        temp_image_paths = []
        temp_audio_paths = []

        for i, (image_name, audio_name) in enumerate(
            zip(image_files, audio_files)
        ):
            # Luu artifact hinh anh ra file tam
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
                temp_files.append(temp_image.name)  # Them vao danh sach don dep

            # Xu ly tuong tu cho artifact audio
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

**(b) Cau hinh FFmpeg filter graph**

Su dung `filter_complex` cua FFmpeg de ghep nhieu hinh anh va audio thanh mot video:

```python
        input_args = []
        filter_parts = []

        for i, (temp_image, temp_audio) in enumerate(
            zip(temp_image_paths, temp_audio_paths)
        ):
            # Them hinh anh va audio cua tung canh lam dau vao
            input_args.extend(["-i", temp_image, "-i", temp_audio])

            scene_duration = scenes[i].get("duration", 4)

            # Tao video stream: lap lai hinh anh tinh trong thoi gian chi dinh
            total_frames = int(30 * scene_duration)  # 30fps * giay
            filter_parts.append(
                f"[{i*2}:v]scale=1080:1920:"
                f"force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps=30,"
                f"loop={total_frames-1}:size=1:start=0[v{i}]"
            )

            # Audio stream (truyen thang khong chuyen doi)
            filter_parts.append(f"[{i*2+1}:a]anull[a{i}]")

        # Noi tat ca stream (concat)
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

**Giai thich FFmpeg filter graph:**

Cho moi canh:
1. `scale=1080:1920:force_original_aspect_ratio=decrease` - Scale ve 1080x1920 (doc), giu ti le
2. `pad=1080:1920:(ow-iw)/2:(oh-ih)/2` - Them padding cho phan thieu (letterbox)
3. `setsar=1` - Dat ti le pixel 1:1
4. `fps=30` - Dat 30fps
5. `loop={total_frames-1}:size=1:start=0` - Lap lai hinh anh tinh trong so frame chi dinh

Filter `concat` noi tat ca video stream va audio stream theo thu tu.

**(c) Thuc thi FFmpeg va luu video cuoi cung**

```python
        ffmpeg_cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",    # Video codec H.264
                "-c:a", "aac",        # Audio codec AAC
                "-pix_fmt", "yuv420p", # Dinh dang pixel tuong thich cao
                "-r", "30",           # 30fps
                output_path,
            ]
        )

        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        # Luu video cuoi cung lam artifact
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

**(d) Don dep file tam - Pattern finally**

```python
    finally:
        # Don dep file tam
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Failed to cleanup {temp_file}: {e}")
```

Su dung pattern `try/finally` dam bao file tam luon duoc don dep du co loi xay ra. Day la pattern quan ly tai nguyen quan trong trong code production.

#### Diem thuc hanh
- Thay doi tuy chon `scale` va `pad` cua FFmpeg filter de thu cac do phan giai khac nhau.
- Dieu chinh gia tri `loop` de thay doi do dai canh.
- Kiem tra phan xu ly loi va quan sat loi gi xay ra khi moi truong khong cai FFmpeg.
- Nghien cuu cach su dung pipe thay vi file tam.

---

### 11.6 Callbacks - Dieu khien hanh vi agent qua callback

#### Chu de va Muc tieu
Hieu he thong callback cua ADK va hoc cach su dung `before_model_callback` de kiem tra va loc dau vao nguoi dung truoc khi goi LLM.

#### Giai thich khai niem cot loi

**1) Callback la gi?**

Callback la ham cho phep can thiep vao cac thoi diem cu the trong hanh vi cua agent. ADK cung cap nhieu diem callback:
- `before_model_callback`: Thuc thi **truoc** khi goi LLM
- `after_model_callback`: Thuc thi **sau** khi goi LLM
- `before_tool_callback`: Thuc thi **truoc** khi goi cong cu
- `after_tool_callback`: Thuc thi **sau** khi goi cong cu

**2) Hien thuc before_model_callback**

```python
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    # Kiem tra tin nhan cuoi trong lich su hoi thoai
    history = llm_request.contents
    last_message = history[-1]

    # Chi kiem tra neu la tin nhan nguoi dung
    if last_message.role == "user":
        text = last_message.parts[0].text

        # Kiem tra tu khoa bi cam
        if "hummus" in text:
            # Tra ve phan hoi tu choi ngay lap tuc ma khong goi LLM
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

    # Tra ve None thi goi LLM binh thuong
    return None
```

**Nguyen ly hoat dong callback:**

```
Dau vao nguoi dung
    |
    v
Thuc thi before_model_callback
    |
    +-- Tra ve LlmResponse --> Bo qua goi LLM, tra phan hoi ngay
    |
    +-- Tra ve None ---------> Tiep tuc goi LLM binh thuong
                                |
                                v
                            LLM tao phan hoi
```

- **Tra ve `LlmResponse`**: Goi LLM bi chan hoan toan, phan hoi tra ve duoc su dung truc tiep. Khong phat sinh chi phi API.
- **Tra ve `None`**: Goi LLM dien ra binh thuong.

**3) Dang ky callback**

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
    before_model_callback=before_model_callback,  # Dang ky callback!
)
```

Khi truyen ham vao tham so `before_model_callback`, moi lan agent nay goi LLM, callback se duoc thuc thi truoc.

**4) Cac truong hop su dung callback**

Vi du nay chi minh hoa loc tu khoa don gian, nhung trong du an thuc te co the ung dung da dang:

| Truong hop su dung | Mo ta |
|---------------------|-------|
| **Loc noi dung** | Chan cac yeu cau khong phu hop truoc khi gui den LLM |
| **Kiem soat chi phi** | Gioi han goi LLM khi vuot qua so token nhat dinh |
| **Logging/Monitoring** | Ghi log moi lan goi LLM |
| **Caching** | Tra ve phan hoi da cache cho cac yeu cau giong nhau |
| **Sua doi prompt** | Thay doi dong prompt gui den LLM |
| **Kiem tra xac thuc/quyen** | Gioi han chuc nang theo quyen nguoi dung |

**5) CallbackContext va LlmRequest**

```python
def before_model_callback(
    callback_context: CallbackContext,  # Truy cap ten agent, trang thai...
    llm_request: LlmRequest,           # Noi dung yeu cau gui den LLM
):
```

- `callback_context.agent_name`: Ten cua agent dang thuc thi
- `llm_request.contents`: Lich su hoi thoai (gom role, parts)
- Sua doi `llm_request` co the thay doi chinh yeu cau gui den LLM

#### Diem thuc hanh
- Them cac bo loc tu khoa khac ngoai "hummus".
- Hien thuc `after_model_callback` de hau xu ly phan hoi LLM.
- In `callback_context.agent_name` de quan sat callback duoc goi tu agent nao.
- Them `before_tool_callback` de hien thuc logging khi goi cong cu.
- Sua doi `llm_request.contents` trong callback de them dong system message.

---

## 3. Tong hop cot loi chuong

### Tong hop cac loai agent ADK

| Loai agent | Lop | Dac diem | Su dung trong du an nay |
|------------|-----|---------|-------------------------|
| **Agent** | `google.adk.agents.Agent` | Suy luan dua tren LLM, co the su dung cong cu | ShortsProducer, ContentPlanner, PromptBuilder, ImageBuilder, VoiceGenerator, VideoAssembler |
| **ParallelAgent** | `google.adk.agents.ParallelAgent` | Thuc thi agent con song song | AssetGenerator (tao hinh anh + giong doc dong thoi) |
| **SequentialAgent** | `google.adk.agents.SequentialAgent` | Thuc thi agent con tuan tu | ImageGenerator (toi uu prompt -> tao hinh anh) |

### Tong hop khai niem ADK cot loi

| Khai niem | Mo ta | Code lien quan |
|-----------|-------|---------------|
| **AgentTool** | Boc agent thanh cong cu de agent khac goi | `AgentTool(agent=content_planner_agent)` |
| **output_schema** | Buoc cau truc dau ra agent bang Pydantic model | `output_schema=ContentPlanOutput` |
| **output_key** | Khoa luu dau ra agent vao trang thai phien | `output_key="content_planner_output"` |
| **ToolContext** | Truy cap trang thai/artifact tu ham cong cu | `tool_context.state.get(...)` |
| **Artifact** | Luu tru/quan ly du lieu nhi phan (hinh anh, audio...) | `tool_context.save_artifact(...)` |
| **Callback** | Ham can thiep vao thoi diem hoat dong cua agent | `before_model_callback=...` |
| **Bien template trong prompt** | Tu dong inject gia tri state qua `{ten_bien}` | `{content_planner_output}` |

### Tong hop pattern thiet ke

1. **Pattern Orchestrator**: Root agent dieu phoi cac agent con theo thu tu
2. **Pattern Pipeline**: SequentialAgent ket noi tuan tu cac buoc xu ly du lieu
3. **Pattern xu ly song song**: ParallelAgent thuc thi dong thoi cac tac vu doc lap
4. **Pattern Artifact caching**: Kiem tra artifact ton tai truoc khi tao de tranh trung lap
5. **Pattern quan ly file tam**: try/finally dam bao don dep tai nguyen
6. **Pattern Callback guard**: before_model_callback de xac thuc va loc dau vao

---

## 4. Bai tap thuc hanh

### Bai tap 1: Co ban - Them loai noi dung moi
Sua doi prompt cua ContentPlannerAgent de ho tro phong cach "tom tat tin tuc" cho YouTube Shorts. Can bao gom giong dieu khan cap dac trung cua tin tuc, narration tap trung vao du kien, va mo ta hinh anh theo phong cach do hoa tin tuc.

### Bai tap 2: Trung cap - Them agent tao phu de (Subtitle)
Tao `SubtitleGeneratorAgent` moi de tao file phu de SRT dua tren van ban narration cua tung canh, va them vao ParallelAgent cua AssetGeneratorAgent. Sua lenh FFmpeg cua VideoAssemblerAgent de dong bo phu de (burn-in) vao video.

### Bai tap 3: Trung cap - Hien thuc nhieu callback
Hien thuc cac callback sau:
- `before_model_callback`: Loc dau vao nguoi dung bang danh sach tu khoa bi cam
- `after_model_callback`: Dem so token cua phan hoi LLM va ghi log
- `before_tool_callback`: Ghi lai thoi diem bat dau goi cong cu
- `after_tool_callback`: Tinh thoi gian thuc thi cong cu va xuat log hieu suat

### Bai tap 4: Nang cao - Them agent nhac nen (BGM)
Hien thuc `BGMAgent` de tao hoac chon nhac nen va them vao AssetGeneratorAgent. Sua lenh FFmpeg cua VideoAssemblerAgent de mix narration va BGM (bao gom dieu chinh am luong) vao video cuoi cung.

### Bai tap 5: Nang cao - Co che phuc hoi loi
Hien thuc logic tu dong thu lai khi xay ra loi API trong qua trinh tao hinh anh hoac giong doc. Bao gom toi da 3 lan thu lai, exponential backoff, va bao toan cac artifact da thanh cong khi chi thanh cong mot phan.

---

## Phu luc: Cach chay du an

### Thiet lap moi truong

```bash
# Di chuyen den thu muc du an
cd youtube-shorts-maker

# Cai dat phu thuoc (su dung uv)
uv sync

# Thiet lap bien moi truong
export OPENAI_API_KEY="your-openai-api-key"

# Kiem tra cai dat FFmpeg
ffmpeg -version
```

### Chay ADK

```bash
# Chay bang server phat trien ADK
adk web youtube_shorts_maker

# Hoac chay bang CLI
adk run youtube_shorts_maker
```

### Cau truc thu muc cuoi cung

```
youtube-shorts-maker/
    youtube_shorts_maker/
        __init__.py
        agent.py                          # Root agent + callback
        prompt.py                         # Prompt orchestrator
        sub_agents/
            content_planner/
                agent.py                  # ContentPlannerAgent
                prompt.py                 # Prompt content planner
            asset_generator/
                agent.py                  # AssetGeneratorAgent (ParallelAgent)
                prompt.py                 # Mo ta asset generator
                image_generator/
                    agent.py              # ImageGeneratorAgent (SequentialAgent)
                    prompt_builder/
                        agent.py          # PromptBuilderAgent
                        prompt.py         # Prompt cua prompt builder
                    image_builder/
                        agent.py          # ImageBuilderAgent
                        prompt.py         # Prompt cua image builder
                        tools.py          # Cong cu generate_images
                voice_generator/
                    agent.py              # VoiceGeneratorAgent
                    prompt.py             # Prompt cua voice generator
                    tools.py              # Cong cu generate_narrations
            video_assembler/
                agent.py                  # VideoAssemblerAgent
                prompt.py                 # Prompt cua video assembler
                tools.py                  # Cong cu assemble_video (FFmpeg)
```
