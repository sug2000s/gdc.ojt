# Chapter 11: ADK 멀티미디어 프로젝트 - YouTube Shorts 자동 생성기

---

## 1. 챕터 개요

이 챕터에서는 Google ADK(Agent Development Kit)를 활용하여 **YouTube Shorts 동영상을 자동으로 생성하는 멀티 에이전트 시스템**을 구축합니다. 사용자가 주제를 입력하면 콘텐츠 기획, 이미지 생성, 음성 나레이션 생성, 비디오 조립까지 전 과정을 AI 에이전트들이 자동으로 처리합니다.

### 프로젝트 이름
**youtube-shorts-maker** - 세로 형식(9:16) YouTube Shorts 동영상 자동 제작기

### 학습 목표
- Google ADK의 다양한 에이전트 유형(Agent, ParallelAgent, SequentialAgent) 이해 및 활용
- 멀티 에이전트 오케스트레이션 패턴 설계
- Pydantic 스키마를 활용한 에이전트 출력 구조화
- OpenAI API(GPT-Image-1, TTS)를 도구(Tool)로 통합하는 방법
- ADK의 Artifact 시스템을 활용한 멀티미디어 파일 관리
- FFmpeg를 이용한 비디오 조립
- 콜백(Callback)을 통한 에이전트 동작 제어

### 전체 아키텍처

```
ShortsProducerAgent (루트 오케스트레이터)
    |
    +-- ContentPlannerAgent (콘텐츠 기획)
    |
    +-- AssetGeneratorAgent (ParallelAgent - 병렬 에셋 생성)
    |       |
    |       +-- ImageGeneratorAgent (SequentialAgent - 순차 이미지 생성)
    |       |       |
    |       |       +-- PromptBuilderAgent (프롬프트 최적화)
    |       |       +-- ImageBuilderAgent (이미지 생성)
    |       |
    |       +-- VoiceGeneratorAgent (음성 나레이션 생성)
    |
    +-- VideoAssemblerAgent (최종 비디오 조립)
```

### 워크플로우 흐름
1. **Phase 1**: 사용자 입력 수집 및 요구사항 확인
2. **Phase 2**: ContentPlannerAgent가 구조화된 스크립트 생성
3. **Phase 3**: AssetGeneratorAgent가 이미지와 음성을 **병렬로** 생성
4. **Phase 4**: VideoAssemblerAgent가 FFmpeg로 최종 MP4 비디오 조립
5. **Phase 5**: 최종 결과물 전달

---

## 2. 섹션별 상세 설명

---

### 11.0 Introduction - 프로젝트 초기 설정

#### 주제 및 목표
프로젝트의 기본 구조를 설정하고, 필요한 의존성 패키지를 정의합니다. ADK 프로젝트의 표준 디렉터리 구조를 이해합니다.

#### 핵심 개념 설명

**ADK 프로젝트 구조**: Google ADK는 특정 디렉터리 구조를 기대합니다. 패키지 이름과 동일한 디렉터리 안에 `agent.py`와 `__init__.py`를 배치해야 합니다.

**프로젝트 의존성**:

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

각 의존성의 역할:
- **google-adk**: Google Agent Development Kit - 에이전트 프레임워크의 핵심
- **google-genai**: Google의 Generative AI 라이브러리 (타입 정의 및 Artifact 관리에 사용)
- **litellm**: 다양한 LLM 프로바이더를 통합하는 라이브러리 (OpenAI 모델을 ADK에서 사용 가능하게 함)
- **openai**: OpenAI API 클라이언트 (이미지 생성, TTS에 직접 사용)

**`__init__.py` 파일**:

```python
from . import agent
```

이 한 줄은 ADK가 패키지를 로드할 때 `agent.py` 모듈을 자동으로 임포트하도록 합니다. ADK는 이 모듈에서 `root_agent`를 찾아 실행합니다.

**초기 디렉터리 구조**:

```
youtube-shorts-maker/
    .python-version          # Python 3.13 지정
    pyproject.toml           # 프로젝트 설정 및 의존성
    README.md
    uv.lock                  # uv 패키지 매니저 잠금 파일
    youtube_shorts_maker/
        __init__.py          # 패키지 초기화
        agent.py             # 루트 에이전트 (빈 파일)
        sub_agents/
            content_planner/
                agent.py     # 콘텐츠 플래너 에이전트 (빈 파일)
                prompt.py    # 프롬프트 정의 (빈 파일)
```

#### 실습 포인트
- `uv`를 사용하여 프로젝트를 초기화하는 방법을 익히세요.
- ADK가 요구하는 디렉터리 구조(`__init__.py`에서 `agent` 모듈 임포트, `root_agent` 변수)를 숙지하세요.
- `sub_agents/` 디렉터리로 에이전트를 모듈화하는 패턴을 기억하세요.

---

### 11.1 Content Planner Agent - 콘텐츠 기획 에이전트

#### 주제 및 목표
YouTube Shorts의 전체 콘텐츠 계획을 구조화된 JSON으로 출력하는 에이전트를 구현합니다. Pydantic 모델을 사용한 `output_schema`와 `output_key`의 활용법을 학습합니다.

#### 핵심 개념 설명

**1) 루트 에이전트 (ShortsProducerAgent) - 오케스트레이터 패턴**

오케스트레이터란 여러 하위 에이전트를 조율하여 복잡한 작업을 수행하는 상위 에이전트입니다. `AgentTool`을 사용하여 하위 에이전트를 도구처럼 호출합니다.

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

핵심 포인트:
- **`LiteLlm(model="openai/gpt-4o")`**: ADK는 기본적으로 Google 모델을 사용하지만, `LiteLlm` 래퍼를 통해 OpenAI의 GPT-4o 등 다른 모델도 사용할 수 있습니다.
- **`AgentTool(agent=...)`**: 하위 에이전트를 도구(Tool)로 감싸서 루트 에이전트가 필요할 때 호출할 수 있게 합니다. 이는 ADK의 핵심 패턴 중 하나입니다.
- **`root_agent`**: ADK가 실행 시 찾는 진입점 변수입니다.

**2) 오케스트레이터 프롬프트 설계**

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

오케스트레이터 프롬프트의 설계 원칙:
- **단계별 워크플로우** 명시: 에이전트가 어떤 순서로 작업해야 하는지 명확히 지시
- **각 단계에서 사용할 하위 에이전트** 지정: ContentPlanner -> AssetGenerator -> VideoAssembler
- **에러 처리 및 사용자 소통** 가이드라인 포함

**3) ContentPlannerAgent - 구조화된 출력(output_schema)**

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

핵심 포인트:
- **`output_schema`**: Pydantic 모델을 지정하면, 에이전트의 출력이 반드시 해당 스키마에 맞게 구조화됩니다. LLM이 자유 형식 텍스트 대신 정해진 JSON 구조로 응답하도록 강제합니다.
- **`output_key`**: 에이전트의 출력을 세션 상태(state)에 저장할 키 이름입니다. `"content_planner_output"`으로 지정하면, 이후 다른 에이전트에서 `tool_context.state.get("content_planner_output")`으로 접근할 수 있습니다.
- **`Field(description=...)`**: 각 필드의 설명은 LLM이 올바른 값을 생성하는 데 가이드 역할을 합니다.

**4) 콘텐츠 플래너 프롬프트의 핵심 전략**

프롬프트에서 주목할 부분:
- **최대 20초 제한**: YouTube Shorts의 특성에 맞는 제약 조건
- **장면 수 유연성**: 3~6개 장면으로 에이전트가 최적 구성 결정
- **타이밍 전략**: 빠른 인트로(2-3초), 메인 콘텐츠(3-5초), 강한 마무리(2-4초)
- **검증 요구**: 출력 전 총 시간이 20초를 초과하지 않는지 확인하도록 지시
- **구체적 예시 제공**: "Perfect Scrambled Eggs" 예시로 기대하는 출력 형태를 명확히 보여줌

#### 코드 분석 - 에이전트 간 데이터 흐름

```
사용자 입력 ("요리 주제로 Shorts 만들어줘")
    |
    v
ShortsProducerAgent (오케스트레이터)
    |  AgentTool로 호출
    v
ContentPlannerAgent
    |  output_schema: ContentPlanOutput
    |  output_key: "content_planner_output"
    v
세션 상태에 저장: state["content_planner_output"] = {
    "topic": "...",
    "total_duration": 18,
    "scenes": [
        {"id": 1, "narration": "...", "visual_description": "...", ...},
        {"id": 2, ...},
        ...
    ]
}
```

#### 실습 포인트
- Pydantic 모델의 `Field(description=...)`이 LLM 출력에 미치는 영향을 실험해 보세요.
- `output_key`를 변경하고 후속 에이전트에서 접근하는 방법을 테스트해 보세요.
- 프롬프트의 예시(Example)를 변경하면 출력 품질이 어떻게 달라지는지 확인해 보세요.

---

### 11.2 Prompt Builder Agent - 이미지 프롬프트 최적화 에이전트

#### 주제 및 목표
콘텐츠 플랜의 시각적 설명을 GPT-Image-1에 최적화된 프롬프트로 변환하는 에이전트를 구현합니다. ParallelAgent와 SequentialAgent의 차이를 이해하고, 에이전트 계층 구조를 설계합니다.

#### 핵심 개념 설명

**1) 세 가지 에이전트 유형의 활용**

이 섹션에서는 ADK의 세 가지 주요 에이전트 유형이 모두 등장합니다:

```python
# 1. ParallelAgent - 하위 에이전트를 동시에 실행
from google.adk.agents import ParallelAgent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,   # 이미지 생성 파이프라인
        # voice_generator_agent, # (11.4에서 추가 예정)
    ],
)
```

**ParallelAgent**는 `sub_agents` 목록에 있는 에이전트들을 **동시에(병렬로)** 실행합니다. 이미지 생성과 음성 생성은 서로 의존하지 않으므로 병렬 처리가 가능합니다.

```python
# 2. SequentialAgent - 하위 에이전트를 순서대로 실행
from google.adk.agents import SequentialAgent

image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,   # 먼저: 프롬프트 최적화
        # image_builder_agent,  # 다음: 이미지 생성 (11.3에서 추가 예정)
    ],
)
```

**SequentialAgent**는 `sub_agents`를 **순서대로** 실행합니다. 프롬프트를 먼저 최적화한 후에야 이미지를 생성할 수 있으므로 순차 처리가 필수입니다.

```python
# 3. Agent - LLM 기반 일반 에이전트 (도구 사용, 추론 가능)
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

**Agent**는 LLM을 사용하여 추론하고, 도구를 호출하며, 구조화된 출력을 생성하는 가장 기본적인 에이전트입니다.

**2) 에이전트 계층 구조 설계 패턴**

```
AssetGeneratorAgent (ParallelAgent)
    |
    +-- ImageGeneratorAgent (SequentialAgent)
    |       |
    |       +-- PromptBuilderAgent (Agent) -- 프롬프트 최적화
    |       +-- ImageBuilderAgent (Agent)  -- 이미지 생성
    |
    +-- VoiceGeneratorAgent (Agent)         -- 음성 생성
```

이 설계의 핵심 원리:
- **이미지 생성**은 프롬프트 최적화 -> 이미지 생성이 **순차적**이어야 함 (SequentialAgent)
- **이미지 생성**과 **음성 생성**은 **병렬**로 진행 가능 (ParallelAgent)
- 각 개별 작업은 LLM 추론이 필요 (Agent)

**3) 프롬프트 빌더의 출력 스키마**

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

**4) 상태(State)를 통한 에이전트 간 데이터 전달**

프롬프트에서 `{content_planner_output}`을 사용하여 이전 에이전트의 출력을 참조합니다:

```python
PROMPT_BUILDER_PROMPT = """
...
## Your Task:
Take the structured content plan: {content_planner_output} and create
optimized vertical image generation prompts for each scene...
"""
```

ADK의 `instruction`에서 `{변수명}` 구문을 사용하면 세션 상태에서 해당 키의 값을 자동으로 주입합니다. 이것이 `output_key`와 연결되는 방식입니다:
1. ContentPlannerAgent가 `output_key="content_planner_output"`으로 결과 저장
2. PromptBuilderAgent의 `instruction`에서 `{content_planner_output}`으로 해당 데이터 참조

**5) 프롬프트 최적화 전략**

프롬프트 빌더가 수행하는 향상(Enhancement) 작업:
- **기술 사양 추가**: 9:16 세로 비율, 1080x1920 해상도
- **시각적 세부사항**: 조명, 카메라 앵글, 구도 등
- **텍스트 오버레이 지시**: 위치, 패딩, 가독성
- **스타일 일관성**: 모든 장면에 동일한 시각적 스타일 유지

예시 변환:
```
원본: "Stovetop dial on low"
최적화: "Close-up shot of modern stovetop control dial set to low heat
setting, 9:16 portrait aspect ratio, 1080x1920 resolution, vertical
composition, warm kitchen lighting, shallow depth of field, photorealistic,
sharp focus, with bold white text 'Secret #1: Low Heat' positioned at
top center of image with generous padding from borders..."
```

#### 실습 포인트
- `ParallelAgent`와 `SequentialAgent`를 바꿔보면서 실행 순서가 어떻게 달라지는지 관찰하세요.
- 프롬프트의 `{content_planner_output}` 템플릿 변수가 실제로 어떤 값으로 치환되는지 로그를 통해 확인하세요.
- 프롬프트 최적화 지침을 수정하여 생성되는 이미지의 스타일을 변경해 보세요.

---

### 11.3 Image Builder Agent - 이미지 생성 에이전트

#### 주제 및 목표
최적화된 프롬프트를 사용하여 OpenAI GPT-Image-1 API로 실제 이미지를 생성하고, ADK의 Artifact 시스템으로 저장하는 에이전트를 구현합니다.

#### 핵심 개념 설명

**1) 에이전트 정의 - 도구를 가진 Agent**

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

이 에이전트는 `generate_images`라는 Python 함수를 도구로 사용합니다. ADK에서는 일반 Python 함수를 `tools` 리스트에 전달하면 자동으로 도구로 등록됩니다.

**2) SequentialAgent에 ImageBuilder 추가**

```python
image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,    # 1단계: 프롬프트 최적화
        image_builder_agent,     # 2단계: 이미지 생성
    ],
)
```

이제 이미지 생성 파이프라인이 완성되었습니다. PromptBuilder가 먼저 실행되어 최적화된 프롬프트를 상태에 저장하면, ImageBuilder가 그 프롬프트를 읽어 이미지를 생성합니다.

**3) generate_images 도구 - ADK Artifact 시스템의 핵심**

```python
import base64
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext

client = OpenAI()


async def generate_images(tool_context: ToolContext):

    # 1. 세션 상태에서 이전 에이전트의 출력 가져오기
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")

    # 2. 이미 생성된 아티팩트 목록 확인 (중복 생성 방지)
    existing_artifacts = await tool_context.list_artifacts()

    generated_images = []

    for prompt in optimized_prompts:
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        filename = f"scene_{scene_id}_image.jpeg"

        # 3. 이미 존재하면 건너뛰기 (캐싱)
        if filename in existing_artifacts:
            generated_images.append({
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100],
                "filename": filename,
            })
            continue

        # 4. OpenAI GPT-Image-1 API로 이미지 생성
        image = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",    # 세로 형식 (2:3 비율)
        )

        # 5. Base64 디코딩
        image_bytes = base64.b64decode(image.data[0].b64_json)

        # 6. ADK Artifact로 저장
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

**핵심 개념 상세 설명:**

**(a) `ToolContext` - 도구의 핵심 컨텍스트**

`tool_context: ToolContext`는 ADK가 도구 함수에 자동으로 주입하는 특별한 매개변수입니다. 이를 통해:
- `tool_context.state`: 세션의 공유 상태에 접근 (에이전트 간 데이터 공유)
- `tool_context.list_artifacts()`: 저장된 아티팩트 목록 조회
- `tool_context.save_artifact()`: 새 아티팩트 저장
- `tool_context.load_artifact()`: 아티팩트 로드

ADK는 함수 시그니처에 `tool_context: ToolContext`가 있으면 자동으로 주입합니다. 사용자가 직접 전달할 필요가 없습니다.

**(b) Artifact 시스템**

Artifact는 ADK에서 바이너리 데이터(이미지, 오디오, 비디오 등)를 관리하는 메커니즘입니다. `google.genai.types.Part`와 `types.Blob`을 사용하여 MIME 타입과 함께 데이터를 저장합니다.

```python
artifact = types.Part(
    inline_data=types.Blob(
        mime_type="image/jpeg",
        data=image_bytes,      # raw bytes
    )
)
await tool_context.save_artifact(filename=filename, artifact=artifact)
```

이는 파일 시스템 대신 ADK의 세션 관리 시스템에 파일을 저장하여, 세션 간 데이터가 유지되고 다른 에이전트에서 접근 가능하게 합니다.

**(c) 중복 생성 방지 패턴**

```python
existing_artifacts = await tool_context.list_artifacts()
if filename in existing_artifacts:
    # 이미 존재하면 건너뛰기
    continue
```

이미 생성된 이미지가 있다면 다시 생성하지 않습니다. API 호출 비용과 시간을 절약하는 중요한 패턴입니다.

**(d) OpenAI 이미지 생성 API 파라미터**

```python
image = client.images.generate(
    model="gpt-image-1",       # OpenAI의 이미지 생성 모델
    prompt=enhanced_prompt,     # 최적화된 프롬프트
    n=1,                        # 1장 생성
    quality="low",              # 품질 (low/medium/high)
    moderation="low",           # 콘텐츠 필터링 수준
    output_format="jpeg",       # 출력 형식
    background="opaque",        # 배경 (opaque: 불투명)
    size="1024x1536",           # 세로 비율 (YouTube Shorts)
)
```

#### 실습 포인트
- `quality` 파라미터를 `"high"`로 변경하여 이미지 품질 차이를 비교해 보세요.
- `size`를 `"1024x1024"`로 변경하면 어떤 결과가 나오는지 확인해 보세요.
- `tool_context.list_artifacts()`를 통한 캐싱 메커니즘이 재실행 시 어떻게 작동하는지 테스트하세요.

---

### 11.4 Audio Narration Agent - 음성 나레이션 에이전트

#### 주제 및 목표
OpenAI TTS(Text-to-Speech) API를 사용하여 각 장면의 나레이션 오디오를 생성하는 에이전트를 구현합니다. ParallelAgent에 음성 생성기를 추가하여 이미지와 음성을 동시에 생성하는 구조를 완성합니다.

#### 핵심 개념 설명

**1) ParallelAgent에 VoiceGenerator 추가**

```python
from google.adk.agents import ParallelAgent
from .prompt import ASSET_GENERATOR_DESCRIPTION
from .image_generator.agent import image_generator_agent
from .voice_generator.agent import voice_generator_agent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,     # 이미지 생성 파이프라인
        voice_generator_agent,     # 음성 생성 에이전트 (추가!)
    ],
)
```

이제 `AssetGeneratorAgent`가 호출되면 이미지 생성(`ImageGeneratorAgent`)과 음성 생성(`VoiceGeneratorAgent`)이 **동시에** 실행됩니다. 두 작업은 서로 독립적이므로 병렬 처리로 전체 실행 시간을 크게 단축할 수 있습니다.

**2) VoiceGeneratorAgent 정의**

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

**3) 음성 선택 전략 - 프롬프트 설계**

이 에이전트의 프롬프트는 LLM이 콘텐츠의 분위기에 따라 적절한 음성을 **스스로 선택**하도록 설계되었습니다:

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

핵심 설계 원칙:
- `{content_planner_output}`을 통해 이전 단계의 전체 콘텐츠 계획을 참조
- 음성 선택 가이드라인을 제공하되, **최종 결정은 LLM에게 위임**
- 각 장면별 `input`(읽을 텍스트)과 `instructions`(톤, 속도 등)를 분리하여 세밀한 제어 가능

**4) generate_narrations 도구 - TTS API 활용**

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

        # 캐싱: 이미 존재하면 건너뛰기
        if filename in existing_artifacts:
            generated_narrations.append({
                "scene_id": scene_id,
                "filename": filename,
                "input": text_input,
                "instructions": instructions[:50],
            })
            continue

        # OpenAI TTS API 호출 (스트리밍 응답)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text_input,
            instructions=instructions,
        ) as response:
            audio_data = response.read()

        # ADK Artifact로 저장
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

**핵심 개념 상세 설명:**

**(a) 도구 함수의 매개변수 설계**

```python
async def generate_narrations(
    tool_context: ToolContext,    # ADK가 자동 주입
    voice: str,                   # LLM이 선택하여 전달
    voice_instructions: List[Dict[str, Any]]  # LLM이 구성하여 전달
):
```

`tool_context`는 ADK가 자동으로 주입하고, `voice`와 `voice_instructions`는 LLM이 프롬프트 지시에 따라 적절한 값을 생성하여 전달합니다. 이것이 ADK 도구의 강력한 점입니다 -- **LLM이 도구 호출 시 인자를 지능적으로 결정**합니다.

**(b) 스트리밍 응답으로 TTS 데이터 수신**

```python
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice=voice,
    input=text_input,
    instructions=instructions,
) as response:
    audio_data = response.read()
```

`with_streaming_response`를 사용하면 오디오 데이터를 스트리밍 방식으로 수신할 수 있어 메모리 효율이 좋습니다. `gpt-4o-mini-tts` 모델은 `instructions` 매개변수를 통해 말하는 속도, 톤, 감정 등을 세밀하게 제어할 수 있습니다.

**(c) 파일 명명 규칙**

```python
filename = f"scene_{scene_id}_narration.mp3"   # 오디오
filename = f"scene_{scene_id}_image.jpeg"       # 이미지 (11.3에서)
```

일관된 명명 규칙을 사용하여 VideoAssembler가 나중에 장면 번호로 파일을 쉽게 찾고 정렬할 수 있게 합니다.

#### 실습 포인트
- `voice` 매개변수를 다양하게 변경하여 동일 텍스트의 음성 차이를 비교해 보세요.
- `instructions`에 "Speak very slowly and dramatically"와 같은 다른 지시를 넣어보세요.
- ParallelAgent 대신 SequentialAgent로 변경하여 실행 시간 차이를 측정해 보세요.

---

### 11.5 Video Assembly - 비디오 조립 에이전트

#### 주제 및 목표
생성된 모든 이미지와 오디오 아티팩트를 FFmpeg를 사용하여 최종 MP4 비디오로 조립하는 에이전트를 구현합니다. Artifact 로드, 임시 파일 관리, FFmpeg 필터 그래프 구성을 학습합니다.

#### 핵심 개념 설명

**1) VideoAssemblerAgent 정의**

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

**2) 루트 에이전트에 전체 파이프라인 연결**

```python
shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),     # 1단계
        AgentTool(agent=asset_generator_agent),      # 2단계
        AgentTool(agent=video_assembler_agent),      # 3단계 (추가!)
    ],
)
```

이제 전체 파이프라인이 완성되었습니다.

**3) assemble_video 도구 - FFmpeg를 활용한 비디오 조립**

이 도구는 이 프로젝트에서 가장 복잡한 부분입니다. 단계별로 분석하겠습니다.

**(a) 아티팩트 로드 및 임시 파일 생성**

```python
async def assemble_video(tool_context: ToolContext) -> str:
    temp_files = []  # 정리할 임시 파일 추적

    try:
        # 콘텐츠 플랜에서 장면 정보 가져오기
        content_planner_output = tool_context.state.get(
            "content_planner_output", {}
        )
        scenes = content_planner_output.get("scenes", [])

        # 저장된 아티팩트 목록 조회
        existing_artifacts = await tool_context.list_artifacts()

        # 이미지/오디오 파일 분류 및 정렬
        image_files = []
        audio_files = []
        for artifact_name in existing_artifacts:
            if artifact_name.endswith("_image.jpeg"):
                image_files.append(artifact_name)
            elif artifact_name.endswith("_narration.mp3"):
                audio_files.append(artifact_name)

        # 장면 번호로 정렬
        def extract_scene_number(filename):
            match = re.search(r"scene_(\d+)_", filename)
            return int(match.group(1)) if match else 0

        image_files.sort(key=extract_scene_number)
        audio_files.sort(key=extract_scene_number)
```

ADK의 Artifact는 메모리/세션에 저장되어 있으므로, FFmpeg에서 사용하려면 **임시 파일**로 추출해야 합니다:

```python
        temp_image_paths = []
        temp_audio_paths = []

        for i, (image_name, audio_name) in enumerate(
            zip(image_files, audio_files)
        ):
            # 이미지 아티팩트를 임시 파일로 저장
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
                temp_files.append(temp_image.name)  # 정리 목록에 추가

            # 오디오 아티팩트도 동일하게 처리
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

**(b) FFmpeg 필터 그래프 구성**

FFmpeg의 `filter_complex`를 사용하여 여러 이미지와 오디오를 하나의 비디오로 합칩니다:

```python
        input_args = []
        filter_parts = []

        for i, (temp_image, temp_audio) in enumerate(
            zip(temp_image_paths, temp_audio_paths)
        ):
            # 각 장면의 이미지와 오디오를 입력으로 추가
            input_args.extend(["-i", temp_image, "-i", temp_audio])

            scene_duration = scenes[i].get("duration", 4)

            # 비디오 스트림 생성: 정적 이미지를 지정 시간만큼 반복
            total_frames = int(30 * scene_duration)  # 30fps * 초
            filter_parts.append(
                f"[{i*2}:v]scale=1080:1920:"
                f"force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps=30,"
                f"loop={total_frames-1}:size=1:start=0[v{i}]"
            )

            # 오디오 스트림 (변환 없이 통과)
            filter_parts.append(f"[{i*2+1}:a]anull[a{i}]")

        # 모든 스트림을 연결(concat)
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

**FFmpeg 필터 그래프 해설:**

각 장면에 대해:
1. `scale=1080:1920:force_original_aspect_ratio=decrease` - 1080x1920(세로)으로 스케일링, 비율 유지
2. `pad=1080:1920:(ow-iw)/2:(oh-ih)/2` - 부족한 부분은 패딩으로 채움 (레터박스)
3. `setsar=1` - 픽셀 비율을 1:1로 설정
4. `fps=30` - 30fps로 설정
5. `loop={total_frames-1}:size=1:start=0` - 정적 이미지를 지정 프레임 수만큼 반복

`concat` 필터로 모든 비디오 스트림과 오디오 스트림을 순서대로 연결합니다.

**(c) FFmpeg 실행 및 최종 비디오 저장**

```python
        ffmpeg_cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",    # H.264 비디오 코덱
                "-c:a", "aac",        # AAC 오디오 코덱
                "-pix_fmt", "yuv420p", # 호환성 높은 픽셀 형식
                "-r", "30",           # 30fps
                output_path,
            ]
        )

        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        # 최종 비디오를 아티팩트로 저장
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

**(d) 임시 파일 정리 - finally 패턴**

```python
    finally:
        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Failed to cleanup {temp_file}: {e}")
```

`try/finally` 패턴을 사용하여 에러가 발생하더라도 임시 파일이 반드시 정리되도록 합니다. 이는 프로덕션 코드에서 중요한 리소스 관리 패턴입니다.

#### 실습 포인트
- FFmpeg 필터의 `scale`과 `pad` 옵션을 변경하여 다양한 해상도를 시험해 보세요.
- `loop` 값을 조절하여 장면 길이를 변경해 보세요.
- 에러 처리 부분을 확인하고, FFmpeg가 설치되지 않은 환경에서 어떤 에러가 발생하는지 관찰해 보세요.
- 임시 파일 대신 파이프(pipe)를 사용하는 방법을 연구해 보세요.

---

### 11.6 Callbacks - 콜백을 통한 에이전트 동작 제어

#### 주제 및 목표
ADK의 콜백 시스템을 이해하고, `before_model_callback`을 사용하여 LLM 호출 전에 사용자 입력을 검사하고 필터링하는 방법을 학습합니다.

#### 핵심 개념 설명

**1) 콜백(Callback)이란?**

콜백은 에이전트의 특정 동작 시점에 개입할 수 있는 함수입니다. ADK에서는 다양한 콜백 포인트를 제공합니다:
- `before_model_callback`: LLM 호출 **전**에 실행
- `after_model_callback`: LLM 호출 **후**에 실행
- `before_tool_callback`: 도구 호출 **전**에 실행
- `after_tool_callback`: 도구 호출 **후**에 실행

**2) before_model_callback 구현**

```python
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    # 대화 기록에서 마지막 메시지 확인
    history = llm_request.contents
    last_message = history[-1]

    # 사용자 메시지인 경우에만 검사
    if last_message.role == "user":
        text = last_message.parts[0].text

        # 금지 키워드 검사
        if "hummus" in text:
            # LLM을 호출하지 않고 즉시 거부 응답 반환
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

    # None을 반환하면 정상적으로 LLM 호출 진행
    return None
```

**콜백 동작 원리:**

```
사용자 입력
    |
    v
before_model_callback 실행
    |
    +-- LlmResponse 반환 --> LLM 호출 건너뛰고 즉시 응답
    |
    +-- None 반환 ---------> 정상적으로 LLM 호출 진행
                                |
                                v
                            LLM 응답 생성
```

- **`LlmResponse`를 반환하면**: LLM 호출이 완전히 차단되고, 반환된 응답이 그대로 사용됩니다. API 비용이 전혀 발생하지 않습니다.
- **`None`을 반환하면**: 정상적인 LLM 호출이 진행됩니다.

**3) 콜백 등록**

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
    before_model_callback=before_model_callback,  # 콜백 등록!
)
```

`before_model_callback` 매개변수에 함수를 전달하면, 이 에이전트에서 LLM을 호출할 때마다 먼저 콜백이 실행됩니다.

**4) 콜백의 활용 사례**

이 예제에서는 단순한 키워드 필터링을 보여주지만, 실제 프로젝트에서는 다양하게 활용할 수 있습니다:

| 활용 사례 | 설명 |
|-----------|------|
| **콘텐츠 필터링** | 부적절한 요청을 LLM에 전달하기 전에 차단 |
| **비용 제어** | 일정 토큰 수 초과 시 LLM 호출 제한 |
| **로깅/모니터링** | 모든 LLM 호출을 로그에 기록 |
| **캐싱** | 동일한 요청에 대해 캐시된 응답 반환 |
| **프롬프트 수정** | LLM에 전달되는 프롬프트를 동적으로 수정 |
| **인증/권한 검사** | 사용자 권한에 따라 기능 제한 |

**5) CallbackContext와 LlmRequest**

```python
def before_model_callback(
    callback_context: CallbackContext,  # 에이전트 이름, 상태 등 접근
    llm_request: LlmRequest,           # LLM에 보낼 요청 내용
):
```

- `callback_context.agent_name`: 현재 실행 중인 에이전트의 이름
- `llm_request.contents`: 대화 기록 (role, parts로 구성)
- `llm_request`를 수정하면 LLM에 전달되는 요청 자체를 변경할 수도 있습니다

#### 실습 포인트
- "hummus" 외에 다른 키워드 필터를 추가해 보세요.
- `after_model_callback`을 구현하여 LLM 응답을 후처리해 보세요.
- `callback_context.agent_name`을 출력하여 어떤 에이전트에서 콜백이 호출되는지 관찰하세요.
- `before_tool_callback`을 추가하여 도구 호출 전에 로깅을 구현해 보세요.
- 콜백에서 `llm_request.contents`를 수정하여 시스템 메시지를 동적으로 추가해 보세요.

---

## 3. 챕터 핵심 정리

### ADK 에이전트 유형 정리

| 에이전트 유형 | 클래스 | 특징 | 이 프로젝트에서의 사용 |
|-------------|--------|------|---------------------|
| **Agent** | `google.adk.agents.Agent` | LLM 기반 추론, 도구 사용 가능 | ShortsProducer, ContentPlanner, PromptBuilder, ImageBuilder, VoiceGenerator, VideoAssembler |
| **ParallelAgent** | `google.adk.agents.ParallelAgent` | 하위 에이전트를 병렬 실행 | AssetGenerator (이미지 + 음성 동시 생성) |
| **SequentialAgent** | `google.adk.agents.SequentialAgent` | 하위 에이전트를 순차 실행 | ImageGenerator (프롬프트 최적화 -> 이미지 생성) |

### 핵심 ADK 개념 정리

| 개념 | 설명 | 관련 코드 |
|------|------|----------|
| **AgentTool** | 에이전트를 도구로 감싸서 다른 에이전트가 호출 가능 | `AgentTool(agent=content_planner_agent)` |
| **output_schema** | Pydantic 모델로 에이전트 출력 구조 강제 | `output_schema=ContentPlanOutput` |
| **output_key** | 에이전트 출력을 세션 상태에 저장할 키 | `output_key="content_planner_output"` |
| **ToolContext** | 도구 함수에서 상태/아티팩트 접근 | `tool_context.state.get(...)` |
| **Artifact** | 바이너리 데이터(이미지, 오디오 등) 저장/관리 | `tool_context.save_artifact(...)` |
| **Callback** | 에이전트 동작 시점에 개입하는 함수 | `before_model_callback=...` |
| **프롬프트 템플릿 변수** | `{변수명}`으로 상태 값 자동 주입 | `{content_planner_output}` |

### 설계 패턴 정리

1. **오케스트레이터 패턴**: 루트 에이전트가 하위 에이전트들을 순서대로 조율
2. **파이프라인 패턴**: SequentialAgent로 데이터 처리 단계를 순차 연결
3. **병렬 처리 패턴**: ParallelAgent로 독립적인 작업을 동시 실행
4. **아티팩트 캐싱 패턴**: 기존 아티팩트 존재 여부 확인 후 중복 생성 방지
5. **임시 파일 관리 패턴**: try/finally로 리소스 정리 보장
6. **콜백 가드 패턴**: before_model_callback으로 입력 검증 및 필터링

---

## 4. 실습 과제

### 과제 1: 기본 - 새로운 콘텐츠 유형 추가
ContentPlannerAgent의 프롬프트를 수정하여 "뉴스 요약" 스타일의 YouTube Shorts를 지원하도록 확장하세요. 뉴스 특유의 긴급한 톤, 팩트 중심 나레이션, 뉴스 그래픽 스타일의 시각적 설명을 포함해야 합니다.

### 과제 2: 중급 - 자막(Subtitle) 에이전트 추가
각 장면의 나레이션 텍스트를 기반으로 SRT 자막 파일을 생성하는 `SubtitleGeneratorAgent`를 새로 만들고, AssetGeneratorAgent의 ParallelAgent에 추가하세요. VideoAssemblerAgent의 FFmpeg 명령을 수정하여 자막을 비디오에 합성(burn-in)하세요.

### 과제 3: 중급 - 다중 콜백 구현
다음 콜백들을 구현하세요:
- `before_model_callback`: 사용자 입력에 금지 키워드 목록(리스트)을 사용한 필터링
- `after_model_callback`: LLM 응답의 토큰 수를 세어 로그에 기록
- `before_tool_callback`: 도구 호출 시작 시간을 기록
- `after_tool_callback`: 도구 실행 시간을 계산하여 성능 로그 출력

### 과제 4: 고급 - 배경 음악 에이전트 추가
배경 음악(BGM)을 생성하거나 선택하는 `BGMAgent`를 구현하고, AssetGeneratorAgent에 추가하세요. VideoAssemblerAgent의 FFmpeg 명령을 수정하여 나레이션과 BGM을 믹싱(볼륨 조절 포함)하여 최종 비디오에 포함하세요.

### 과제 5: 고급 - 에러 복구 메커니즘
이미지 생성이나 음성 생성 중 API 에러가 발생했을 때 자동으로 재시도하는 로직을 구현하세요. 최대 3회 재시도, 지수 백오프(exponential backoff), 부분 성공 시 성공한 아티팩트는 보존하는 기능을 포함해야 합니다.

---

## 부록: 프로젝트 실행 방법

### 환경 설정

```bash
# 프로젝트 디렉터리로 이동
cd youtube-shorts-maker

# 의존성 설치 (uv 사용)
uv sync

# 환경 변수 설정
export OPENAI_API_KEY="your-openai-api-key"

# FFmpeg 설치 확인
ffmpeg -version
```

### ADK 실행

```bash
# ADK 개발 서버로 실행
adk web youtube_shorts_maker

# 또는 CLI로 실행
adk run youtube_shorts_maker
```

### 디렉터리 최종 구조

```
youtube-shorts-maker/
    youtube_shorts_maker/
        __init__.py
        agent.py                          # 루트 에이전트 + 콜백
        prompt.py                         # 오케스트레이터 프롬프트
        sub_agents/
            content_planner/
                agent.py                  # ContentPlannerAgent
                prompt.py                 # 콘텐츠 플래너 프롬프트
            asset_generator/
                agent.py                  # AssetGeneratorAgent (ParallelAgent)
                prompt.py                 # 에셋 생성기 설명
                image_generator/
                    agent.py              # ImageGeneratorAgent (SequentialAgent)
                    prompt_builder/
                        agent.py          # PromptBuilderAgent
                        prompt.py         # 프롬프트 빌더 프롬프트
                    image_builder/
                        agent.py          # ImageBuilderAgent
                        prompt.py         # 이미지 빌더 프롬프트
                        tools.py          # generate_images 도구
                voice_generator/
                    agent.py              # VoiceGeneratorAgent
                    prompt.py             # 음성 생성기 프롬프트
                    tools.py              # generate_narrations 도구
            video_assembler/
                agent.py                  # VideoAssemblerAgent
                prompt.py                 # 비디오 조립 프롬프트
                tools.py                  # assemble_video 도구 (FFmpeg)
```
