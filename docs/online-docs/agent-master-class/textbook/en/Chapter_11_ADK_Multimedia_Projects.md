# Chapter 11: ADK Multimedia Project - YouTube Shorts Auto Generator

---

## 1. Chapter Overview

In this chapter, we build a **multi-agent system that automatically generates YouTube Shorts videos** using Google ADK (Agent Development Kit). When a user provides a topic, AI agents automatically handle the entire process from content planning, image generation, voice narration generation, to video assembly.

### Project Name
**youtube-shorts-maker** - Automatic vertical format (9:16) YouTube Shorts video creator

### Learning Objectives
- Understand and utilize various ADK agent types (Agent, ParallelAgent, SequentialAgent)
- Design multi-agent orchestration patterns
- Structure agent outputs using Pydantic schemas
- Integrate OpenAI APIs (GPT-Image-1, TTS) as tools
- Manage multimedia files using ADK's Artifact system
- Assemble videos using FFmpeg
- Control agent behavior through Callbacks

### Overall Architecture

```
ShortsProducerAgent (Root Orchestrator)
    |
    +-- ContentPlannerAgent (Content Planning)
    |
    +-- AssetGeneratorAgent (ParallelAgent - Parallel Asset Generation)
    |       |
    |       +-- ImageGeneratorAgent (SequentialAgent - Sequential Image Generation)
    |       |       |
    |       |       +-- PromptBuilderAgent (Prompt Optimization)
    |       |       +-- ImageBuilderAgent (Image Generation)
    |       |
    |       +-- VoiceGeneratorAgent (Voice Narration Generation)
    |
    +-- VideoAssemblerAgent (Final Video Assembly)
```

### Workflow
1. **Phase 1**: Collect user input and confirm requirements
2. **Phase 2**: ContentPlannerAgent generates a structured script
3. **Phase 3**: AssetGeneratorAgent generates images and voice **in parallel**
4. **Phase 4**: VideoAssemblerAgent assembles the final MP4 video with FFmpeg
5. **Phase 5**: Deliver the final output

---

## 2. Section-by-Section Details

---

### 11.0 Introduction - Initial Project Setup

#### Topic and Objectives
Set up the basic project structure and define required dependency packages. Understand the standard directory structure for ADK projects.

#### Key Concepts

**ADK Project Structure**: Google ADK expects a specific directory structure. You must place `agent.py` and `__init__.py` inside a directory that matches the package name.

**Project Dependencies**:

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

Role of each dependency:
- **google-adk**: Google Agent Development Kit - the core of the agent framework
- **google-genai**: Google's Generative AI library (used for type definitions and Artifact management)
- **litellm**: A library that integrates various LLM providers (enables using OpenAI models within ADK)
- **openai**: OpenAI API client (used directly for image generation and TTS)

**`__init__.py` file**:

```python
from . import agent
```

This single line makes ADK automatically import the `agent.py` module when loading the package. ADK looks for `root_agent` in this module and executes it.

**Initial Directory Structure**:

```
youtube-shorts-maker/
    .python-version          # Specifies Python 3.13
    pyproject.toml           # Project settings and dependencies
    README.md
    uv.lock                  # uv package manager lock file
    youtube_shorts_maker/
        __init__.py          # Package initialization
        agent.py             # Root agent (empty file)
        sub_agents/
            content_planner/
                agent.py     # Content planner agent (empty file)
                prompt.py    # Prompt definitions (empty file)
```

#### Practice Points
- Learn how to initialize a project using `uv`.
- Familiarize yourself with the directory structure required by ADK (importing the `agent` module from `__init__.py`, the `root_agent` variable).
- Remember the pattern of modularizing agents using the `sub_agents/` directory.

---

### 11.1 Content Planner Agent - Content Planning Agent

#### Topic and Objectives
Implement an agent that outputs the entire content plan for YouTube Shorts as structured JSON. Learn how to use `output_schema` and `output_key` with Pydantic models.

#### Key Concepts

**1) Root Agent (ShortsProducerAgent) - Orchestrator Pattern**

An orchestrator is a higher-level agent that coordinates multiple sub-agents to perform complex tasks. It uses `AgentTool` to call sub-agents as if they were tools.

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

Key points:
- **`LiteLlm(model="openai/gpt-4o")`**: ADK uses Google models by default, but through the `LiteLlm` wrapper, you can also use other models like OpenAI's GPT-4o.
- **`AgentTool(agent=...)`**: Wraps a sub-agent as a Tool so the root agent can call it when needed. This is one of ADK's core patterns.
- **`root_agent`**: The entry point variable that ADK looks for at runtime.

**2) Orchestrator Prompt Design**

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

Design principles for the orchestrator prompt:
- **Specify step-by-step workflow**: Clearly instruct the agent on what order to perform tasks
- **Designate which sub-agent to use at each step**: ContentPlanner -> AssetGenerator -> VideoAssembler
- **Include error handling and user communication** guidelines

**3) ContentPlannerAgent - Structured Output (output_schema)**

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

Key points:
- **`output_schema`**: When a Pydantic model is specified, the agent's output is forced to conform to the schema structure. It forces the LLM to respond in a defined JSON structure instead of free-form text.
- **`output_key`**: The key name for storing the agent's output in the session state. When set to `"content_planner_output"`, other agents can access it later via `tool_context.state.get("content_planner_output")`.
- **`Field(description=...)`**: The description of each field serves as a guide for the LLM to generate correct values.

**4) Key Strategies in the Content Planner Prompt**

Notable aspects of the prompt:
- **20-second maximum limit**: A constraint matching the characteristics of YouTube Shorts
- **Flexible scene count**: Agent determines the optimal composition with 3-6 scenes
- **Timing strategy**: Quick intro (2-3 seconds), main content (3-5 seconds), strong ending (2-4 seconds)
- **Validation requirement**: Instructions to verify total time doesn't exceed 20 seconds before output
- **Concrete examples provided**: The "Perfect Scrambled Eggs" example clearly shows the expected output format

#### Code Analysis - Data Flow Between Agents

```
User input ("Make a Shorts about cooking")
    |
    v
ShortsProducerAgent (Orchestrator)
    |  Called via AgentTool
    v
ContentPlannerAgent
    |  output_schema: ContentPlanOutput
    |  output_key: "content_planner_output"
    v
Saved in session state: state["content_planner_output"] = {
    "topic": "...",
    "total_duration": 18,
    "scenes": [
        {"id": 1, "narration": "...", "visual_description": "...", ...},
        {"id": 2, ...},
        ...
    ]
}
```

#### Practice Points
- Experiment with how `Field(description=...)` in Pydantic models affects LLM output.
- Test changing the `output_key` and accessing it from subsequent agents.
- See how output quality changes when you modify the examples in the prompt.

---

### 11.2 Prompt Builder Agent - Image Prompt Optimization Agent

#### Topic and Objectives
Implement an agent that transforms the visual descriptions from the content plan into prompts optimized for GPT-Image-1. Understand the differences between ParallelAgent and SequentialAgent and design agent hierarchies.

#### Key Concepts

**1) Utilizing Three Agent Types**

This section introduces all three main ADK agent types:

```python
# 1. ParallelAgent - Executes sub-agents simultaneously
from google.adk.agents import ParallelAgent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,   # Image generation pipeline
        # voice_generator_agent, # (to be added in 11.4)
    ],
)
```

**ParallelAgent** executes agents in the `sub_agents` list **simultaneously (in parallel)**. Image generation and voice generation don't depend on each other, so parallel processing is possible.

```python
# 2. SequentialAgent - Executes sub-agents in order
from google.adk.agents import SequentialAgent

image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,   # First: Prompt optimization
        # image_builder_agent,  # Next: Image generation (to be added in 11.3)
    ],
)
```

**SequentialAgent** executes `sub_agents` **in order**. Since you must optimize the prompt before generating images, sequential processing is mandatory.

```python
# 3. Agent - General LLM-based agent (can use tools and reason)
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

**Agent** is the most basic agent that uses an LLM to reason, call tools, and generate structured outputs.

**2) Agent Hierarchy Design Pattern**

```
AssetGeneratorAgent (ParallelAgent)
    |
    +-- ImageGeneratorAgent (SequentialAgent)
    |       |
    |       +-- PromptBuilderAgent (Agent) -- Prompt optimization
    |       +-- ImageBuilderAgent (Agent)  -- Image generation
    |
    +-- VoiceGeneratorAgent (Agent)         -- Voice generation
```

Core principles of this design:
- **Image generation** must be **sequential**: prompt optimization -> image generation (SequentialAgent)
- **Image generation** and **voice generation** can proceed **in parallel** (ParallelAgent)
- Each individual task requires LLM reasoning (Agent)

**3) Prompt Builder Output Schema**

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

**4) Data Transfer Between Agents via State**

The prompt references the previous agent's output using `{content_planner_output}`:

```python
PROMPT_BUILDER_PROMPT = """
...
## Your Task:
Take the structured content plan: {content_planner_output} and create
optimized vertical image generation prompts for each scene...
"""
```

When you use `{variable_name}` syntax in ADK's `instruction`, the corresponding key's value is automatically injected from the session state. This is how it connects with `output_key`:
1. ContentPlannerAgent saves its result with `output_key="content_planner_output"`
2. PromptBuilderAgent's `instruction` references that data via `{content_planner_output}`

**5) Prompt Optimization Strategy**

Enhancement tasks performed by the prompt builder:
- **Adding technical specs**: 9:16 vertical ratio, 1080x1920 resolution
- **Visual details**: Lighting, camera angles, composition, etc.
- **Text overlay instructions**: Position, padding, readability
- **Style consistency**: Maintaining the same visual style across all scenes

Example transformation:
```
Original: "Stovetop dial on low"
Optimized: "Close-up shot of modern stovetop control dial set to low heat
setting, 9:16 portrait aspect ratio, 1080x1920 resolution, vertical
composition, warm kitchen lighting, shallow depth of field, photorealistic,
sharp focus, with bold white text 'Secret #1: Low Heat' positioned at
top center of image with generous padding from borders..."
```

#### Practice Points
- Swap `ParallelAgent` and `SequentialAgent` to observe how execution order changes.
- Check the logs to see what actual values replace the `{content_planner_output}` template variable.
- Modify the prompt optimization guidelines to change the style of generated images.

---

### 11.3 Image Builder Agent - Image Generation Agent

#### Topic and Objectives
Implement an agent that generates actual images using the optimized prompts with the OpenAI GPT-Image-1 API and saves them using ADK's Artifact system.

#### Key Concepts

**1) Agent Definition - An Agent with Tools**

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

This agent uses a Python function called `generate_images` as a tool. In ADK, when you pass a regular Python function to the `tools` list, it is automatically registered as a tool.

**2) Adding ImageBuilder to SequentialAgent**

```python
image_generator_agent = SequentialAgent(
    name="ImageGeneratorAgent",
    sub_agents=[
        prompt_builder_agent,    # Step 1: Prompt optimization
        image_builder_agent,     # Step 2: Image generation
    ],
)
```

The image generation pipeline is now complete. PromptBuilder runs first and saves the optimized prompts to state, then ImageBuilder reads those prompts and generates images.

**3) generate_images Tool - The Core of ADK's Artifact System**

```python
import base64
from google.genai import types
from openai import OpenAI
from google.adk.tools.tool_context import ToolContext

client = OpenAI()


async def generate_images(tool_context: ToolContext):

    # 1. Get previous agent's output from session state
    prompt_builder_output = tool_context.state.get("prompt_builder_output")
    optimized_prompts = prompt_builder_output.get("optimized_prompts")

    # 2. Check existing artifact list (prevent duplicate generation)
    existing_artifacts = await tool_context.list_artifacts()

    generated_images = []

    for prompt in optimized_prompts:
        scene_id = prompt.get("scene_id")
        enhanced_prompt = prompt.get("enhanced_prompt")
        filename = f"scene_{scene_id}_image.jpeg"

        # 3. Skip if already exists (caching)
        if filename in existing_artifacts:
            generated_images.append({
                "scene_id": scene_id,
                "prompt": enhanced_prompt[:100],
                "filename": filename,
            })
            continue

        # 4. Generate image with OpenAI GPT-Image-1 API
        image = client.images.generate(
            model="gpt-image-1",
            prompt=enhanced_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",    # Vertical format (2:3 ratio)
        )

        # 5. Base64 decode
        image_bytes = base64.b64decode(image.data[0].b64_json)

        # 6. Save as ADK Artifact
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

**Detailed Explanation of Key Concepts:**

**(a) `ToolContext` - The Core Context for Tools**

`tool_context: ToolContext` is a special parameter that ADK automatically injects into tool functions. Through it, you can:
- `tool_context.state`: Access the session's shared state (data sharing between agents)
- `tool_context.list_artifacts()`: Query the list of saved artifacts
- `tool_context.save_artifact()`: Save a new artifact
- `tool_context.load_artifact()`: Load an artifact

ADK automatically injects the context when it finds `tool_context: ToolContext` in the function signature. You don't need to pass it manually.

**(b) Artifact System**

Artifact is the mechanism for managing binary data (images, audio, video, etc.) in ADK. It uses `google.genai.types.Part` and `types.Blob` to store data along with MIME types.

```python
artifact = types.Part(
    inline_data=types.Blob(
        mime_type="image/jpeg",
        data=image_bytes,      # raw bytes
    )
)
await tool_context.save_artifact(filename=filename, artifact=artifact)
```

Instead of the file system, this stores files in ADK's session management system, allowing data to persist across sessions and be accessible by other agents.

**(c) Duplicate Generation Prevention Pattern**

```python
existing_artifacts = await tool_context.list_artifacts()
if filename in existing_artifacts:
    # Skip if already exists
    continue
```

If an image has already been generated, it won't be regenerated. This is an important pattern that saves API call costs and time.

**(d) OpenAI Image Generation API Parameters**

```python
image = client.images.generate(
    model="gpt-image-1",       # OpenAI's image generation model
    prompt=enhanced_prompt,     # Optimized prompt
    n=1,                        # Generate 1 image
    quality="low",              # Quality (low/medium/high)
    moderation="low",           # Content filtering level
    output_format="jpeg",       # Output format
    background="opaque",        # Background (opaque: non-transparent)
    size="1024x1536",           # Vertical ratio (YouTube Shorts)
)
```

#### Practice Points
- Change the `quality` parameter to `"high"` and compare the image quality difference.
- See what happens when you change `size` to `"1024x1024"`.
- Test how the caching mechanism via `tool_context.list_artifacts()` works on re-execution.

---

### 11.4 Audio Narration Agent - Voice Narration Agent

#### Topic and Objectives
Implement an agent that generates narration audio for each scene using the OpenAI TTS (Text-to-Speech) API. Complete the structure for generating images and voice simultaneously by adding the voice generator to ParallelAgent.

#### Key Concepts

**1) Adding VoiceGenerator to ParallelAgent**

```python
from google.adk.agents import ParallelAgent
from .prompt import ASSET_GENERATOR_DESCRIPTION
from .image_generator.agent import image_generator_agent
from .voice_generator.agent import voice_generator_agent

asset_generator_agent = ParallelAgent(
    name="AssetGeneratorAgent",
    description=ASSET_GENERATOR_DESCRIPTION,
    sub_agents=[
        image_generator_agent,     # Image generation pipeline
        voice_generator_agent,     # Voice generation agent (added!)
    ],
)
```

Now when `AssetGeneratorAgent` is called, image generation (`ImageGeneratorAgent`) and voice generation (`VoiceGeneratorAgent`) execute **simultaneously**. Since the two tasks are independent, parallel processing can significantly reduce overall execution time.

**2) VoiceGeneratorAgent Definition**

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

**3) Voice Selection Strategy - Prompt Design**

This agent's prompt is designed to have the LLM **select the appropriate voice on its own** based on the content's mood:

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

Key design principles:
- References the entire content plan from the previous step via `{content_planner_output}`
- Provides voice selection guidelines but **delegates the final decision to the LLM**
- Separates `input` (text to read) and `instructions` (tone, speed, etc.) for each scene for fine-grained control

**4) generate_narrations Tool - TTS API Usage**

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

        # Caching: skip if already exists
        if filename in existing_artifacts:
            generated_narrations.append({
                "scene_id": scene_id,
                "filename": filename,
                "input": text_input,
                "instructions": instructions[:50],
            })
            continue

        # OpenAI TTS API call (streaming response)
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text_input,
            instructions=instructions,
        ) as response:
            audio_data = response.read()

        # Save as ADK Artifact
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

**Detailed Explanation of Key Concepts:**

**(a) Tool Function Parameter Design**

```python
async def generate_narrations(
    tool_context: ToolContext,    # Automatically injected by ADK
    voice: str,                   # Selected and passed by the LLM
    voice_instructions: List[Dict[str, Any]]  # Composed and passed by the LLM
):
```

`tool_context` is automatically injected by ADK, while `voice` and `voice_instructions` are generated by the LLM with appropriate values according to the prompt instructions. This is the power of ADK tools -- **the LLM intelligently determines the arguments when calling a tool**.

**(b) Receiving TTS Data via Streaming Response**

```python
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice=voice,
    input=text_input,
    instructions=instructions,
) as response:
    audio_data = response.read()
```

Using `with_streaming_response` allows receiving audio data in a streaming fashion, which is memory-efficient. The `gpt-4o-mini-tts` model allows fine-grained control over speaking speed, tone, emotion, etc. through the `instructions` parameter.

**(c) File Naming Convention**

```python
filename = f"scene_{scene_id}_narration.mp3"   # Audio
filename = f"scene_{scene_id}_image.jpeg"       # Image (from 11.3)
```

Using a consistent naming convention allows the VideoAssembler to easily find and sort files by scene number later.

#### Practice Points
- Try various `voice` parameters to compare voice differences for the same text.
- Try different instructions like "Speak very slowly and dramatically".
- Switch from ParallelAgent to SequentialAgent and measure the execution time difference.

---

### 11.5 Video Assembly - Video Assembly Agent

#### Topic and Objectives
Implement an agent that assembles all generated image and audio artifacts into a final MP4 video using FFmpeg. Learn about Artifact loading, temporary file management, and FFmpeg filter graph construction.

#### Key Concepts

**1) VideoAssemblerAgent Definition**

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

**2) Connecting the Full Pipeline to the Root Agent**

```python
shorts_producer_agent = Agent(
    name="ShortsProducerAgent",
    model=MODEL,
    description=SHORTS_PRODUCER_DESCRIPTION,
    instruction=SHORTS_PRODUCER_PROMPT,
    tools=[
        AgentTool(agent=content_planner_agent),     # Step 1
        AgentTool(agent=asset_generator_agent),      # Step 2
        AgentTool(agent=video_assembler_agent),      # Step 3 (added!)
    ],
)
```

The full pipeline is now complete.

**3) assemble_video Tool - Video Assembly with FFmpeg**

This tool is the most complex part of the project. Let's analyze it step by step.

**(a) Loading Artifacts and Creating Temporary Files**

```python
async def assemble_video(tool_context: ToolContext) -> str:
    temp_files = []  # Track temporary files for cleanup

    try:
        # Get scene information from content plan
        content_planner_output = tool_context.state.get(
            "content_planner_output", {}
        )
        scenes = content_planner_output.get("scenes", [])

        # Query saved artifact list
        existing_artifacts = await tool_context.list_artifacts()

        # Classify and sort image/audio files
        image_files = []
        audio_files = []
        for artifact_name in existing_artifacts:
            if artifact_name.endswith("_image.jpeg"):
                image_files.append(artifact_name)
            elif artifact_name.endswith("_narration.mp3"):
                audio_files.append(artifact_name)

        # Sort by scene number
        def extract_scene_number(filename):
            match = re.search(r"scene_(\d+)_", filename)
            return int(match.group(1)) if match else 0

        image_files.sort(key=extract_scene_number)
        audio_files.sort(key=extract_scene_number)
```

Since ADK Artifacts are stored in memory/sessions, they must be extracted to **temporary files** for use with FFmpeg:

```python
        temp_image_paths = []
        temp_audio_paths = []

        for i, (image_name, audio_name) in enumerate(
            zip(image_files, audio_files)
        ):
            # Save image artifact to temporary file
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
                temp_files.append(temp_image.name)  # Add to cleanup list

            # Process audio artifact the same way
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

**(b) FFmpeg Filter Graph Construction**

Using FFmpeg's `filter_complex` to combine multiple images and audio into a single video:

```python
        input_args = []
        filter_parts = []

        for i, (temp_image, temp_audio) in enumerate(
            zip(temp_image_paths, temp_audio_paths)
        ):
            # Add each scene's image and audio as inputs
            input_args.extend(["-i", temp_image, "-i", temp_audio])

            scene_duration = scenes[i].get("duration", 4)

            # Create video stream: repeat static image for specified duration
            total_frames = int(30 * scene_duration)  # 30fps * seconds
            filter_parts.append(
                f"[{i*2}:v]scale=1080:1920:"
                f"force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps=30,"
                f"loop={total_frames-1}:size=1:start=0[v{i}]"
            )

            # Audio stream (pass through without conversion)
            filter_parts.append(f"[{i*2+1}:a]anull[a{i}]")

        # Concatenate all streams
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

**FFmpeg Filter Graph Explained:**

For each scene:
1. `scale=1080:1920:force_original_aspect_ratio=decrease` - Scale to 1080x1920 (vertical) while maintaining aspect ratio
2. `pad=1080:1920:(ow-iw)/2:(oh-ih)/2` - Fill insufficient areas with padding (letterbox)
3. `setsar=1` - Set pixel aspect ratio to 1:1
4. `fps=30` - Set to 30fps
5. `loop={total_frames-1}:size=1:start=0` - Repeat the static image for the specified number of frames

The `concat` filter joins all video and audio streams in order.

**(c) FFmpeg Execution and Final Video Storage**

```python
        ffmpeg_cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-filter_complex", ";".join(filter_parts),
                "-map", "[outv]",
                "-map", "[outa]",
                "-c:v", "libx264",    # H.264 video codec
                "-c:a", "aac",        # AAC audio codec
                "-pix_fmt", "yuv420p", # Highly compatible pixel format
                "-r", "30",           # 30fps
                output_path,
            ]
        )

        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        # Save final video as artifact
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

**(d) Temporary File Cleanup - finally Pattern**

```python
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Failed to cleanup {temp_file}: {e}")
```

The `try/finally` pattern ensures temporary files are always cleaned up even if an error occurs. This is an important resource management pattern in production code.

#### Practice Points
- Change the FFmpeg filter's `scale` and `pad` options to test various resolutions.
- Adjust the `loop` value to change scene duration.
- Check the error handling section and observe what errors occur when FFmpeg is not installed.
- Research how to use pipes instead of temporary files.

---

### 11.6 Callbacks - Controlling Agent Behavior Through Callbacks

#### Topic and Objectives
Understand ADK's callback system and learn how to use `before_model_callback` to inspect and filter user input before LLM calls.

#### Key Concepts

**1) What are Callbacks?**

Callbacks are functions that can intervene at specific points during an agent's operation. ADK provides various callback points:
- `before_model_callback`: Executes **before** an LLM call
- `after_model_callback`: Executes **after** an LLM call
- `before_tool_callback`: Executes **before** a tool call
- `after_tool_callback`: Executes **after** a tool call

**2) Implementing before_model_callback**

```python
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    # Check the last message in conversation history
    history = llm_request.contents
    last_message = history[-1]

    # Only inspect if it's a user message
    if last_message.role == "user":
        text = last_message.parts[0].text

        # Check for banned keywords
        if "hummus" in text:
            # Return immediate rejection response without calling LLM
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

    # Returning None proceeds with normal LLM call
    return None
```

**How Callbacks Work:**

```
User Input
    |
    v
before_model_callback executes
    |
    +-- Returns LlmResponse --> Skips LLM call and returns immediate response
    |
    +-- Returns None ---------> Proceeds with normal LLM call
                                |
                                v
                            LLM generates response
```

- **If `LlmResponse` is returned**: The LLM call is completely blocked, and the returned response is used as-is. No API costs are incurred.
- **If `None` is returned**: The normal LLM call proceeds.

**3) Registering Callbacks**

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
    before_model_callback=before_model_callback,  # Register callback!
)
```

When you pass a function to the `before_model_callback` parameter, the callback is executed before every LLM call from this agent.

**4) Use Cases for Callbacks**

This example shows simple keyword filtering, but in real projects, callbacks can be used in various ways:

| Use Case | Description |
|----------|-------------|
| **Content Filtering** | Block inappropriate requests before sending to the LLM |
| **Cost Control** | Limit LLM calls when a certain token count is exceeded |
| **Logging/Monitoring** | Log all LLM calls |
| **Caching** | Return cached responses for identical requests |
| **Prompt Modification** | Dynamically modify prompts sent to the LLM |
| **Authentication/Authorization** | Restrict features based on user permissions |

**5) CallbackContext and LlmRequest**

```python
def before_model_callback(
    callback_context: CallbackContext,  # Access agent name, state, etc.
    llm_request: LlmRequest,           # Content of the request to send to LLM
):
```

- `callback_context.agent_name`: Name of the currently executing agent
- `llm_request.contents`: Conversation history (composed of role, parts)
- Modifying `llm_request` can change the request itself that is sent to the LLM

#### Practice Points
- Add keyword filters other than "hummus".
- Implement `after_model_callback` to post-process LLM responses.
- Print `callback_context.agent_name` to observe which agent triggers the callback.
- Add `before_tool_callback` to implement logging before tool calls.
- Modify `llm_request.contents` in a callback to dynamically add system messages.

---

## 3. Chapter Key Summary

### ADK Agent Types Summary

| Agent Type | Class | Characteristics | Usage in This Project |
|-----------|-------|----------------|----------------------|
| **Agent** | `google.adk.agents.Agent` | LLM-based reasoning, can use tools | ShortsProducer, ContentPlanner, PromptBuilder, ImageBuilder, VoiceGenerator, VideoAssembler |
| **ParallelAgent** | `google.adk.agents.ParallelAgent` | Executes sub-agents in parallel | AssetGenerator (simultaneous image + voice generation) |
| **SequentialAgent** | `google.adk.agents.SequentialAgent` | Executes sub-agents sequentially | ImageGenerator (prompt optimization -> image generation) |

### Key ADK Concepts Summary

| Concept | Description | Related Code |
|---------|-------------|-------------|
| **AgentTool** | Wraps an agent as a tool so other agents can call it | `AgentTool(agent=content_planner_agent)` |
| **output_schema** | Forces agent output structure with a Pydantic model | `output_schema=ContentPlanOutput` |
| **output_key** | Key for saving agent output to session state | `output_key="content_planner_output"` |
| **ToolContext** | Access state/artifacts from tool functions | `tool_context.state.get(...)` |
| **Artifact** | Store/manage binary data (images, audio, etc.) | `tool_context.save_artifact(...)` |
| **Callback** | Functions that intervene at specific agent operation points | `before_model_callback=...` |
| **Prompt Template Variables** | Auto-inject state values via `{variable_name}` | `{content_planner_output}` |

### Design Patterns Summary

1. **Orchestrator Pattern**: Root agent coordinates sub-agents in order
2. **Pipeline Pattern**: SequentialAgent connects data processing steps sequentially
3. **Parallel Processing Pattern**: ParallelAgent runs independent tasks simultaneously
4. **Artifact Caching Pattern**: Check existing artifact existence to prevent duplicate generation
5. **Temporary File Management Pattern**: Guarantee resource cleanup with try/finally
6. **Callback Guard Pattern**: Input validation and filtering with before_model_callback

---

## 4. Practice Exercises

### Exercise 1: Basic - Add a New Content Type
Modify the ContentPlannerAgent's prompt to support "news summary" style YouTube Shorts. Include the urgent tone characteristic of news, fact-focused narration, and news-graphic style visual descriptions.

### Exercise 2: Intermediate - Add a Subtitle Agent
Create a new `SubtitleGeneratorAgent` that generates SRT subtitle files based on each scene's narration text, and add it to the AssetGeneratorAgent's ParallelAgent. Modify the VideoAssemblerAgent's FFmpeg command to burn subtitles into the video.

### Exercise 3: Intermediate - Implement Multiple Callbacks
Implement the following callbacks:
- `before_model_callback`: Filter using a banned keyword list on user input
- `after_model_callback`: Count and log the token count of LLM responses
- `before_tool_callback`: Record the start time of tool calls
- `after_tool_callback`: Calculate tool execution time and output performance logs

### Exercise 4: Advanced - Add a Background Music Agent
Implement a `BGMAgent` that generates or selects background music and add it to the AssetGeneratorAgent. Modify the VideoAssemblerAgent's FFmpeg command to mix narration and BGM (with volume adjustment) into the final video.

### Exercise 5: Advanced - Error Recovery Mechanism
Implement automatic retry logic for when API errors occur during image or voice generation. Include a maximum of 3 retries, exponential backoff, and preservation of successfully generated artifacts on partial success.

---

## Appendix: How to Run the Project

### Environment Setup

```bash
# Navigate to the project directory
cd youtube-shorts-maker

# Install dependencies (using uv)
uv sync

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"

# Verify FFmpeg installation
ffmpeg -version
```

### Running ADK

```bash
# Run with ADK development server
adk web youtube_shorts_maker

# Or run with CLI
adk run youtube_shorts_maker
```

### Final Directory Structure

```
youtube-shorts-maker/
    youtube_shorts_maker/
        __init__.py
        agent.py                          # Root agent + callbacks
        prompt.py                         # Orchestrator prompt
        sub_agents/
            content_planner/
                agent.py                  # ContentPlannerAgent
                prompt.py                 # Content planner prompt
            asset_generator/
                agent.py                  # AssetGeneratorAgent (ParallelAgent)
                prompt.py                 # Asset generator description
                image_generator/
                    agent.py              # ImageGeneratorAgent (SequentialAgent)
                    prompt_builder/
                        agent.py          # PromptBuilderAgent
                        prompt.py         # Prompt builder prompt
                    image_builder/
                        agent.py          # ImageBuilderAgent
                        prompt.py         # Image builder prompt
                        tools.py          # generate_images tool
                voice_generator/
                    agent.py              # VoiceGeneratorAgent
                    prompt.py             # Voice generator prompt
                    tools.py              # generate_narrations tool
            video_assembler/
                agent.py                  # VideoAssemblerAgent
                prompt.py                 # Video assembly prompt
                tools.py                  # assemble_video tool (FFmpeg)
```
