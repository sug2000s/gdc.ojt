# Chapter 15: LangGraph YouTube Thumbnail Auto-Generation Project

---

## 1. Chapter Overview

In this chapter, we build a complete pipeline that automatically generates thumbnails from YouTube videos using **LangGraph**. This project goes beyond simple image generation, designing the entire workflow as a graph -- from extracting audio from videos, speech recognition (transcription), text summarization, AI image generation, to generating the final high-definition thumbnail incorporating human feedback.

### Key Learning Objectives

| Section | Topic | Core Technology |
|---------|-------|-----------------|
| 15.0 | Project Introduction & Environment Setup | uv, pyproject.toml, dependency management |
| 15.1 | Audio Extraction & Speech Recognition | ffmpeg, OpenAI Whisper API, StateGraph |
| 15.2 | Parallel Summarizer Nodes | Send API, Map-Reduce pattern, Annotated reducers |
| 15.3 | Thumbnail Sketch Generation Nodes | GPT Image API, parallel image generation |
| 15.4 | Human Feedback Loop | interrupt, Command, InMemorySaver |
| 15.5 | HD Thumbnail Generation & Deployment | LangGraph CLI, langgraph.json, production deployment |

### Overall Graph Flow

```
[START]
   |
   v
[extract_audio] -----> Extract mp3 from mp4 using ffmpeg
   |
   v
[transcribe_audio] --> Convert speech to text using OpenAI Whisper
   |
   v (conditional: dispatch_summarizers)
[summarize_chunk] x N --> Split text into chunks and summarize in parallel
   |
   v
[mega_summary] --------> Consolidate all chunk summaries into one final summary
   |
   v (conditional: dispatch_artists)
[generate_thumbnails] x 5 --> Generate 5 different thumbnail sketches in parallel
   |
   v
[human_feedback] ------> Collect user feedback via interrupt
   |
   v
[generate_hd_thumbnail] -> Generate final HD thumbnail incorporating feedback
   |
   v
[END]
```

---

## 2. Detailed Section Descriptions

---

### 15.0 Project Introduction & Environment Setup

**Commit:** `c8e3d16` "15.0 Introduction"

#### Topic and Objectives

This is the stage where we lay the project's foundation. We initialize the project using the Python package manager **uv** and configure all necessary dependencies. We also include `ipykernel` as a development dependency so we can start developing in a Jupyter Notebook environment.

#### Core Concepts

**uv Package Manager**: A modern package manager in the Python ecosystem that replaces `pip` and `venv`. Written in Rust, it is extremely fast and manages projects based on `pyproject.toml`. It ensures reproducible environments by locking exact dependency versions through the `uv.lock` file.

**pyproject.toml**: The standard configuration file for Python projects. It declaratively defines project metadata and dependencies.

#### Code Analysis

```toml
[project]
name = "youtube-thumbnail-maker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.7",
    "langgraph-cli[inmem]>=0.4.2",
    "langsmith>=0.4.28",
    "openai[aiohttp]>=1.107.3",
    "python-dotenv>=1.1.1",
    "tiktoken>=0.11.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

**Dependency Analysis:**

- **`langchain[openai]`**: LangChain framework's OpenAI integration. `[openai]` is an extras specifier that installs additional OpenAI-related packages.
- **`langgraph`**: LangGraph core library. The essential tool for building graph-based agent workflows.
- **`langgraph-cli[inmem]`**: LangGraph CLI tool. `[inmem]` includes in-memory checkpointer support. Used for running local development servers.
- **`langsmith`**: LangSmith observability platform. Enables monitoring and debugging of graph executions.
- **`openai[aiohttp]`**: OpenAI Python SDK. `[aiohttp]` adds asynchronous HTTP support. Used for the Whisper API and image generation API.
- **`python-dotenv`**: A utility for loading environment variables (API keys, etc.) from `.env` files.
- **`tiktoken`**: OpenAI's tokenizer. Used for counting the number of tokens in text.
- **`ipykernel`**: A development dependency that enables using a Python kernel in Jupyter Notebooks.

#### Practice Points

1. Create a new project with `uv init youtube-thumbnail-maker`.
2. Add dependencies with `uv add langchain[openai] langgraph openai[aiohttp]`, etc.
3. Create a `.env` file and set the `OPENAI_API_KEY`.
4. Launch the Jupyter environment with `uv run jupyter notebook`.

---

### 15.1 Audio Extraction & Speech Recognition

**Commit:** `4133ae6` "15.1 Audio Extraction and Transcription"

#### Topic and Objectives

Build a LangGraph graph that extracts audio (mp3) from a YouTube video file (mp4) and converts speech to text using OpenAI's Whisper model. In this step, we learn the **basic structure of StateGraph** and the concepts of **Nodes and Edges**.

#### Core Concepts

**StateGraph**: The core class of LangGraph that defines a graph where data is passed between nodes centered around state. Each node receives state as input and returns an updated portion of the state.

**State Definition with TypedDict**: Using Python's `TypedDict` to define the schema of the shared state across the entire graph in a type-safe manner. This can be thought of as the graph's "data contract."

**ffmpeg**: A powerful command-line tool for audio/video processing. Executed as an external process from Python via `subprocess.run()`.

**OpenAI Whisper API**: A speech-to-text model. It supports multiple languages, and recognition accuracy for domain-specific terms can be improved through the `prompt` parameter.

#### Code Analysis

**Step 1: State Definition**

```python
from langgraph.graph import END, START, StateGraph
from typing import TypedDict
import subprocess
from openai import OpenAI


class State(TypedDict):
    video_file: str       # Input video file path
    audio_file: str       # Extracted audio file path
    transcription: str    # Speech recognition result text
```

`State` is the data structure shared across the entire graph. Each node reads part or all of this State and returns only the portions to update as a dictionary. The returned values are **merged** into the existing State.

**Step 2: Audio Extraction Node**

```python
def extract_audio(state: State):
    output_file = state["video_file"].replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file,
    ]
    subprocess.run(command)
    return {
        "audio_file": output_file,
    }
```

Key points in this node:

- **`-filter:a "atempo=2.0"`**: Accelerates the audio playback speed by 2x. Since the Whisper API has an upload file size limit (25MB), speeding up the audio is a practical trick to reduce file size. It has little impact on speech recognition quality.
- **`-y`**: A flag that automatically allows overwriting if the output file already exists.
- **Return value**: Only returns `{"audio_file": output_file}`, updating just the `audio_file` field of the State.

**Step 3: Speech Recognition Node**

```python
def transcribe_audio(state: State):
    client = OpenAI()
    with open(state["audio_file"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en",
            prompt="Netherlands, Rotterdam, Amsterdam, The Hage",
        )
        return {
            "transcription": transcription,
        }
```

- **`model="whisper-1"`**: Uses OpenAI's Whisper speech recognition model.
- **`response_format="text"`**: Returns results in plain text format. Other formats such as `"json"`, `"verbose_json"`, `"srt"`, and `"vtt"` are also supported.
- **`language="en"`**: Specifies English as the language to improve recognition accuracy.
- **`prompt` parameter**: This is a very useful feature. By providing proper nouns (city names, etc.) that are expected to appear in the video, it helps Whisper recognize those words accurately.

**Step 4: Graph Construction and Execution**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_edge("transcribe_audio", END)

graph = graph_builder.compile()
```

LangGraph graph construction consists of three steps:

1. **Add nodes** (`add_node`): Register functions to execute along with their names.
2. **Add edges** (`add_edge`): Define the execution order (direction) between nodes. `START` and `END` are special nodes representing the beginning and end of the graph.
3. **Compile** (`compile`): Transform the graph into an executable form.

```python
graph.invoke({"video_file": "netherlands.mp4"})
```

Pass the initial state to `invoke()` to execute the graph. The graph runs in the order `START` -> `extract_audio` -> `transcribe_audio` -> `END`, and the final state is returned.

#### Practice Points

1. Verify that `ffmpeg` is installed on the system (`ffmpeg -version`).
2. Test the pipeline first with a short test video (1-2 minutes).
3. Compare speech recognition quality differences by changing the `atempo` value (1.5, 2.0, 3.0).
4. Observe recognition accuracy changes by putting different proper nouns in the `prompt` parameter.

---

### 15.2 Parallel Summarizer Nodes

**Commit:** `a664f18` "15.2 Summarizer Nodes"

#### Topic and Objectives

Split a long transcription text into multiple chunks and implement a **Map-Reduce pattern** that summarizes each chunk **in parallel**. Learn dynamic parallel processing using LangGraph's `Send` API and `Annotated` reducers.

#### Core Concepts

**Map-Reduce Pattern**: A classic distributed computing pattern for processing large-scale data.
- **Map phase**: Split data into multiple pieces and process each independently (parallel summarization).
- **Reduce phase**: Combine individual results into one (the mega_summary in the next section).

**Send API**: The core mechanism in LangGraph that enables dynamic parallel processing. When a list of `Send("node_name", data)` is returned, the specified node is executed in parallel for each piece of data.

**Annotated Reducer**: A reducer function is specified on a type like `Annotated[list[str], operator.add]`. When multiple nodes return values to the same State field, this defines how the values should be combined. `operator.add` concatenates lists.

#### Code Analysis

**State Extension: Annotated Reducer**

```python
from langgraph.types import Send
from typing_extensions import Annotated
import operator
import textwrap
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")


class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]  # Reducer added!
```

The key here is using `Annotated[list[str], operator.add]` for the `summaries` field. Without this, when multiple parallel nodes write to `summaries` simultaneously, only the last value would remain. By specifying the `operator.add` reducer, each node's return value is **accumulated** into the existing list.

> **Note**: You must use `typing_extensions.Annotated`, not `typing.Annotated`. Using `typing.Annotated` with function-call syntax like `Annotated(list[str], operator.add)` will raise a `TypeError: Cannot instantiate typing.Annotated` error. Always use bracket syntax: `Annotated[list[str], operator.add]`.

**Dispatcher Function: Dynamic Parallel Branching**

```python
def dispatch_summarizers(state: State):
    transcription = state["transcription"]
    chunks = []
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append({"id": i + 1, "chunk": chunk})
    return [Send("summarize_chunk", chunk) for chunk in chunks]
```

How this function works:

1. **Text splitting**: `textwrap.wrap(transcription, 500)` splits long text into segments of up to 500 characters. It respects word boundaries so words are not split in the middle.
2. **Chunk construction**: Each chunk is assigned a unique ID to track ordering.
3. **Return Send list**: Returning a list of `Send("summarize_chunk", chunk)` causes LangGraph to execute the `summarize_chunk` node **in parallel** for each chunk.

This function is used as a router for `add_conditional_edges`. While typical conditional edges return the next node name (a string), returning `Send` objects creates dynamic parallel branching.

**Summarizer Node**

```python
def summarize_chunk(chunk):
    chunk_id = chunk["id"]
    chunk = chunk["chunk"]

    response = llm.invoke(
        f"""
        Please summarize the following text.

        Text: {chunk}
        """
    )
    summary = f"[Chunk {chunk_id}] {response.content}"
    return {
        "summaries": [summary],
    }
```

Key points:

- **The parameter is not `state`**: Nodes called via `Send` receive the data passed from `Send` directly as their parameter, not the usual State. Here, it receives a `{"id": ..., "chunk": ...}` dictionary.
- **Wrapped in a list**: The return must be wrapped as `"summaries": [summary]`. Since the `operator.add` reducer concatenates lists, even a single value must be wrapped in a list for correct operation.
- **`[Chunk N]` prefix**: A chunk ID is prepended to each summary to enable later ordering.

**Graph Construction: Conditional Edges**

```python
graph_builder.add_node("summarize_chunk", summarize_chunk)

graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", END)
```

The third argument `["summarize_chunk"]` of `add_conditional_edges` specifies the list of nodes reachable from this conditional edge. LangGraph uses this information when validating the graph.

#### Practice Points

1. Observe summarization quality changes by modifying the chunk size (500) in `textwrap.wrap`.
2. Implement sequential processing (for loop) instead of `Send` and compare execution times.
3. Experiment with other reducers. For example, you can use a custom function instead of `operator.add`.
4. Compare summarization quality by switching the LLM model from `gpt-4o-mini` to `gpt-4o`.

---

### 15.3 Thumbnail Sketch Generation Nodes

**Commit:** `13e2688` "15.3 Thumbnail Sketcher Nodes"

#### Topic and Objectives

Consolidate the individual summaries from the previous section into a single comprehensive summary (`mega_summary`), then generate 5 different thumbnail sketches **in parallel** based on that summary. This uses OpenAI's image generation API (`gpt-image-1`) and applies the Map-Reduce pattern to image generation.

#### Core Concepts

**Mega Summary (Comprehensive Summary)**: This is the Reduce phase of Map-Reduce. It combines multiple chunk summaries into one, creating a single summary that captures the key content of the entire video. This summary becomes the basis for the thumbnails.

**OpenAI Image Generation API**: Generates images from text prompts using the `gpt-image-1` model. The `quality` parameter controls generation quality (`low`/`medium`/`high`), and the `moderation` parameter controls the content filtering level.

**Dual Map-Reduce**: This project uses Map-Reduce **twice**:
1. First: Text chunks -> parallel summarization -> comprehensive summary
2. Second: Comprehensive summary -> parallel thumbnail generation -> user selection

#### Code Analysis

**State Extension**

```python
class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]    # Newly added
    thumbnail_sketches: Annotated[list[str], operator.add]   # Newly added
    final_summary: str                                        # Newly added
```

Newly added fields:
- **`thumbnail_prompts`**: Stores the prompts used for each thumbnail generation. This is for reusing the prompt of the user's chosen thumbnail later.
- **`thumbnail_sketches`**: Stores the file paths of generated thumbnail images.
- **`final_summary`**: Stores the comprehensive summary text.

**Comprehensive Summary Node (mega_summary)**

```python
def mega_summary(state: State):
    all_summaries = "\n".join(state["summaries"])

    prompt = f"""
        You are given multiple summaries of different chunks from a video transcription.

        Please create a comprehensive final summary that combines all the key points.

        Individual summaries:

        {all_summaries}
    """

    response = llm.invoke(prompt)

    return {
        "final_summary": response.content,
    }
```

This node executes after all `summarize_chunk` nodes have completed. The `state["summaries"]` contains all chunk summaries accumulated by the `operator.add` reducer. It combines these into a single prompt and asks the LLM to produce a comprehensive summary.

**Artist Dispatcher: Second Parallel Branch**

```python
def dispatch_artists(state: State):
    return [
        Send(
            "generate_thumbnails",
            {
                "id": i,
                "summary": state["final_summary"],
            },
        )
        for i in [1, 2, 3, 4, 5]
    ]
```

This follows the same pattern as `dispatch_summarizers`, but this time creates 5 fixed tasks. Each task receives the same `final_summary` but has a different `id`. Thanks to the probabilistic nature of LLMs, different visual concepts are generated from the same summary each time.

**Thumbnail Generation Node**

```python
def generate_thumbnails(args):
    concept_id = args["id"]
    summary = args["summary"]

    prompt = f"""
    Based on this video summary, create a detailed visual prompt for a YouTube thumbnail.

    Create a detailed prompt for generating a thumbnail image that would attract viewers. Include:
        - Main visual elements
        - Color scheme
        - Text overlay suggestions
        - Overall composition

    Summary: {summary}
    """

    response = llm.invoke(prompt)
    thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    filename = f"thumbnail_{concept_id}.jpg"

    with open(filename, "wb") as file:
        file.write(image_bytes)

    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [filename],
    }
```

This node operates in two stages:

1. **Prompt generation**: The LLM (`gpt-4o-mini`) generates a visual prompt based on the video summary. It is instructed to include main visual elements, color scheme, text overlay, and overall composition.
2. **Image generation**: The generated prompt is passed to the `gpt-image-1` model to create the actual image.

Key parameters:
- **`quality="low"`**: Since this is the sketch stage, images are generated quickly at low quality. This saves cost and time.
- **`moderation="low"`**: Content filtering is set to be lenient, allowing more creative results.
- **`size="auto"`**: Lets the model automatically determine the image size.
- **`b64_json`**: The image is returned as Base64-encoded JSON. This is decoded and saved as a JPG file.

**Complete Graph Construction**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", END)

graph = graph_builder.compile()
```

The flow of the graph:
1. `summarize_chunk` -> `mega_summary`: Once all parallel summaries are complete, proceed to the comprehensive summary node.
2. `mega_summary` -> `dispatch_artists` -> `generate_thumbnails` x 5: Once the comprehensive summary is complete, generate 5 thumbnails in parallel.

#### Practice Points

1. Try changing the number of thumbnails generated (currently 5).
2. Modify the image generation prompt to request specific styles (illustration, photo, minimal, etc.).
3. Compare quality and generation time by changing the `quality` parameter to `"medium"` or `"high"`.
4. Compare the 5 generated thumbnails to observe the probabilistic nature of prompts.

---

### 15.4 Human Feedback Loop

**Commit:** `910cdef` "15.4 Human Feedback"

#### Topic and Objectives

Implement a **Human-in-the-Loop** pattern where the user selects one of the 5 AI-generated thumbnail sketches and provides additional feedback. This uses LangGraph's `interrupt` function, the `Command` class, and the `InMemorySaver` checkpointer.

#### Core Concepts

**Human-in-the-Loop (HITL)**: A pattern that inserts human judgment or input in the middle of an AI workflow. Rather than full automation, it involves human review and approval at critical decision points. This is essential for improving AI output quality and accurately reflecting user intent.

**interrupt()**: A function in LangGraph that **pauses** graph execution. When this function is called, graph execution stops and the current state is saved to the checkpointer. After receiving the user's response, execution resumes with `Command(resume=response)`.

**Command**: A class that delivers the user's response to an interrupted graph and resumes execution.

**InMemorySaver**: A checkpointer that saves graph state in memory. A checkpointer is required for `interrupt` to work. When pausing and resuming a graph, the previous state must be stored somewhere.

**thread_id**: A unique ID that identifies the same conversation/session. The checkpointer uses `thread_id` as the key to save and restore state.

#### Code Analysis

**New Imports and Setup**

```python
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
```

**State Extension**

```python
class State(TypedDict):
    # ... existing fields ...
    user_feedback: str    # User's modification feedback
    chosen_prompt: str    # Prompt of the user's chosen thumbnail
```

**Human Feedback Node**

```python
def human_feedback(state: State):
    answer = interrupt(
        {
            "chosen_thumbnail": "Which thumbnail do you like the most?",
            "feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
        }
    )
    user_feedback = answer["user_feedback"]
    chosen_prompt = answer["chosen_prompt"]
    return {
        "user_feedback": user_feedback,
        "chosen_prompt": state["thumbnail_prompts"][chosen_prompt - 1],
    }
```

How the `interrupt()` function works:

1. The dictionary passed as the argument to `interrupt()` is the **question/guidance message** shown to the user. This dictionary is included in the graph's return value and can be displayed by the client (UI).
2. Graph execution **stops** here. All state up to this point (5 thumbnail image paths, prompts, etc.) is saved to the checkpointer.
3. When the user provides a response, `interrupt()` returns that response and execution continues.

In the return value, `state["thumbnail_prompts"][chosen_prompt - 1]` retrieves the original prompt corresponding to the user's selected number (1-5). This prompt becomes the basis for HD thumbnail generation.

**HD Thumbnail Generation Node**

```python
def generate_hd_thumbnail(state: State):
    chosen_prompt = state["chosen_prompt"]
    user_feedback = state["user_feedback"]

    prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail
    prompt and create an enhanced version that incorporates the user's specific feedback.

    ORIGINAL PROMPT:
    {chosen_prompt}

    USER FEEDBACK TO INCORPORATE:
    {user_feedback}

    Create an enhanced prompt that:
        1. Maintains the core concept from the original prompt
        2. Specifically addresses and implements the user's feedback requests
        3. Adds professional YouTube thumbnail specifications:
            - High contrast and bold visual elements
            - Clear focal points that draw the eye
            - Professional lighting and composition
            - Optimal text placement and readability with generous padding from edges
            - Colors that pop and grab attention
            - Elements that work well at small thumbnail sizes
            - IMPORTANT: Always ensure adequate white space/padding between any text
              and the image borders
    """

    response = llm.invoke(prompt)
    final_thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=final_thumbnail_prompt,
        quality="high",           # Changed to high quality!
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    with open("thumbnail_final.jpg", "wb") as file:
        file.write(image_bytes)
```

Design points of this node:

1. **Prompt augmentation**: The user's selected original prompt is enhanced with their feedback and professional YouTube thumbnail specifications (high contrast, focal points, text margins, etc.).
2. **`quality="high"`**: Unlike the sketch stage's `"low"`, this generates at high quality since it is the final deliverable.
3. **Fixed filename**: Saved as `thumbnail_final.jpg` to clearly indicate it is the final result.

**Graph Compilation (Adding Checkpointer)**

```python
graph = graph_builder.compile(checkpointer=memory)
```

Passing `checkpointer=memory` activates state persistence for the graph. Without this, `interrupt()` will not work.

**Execution: Two-Phase Invocation**

```python
# Phase 1: Start graph execution (pauses at human_feedback)
config = {
    "configurable": {
        "thread_id": "1",
    },
}

graph.invoke(
    {"video_file": "netherlands.mp4"},
    config=config,
)

# Phase 2: Resume with user response
response = {
    "user_feedback": "Make sure the fella is smiling, remove any mention of audible, "
                     "or any logo, and give it a photo realistic, 3d style.",
    "chosen_prompt": 1,
}

graph.invoke(
    Command(resume=response),
    config=config,
)
```

The first `invoke` runs from `extract_audio` -> `transcribe_audio` -> `summarize_chunk` x N -> `mega_summary` -> `generate_thumbnails` x 5 -> `human_feedback` and pauses. The user reviews the 5 generated thumbnails and provides their preferred number and modification feedback.

In the second `invoke`, passing `Command(resume=response)` causes `interrupt()` to return the `response`, completing the `human_feedback` node. Then `generate_hd_thumbnail` executes to produce the final high-quality thumbnail.

> **Important**: The same `config` (same `thread_id`) must be used in both `invoke` calls. This is because the checkpointer manages state using `thread_id` as the key.

#### Practice Points

1. Try various feedback ("brighter", "remove text", "illustration style", etc.).
2. Run separate sessions by changing the `thread_id`.
3. Modify the message passed to `interrupt()` to provide more detailed guidance to the user.
4. Experiment with persistent storage by switching the checkpointer to `SqliteSaver`.

---

### 15.5 HD Thumbnail Generation & Production Deployment

**Commit:** `257d1b8` "15.5 HD Thumbnail Generation"

#### Topic and Objectives

Convert the code developed in Jupyter Notebook into a **production-deployable form**. Create a standalone `graph.py` module and configure it to be served via the **LangGraph CLI** through the `langgraph.json` configuration file.

#### Core Concepts

**LangGraph CLI (langgraph-cli)**: A CLI tool that enables running LangGraph graphs as a local server or deploying them to the cloud. Running the `langgraph dev` command starts a local server, allowing interaction with the graph through a REST API.

**langgraph.json**: The configuration file for LangGraph CLI. It defines which file contains which graph, what the dependencies are, and where to load environment variables from.

**Transition from Notebook to Module**: While Jupyter Notebooks are convenient during development, they must be converted to `.py` files for deployment. This process involves organizing and modularizing the code structure.

#### Code Analysis

**langgraph.json Configuration File**

```json
{
    "dependencies": [
        "graph.py"
    ],
    "graphs": {
        "mr_thumbs": "./graph.py:graph"
    },
    "env": ".env"
}
```

- **`dependencies`**: Specifies the project's dependency files. Here, `graph.py` itself is specified as a dependency. If a `pyproject.toml` exists, dependencies are typically installed automatically.
- **`graphs`**: Defines graphs to serve as name-path pairs.
  - `"mr_thumbs"`: The service name for the graph. Accessed by this name in API endpoints.
  - `"./graph.py:graph"`: File path and variable name separated by a colon. Serves the `graph` variable from the `graph.py` file.
- **`env`**: The path to the `.env` file from which to load environment variables.

**graph.py: Complete Production Code**

```python
graph = graph_builder.compile(name="mr_thumbs")
```

Differences from the Notebook version:
- `checkpointer=memory` has been removed. The LangGraph CLI automatically manages checkpointers.
- `name="mr_thumbs"` gives the graph a name for identification.

**The overall structure of graph.py** is as follows:

```python
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
import operator
import base64

llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[str], operator.add]
    final_summary: str
    user_feedback: str
    chosen_prompt: str

# --- Node functions (extract_audio, transcribe_audio, ...) ---

# --- Graph construction ---
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", END)

graph = graph_builder.compile(name="mr_thumbs")
```

**LangGraph CLI Execution**

```bash
# Start development server
langgraph dev

# Or run in in-memory mode
langgraph dev --no-browser
```

When executed, a local server starts, and you can visually inspect the graph in the LangGraph Studio UI, run it, and provide feedback at `interrupt` points.

#### Practice Points

1. Run the local server with `langgraph dev` and inspect the graph in LangGraph Studio.
2. Enter a `video_file` in the Studio UI and run the graph.
3. Provide feedback through the Studio UI at the `interrupt` point and check the results.
4. Try adding new nodes to the graph (e.g., watermark addition, image resizing).

---

## 3. Chapter Key Summary

### Core LangGraph Patterns

| Pattern | Description | Use Cases |
|---------|-------------|-----------|
| **StateGraph** | State-centric graph workflow | Foundation of all LangGraph projects |
| **Send (Map-Reduce)** | Dynamic parallel branching and result collection | Parallel text chunk summarization, multiple image generation |
| **Annotated Reducer** | Strategy for combining results from parallel nodes | List accumulation with `operator.add` |
| **interrupt / Command** | Human-in-the-Loop pattern | User selection, feedback collection |
| **Checkpointer** | Persistent storage of graph state | interrupt support, session management |
| **conditional_edges** | Conditional routing | Send-based dynamic branching |

### External Tools and APIs Used

| Tool/API | Purpose |
|----------|---------|
| **ffmpeg** | Extract audio from video, speed adjustment |
| **OpenAI Whisper** (`whisper-1`) | Speech-to-text conversion |
| **OpenAI Chat** (`gpt-4o-mini`) | Text summarization, prompt generation |
| **OpenAI Image** (`gpt-image-1`) | Text-to-image generation |
| **LangGraph CLI** | Graph service deployment |

### Architecture Design Principles

1. **Incremental complexity**: Started with a simple 2-node graph and gradually expanded to a complex 7-node graph.
2. **Cost-efficient design**: Used `quality="low"` during the sketch stage and applied `quality="high"` only for the final deliverable to optimize API costs.
3. **Modularization**: Each node is designed with a single clear responsibility, so modifications to individual nodes do not affect other nodes.
4. **Human-in-the-Loop**: Implemented a realistic workflow that reflects user judgment at critical decision points, rather than full automation.

---

## 4. Practice Exercises

### Exercise 1: Basic - Add Multi-Language Support (Difficulty: 2/5)

Modify the `transcribe_audio` node to add a `language` field to the State, allowing users to specify the video's language. Test with various languages such as Korean (`"ko"`), Japanese (`"ja"`), etc.

```python
class State(TypedDict):
    video_file: str
    language: str  # Added
    # ...
```

### Exercise 2: Intermediate - Improve Summary Quality (Difficulty: 3/5)

Currently, `textwrap.wrap` simply splits text based on character count. Improve `dispatch_summarizers` to split chunks based on token count using `tiktoken`. Also, add slight overlap between chunks to reduce context loss.

### Exercise 3: Intermediate - Thumbnail Style Selection (Difficulty: 3/5)

Add an additional `interrupt` **before** the `human_feedback` node, allowing users to first select their desired thumbnail style (photorealism, illustration, minimal, 3D rendering, etc.). Reflect the selected style in the `generate_thumbnails` prompts.

### Exercise 4: Advanced - Iterative Feedback Loop (Difficulty: 4/5)

Currently, feedback can only be provided once. Implement an **iterative feedback loop** where users can provide additional feedback if they are not satisfied with the final HD thumbnail. Design conditional edges so the `generate_hd_thumbnail` -> `interrupt` -> `generate_hd_thumbnail` cycle repeats until the user responds with "done."

### Exercise 5: Advanced - Full Pipeline Extension (Difficulty: 5/5)

Extend the pipeline by adding the following features:

1. **YouTube URL input**: Add a node that accepts a YouTube URL instead of `video_file` and automatically downloads the video using `yt-dlp` or similar tools.
2. **A/B testing mode**: Generate 2 final thumbnails for A/B testing purposes.
3. **Image post-processing**: Create a node that automatically adds a channel logo watermark to generated thumbnails using the Pillow library.
4. **Result storage**: Add a node that organizes and saves all intermediate results (summaries, prompts, images) into a JSON file.

---

## Appendix: Key API Reference

### LangGraph StateGraph API

```python
from langgraph.graph import END, START, StateGraph

# Create graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("name", function)

# Add edges (sequential)
graph_builder.add_edge("start_node", "end_node")

# Conditional edges (using a router function)
graph_builder.add_conditional_edges("start_node", router_function, ["possible_node1", "possible_node2"])

# Compile
graph = graph_builder.compile(checkpointer=checkpointer, name="graph_name")

# Execute
result = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
```

### LangGraph Send API

```python
from langgraph.types import Send

# Return a list of Send objects from a dispatcher function
def dispatcher(state: State):
    return [Send("node_name", data) for data in data_list]
```

### LangGraph interrupt / Command

```python
from langgraph.types import interrupt, Command

# Pause execution within a node
def my_node(state: State):
    answer = interrupt({"question": "Message to show the user"})
    # answer is the value passed via Command(resume=response)
    return {"field": answer["key"]}

# Resume execution
graph.invoke(Command(resume=response), config=config)
```

### OpenAI Image Generation API

```python
from openai import OpenAI
import base64

client = OpenAI()
result = client.images.generate(
    model="gpt-image-1",
    prompt="image description",
    quality="low" | "medium" | "high",
    moderation="low" | "auto",
    size="auto" | "1024x1024" | "1792x1024",
)
image_bytes = base64.b64decode(result.data[0].b64_json)
```
