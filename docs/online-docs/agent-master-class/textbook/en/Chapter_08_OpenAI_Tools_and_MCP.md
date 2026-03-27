# Chapter 8: Building a ChatGPT Clone with OpenAI Tools and MCP

---

## 1. Chapter Overview

In this chapter, we build a full-fledged ChatGPT clone application using the various **Built-in Tools** and **MCP (Model Context Protocol)** provided by the OpenAI Agents SDK. Using Streamlit as the UI framework, we start from simple text conversations and progressively extend functionality to include web search, file search, image input/generation, code execution, and external MCP server integration.

### Learning Objectives

- Implement a chat UI by integrating Streamlit with the OpenAI Agents SDK
- Understand conversation history persistence using `SQLiteSession`
- Learn how to use OpenAI built-in tools such as `WebSearchTool`, `FileSearchTool`, `ImageGenerationTool`, and `CodeInterpreterTool`
- Learn how to handle multi-modal (image) inputs
- Understand external tool integration patterns through `HostedMCPTool` and `MCPServerStdio`
- Master real-time UI update techniques using streaming events

### Technology Stack

| Technology | Role |
|------|------|
| **Streamlit** | Web-based chat UI framework |
| **OpenAI Agents SDK** | Agent, Runner, and tool management |
| **SQLiteSession** | Local storage for conversation history |
| **OpenAI API** | File upload and Vector Store management |
| **MCP (Model Context Protocol)** | Protocol for external tool server integration |

### Project Structure

```
chatgpt-clone/
├── main.py                      # Main application
├── chat-gpt-clone-memory.db     # SQLite conversation history DB
├── facts.txt                    # Sample data for File Search
└── international.png            # Image for multi-modal testing
```

---

## 2. Detailed Section Descriptions

---

### 8.0 Chat UI - Building the Streamlit Chat Interface

#### Topic and Objective

Build a basic chat UI using Streamlit and create a foundational structure that integrates the OpenAI Agents SDK's `Agent` and `Runner` to display streaming responses in real time.

#### Key Concepts

**Streamlit's `session_state`** is the core mechanism for maintaining state in a web app. Since Streamlit re-runs the entire script on every user interaction, objects like Agent and Session must be stored in `session_state` to avoid recreating them each time.

**`SQLiteSession`** is a conversation history persistence tool provided by the OpenAI Agents SDK that automatically saves and loads conversations in a SQLite database. This ensures that previous conversations are preserved even after a page refresh.

**`Runner.run_streamed()`** runs the agent in streaming mode, allowing real-time event reception as the response is being generated.

#### Code Analysis

```python
import dotenv
dotenv.load_dotenv()

import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession

# Store the Agent in session_state to maintain the same instance across reruns
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.
        """,
    )
agent = st.session_state["agent"]

# Persist conversation history to a local DB using SQLiteSession
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",               # Session identifier
        "chat-gpt-clone-memory.db",   # SQLite DB file path
    )
session = st.session_state["session"]
```

The important pattern in the code above is the `if "key" not in st.session_state` guard. Since Streamlit re-runs the entire `main.py` on every user interaction, without this guard the Agent and Session would be recreated each time and all previous state would be lost.

```python
async def run_agent(message):
    stream = Runner.run_streamed(
        agent,
        message,
        session=session,  # Pass session for automatic conversation history management
    )

    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                with st.chat_message("ai"):
                    st.write_stream(event.data.delta)
```

`stream.stream_events()` is an async iterator that delivers various events one by one as they occur during response generation. Among `raw_response_event` types, events with `response.output_text.delta` type contain the actual text fragments (deltas).

```python
# Chat input UI
prompt = st.chat_input("Write a message for your assistant")

if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))

# Sidebar: Memory reset and debugging
with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```

`st.chat_input()` is Streamlit's dedicated chat input widget, and `st.chat_message()` is a container that visually distinguishes user/AI messages. The sidebar displays a session reset button and currently stored conversation items for debugging purposes.

#### Practice Points

- Run the app with `streamlit run main.py` and try having a conversation
- Refresh the browser and verify that conversation history is preserved in the sidebar
- Clear the conversation with `session.clear_session()` and verify the behavior

---

### 8.1 Conversation History - Rendering Conversation History

#### Topic and Objective

Restore previous conversation history on page refresh and improve the streaming response display so that text appears progressively.

#### Key Concepts

In the previous section, conversation history was saved to the DB but was not displayed on screen after a page refresh. By adding a **`paint_history()`** function, we implement functionality that reads stored messages from SQLiteSession and redraws them on screen every time the app loads.

Also, previously each delta triggered a new `st.write()` call, causing messages to be duplicated across multiple lines. This is improved by using an **`st.empty()`** placeholder to accumulate and update text in a single area.

#### Code Analysis

```python
async def paint_history():
    messages = await session.get_items()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                if message["type"] == "message":
                    st.write(message["content"][0]["text"])

# Execute immediately when app loads
asyncio.run(paint_history())
```

`session.get_items()` returns the entire stored conversation history as a list. Each message is a dictionary, with different structures depending on whether the `role` field is `"user"` or `"assistant"`. User messages have `content` as a simple string, while AI responses have `content` as a list (`[{"text": "..."}]`).

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()  # Create an empty placeholder
        response = ""
        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta        # Accumulate text
                    text_placeholder.write(response)    # Update in the same location
```

**The role of `st.empty()`**: In Streamlit, `st.empty()` creates an empty container that can be filled with content later. When `.write()` is called on this container, it **replaces** the previous content, creating a natural effect where streaming text progressively grows in one place.

#### Practice Points

- Have several conversations, then refresh the page to verify that history is restored
- Compare the difference when using `st.write()` directly instead of `st.empty()`
- Analyze the message dictionary structure through the `get_items()` output in the sidebar

---

### 8.2 Web Search Tool - Adding Web Search Capability

#### Topic and Objective

Add `WebSearchTool` to the agent to provide real-time web search functionality, and build a status management system that displays search progress in real time on the UI.

#### Key Concepts

**`WebSearchTool`** is a built-in tool provided by the OpenAI Agents SDK that allows the agent to search the web for the latest information not present in its training data. It is important to specify tool usage guidelines in the agent's `instructions` to indicate when web search should be performed.

**Status container (`st.status`)** is a progress indicator widget provided by Streamlit that visually informs the user about the tool execution process.

#### Code Analysis

```python
from agents import Agent, Runner, SQLiteSession, WebSearchTool

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data. Use this tool when the users asks about current
              or future events, when you think you don't know the answer, try
              searching for it in the web first.
        """,
        tools=[
            WebSearchTool(),  # Register web search tool
        ],
    )
```

The tool usage conditions are specified in the agent's `instructions`. The guidance says to try web search first "when asked about current or future events" or "when the answer is unknown." This is a prompt engineering technique that improves tool selection accuracy.

```python
def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": (
            "🔍 Starting web search...",
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 Web search in progress...",
            "running",
        ),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)
```

The `update_status()` function acts as an **event dispatcher** that updates the UI status display based on the streaming event type. Web search-related events are divided into three stages:

1. `in_progress` - Search started
2. `searching` - Search in progress
3. `completed` - Search completed

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)  # Status container
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(agent, message, session=session)

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)  # Update status

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response)
```

When restoring conversation history, web search call records are also displayed:

```python
if "type" in message and message["type"] == "web_search_call":
    with st.chat_message("ai"):
        st.write("🔍 Searched the web...")
```

#### Practice Points

- Ask real-time questions like "What's the weather like today?" and verify that web search is triggered
- Observe the status container transition process (start -> in progress -> complete)
- Verify that web search is not triggered when asking general knowledge questions that exist in the training data

---

### 8.3 File Search Tool - File Search Tool and Vector Store

#### Topic and Objective

Add the ability to search uploaded file contents using `FileSearchTool` and OpenAI's Vector Store. Also enable users to upload text files directly through Streamlit's file upload feature.

#### Key Concepts

**Vector Store** is a vector database hosted by OpenAI that automatically embeds uploaded text files, enabling semantic-based search. `FileSearchTool` is the tool that allows the agent to search this Vector Store for relevant information.

**The file upload workflow** consists of two steps:
1. Upload the file to OpenAI with `client.files.create()`
2. Link the file to a Vector Store with `client.vector_stores.files.create()`

#### Code Analysis

```python
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool

client = OpenAI()

# Vector Store ID (pre-created via OpenAI dashboard or API)
VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: ...
            - File Search Tool: Use this tool when the user asks a question
              about facts related to themselves. Or when they ask questions
              about specific files.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],  # Target Vector Store for search
                max_num_results=3,                    # Maximum number of search results
            ),
        ],
    )
```

`FileSearchTool` specifies the target Vector Store via `vector_store_ids` and limits the maximum number of results via `max_num_results`.

File upload and Vector Store linking code:

```python
# Enable file attachment in chat input
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt"],
)

if prompt:
    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    # Step 1: Upload file to OpenAI
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching file...")

                    # Step 2: Link file to Vector Store
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded", state="complete")
```

The sample data file (`facts.txt`) used in this project contains a hypothetical investment portfolio and spending records. Once uploaded, the agent can answer personal information questions like "How many shares of Apple do I have?"

Note the `replace("$", "\$")` applied to responses containing the `$` symbol to prevent Streamlit's LaTeX rendering issue:

```python
st.write(message["content"][0]["text"].replace("$", "\$"))
```

#### Practice Points

- Upload `facts.txt` and ask questions like "What's my total portfolio value?"
- Create a Vector Store directly from the OpenAI dashboard and replace the ID
- Compare and observe when file search vs. web search is triggered

---

### 8.4 Multi Modal Agent - Multi-Modal Image Input

#### Topic and Objective

Add multi-modal capabilities so the agent can receive and analyze images as input. When a user uploads an image, encode it in Base64, store it in the session, and enable the agent to understand it.

#### Key Concepts

**Multi-Modal** refers to the ability to process multiple types of input beyond just text, such as images and audio. OpenAI's GPT-4 series models can receive images as input and analyze and describe their contents.

**Base64 encoding** is used to pass images to the API. Image byte data is converted to a Base64 string, then passed as a Data URI in the format `data:image/png;base64,...`.

#### Code Analysis

```python
import base64

# Allow image files in chat input
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=[
        "txt",
        "jpg",
        "jpeg",
        "png",
    ],
)
```

Image upload handling:

```python
elif file.type.startswith("image/"):
    with st.status("⏳ Uploading image...") as status:
        file_bytes = file.getvalue()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{file.type};base64,{base64_data}"

        # Save image as a user message in the session
        asyncio.run(
            session.add_items(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": data_uri,
                            }
                        ],
                    }
                ]
            )
        )
        status.update(label="✅ Image uploaded", state="complete")
    with st.chat_message("human"):
        st.image(data_uri)
```

The key is adding the image directly to the conversation history via `session.add_items()`. It uses the `input_image` type and `image_url` field required by the OpenAI API format. `detail: "auto"` lets the model automatically determine the image resolution.

Modifying `paint_history()` to also display images when restoring conversation history:

```python
if message["role"] == "user":
    content = message["content"]
    if isinstance(content, str):
        st.write(content)           # Text message
    elif isinstance(content, list):
        for part in content:
            if "image_url" in part:
                st.image(part["image_url"])  # Image message
```

The user message's `content` can be either a string (plain text) or a list (multi-modal). The `isinstance()` check handles both cases.

#### Practice Points

- Upload a chart or graph image and ask "What do you see in this image?"
- Upload an image, then ask follow-up text questions to check if the agent remembers the image context
- Analyze the structure of the Base64-encoded Data URI

---

### 8.5 Image Generation Tool - Image Generation Tool

#### Topic and Objective

Add `ImageGenerationTool` to enable the agent to generate images based on user requests. Also implement a technique to display intermediate results (partial images) in real time during the generation process.

#### Key Concepts

**`ImageGenerationTool`** wraps OpenAI's image generation API (DALL-E) so the agent can call it as a tool. Through the `partial_images` setting, low-resolution preview images can be received during generation, allowing visual progress feedback to the user.

#### Code Analysis

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
)

# Add ImageGenerationTool to the agent's tool list
ImageGenerationTool(
    tool_config={
        "type": "image_generation",
        "quality": "high",           # High-quality image generation
        "output_format": "jpeg",     # Output format
        "partial_images": 1,         # Number of intermediate preview images
    }
),
```

Key `tool_config` options:
- `quality`: `"high"` or `"standard"`. High quality is more refined but takes longer to generate
- `output_format`: `"jpeg"` or `"png"`
- `partial_images`: Number of intermediate images to receive during generation. Setting to 1 or more enables progressive rendering effects

Handling image events in streaming:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        image_placeholder = st.empty()  # Add image placeholder
        response = ""

        # ... streaming loop ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # Display intermediate image generation results
                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)

                elif event.data.type == "response.completed":
                    image_placeholder.empty()
                    text_placeholder.empty()
```

Intermediate images (`partial_image`) are delivered as Base64-encoded data, so they are decoded with `base64.b64decode()` and displayed with `st.image()`.

Displaying generated images when restoring conversation history:

```python
elif message_type == "image_generation_call":
    image = base64.b64decode(message["result"])
    with st.chat_message("ai"):
        st.image(image)
```

Image generation-related status messages are also added:

```python
"response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
"response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
```

#### Practice Points

- Test image generation with requests like "Draw a picture of a cat eating pizza in space"
- Toggle `partial_images` between 0 and 1 and compare the preview effect
- Verify that generated images are saved in conversation history and displayed after refresh

---

### 8.6 Code Interpreter Tool - Code Execution Tool

#### Topic and Objective

Add `CodeInterpreterTool` to enable the agent to write and execute Python code for calculations, data analysis, chart generation, and more.

#### Key Concepts

**`CodeInterpreterTool`** is a tool that allows Python code to be executed in a sandboxed environment hosted by OpenAI. When the agent writes code, it is executed in a secure container and the results are returned. It is useful for mathematical calculations, data analysis, visualization, and more.

The `container` setting with `"type": "auto"` lets OpenAI automatically select the appropriate execution environment.

#### Code Analysis

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool, CodeInterpreterTool,
)

CodeInterpreterTool(
    tool_config={
        "type": "code_interpreter",
        "container": {
            "type": "auto",      # Automatic container selection
        },
    }
),
```

Streaming processing of the code execution process:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        code_placeholder = st.empty()    # Placeholder for code display
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""
        code_response = ""

        # Store placeholders in session_state (for cleanup in next run)
        st.session_state["code_placeholder"] = code_placeholder
        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        # ... streaming loop ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # Display code writing process in real time
                if event.data.type == "response.code_interpreter_call_code.delta":
                    code_response += event.data.delta
                    code_placeholder.code(code_response)  # Display as code block
```

`st.code()` displays a code block with syntax highlighting applied. Being able to watch the code being written in real time enhances the user experience.

Code to clean up previous placeholders on the next message run:

```python
if prompt:
    # Clean up previous placeholders
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()
```

Without this cleanup code, streaming placeholders from the previous response would remain on screen and cause duplicate display with messages restored by `paint_history()`.

Code execution-related status messages:

```python
"response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
"response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
```

#### Practice Points

- Try code execution requests like "Calculate the first 20 terms of the Fibonacci sequence"
- Also try visualization requests like "Draw a sine function graph"
- Observe the real-time code writing process

---

### 8.7 Hosted MCP Tool - Hosted MCP Tool Integration

#### Topic and Objective

Use **HostedMCPTool** to connect to an external hosted MCP server (Context7) and add the ability to search software project documentation.

#### Key Concepts

**MCP (Model Context Protocol)** is an open protocol for AI models to interact with external tools and data sources. Through MCP, agents can leverage various capabilities provided by third parties beyond their built-in tools.

**HostedMCPTool** connects to MCP servers published on the internet via HTTP. It is simple to set up since you only need to know the server URL.

**Context7** is an MCP server that provides up-to-date documentation for software projects, enabling agents to search official documentation for specific libraries or frameworks.

#### Code Analysis

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
    CodeInterpreterTool, HostedMCPTool,
)

HostedMCPTool(
    tool_config={
        "server_url": "https://mcp.context7.com/mcp",  # MCP server URL
        "type": "mcp",
        "server_label": "Context7",                      # Display label
        "server_description": "Use this to get the docs from software projects.",
        "require_approval": "never",                     # Auto-approve (no user confirmation needed)
    }
),
```

Key `tool_config` fields:
- `server_url`: The MCP server endpoint URL
- `server_label`: Server name displayed in the UI
- `server_description`: Description used by the agent to determine when to use this tool
- `require_approval`: Setting to `"never"` automatically calls the tool without user approval

MCP-related conversation history restoration:

```python
elif message_type == "mcp_list_tools":
    with st.chat_message("ai"):
        st.write(f"Listed {message['server_label']}'s tools")
elif message_type == "mcp_call":
    with st.chat_message("ai"):
        st.write(
            f"Called {message['server_label']}'s {message['name']} "
            f"with args {message['arguments']}"
        )
```

MCP calls occur in two stages:
1. **`mcp_list_tools`**: Query the list of available tools from the server
2. **`mcp_call`**: Actually call a specific tool with arguments

MCP-related status messages:

```python
"response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
"response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
"response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
"response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
"response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
"response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
```

#### Practice Points

- Test the Context7 MCP with questions like "Tell me how to use Streamlit's st.chat_input"
- Observe whether `mcp_list_tools` and `mcp_call` occur in sequence during MCP calls
- Change `require_approval` to `"always"` and observe the behavioral difference

---

### 8.8 Local MCP Server - Local MCP Server Integration

#### Topic and Objective

Use **`MCPServerStdio`** to connect to a locally running MCP server (Yahoo Finance). Through this, understand the differences between hosted MCP and local MCP, and refactor the agent creation structure to use the async context manager pattern.

#### Key Concepts

**`MCPServerStdio`** runs an MCP server as a local process and communicates via standard input/output (stdin/stdout). It uses `uvx` (a Python package runner) to directly execute MCP server packages.

**Differences between Hosted MCP and Local MCP**:
| Property | Hosted MCP | Local MCP |
|------|-----------|-----------|
| Execution location | Remote server | Local machine |
| Connection method | HTTP | stdin/stdout |
| Configuration | Specify URL only | Specify execution command |
| Lifecycle | Always available | Requires process start/stop |

Local MCP servers must have their lifecycle managed with `async with` statements (async context managers). Because of this, the agent creation location moves from `session_state` initialization to inside the `run_agent()` function.

#### Code Analysis

```python
from agents.mcp.server import MCPServerStdio

async def run_agent(message):
    # Define local MCP server
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",                    # Command to execute
            "args": ["mcp-yahoo-finance"],       # Package name
        },
        cache_tools_list=True,  # Cache tool list for performance optimization
    )

    # Manage server lifecycle with async context manager
    async with yfinance_server:

        # Create Agent inside the context (MCP server must be active)
        agent = Agent(
            mcp_servers=[
                yfinance_server,       # Connect local MCP server
            ],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.
        ...
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID],
                    max_num_results=3,
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        # Now use the agent for streaming execution
        with st.chat_message("ai"):
            # ... same streaming logic as before ...
```

**The key structural change**: Until the previous section, the Agent was created once in `st.session_state` and reused. However, since local MCP servers are only valid inside an `async with` block, the Agent must also be recreated inside that block each time. This has a slight performance overhead but is a necessary trade-off for reliable lifecycle management of local MCP servers.

`cache_tools_list=True` caches the MCP server's tool list so that the tool list doesn't need to be queried every time. This is useful for servers whose tool list doesn't change frequently.

#### Practice Points

- Test the Yahoo Finance MCP with finance-related questions like "Tell me the current price of Apple stock"
- Run `uvx mcp-yahoo-finance` directly in the terminal and observe MCP server behavior
- Check what error occurs when creating the Agent outside the `async with` block

---

### 8.9 Conclusions - Adding a Second Local MCP Server

#### Topic and Objective

Add a second local MCP server (timezone server) to learn the pattern of using multiple MCP servers simultaneously.

#### Key Concepts

Python's `async with` statement can manage **multiple context managers simultaneously** by separating them with commas (`,`). This allows running multiple local MCP servers at the same time and connecting them all to the same Agent.

#### Code Analysis

```python
async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-yahoo-finance"],
        },
        cache_tools_list=True,
    )

    # Second local MCP server: provides timezone information
    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    # Manage both servers simultaneously as context managers
    async with yfinance_server, timezone_server:

        agent = Agent(
            mcp_servers=[
                yfinance_server,
                timezone_server,      # Add second MCP server
            ],
            name="ChatGPT Clone",
            # ... rest of configuration is the same ...
        )
```

`mcp-server-time` can specify a default timezone with the `--local-timezone` argument. The agent can use this server to perform tasks like getting the current time in a specific timezone or converting between timezones.

The `async with yfinance_server, timezone_server:` syntax starts both servers simultaneously and cleanly shuts them both down when the block ends. Even if one error occurs, the remaining servers are properly cleaned up.

#### Practice Points

- Test the timezone MCP with questions like "What time is it in New York?"
- Try compound questions that use multiple MCP servers in a single conversation (e.g., "Tell me the Apple stock price and the current time in New York")
- Find and add new MCP server packages

---

## 3. Chapter Key Summary

### Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │chat_input│  │chat_msg  │  │  sidebar (debug)   │   │
│  └────┬─────┘  └──────────┘  └────────────────────┘   │
│       │                                                 │
│       v                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              run_agent()                         │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  async with MCP_Server_1, MCP_Server_2  │    │   │
│  │  │  ┌───────────────────────────────────┐  │    │   │
│  │  │  │           Agent                   │  │    │   │
│  │  │  │  ┌──────────┐ ┌──────────────┐   │  │    │   │
│  │  │  │  │WebSearch │ │ FileSearch   │   │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │ImageGen  │ │CodeInterpreter│  │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │HostedMCP │ │ Local MCP x2 │  │  │    │   │
│  │  │  │  └──────────┘ └──────────────┘   │  │    │   │
│  │  │  └───────────────────────────────────┘  │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│              ┌─────────────────┐                       │
│              │  SQLiteSession   │                       │
│              │  (History Store) │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### Tool Summary

| Tool | Purpose | Key Event Type |
|------|------|-----------------|
| **WebSearchTool** | Real-time web information search | `response.web_search_call.*` |
| **FileSearchTool** | Vector Store-based file content search | `response.file_search_call.*` |
| **ImageGenerationTool** | DALL-E image generation | `response.image_generation_call.*` |
| **CodeInterpreterTool** | Python code execution | `response.code_interpreter_call*` |
| **HostedMCPTool** | Remote MCP server integration | `response.mcp_call.*`, `response.mcp_list_tools.*` |
| **MCPServerStdio** | Local MCP server integration | `response.mcp_call.*`, `response.mcp_list_tools.*` |

### Key Concepts Summary

1. **`st.session_state`**: Streamlit's state management mechanism. Maintains data across script reruns.

2. **`st.empty()`**: A placeholder whose content can be replaced later. Essential for streaming text display.

3. **`st.status()`**: A widget that visually displays task progress. Supports `running`/`complete` states.

4. **`SQLiteSession`**: Automatically saves/restores conversation history to a SQLite DB. Passed to Runner via the `session` parameter.

5. **Streaming event pattern**: Determine the event type via `raw_response_event` > `event.data.type` and perform appropriate UI updates for each tool's events.

6. **`async with` context manager**: Safely manages the lifecycle (start/stop) of local MCP servers. Multiple servers can be managed simultaneously by connecting them with commas.

7. **Vector Store**: A vector DB hosted by OpenAI that automatically embeds uploaded files, enabling semantic-based search.

---

## 4. Practice Exercises

### Exercise 1: Basic - Tool Usage Logging System

Extend the `update_status()` function to create a system that logs all tool calls with timestamps to a log file. Implement it so that logs can be viewed in the sidebar.

**Hints**:
- Use Python's `logging` module
- Display log contents in `st.sidebar`
- Record each tool call's start time, end time, and duration

### Exercise 2: Intermediate - Adding a New MCP Server

Find and add `mcp-server-fetch` (for fetching web page content) or another MCP server package to the project. Connect it as a local MCP server and modify the `instructions` so the agent can use the tool appropriately.

**Hints**:
- Search PyPI for MCP server packages that can be run with `uvx`
- Set the appropriate command and arguments in `MCPServerStdio`'s `params`
- Add the new server to the `async with` statement and include it in the `mcp_servers` list

### Exercise 3: Intermediate - Multiple Conversation Session Management

Currently only one conversation session is supported. Implement functionality to create and switch between multiple conversation sessions from the sidebar.

**Hints**:
- Dynamically change the first argument (session ID) of `SQLiteSession`
- Display the session list with `st.sidebar.selectbox()`
- Assign a unique ID when creating a new session

### Exercise 4: Advanced - Custom Tool Integration

Use the `@function_tool` decorator to create custom Python functions as tools and register them alongside existing built-in tools in the agent. For example, you could create tools that explore the local file system or perform simple calculations.

**Hints**:
- Apply the `@function_tool` pattern learned in previous chapters
- The tool's docstring affects the agent's tool selection
- Specify usage conditions for the new tool in the `instructions`

### Exercise 5: Advanced - Audio Input Multi-Modal Extension

Extend the current image-only multi-modal functionality to also support audio files. Implement this by using OpenAI's Whisper API to convert speech to text before passing it to the agent.

**Hints**:
- Add audio formats to `st.chat_input`'s `file_type`
- Convert speech to text with `client.audio.transcriptions.create()`
- Pass the converted text to the agent as a regular message

---

## Appendix: Final Complete Code (main.py)

Below is the final `main.py` with all Chapter 8 features integrated:

```python
import dotenv

dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    HostedMCPTool,
)
from agents.mcp.server import MCPServerStdio

client = OpenAI()

VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"


if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )
session = st.session_state["session"]


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)
            elif message_type == "code_interpreter_call":
                with st.chat_message("ai"):
                    st.code(message["code"])
            elif message_type == "mcp_list_tools":
                with st.chat_message("ai"):
                    st.write(f"Listed {message['server_label']}'s tools")
            elif message_type == "mcp_call":
                with st.chat_message("ai"):
                    st.write(
                        f"Called {message['server_label']}'s {message['name']} "
                        f"with args {message['arguments']}"
                    )


asyncio.run(paint_history())


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
        "response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
        "response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
        "response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
        "response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
        "response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
        "response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
        "response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
        "response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
        "response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={"command": "uvx", "args": ["mcp-yahoo-finance"]},
        cache_tools_list=True,
    )

    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    async with yfinance_server, timezone_server:
        agent = Agent(
            mcp_servers=[yfinance_server, timezone_server],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data.
            - File Search Tool: Use this tool when the user asks a question about
              facts related to themselves.
            - Code Interpreter Tool: Use this tool when you need to write and run
              code to answer the user's question.
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID], max_num_results=3
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        with st.chat_message("ai"):
            status_container = st.status("⏳", expanded=False)
            code_placeholder = st.empty()
            image_placeholder = st.empty()
            text_placeholder = st.empty()
            response = ""
            code_response = ""

            st.session_state["code_placeholder"] = code_placeholder
            st.session_state["image_placeholder"] = image_placeholder
            st.session_state["text_placeholder"] = text_placeholder

            stream = Runner.run_streamed(agent, message, session=session)

            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    update_status(status_container, event.data.type)

                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                    if event.data.type == "response.code_interpreter_call_code.delta":
                        code_response += event.data.delta
                        code_placeholder.code(code_response)

                    elif event.data.type == "response.image_generation_call.partial_image":
                        image = base64.b64decode(event.data.partial_image_b64)
                        image_placeholder.image(image)

prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
)

if prompt:
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()), purpose="user_data"
                    )
                    status.update(label="⏳ Attaching file...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID, file_id=uploaded_file.id
                    )
                    status.update(label="✅ File uploaded", state="complete")
        elif file.type.startswith("image/"):
            with st.status("⏳ Uploading image...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")
            with st.chat_message("human"):
                st.image(data_uri)

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```
