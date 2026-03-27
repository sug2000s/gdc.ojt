# Chapter 21: AI Agent Deployment

## Chapter Overview

In this chapter, we will learn the entire process of **deploying an AI agent built with the OpenAI Agents SDK to a real production environment**. Going beyond simply running an agent locally, we cover wrapping it as a REST API using the FastAPI web framework, managing conversation state with OpenAI's Conversations API, handling synchronous/streaming responses, and finally deploying to the Railway cloud platform.

### Learning Objectives

- Understand how to wrap an AI agent as a REST API using FastAPI
- Master conversation state (context) management using the OpenAI Conversations API
- Learn the differences between synchronous (Sync) and streaming responses, and how to implement them
- Practice cloud deployment using the Railway platform

### Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13 | Programming language |
| FastAPI | 0.118.3 | Web framework |
| OpenAI Agents SDK | 0.3.3 | AI agent framework |
| Uvicorn | 0.37.0 | ASGI server |
| python-dotenv | 1.1.1 | Environment variable management |
| Railway | - | Cloud deployment platform |

---

## 21.0 Introduction - Project Initial Setup

### Topic and Objectives

Create the basic skeleton of a deployment project. Initialize a new Python project using `uv` (Python package manager) and set up the required dependencies.

### Core Concept Explanation

#### Project Structure

In this chapter, we create an independent directory called `deployment/`, separate from the existing masterclass project. This is to design it as a standalone deployable application.

```
deployment/
├── .python-version    # Python version specification (3.13)
├── README.md          # Project description
├── main.py            # Main application file
└── pyproject.toml     # Project metadata and dependencies
```

#### pyproject.toml - Dependency Management

`pyproject.toml` is the standard configuration file for modern Python projects. This file declares the project's metadata and dependencies.

```toml
[project]
name = "deployment"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi==0.118.3",
    "openai-agents==0.3.3",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

The role of each dependency:

- **`fastapi`**: A high-performance Python web framework. Provides automatic API documentation generation, type validation, and more.
- **`openai-agents`**: OpenAI's official agent SDK. Provides core classes needed for agent execution, such as Agent and Runner.
- **`python-dotenv`**: Loads environment variables from `.env` files. Used to separate sensitive information like API keys from the code.
- **`uvicorn`**: An ASGI server. The server that enables your FastAPI application to actually receive HTTP requests.

#### Initial main.py

```python
def main():
    print("Hello from deployment!")


if __name__ == "__main__":
    main()
```

At this point, `main.py` is still just skeleton code. Starting from the next section, we will transform it into a full FastAPI application.

### Practice Points

1. Create a new project with the `uv init deployment` command.
2. Add dependencies with `uv add fastapi openai-agents python-dotenv uvicorn`.
3. Verify that `3.13` is set in the `.python-version` file.

---

## 21.1 Conversations API - Building the Conversation Management API

### Topic and Objectives

Build a REST API using FastAPI that can manage conversations with the AI agent. Create endpoints to establish conversation sessions and add messages to each conversation using OpenAI's **Conversations API**.

### Core Concept Explanation

#### What is the OpenAI Conversations API?

The Conversations API is a conversation state management feature provided by OpenAI. Previously, you had to manage conversation history yourself, but with the Conversations API, OpenAI maintains the conversation state on the server side.

Core flow:
1. Creating a new conversation session with `client.conversations.create()` returns a unique `conversation_id`.
2. When you subsequently pass this `conversation_id` during agent execution, OpenAI automatically maintains the previous conversation context.

Advantages of this approach:
- **Serverless compatible**: No need to store state on the server, so conversations persist even if the server restarts.
- **Simple implementation**: Eliminates complex logic for directly managing conversation history arrays.
- **Scalability**: The same conversation can be continued across multiple server instances.

#### FastAPI Application Structure

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner
```

**Important note**: `load_dotenv()` is called **before** `from agents import ...`. This is because the `agents` module references environment variables (especially `OPENAI_API_KEY`) when it is imported. If the order is reversed, the API key cannot be found and an error will occur.

#### Agent Definition

```python
agent = Agent(
    name="Assistant",
    instructions="You help users with their questions."
)
```

The agent is created only once at the module level. There is no need to create a new one for each request. `instructions` serves as the agent's system prompt.

#### FastAPI App and OpenAI Client Initialization

```python
app = FastAPI()
client = AsyncOpenAI()
```

`AsyncOpenAI()` is an asynchronous OpenAI client. Since FastAPI is an asynchronous framework, using an asynchronous client is appropriate for performance.

#### Conversation Creation Endpoint

```python
class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }
```

Key points of this code:

1. **Pydantic BaseModel**: `CreateConversationResponse` defines the response schema. FastAPI automatically serializes this to JSON and reflects it in the Swagger documentation.
2. **`client.conversations.create()`**: Calls the OpenAI API to create a new conversation session. The `.id` field of the return value contains a unique ID starting with `conv_`.
3. **Asynchronous processing**: Uses the `await` keyword to asynchronously wait for the API call to complete.

#### Message Endpoint (Skeleton)

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(conversation_id: str):
    pass
```

At this point, the message endpoint is not yet implemented. The URL path includes `{conversation_id}`, creating a structure that allows sending messages to a specific conversation.

### Code Analysis - Overall Flow

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │                            │── conversations.create()─>│
   │                            │<── conversation object ───│
   │<── { conversation_id } ────│                          │
```

### Practice Points

1. Start the development server with the `uvicorn main:app --reload` command.
2. Access `http://127.0.0.1:8000/docs` in your browser to see FastAPI's auto-generated Swagger documentation.
3. Call the `POST /conversations` endpoint and verify that a `conversation_id` is returned correctly.

---

## 21.2 Sync Responses - Implementing Synchronous Responses

### Topic and Objectives

Complete the endpoint that sends a message to a conversation and receives the agent's response **synchronously**. Use `Runner.run()` to execute the agent and deliver the complete response to the client all at once after the entire response is generated.

### Core Concept Explanation

#### Synchronous Response vs Streaming Response

| Characteristic | Synchronous Response | Streaming Response |
|----------------|---------------------|--------------------|
| Response method | Sends all at once after complete response | Sends in real-time token by token |
| User experience | Wait time until response | Text appears immediately |
| Implementation difficulty | Relatively simple | Requires event stream handling |
| Suitable for | Backend-to-backend communication, short responses | User-facing UI, long responses |

In this section, we first implement synchronous responses.

#### Request/Response Model Definition

```python
class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str
```

Strictly define the input/output schemas using Pydantic `BaseModel`.

- **`CreateMessageInput`**: The request body sent by the client. The user's question is contained in the `question` field.
- **`CreateMessageOutput`**: The response returned by the server. The agent's answer is contained in the `answer` field.

Based on these models, FastAPI:
- Automatically parses the JSON request body and validates types.
- Automatically returns a 422 Validation Error for malformed requests.

#### Message Processing Endpoint

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }
```

Key points of this code:

1. **`conversation_id` path parameter**: Extracted from the URL and passed as a function argument. This identifies which conversation to send the message to.
2. **`message_input` body parameter**: FastAPI automatically converts the JSON request body into a `CreateMessageInput` object.
3. **`Runner.run()`**: The core method for executing the agent.
   - `starting_agent`: The agent object to execute
   - `input`: The user's question text
   - `conversation_id`: The conversation ID from the OpenAI Conversations API. This is **the key to maintaining conversation context**.
4. **`answer.final_output`**: Extracts the agent's final text output from the return value of `Runner.run()`.

#### API Testing (api.http)

```http
POST http://127.0.0.1:8000/conversations

###

POST http://127.0.0.1:8000/conversations/conv_68ecdf11ff6081969cc4e8e9d126c015082054e6371dc260/message
Content-Type: application/json

{
    "question": "What is the first question i asked you?"
}
```

The `api.http` file is an HTTP request test file used with VS Code's REST Client extension, etc. Requests are separated by `###`, and each request can be executed individually.

In the test above, the question "What is the first question i asked you?" is designed to **verify conversation context persistence**. Since previous conversation content is maintained through the `conversation_id`, the agent can remember and respond about previously received questions.

### Code Analysis - Overall Flow

```
Client                       FastAPI Server              OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │<── { conversation_id } ────│                          │
   │                            │                          │
   │── POST /conversations/     │                          │
   │   {id}/message ───────────>│                          │
   │   { question: "..." }      │── Runner.run() ─────────>│
   │                            │   (includes conversation_id) │
   │                            │<── completed response ────│
   │<── { answer: "..." } ──────│                          │
```

### Practice Points

1. After creating a conversation, use the returned `conversation_id` to send multiple messages.
2. Say "My name is [name]", then ask "What is my name?" to verify that conversation context is maintained.
3. Verify that conversations are independently maintained when using different `conversation_id` values.

---

## 21.3 StreamingResponse - Implementing Streaming Responses

### Topic and Objectives

Implement an endpoint that delivers the agent's response via **real-time streaming**. This allows users to see the agent's answer being generated in real-time. We implement two streaming methods (text only / all events).

### Core Concept Explanation

#### The Need for Streaming Responses

Synchronous responses (`Runner.run()`) require the client to wait until the entire answer is generated. For long answers, this can take several seconds to tens of seconds, resulting in a poor user experience.

Streaming responses (`Runner.run_streamed()`) send tokens to the client as they are generated. This works on the same principle as text appearing one character at a time in the ChatGPT web interface.

#### FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse
```

FastAPI's `StreamingResponse` accepts a generator function and sends data to the client in chunks. It sends data incrementally while maintaining the HTTP connection.

#### Method 1: Streaming Text Deltas Only

```python
@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")
```

Let's analyze this code step by step:

**Step 1 - Calling `Runner.run_streamed()`**:
```python
events = Runner.run_streamed(
    starting_agent=agent,
    input=message_input.question,
    conversation_id=conversation_id,
)
```
Instead of `Runner.run()`, we use `Runner.run_streamed()`. This method does not return the result all at once, but returns an event stream object.

**Step 2 - Event Filtering**:
```python
async for event in events.stream_events():
    if (
        event.type == "raw_response_event"
        and event.data.type == "response.output_text.delta"
    ):
        yield event.data.delta
```

`stream_events()` generates various types of events. Here we filter using two conditions:
- `event.type == "raw_response_event"`: Raw events delivered directly from the OpenAI API
- `event.data.type == "response.output_text.delta"`: Events corresponding to text output **deltas**

`yield` is Python's async generator syntax. It produces data one piece at a time and passes it to `StreamingResponse`.

**Step 3 - Returning StreamingResponse**:
```python
return StreamingResponse(event_generator(), media_type="text/plain")
```
Setting `media_type="text/plain"` streams as pure text. The client maintains the connection and receives text chunks sequentially.

#### Method 2: Streaming All Events

```python
@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

This endpoint streams not just text deltas, but **all raw_response_events** in JSON format.

Key differences:
- The filter condition is reduced to only `event.type == "raw_response_event"` (text delta condition removed).
- `event.data.to_json()` converts the entire event to a JSON string.
- A `\n` (newline) is appended after each event so the client can distinguish between events.

This approach is useful when more fine-grained control is needed on the frontend. For example, you can receive tool call events, agent switch events, etc., and reflect them in the UI.

#### Comparison of the Two Methods

| Characteristic | `/message-stream` | `/message-stream-all` |
|----------------|-------------------|----------------------|
| Content sent | Text fragments only | All events (JSON) |
| Data format | Pure text | JSON (newline-separated) |
| Data volume | Small | Large |
| Suitable for | Simple chat UI | Advanced UI (tool execution display, etc.) |

#### Streaming Test with curl

```bash
curl -N -X POST http://127.0.0.1:8000/conversations/{conv_id}/message-stream \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the size of the great wall of china?"}'
```

The `-N` flag in `curl` disables output buffering. Without this option, curl would accumulate data in a buffer and output it all at once, making it impossible to observe the streaming effect.

### Practice Points

1. Call the `/message-stream` endpoint with `curl -N` and verify that text is output in real-time.
2. Call the `/message-stream-all` endpoint and observe what types of events are delivered.
3. Compare the perceived speed difference between synchronous responses (`/message`) and streaming responses (`/message-stream`).
4. Analyze what event types exist in the streamed JSON events besides `response.output_text.delta`.

---

## 21.4 Deployment - Railway Cloud Deployment

### Topic and Objectives

Deploy the completed AI agent API to the **Railway** cloud platform to make it an actual service accessible from the internet. We cover writing deployment configuration files, environment variable management, and security settings.

### Core Concept Explanation

#### What is Railway?

Railway is a developer-friendly cloud deployment platform. Connect a Git repository and it automatically builds and deploys, making it easy to manage environment variables, check logs, configure domains, and more.

Advantages of Railway:
- Git push-based automatic deployment (CI/CD)
- Automatic builds using NIXPACKS (no Dockerfile required)
- Free tier available
- Simple environment variable management

#### Adding a Health Check Endpoint

```python
@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }
```

A simple GET endpoint is added to the root path (`/`). This serves multiple purposes:
- **Health Check**: Used to verify the server is operating normally. Cloud platforms like Railway periodically call this endpoint to check service status.
- **Quick Verification**: Allows you to immediately check if the service is working when accessing the deployed URL in a browser.
- **Synchronous function**: Defined with `def` instead of `async def`. Since there are no external API calls, async is unnecessary.

#### railway.json - Deployment Configuration

```json
{
    "$schema": "https://railway.app/railway.schema.json",
    "build": {
        "builder": "NIXPACKS"
    },
    "deploy": {
        "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
    }
}
```

Analysis of each configuration item:

- **`$schema`**: JSON schema URL. Supports autocomplete and validation in IDEs.
- **`build.builder: "NIXPACKS"`**: NIXPACKS is a tool that analyzes source code and automatically configures the build environment. When it detects `pyproject.toml`, it automatically sets up the Python environment and installs dependencies. No need to write a Dockerfile manually.
- **`deploy.startCommand`**: The command to run the application after deployment.
  - `uvicorn main:app`: Runs the `app` object from `main.py` as an ASGI server
  - `--host 0.0.0.0`: Allows connections from all network interfaces (required in container environments)
  - `--port $PORT`: Uses the dynamically assigned port from Railway (`$PORT` environment variable)

#### .gitignore - Security and Cleanup

```
.env
.venv
__pycache__
```

Files that should not be included in the Git repository during deployment:
- **`.env`**: File containing sensitive environment variables like API keys. Must **never** be committed to Git.
- **`.venv`**: Python virtual environment directory. Created separately in the deployment environment.
- **`__pycache__`**: Python bytecode cache. Unnecessary files.

#### Post-Deployment URL Change

```http
POST https://my-agent-deployment-production.up.railway.app/conversations
```

The URL changes from `http://127.0.0.1:8000` during local development to `https://my-agent-deployment-production.up.railway.app` after Railway deployment. Railway automatically provides HTTPS and assigns a subdomain based on the project name.

### Deployment Procedure Summary

```
1. Create a Railway account and project
2. Connect GitHub repository
3. Set environment variables (OPENAI_API_KEY)
4. Automatic build and deployment according to railway.json configuration
5. Test the API with the assigned URL
```

### Final Project Structure

```
deployment/
├── .gitignore         # Git exclusion file list
├── .python-version    # Python 3.13
├── .env               # Environment variables (Git excluded)
├── README.md          # Project description
├── api.http           # API test file
├── main.py            # Main application (FastAPI + Agent)
├── pyproject.toml     # Dependency management
├── railway.json       # Railway deployment configuration
└── uv.lock            # Dependency lock file
```

### Final main.py Complete Code

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner


agent = Agent(
    name="Assistant",
    instructions="You help users with their questions.",
)

app = FastAPI()
client = AsyncOpenAI()


class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }


class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str


@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }


@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### Practice Points

1. Sign up at Railway (https://railway.app) and create a new project.
2. Connect your GitHub repository and set the `OPENAI_API_KEY` environment variable in the Railway dashboard.
3. Call `POST /conversations` with the deployed URL to create a conversation and send messages.
4. Check the deployed service logs in the Railway dashboard and observe the request processing flow.

---

## Chapter Key Summary

### 1. Architecture Pattern
When deploying AI agents to production, **wrapping them as a REST API** is the standard approach. FastAPI is optimized for this task, providing async support, automatic documentation generation, type validation, and more.

### 2. Conversation State Management
Using OpenAI's **Conversations API** eliminates the need to manage conversation history on the server side. Just passing the `conversation_id` allows OpenAI to automatically maintain previous context. This is a significant advantage in serverless environments or when horizontal scaling.

### 3. Synchronous vs Streaming
- **`Runner.run()`**: Requires waiting for the complete response, but implementation is simple. Suitable for backend-to-backend communication.
- **`Runner.run_streamed()`**: Supports real-time token streaming. Essential for user-facing UIs.

### 4. Event Filtering
When streaming, it is important to filter only the events that match your purpose from the various events generated by `stream_events()`:
- Text only needed: `raw_response_event` + `response.output_text.delta`
- All events needed: All `raw_response_event`

### 5. Cloud Deployment
Using the Railway + NIXPACKS combination enables automatic build and deployment with just `pyproject.toml`, without a Dockerfile. The key configurations are `startCommand` in `railway.json` and the environment variable (`OPENAI_API_KEY`).

### 6. Security
The `.env` file must be included in `.gitignore` to prevent it from being committed to the Git repository. In deployment environments, use the platform's environment variable management features.

---

## Practice Exercises

### Exercise 1: Basic Deployment (Difficulty: 2/5)
Follow the chapter's code to build a locally functioning AI agent API.

**Requirements:**
- `POST /conversations` - Create conversation
- `POST /conversations/{id}/message` - Send synchronous message
- `POST /conversations/{id}/message-stream` - Send streaming message
- Verify that conversation context is properly maintained

### Exercise 2: Agent Customization (Difficulty: 3/5)
Change the basic Assistant agent to an agent specialized in a specific domain.

**Example:**
```python
agent = Agent(
    name="Korean Teacher",
    instructions="""You are a Korean language teacher.
    Help users learn Korean.
    Correct grammatical errors and suggest natural expressions.""",
)
```

### Exercise 3: Adding Tools (Difficulty: 4/5)
Add function tools to the agent to extend it with the ability to utilize external data.

**Hint:**
```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Returns the weather for a given city."""
    # Connect to a real weather API or return dummy data
    return f"Current weather in {city}: Clear, 22 degrees"

agent = Agent(
    name="Weather Assistant",
    instructions="You help users check the weather.",
    tools=[get_weather],
)
```

Observe how tool call events are delivered in the streaming endpoint (`/message-stream-all`).

### Exercise 4: Railway Deployment (Difficulty: 4/5)
Actually deploy to Railway and call the API with the deployed URL.

**Checklist:**
- [ ] Create Railway project
- [ ] Connect GitHub repository
- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] Health check via `/` endpoint after deployment is complete
- [ ] Test conversation creation and message sending
- [ ] Test streaming responses (using `curl -N`)

### Exercise 5: Frontend Integration (Difficulty: 5/5)
Implement a simple chat frontend that connects to the deployed API.

**Hint:**
- Streaming read with the `fetch()` API:
```javascript
const response = await fetch(url, { method: 'POST', body: JSON.stringify({ question }), headers: { 'Content-Type': 'application/json' } });
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // Append text to UI
}
```

---

> **Next Chapter Preview**: In the next chapter, we will learn how to test and verify the quality of AI agent responses.
