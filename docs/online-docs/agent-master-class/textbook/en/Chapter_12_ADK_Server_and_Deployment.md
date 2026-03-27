# Chapter 12: ADK Server and Deployment

---

## 12.0 Chapter Overview

In this chapter, we learn the entire process of building AI agents using Google ADK (Agent Development Kit), operating them as servers, and finally deploying them to Google Cloud Vertex AI.

Throughout the chapter, we cover two agent projects:

1. **Email Refiner Agent** - A LoopAgent-based system where multiple specialized agents collaborate to iteratively improve emails
2. **Travel Advisor Agent** - A tool-based travel advisor agent that provides weather, exchange rate, and attraction information

Through these two projects, we learn the following core topics:

| Section | Topic | Key Concepts |
|---------|-------|-------------|
| 12.0 | Introduction | ADK project structure, Agent definition, prompt design |
| 12.1 | LoopAgent | Loop agent, output_key, escalate, ToolContext |
| 12.3 | API Server | ADK built-in API server, REST endpoints, session management |
| 12.4 | Server Sent Events | SSE streaming, real-time response handling |
| 12.6 | Runner | Runner class, DatabaseSessionService, code-mode execution |
| 12.7 | Deployment to VertexAI | Vertex AI deployment, reasoning_engines, remote execution |

---

## 12.0 Introduction - ADK Project Structure and Agent Definition

### Topic and Objectives

Understand the basic structure of an ADK-based agent project and design each component of a multi-agent system called Email Refiner.

### Key Concepts

#### 1) ADK Project Directory Structure

ADK follows specific directory structure conventions. `agent.py` and `__init__.py` must exist inside the agent package, and `__init__.py` must import the `agent` module for ADK to automatically recognize the agent.

```
email-refiner-agent/
├── .python-version          # Python version (3.13)
├── pyproject.toml           # Project dependency definitions
├── uv.lock                  # Dependency lock file
├── README.md
└── email_refiner/           # Agent package
    ├── __init__.py          # Agent module registration
    ├── agent.py             # Agent definition
    └── prompt.py            # Prompt and description collection
```

**Role of `__init__.py`:**

```python
from . import agent
```

This single line is very important. The ADK framework automatically searches for the `agent` module within the package, and this explicit import in `__init__.py` is required for ADK's agent discovery to work.

#### 2) Dependency Configuration (pyproject.toml)

```toml
[project]
name = "email-refiner-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "google-adk>=1.12.0",
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Key dependencies:
- **`google-adk`**: Google Agent Development Kit core library
- **`google-genai`**: Google Generative AI client
- **`litellm`**: A library that enables using various LLM providers (OpenAI, Anthropic, Google, etc.) through a unified interface

#### 3) Multi-Specialist Agent Design

The Email Refiner consists of 5 specialized agents. Each agent handles a different aspect of email improvement:

```python
from google.adk.agents import Agent, LoopAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="openai/gpt-4o-mini")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
)

literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
)
```

**Role of Each Agent:**

| Agent | Role | Core Mission |
|-------|------|-------------|
| ClarityEditorAgent | Clarity Editor | Remove ambiguity, eliminate redundant phrases, make sentences concise |
| ToneStylistAgent | Tone Stylist | Maintain warm and confident tone, preserve professionalism |
| PersuationAgent | Persuasion Strategist | Strengthen CTAs, structure arguments, remove passive expressions |
| EmailSynthesizerAgent | Email Synthesizer | Integrate all improvements into a single email |
| LiteraryCriticAgent | Literary Critic | Final quality review and approve/rework decision |

#### 4) Prompt Design Pattern

Prompts are managed by separating `description` (agent role description) and `instruction` (detailed directives). This follows the Separation of Concerns principle.

```python
# Description - Briefly defines what the agent is
CLARITY_EDITOR_DESCRIPTION = "Expert editor focused on clarity and simplicity."

# Instruction - Describes in detail how the agent should behave
CLARITY_EDITOR_INSTRUCTION = """
You are an expert editor focused on clarity and simplicity. Your job is to
eliminate ambiguity, redundancy, and make every sentence crisp and clear.

Take the email draft and improve it for clarity:
- Remove redundant phrases
- Simplify complex sentences
- Eliminate ambiguity
- Make every sentence clear and direct

Provide your improved version with focus on clarity.
"""
```

Particularly notable is the **pipeline pattern**. Each agent's instruction uses template variables that reference the output of previous agents:

```python
TONE_STYLIST_INSTRUCTION = """
...
Here's the clarity-improved version:
{clarity_output}
"""

PERSUASION_STRATEGIST_INSTRUCTION = """
...
Here's the tone-improved version:
{tone_output}
"""

EMAIL_SYNTHESIZER_INSTRUCTION = """
...
Clarity version: {clarity_output}
Tone version: {tone_output}
Persuasion version: {persuasion_output}

Synthesize the best elements from all versions into one polished final email.
"""
```

The variables `{clarity_output}`, `{tone_output}`, etc. connect with the `output_key` that we'll learn about in the next section.

### Practice Points

1. Create an ADK project directory yourself and build the structure that imports the agent module from `__init__.py`.
2. Read each agent's prompt and draw a diagram of the email improvement pipeline flow.
3. Use `LiteLlm` to replace the OpenAI model with another model (e.g., `anthropic/claude-3-haiku`).

---

## 12.1 LoopAgent - Loop Agent and Escalation

### Topic and Objectives

Build a system where multiple agents collaborate iteratively using ADK's `LoopAgent`. Learn about data sharing between agents through `output_key` and the loop termination mechanism through `escalate`.

### Key Concepts

#### 1) output_key - Data Transfer Between Agents

`output_key` specifies the key name for storing an agent's output in the session state. The template variables `{clarity_output}`, `{tone_output}`, etc. seen in the previous section are populated through this `output_key`.

```python
MODEL = LiteLlm(model="openai/gpt-4o")

clarity_agent = Agent(
    name="ClarityEditorAgent",
    description=CLARITY_EDITOR_DESCRIPTION,
    instruction=CLARITY_EDITOR_INSTRUCTION,
    output_key="clarity_output",    # Output saved to state["clarity_output"]
    model=MODEL,
)

tone_stylist_agent = Agent(
    name="ToneStylistAgent",
    description=TONE_STYLIST_DESCRIPTION,
    instruction=TONE_STYLIST_INSTRUCTION,
    output_key="tone_output",       # Output saved to state["tone_output"]
    model=MODEL,
)

persuation_agent = Agent(
    name="PersuationAgent",
    description=PERSUASION_STRATEGIST_DESCRIPTION,
    instruction=PERSUASION_STRATEGIST_INSTRUCTION,
    output_key="persuasion_output", # Output saved to state["persuasion_output"]
    model=MODEL,
)

email_synthesizer_agent = Agent(
    name="EmailSynthesizerAgent",
    description=EMAIL_SYNTHESIZER_DESCRIPTION,
    instruction=EMAIL_SYNTHESIZER_INSTRUCTION,
    output_key="synthesized_output", # Output saved to state["synthesized_output"]
    model=MODEL,
)
```

**Data Flow:**

```
User email input
    │
    ▼
ClarityEditorAgent ──── output_key="clarity_output" ──────► Saved to state
    │
    ▼
ToneStylistAgent ────── output_key="tone_output" ──────────► Saved to state
    │                    (references {clarity_output} in instruction)
    ▼
PersuationAgent ─────── output_key="persuasion_output" ────► Saved to state
    │                    (references {tone_output} in instruction)
    ▼
EmailSynthesizerAgent ─ output_key="synthesized_output" ───► Saved to state
    │                    (references all 3 outputs)
    ▼
LiteraryCriticAgent ─── Quality judgment
    │                    (references {synthesized_output} in instruction)
    ├── Fail → Restart loop
    └── Pass → Exit loop via escalate
```

#### 2) ToolContext and escalate - Loop Termination Mechanism

`LoopAgent` loops indefinitely by default (or up to `max_iterations`), and the `escalate` mechanism is used to break out of the loop under specific conditions.

```python
from google.adk.tools.tool_context import ToolContext

async def escalate_email_complete(tool_context: ToolContext):
    """Use this tool only when the email is good to go."""
    tool_context.actions.escalate = True
    return "Email optimization complete."
```

**Key Points:**
- `ToolContext` is a context object that ADK automatically injects when executing a tool.
- Setting `tool_context.actions.escalate = True` immediately terminates the current loop.
- This tool is only given to the `LiteraryCriticAgent`, so the loop only terminates when the critic is satisfied with the email quality.

```python
literary_critic_agent = Agent(
    name="LiteraryCriticAgent",
    description=LITERARY_CRITIC_DESCRIPTION,
    instruction=LITERARY_CRITIC_INSTRUCTION,
    tools=[
        escalate_email_complete,   # Escalate tool assigned
    ],
    model=MODEL,
)
```

#### 3) LoopAgent Configuration

All sub-agents are wrapped in a `LoopAgent` to complete the iterative execution structure:

```python
email_refiner_agent = LoopAgent(
    name="EmailRefinerAgent",
    max_iterations=50,                    # Maximum 50 iterations (safety measure)
    description=EMAIL_OPTIMIZER_DESCRIPTION,
    sub_agents=[
        clarity_agent,                     # 1. Clarity improvement
        tone_stylist_agent,                # 2. Tone adjustment
        persuation_agent,                  # 3. Persuasion strengthening
        email_synthesizer_agent,           # 4. Synthesis
        literary_critic_agent,             # 5. Final review (can escalate)
    ],
)

root_agent = email_refiner_agent
```

**Importance of the `root_agent` variable:** The ADK framework automatically searches for a variable named `root_agent` and uses it as the entry point agent. You must use this name.

#### 4) Prompt Reinforcement - Ensuring LLM Actually Calls the Tool

In practice, the LLM may "say" it will call a tool but not actually call it. To prevent this, the prompt was reinforced:

```python
LITERARY_CRITIC_INSTRUCTION = """
...
2. If the email meets professional standards and communicates effectively:
   - Call the `escalate_email_complete` tool, CALL IT DONT JUST SAY YOU ARE
     GOING TO CALL IT. CALL THE THING!
   - Provide your final positive assessment of the email
...
## Tool Usage:
When the email is ready, CALL the tool: `escalate_email_complete()`
...
"""
```

Using uppercase and emphatic expressions to clearly instruct the LLM to execute the tool call is a very useful prompt engineering technique in practice.

### Practice Points

1. Lower `max_iterations` to 3 and run it to observe the behavior when the loop reaches the maximum iteration count.
2. See what happens if you only provide a return value instead of setting `escalate = True` in the `escalate_email_complete` function.
3. Remove `output_key` and run to confirm that the next agent cannot reference the previous result.

---

## 12.3 API Server - ADK Built-in API Server

### Topic and Objectives

Learn how to serve agents as REST APIs using ADK's built-in web server. Create a new Travel Advisor Agent and interact with it through the API server.

### Key Concepts

#### 1) Travel Advisor Agent - Tool-Based Agent

Build a new travel advisor agent that utilizes tools for the API server demonstration:

```python
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext

MODEL = LiteLlm(model="openai/gpt-4o")


async def get_weather(tool_context: ToolContext, location: str):
    """Get current weather information for a location."""
    return {
        "location": location,
        "temperature": "22°C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind": "12 km/h",
        "forecast": "Mild weather with occasional clouds expected throughout the day",
    }


async def get_exchange_rate(
    tool_context: ToolContext, from_currency: str, to_currency: str, amount: float
):
    """Get exchange rate between two currencies.
    Args should always be from_currency str, to_currency str, amount flot
    """
    mock_rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("USD", "KRW"): 1325.00,
        ("EUR", "USD"): 1.09,
        ("EUR", "GBP"): 0.86,
        ("GBP", "USD"): 1.27,
        ("JPY", "USD"): 0.0067,
        ("KRW", "USD"): 0.00075,
    }

    rate = mock_rates.get((from_currency, to_currency), 1.0)
    converted_amount = amount * rate

    return {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "amount": amount,
        "exchange_rate": rate,
        "converted_amount": converted_amount,
        "timestamp": "2024-03-15 10:30:00 UTC",
    }


async def get_local_attractions(
    tool_context: ToolContext, location: str, category: str = "all"
):
    """Get popular attractions and points of interest for a location."""
    attractions = {
        "Paris": [
            {"name": "Eiffel Tower", "type": "landmark", "rating": 4.8,
             "description": "Iconic iron lattice tower"},
            {"name": "Louvre Museum", "type": "museum", "rating": 4.7,
             "description": "World's largest art museum"},
            # ... more attraction data
        ],
        "Tokyo": [
            {"name": "Tokyo Tower", "type": "landmark", "rating": 4.5,
             "description": "Communications and observation tower"},
            {"name": "Senso-ji", "type": "temple", "rating": 4.6,
             "description": "Ancient Buddhist temple"},
            # ... more attraction data
        ],
        "default": [
            {"name": "City Center", "type": "area", "rating": 4.2,
             "description": "Main downtown area"},
            # ... default attraction data
        ],
    }

    location_attractions = attractions.get(location, attractions["default"])

    if category != "all":
        location_attractions = [
            a for a in location_attractions if a["type"] == category
        ]

    return {
        "location": location,
        "category": category,
        "attractions": location_attractions,
        "total_count": len(location_attractions),
    }
```

**Tool Function Design Patterns:**
- All tool functions are defined as `async` asynchronous functions.
- The first parameter must be `tool_context: ToolContext` (automatically injected by ADK).
- The docstring serves to explain the tool's purpose to the LLM.
- Return values are in dictionary format, which the LLM interprets to respond to the user.

Agent registration:

```python
travel_advisor_agent = Agent(
    name="TravelAdvisorAgent",
    description=TRAVEL_ADVISOR_DESCRIPTION,
    instruction=TRAVEL_ADVISOR_INSTRUCTION,
    tools=[
        get_weather,
        get_exchange_rate,
        get_local_attractions,
    ],
    model=MODEL,
)

root_agent = travel_advisor_agent
```

#### 2) Running the ADK Built-in API Server

ADK can instantly launch a built-in web server with the `adk api_server` command. This server is FastAPI-based and automatically provides REST endpoints for interacting with agents.

```bash
# Run from the parent directory containing the agent project
adk api_server email-refiner-agent/
```

Once the server starts, it is accessible at `http://127.0.0.1:8000`.

#### 3) Interacting with Agents via REST API

**Creating a Session:**

```python
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"

# Create a new session
response = requests.post(
    f"{BASE_URL}/apps/{APP_NAME}/users/{USER_ID}/sessions"
)
print(response.json())
# Returns a response containing the session ID
```

ADK API server session creation endpoint pattern:
```
POST /apps/{app_name}/users/{user_id}/sessions
```

**Sending Messages (Synchronous Mode):**

```python
SESSION_ID = "ce085ce3-9637-4eca-b7a1-b0be58fa39f1"  # ID received during session creation

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "Yes, I want to know the currency exchange rate"}],
        "role": "user",
    },
}
response = requests.post(f"{BASE_URL}/run", json=message)
print(response.json())
```

**Parsing Responses:**

```python
data = response.json()

for event in data:
    content = event.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

Responses come as an array of events, where each event's `content.parts` contains:
- `functionCall`: Information about tools called by the agent
- `text`: The agent's text response

#### 4) Dependency Updates

Dependencies were added for API server and evaluation (eval) features:

```toml
dependencies = [
    "google-adk[eval]>=1.12.0",   # [eval] extra added
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",            # HTTP client (for API calls)
    "sseclient-py>=1.8.0",        # SSE client (next section)
]
```

### Practice Points

1. Run the server with the `adk api_server` command and visit `http://127.0.0.1:8000/docs` in your browser to check the auto-generated Swagger UI.
2. Create a session and send multiple messages sequentially to verify that conversation context is maintained.
3. Send API requests to the `email_refiner` agent using a different `APP_NAME`.

---

## 12.4 Server Sent Events (SSE) - Real-time Streaming Responses

### Topic and Objectives

Learn how to handle real-time streaming responses based on Server-Sent Events using the `/run_sse` endpoint instead of the synchronous `/run` endpoint.

### Key Concepts

#### 1) What are SSE (Server-Sent Events)?

SSE is an HTTP-based protocol for unidirectional real-time data streaming from server to client. Unlike WebSocket, it uses regular HTTP connections, making implementation simple.

**Synchronous Mode vs SSE Mode Comparison:**

| Feature | `/run` (Synchronous) | `/run_sse` (Streaming) |
|---------|---------------------|----------------------|
| Response Method | Returns entire response at once | Sends in real-time as event units |
| User Experience | Wait until response is complete | Monitor progress in real-time |
| Tool Call Observation | Included in result | Observe call process in real-time |
| Suitable For | Short responses, backend processing | Long responses, frontend UI |

#### 2) SSE Client Implementation

```python
import sseclient
import json
import requests

BASE_URL = "http://127.0.0.1:8000"
APP_NAME = "travel_advisor_agent"
USER_ID = "u_123"
SESSION_ID = "3f673a5a-04ab-4edb-af23-6f42449a970b"

message = {
    "appName": APP_NAME,
    "userId": USER_ID,
    "sessionId": SESSION_ID,
    "newMessage": {
        "parts": [{"text": "What is the weather there?"}],
        "role": "user",
    },
    "streaming": True,              # Enable streaming flag
}

response = requests.post(
    f"{BASE_URL}/run_sse",           # SSE-dedicated endpoint
    json=message,
    stream=True,                     # Enable requests streaming mode
)

client = sseclient.SSEClient(response)

for event in client.events():
    data = json.loads(event.data)
    content = data.get("content")
    parts = content.get("parts")
    for part in parts:
        function_call = part.get("functionCall", None)
        if function_call:
            print(function_call.get("name"))
        text = part.get("text", None)
        if text:
            print(text)
    print("=" * 60)
```

**Code Differences from Synchronous Mode:**

1. **Add `"streaming": True` to request message** - Informs the server of streaming mode.
2. **Endpoint change**: Use `/run_sse` instead of `/run`
3. **`stream=True` option**: Enable streaming mode on `requests.post()`
4. **Wrap with `sseclient.SSEClient`**: Parse the response as an SSE event stream
5. **Event loop**: Process events one by one with `client.events()`

#### 3) SSE Event Structure

Each SSE event contains a `data` field in JSON format:

```json
{
    "content": {
        "parts": [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"}
                }
            }
        ]
    }
}
```

Or text response:

```json
{
    "content": {
        "parts": [
            {
                "text": "The current weather in Paris is 22 degrees..."
            }
        ]
    }
}
```

### Practice Points

1. Send the same question in both synchronous mode (`/run`) and SSE mode (`/run_sse`) and compare the response time and user experience differences.
2. While receiving SSE events, observe that tool call (functionCall) events arrive before text responses.
3. Try writing code that parses the HTTP stream directly instead of using `sseclient-py` (using `response.iter_lines()`).

---

## 12.6 Runner - Running Agents Directly from Code

### Topic and Objectives

Learn how to run agents directly in pure Python code using the `Runner` class without the ADK CLI or API server. Also covers persistent session management via `DatabaseSessionService` and `InMemoryArtifactService`.

### Key Concepts

#### 1) Role of the Runner

`Runner` is the core orchestrator for agent execution. Use it when you want to run agents directly within code without an API server. The Runner manages:
- Agent execution flow
- Session state management
- Artifact (file, etc.) management
- Event streaming

#### 2) Session Service and Artifact Service

```python
from google.adk.sessions import DatabaseSessionService
from google.adk.artifacts import InMemoryArtifactService

# Artifact service: Memory-based (temporary storage for files, etc.)
in_memory_service_py = InMemoryArtifactService()

# Session service: SQLite DB-based (persistent session storage)
session_service = DatabaseSessionService(db_url="sqlite:///./session.db")
```

**Advantages of `DatabaseSessionService`:**
- Session data is persistently stored in a SQLite file (`session.db`).
- Previous conversations can be resumed even after server restart.
- Changing `db_url` to PostgreSQL, etc. makes it usable in production environments.

#### 3) Session Creation and State Initialization

```python
session = await session_service.create_session(
    app_name="weather_agent",
    user_id="u_123",
    state={
        "user_name": "nico",    # Store user name in initial state
    },
)
```

You can set initial values in the `state` dictionary. These values are referenced as template variables in the agent's instruction:

```python
# prompt.py
TRAVEL_ADVISOR_INSTRUCTION = """
You are a helpful travel advisor agent...

You call the user by their name:

Their name is {user_name}
...
"""
```

`{user_name}` is automatically substituted with the `"user_name"` value from the session state. This is ADK's **state-based prompt template** feature.

#### 4) Running Agents via Runner

```python
from google.genai import types
from google.adk.runners import Runner

# Create Runner
runner = Runner(
    agent=travel_advisor_agent,           # Agent to run
    session_service=session_service,      # Session management service
    app_name="weather_agent",             # App name (must match session service)
    artifact_service=in_memory_service_py, # Artifact management service
)

# Create user message
message = types.Content(
    role="user",
    parts=[
        types.Part(text="Im going to Vietnam, tell me all about it."),
    ],
)

# Asynchronous streaming execution
async for event in runner.run_async(
    user_id="u_123",
    session_id=session.id,
    new_message=message
):
    if event.is_final_response():
        print(event.content.parts[0].text)
    else:
        print(event.get_function_calls())
        print(event.get_function_responses())
```

**Event Processing Pattern:**
- `event.is_final_response()`: Check if it's the final text response
- `event.get_function_calls()`: Check tool call events
- `event.get_function_responses()`: Check tool response events

#### 5) Execution Result Analysis

Looking at the actual execution results, you can clearly observe the agent's operation process:

```
# Step 1: Agent calls 3 tools simultaneously (parallel tool calling)
[FunctionCall(name='get_weather', args={'location': 'Vietnam'}),
 FunctionCall(name='get_exchange_rate', args={'from_currency': 'USD', 'to_currency': 'VND', 'amount': 1}),
 FunctionCall(name='get_local_attractions', args={'location': 'Vietnam'})]

# Step 2: Receive tool responses
[FunctionResponse(name='get_weather', response=<dict len=6>),
 FunctionResponse(name='get_exchange_rate', response=<dict len=6>),
 FunctionResponse(name='get_local_attractions', response={
     'error': "Invoking `get_local_attractions()` failed as the following
     mandatory input parameters are not present: category..."
 })]

# Step 3: Final response (synthesizes tool results into natural language)
Hello Nico! Here's some information to help you prepare for your trip to Vietnam:

### Weather in Vietnam
- **Current Temperature:** 22°C
- **Condition:** Partly cloudy
...
```

Notable points:
1. The agent reads `{user_name}` from the session state and greets with "Hello Nico!".
2. It calls 3 tools **in parallel** for efficient information gathering.
3. Although a `category` parameter missing error occurred with `get_local_attractions`, the agent handled it on its own and directly generated general Vietnam tourist information.

### Practice Points

1. Replace `DatabaseSessionService` with `InMemorySessionService` and confirm that sessions are not preserved after server restart.
2. Add `"preferred_language": "Korean"` to `state` and use it in the prompt to make the agent respond in Korean.
3. Find out how to use the synchronous `run` method instead of `run_async`.
4. Use `output_key` to save agent responses to session state and reference them in the next conversation.

---

## 12.7 Deployment to Vertex AI - Cloud Deployment

### Topic and Objectives

Learn how to deploy built ADK agents to Google Cloud's Vertex AI Agent Engine for operation in a production environment.

### Key Concepts

#### 1) What is Vertex AI Agent Engine?

Vertex AI Agent Engine (formerly Reasoning Engine) is a Google Cloud service for hosting and managing AI agents. Deploying ADK agents to the cloud provides:
- No server infrastructure management needed
- Auto-scaling
- Leveraging Google Cloud's security and monitoring features
- Remote session management and execution

#### 2) Deployment Script (deploy.py)

```python
import dotenv

dotenv.load_dotenv()

import os
import vertexai
import vertexai.agent_engines
from vertexai.preview import reasoning_engines
from travel_advisor_agent.agent import travel_advisor_agent

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"
BUCKET = "gs://nico-awesome-weather_agent"

# Initialize Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=BUCKET,         # GCS bucket for deployment file staging
)

# Wrap ADK agent as AdkApp
app = reasoning_engines.AdkApp(
    agent=travel_advisor_agent,
    enable_tracing=True,            # Enable execution tracing
)

# Deploy to Vertex AI
remote_app = vertexai.agent_engines.create(
    display_name="Travel Advisor Agent",
    agent_engine=app,
    requirements=[                  # Required Python packages
        "google-cloud-aiplatform[adk,agent_engines]",
        "litellm",
    ],
    extra_packages=["travel_advisor_agent"],  # Include agent package
    env_vars={                      # Pass environment variables
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    },
)
```

**Detailed Deployment Process Analysis:**

| Step | Code | Description |
|------|------|-------------|
| 1. Environment Setup | `dotenv.load_dotenv()` | Load environment variables like API keys from `.env` file |
| 2. Vertex AI Init | `vertexai.init(...)` | Configure project, region, staging bucket |
| 3. App Wrapping | `reasoning_engines.AdkApp(...)` | Wrap ADK agent in Vertex AI compatible format |
| 4. Deployment | `agent_engines.create(...)` | Execute actual deployment to cloud |

**Role of `extra_packages` parameter:**
Includes the local package directory (`travel_advisor_agent`) in the deployment bundle. This is necessary for the agent code to be importable in the cloud environment.

**Secret Management via `env_vars`:**
Sensitive information like API keys is passed as environment variables. It's important for security not to hardcode them directly in the code.

#### 3) Additional Dependencies

Packages added for deployment:

```toml
dependencies = [
    "cloudpickle>=3.1.1",                                    # Object serialization
    "google-adk[eval]>=1.12.0",
    "google-cloud-aiplatform[adk,agent-engines]>=1.111.0",   # Vertex AI SDK
    "google-genai>=1.31.0",
    "litellm>=1.76.0",
    "requests>=2.32.5",
    "sseclient-py>=1.8.0",
]
```

- **`cloudpickle`**: Used to serialize Python objects for transmission to the cloud
- **`google-cloud-aiplatform[adk,agent-engines]`**: Includes Vertex AI's ADK and Agent Engine features

#### 4) Remote Agent Management and Execution (remote.py)

```python
import vertexai
from vertexai import agent_engines

PROJECT_ID = "gen-lang-client-0125196626"
LOCATION = "europe-southwest1"

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
)

# Query deployment list
# deployments = agent_engines.list()
# for deployment in deployments:
#     print(deployment)

# Get remote app by specific deployment ID
DEPLOYMENT_ID = "projects/23382131925/locations/europe-southwest1/reasoningEngines/2153529862441140224"
remote_app = agent_engines.get(DEPLOYMENT_ID)

# Delete deployment (force delete with force=True)
remote_app.delete(force=True)
```

**Remote Session Creation and Streaming Query:**

```python
# Create remote session
# remote_session = remote_app.create_session(user_id="u_123")
# print(remote_session["id"])

SESSION_ID = "5724511082748313600"

# Send streaming query to remote agent
# for event in remote_app.stream_query(
#     user_id="u_123",
#     session_id=SESSION_ID,
#     message="I'm going to Laos, any tips?",
# ):
#     print(event, "\n", "=" * 50)
```

**Remote Execution API Summary:**

| Method | Purpose |
|--------|---------|
| `agent_engines.list()` | Query all deployments |
| `agent_engines.get(id)` | Get specific deployment |
| `remote_app.create_session(user_id=...)` | Create remote session |
| `remote_app.stream_query(...)` | Query with streaming |
| `remote_app.delete(force=True)` | Delete deployment |

### Practice Points

1. Create a GCP project and GCS bucket, and actually deploy an agent.
2. Deploy with `enable_tracing=True` and check the tracing logs in the Google Cloud Console.
3. Compare response times between `remote_app.stream_query()` and local Runner execution.
4. Create sessions with multiple user IDs and verify that session isolation works correctly.

---

## Chapter Key Summary

### 1. ADK Agent Architecture

```
                    ┌─────────────────────┐
                    │      ADK Agent      │
                    │                     │
                    │  - name             │
                    │  - description      │
                    │  - instruction      │
                    │  - model            │
                    │  - tools            │
                    │  - output_key       │
                    │  - sub_agents       │
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
         │  Agent  │   │ LoopAgent │   │  Runner   │
         │ (Single)│   │  (Loop)   │   │ (Executor)│
         └─────────┘   └───────────┘   └───────────┘
```

### 2. Execution Mode Comparison

| Execution Method | Description | When to Use |
|-----------------|-------------|-------------|
| `adk web` | Test agent with web UI | Quick testing during development |
| `adk api_server` | Run REST API server | Frontend integration, local service |
| `Runner` (code mode) | Execute directly from Python code | Custom application integration |
| Vertex AI Deployment | Cloud production environment | Production service operation |

### 3. Key ADK Classes/Components Summary

| Component | Role |
|-----------|------|
| `Agent` | Single agent definition (name, description, instruction, model, tools) |
| `LoopAgent` | Orchestrator that iteratively executes sub-agents |
| `LiteLlm` | Use various LLM providers through a unified interface |
| `ToolContext` | Access session state and actions from tool functions |
| `output_key` | Key for saving agent output to session state |
| `escalate` | Early termination of loops or agent chains |
| `Runner` | Orchestrator that manages agent execution from code |
| `DatabaseSessionService` | DB-based persistent session management |
| `InMemoryArtifactService` | Memory-based artifact management |
| `reasoning_engines.AdkApp` | Wraps ADK agent for Vertex AI deployment format |

### 4. Core Data Flow Pattern

```
Save with output_key → Accumulate in state → Reference via {variable_name} in instruction
```

This pattern is the most important mechanism for transferring data between agents in ADK.

---

## Practice Exercises

### Exercise 1: Code Review Agent (Using LoopAgent)

Referencing the Email Refiner Agent structure, create a **Code Review Agent**.

**Requirements:**
- `SecurityReviewAgent`: Security vulnerability review
- `PerformanceReviewAgent`: Performance optimization suggestions
- `StyleReviewAgent`: Code style and readability review
- `ReviewSynthesizerAgent`: Synthesize all reviews
- `ApprovalAgent`: Final approve/reject decision (using escalate tool)

**Hints:**
- Set `output_key` on each agent to save review results to state
- Give `ApprovalAgent` an `escalate_review_complete` tool
- Set `LoopAgent`'s `max_iterations` appropriately

### Exercise 2: API Server and SSE Client

Extend the Travel Advisor Agent to add **restaurant recommendation functionality**, and implement an API server and SSE client.

**Requirements:**
1. Add a `get_restaurant_recommendations(location, cuisine_type)` tool function
2. Run the server with `adk api_server`
3. Receive real-time streaming responses with an SSE client
4. Distinguish between tool call events and text response events for UI display

### Exercise 3: Interactive CLI Using Runner

Create a CLI program that interactively communicates with the agent in the terminal using Runner.

**Requirements:**
1. Use `DatabaseSessionService` for persistent conversation history storage
2. Choose to continue an existing session or create a new session at program start
3. Store the user's preferred language in `state` and use it in the prompt
4. Print the session ID on `Ctrl+C` exit so it can be resumed next time

### Exercise 4: Vertex AI Deployment (Advanced)

Actually deploy the Travel Advisor Agent to Vertex AI and use it remotely.

**Requirements:**
1. Create a GCP project and enable the Vertex AI API
2. Create a GCS bucket (for staging)
3. Write a deployment script referencing `deploy.py`
4. Create a remote session and execute queries referencing `remote.py`
5. Compare and analyze the deployed agent's response time with local execution

**Cautions:**
- GCP charges may apply, so be sure to delete with `remote_app.delete(force=True)` after testing
- Never hardcode API keys in code; always pass them as environment variables

---

> **Next Chapter Preview:** The next chapter covers the Agent Evaluation framework, where we learn how to systematically measure and improve agent response quality.
