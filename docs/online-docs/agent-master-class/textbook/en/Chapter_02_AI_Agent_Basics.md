# Chapter 2: AI Agent Basics

---

## Chapter Overview

In this chapter, we learn step by step how to build an AI agent from scratch using the OpenAI API. Starting from a simple API call, we progressively add conversation memory, define external tools, and implement Function Calling so the AI can execute real functions, culminating in a fully functional agent.

### Learning Objectives

1. Set up a Python development environment and verify OpenAI API connectivity
2. Learn the basics of the OpenAI Chat Completions API
3. Build a chatbot that maintains context by managing conversation history (memory)
4. Define Tool schemas to inform the AI of available functions
5. Implement Function Calling so the AI can actually execute chosen functions
6. Pass tool execution results back to the AI to generate the final response

### Chapter Structure

| Section | Topic | Key Keywords |
|---------|-------|------------|
| 2.0 | Project Setup | uv, Python 3.13, OpenAI SDK |
| 2.2 | Your First AI Agent | Chat Completions API, Prompt Engineering |
| 2.3 | Adding Memory | Conversation History, messages Array, while Loop |
| 2.4 | Adding Tools | Tools Schema, JSON Schema, FUNCTION_MAP |
| 2.5 | Adding Function Calling | Function Calling, tool_calls, process_ai_response |
| 2.6 | Tool Results | Tool Results, Recursive Calls, Completing the Agent Loop |

---

## 2.0 Project Setup

### Topic and Objective

Set up the Python project environment for developing an AI agent. Initialize the project using the `uv` package manager, install the OpenAI Python SDK, and verify that the API key loads correctly in a Jupyter Notebook environment.

### Core Concepts

#### uv Package Manager

`uv` is a next-generation Python package manager written in Rust that provides much faster dependency resolution and installation speed compared to traditional tools like `pip` or `poetry`. In this project, we use `uv` to create virtual environments and manage packages.

#### Project Structure

The project is organized as follows:

```
my-first-agent/
├── .gitignore          # List of files not tracked by Git
├── .python-version     # Python version specification (3.13)
├── README.md           # Project description
├── main.ipynb          # Main notebook (code workspace)
├── pyproject.toml      # Project configuration and dependency definitions
└── uv.lock             # Dependency lock file
```

### Code Analysis

#### pyproject.toml - Project Configuration File

```toml
[project]
name = "my-first-agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "openai>=1.98.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.0",
]
```

**Key Points:**

- `requires-python = ">=3.13"`: Requires Python 3.13 or higher. This is to leverage the latest Python features.
- `dependencies`: Specifies `openai>=1.98.0` as a runtime dependency. This is the official Python SDK for communicating with the OpenAI API.
- `[dependency-groups] dev`: Includes `ipykernel` as a development dependency. This is required to run Python code in Jupyter Notebook.

#### .python-version

```
3.13
```

This file allows tools like `uv` or `pyenv` to automatically detect the Python version to use for the project.

#### main.ipynb - API Key Verification

```python
import os

print(os.getenv("OPENAI_API_KEY"))
```

This code verifies that the OpenAI API key is correctly set in the environment variables. For security reasons, API keys should never be hardcoded in the source code but managed through `.env` files or system environment variables.

#### .gitignore - Ignored Files Configuration

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv
.env
```

**Important:** The `.env` file is included in `.gitignore`. This prevents sensitive information such as API keys from being uploaded to the Git repository. This is a critically important security configuration.

### Practice Points

1. Initialize the project with the `uv init my-first-agent` command
2. Install the OpenAI SDK with `uv add openai`
3. Install the Jupyter kernel with `uv add --dev ipykernel`
4. Create a `.env` file and store your API key in the format `OPENAI_API_KEY=sk-...`
5. Run Jupyter Notebook and verify that the API key prints correctly

---

## 2.2 Your First AI Agent

### Topic and Objective

Create the most basic form of an AI agent using the OpenAI Chat Completions API. At this stage, we experiment with informing the AI of available functions through prompt engineering (as plain text) and guiding it to select the appropriate function.

### Core Concepts

#### Chat Completions API

OpenAI's Chat Completions API is the primary interface for interacting with conversational AI models. You send messages and the AI generates and returns responses. Each message includes a `role` and `content`.

Types of roles:
- `system`: System messages that instruct the AI's behavior
- `user`: Messages sent by the user
- `assistant`: Response messages generated by the AI
- `tool`: Messages delivering tool execution results (covered in later sections)

#### Function Selection via Prompt Engineering

At this stage, we do not yet use OpenAI's official Function Calling feature. Instead, we use prompts (text instructions) to ask the AI "here are some available functions, please pick the right one." This is the most primitive form of an AI agent.

### Code Analysis

#### Prompt-Based Function Selection

```python
import openai

client = openai.OpenAI()

PROMPT = """
I have the following functions in my system.

`get_weather`
`get_currency`
`get_news`

All of them receive the name of a country as an argumet (i.e get_news('Spain'))

Please answer with the name of the function that you would like me to run.

Please say nothing else, just the name of the function with the arguments.

Answer the following question:

What is the weather in Greece?
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": PROMPT}],
)

response
```

**Code Analysis:**

1. **`openai.OpenAI()`**: Creates an OpenAI client. It automatically reads the `OPENAI_API_KEY` environment variable.
2. **`PROMPT`**: Defines a multi-line prompt. It tells the AI about the available functions (`get_weather`, `get_currency`, `get_news`) and asks it to select the one matching the question.
3. **`client.chat.completions.create()`**: Makes the API call. The `model` parameter specifies which model to use, and the `messages` parameter passes the conversation content.

#### Extracting the Message from the Response

```python
message = response.choices[0].message.content
message
```

**Output:**
```
"get_weather('Greece')"
```

**Key Understanding:**

- `response.choices[0]`: Gets the first choice from the API response (typically only one is returned)
- `.message.content`: Extracts the message content (text) from that choice
- The AI followed the prompt instructions and returned text in function call format: `get_weather('Greece')`

#### Limitations of This Approach

While this approach works, it has several issues:

- There is no guarantee that the AI's response will always be in a consistent format (e.g., `"get_weather('Greece')"` vs `"I would call get_weather with Greece"`)
- Additional string processing is needed to parse the returned text and actually call the function
- It is difficult to precisely convey function parameter types and whether they are required

To address these limitations, OpenAI provides the official **Function Calling** feature, which is covered from section 2.4 onward.

### Practice Points

1. Modify the prompt to ask a different question (e.g., "What is the currency of Japan?")
2. Verify that the AI returns `get_currency('Japan')`
3. Remove the "Please say nothing else" part from the prompt and observe how the AI response changes
4. Compare the response differences when using a different model (e.g., `gpt-4o`)

---

## 2.3 Adding Memory

### Topic and Objective

Build an AI chatbot that remembers previous conversations. Use the `messages` array to manage conversation history, enabling continuous dialogue between the user and AI.

### Core Concepts

#### How Conversation Memory Works

LLMs (Large Language Models) are fundamentally **stateless** systems. Each API call is independent, and previous conversations are not automatically remembered. Therefore, to maintain conversation context, **all previous messages must be sent along with each API call**.

This is exactly what the `messages` array does. Each time a user sends a new message:

1. Add the user's message to the `messages` array
2. Send the entire `messages` array to the API
3. Add the AI's response back to the `messages` array
4. In the next conversation turn, all this history is sent together

```
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico!"},
    {"role": "user", "content": "What is my name?"},
    # The AI can see the history above and answer "Your name is Nico."
]
```

### Code Analysis

#### Initial Setup

```python
import openai

client = openai.OpenAI()
messages = []
```

Initialize `messages` as an empty array. This array serves as the **memory** that stores the entire conversation history.

#### AI Call Function Definition

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": message})
    print(f"AI: {message}")
```

**Code Analysis:**

1. The entire `messages` array is passed to the API to maintain conversation context
2. The AI's response (`message.content`) is extracted
3. The response is added to the `messages` array as `{"role": "assistant", "content": message}` -- this is the act of **saving to memory**
4. The response is printed to the screen

#### Conversation Loop

```python
while True:
    message = input("Send a message to the LLM...")
    if message == "quit" or message == "q":
        break
    else:
        messages.append({"role": "user", "content": message})
        print(f"User: {message}")
        call_ai()
```

**Code Analysis:**

1. `while True`: An infinite loop to continue the conversation
2. `input()`: Receives a message from the user
3. Entering `"quit"` or `"q"` exits the loop
4. The user message is added to `messages` as `{"role": "user", "content": message}`, then `call_ai()` is called

#### Execution Results (Verifying Memory Operation)

```
User: My name is Nico
AI: Nice to meet you, Nico! How can I assist you today?
User: What is my name?
AI: Your name is Nico.
User: I'm from Korea
AI: That's great! Korea has a rich culture and history. ...
User: What was the first question I asked you and what is the closest Island country to where I was born?
AI: The first question you asked was, "What is my name?" As for the closest island country to Korea, that would be Japan...
```

**What these results demonstrate:**

- The AI remembers the user's name ("Nico")
- The AI remembers the user's origin ("Korea")
- The AI even remembers what the first question was
- All of this is possible because previous conversations are accumulated in the `messages` array

### Visualizing the messages Array Structure

```
API Call #1:
messages = [
    {"role": "user", "content": "My name is Nico"}
]

API Call #2:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"}
]

API Call #3:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"},
    {"role": "assistant", "content": "Your name is Nico."},
    {"role": "user", "content": "I'm from Korea"}
]
```

Note that the entire history is sent with every call, so API costs (token usage) increase as conversations grow longer.

### Practice Points

1. Continue a long conversation and test how well the AI remembers
2. Print the `messages` array directly to examine its internal structure
3. Reset with `messages = []` mid-conversation and confirm the AI forgets previous dialogue
4. Add a `system` role message to change the AI's personality (e.g., `{"role": "system", "content": "You are a pirate. Speak like a pirate."}`)

---

## 2.4 Adding Tools

### Topic and Objective

Use OpenAI's official **Tools** feature to structurally inform the AI of available functions. Define function names, descriptions, and parameters using JSON Schema, and observe how the `finish_reason` changes to `tool_calls` when the AI decides to invoke a tool.

### Core Concepts

#### Prompt-Based vs Tools-Based Comparison

In section 2.2, we used prompt text to inform the AI about available functions. This was unreliable and difficult to parse. OpenAI's **Tools** feature replaces this with structured JSON Schema:

| Aspect | Prompt-Based (2.2) | Tools-Based (2.4) |
|--------|-------------------|-----------------|
| Function Definition Method | Natural language text | JSON Schema |
| Response Format | Free-form text | Structured tool_calls object |
| Parameter Definition | Unclear | Clear types, required fields, etc. |
| Parsing Difficulty | High | Low (handled by SDK) |

#### FUNCTION_MAP Pattern

When the AI returns a function name, you need to find and execute the actual Python function by that name. For this, we use a **dictionary that maps function names (strings) to function objects**:

```python
FUNCTION_MAP = {"get_weather": get_weather}
```

This pattern allows dynamic invocation of the `get_weather` function using the string `"get_weather"` returned by the AI.

### Code Analysis

#### Function Definition and Mapping

```python
def get_weather(city):
    return "33 degrees celcius."


FUNCTION_MAP = {"get_weather": get_weather}
```

- `get_weather`: A function that returns weather information. Currently it returns a hardcoded value, but in production it would call a weather API.
- `FUNCTION_MAP`: Maps string keys to function objects. Later, when the AI responds with `"get_weather"`, we find the actual function via `FUNCTION_MAP["get_weather"]`.

#### Tools Schema Definition (Core)

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "A function to get the weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the weather of.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]
```

**Detailed Schema Structure Analysis:**

1. **`"type": "function"`**: Specifies that this tool's type is a function.

2. **`"function"` object:**
   - `"name"`: The function name. The AI uses this name to request function calls.
   - `"description"`: A description of the function. The AI uses this to determine in which situations to use this function. **Writing a good description is critically important.**
   - `"parameters"`: Defines parameters in JSON Schema format.
     - `"type": "object"`: Indicates that the parameters are in object form.
     - `"properties"`: Defines the name, type, and description of each parameter.
     - `"required"`: Specifies the list of required parameters as an array.

3. **`TOOLS` is an array.** Multiple tools can be defined and provided to the AI.

#### Passing Tools to the API Call

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    print(response)
    message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": message})
    print(f"AI: {message}")
```

**Change:** The `tools=TOOLS` parameter has been added. Now the AI can invoke appropriate functions during the conversation.

#### Execution Result Analysis

For normal conversation:
```
User: my name is nico
AI: Nice to meet you, Nico! How can I assist you today?
```
- `finish_reason` is `'stop'` -- the AI responded with normal text.
- `tool_calls` is `None`.

For a question requiring a tool:
```
User: what is the weather in Spain
AI: None
```
- `finish_reason` has changed to `'tool_calls'`!
- The `tool_calls` array contains information about the function to call:
  ```python
  tool_calls=[
      ChatCompletionMessageToolCall(
          id='call_yTID1R7DPur7eJMWlobM8tgu',
          function=Function(
              arguments='{"city":"Spain"}',
              name='get_weather'
          ),
          type='function'
      )
  ]
  ```
- `message.content` is `None` -- because the AI chose to make a tool call instead of generating text.

**This is the key point:** Instead of directly answering "tell me the weather," the AI requested "please call the get_weather function with the argument city='Spain'." However, since there is no code to handle this request yet, `AI: None` is printed.

### Practice Points

1. Add a `get_news` function schema to `TOOLS`
2. Alternate between questions requiring a tool and those not requiring one, observing changes in `finish_reason`
3. Modify the `description` and experiment with how the AI's tool selection behavior changes
4. Print the `response` object in detail to inspect the `tool_calls` structure yourself

---

## 2.5 Adding Function Calling

### Topic and Objective

Implement the `process_ai_response` function that actually processes tool calls requested by the AI. Write branching logic that executes the corresponding function when the AI response contains `tool_calls`, and otherwise processes it as a normal text response.

### Core Concepts

#### The Complete Flow of Function Calling

```
User Question -> AI Judgment -> tool_calls Response -> Function Execution -> Add Result to messages -> Re-call AI -> Final Response
```

In this section, we implement up to "Function Execution" and "Add Result to messages." The "Re-call AI" part is completed in the next section (2.6).

#### Structure of the tool_calls Response

When the AI decides to use a tool, the response's `message` object includes the following information:

```python
message.tool_calls = [
    ChatCompletionMessageToolCall(
        id='call_yTID1R7DPur7eJMWlobM8tgu',    # Unique ID
        function=Function(
            name='get_weather',                   # Function name to call
            arguments='{"city":"Spain"}'           # Arguments as JSON string
        ),
        type='function'
    )
]
```

Key points:
- `id`: Each tool call is assigned a unique ID. This ID is used later to match which call the result belongs to.
- `arguments`: This is a **JSON string**, not a Python dictionary, so it must be parsed with `json.loads()`.
- `tool_calls` is an **array**. The AI can call multiple functions at once.

### Code Analysis

#### Adding the Import

```python
import openai, json
```

The `json` module has been added. It is needed to parse the function arguments returned by the AI as JSON strings.

#### process_ai_response Function (Core Logic)

```python
from openai.types.chat import ChatCompletionMessage


def process_ai_response(message: ChatCompletionMessage):
    if message.tool_calls > 0:
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Calling function: {function_name} with {arguments}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

            function_to_run = FUNCTION_MAP.get(function_name)

            result = function_to_run(**arguments)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result,
                }
            )
    else:
        messages.append({"role": "assistant", "content": message.content})
        print(f"AI: {message.content}")
```

**Step-by-Step Detailed Analysis:**

**Step 1: Branch Decision**
```python
if message.tool_calls > 0:
```
Check if the AI response contains `tool_calls`. If yes, execute the tool call logic; otherwise, execute the normal text response logic.

> Note: This conditional is later modified to `if message.tool_calls:` in section 2.6. This is because `None > 0` can raise a `TypeError` in Python. A truthy/falsy check is safer.

**Step 2: Add the assistant message to the history**
```python
messages.append(
    {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ],
    }
)
```

Record the AI's tool call response in the `messages` array. **This is critically important.** The OpenAI API looks at this history on the next call to understand that a tool call was made. Since `content` can be `None`, we use `message.content or ""` to default to an empty string.

**Step 3: Iterate through each tool call and execute**
```python
for tool_call in message.tool_calls:
    function_name = tool_call.function.name      # "get_weather"
    arguments = tool_call.function.arguments      # '{"city":"Spain"}'

    print(f"Calling function: {function_name} with {arguments}")

    try:
        arguments = json.loads(arguments)         # {"city": "Spain"}
    except json.JSONDecodeError:
        arguments = {}

    function_to_run = FUNCTION_MAP.get(function_name)  # get_weather function object

    result = function_to_run(**arguments)          # get_weather(city="Spain")
```

- `json.loads()`: Converts the JSON string to a Python dictionary
- `try/except`: Defensive coding to handle JSON parsing failure
- `FUNCTION_MAP.get()`: Finds the actual function by its string name
- `**arguments`: Unpacks the dictionary as keyword arguments. `{"city": "Spain"}` becomes `city="Spain"`

**Step 4: Add tool execution result to history**
```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": function_name,
        "content": result,
    }
)
```

- `"role": "tool"`: Indicates this message is a tool execution result
- `"tool_call_id"`: The ID matching which tool call this result corresponds to. Omitting this causes an API error
- `"content"`: The function execution result (in this case, "33 degrees celcius.")

#### Supplementary Code for Understanding the ** Operator (Unpacking)

The commit includes experimental code to understand the `**` operator:

```python
a = '{"city": "Spain"}'

b = json.loads(a)    # b = {"city": "Spain"}

**b                   # Unpacking: city="Spain"

get_weather(city='Spain')
```

This code is a learning example showing how a JSON string is converted into function call arguments.

#### Simplified call_ai Function

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

The response processing logic from the previous `call_ai` has been separated into `process_ai_response` for cleaner code organization.

### Practice Points

1. Add `print(messages)` at each step within `process_ai_response` to track how the history changes
2. Check what error occurs when the AI tries to call a function not in `FUNCTION_MAP` (hint: `None(**arguments)` raises a `TypeError`)
3. Add a `get_currency` function and schema to extend the agent to support multiple tools

---

## 2.6 Tool Results

### Topic and Objective

Pass tool execution results back to the AI so it can generate a natural final response based on those results. This completes the **agent loop**.

### Core Concepts

#### Agent Loop

A complete AI agent forms the following loop:

```
User Question
    |
AI Judgment ---> If normal response -> Output text to user (loop ends)
    |
If tool call is needed
    |
Execute function -> Add result to messages
    |
Re-call AI (call call_ai again)
    |
AI Judgment ---> If normal response -> Output text to user (loop ends)
    |
If another tool call is needed -> Execute function again... (repeat)
```

The key to this loop is that **the AI is called again after tool execution**. The AI receives the tool execution result and formats it appropriately for the user.

#### Problem with the Previous Section (2.5)

In 2.5, we only implemented executing tools and adding results to `messages`. However, we did not pass those results back to the AI, so the AI could not generate a final answer. This section resolves that.

### Code Analysis

#### Modified process_ai_response (Final Version)

```python
from openai.types.chat import ChatCompletionMessage


def process_ai_response(message: ChatCompletionMessage):

    if message.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(f"Calling function: {function_name} with {arguments}")

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

            function_to_run = FUNCTION_MAP.get(function_name)

            result = function_to_run(**arguments)

            print(f"Ran {function_name} with args {arguments} for a result of {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result,
                }
            )

        call_ai()
    else:
        messages.append({"role": "assistant", "content": message.content})
        print(f"AI: {message.content}")


def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

**Three Key Changes:**

**1. Conditional Fix: `message.tool_calls > 0` -> `message.tool_calls`**

```python
# Before (2.5)
if message.tool_calls > 0:

# After (2.6)
if message.tool_calls:
```

In Python, `None > 0` can raise a `TypeError`. Since `tool_calls` is falsy when `None` and truthy when a list exists, this approach is safer.

**2. Debug Output Added**

```python
print(f"Ran {function_name} with args {arguments} for a result of {result}")
```

Prints the function execution result to the console to aid debugging.

**3. Re-call AI After Tool Execution (The Most Important Change)**

```python
        # After all tools have been executed
        call_ai()
```

This single line completes the agent loop. After adding all tool results to `messages`, `call_ai()` is called again. The AI then receives the full history including tool execution results and generates the final response.

**Tracing the call flow:**

```
call_ai()                          # 1st call
  -> AI: returns tool_calls
  -> process_ai_response()
    -> Execute tool, add result to messages
    -> call_ai()                    # 2nd call (recursive)
      -> AI: returns normal text response
      -> process_ai_response()
        -> else branch: print text
```

This is a **mutual recursion** pattern: `call_ai` -> `process_ai_response` -> `call_ai` -> `process_ai_response` -> ...

#### Execution Results (Complete Agent Operation)

```
User: My name is Nico
AI: Hello, Nico! How can I assist you today?
User: What is my name
AI: Your name is Nico.
User: What is the weather in Spain
Calling function: get_weather with {"city":"Spain"}
Ran get_weather with args {'city': 'Spain'} for a result of 33 degrees celcius.
AI: The weather in Spain is 33 degrees Celsius. If you need more specific weather details for a particular city or region in Spain, just let me know!
```

**Operation Process Analysis:**

1. For the question "What is the weather in Spain," the AI decided to call `get_weather(city="Spain")`
2. The function executed and returned "33 degrees celcius."
3. This result was added to `messages` and the AI was called again
4. The AI transformed the raw data "33 degrees celcius." into a natural sentence: "The weather in Spain is 33 degrees Celsius. If you need more specific weather details..."

#### Final messages Array Inspection

```python
messages
```

Output:
```python
[
    {'role': 'user', 'content': 'My name is Nico'},
    {'role': 'assistant', 'content': 'Hello, Nico! How can I assist you today?'},
    {'role': 'user', 'content': 'What is my name'},
    {'role': 'assistant', 'content': 'Your name is Nico.'},
    {'role': 'user', 'content': 'What is the weather in Spain'},
    {'role': 'assistant',
     'content': '',
     'tool_calls': [{'id': 'call_za6hozI93riBO1tzf0gdPOwt',
       'type': 'function',
       'function': {'name': 'get_weather', 'arguments': '{"city":"Spain"}'}}]},
    {'role': 'tool',
     'tool_call_id': 'call_za6hozI93riBO1tzf0gdPOwt',
     'name': 'get_weather',
     'content': '33 degrees celcius.'},
    {'role': 'assistant',
     'content': 'The weather in Spain is 33 degrees Celsius. If you need more specific weather details for a particular city or region in Spain, just let me know!'}
]
```

This array shows the agent's entire operation process:
1. Normal conversation (user -> assistant)
2. Tool call request (assistant with tool_calls)
3. Tool execution result (tool)
4. Final response based on the result (assistant)

### Practice Points

1. Add `get_news` and `get_currency` functions, and test with a compound question ("What is the weather and news in Korea?") to see if multiple tools are called simultaneously
2. Think about why the recursive calls do not fall into an infinite loop (hint: once the AI receives tool results, it responds with normal text, falling into the `else` branch)
3. Print the `messages` array to visually confirm how messages of each role (user, assistant, tool) accumulate
4. Deliberately change a tool execution result to an error message and test how the AI reacts

---

## Chapter Key Takeaways

### 1. The Essence of an AI Agent

An AI agent is the combination of **LLM + Tools + Loop**. The LLM makes decisions, tools execute actions, and the loop repeatedly connects them.

### 2. The messages Array IS the Memory

LLMs do not maintain state. Accumulating previous conversations in the `messages` array and sending them each time is the actual substance of "memory."

### 3. Tools Schema Is a Contract with the AI

When tools are defined via JSON Schema, the AI requests tool calls in a structured format. The quality of the `description` directly impacts the AI's decision accuracy.

### 4. The Core Flow of Function Calling

```
User Question -> AI Judgment -> tool_calls Return -> Function Execution -> Add Result to messages -> Re-call AI -> Final Response
```

### 5. Agent Loop Termination Condition

The loop terminates when the AI responds with normal text without any more tool calls. This means the AI itself has determined that "no more tools are needed."

### 6. Key Data Flow Summary

```python
# User message
{"role": "user", "content": "What is the weather in Spain?"}

# AI's tool call request
{"role": "assistant", "content": "", "tool_calls": [...]}

# Tool execution result
{"role": "tool", "tool_call_id": "call_xxx", "name": "get_weather", "content": "33 degrees"}

# AI's final response
{"role": "assistant", "content": "The weather in Spain is 33 degrees Celsius."}
```

---

## Practice Assignments

### Assignment 1: Multi-Tool Agent (Basic)

Implement all three functions -- `get_weather`, `get_news`, `get_currency` -- and define `TOOLS` schemas so the AI selects the appropriate tool based on the question.

**Requirements:**
- Each function may return hardcoded results
- Register all three functions in `FUNCTION_MAP`
- Verify correct responses for questions like "What is the news in Japan?" and "What is the currency of Brazil?"

### Assignment 2: Adding System Prompt (Basic)

Add a `system` role message at the beginning of the `messages` array to change the AI's personality.

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful weather assistant. Always respond in Korean."}
]
```

### Assignment 3: Enhanced Error Handling (Intermediate)

Add error handling for the following situations:
- When the AI calls a function that does not exist in `FUNCTION_MAP`
- When an exception occurs during function execution
- When `json.loads()` fails (basic handling already exists, but improve it to pass the error message to the AI)

**Hint:** When an error occurs, you can put the error message in the `content` of the `"role": "tool"` message to inform the AI.

### Assignment 4: Conversation History Management (Intermediate)

As conversations grow longer, token usage increases. Implement one of the following strategies:
- Remove old messages when the `messages` array exceeds a certain length (but keep `system` messages)
- Estimate total token count and set a limit

### Assignment 5: Real API Integration (Advanced)

Connect the `get_weather` function to a real weather API (e.g., OpenWeatherMap API) to return real-time weather data. Verify that when the function's return value changes, the AI's response changes accordingly.
