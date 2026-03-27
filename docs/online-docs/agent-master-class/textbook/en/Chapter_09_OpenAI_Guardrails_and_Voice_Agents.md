# Chapter 9: OpenAI Agents SDK - Guardrails, Handoffs, and Voice Agents

---

## Chapter Overview

In this chapter, we use the OpenAI Agents SDK (`openai-agents`) to progressively build a production-level **customer support agent system**. Going beyond a simple chatbot, we complete a comprehensive project covering context management, dynamic instructions, input/output guardrails, inter-agent handoffs, lifecycle hooks, and voice agents.

### Learning Objectives

| Section | Topic | Key Keywords |
|------|------|-------------|
| 9.0 | Project Introduction and Basic Structure | Streamlit, SQLiteSession, Runner |
| 9.1 | Context Management | RunContextWrapper, Pydantic Model |
| 9.2 | Dynamic Instructions | Dynamic Instructions, Function-based Prompts |
| 9.3 | Input Guardrails | Input Guardrail, Tripwire |
| 9.4 | Agent Handoffs | Handoff, Specialist Agent Routing |
| 9.5 | Handoff UI | agent_updated_stream_event, Real-time Transition Display |
| 9.6 | Hooks | AgentHooks, Tool Usage Logging |
| 9.7 | Output Guardrails | Output Guardrail, Response Validation |
| 9.8 | Voice Agent I | AudioInput, WAV Conversion |
| 9.9 | Voice Agent II | VoicePipeline, VoiceWorkflowBase, sounddevice |

### Project Structure (Final)

```
customer-support-agent/
├── main.py                      # Streamlit main application
├── models.py                    # Pydantic data models
├── tools.py                     # Agent tool functions
├── output_guardrails.py         # Output guardrail definitions
├── workflow.py                  # Voice agent custom workflow
├── my_agents/
│   ├── triage_agent.py          # Triage (classification) agent
│   ├── technical_agent.py       # Technical support agent
│   ├── billing_agent.py         # Billing support agent
│   ├── order_agent.py           # Order management agent
│   └── account_agent.py         # Account management agent
├── pyproject.toml               # Project dependencies
└── customer-support-memory.db   # SQLite session storage
```

---

## 9.0 Project Introduction and Basic Structure

### Topic and Objective

Combine the OpenAI Agents SDK with Streamlit to create the basic skeleton of a customer support chatbot. Set up a structure that stores conversation history in SQLite and displays streaming responses in real time.

### Key Concepts

**OpenAI Agents SDK** is a framework for developing agent-based applications. It runs agents through `Runner` and can permanently store conversation history with `SQLiteSession`. Streamlit is a web UI framework for rapid prototyping, allowing easy implementation of chat interfaces using `st.chat_message` and `st.chat_input`.

### Code Analysis

**Project Dependency Setup (`pyproject.toml`)**

```toml
[project]
name = "customer-support-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "openai-agents[voice]>=0.2.8",
    "python-dotenv>=1.1.1",
    "streamlit>=1.48.1",
]
```

- `openai-agents[voice]`: OpenAI Agents SDK package including voice agent features
- `python-dotenv`: Loads environment variables such as API keys from `.env` files
- `streamlit`: Web UI framework

**Main Application (`main.py`)**

```python
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import streamlit as st
from agents import Runner, SQLiteSession

client = OpenAI()

# SQLite-based session management
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]
```

**Key Points:**
- `SQLiteSession` takes two arguments: the session name (`"chat-history"`) and the database file path (`"customer-support-memory.db"`)
- `st.session_state` is used to ensure the session object persists across Streamlit re-renders

**Conversation History Display Function:**

```python
async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))

asyncio.run(paint_history())
```

- `session.get_items()` retrieves all stored messages
- The reason for escaping the `$` symbol with `\$` is to prevent Streamlit from interpreting it as a LaTeX formula

**Streaming Agent Execution:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))
```

- `Runner.run_streamed()` runs the agent in streaming mode, generating events as each token is produced
- Within `raw_response_event` type, it detects `response.output_text.delta` to display text in real time by accumulation
- `st.empty()` creates a placeholder whose content can be updated later

### Practice Points

1. Initialize the project using the `uv` package manager and install dependencies
2. Run the app with `streamlit run main.py` and verify the basic chat interface
3. Open the SQLite DB file and examine how conversation history is stored

---

## 9.1 Context Management

### Topic and Objective

Learn how to pass **user context information** during agent execution. Define type-safe contexts with Pydantic models and learn the pattern of accessing this information within tool functions through `RunContextWrapper`.

### Key Concepts

**Context** refers to external information that the agent can reference during execution. For example, the currently logged-in user's ID, name, subscription tier, etc. This information is used in the agent's prompts and tool functions.

`RunContextWrapper` is a generic type. By specifying the context type like `RunContextWrapper[UserAccountContext]`, you can get IDE auto-completion and type checking support.

### Code Analysis

**Context Model Definition (`models.py`)**

```python
from pydantic import BaseModel

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"  # premium, enterprise
```

- Inherits from Pydantic `BaseModel` to automatically handle data validation and serialization
- The `tier` field has a default value of `"basic"`, making it an optional field

**Using Context in Tool Functions (`main.py`)**

```python
from agents import Runner, SQLiteSession, function_tool, RunContextWrapper
from models import UserAccountContext

@function_tool
def get_user_tier(wrapper: RunContextWrapper[UserAccountContext]):
    return (
        f"The user {wrapper.context.customer_id} has a {wrapper.context.tier} account."
    )
```

- The `@function_tool` decorator converts a regular Python function into a tool that the agent can call
- Access the `UserAccountContext` instance passed at runtime through `wrapper.context`

**Context Creation and Passing:**

```python
user_account_ctx = UserAccountContext(
    customer_id=1,
    name="nico",
    tier="basic",
)

# Pass context when running Runner
stream = Runner.run_streamed(
    agent,
    message,
    session=session,
    context=user_account_ctx,  # Context injection
)
```

- Pass the context object via the `context` parameter of `Runner.run_streamed()`
- This context is accessible from all tool functions and instructions of the agent

### Practice Points

1. Add new fields like `phone_number`, `preferred_language` to `UserAccountContext`
2. Create a new `@function_tool` that utilizes context information
3. Implement a tool that returns different responses based on the `tier` value

---

## 9.2 Dynamic Instructions

### Topic and Objective

Learn how to define the agent's instructions not as a **static string** but as a **function**, generating prompts that dynamically change based on the context at execution time.

### Key Concepts

Typically, an agent's `instructions` is a fixed string. However, to include runtime information such as the user's name, tier, or email in the prompt, function-based dynamic instructions are needed. This function receives `RunContextWrapper` and `Agent` objects as arguments and returns a string.

### Code Analysis

**Creating the Triage Agent (`my_agents/triage_agent.py`)**

```python
from agents import Agent, RunContextWrapper
from models import UserAccountContext

def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a customer support agent. You ONLY help customers with their
    questions about their User Account, Billing, Orders, or Technical Support.
    You call customers by their name.

    The customer's name is {wrapper.context.name}.
    The customer's email is {wrapper.context.email}.
    The customer's tier is {wrapper.context.tier}.

    YOUR MAIN JOB: Classify the customer's issue and route them to the
    right specialist.

    ISSUE CLASSIFICATION GUIDE:

    TECHNICAL SUPPORT - Route here for:
    - Product not working, errors, bugs
    - App crashes, loading issues, performance problems
    ...

    BILLING SUPPORT - Route here for:
    - Payment issues, failed charges, refunds
    ...

    ORDER MANAGEMENT - Route here for:
    - Order status, shipping, delivery questions
    ...

    ACCOUNT MANAGEMENT - Route here for:
    - Login problems, password resets, account access
    ...

    SPECIAL HANDLING:
    - Premium/Enterprise customers: Mention their priority status when routing
    - Multiple issues: Handle the most urgent first, note others for follow-up
    - Unclear issues: Ask 1-2 clarifying questions before routing
    """

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,  # Pass function directly
)
```

**Key Points:**
- A **function reference** is passed to the `instructions` parameter instead of a string (without parentheses: `dynamic_triage_agent_instructions`)
- The function signature must be `(wrapper: RunContextWrapper[T], agent: Agent[T]) -> str`
- f-strings are used to insert context values like `wrapper.context.name`, `wrapper.context.tier` into the prompt
- The Triage agent's role is to classify customer inquiries and route them to the appropriate specialist agent

**Changes in main.py:**

In this commit, the `get_user_tier` tool function that was previously in `main.py` has been removed. The approach has shifted from accessing context information through a tool to handling it directly in dynamic instructions.

### Practice Points

1. Modify the dynamic instructions to respond in different tones (formal/informal) based on the `wrapper.context.tier` value
2. Include the current time (`datetime.now()`) in the instructions to add time-based greetings
3. Write dynamic instructions that utilize properties of the agent object (`agent`)

---

## 9.3 Input Guardrails

### Topic and Objective

Implement input guardrails that **automatically check** whether user input falls outside the agent's scope of work. Learn the pattern where a separate "guardrail agent" analyzes the input and blocks the conversation if the request is inappropriate.

### Key Concepts

**Input Guardrail** is a validation step that runs before the agent processes user input. During this process, a separate small agent (guardrail agent) analyzes the input to determine whether it is "off-topic." If inappropriate input is detected, the **Tripwire** triggers an `InputGuardrailTripwireTriggered` exception.

The advantages of this pattern are:
- No need to complicate the main agent's instructions
- Guardrail checks are **executed asynchronously in parallel**, minimizing performance degradation
- Validation logic is separated into a separate module for reusability and testability

### Code Analysis

**Guardrail Output Model (`models.py`)**

```python
from pydantic import BaseModel
from typing import Optional

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"
    email: Optional[str] = None

class InputGuardRailOutput(BaseModel):
    is_off_topic: bool
    reason: str
```

- `InputGuardRailOutput` is the structured output format of the guardrail agent
- `is_off_topic`: Whether the request is outside the scope of work
- `reason`: The basis for the judgment (for debugging and logging)

**Guardrail Agent and Decorator (`my_agents/triage_agent.py`)**

```python
from agents import (
    Agent, RunContextWrapper, input_guardrail,
    Runner, GuardrailFunctionOutput,
)
from models import UserAccountContext, InputGuardRailOutput

# 1. Define a dedicated guardrail agent
input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="""
    Ensure the user's request specifically pertains to User Account details,
    Billing inquiries, Order information, or Technical Support issues, and
    is not off-topic. If the request is off-topic, return a reason for the
    tripwire. You can make small conversation with the user, specially at
    the beginning of the conversation, but don't help with requests that
    are not related to User Account details, Billing inquiries, Order
    information, or Technical Support issues.
    """,
    output_type=InputGuardRailOutput,  # Force structured output
)

# 2. Define the guardrail function
@input_guardrail
async def off_topic_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
    input: str,
):
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=wrapper.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_off_topic,
    )

# 3. Connect guardrail to the triage agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[
        off_topic_guardrail,
    ],
)
```

**Operation Sequence:**
1. User sends a message
2. The `off_topic_guardrail` function executes
3. Internally, the `input_guardrail_agent` analyzes the message
4. Returns the result in `InputGuardRailOutput` format
5. If `is_off_topic` is `True`, `tripwire_triggered=True` is set
6. When the Tripwire triggers, an `InputGuardrailTripwireTriggered` exception is raised

**Exception Handling (`main.py`)**

```python
from agents import Runner, SQLiteSession, InputGuardrailTripwireTriggered

async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                triage_agent,
                message,
                session=session,
                context=user_account_ctx,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))
        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

- `try/except` catches the `InputGuardrailTripwireTriggered` exception and displays an appropriate message to the user

### Practice Points

1. Modify the guardrail agent's instructions to apply stricter/looser filtering
2. Test the guardrail with off-topic messages like "What's the weather today?" and legitimate messages like "I want to change my password"
3. Add functionality to display the `reason` field in the UI to explain why it was blocked

---

## 9.4 Agent Handoffs

### Topic and Objective

Implement a multi-agent structure where the triage agent classifies customer inquiries and hands off (transfers) the conversation to the appropriate **specialist agent**. Create 4 specialist agents (technical support, billing, order, account) and set up the handoff mechanism.

### Key Concepts

**Handoff** is when one agent transfers conversation control to another agent. This is similar to a call center agent transferring a call to a specialized department.

In the OpenAI Agents SDK, handoffs are composed of the following elements:
- `handoff()` function: Defines the handoff configuration
- `on_handoff`: Callback function executed when a handoff occurs
- `input_type`: Schema of data passed during handoff
- `input_filter`: Filter that cleans up the previous agent's tool call history during handoff

### Code Analysis

**Handoff Data Model (`models.py`)**

```python
class HandoffData(BaseModel):
    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
```

This model defines the metadata that the triage agent passes to the specialist agent during handoff.

**Specialist Agent Example - Technical Support (`my_agents/technical_agent.py`)**

```python
from agents import Agent, RunContextWrapper
from models import UserAccountContext

def dynamic_technical_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a Technical Support specialist helping {wrapper.context.name}.
    Customer tier: {wrapper.context.tier}
    {"(Premium Support)" if wrapper.context.tier != "basic" else ""}

    YOUR ROLE: Solve technical issues with our products and services.

    TECHNICAL SUPPORT PROCESS:
    1. Gather specific details about the technical issue
    2. Ask for error messages, steps to reproduce, system info
    3. Provide step-by-step troubleshooting solutions
    4. Test solutions with the customer
    5. Escalate to engineering if needed

    {"PREMIUM PRIORITY: Offer direct escalation to senior engineers
    if standard solutions don't work." if wrapper.context.tier != "basic" else ""}
    """

technical_agent = Agent(
    name="Technical Support Agent",
    instructions=dynamic_technical_agent_instructions,
)
```

- All specialist agents follow the same pattern: dynamic instructions + `Agent` creation
- Additional benefits are communicated to premium customers based on `wrapper.context.tier`

**Billing Agent (`my_agents/billing_agent.py`)**

```python
def dynamic_billing_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a Billing Support specialist helping {wrapper.context.name}.
    Customer tier: {wrapper.context.tier}
    {"(Premium Billing Support)" if wrapper.context.tier != "basic" else ""}

    YOUR ROLE: Resolve billing, payment, and subscription issues.
    ...
    {"PREMIUM BENEFITS: Fast-track refund processing and flexible
    payment options available." if wrapper.context.tier != "basic" else ""}
    """

billing_agent = Agent(
    name="Billing Support Agent",
    instructions=dynamic_billing_agent_instructions,
)
```

**Handoff Configuration (`my_agents/triage_agent.py`)**

```python
import streamlit as st
from agents import (
    Agent, RunContextWrapper, input_guardrail, Runner,
    GuardrailFunctionOutput, handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from models import UserAccountContext, InputGuardRailOutput, HandoffData
from my_agents.account_agent import account_agent
from my_agents.technical_agent import technical_agent
from my_agents.order_agent import order_agent
from my_agents.billing_agent import billing_agent

# Handoff callback: Display handoff info in sidebar
def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
):
    with st.sidebar:
        st.write(f"""
            Handing off to {input_data.to_agent_name}
            Reason: {input_data.reason}
            Issue Type: {input_data.issue_type}
            Description: {input_data.issue_description}
        """)

# Handoff factory function
def make_handoff(agent):
    return handoff(
        agent=agent,
        on_handoff=handle_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[off_topic_guardrail],
    handoffs=[
        make_handoff(technical_agent),
        make_handoff(billing_agent),
        make_handoff(account_agent),
        make_handoff(order_agent),
    ],
)
```

**Key Points:**
- `RECOMMENDED_PROMPT_PREFIX`: A recommended prompt prefix provided by OpenAI for handoffs, which tells the agent how to perform handoffs
- `handoff_filters.remove_all_tools`: Removes the previous agent's tool call history during handoff so the new agent starts with a clean state
- The `make_handoff()` factory function reduces duplicate code
- Handoffs can also be implemented by converting agents to tools using the `as_tool()` method (see commented code)

### Practice Points

1. Add a new specialist agent (e.g., "Returns Specialist Agent") and connect the handoff
2. Replace `input_filter` with a custom filter instead of `handoff_filters.remove_all_tools`
3. Experiment with the differences between the `as_tool()` approach and the `handoff()` approach

---

## 9.5 Handoff UI

### Topic and Objective

Display **real-time transition status in the UI** when inter-agent handoffs occur, and track the currently active agent so that subsequent messages are delivered to the correct agent.

### Key Concepts

Among streaming events, `agent_updated_stream_event` is fired when the agent changes. By detecting this and displaying a transition message in the UI while saving the current agent to `st.session_state`, the user's next message is delivered to the correct specialist agent.

### Code Analysis

**Agent State Tracking (`main.py`)**

```python
# Store the currently active agent in session state
if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent
```

**Detecting Handoffs in Streaming Events:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                st.session_state["agent"],  # Use the currently active agent
                message,
                session=session,
                context=user_account_ctx,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                # Detect agent transition event
                elif event.type == "agent_updated_stream_event":
                    if st.session_state["agent"].name != event.new_agent.name:
                        st.write(
                            f"Transfered from "
                            f"{st.session_state['agent'].name} to "
                            f"{event.new_agent.name}"
                        )
                        # Update current agent to the new agent
                        st.session_state["agent"] = event.new_agent
                        # Initialize placeholder for the new agent's response
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder
                        response = ""

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

**Key Points:**
- `agent_updated_stream_event`: Streaming event fired when the agent changes
- `event.new_agent`: The newly activated agent object
- When the agent changes, `response` is reset and a new `text_placeholder` is created so the new agent's response displays from the beginning
- Updating `st.session_state["agent"]` ensures the user's next message is sent to the new agent

### Practice Points

1. Improve the transition message style during handoffs (e.g., add dividers)
2. Add functionality to always display the currently active agent name in the sidebar
3. Create a "Return to Triage" button to manually reset the agent

---

## 9.6 Hooks

### Topic and Objective

Implement **AgentHooks** that insert custom logic into the agent's **lifecycle events** (start, end, tool execution, handoff). Also add actual business tools to each specialist agent.

### Key Concepts

**Hooks** are collections of callback functions that are automatically called at specific points during agent execution. By inheriting the `AgentHooks` class, you can override the following methods:

| Method | Trigger Point |
|--------|-----------|
| `on_start` | Agent execution start |
| `on_end` | Agent execution complete |
| `on_tool_start` | Just before a tool function is called |
| `on_tool_end` | After a tool function completes |
| `on_handoff` | When handoff to another agent occurs |

### Code Analysis

**Tool Function Examples (`tools.py`)**

In this commit, a 441-line `tools.py` is added. Tool functions for each specialist agent are organized by category.

```python
import streamlit as st
from agents import function_tool, AgentHooks, Agent, Tool, RunContextWrapper
from models import UserAccountContext
import random
from datetime import datetime, timedelta

# === Technical Support Tools ===

@function_tool
def run_diagnostic_check(
    context: UserAccountContext, product_name: str, issue_description: str
) -> str:
    """
    Run a diagnostic check on the customer's product to identify potential issues.
    """
    diagnostics = [
        "Server connectivity: Normal",
        "API endpoints: Responsive",
        "Cache memory: 85% full (recommend clearing)",
        "Database connections: Stable",
        "Last update: 7 days ago (update available)",
    ]
    return f"Diagnostic results for {product_name}:\n" + "\n".join(diagnostics)

@function_tool
def escalate_to_engineering(
    context: UserAccountContext, issue_summary: str, priority: str = "medium"
) -> str:
    """Escalate a technical issue to the engineering team."""
    ticket_id = f"ENG-{random.randint(10000, 99999)}"
    return f"""
Issue escalated to Engineering Team
Ticket ID: {ticket_id}
Priority: {priority.upper()}
Summary: {issue_summary}
Expected response: {2 if context.is_premium_customer() else 4} hours
    """.strip()
```

**Tool Function Design Patterns:**
- Takes `context: UserAccountContext` as the first argument to access user information
- The docstring serves to explain the tool's purpose to the agent
- The `Args` section descriptions are also referenced by the agent
- Different processing based on premium customer status (e.g., differentiated response times)

```python
# === Billing Support Tools ===

@function_tool
def process_refund_request(
    context: UserAccountContext, refund_amount: float, reason: str
) -> str:
    """Process a refund request for the customer."""
    processing_days = 3 if context.is_premium_customer() else 5
    refund_id = f"REF-{random.randint(100000, 999999)}"
    return f"""
Refund request processed
Refund ID: {refund_id}
Amount: ${refund_amount}
Processing time: {processing_days} business days
    """.strip()

# === Order Management Tools ===

@function_tool
def lookup_order_status(context: UserAccountContext, order_number: str) -> str:
    """Look up the current status and details of an order."""
    statuses = ["processing", "shipped", "in_transit", "delivered"]
    current_status = random.choice(statuses)
    tracking_number = f"1Z{random.randint(100000, 999999)}"
    estimated_delivery = datetime.now() + timedelta(days=random.randint(1, 5))
    return f"""
Order Status: {order_number}
Status: {current_status.title()}
Tracking: {tracking_number}
Estimated delivery: {estimated_delivery.strftime('%B %d, %Y')}
    """.strip()

# === Account Management Tools ===

@function_tool
def reset_user_password(context: UserAccountContext, email: str) -> str:
    """Send password reset instructions to the customer's email."""
    reset_token = f"RST-{random.randint(100000, 999999)}"
    return f"""
Password reset initiated
Reset link sent to: {email}
Reset token: {reset_token}
Link expires in: 1 hour
    """.strip()
```

**AgentHooks Implementation:**

```python
class AgentToolUsageLoggingHooks(AgentHooks):

    async def on_tool_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** starting tool: `{tool.name}`")

    async def on_tool_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
        result: str,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** used tool: `{tool.name}`")
            st.code(result)

    async def on_handoff(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        source: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"Handoff: **{source.name}** -> **{agent.name}**")

    async def on_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** activated")

    async def on_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        output,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** completed")
```

**Connecting Tools and Hooks to Agents (e.g., `my_agents/account_agent.py`)**

```python
from tools import (
    reset_user_password,
    enable_two_factor_auth,
    update_account_email,
    deactivate_account,
    export_account_data,
    AgentToolUsageLoggingHooks,
)

account_agent = Agent(
    name="Account Management Agent",
    instructions=dynamic_account_agent_instructions,
    tools=[
        reset_user_password,
        enable_two_factor_auth,
        update_account_email,
        deactivate_account,
        export_account_data,
    ],
    hooks=AgentToolUsageLoggingHooks(),
)
```

**Tools Assigned to Each Specialist Agent:**

| Agent | Tools |
|----------|------|
| Technical | `run_diagnostic_check`, `provide_troubleshooting_steps`, `escalate_to_engineering` |
| Billing | `lookup_billing_history`, `process_refund_request`, `update_payment_method`, `apply_billing_credit` |
| Order | `lookup_order_status`, `initiate_return_process`, `schedule_redelivery`, `expedite_shipping` |
| Account | `reset_user_password`, `enable_two_factor_auth`, `update_account_email`, `deactivate_account`, `export_account_data` |

### Practice Points

1. Record tool execution start time in `on_tool_start` and calculate elapsed time in `on_tool_end` for display
2. Add a hook that displays a confirmation message when a specific tool (e.g., `deactivate_account`) is called
3. Implement a hook that saves tool usage history to a log file

---

## 9.7 Output Guardrails

### Topic and Objective

Implement output guardrails that verify whether the agent's **response** contains content outside the scope of that agent's work. While structurally symmetric to input guardrails, the difference is that it verifies the agent's final output.

### Key Concepts

**Output Guardrail** is a step that verifies whether the agent's response is appropriate after it has been generated. For example, if the technical support agent generates a response that includes billing information or account management information, this is outside its domain and should be blocked.

When the `OutputGuardrailTripwireTriggered` exception is raised, the text already displayed via streaming is removed and replaced with a substitute message.

### Code Analysis

**Output Guardrail Model (`models.py`)**

```python
class TechnicalOutputGuardRailOutput(BaseModel):
    contains_off_topic: bool
    contains_billing_data: bool
    contains_account_data: bool
    reason: str
```

- Checks for multiple types of inappropriate content individually
- Has more granular validation criteria than input guardrails

**Output Guardrail Definition (`output_guardrails.py`)**

```python
from agents import (
    Agent, output_guardrail, Runner,
    RunContextWrapper, GuardrailFunctionOutput,
)
from models import TechnicalOutputGuardRailOutput, UserAccountContext

# Dedicated output validation agent
technical_output_guardrail_agent = Agent(
    name="Technical Support Guardrail",
    instructions="""
    Analyze the technical support response to check if it
    inappropriately contains:

    - Billing information (payments, refunds, charges, subscriptions)
    - Order information (shipping, tracking, delivery, returns)
    - Account management info (passwords, email changes, account settings)

    Technical agents should ONLY provide technical troubleshooting,
    diagnostics, and product support.
    Return true for any field that contains inappropriate content.
    """,
    output_type=TechnicalOutputGuardRailOutput,
)

@output_guardrail
async def technical_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        technical_output_guardrail_agent,
        output,
        context=wrapper.context,
    )

    validation = result.final_output

    # Trigger tripwire if any of the three validation criteria are violated
    triggered = (
        validation.contains_off_topic
        or validation.contains_billing_data
        or validation.contains_account_data
    )

    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )
```

**Comparison Between Input Guardrails and Output Guardrails:**

| Item | Input Guardrail | Output Guardrail |
|------|--------------|--------------|
| Decorator | `@input_guardrail` | `@output_guardrail` |
| Inspection target | User's message | Agent's response |
| Third argument | `input: str` | `output: str` |
| Exception type | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
| Application location | `input_guardrails=[]` | `output_guardrails=[]` |

**Connecting Output Guardrail to Agent (`my_agents/technical_agent.py`)**

```python
from output_guardrails import technical_output_guardrail

technical_agent = Agent(
    name="Technical Support Agent",
    instructions=dynamic_technical_agent_instructions,
    tools=[
        run_diagnostic_check,
        provide_troubleshooting_steps,
        escalate_to_engineering,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[
        technical_output_guardrail,
    ],
)
```

**Exception Handling (`main.py`)**

```python
from agents import (
    Runner, SQLiteSession,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

# Inside run_agent function:
except OutputGuardrailTripwireTriggered:
    st.write("Cant show you that answer.")
    st.session_state["text_placeholder"].empty()  # Remove already displayed text
```

- `text_placeholder.empty()` removes the inappropriate response that was already displayed on screen during streaming

### Practice Points

1. Add output guardrails to the billing agent to verify that technical information is not included
2. Add allowed/prohibited keyword lists to the output guardrail agent's instructions
3. Implement functionality to log the `reason` when a guardrail triggers

---

## 9.8 Voice Agent I

### Topic and Objective

Convert the text-based chat interface to a **voice input** interface. Build a pipeline that records audio with Streamlit's `st.audio_input`, converts WAV format audio to a NumPy array, and passes it to the OpenAI Agents SDK's `AudioInput`.

### Key Concepts

The voice agent operates in the following stages:
1. **Voice Recording**: Record directly in the browser using Streamlit's `st.audio_input` widget
2. **Audio Conversion**: Convert WAV file to NumPy `int16` array
3. **AudioInput Creation**: Wrap the converted array in `AudioInput(buffer=array)` format
4. **Agent Execution**: Pass voice data to the agent

### Code Analysis

**New Dependencies Added (`pyproject.toml`)**

```toml
dependencies = [
    "numpy>=2.3.2",
    "openai-agents[voice]>=0.2.8",
    "python-dotenv>=1.1.1",
    "sounddevice>=0.5.2",
    "streamlit>=1.48.1",
]
```

- `numpy`: For processing audio data as arrays
- `sounddevice`: For audio output (playback)

**Audio Conversion Function (`main.py`)**

```python
from agents.voice import AudioInput
import numpy as np
import wave, io

def convert_audio(audio_input):
    # Convert Streamlit audio input to bytes
    audio_data = audio_input.getvalue()

    # Parse as WAV file and extract frames
    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        audio_frames = wav_file.readframes(-1)  # -1 means all frames

    # Convert to NumPy int16 array
    return np.frombuffer(
        audio_frames,
        dtype=np.int16,
    )
```

**How it works:**
1. `audio_input.getvalue()`: Extract binary data from Streamlit's `UploadedFile` object
2. `wave.open(io.BytesIO(...))`: Open byte data as an in-memory WAV file
3. `wav_file.readframes(-1)`: Read all audio frames (raw PCM data)
4. `np.frombuffer(..., dtype=np.int16)`: Convert PCM data to 16-bit integer array

**Voice Input UI Transition:**

```python
# Use voice input instead of text input
audio_input = st.audio_input(
    "Record your message",
)

if audio_input:
    with st.chat_message("human"):
        st.audio(audio_input)  # Display recorded audio for playback
    asyncio.run(run_agent(audio_input))
```

**Agent Execution Function Changes:**

```python
async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)
            # Completed with VoicePipeline in the next section

            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=user_account_ctx,
            )
        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**What was removed in this commit:**
- `paint_history()` function (conversation history display) - Removed because displaying previous conversations as text is unnatural in a voice agent
- Text-based streaming event processing logic - Replaced with voice pipeline in the next section

### Practice Points

1. Examine the WAV file structure (header, channel count, sample rate) using the `wave` module
2. Print the audio array's shape and dtype to understand the data format
3. Compare `st.audio(audio_input)` with the directly converted array to verify the conversion is correct

---

## 9.9 Voice Agent II

### Topic and Objective

Implement `VoicePipeline` and a custom `VoiceWorkflowBase` to complete the full pipeline: **voice input -> text conversion -> agent processing -> voice output**. Also implement real-time voice output using `sounddevice`.

### Key Concepts

**VoicePipeline** is a voice processing pipeline provided by the OpenAI Agents SDK that automatically handles the following stages:
1. **STT (Speech-to-Text)**: Convert audio to text
2. **Workflow Execution**: Pass converted text to the agent for response generation
3. **TTS (Text-to-Speech)**: Convert agent response to voice

By inheriting **VoiceWorkflowBase** and defining a custom workflow, you can freely customize the agent execution logic.

### Code Analysis

**Custom Workflow (`workflow.py`)**

```python
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
from agents import Runner
import streamlit as st

class CustomWorkflow(VoiceWorkflowBase):

    def __init__(self, context):
        self.context = context

    async def run(self, transcription):
        # Receive text converted by STT (transcription) and run the agent
        result = Runner.run_streamed(
            st.session_state["agent"],
            transcription,
            session=st.session_state["session"],
            context=self.context,
        )

        # Stream agent response in text chunks
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Update the last active agent since handoff may have occurred
        st.session_state["agent"] = result.last_agent
```

**Key Points:**
- Inherits `VoiceWorkflowBase` and implements the `run()` method
- The `run()` method is defined as an **async generator** (using `yield`)
- `transcription`: The result of voice-to-text conversion (STT result)
- `VoiceWorkflowHelper.stream_text_from()`: Utility that extracts text chunks from `Runner` results
- `result.last_agent`: Returns the last activated agent if a handoff occurred. Saving this to `session_state` ensures the correct agent is used for the next voice input

**VoicePipeline Integration (`main.py`)**

```python
from agents.voice import AudioInput, VoicePipeline
from workflow import CustomWorkflow
import sounddevice as sd

async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            # 1. Audio conversion
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)

            # 2. Create custom workflow
            workflow = CustomWorkflow(context=user_account_ctx)

            # 3. Create and run voice pipeline
            pipeline = VoicePipeline(workflow=workflow)

            status_container.update(label="Running workflow", state="running")

            result = await pipeline.run(audio)

            # 4. Set up audio output stream
            player = sd.OutputStream(
                samplerate=24000,  # 24kHz sample rate
                channels=1,        # Mono audio
                dtype=np.int16,    # 16-bit integer
            )
            player.start()

            status_container.update(state="complete")

            # 5. Play voice response in real time
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**Voice Pipeline Operation Sequence:**
1. `convert_audio()`: WAV -> NumPy array
2. `AudioInput(buffer=audio_array)`: Array -> AudioInput object
3. `CustomWorkflow(context=...)`: Create workflow with context
4. `VoicePipeline(workflow=workflow)`: Create pipeline
5. `pipeline.run(audio)`: STT -> Agent execution -> TTS (async)
6. `result.stream()`: Stream TTS result in chunks
7. `player.write(event.data)`: Output each audio chunk to speaker

**Triage Agent Modification (`my_agents/triage_agent.py`)**

```python
def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    SPEAK TO THE USER IN ENGLISH

    {RECOMMENDED_PROMPT_PREFIX}

    You are a customer support agent...
    """

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    # input_guardrails=[
    #     # off_topic_guardrail,
    # ],
    handoffs=[
        make_handoff(technical_agent),
        make_handoff(billing_agent),
        make_handoff(account_agent),
        make_handoff(order_agent),
    ],
)
```

**Changes:**
- `"SPEAK TO THE USER IN ENGLISH"` is added at the beginning of the instructions to specify the TTS output language
- Input guardrails are commented out - because STT conversion results from voice input may not be suitable for the guardrail agent

### Practice Points

1. Change the `samplerate` and check audio quality differences (16000, 24000, 48000)
2. Log the `transcription` value in `CustomWorkflow.run()` to check STT conversion quality
3. Add functionality to save TTS responses to a file for later playback
4. Test whether handoffs work correctly in the voice agent

---

## Chapter Key Summary

### 1. Architecture Pattern

The system implemented in this chapter follows the following hierarchical multi-agent architecture:

```
User Input
    |
    v
[Input Guardrail] -- Inappropriate --> Block Message
    |
    v (Pass)
[Triage Agent] -- Classification --> Handoff
    |           |           |           |
    v           v           v           v
[Technical]  [Billing]  [Order]     [Account]
   |           |           |           |
   v           v           v           v
[Output Guardrail] -- Inappropriate --> Block Message
    |
    v (Pass)
Response to User
```

### 2. Key SDK Component Summary

| Component | Role | Section Used |
|----------|------|-----------|
| `Agent` | Agent definition (instructions, tools, guardrails) | All |
| `Runner.run_streamed()` | Streaming agent execution | All |
| `SQLiteSession` | Permanent conversation history storage | 9.0 |
| `RunContextWrapper` | Execution context access | 9.1+ |
| `@function_tool` | Tool function definition | 9.1, 9.6 |
| `@input_guardrail` | Input validation | 9.3 |
| `@output_guardrail` | Output validation | 9.7 |
| `handoff()` | Inter-agent transition | 9.4 |
| `AgentHooks` | Lifecycle callbacks | 9.6 |
| `VoicePipeline` | Voice processing pipeline | 9.9 |
| `VoiceWorkflowBase` | Custom voice workflow | 9.9 |

### 3. Guardrail Design Principles

- **Input Guardrails**: Block inappropriate requests before agent processing (cost saving, security)
- **Output Guardrails**: Block inappropriate content after agent response (quality assurance, data leak prevention)
- Guardrail agents are separated as lightweight independent agents to maintain separation of concerns
- Specifying Pydantic models in `output_type` enforces structured judgment results

### 4. Handoff Design Principles

- Each specialist agent has a clearly defined area of responsibility
- `handoff_filters.remove_all_tools` cleans up the previous agent's tool history
- `on_handoff` callback logs handoff metadata
- `result.last_agent` or `agent_updated_stream_event` tracks the currently active agent

---

## Practice Exercises

### Exercise 1: Add a New Specialist Agent (Difficulty: Medium)

**Objective**: Add a refund specialist agent.

**Requirements**:
- Create a `my_agents/refund_agent.py` file
- Define dynamic instructions and inform premium customers of additional benefits
- Add 2 or more refund-related tool functions to `tools.py`
- Connect `AgentToolUsageLoggingHooks`
- Add to the triage agent's handoff list
- Add refund-related items to the triage agent's classification guide

### Exercise 2: Extend Output Guardrails (Difficulty: Medium)

**Objective**: Add output guardrails to all specialist agents.

**Requirements**:
- Billing agent: Verify that technical information is not included
- Order agent: Verify that billing or account information is not included
- Account agent: Verify that order or billing information is not included
- Define Pydantic output models for each guardrail in `models.py`

### Exercise 3: Custom Hook System (Difficulty: Hard)

**Objective**: Implement an advanced hook system that collects agent usage statistics.

**Requirements**:
- Track each agent's call count, average response time, and tool usage frequency
- Display statistics data in the sidebar in real time
- Reset statistics when the session is initialized
- Record start time in `on_start` and calculate elapsed time in `on_end`

### Exercise 4: Bidirectional Voice Agent (Difficulty: Hard)

**Objective**: Implement an agent that allows continuous voice conversation.

**Requirements**:
- Automatically start the next recording when voice response playback ends
- Announce via voice which agent the conversation was transferred to when a handoff occurs
- Display conversation history as text in the sidebar (both STT results and agent responses)
- Add functionality to recognize "end conversation" as a voice command to terminate the session

### Exercise 5: Agent Return Mechanism (Difficulty: Hard)

**Objective**: Implement a mechanism to return from specialist agents to the triage agent.

**Requirements**:
- Add a "return to triage" handoff to each specialist agent
- Automatically return to triage when a specialist agent receives a request outside its scope
- Include a summary of the conversation so far in the handoff data when returning
- Visually display return events in the UI
