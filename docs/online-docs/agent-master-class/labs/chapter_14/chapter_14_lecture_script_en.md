# Chapter 14: Building a LangGraph Chatbot — Lecture Script

---

## Opening (2 min)

> Alright, in the last chapter we learned the fundamental building blocks of LangGraph.
>
> Today we'll use those to build a **practical chatbot** step by step.
>
> We start with a simple chatbot and add features one at a time:
>
> ```
> 14.0 Basic Chatbot       (MessagesState + init_chat_model)
> 14.1 Tool Node            (Tool Calling + ReAct Pattern)
> 14.2 Memory               (SQLite Checkpointer)
> 14.3 Human-in-the-loop    (interrupt + Command resume)
> 14.4 Time Travel           (State Fork)
> ```
>
> Each step builds on the previous one.
> By the end, we'll have a chatbot with memory, human intervention, and the ability to travel back in time.
>
> Let's get started.

---

## 14.0 Setup & Environment (2 min)

### Before running the cell

> First, let's check our environment.
> We load the API key from the `.env` file and check `langgraph` and `langchain` versions.
>
> Same environment as Chapter 13. No additional packages needed.

### After running the cell

> If the API key and versions print correctly, we're good to go.
> If you get an error, check `uv sync` or `pip install`.

---

## 14.0 Basic Chatbot — MessagesState (8 min)

### Concept

> In Chapter 13, we defined state manually with `TypedDict`.
> For chatbots, there's something more convenient: **`MessagesState`**.
>
> What is `MessagesState`?
> - It comes with a built-in `messages: list` field
> - A **reducer is built in**, so messages accumulate automatically
> - Remember `Annotated[list, operator.add]` from Chapter 13? That's already included
>
> LLM initialization is also simple:
> ```python
> llm = init_chat_model("openai:gpt-4o-mini")
> ```
> `init_chat_model` initializes any LLM in one line using the provider:model_name format.
>
> The graph structure is the simplest possible:
> ```
> START -> chatbot -> END
> ```

### Code walkthrough (before running)

> Let's look at the code.
>
> ```python
> class State(MessagesState):
>     pass
> ```
>
> Just inherit from `MessagesState`. The `messages` field is automatically included.
>
> ```python
> def chatbot(state: State):
>     response = llm.invoke(state["messages"])
>     return {"messages": [response]}
> ```
>
> The `chatbot` node:
> 1. Pulls `messages` from the state and passes them to the LLM
> 2. Adds the LLM response to `messages`
>
> Thanks to the reducer, existing messages don't disappear — the response is appended.
>
> Edges: `START -> chatbot -> END`. The simplest linear graph.
> Let's run it.

### After running

> Look at the output:
> ```
> human: how are you?
> ai: I'm just a computer program, so I don't have feelings...
> ```
>
> The `human` message and `ai` response are stacked in the `messages` list.
>
> This is the most basic chatbot.
> But there's a problem — **conversation history doesn't persist.**
> Once `invoke()` finishes, the state is gone. We'll fix this in 14.2.

### Exercise 14.0 (3 min)

> **Exercise 1**: What happens if you use `TypedDict` with `messages: list` instead of `MessagesState`? Check what happens without the reducer.
>
> **Exercise 2**: Change the provider in `init_chat_model`. Try something like `"anthropic:claude-sonnet-4-20250514"`.
>
> **Exercise 3**: Add a `system_prompt: str` field to `State` and implement a dynamic system prompt.

---

## 14.1 Tool Node — Tool Calling + ReAct Pattern (10 min)

### Concept

> The basic chatbot only answers from LLM knowledge.
> What about real-time weather or database lookups — when you need **external tools**?
>
> That's the **ReAct pattern**:
> 1. The LLM decides "I need to call this tool" and generates tool_calls
> 2. The tool executes and returns results
> 3. The LLM sees the results and generates the final response
>
> Graph structure:
> ```
> START -> chatbot -> [tool_calls?] -> tools -> chatbot -> ... -> END
> ```
>
> Three key components:
> - `@tool` — converts a Python function into an LLM-callable tool
> - `ToolNode` — the node responsible for executing tools
> - `tools_condition` — routes to `tools` if tool_calls exist, otherwise to `END`

### Code walkthrough (before running)

> Let's go through the code step by step.
>
> **Step 1: Define the tool**
> ```python
> @tool
> def get_weather(city: str):
>     """Gets weather in city"""
>     return f"The weather in {city} is sunny."
> ```
> The `@tool` decorator turns this function into an LLM-callable tool.
> **The docstring matters** — the LLM reads it to decide when to use this tool.
>
> **Step 2: Bind tools to the LLM**
> ```python
> llm_with_tools = llm.bind_tools(tools=[get_weather])
> ```
> This tells the LLM "you have access to this tool".
>
> **Step 3: Build the graph**
> ```python
> graph_builder.add_conditional_edges("chatbot", tools_condition)
> graph_builder.add_edge("tools", "chatbot")
> ```
>
> `tools_condition` is a built-in function from LangGraph.
> If the LLM response has `tool_calls`, it returns `"tools"`; otherwise `"__end__"`.
>
> Tool results go back to `chatbot` because the LLM needs to see the results to compose the final answer.
>
> Let's run it.

### After running — question that needs a tool

> ```
> human: what is the weather in machupichu
> ai:                          <- only tool_calls, content is empty
> tool: The weather in Machupichu is sunny.
> ai: The weather in Machupicchu is sunny.
> ```
>
> See the flow?
> 1. User asks about weather
> 2. LLM decides it needs `get_weather` and generates tool_calls
> 3. `tools_condition` detects tool_calls and routes to `tools` node
> 4. `ToolNode` executes `get_weather("Machupichu")`
> 5. Result goes back to `chatbot`, LLM generates the final response
>
> This is the **ReAct loop**. Reasoning + Acting.

### After running — question that doesn't need a tool

> ```
> human: hello, how are you?
> ai: Hello! I'm just a computer program...
> ```
>
> No `tool` message this time.
> The LLM decided it could answer without tools.
> `tools_condition` routed straight to `END`.
>
> **Same graph, different paths depending on input.** Same principle as conditional edges in Chapter 13.

### Exercise 14.1 (3 min)

> **Exercise 1**: Add another tool like `get_time`. Does the LLM choose the right tool for each situation?
>
> **Exercise 2**: Change the tool's docstring. How does it affect the LLM's tool selection behavior?
>
> **Exercise 3**: Replace `tools_condition` with a custom condition function.

---

## 14.2 Memory — SQLite Checkpointer (10 min)

### Concept

> Remember the problem from 14.0? State disappears after `invoke()` finishes.
>
> For a chatbot to remember previous conversations, it needs a **checkpointer**.
>
> What a checkpointer does:
> - Saves state to a database after each node execution
> - For the same session (thread_id), loads previous state and continues
>
> ```python
> conn = sqlite3.connect("memory.db")
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> **`thread_id`** is the key:
> - Same thread_id means the conversation continues (memory preserved)
> - Different thread_id means a completely new conversation
>
> Think of thread_id as a "chat room" in a real messaging app.

### Code walkthrough (before running)

> The code is almost identical to the tool chatbot in 14.1.
> Only two lines changed:
>
> ```python
> conn = sqlite3.connect("memory.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> Just passing `checkpointer` to `compile()` activates memory.
>
> And when calling `invoke()`:
> ```python
> config = {"configurable": {"thread_id": "1"}}
> result = graph.invoke({"messages": [...]}, config=config)
> ```
>
> The `config` includes `thread_id` to identify the session.
> Let's run it.

### After running — thread_id="1", first conversation

> ```
> Hello Alice! How can I assist you today?
> ```
>
> We told it our name. Let's ask about it in the next cell with the same thread_id.

### After running — thread_id="1", continued

> ```
> Your name is Alice.
> ```
>
> **It remembers!** Same thread_id, so the previous conversation is in memory.

### After running — thread_id="2", new conversation

> ```
> I'm sorry, but I don't have access to personal information about you...
> ```
>
> Different thread_id, so it's a **completely new conversation**. It doesn't know the name.
>
> This is the core of the checkpointer:
> - Same thread means memory is preserved
> - Different thread means complete isolation

### After running — get_state_history

> ```
> next: (), messages: 4
> next: ('chatbot',), messages: 3
> next: ('__start__',), messages: 2
> ...
> ```
>
> `get_state_history()` shows all snapshots for that thread.
> Each node execution's before-and-after state is recorded.
>
> Empty `next` means the graph completed at that point,
> `('chatbot',)` means just before the chatbot node executed.
>
> This history becomes important in 14.4 Time Travel.

### Exercise 14.2 (3 min)

> **Exercise 1**: Have multiple conversations with the same thread_id. Does memory persist?
>
> **Exercise 2**: Try streaming execution with `stream_mode="updates"`.
>
> **Exercise 3**: Analyze each snapshot from `get_state_history()` to track state changes.

---

## 14.3 Human-in-the-loop (10 min)

### Concept

> The chatbot we've built so far is **fully automatic**.
> User makes a request, the LLM handles it, done.
>
> But in practice, **humans often need to intervene**:
> - Reviewing AI-generated code
> - Approving important decisions
> - Providing feedback on AI output
>
> LangGraph's solution: **`interrupt()` and `Command(resume=...)`**
>
> The flow:
> 1. During graph execution, `interrupt()` is called and execution pauses
> 2. The user provides feedback
> 3. `Command(resume=feedback)` passes the feedback and resumes execution
>
> Because the checkpointer saves the paused state, we can pick up right where we left off.

### Code walkthrough (before running)

> This example is a **poem-writing chatbot**. The LLM writes poems and receives human feedback.
>
> The key is the `get_human_feedback` tool:
> ```python
> @tool
> def get_human_feedback(poem: str):
>     """Asks the user for feedback on the poem."""
>     feedback = interrupt(f"Here is the poem, tell me what you think\n{poem}")
>     return feedback
> ```
>
> When `interrupt()` is called:
> 1. Graph execution **stops**
> 2. The value passed as argument is displayed to the user
> 3. When resumed with `Command(resume=...)`, that value becomes the return value of `interrupt()`
>
> Look at the LLM's system prompt:
> ```
> ALWAYS ASK FOR FEEDBACK FIRST.
> Only after you receive positive feedback you can return the final poem.
> ```
>
> The LLM must always request feedback after writing a poem, and only return the final result after positive feedback.
> Let's run it.

### After running — Step 1: Request a poem

> ```
> Next: ('tools',)
> ```
>
> `next` is `('tools',)`. That means we're **paused** at the tools node.
> The LLM called `get_human_feedback`, and `interrupt()` halted execution.
>
> Now it's the user's turn to provide feedback.

### After running — Step 2: Negative feedback

> ```python
> Command(resume="It is too long! Make it shorter, 4 lines max.")
> ```
>
> We gave negative feedback, so the LLM revises the poem and asks for feedback again.
> `next` is still `('tools',)` — waiting for another interrupt.

### After running — Step 3: Positive feedback

> ```python
> Command(resume="It looks great!")
> ```
>
> This time we gave positive feedback, so the LLM returns the final poem.
> `next` is `()` — graph complete.
>
> Look at the full conversation flow:
> ```
> human: Please make a poem about Python code.
> ai:                          <- writes poem and calls feedback tool
> tool: It is too long!...     <- human feedback (negative)
> ai:                          <- revises and asks again
> tool: It looks great!        <- human feedback (positive)
> ai: Here's the final poem... <- final result
> ```
>
> **The human is inside the loop.** That's why it's called Human-in-the-loop.

### Exercise 14.3 (3 min)

> **Exercise 1**: Give negative feedback multiple times in a row. How does the LLM react?
>
> **Exercise 2**: Pass a dictionary (structured data) to `Command(resume=...)`.
>
> **Exercise 3**: Design a pipeline with multiple interrupt points, like review then approve then deploy.

---

## 14.4 Time Travel — State Fork (10 min)

### Concept

> We said the checkpointer saves all state, right?
> So **can we go back to a past point in time?**
>
> Yes. That's **Time Travel**.
>
> Key APIs:
> - `get_state_history()` — retrieve all snapshots in chronological order
> - `update_state()` — modify a past checkpoint to create a new branch
>
> Why is this useful?
> - **Debugging**: go back to when an error occurred and analyze the cause
> - **A/B testing**: branch from the same point with different inputs
> - **Rollback**: undo a bad execution and start over

### Code walkthrough (before running)

> We build a simple chatbot and have two conversations.
>
> 1. "I live in Europe. My city is Valencia." followed by a Valencia-related response
> 2. "What are some good restaurants near me?" followed by Valencia-based restaurant recommendations
>
> Then we use `get_state_history()` to view all snapshots,
> and **fork** from a past point by changing "Valencia" to "Zagreb".
>
> Let's run it.

### After running — starting the conversation

> ```
> Valencia is a beautiful city located on the eastern coast of Spain...
> ```
>
> A normal conversation.

### After running — follow-up question

> ```
> Valencia-based response:
> La Pepica — Famous for its paella...
> ```
>
> It recommends restaurants based on Valencia.

### After running — state history

> ```
> Snapshot 0: next=(), messages=4
> Snapshot 1: next=('chatbot',), messages=3
> ...
> Snapshot 5: next=('__start__',), messages=0
> ```
>
> 6 snapshots total. Each one is a specific point during graph execution.
> We're going to find the **snapshot right after the user mentioned their city**.

### After running — fork (Valencia to Zagreb)

> ```python
> graph.update_state(
>     fork_config,
>     {"messages": [HumanMessage(content="I live in Europe. My city is Zagreb.")]},
> )
> result_fork = graph.invoke(None, config=fork_config)
> ```
>
> We update the state using the past snapshot's config (`fork_config`),
> then `invoke(None)` to re-execute from that point.
>
> The result now shows a Zagreb-based response.
> **We forked from a past point in the same conversation.**
>
> The original conversation is untouched. A fork creates a new branch, it doesn't overwrite the existing one.

### Exercise 14.4 (3 min)

> **Exercise 1**: Have a longer conversation, then fork from a midpoint.
>
> **Exercise 2**: Ask different questions in the forked branch and compare results with the original.
>
> **Exercise 3**: Think of real-world scenarios where time travel is useful (A/B testing, debugging, rollback).

---

## Final Exercises Guide (3 min)

> There are 4 comprehensive exercises at the end of the notebook.
>
> **Exercise 1** (★★☆): Multi-tool chatbot — 3 tools: `get_weather`, `get_time`, `get_news`
> **Exercise 2** (★★☆): Conversation persistence — session management with thread_id
> **Exercise 3** (★★★): Code review HITL — human review via interrupt()
> **Exercise 4** (★★★): Time travel A/B test — compare branches from the same point
>
> Exercises 1–2 are basics, 3–4 are challenges.
> Time allocation: 10 min for easy ones, 15 min for harder ones.

---

## Wrap-up (2 min)

> Let's summarize what we learned today.
>
> | Concept | Key Point |
> |---------|-----------|
> | MessagesState | Built-in message reducer, the standard for chatbot state |
> | init_chat_model | One-line initialization with provider:model_name |
> | @tool + ToolNode | Integrate external tools into the graph |
> | tools_condition | Auto-routing based on tool_calls presence |
> | SqliteSaver | DB-based state persistence, session isolation via thread_id |
> | interrupt() | Pause graph execution for human intervention |
> | Command(resume) | Resume execution with feedback |
> | get_state_history | Retrieve full snapshot history |
> | update_state | Create new branches (forks) from past checkpoints |
>
> From a basic chatbot to tools, memory, human intervention, and time travel.
> These patterns are the core building blocks of production AI agents.
>
> Great work today.
