# Chapter 18: Multi-Agent Architectures — Lecture Script

---

## Opening (2 min)

> Alright, today we're learning **Multi-Agent Architectures**.
>
> Until now, a single agent handled everything, right?
> But in reality, you often need **multiple agents working together**.
>
> For example, if a customer asks in Korean, a Korean agent responds.
> If they ask in Spanish, a Spanish agent takes over.
>
> Here's what we'll cover today:
>
> ```
> 18.1 Network Architecture    (P2P — agents hand off autonomously)
> 18.2 Supervisor Architecture  (central coordinator routes)
> 18.3 Supervisor as Tools      (agents encapsulated as tools)
> ```
>
> We'll solve the same problem with three different architectures.
> You'll experience each one and compare their trade-offs. Let's begin.

---

## 18.0 Setup & Environment (3 min)

### Before running the cell

> First, let's check our environment.
> We load the API key from the `.env` file and check `langgraph` and `langchain` versions.
>
> Key packages needed today:
> - `langgraph >= 1.1` — multi-agent support
> - `langchain >= 1.2` — LLM integration
> - `langchain-openai` — OpenAI models

### After running the cell

> If the versions print correctly, we're good to go.
> If you get an error, check `uv sync` or `pip install`.

---

## 18.1 Network Architecture — P2P Agent Handoff (15 min)

### Concept

> The first architecture is the **Network (P2P)** approach.
>
> There's no central coordinator. Each agent **autonomously decides**
> when to hand off the conversation to another agent.
>
> ```
> korean_agent ◄──► greek_agent
>       ▲               ▲
>       └──► spanish_agent ◄──┘
> ```
>
> Three key concepts:
>
> 1. **`handoff_tool`** — the handoff tool. Uses `Command(goto=..., graph=Command.PARENT)` to switch agents at the parent graph level.
> 2. **`make_agent()` factory** — creates agents with identical structure but different parameters.
> 3. **Subgraphs** — each agent runs its own independent ReAct loop.
>
> `Command.PARENT` is the key. It lets a subgraph jump to another node in the parent graph.

### Code walkthrough (before running)

> Let's look at the code. It's organized in 4 steps.
>
> **Step 1 — State definition:**
> ```python
> class AgentsState(MessagesState):
>     current_agent: str
>     transfered_by: str
> ```
> We extend `MessagesState` to track the current agent and who transferred.
>
> **Step 2 — `make_agent()` factory:**
> Each agent has a different prompt and tools but the same structure.
> `agent → tools_condition → tools → agent` loop.
> This is the ReAct pattern we've seen before.
>
> **Step 3 — `handoff_tool`:**
> ```python
> return Command(
>     update={"current_agent": transfer_to},
>     goto=transfer_to,
>     graph=Command.PARENT,  # Switch at parent graph level!
> )
> ```
> Without `graph=Command.PARENT`, the transition would stay inside the subgraph.
> With it, we jump to another agent node in the parent graph.
>
> There's also infinite loop protection — an agent can't transfer to itself.
>
> **Step 4 — Top-level graph:**
> Each agent is registered as a node with `destinations` specifying possible transfer targets.
> `START → korean_agent` sets the default entry point.
>
> Let's run it.

### After running — Korean message

> We sent "Hello! I have an account issue" in Korean.
>
> `korean_agent` handled it directly. No handoff needed.
> A Korean message to a Korean agent — naturally, it handles it itself.

### After running — Spanish message

> Now a Spanish message: "Hola! Necesito ayuda con mi cuenta."
>
> Look at the output:
> ```
> [korean_agent] current_agent=spanish_agent    ← detected, handing off!
> [spanish_agent] current_agent=spanish_agent    ← responds in Spanish
> ```
>
> `korean_agent` detected Spanish and called `handoff_tool`
> to transfer to `spanish_agent`. An autonomous decision.
>
> This is the hallmark of the Network architecture.
> Each agent decides on its own: "If I can't handle this, I'll pass it along."

### Exercise 18.1 (5 min)

> **Exercise 1**: Send a Greek message and observe the handoff flow.
>
> **Exercise 2**: Add a Japanese agent. Remember to update the `handoff_tool` docstring too.
>
> **Exercise 3**: Remove `Command.PARENT` and see what error occurs.

---

## 18.2 Supervisor Architecture — Central Coordinator (15 min)

### Concept

> The second architecture is the **Supervisor** approach.
>
> In the Network, every agent had routing logic.
> The Supervisor approach is different. **One central coordinator** handles all routing.
>
> ```
>          Supervisor
>         /    |     \
>    korean  greek  spanish
>         \    |     /
>          Supervisor  ← comes back
> ```
>
> Agents focus solely on their role. They don't need to know about routing.
>
> Key concepts:
> - **Structured Output** — `SupervisorOutput(next_agent, reasoning)` for safe routing
> - **`Literal` type** — restricts possible values to prevent invalid routing
> - **Circular graph** — agent → supervisor → agent → ... → `__end__`
> - **`reasoning` field** — tracks why an agent was selected (great for debugging)

### Code walkthrough (before running)

> Let's look at the code.
>
> **Supervisor output schema:**
> ```python
> class SupervisorOutput(BaseModel):
>     next_agent: Literal["korean_agent", "spanish_agent", "greek_agent", "__end__"]
>     reasoning: str
> ```
> `Literal` restricts possible values. The LLM can't return garbage.
> `__end__` allows the conversation to terminate.
>
> **Agent factory — `make_agent()`:**
> Compare this with 18.1. There's no `tools` parameter!
> Agents don't need handoff tools. They simply respond.
> The structure is much simpler.
>
> **Supervisor node:**
> ```python
> structured_llm = llm.with_structured_output(SupervisorOutput)
> ```
> `with_structured_output` forces the LLM to respond in `SupervisorOutput` format.
> Then `Command(goto=response.next_agent)` handles the routing.
>
> **Graph structure — circular:**
> ```
> START → supervisor → agent → supervisor → agent → ... → END
> ```
> After an agent responds, it goes back to the supervisor.
> When the supervisor returns `__end__`, the conversation ends.
>
> Let's run it.

### After running — Korean message

> Look at the output:
> ```
> Supervisor → korean_agent (reason: ...)
> ```
>
> The supervisor detected Korean and routed to `korean_agent`.
> The `reasoning` field shows why. Very useful for debugging.
>
> After the agent responds, it goes back to the supervisor,
> which then returns `__end__` to finish the conversation.
>
> Compared to Network: agent code is much simpler.
> But the supervisor carries all the decision-making burden.

### After running — Spanish message

> Now Spanish. The supervisor routes to `spanish_agent`.
>
> The agent doesn't need to determine whether it's the right one for the language.
> The supervisor handles all of that. This is the benefit of separation of concerns.

### Exercise 18.2 (5 min)

> **Exercise 1**: Print the `reasoning` field to analyze the supervisor's decision-making.
>
> **Exercise 2**: When adding an agent, compare what code changes are needed in Network vs Supervisor.
>
> **Exercise 3**: Remove the `__end__` option from `SupervisorOutput` and see what happens.

---

## 18.3 Supervisor as Tools — Agents as Tool Functions (10 min)

### Concept

> The third architecture. **The cleanest structure** of all.
>
> Core idea: make agents into `@tool` functions.
> The supervisor uses `bind_tools` to bind agent tools,
> and invokes them through LLM tool calling naturally.
>
> ```
> Supervisor ──tools_condition──► ToolNode
>                                ├ korean_agent
>                                ├ greek_agent
>                                └ spanish_agent
> ```
>
> No separate routing logic needed. The LLM picks the right tool on its own.
> No Structured Output, no Command, no handoff_tool.
>
> Adding an agent = adding one `@tool` function. That's it.

### Code walkthrough (before running)

> Let's look at the code. It's surprisingly simple.
>
> **Agents = @tool functions:**
> ```python
> @tool
> def korean_agent(message: str) -> str:
>     """Transfer to Korean customer support agent."""
>     response = llm.invoke(...)
>     return response.content
> ```
>
> The docstring matters. The LLM reads this description to decide when to call the tool.
>
> **Supervisor:**
> ```python
> llm_with_tools = llm.bind_tools(agent_tools)
> ```
> Bind agent tools to the LLM. The system prompt says "route to the appropriate language agent."
>
> **Graph — ReAct structure:**
> ```
> START → supervisor → tools_condition → ToolNode → supervisor → ... → END
> ```
>
> This is actually the same ReAct pattern from Chapter 15.
> The only difference is that the tools are **agents** instead of regular functions.
>
> Let's run it.

### After running — Korean message

> Look at the result:
> ```
> 고객님, 비밀번호 변경을 도와드리겠습니다...
> ```
>
> The supervisor called the `korean_agent` tool,
> and the response came back in Korean.
>
> Compare with 18.2 — the code is less than half the size.
> No Structured Output schema, no Command, no circular graph setup.

### After running — Spanish message

> Same for Spanish. The `spanish_agent` tool gets called.
>
> The LLM naturally decides "this customer speaks Spanish,
> so I'll call spanish_agent" through tool calling.
>
> You can feel the advantage of this architecture — least code, cleanest structure.

### Exercise 18.3 (5 min)

> **Exercise 1**: Add a Japanese agent as a `@tool`. Notice how simple the code change is.
>
> **Exercise 2**: Compare the three architectures (Network, Supervisor, Supervisor+Tools):
> - Routing method, agent complexity, scalability, debuggability
>
> **Exercise 3**: Design a real-world scenario beyond language routing (e.g., department routing, tech stack classification).

---

## Architecture Comparison Summary (3 min)

> Let's compare the three architectures.
>
> | | Network (18.1) | Supervisor (18.2) | Supervisor+Tools (18.3) |
> |--|---------|------------|------------------|
> | **Routing** | Each agent autonomous | Central supervisor | LLM tool calling |
> | **Agent complexity** | High (handoff logic) | Low (respond only) | Minimal (@tool function) |
> | **Scalability** | Modify all agents | Modify supervisor only | Just add a tool |
> | **Debugging** | Difficult | reasoning tracking | tool call tracking |
> | **Best for** | Few agents, autonomy needed | Medium scale, control needed | Large scale, clean structure |
>
> In practice, **Supervisor+Tools (18.3)** is the most commonly used.
> But Network is good when agents need autonomous collaboration,
> and Supervisor is appropriate when you need fine-grained routing control.
>
> Choosing the right architecture for the situation is what matters.

---

## Final Exercises Guide (2 min)

> There are 3 comprehensive exercises at the end of the notebook.
>
> **Exercise 1** (★★☆): 4-language Network Architecture — add a Japanese agent
> **Exercise 2** (★★★): Department-based Supervisor routing — billing, tech, general
> **Exercise 3** (★★★): Supervisor+Tools with specialized tools — weather, calculator, search
>
> Budget 15 minutes for Exercise 1, 20 minutes each for Exercises 2-3.

---

## Closing (2 min)

> Let's wrap up what we learned today.
>
> | Concept | Key takeaway |
> |---------|-------------|
> | Network (P2P) | `handoff_tool` + `Command.PARENT` for autonomous handoff |
> | Supervisor | Structured Output for central routing, circular graph |
> | Supervisor+Tools | Agents as `@tool`, reuses ReAct pattern |
> | `make_agent()` | Agent factory — same structure, different parameters |
> | `Command.PARENT` | Jump from subgraph to parent graph |
> | `destinations` | Declare possible transfer targets in graph |
>
> Multi-agent is the crown jewel of LangGraph.
> When building agent systems in production, you'll use one of these three patterns.
>
> Great work today.
