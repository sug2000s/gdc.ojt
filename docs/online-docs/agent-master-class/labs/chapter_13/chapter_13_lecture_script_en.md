# Chapter 13: LangGraph Fundamentals — Lecture Script

---

## Opening (2 min)

> Alright, today we're learning **LangGraph** from scratch.
>
> So far, the agents we've built used manually coded loops, right?
> LangGraph lets you design them as a **Graph** structure.
>
> A Node is "what to do", an Edge is "what comes next".
> Combine these and you can build complex AI workflows cleanly.
>
> Here's what we'll cover today:
>
> ```
> 13.1 Your First Graph       (Node, Edge, START, END)
> 13.2 Graph State             (Reading/Modifying State)
> 13.4 Multiple Schemas        (Input/Output/Internal separation)
> 13.5 Reducer Functions       (State accumulation)
> 13.6 Node Caching            (CachePolicy)
> 13.7 Conditional Edges       (Dynamic branching)
> 13.8 Send API                (Dynamic parallel processing)
> 13.9 Command                 (In-node routing)
> ```
>
> It looks like a lot, but we'll build up step by step. Let's get started.

---

## 13.0 Setup & Environment (3 min)

### Before running the cell

> First, let's check our environment.
> We load the API key from the `.env` file and check `langgraph` and `langchain` versions.
>
> Packages we need today:
> - `langgraph >= 0.6.6` — the core framework
> - `langchain[openai] >= 0.3.27` — OpenAI integration
> - `grandalf` — graph visualization

### After running the cell

> If the versions print correctly, we're good to go.
> If you get an error, check `uv sync` or `pip install`.

---

## 13.1 Your First Graph (10 min)

### Concept

> LangGraph has three core building blocks:
>
> 1. **State** — shared data across the entire graph. Defined with `TypedDict`.
> 2. **Node** — a function to execute. Receives state as input.
> 3. **Edge** — connections between nodes. Determines execution order.
>
> And two special nodes:
> - `START` — the entry point of the graph
> - `END` — the exit point of the graph
>
> With just these five things, you can build any graph.

### Code walkthrough (before running)

> Let's look at the code.
>
> ```python
> class State(TypedDict):
>     hello: str
> ```
>
> A state with a single string field `hello`.
>
> ```python
> graph_builder = StateGraph(State)
> ```
>
> Pass the state schema to `StateGraph` to create a graph builder.
>
> The three node functions simply `print`. They don't modify state yet.
>
> Then `add_node` registers nodes, and `add_edge` connects them:
>
> ```
> START -> node_one -> node_two -> node_three -> END
> ```
>
> Finally, `compile()` → `invoke()` to run it.
> Let's execute.

### After running

> `node_one`, `node_two`, `node_three` printed in order, right?
> This is the most basic **linear graph**.
>
> In the next cell, `draw_mermaid_png()` visualizes the graph.
> You can see the arrows connected in sequence — that's the graph we built.

### Exercise 13.1 (5 min)

> **Exercise 1**: Remove the `START` edge. What error do you get?
>
> **Exercise 2**: Change the node order. `add_edge` order determines the execution flow.
>
> **Exercise 3**: Instead of `add_node("node_one", node_one)`, try `add_node(node_one)`.
> Does the function name automatically become the node name?

---

## 13.2 Graph State (10 min)

### Concept

> Now the key part. **How nodes read and modify state.**
>
> The rules are simple:
> 1. Nodes receive `state` as input
> 2. Return a dictionary to update the state
> 3. **Default behavior is overwrite**
>
> Keys not returned keep their previous values.

### Code walkthrough

> The state now has two fields: `hello` (string) and `a` (boolean).
>
> `node_one` updates both:
> ```python
> return {"hello": "from node one.", "a": True}
> ```
>
> `node_two` only updates `hello`. It doesn't touch `a`.
> So what happens to `a`? **The previous value `True` is preserved.**
>
> Let's run it.

### After running

> Look at the output:
> ```
> node_one {'hello': 'world'}           ← received initial input
> node_two {'hello': 'from node one.', 'a': True}  ← updated by node_one
> node_three {'hello': 'from node two.', 'a': True} ← a is still True
> ```
>
> See how `a` stays `True`? Because `node_two` and `node_three` didn't return `a`.
> **Only returned keys get overwritten; the rest persist.** That's the default strategy.
>
> Final result: `{'hello': 'from node three.', 'a': True}`

### Exercise 13.2 (5 min)

> **Exercise 1**: Include `"a": False` in the initial input. What does `node_one` see?
>
> **Exercise 2**: Return a key not in `State` from a node. e.g., `return {"unknown": 123}`
>
> **Exercise 3**: Change `a` to `False` in `node_two`. Check what `node_three` sees.

---

## 13.4 Multiple Schemas (10 min)

### Concept

> In real apps, these situations come up a lot:
>
> - The input from the user differs from internal processing data
> - The final output should only expose part of the internal state
> - Some nodes need private state that others shouldn't see
>
> LangGraph lets you **separate three schemas**:
>
> | Parameter | Role |
> |-----------|------|
> | First argument | Internal full state (Private) |
> | `input_schema` | External input shape |
> | `output_schema` | External output shape |

### Code walkthrough

> `PrivateState` is internal (`a`, `b`).
> `InputState` is external input (`hello`).
> `OutputState` is external output (`bye`).
>
> Each node uses a **different schema as its type hint**.
> `node_one` only sees `InputState`, `node_two` sees `PrivateState`.
>
> Let's run it.

### After running

> Check the output:
> ```
> node_one -> {'hello': 'world'}       ← only sees InputState
> node_two -> {}                        ← PrivateState but a, b not set yet
> node_three -> {'a': 1}                ← node_two set a
> node_four -> {'a': 1, 'b': 1}        ← full PrivateState
> {'secret': True}                      ← MegaPrivate
> ```
>
> **Final result: `{'bye': 'world'}`**
>
> `a`, `b`, `secret` don't appear in the output.
> Since `OutputState` only defines `bye`, that's all that's returned.
>
> This matters for API design. Internal processing data doesn't leak out.

### Exercise 13.4 (5 min)

> **Exercise 1**: Remove `output_schema`. How does the return value change?
>
> **Exercise 2**: `invoke({"hello": "world", "extra": 123})` — what happens with a non-existent field?
>
> **Exercise 3**: Think about why this matters from a security perspective.

---

## 13.5 Reducer Functions (10 min)

### Concept

> In 13.2 we said the default strategy is "overwrite", right?
>
> But think about **chat messages**.
> If previous messages disappear every time a new one arrives, that's broken.
> Messages need to **accumulate**.
>
> That's what **Reducer functions** solve.
>
> ```python
> messages: Annotated[list[str], operator.add]
> ```
>
> This single line means:
> "When `messages` is updated, don't overwrite — **append** to the existing list."
>
> `operator.add` performs the `+` operation for lists.

### After running

> Result: `{'messages': ['Hello!', 'Hello, nice to meet you!']}`
>
> The initial input `["Hello!"]` was **combined** with `["Hello, nice to meet you!"]` from `node_one`.
>
> **Without the reducer?** Only `["Hello, nice to meet you!"]` would remain, and `["Hello!"]` would be gone.
>
> In chat apps, reducers are essential — you need to build up conversation history.

### Exercise 13.5 (5 min)

> **Exercise 1**: Add a message in `node_two` as well. Do three messages accumulate?
>
> **Exercise 2**: Remove `Annotated` and use plain `messages: list[str]`. Compare results.
>
> **Exercise 3**: Write a custom reducer. e.g., deduplicate, keep only the last 5.

---

## 13.6 Node Caching (7 min)

### Concept

> Some nodes are expensive to run — LLM calls, external API calls.
> If the same input produces the same result, **cache it**.
>
> ```python
> cache_policy=CachePolicy(ttl=20)  # cache for 20 seconds
> ```
>
> `ttl` is Time-To-Live. After this duration, the cache expires and the node re-executes.

### Running the cell

> `node_two` returns the current time.
> We run it 6 times at 5-second intervals — 30 seconds total.
> With `ttl=20`:
>
> - **Runs 1–4** (0–20s): same time printed — cache hit!
> - **Runs 5–6** (after 20s): new time — cache expired, re-executed
>
> Let's verify. (Takes about 30 seconds)

### After running

> See? The first few runs show the same time, then it changes.
> Very useful for reducing API call costs.

---

## 13.7 Conditional Edges (15 min)

### Concept

> So far, we've only built **linear graphs**. A → B → C.
>
> But in reality, you need "different paths based on conditions".
>
> `add_conditional_edges` determines the next node based on a **routing function's** return value.
>
> ```python
> add_conditional_edges(
>     source_node,
>     routing_function,       # examines state, returns a value
>     {value: target_node}    # return value → node mapping
> )
> ```

### Code walkthrough

> Look at the `decide_path` function:
>
> ```python
> def decide_path(state: State):
>     return state["seed"] % 2 == 0  # True or False
> ```
>
> Even `seed` → `True`, odd → `False`.
>
> Mapping:
> ```python
> {True: "node_one", False: "node_two"}
> ```
>
> So:
> - seed=42 (even) → `True` → `node_one` → `node_two` → ...
> - seed=7 (odd) → `False` → straight to `node_two` → ...
>
> After `node_two`, there's another conditional edge. It reuses the same `decide_path`.

### Running — seed=42

> Even number:
> ```
> node_one -> {'seed': 42}
> node_two -> {'seed': 42}
> node_three -> {'seed': 42}
> ```
> Path: `START → node_one → node_two → node_three → END`.

### Running — seed=7

> Odd number:
> ```
> node_two -> {'seed': 7}
> node_four -> {'seed': 7}
> ```
> Path: `START → node_two → node_four → END`.
>
> Same graph, but completely different paths based on input!
> The graph visualization makes the branching clear.

### Exercise 13.7 (5 min)

> **Exercise 1**: Try various `seed` values and observe how paths change.
>
> **Exercise 2**: Switch to directly returning node names:
> ```python
> def decide_path(state) -> Literal["node_three", "node_four"]:
>     if state["seed"] % 2 == 0:
>         return "node_three"
>     else:
>         return "node_four"
> ```
>
> **Exercise 3**: Design a conditional edge with 3 or more branches.

---

## 13.8 Send API (15 min)

### Concept

> Conditional edges decide "where to go".
> **Send API** goes one step further:
>
> **Run the same node multiple times simultaneously with different inputs.**
>
> ```python
> Send("node_two", word)  # run node_two with word as input
> ```
>
> When the `dispatcher` function returns a list of `Send` objects,
> that many `node_two` instances run **in parallel**.
>
> This is the **Map-Reduce** pattern:
> - Map: split data and process each piece
> - Reduce: gather and merge results (Reducer functions!)

### Code walkthrough

> Key point:
>
> ```python
> def node_two(word: str):  # receives str, not State!
> ```
>
> Values passed via Send API are **custom inputs, not State**.
> Each word is individually passed to `node_two`.
>
> The `output` field has an `Annotated[..., operator.add]` reducer,
> so all parallel execution results are automatically merged.
>
> `dispatcher`:
> ```python
> def dispatcher(state):
>     return [Send("node_two", word) for word in state["words"]]
> ```
> 6 words → 6 `Send` objects → 6 parallel `node_two` executions.

### After running

> Result:
> ```
> hello -> 5 letters
> world -> 5 letters
> how   -> 3 letters
> are   -> 3 letters
> you   -> 3 letters
> doing -> 5 letters
> ```
>
> 6 words processed individually, results merged into the `output` list.
>
> Where do you use this in practice?
> - Summarize multiple documents simultaneously
> - Collect data from multiple APIs at once
> - Process multiple user requests in parallel

### Exercise 13.8 (5 min)

> **Exercise 1**: Increase to 20 words.
>
> **Exercise 2**: Add `time.sleep(1)` to `node_two`. Is total execution time 1 second or 6?
>
> **Exercise 3**: Implement one real-world use case yourself.

---

## 13.9 Command (10 min)

### Concept

> So far:
> - State updates = node returns a dictionary
> - Routing = `add_conditional_edges` + routing function
>
> These two were **separate**.
>
> **Command** combines them **into one**:
>
> ```python
> Command(
>     goto="account_support",       # where to go
>     update={"reason": "..."},     # state update
> )
> ```
>
> Inside a node: "update state + decide next node" in one shot.
> No `add_conditional_edges`, no routing function needed.

### Code walkthrough

> Look at `triage_node`:
>
> ```python
> def triage_node(state) -> Command[Literal["account_support", "tech_support"]]:
>     return Command(
>         goto="account_support",
>         update={"transfer_reason": "The user wants to change password."},
>     )
> ```
>
> **The `Command[Literal[...]]` return type is the key.**
> This type hint lets LangGraph know the possible routes,
> so the graph works without `add_edge`.
>
> Notice there's **no edge** after `triage_node` in the graph setup.
> Command determines the next node at runtime.

### After running

> ```
> account_support running
> Result: {'transfer_reason': 'The user wants to change password.'}
> ```
>
> `triage_node` used Command to:
> 1. Update `transfer_reason`
> 2. Route to `account_support`
>
> The graph visualization shows branching from `triage_node` to both nodes.
> This is possible just from the type hint.

### Exercise 13.9 (5 min)

> **Exercise 1**: Add a condition so it can also route to `tech_support`.
>
> **Exercise 2**: `Command` vs `add_conditional_edges` — what are the pros and cons?
>
> **Exercise 3**: Implement multi-step routing like a real customer support system.

---

## Final Exercises Guide (3 min)

> There are 5 comprehensive exercises at the end of the notebook.
>
> **Exercise 1** (★☆☆): Counter graph — each node increments counter by 1
> **Exercise 2** (★★☆): Chat simulator — accumulate messages with reducer
> **Exercise 3** (★★☆): Age-based routing — conditional edges
> **Exercise 4** (★★★): Uppercase conversion — Send API
> **Exercise 5** (★★★): Support router — Command
>
> Exercises 1–2 are basics, 3–5 are challenges.
> Time allocation: 10 min for easy ones, 15 min for harder ones.

---

## Wrap-up (3 min)

> Let's summarize what we learned today.
>
> | Concept | Key Point |
> |---------|-----------|
> | StateGraph | The backbone of state-based graphs |
> | Node / Edge | What to do / In what order |
> | State | Default is overwrite; Reducer enables accumulation |
> | Multiple Schemas | Separate input/output/internal |
> | CachePolicy | Per-node caching for performance optimization |
> | Conditional Edges | Dynamic branching based on state |
> | Send API | Dynamic parallel execution (Map-Reduce) |
> | Command | Routing + state update in one shot |
>
> These are the fundamental building blocks of LangGraph.
> In the next chapter, we'll use them to build an actual **chatbot**.
>
> Great work today.
