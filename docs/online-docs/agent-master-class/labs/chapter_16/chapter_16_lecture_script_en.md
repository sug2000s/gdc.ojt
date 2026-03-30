# Chapter 16: Workflow Architecture Patterns — Lecture Script

---

## Opening (2 min)

> Alright, today we're learning **5 Workflow Architecture Patterns**.
>
> In the previous chapter, we learned the basic building blocks of LangGraph.
> Today, we combine them into **patterns you'll use constantly in real projects**.
>
> Here's what we'll cover:
>
> ```
> 16.1 Prompt Chaining        (Sequential LLM calls)
> 16.2 Prompt Chaining + Gate (Quality check with retry)
> 16.3 Routing                (Dynamic routing by difficulty)
> 16.4 Parallelization        (Parallel execution + aggregation)
> 16.5 Orchestrator-Workers   (Dynamic task distribution)
> ```
>
> Each pattern builds on the previous one.
> Chaining is the foundation, then we add Gate, Routing, Parallelization,
> and finally combine everything into Orchestrator-Workers.
>
> Let's get started.

---

## 16.0 Setup & Environment (2 min)

### Before running the cell

> First, let's check our environment.
> We load API keys and model names from the `.env` file.
>
> Packages we need today:
> - `langgraph >= 1.1` — core framework
> - `langchain >= 1.2` — LLM integration
> - `pydantic` — structured output schemas

### After running the cell

> If `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL_NAME` print correctly, we're ready.
> The next cell also verifies `langgraph` and `langchain` versions.

---

## 16.1 Prompt Chaining — Sequential Chaining (10 min)

### Concept

> This is the most fundamental pattern. **Chain multiple LLM calls sequentially.**
>
> The output of each step becomes the input for the next.
>
> ```
> START → list_ingredients → create_recipe → describe_plating → END
> ```
>
> We'll use a cooking recipe as our example:
> 1. Dish name → generate **ingredient list**
> 2. Ingredient list → generate **cooking instructions**
> 3. Cooking instructions → generate **plating description**
>
> The key technique is `with_structured_output()`.
> In the first node, we **force the LLM response into a Pydantic model**
> so the next node receives a guaranteed structure.

### Code walkthrough (before running)

> Let's look at the code.
>
> First, the State definition:
> ```python
> class State(TypedDict):
>     dish: str                    # input: dish name
>     ingredients: list[dict]      # step 1 output
>     recipe_steps: str            # step 2 output
>     plating_instructions: str    # step 3 output
> ```
>
> Then we define a Pydantic model for structured output:
> ```python
> class Ingredient(BaseModel):
>     name: str
>     quantity: str
>     unit: str
> ```
>
> The key part in the `list_ingredients` node:
> ```python
> structured_llm = llm.with_structured_output(IngredientsOutput)
> ```
>
> This forces the LLM to respond with **exact JSON structure** instead of free text.
> `name`, `quantity`, `unit` are guaranteed to be present.
>
> The other two nodes (`create_recipe`, `describe_plating`) use regular `invoke()`.
> They include the previous step's result in the prompt.
>
> The graph setup is straightforward — `add_edge` to connect in sequence.
> Let's run it.

### After running

> Let's look at the result.
>
> In the `=== Ingredients ===` section, ingredients are output structurally:
> `chickpeas: 1 can`, `tahini: 1/4 cup`, etc.
> That's thanks to `with_structured_output()`.
>
> Below that, Recipe and Plating are free text.
> The key point is that each step used the previous step's result as input.
>
> **The advantage of Chaining**: breaking complex tasks into steps
> makes each prompt simpler and improves overall output quality.

### Exercise 16.1 (3 min)

> **Exercise 1**: Change `dish` to `"bibimbap"` or `"sushi"` and compare results.
>
> **Exercise 2**: Compare the difference between a node using `with_structured_output()` and one using plain `invoke()`.
>
> **Exercise 3**: Add a 4th step (e.g., wine pairing recommendation).

---

## 16.2 Prompt Chaining + Gate — Conditional Gate (8 min)

### Concept

> We add **quality verification** to the 16.1 Chaining pattern.
>
> If the LLM output doesn't meet the criteria, **re-execute the previous step**.
>
> ```
> START → list_ingredients → [gate: 3~8?] → Yes → create_recipe → ...
>                              ↑              → No ─┘ (retry)
> ```
>
> The gate function is simple: return `True` to pass, `False` to retry.
>
> This is a practical application of the **conditional edges** from 13.7.
> In `add_conditional_edges`, `True` goes to the next step, `False` loops back to itself.

### Code walkthrough (before running)

> Almost the same code as 16.1, but with a gate function added:
>
> ```python
> def gate(state: State):
>     count = len(state["ingredients"])
>     if count > 8 or count < 3:
>         print(f"  GATE FAIL: {count} ingredients (need 3-8). Retrying...")
>         return False
>     print(f"  GATE PASS: {count} ingredients")
>     return True
> ```
>
> Fewer than 3 or more than 8 ingredients means failure — calls `list_ingredients` again.
>
> The key part of the graph:
> ```python
> graph_builder.add_conditional_edges(
>     "list_ingredients",
>     gate,
>     {True: "create_recipe", False: "list_ingredients"},
> )
> ```
>
> When `False`, it loops back to itself — creating a **retry loop**.
> Let's run it.

### After running

> `Generated 8 ingredients` → `GATE PASS: 8 ingredients`
>
> It passed on the first try this time. Since the prompt asks for "5-8", it usually passes.
>
> But if you tighten the gate condition to `len == 5`?
> You'll see multiple retries.
>
> **Warning**: there's a risk of infinite loops.
> In production, always add a `retry_count` field to State and cap the maximum retries.

### Exercise 16.2 (3 min)

> **Exercise 1**: Change the gate condition to `len(ingredients) == 5` and observe retry counts.
>
> **Exercise 2**: Add a `retry_count` field to State and limit retries to 3.
>
> **Exercise 3**: Think about scenarios where the gate pattern is useful — code syntax validation, translation language verification, required field checks, etc.

---

## 16.3 Routing — Dynamic Routing (10 min)

### Concept

> The third pattern. **Route to different paths based on input.**
>
> The LLM assesses question difficulty and routes to different model nodes:
>
> ```
> START → assess_difficulty → easy   → dumb_node   (GPT-3.5)
>                           → medium → average_node (GPT-4o)
>                           → hard   → smart_node   (GPT-5)
> ```
>
> Two key techniques:
>
> 1. **Structured Output + Literal**: Constrain LLM response to exactly `"easy"`, `"medium"`, or `"hard"`
> 2. **Command**: From 13.9 — routing + state update in one shot
>
> This is extremely useful for cost optimization in production.
> No need to use an expensive model for easy questions.

### Code walkthrough (before running)

> Look at `DifficultyResponse`:
> ```python
> class DifficultyResponse(BaseModel):
>     difficulty_level: Literal["easy", "medium", "hard"]
> ```
>
> `Literal` forces the LLM to return only one of these three values.
>
> The `assess_difficulty` node:
> ```python
> def assess_difficulty(state: State):
>     structured_llm = llm.with_structured_output(DifficultyResponse)
>     response = structured_llm.invoke(...)
>     level = response.difficulty_level
>     goto_map = {"easy": "dumb_node", "medium": "average_node", "hard": "smart_node"}
>     return Command(goto=goto_map[level], update={"difficulty": level})
> ```
>
> `Command` handles both **where to go** and **state update** simultaneously.
> No `add_conditional_edges` needed.
>
> Notice the new `destinations` parameter in graph setup:
> ```python
> graph_builder.add_node(
>     "assess_difficulty", assess_difficulty,
>     destinations=("dumb_node", "average_node", "smart_node"),
> )
> ```
>
> This tells LangGraph the possible destinations when using Command.
>
> We'll test with two questions — one easy, one hard.

### Running — easy question

> `"What is the capital of France?"` — capital of France.
>
> ```
> Difficulty: easy → dumb_node
> Model: gpt-3.5 (simulated)
> Answer: The capital of France is Paris.
> ```
>
> Simple question, assessed as `easy`, routed to `dumb_node`.

### Running — hard question

> `"Explain the economic implications of quantum computing on global supply chains"`
>
> ```
> Difficulty: hard → smart_node
> Model: gpt-5 (simulated)
> ```
>
> Complex question, assessed as `hard`, routed to `smart_node`.
>
> We're simulating with the same model here, but in production, using different models
> can **significantly reduce costs**. Cheap model for easy, premium model for hard.

### Exercise 16.3 (3 min)

> **Exercise 1**: Try questions of various difficulty levels and verify routing is correct.
>
> **Exercise 2**: Check the `model_used` field to verify which path was taken.
>
> **Exercise 3**: Design a scenario that routes to different prompt templates instead of different models.

---

## 16.4 Parallelization — Parallel Execution (10 min)

### Concept

> The fourth pattern. **Run independent LLM calls simultaneously.**
>
> ```
>         → get_summary ────────┐
>         → get_sentiment ──────┤
> START → → get_key_points ─────┤→ get_final_analysis → END
>         → get_recommendation ─┘
> ```
>
> This is also called the **Fan-out / Fan-in** pattern:
> - Fan-out: 4 nodes launch simultaneously from START
> - Fan-in: all 4 must complete before merging into one
>
> Implementing this in LangGraph is surprisingly simple.
> Connect edges from `START` to multiple nodes — they run **automatically in parallel**.
> Connect multiple nodes to one — it **automatically joins** (waits for all to complete).
>
> How this differs from Send API in 13.8: here we run **different functions** simultaneously.
> Send API runs the **same function with different inputs** simultaneously.

### Code walkthrough (before running)

> The State has 6 fields:
> ```python
> class State(TypedDict):
>     document: str          # input document
>     summary: str           # parallel node 1
>     sentiment: str         # parallel node 2
>     key_points: str        # parallel node 3
>     recommendation: str    # parallel node 4
>     final_analysis: str    # aggregation result
> ```
>
> All 4 parallel nodes read the same `document` but **write to different fields**.
> No overlap, so no conflicts when running simultaneously.
>
> The key part of graph setup:
> ```python
> # Fan-out: START to 4 nodes
> graph_builder.add_edge(START, "get_summary")
> graph_builder.add_edge(START, "get_sentiment")
> graph_builder.add_edge(START, "get_key_points")
> graph_builder.add_edge(START, "get_recommendation")
>
> # Fan-in: 4 nodes to 1
> graph_builder.add_edge("get_summary", "get_final_analysis")
> graph_builder.add_edge("get_sentiment", "get_final_analysis")
> graph_builder.add_edge("get_key_points", "get_final_analysis")
> graph_builder.add_edge("get_recommendation", "get_final_analysis")
> ```
>
> That's all it takes for parallel + join. Let's run it.

### After running

> Look at the output:
> ```
> [parallel] get_key_points started
> [parallel] get_recommendation started
> [parallel] get_sentiment started
> [parallel] get_summary started
> ```
>
> All 4 started **almost simultaneously**. The random order is proof of parallel execution.
>
> Then:
> ```
> [join] get_final_analysis started
> ```
>
> `get_final_analysis` only ran after all 4 completed.
>
> The Final Analysis combines summary, sentiment, key_points, and recommendation
> into a comprehensive analysis.
>
> **Performance perspective**: sequential execution means LLM call times are summed.
> Parallel execution takes only **the time of the slowest single call**. That can be 3-4x faster.

### Exercise 16.4 (3 min)

> **Exercise 1**: Add a 5th parallel node (e.g., `get_risks`) and verify it's included in the aggregation.
>
> **Exercise 2**: Run with `graph.stream()` to observe the parallel starts directly.
>
> **Exercise 3**: Compare total execution time between sequential and parallel execution.

---

## 16.5 Orchestrator-Workers — Dynamic Task Distribution (10 min)

### Concept

> The final pattern. The most powerful, combining all previous patterns.
>
> ```
> START → orchestrator → [Send x N] → worker x N → synthesizer → END
> ```
>
> The core idea:
> 1. **Orchestrator**: analyzes the topic and generates a list of sections (unknown count ahead of time)
> 2. **Send API**: dynamically launches N workers in parallel, one per section
> 3. **Synthesizer**: combines all worker results into a final report
>
> How this differs from 16.4 Parallelization:
> - 16.4: **fixed number of nodes** (4 defined in code)
> - 16.5: **dynamic number of nodes** (LLM decides, Send API executes)
>
> This is the **real-world combination** of 13.8 Send API + 16.4 Parallelization.

### Code walkthrough (before running)

> Let's look at the State:
> ```python
> class State(TypedDict):
>     topic: str
>     sections: list[str]
>     results: Annotated[list[dict], operator.add]  # reducer!
>     final_report: str
> ```
>
> `results` has an `operator.add` reducer.
> When workers return results in parallel, they're automatically accumulated.
>
> **Orchestrator**:
> ```python
> def orchestrator(state: State):
>     structured_llm = llm.with_structured_output(Sections)
>     response = structured_llm.invoke(
>         f"Break down this topic into 3-5 research sections: {state['topic']}"
>     )
>     return {"sections": response.sections}
> ```
>
> The LLM breaks the topic into 3-5 sections. Structured with `with_structured_output()`.
>
> **Dispatcher** (Send API):
> ```python
> def dispatch_workers(state: State):
>     return [Send("worker", section) for section in state["sections"]]
> ```
>
> Returns as many `Send` objects as there are sections. 5 sections = 5 workers in parallel.
>
> **Worker**:
> ```python
> def worker(section: str):  # receives str, not State!
>     response = llm.invoke(f"Write a brief paragraph about: {section}")
>     return {"results": [{"section": section, "content": response.content}]}
> ```
>
> As we learned in 13.8, values passed via Send API are **custom inputs, not State**.
>
> **Synthesizer**: combines all worker results into the final report.
>
> Let's run it.

### After running

> ```
> Orchestrator: 5 sections
>   - Introduction to AI in Healthcare
>   - Applications of AI in Clinical Settings
>   - Ethical Considerations and Challenges
>   - Impact on Patient Outcomes
>   - Future Trends and Predictions
> ```
>
> The Orchestrator split the topic into 5 sections.
>
> Then 5 workers ran in parallel:
> ```
> Worker done: Introduction to AI in Healthcare...
> Worker done: Future Trends and Predictions...
> Worker done: Impact on Patient Outcomes...
> ```
>
> Notice the random order — that's parallel execution.
>
> Finally, the Final Report is a complete report combining all 5 sections.
>
> **This is the power of the Orchestrator-Workers pattern.**
> Just change the topic and a completely different report structure is generated automatically.
> The LLM decides both the number and content of sections.

### Exercise 16.5 (3 min)

> **Exercise 1**: Change `topic` to see if the LLM produces different section breakdowns.
>
> **Exercise 2**: Add `time.sleep(1)` to the worker to feel the effect of parallel execution.
>
> **Exercise 3**: Modify the worker to return structured output using `BaseModel`.

---

## Final Exercises Guide (3 min)

> There are 4 comprehensive exercises at the end of the notebook.
>
> **Exercise 1** (★★☆): Translation chain + gate — Korean→English→Japanese sequential translation, retry if under 50 chars
> **Exercise 2** (★★☆): Sentiment-based routing — analyze user message sentiment, route to positive/negative/neutral paths
> **Exercise 3** (★★★): Code review parallel analysis — 4 perspectives (security/performance/readability/testing) simultaneously, then synthesize
> **Exercise 4** (★★★): Orchestrator-Workers blog — topic→outline generation→section workers→synthesis
>
> Exercises 1-2 are basics, 3-4 are challenges.
> Time allocation: 10 min for easy ones, 15 min for harder ones.

---

## Wrap-up (2 min)

> Let's summarize the 5 patterns we learned today.
>
> | Pattern | Structure | Key Point |
> |---------|-----------|-----------|
> | Prompt Chaining | A → B → C | Sequential connection, structured output for stability |
> | Chaining + Gate | A → [check] → B / ↩ A | Quality check with retry, watch for infinite loops |
> | Routing | A → branch → B or C or D | Command + Literal for safe routing |
> | Parallelization | Fan-out → Fan-in | Just connect edges for auto-parallel, 3-4x performance gain |
> | Orchestrator-Workers | O → Send x N → S | Dynamic parallel, Send API + Reducer for result accumulation |
>
> These 5 are the **most commonly used patterns in production**.
> Most AI workflows can be built as combinations of these patterns.
>
> In the next chapter, we'll learn more advanced patterns.
> Great work today.
