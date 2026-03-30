# Chapter 15: LangGraph Project Pipeline — Lecture Script

---

## Opening (2 min)

> Alright, in Chapter 14 we built a chatbot and learned the core LangGraph patterns.
>
> Today we **combine all of them** into a complete, end-to-end pipeline.
>
> The project is a **Blog Post Generator** — an automated blog writing pipeline.
>
> ```
> 15.0 Setup
> 15.1 Basic Pipeline          (Linear 2-node graph)
> 15.2 Parallel Writing Nodes  (Send API + Map-Reduce)
> 15.3 Human-in-the-loop       (interrupt + Command resume)
> 15.4 Complete Pipeline        (All patterns combined)
> 15.5 Production Deployment    (langgraph.json, graph.py)
> ```
>
> Just like Chapter 14, each section builds on the previous one.
> By the end, we'll have a pipeline that automatically researches a topic, writes sections in parallel, gets human review, and produces a final post.
>
> Let's go.

---

## 15.0 Setup & Environment (2 min)

### Before running the cell

> First, let's check our environment.
> We load the API key from `.env` and check `langgraph` and `langchain` versions.
>
> Same environment as Chapter 14. No additional packages needed.

### After running the cell

> If the API key and versions print correctly, we're good to go.
> If you get an error, check `uv sync` or `pip install`.

---

## 15.1 Basic Pipeline — 2-Node Linear Graph (8 min)

### Concept

> We start with the simplest possible pipeline.
>
> The structure:
> ```
> START → get_topic_info → write_draft → END
> ```
>
> **How is this different from the chatbot?**
> In Chapter 14, the chatbot used `MessagesState` — messages accumulated in a conversation.
> A pipeline uses **`TypedDict` to design its own state**.
> Because a pipeline isn't a conversation — it's about **data transforming step by step**.
>
> Look at the state:
> ```python
> class PipelineState(TypedDict):
>     topic: str             # Input: blog topic
>     background_info: str   # Step 1 output: background research
>     draft: str             # Step 2 output: draft
> ```
>
> Each node fills one field in the state.
> `get_topic_info` fills `background_info`, `write_draft` fills `draft`.
> Data flows through like a pipe.

### Code walkthrough (before running)

> Let's look at the code.
>
> ```python
> def get_topic_info(state: PipelineState):
>     topic = state["topic"]
>     response = llm.invoke(f"Provide a concise background summary about: {topic}...")
>     return {"background_info": response.content}
> ```
>
> The node function pattern is the same as Chapters 13 and 14:
> 1. Extract needed data from state
> 2. Call the LLM
> 3. Return the result to a state field
>
> `write_draft` follows the same pattern. It receives `background_info` and produces `draft`.
>
> Graph assembly is familiar too:
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_draft", write_draft)
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_edge("get_topic_info", "write_draft")
> graph_builder.add_edge("write_draft", END)
> ```
>
> Linear edges connecting nodes in order. The most basic pipeline.
>
> Let's run it.

### After running the cell

> Look at the results.
> `background_info` contains the research about the topic,
> and `draft` contains a blog post written based on that research.
>
> Just 2 nodes give us a "research → write" pipeline.
>
> But there's a problem — the blog post is **one big chunk**.
> In reality, we want to write multiple sections separately.
> That's why we add **parallel writing** next.

---

## 15.2 Parallel Writing Nodes — Send API + Map-Reduce (15 min)

### Concept

> This section is the **highlight** of Chapter 15.
>
> We split the blog post into N sections, **write each one simultaneously**, and merge them.
>
> ```
> get_topic_info → dispatch_writers ──→ write_section (x3) → combine_sections
>                                  ├─→ write_section
>                                  └─→ write_section
> ```
>
> Three new concepts appear here:
>
> 1. **`Send` API** — dynamically dispatch parallel nodes. We learned this in Chapter 13.
> 2. **`Annotated[list[str], operator.add]`** — a reducer that automatically merges parallel results
> 3. **`with_structured_output()`** — LLM generates structured output as a Pydantic model
>
> Let's go through each one.

### with_structured_output explained

> First, `with_structured_output`.
>
> Until now, LLM responses were always strings — we got text via `response.content`.
>
> But we want the blog post's **section titles and key points** as structured data.
>
> So we define the output shape with Pydantic models:
>
> ```python
> class SectionPlan(BaseModel):
>     title: str = Field(description="Section title")
>     key_points: list[str] = Field(description="Key points to cover")
>
> class BlogOutline(BaseModel):
>     sections: list[SectionPlan] = Field(description="List of sections")
> ```
>
> Then we tell the LLM to output in this format:
>
> ```python
> planner = llm.with_structured_output(BlogOutline)
> outline = planner.invoke("Create an outline with 3 sections")
> ```
>
> Now `outline.sections` is a list of `SectionPlan` objects.
> No string parsing — direct access to `.title` and `.key_points`!
>
> This is what makes LLMs a **programmable tool**.

### dispatch_writers explained — This is NOT a node!

> Now, this is really important.
>
> **`dispatch_writers` is NOT a node!**
>
> Look at the code:
>
> ```python
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> ```
>
> It goes into `add_conditional_edges` as the second argument.
> This is the same position as `tools_condition` in Chapter 14.
> It's an **edge routing function**.
>
> It's not registered with `add_node`. It doesn't appear as a node in graph visualization.
>
> **What does it do?**
>
> After `get_topic_info` finishes, LangGraph asks "where to go next?" and
> calls `dispatch_writers` to decide.
>
> This function returns a list of `Send` objects:
>
> ```python
> def dispatch_writers(state: PipelineState):
>     planner = llm.with_structured_output(BlogOutline)
>     outline = planner.invoke("Create an outline")
>     return [
>         Send("write_section", {
>             "topic": state["topic"],
>             "section_title": section.title,
>             "section_key_points": section.key_points,
>         })
>         for section in outline.sections
>     ]
> ```
>
> Each `Send` object creates an **independent execution instance** of `write_section`.
> If there are 3 sections, 3 `write_section` nodes run **simultaneously**.
>
> Think of it this way:
> - `dispatch_writers` is a **mail sorting facility**
> - It sorts packages (section data) and sends each to a different delivery driver (`write_section`)
> - The sorting facility itself doesn't deliver anything (it's not a node)

### Map-Reduce pattern explained

> How are the parallel results merged?
>
> Look at the state definition:
>
> ```python
> class PipelineState(TypedDict):
>     sections: Annotated[list[str], operator.add]  # Reducer!
> ```
>
> `operator.add` is the reducer.
> Each `write_section` returns `{"sections": ["section content"]}`,
> and LangGraph automatically **concatenates** the lists.
>
> 3 nodes each return 1 item → `sections` ends up with 3 items.
>
> This is the **Map-Reduce** pattern:
> - **Map** = dispatch with `Send`, process each in parallel
> - **Reduce** = merge results with the `operator.add` reducer
>
> The `combine_sections` node receives `state["sections"]` (all 3 collected) and
> joins them into a single `combined_draft`.

### Running the code

> Let's run it.

### After running the cell

> Look at the results:
> - "3 sections written in parallel!" — 3 sections written simultaneously
> - Each section starts with `## Title` structure
> - `combined_draft` contains the merged full draft
>
> This wasn't one node writing 3 times sequentially.
> `Send` dispatched 3 instances that ran **in parallel**.
>
> In production, even with 10 or 20 sections, the same pattern scales.

---

## 15.3 Human-in-the-loop — interrupt + Command resume (10 min)

### Concept

> In 15.2, the draft was created automatically.
> But we shouldn't publish it directly — **a human needs to review it**.
>
> We apply the `interrupt` and `Command` from Chapter 14.3 here.
>
> ```
> write_draft → human_feedback(interrupt!) → [human feedback] → finalize_post → END
> ```
>
> Three key points:
> - `interrupt(value)` — pauses the graph and shows the draft to the user
> - `Command(resume=feedback)` — resumes with user feedback
> - `SqliteSaver` — checkpointer to preserve state between pauses (required!)

### Code walkthrough (before running)

> Let's look at the state:
>
> ```python
> class ReviewState(TypedDict):
>     topic: str
>     draft: str
>     feedback: str
>     final_post: str
> ```
>
> `feedback` and `final_post` fields are added.
>
> The `human_feedback` node is the key:
>
> ```python
> def human_feedback(state: ReviewState):
>     feedback = interrupt(
>         f"DRAFT FOR REVIEW\n{state['draft'][:500]}...\n"
>         f"Please provide your feedback:"
>     )
>     return {"feedback": feedback}
> ```
>
> When `interrupt()` is called:
> 1. The graph **pauses** — state is automatically saved to SQLite
> 2. The argument to `interrupt()` is returned to the user (draft preview)
> 3. Waits until the user provides feedback
>
> To resume:
> ```python
> result = graph.invoke(
>     Command(resume="Make it more concise. Add code examples."),
>     config=config,
> )
> ```
>
> The value in `Command(resume=...)` becomes the **return value** of `interrupt()`.
> So `feedback = "Make it more concise..."` is stored in state.
>
> The `finalize_post` node sends the draft + feedback to the LLM to generate the final version.
>
> A checkpointer is **mandatory**:
> ```python
> conn = sqlite3.connect("pipeline_review.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```
>
> Using `interrupt` without a checkpointer will throw an error.
> The state has to be stored somewhere while the graph is paused.

### Running the cell — Step 1: Write draft

> Run the first cell.

### After running (Step 1)

> `snapshot.next` is `('human_feedback',)`.
> The graph is paused at the `human_feedback` node because of `interrupt`.
>
> The draft preview is displayed, waiting for feedback.

### Running the cell — Step 2: Provide feedback

> Resume with `Command(resume="Make it more concise. Add a code example.")`.

### After running (Step 2)

> `Status: COMPLETE` — pipeline finished.
> `snapshot.next` is an empty tuple `()`.
>
> `final_post` contains the revised final version with feedback applied.
> It should be more concise with code examples added.
>
> This is the basic structure of AI + human collaboration.
> AI creates the first draft, human provides direction, AI refines it.

---

## 15.4 Complete Pipeline — All Patterns Combined (10 min)

### Concept

> Now we **merge everything** from 15.1 through 15.3 into one pipeline.
>
> ```
> [START]
>    |
> [get_topic_info] ───── LLM background research
>    |
> [dispatch_writers] ──── with_structured_output + Send API (router!)
>    |
> [write_section] x N ─── Parallel writing (Map)
>    |
> [combine_sections] ──── Merge draft (Reduce)
>    |
> [human_feedback] ───── interrupt() review
>    |
> [finalize_post] ────── Apply feedback, final version
>    |
> [END]
> ```
>
> The code looks long, but every part is something we've already learned.
> Nothing new. We're just **combining** them.

### State explained

> The state tells the whole story:
>
> ```python
> class FullPipelineState(TypedDict):
>     topic: str                                      # Input
>     background_info: str                             # From 15.1
>     sections: Annotated[list[str], operator.add]     # From 15.2 (Map-Reduce)
>     combined_draft: str                              # From 15.2
>     feedback: str                                    # From 15.3 (HITL)
>     final_post: str                                  # From 15.3
> ```
>
> Six fields showing the entire data flow of the pipeline.
> Each node fills them one by one.

### Graph assembly explained

> Let's look at the graph assembly code:
>
> ```python
> graph_builder.add_node("get_topic_info", get_topic_info)
> graph_builder.add_node("write_section", write_section)
> graph_builder.add_node("combine_sections", combine_sections)
> graph_builder.add_node("human_feedback", human_feedback)
> graph_builder.add_node("finalize_post", finalize_post)
> ```
>
> **5 nodes** are registered.
> `dispatch_writers` is **not here**! Because it's not a node.
>
> ```python
> graph_builder.add_edge(START, "get_topic_info")
> graph_builder.add_conditional_edges("get_topic_info", dispatch_writers, ["write_section"])
> graph_builder.add_edge("write_section", "combine_sections")
> graph_builder.add_edge("combine_sections", "human_feedback")
> graph_builder.add_edge("human_feedback", "finalize_post")
> graph_builder.add_edge("finalize_post", END)
> ```
>
> `dispatch_writers` is registered as a **routing function** in `add_conditional_edges`.
> After `get_topic_info` runs, this router returns `Send` objects to dispatch parallel nodes.
>
> The checkpointer is also required:
> ```python
> conn = sqlite3.connect("pipeline_full.db", check_same_thread=False)
> graph = graph_builder.compile(checkpointer=SqliteSaver(conn))
> ```

### Running the cell — Step 1: Execute pipeline

> Let's run it. Topic is "Building AI Agents with LangGraph".

### After running (Step 1)

> Look at the output:
> - `Dispatching 3 parallel writers...` — 3 sections dispatched in parallel
> - `Paused at: ('human_feedback',)` — paused at the review step
> - `Sections written: 3` — 3 sections completed
> - Draft preview is displayed
>
> Up to this point, automatically: topic research → outline generation → 3 parallel section writes → draft merge.
> Now waiting for human review.

### Running the cell — Step 2: Provide feedback

> Let's give feedback:
> ```python
> Command(resume="Add a practical code example in each section. Make the tone more engaging.")
> ```

### After running (Step 2)

> `Status: COMPLETE` — done!
>
> The final post has the feedback applied.
> Code examples should be added and the tone should be different.
>
> This is a **complete AI writing pipeline**:
> 1. AI researches the topic
> 2. Plans the structure
> 3. Writes sections in parallel
> 4. Human reviews
> 5. AI applies feedback and generates the final version
>
> 6 units (precisely: 5 nodes + 1 router), combining 3 LangGraph patterns.

---

## 15.5 Production Deployment Structure (5 min)

### Concept

> Prototyping in a notebook is great, but how do we deploy to production?
>
> LangGraph provides a standard structure for production deployment.
>
> ```
> my_pipeline/
> ├── langgraph.json       # Entry point config
> ├── graph.py             # Graph definition
> ├── state.py             # State schemas
> ├── nodes.py             # Node functions
> ├── prompts.py           # LLM prompt templates
> └── requirements.txt     # Dependencies
> ```
>
> We **split the notebook code into separate files**.
>
> The key is `langgraph.json`:
> ```json
> {
>     "dependencies": ["."],
>     "graphs": {
>         "blog_pipeline": "./graph.py:graph"
>     },
>     "env": ".env"
> }
> ```
>
> `"blog_pipeline": "./graph.py:graph"` — declares that the `graph` variable in `graph.py` is the entry point.
>
> `graph.py` contains the same code we wrote in the notebook.
> `StateGraph` creation, node registration, edge connections, `compile()`.
>
> The only difference is import paths.

### Deployment options

> There are 3 deployment methods:
>
> | Method | Description |
> |--------|------------|
> | `langgraph dev` | Local development + Studio UI (http://localhost:8123) |
> | `langgraph up` | Run as Docker container |
> | LangGraph Cloud | Managed deployment with LangSmith integration |
>
> Running `langgraph dev` launches a Studio UI where you can **visually** test the graph.
> Click on nodes to inspect inputs/outputs, and provide feedback directly at interrupt points.
>
> For production, use `langgraph up` for Docker, or
> deploy to LangGraph Cloud to expose it as an API endpoint.

---

## Wrap-up (3 min)

> Let's summarize what we built today.
>
> **Blog Post Generator Pipeline:**
> 1. `get_topic_info` — LLM researches the topic
> 2. `dispatch_writers` — router function, creates outline with `with_structured_output`, dispatches with `Send`
> 3. `write_section` x N — parallel section writing (Map)
> 4. `combine_sections` — merge results (Reduce)
> 5. `human_feedback` — human review with `interrupt()`
> 6. `finalize_post` — apply feedback, final version
>
> **LangGraph patterns used:**
>
> | Pattern | Where used |
> |---------|-----------|
> | `TypedDict` state | Pipeline data flow design |
> | `Send` API | Parallel section dispatch (Map) |
> | `operator.add` reducer | Auto-merge parallel results (Reduce) |
> | `with_structured_output` | Structured LLM output (outline) |
> | `conditional_edges` | Router function for dynamic branching |
> | `interrupt` + `Command` | Human review workflow |
> | `SqliteSaver` | State preservation between interrupts |
>
> In Chapter 13 we learned the basics, in Chapter 14 we applied them to a chatbot,
> and today in Chapter 15 we built a **production-grade pipeline combining everything**.
>
> One key takeaway:
> **`dispatch_writers` is NOT a node — it's a router function.**
> It's registered with `add_conditional_edges` and returns `Send` objects to dispatch parallel nodes.
>
> See you in the next chapter.
