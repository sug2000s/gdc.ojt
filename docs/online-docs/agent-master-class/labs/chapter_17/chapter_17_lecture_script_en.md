# Chapter 17: LangGraph Workflow Testing — Lecture Script

---

## Opening (2 min)

> Alright, so far we have been building workflows with LangGraph.
>
> But how do we verify that our graphs **actually work correctly**?
>
> Today we will learn how to **systematically test LangGraph workflows with pytest**.
>
> Here is the roadmap:
>
> ```
> 17.0 Setup
> 17.1 Email Processing Graph    (Rule-based, deterministic)
> 17.2 Pytest Basics              (%%writefile, parametrize)
> 17.3 Node Unit Testing          (graph.nodes, update_state)
> 17.4 Switching to AI Nodes      (with_structured_output)
> 17.5 AI Testing Strategies      (Range-based assertions)
> 17.6 LLM-as-a-Judge             (Golden examples, similarity scoring)
> ```
>
> The key evolution to keep in mind:
> **Exact match --> Range-based --> LLM-as-Judge**
>
> Rule-based code can be tested with `assert ==`,
> but AI outputs change every time, so we need new strategies.
>
> Let us experience this evolution step by step. Let's begin.

---

## 17.0 Setup & Environment (2 min)

### Before running the cell

> First, let us check our environment.
> We load API keys from `.env` and verify the versions of `langgraph`, `langchain`, and `pytest`.
>
> Same environment as previous chapters, plus `pytest`.

### After running the cell

> If the API key and versions print correctly, we are good to go.
> If `pytest` is missing, install it with `pip install pytest`.

---

## 17.1 Email Processing Graph — Rule-based Email Classifier (8 min)

### Concept

> We start by building a **graph that is easy to test**.
>
> Why rule-based first?
> Because rule-based logic is **deterministic**.
> Same input always produces the same output. Testing is straightforward.
>
> The graph structure is linear:
>
> ```
> START --> categorize_email --> assign_priority --> generate_response --> END
> ```
>
> Three nodes:
> - `categorize_email` — classifies by keyword (urgent / spam / normal)
> - `assign_priority` — maps category to priority (high / low / medium)
> - `generate_response` — returns a template response per category
>
> All use `if-elif-else`. No LLM involved.

### Code walkthrough (before running)

> Let us look at the code.
>
> State definition:
>
> ```python
> class EmailState(TypedDict):
>     email: str
>     category: str
>     priority: str
>     response: str
> ```
>
> As the email flows through the graph, `category`, `priority`, and `response` get filled in.
>
> The `categorize_email` function:
>
> ```python
> def categorize_email(state: EmailState) -> dict:
>     email = state["email"].lower()
>     if "urgent" in email or "asap" in email:
>         return {"category": "urgent"}
>     elif "offer" in email or "discount" in email:
>         return {"category": "spam"}
>     return {"category": "normal"}
> ```
>
> Converts to lowercase, then checks for keywords.
> "urgent" or "asap" means urgent, "offer" or "discount" means spam, everything else is normal.
>
> `assign_priority` maps category to priority:
> urgent --> high, spam --> low, otherwise --> medium.
>
> `generate_response` returns a template response based on category.
>
> Graph edges are linear:
>
> ```python
> builder.add_edge(START, "categorize_email")
> builder.add_edge("categorize_email", "assign_priority")
> builder.add_edge("assign_priority", "generate_response")
> builder.add_edge("generate_response", END)
> ```
>
> A straight pipeline. No branching.

### After running

> ```
> "URGENT: Server is down, fix ASAP!" --> category: urgent, priority: high
> "Special offer! 50% discount today!" --> category: spam, priority: low
> "Hi, I have a question about my order." --> category: normal, priority: medium
> ```
>
> Works as expected.
> Now let us replace manual checking with **automated tests**.

---

## 17.2 Pytest Basics — parametrize + %%writefile (8 min)

### Concept

> There is a pattern for running pytest inside notebooks. Three steps:
>
> 1. `%%writefile main.py` — save the code under test to a `.py` file
> 2. `%%writefile tests.py` — save the test code to a `.py` file
> 3. `!pytest tests.py -v` — run the tests
>
> **What is `%%writefile`?**
> It is a Jupyter magic command. It saves the cell contents directly to a file.
> Just put `%%writefile filename` on the first line of the cell.
>
> Why do we need this?
> pytest only runs `.py` files. It cannot read notebook cells directly.
> So we use `%%writefile` to export the code.
>
> And `@pytest.mark.parametrize` — a decorator that lets you run one test function
> with multiple sets of inputs and expected values.
> One function, six cases, ten cases — all at once.

### Code walkthrough (before running)

> First, `%%writefile main.py` saves the email graph code to a file.
> Same code as section 17.1, no changes.
>
> Next, `%%writefile tests.py`:
>
> ```python
> @pytest.mark.parametrize(
>     "email, expected_category, expected_priority",
>     [
>         ("URGENT: Server down!", "urgent", "high"),
>         ("Fix this ASAP please", "urgent", "high"),
>         ("Special offer! Buy now!", "spam", "low"),
>         ("Get 50% discount today", "spam", "low"),
>         ("Hi, I have a question", "normal", "medium"),
>         ("Meeting tomorrow at 3pm", "normal", "medium"),
>     ],
> )
> def test_email_pipeline(email, expected_category, expected_priority):
>     graph = build_email_graph()
>     result = graph.invoke({"email": email})
>     assert result["category"] == expected_category
>     assert result["priority"] == expected_priority
>     assert len(result["response"]) > 0
> ```
>
> `@pytest.mark.parametrize`:
> - First argument: parameter names (comma-separated string)
> - Second argument: list of test cases (list of tuples)
> - 6 tuples means the test runs 6 times
>
> `assert` fails the test when the condition is False.
> `result["category"] == expected_category` — must be an exact match.
>
> Below that are individual node tests:
>
> ```python
> def test_categorize_urgent():
>     result = categorize_email({"email": "This is URGENT!", ...})
>     assert result["category"] == "urgent"
> ```
>
> Calling node functions directly. This is the **foundation of unit testing**.

### After running

> `!pytest tests.py -v` produces:
>
> ```
> tests.py::test_email_pipeline[URGENT: Server down!-urgent-high]   PASSED
> tests.py::test_email_pipeline[Fix this ASAP please-urgent-high]   PASSED
> ...
> tests.py::test_priority_mapping                                    PASSED
> tests.py::test_response_templates                                  PASSED
> ```
>
> All PASSED. Green across the board.
>
> `-v` means verbose. Each test case gets its own line.
> Thanks to parametrize, the input values appear in the test name.
>
> This is the beauty of rule-based testing: **`assert ==` for exact verification.**

---

## 17.3 Node Unit Testing — graph.nodes + update_state (8 min)

### Concept

> In 17.2, we tested the entire graph end-to-end.
> But in practice, you often need to test **specific nodes in isolation**.
>
> Two methods:
>
> **Method 1: `graph.nodes["name"].invoke(state)`**
> - Directly invoke a specific node from the compiled graph
> - Equivalent to calling the node function directly
>
> **Method 2: `graph.update_state(config, values, as_node="name")`**
> - Inject state as if a specific node produced it
> - Then run only the **remaining nodes** (partial execution)
> - Requires a `MemorySaver` checkpointer

### Code walkthrough — Method 1 (before running)

> ```python
> test_state = {"email": "URGENT: Need help ASAP", "category": "", "priority": "", "response": ""}
>
> cat_result = graph.nodes["categorize_email"].invoke(test_state)
> print(f"categorize_email result: {cat_result}")
> ```
>
> `graph.nodes` is a dictionary. Keys are node names, values are node objects.
> `.invoke(state)` runs just that one node.
>
> Important: **this does not run the entire graph.**
> Only `categorize_email` executes and returns its result.
>
> This lets you verify "does my node function return the correct value?" independently.

### After running — Method 1

> ```
> categorize_email result: {'category': 'urgent'}
> assign_priority result: {'priority': 'high'}
> generate_response result: {'response': 'This email has been classified as spam.'}
> ```
>
> Each node works independently.

### Code walkthrough — Method 2: update_state (before running)

> First, create a graph with a checkpointer:
>
> ```python
> from langgraph.checkpoint.memory import MemorySaver
> graph_mem = builder.compile(checkpointer=MemorySaver())
> ```
>
> `MemorySaver` is an in-memory checkpointer. Lighter than SQLite, great for testing.
>
> Then:
>
> ```python
> config2 = {"configurable": {"thread_id": "test_partial_2"}}
>
> # Run full graph first
> graph_mem.invoke({"email": "Hello, normal email"}, config=config2)
>
> # Force-inject state as if categorize_email returned "urgent"
> graph_mem.update_state(
>     config2,
>     {"category": "urgent"},
>     as_node="categorize_email",
> )
> ```
>
> `as_node="categorize_email"` — this is the key!
> It means "treat this value as if categorize_email produced it."
>
> We forcibly changed category from "normal" to "urgent".
>
> ```python
> result_partial = graph_mem.invoke(None, config=config2)
> ```
>
> `invoke(None)` — no new input, just **run the remaining nodes**.
> `assign_priority --> generate_response` execute.
> Since category is "urgent", priority becomes "high".

### After running — Method 2

> ```
> Injected state: category=urgent
> Partial execution result: category=urgent, priority=high
> Assert passed! Partial execution with update_state successful
> ```
>
> Why is this useful?
>
> When you suspect a bug in the 3rd node,
> you do not need to run the first two.
> **Inject the state you want, test only the 3rd node.**
>
> Saves time and makes debugging faster.

---

## 17.4 Switching to AI Nodes — LLM + with_structured_output (8 min)

### Concept

> Here is where the paradigm shifts.
>
> Until now, everything was rule-based and deterministic.
> Now we **replace the node functions with AI, keeping the same graph structure**.
>
> Graph structure stays the same:
> ```
> START --> categorize_email --> assign_priority --> generate_response --> END
> ```
>
> Only the logic inside each node changes.
>
> Key technique: **`with_structured_output`**
>
> LLMs naturally return free-form text.
> But our graph needs exactly `"urgent"`, `"spam"`, or `"normal"`.
>
> `with_structured_output(PydanticModel)` forces the LLM output into a Pydantic model.
> Whatever the LLM says, it gets parsed into the defined fields and types.

### Code walkthrough (before running)

> Pydantic output schemas:
>
> ```python
> class CategoryOutput(BaseModel):
>     category: Literal["urgent", "spam", "normal"] = Field(
>         description="The email category"
>     )
> ```
>
> `Literal["urgent", "spam", "normal"]` — only these 3 values allowed!
> Even if the LLM wants to say "critical", it must pick one of these three.
>
> `Field(description=...)` explains to the LLM what this field represents.
>
> Same pattern for `PriorityOutput` and `ResponseOutput`.
>
> Creating structured LLMs:
>
> ```python
> category_llm = llm.with_structured_output(CategoryOutput)
> priority_llm = llm.with_structured_output(PriorityOutput)
> response_llm = llm.with_structured_output(ResponseOutput)
> ```
>
> Now `category_llm.invoke("...")` returns a `CategoryOutput` object,
> not a string — an object with a `.category` attribute.
>
> AI node function:
>
> ```python
> def ai_categorize_email(state: EmailState) -> dict:
>     result = category_llm.invoke(
>         f"Classify this email into one of: urgent, spam, normal.\n\nEmail: {state['email']}"
>     )
>     return {"category": result.category}
> ```
>
> Comparing rule-based vs AI:
> - Rule-based: `if "urgent" in email` — keyword matching
> - AI-based: LLM understands context and makes a judgment
>
> "The server is down" has no "urgent" keyword, but AI can still classify it as urgent.

### After running

> ```
> "URGENT: Production database is corrupted" --> Category: urgent, Priority: high
> "Congratulations! You won $1,000,000!" --> Category: spam, Priority: low
> ```
>
> AI classifies correctly.
>
> But here is the problem! **Run the same email again and you might get different results.**
> AI output is **non-deterministic**.
>
> This means `assert ==` will not work reliably.
> We need new testing strategies.

---

## 17.5 AI Testing Strategies — Range-based Assertions (8 min)

### Concept

> AI output changes every time.
> So how do we test it?
>
> **We validate with ranges!**
>
> Four strategies:
>
> 1. **Valid value range**: `assert result in ["urgent", "spam", "normal"]`
>    - Whatever the value, as long as it is one of these three, OK
>
> 2. **Length range**: `assert 20 <= len(response) <= 1000`
>    - Not too short, not too long, OK
>
> 3. **Minimum quality threshold**: `assert score >= threshold`
>    - Score above the bar, OK
>
> 4. **Consistency**: Run same input N times, check majority agreement
>    - At least 2 out of 3 match, OK
>
> Since we cannot know the exact value, we verify
> "it should at least be within this range."

### Code walkthrough (before running)

> `%%writefile tests_ai.py` creates the test file.
>
> ```python
> VALID_CATEGORIES = {"urgent", "spam", "normal"}
> VALID_PRIORITIES = {"high", "medium", "low"}
> ```
>
> Valid value sets defined upfront.
>
> ```python
> @pytest.fixture
> def ai_graph():
>     return build_ai_email_graph()
> ```
>
> `@pytest.fixture` — a function that creates reusable objects for tests.
> Multiple test functions that take `ai_graph` as a parameter get it auto-injected.
>
> ```python
> def test_output_in_valid_range(ai_graph, email):
>     result = ai_graph.invoke({"email": email})
>     assert result["category"] in VALID_CATEGORIES
>     assert result["priority"] in VALID_PRIORITIES
>     assert len(result["response"]) > 10
>     assert len(result["response"]) < 2000
> ```
>
> Not `assert ==` but `assert in`!
> Not "is it urgent?" but "is it one of urgent, spam, or normal?"
>
> The consistency test:
>
> ```python
> def test_consistency_over_runs(ai_graph):
>     email = "URGENT: Production is completely down!"
>     categories = []
>     for _ in range(3):
>         result = ai_graph.invoke({"email": email})
>         categories.append(result["category"])
>
>     from collections import Counter
>     most_common_count = Counter(categories).most_common(1)[0][1]
>     assert most_common_count >= 2
> ```
>
> Run 3 times, the most frequent result must appear at least twice.
> That is 66% consistency. "2 out of 3 must agree."

### After running

> ```
> tests_ai.py::test_output_in_valid_range[URGENT: Server...]     PASSED
> tests_ai.py::test_clear_cases_match_expected[CRITICAL...]       PASSED
> tests_ai.py::test_response_length_range                          PASSED
> tests_ai.py::test_consistency_over_runs                          PASSED
> ```
>
> All PASSED!
>
> Key takeaway — comparing rule-based vs AI testing:
>
> | Rule-based | AI-based |
> |------------|----------|
> | `assert ==` exact match | `assert in` range check |
> | One run is enough | N runs for consistency |
> | Always same result | May vary each time |
>
> Completely different testing strategies.

---

## 17.6 LLM-as-a-Judge — Golden Examples + Similarity Scoring (10 min)

### Concept

> Alright, the final and most powerful testing method.
>
> In 17.5, range validation checked "is the value valid?"
> But how do we check "is the response **quality** good?"
>
> Having humans read every response? Inefficient.
> So we let an **LLM act as the Judge**.
>
> The pattern has three parts:
>
> 1. **Golden Examples** — ideal responses prepared per category
> 2. **Judge LLM** — compares generated response to golden example, assigns similarity score
> 3. **Threshold** — score of 70 or above passes
>
> Analogy:
> - Golden Example = answer key
> - Judge LLM = grader
> - Threshold = passing score

### Code walkthrough — Golden Examples (before running)

> ```python
> RESPONSE_EXAMPLES = {
>     "urgent": (
>         "Thank you for alerting us. We have escalated this to our on-call team "
>         "and will provide an update within 1 hour..."
>     ),
>     "spam": (
>         "This message has been identified as unsolicited commercial email..."
>     ),
>     "normal": (
>         "Thank you for your email. We have received your message and a team "
>         "member will respond within 24 hours..."
>     ),
> }
> ```
>
> We pre-write "ideal responses" for each category.
> We will evaluate how **similar** the AI-generated response is to these.

### Code walkthrough — SimilarityScoreOutput + Judge function

> ```python
> class SimilarityScoreOutput(BaseModel):
>     score: int = Field(gt=0, lt=100, description="Similarity score between 1 and 99")
>     reasoning: str = Field(description="Brief explanation of the score")
> ```
>
> The structure the Judge LLM returns:
> - `score` — similarity score between 1 and 99
> - `reasoning` — explanation for the score
>
> `gt=0, lt=100` — Pydantic constraints. Must be greater than 0 and less than 100.
>
> ```python
> judge_llm = llm.with_structured_output(SimilarityScoreOutput)
> ```
>
> Again `with_structured_output`! This time structuring score and reasoning.
>
> Judge function:
>
> ```python
> def judge_response(generated: str, golden: str) -> SimilarityScoreOutput:
>     prompt = f"""You are an expert quality evaluator. Compare the generated response
> with the golden (ideal) response and score their similarity.
>
> Consider:
> - Tone and professionalism (30%)
> - Key information coverage (40%)
> - Appropriate length and format (30%)
>
> Golden Response:
> {golden}
>
> Generated Response:
> {generated}
>
> Score from 1 to 99..."""
>     return judge_llm.invoke(prompt)
> ```
>
> The prompt specifies evaluation criteria:
> - Tone and professionalism 30%
> - Key information coverage 40%
> - Appropriate length and format 30%
>
> The Judge LLM scores according to these criteria.

### After running — Judge test

> ```
> Category: urgent
> AI Response: "We have received your critical alert..."
> Similarity Score: 82
> Reasoning: "Both responses acknowledge urgency and promise timely action..."
> ```
>
> Score 82! Above the threshold of 70, so PASS.

### Code walkthrough — pytest with Judge

> Looking at `%%writefile tests_judge.py`:
>
> ```python
> THRESHOLD = 70
>
> def test_response_quality_above_threshold(ai_graph, email, expected_category):
>     result = ai_graph.invoke({"email": email})
>     golden = RESPONSE_EXAMPLES[expected_category]
>     score_result = judge_response(result["response"], golden)
>
>     assert score_result.score >= THRESHOLD
> ```
>
> Compare AI-generated response against golden example.
> If the Judge's score is 70 or above, the test passes.
>
> There are also two edge case tests:
>
> ```python
> def test_judge_perfect_match():
>     golden = RESPONSE_EXAMPLES["urgent"]
>     score_result = judge_response(golden, golden)
>     assert score_result.score >= 90
> ```
>
> Feeding the golden example to itself should score 90+. (It is comparing to itself.)
>
> ```python
> def test_judge_poor_match():
>     poor_response = "lol ok whatever"
>     score_result = judge_response(poor_response, golden)
>     assert score_result.score < 40
> ```
>
> A terrible response should score below 40.
>
> This validates the Judge itself. "Is the Judge judging correctly?"

### After running — pytest

> ```
> tests_judge.py::test_response_quality_above_threshold[EMERGENCY...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[FREE GIFT...]   PASSED
> tests_judge.py::test_response_quality_above_threshold[Hi, could...]   PASSED
> tests_judge.py::test_judge_perfect_match                               PASSED
> tests_judge.py::test_judge_poor_match                                  PASSED
> ```
>
> All passed!

---

## Closing — The Evolution of Testing Strategies (4 min)

> Let us wrap up what we learned today.
>
> **The evolution of testing strategies:**
>
> ```
> 17.1-17.2  Rule-based  -->  assert == (exact match)
> 17.4-17.5  AI-based    -->  assert in (range validation)
> 17.6       LLM Judge   -->  assert score >= threshold (quality validation)
> ```
>
> As we move from deterministic to non-deterministic systems,
> tests become **more relaxed but more realistic**.
>
> Key tools summary:
>
> | Tool | Purpose |
> |------|---------|
> | `%%writefile` | Create .py files from notebooks |
> | `@pytest.mark.parametrize` | Test multiple cases at once |
> | `graph.nodes["name"].invoke()` | Unit test individual nodes |
> | `update_state(as_node="...")` | Inject state for partial execution |
> | `with_structured_output` | Structure LLM output |
> | `SimilarityScoreOutput` | Judge score schema |
>
> Before moving to the next chapter, try the Final Exercises.
> Especially **Exercise 2 (AI classification accuracy benchmark)** is the
> most commonly used pattern in real-world testing.
>
> Great work today!
