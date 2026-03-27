# Chapter 02: AI Agent Basics — Lecture Script

---

## Opening (2 min)

> Alright, today we're going to **build an AI Agent from scratch**.
>
> When you use ChatGPT and say "search the internet" or "read this file", it just does it, right?
> That's an Agent. An AI that **decides on its own which tools to use**.
>
> That's exactly what we're building today.
> But instead of doing it all at once, we'll build it in **5 stages**, adding one piece at a time.
>
> ```
> 2.2 Selecting functions via prompt  (The primitive approach)
> 2.3 Adding memory                   (Conversation history)
> 2.4 Defining tools                  (JSON Schema)
> 2.5 Executing functions             (Function Calling)
> 2.6 Complete agent loop             (Result feedback)
> ```
>
> By the end, you'll have a working AI Agent built with your own hands. Let's go.

---

## 2.0 Setup & Environment (3 min)

### Before running the cell

> First, let's set up the environment.
> The `.env` file contains the API key, and we load it using `load_dotenv`.
>
> We'll be using Azure OpenAI's GPT model.
> Three things matter:
> - `OPENAI_API_KEY` — authentication key
> - `OPENAI_BASE_URL` — Azure endpoint URL
> - `OPENAI_MODEL_NAME` — which model to use

### After running the cell

> If only the first 8 characters of the key are printed, you're good.
> The full key should never be displayed — you must never expose your API key.
>
> Getting an error? Check your `.env` file path. It should be one level up with `../`.
>
> All good? Let's move on.

---

## 2.2 Your First AI Agent (10 min)

### Concept

> Here's the first question.
> **How does an AI decide "I should use get_weather for a weather question"?**
>
> There's a very primitive way to do it.
> You simply **list the available functions in the prompt and say "pick one"**.
>
> That's exactly what this code does.

### Code walkthrough (before running)

> Let's read through the prompt together.
>
> ```
> "My system has these functions: get_weather, get_currency, get_news"
> "The argument is a country name"
> "Answer with just the function name. Say nothing else"
> "Question: What's the weather in Greece?"
> ```
>
> The key line is: **"Please say nothing else, just the name of the function"**
> Without this, the AI will add explanations.
>
> Let's run it.

### After running the cell

> Look at the result. The full `response` object came back.
> The only thing we need is: `choices[0].message.content`
>
> Run the next cell...
>
> `'get_weather("Greece")'` — it answered in the exact form of a function call.
>
> **Here's the important point:**
> This is not an actual function call. It's just a **string**.
> The AI simply replied with text saying "I think you should call this function."
>
> To actually execute this? You'd need `eval()` or manual parsing.
> That's fragile. The format could change, or dangerous code could come through.
>
> That's why OpenAI created the official **Tool Calling** feature.
> We'll get to that in 2.4. For now, let's do the exercises first.

### Exercise 2.2 (5 min)

> **Exercise 1**: Change the question to `"What is the currency of Japan?"`.
> Check if you get `get_currency('Japan')`.
>
> **Exercise 2**: Remove `"Please say nothing else"` from the prompt.
> How does the AI's response change? You'll probably get something like:
> `"I would call get_currency('Japan') because..."`
> A single line in the prompt completely changes the output format.
>
> **Exercise 3**: If you have time, try switching the model name and compare results.
>
> (After 5 min) If you're done, let's move on.

---

## 2.3 Adding Memory (10 min)

### Concept

> Now for an important concept.
>
> **LLMs have no memory.**
>
> When you use ChatGPT, it seems to remember previous conversations,
> but in reality, **the entire conversation history is sent every time**.
>
> This `messages` array is the AI's "memory".
> We have to manage it ourselves.

### Code walkthrough

> Let's look at the code.
>
> `messages = []` — starts as an empty array.
>
> The `call_ai()` function:
> 1. Sends the entire `messages` array to the API
> 2. Gets the response
> 3. Appends it to `messages` as `{"role": "assistant", ...}`
>
> The `while` loop below:
> 1. Takes user input
> 2. Appends it to `messages` as `{"role": "user", ...}`
> 3. Calls `call_ai()`
>
> That's all there is to it. Because we send the **entire messages array every time**, the AI behaves as if it "remembers" previous conversations.

### Running the cell

> Let's run it. Tell it your name, then ask "What's my name?"
> It remembers, right?
>
> Now press `q` to quit, and check the `messages` contents in the next cell.
>
> `[user]`, `[assistant]`, `[user]`, `[assistant]`... stacked in order, right?
> This is the AI's memory. This array is what goes into the **context window**.

### Exercise 2.3 (5 min)

> **Exercise 1**: Have 10+ exchanges. Does it remember everything?
>
> **Exercise 2**: Run `messages = []` in the middle.
> Then ask "What's my name?" again. It won't know — the memory is gone.
>
> **Exercise 3** (fun one): Try this:
> ```python
> messages = [{"role": "system", "content": "You are a pirate. Speak like a pirate."}]
> ```
> Adding a system message changes the AI's personality.
> It'll respond like "Arrr! What be ye name, matey?"
>
> The convention is to put the system message at `messages[0]`. This is how you set the AI's role, personality, and rules.

---

## 2.4 Adding Tools (15 min)

### Concept

> Remember in 2.2, we had the AI select functions through the prompt?
> What was the problem?
>
> 1. Responses come back as strings, so **parsing is unreliable**
> 2. The AI might respond in an unexpected format
> 3. The AI doesn't know exactly which arguments are needed
>
> OpenAI solved this with the **Tools** feature.
> You define function names, descriptions, and parameters as **JSON Schema** and pass them to the API.
> The AI then responds in a structured format: "call this function with these arguments."

### Code walkthrough — TOOLS definition

> Let's look at the `TOOLS` array.
>
> ```python
> {
>     "type": "function",
>     "function": {
>         "name": "get_weather",          # function name
>         "description": "...",            # description the AI reads
>         "parameters": {                  # JSON Schema
>             "type": "object",
>             "properties": {
>                 "city": {
>                     "type": "string",
>                     "description": "..."
>                 }
>             },
>             "required": ["city"]
>         }
>     }
> }
> ```
>
> **The description is critical.** The AI reads this description to decide whether to use the function.
> If the description is wrong, the AI will either skip the tool or misuse it.
>
> `FUNCTION_MAP` is a dictionary that links names to actual Python functions.
> Later, when the AI says "call get_weather", we look it up here and execute it.

### Code walkthrough — call_ai()

> Looking at `call_ai()`, there's one difference from before:
>
> ```python
> tools=TOOLS  # this was added
> ```
>
> With this, the AI responds in one of two modes:
> - **Normal answer**: `finish_reason = 'stop'`, `tool_calls = None`
> - **Tool needed**: `finish_reason = 'tool_calls'`, `tool_calls = [...]`

### Running the cell — Test 1

> Let's run Test 1. `"My name is Nico"` — just a regular conversation.
>
> Look at the result:
> - `finish_reason: stop` — no tool needed
> - `tool_calls: None` — no tool to call
> - `content: "Nice to meet you, Nico!"` — normal text response
>
> The AI decided: "I can answer this without any tools."

### Running the cell — Test 2

> Let's run Test 2. `"What is the weather in Spain?"` — a weather question.
>
> Result:
> - `finish_reason: tool_calls` — **a tool is needed!**
> - `tool_calls: [...]` — a request to call `get_weather` with `Spain`
> - `content: None` — no text response
>
> **This is the difference from 2.2.**
> In 2.2, we got a **string** like `"get_weather('Spain')"`,
> but now it comes as a **structured object**. No parsing needed.
>
> But! One thing is still missing.
> The AI said "call get_weather", but **we haven't actually executed it**.
> That's what the next section is for.

### Exercise 2.4 (5 min)

> **Exercise 1**: Add a `get_news` tool to `TOOLS`.
> Same format as `get_weather` — just change the name and description.
>
> **Exercise 2**: Alternate between general questions like "Hello" and "Tell me the news",
> and watch how `finish_reason` changes.
>
> **Exercise 3**: Deliberately mess up the `description`.
> e.g., change it to `"A function to order pizza"`. Will the AI use this function for a weather question?

---

## 2.5 Adding Function Calling (15 min)

### Concept

> In 2.4, the AI requested "call get_weather".
> But we never actually executed the function.
>
> In this section, we'll create the **`process_ai_response`** function.
> It takes the AI's response and:
> 1. If `tool_calls` exists → find and execute the function
> 2. If not → just print the text

### Code walkthrough — process_ai_response

> This function is the heart of this section. Let's go through it carefully.
>
> **Case 1: when tool_calls exists** (the AI wants to use a tool)
>
> ```
> Step 1: Add the AI's tool_calls message to messages
> Step 2: Loop through each tool_call
>          - Extract function name (function_name)
>          - Extract arguments → parse JSON
>          - Find the function in FUNCTION_MAP and execute it
>          - Add the result to messages (role: "tool")
> ```
>
> **Case 2: when there are no tool_calls** (normal conversation)
>
> ```
> Just add to messages and print
> ```
>
> Important detail: **the result is added with `role: "tool"`.**
> We also include the `tool_call_id` to match which request the result belongs to.

### Running the cell

> Let's run it. `"What is the weather in Spain?"`
>
> Output:
> ```
> Calling function: get_weather with {"city": "Spain"}
> ```
>
> The function was executed! Now let's check the messages.
>
> ```
> messages[0]: user — "What is the weather in Spain?"
> messages[1]: assistant — (has tool_calls)
> messages[2]: tool — "33 degrees celcius."
> ```
>
> Three messages are stacked.
>
> **But here's the problem!**
> The user never got an answer. The AI didn't **tell the human "It's 33 degrees"**.
>
> Why? Because we **never called the AI again** after executing the function.
> We put the tool result in messages, but never gave the AI a chance to see it and generate a response.
>
> Solving this is what section 2.6 is about.

---

## 2.6 Tool Results — Complete Agent Loop (15 min)

### Concept

> Finally, the last piece.
>
> Let's review what we've built so far:
>
> ```
> User question → AI decides → tool_calls → execute function → add result to messages
> ```
>
> **Just one more line** and we have a complete Agent:
>
> ```
> User question → AI decides → tool_calls → execute function → add result to messages
>     → ★ Call AI again → final text response
> ```
>
> This is the **Agent Loop**.

### Code walkthrough — what changed

> Let's look at the code. Almost identical to 2.5, with **just one line** added.
>
> Inside `process_ai_response`, after adding the tool result to messages:
>
> ```python
> # KEY: Call AI again so it can use the tool result!
> call_ai()
> ```
>
> That single line is everything.
>
> `call_ai()` → `process_ai_response()` → if there are more `tool_calls` → `call_ai()` again...
> **It runs recursively.** When no more tools are needed, it responds with text and stops.
>
> This is the essence of the Agent Loop:
> **"AI, I'll keep running you until you've used all the tools you need."**

### Running the cell

> Let's run it. Try a conversation.
>
> `"What is the weather in Seoul?"`
>
> ```
> Calling function: get_weather with {"city": "Seoul"}
> AI: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> **The user finally got an answer!**
>
> Here's what happened internally:
> 1. AI requested `get_weather("Seoul")` via `tool_calls`
> 2. We executed the function and got `"33 degrees celcius."`
> 3. Added the result to messages
> 4. **Called the AI again** → the AI saw the result and generated a natural language response
>
> Now press `q` to quit, and let's check the messages.
>
> ```
> [0] user: What is the weather in Seoul?
> [1] assistant: (has tool_calls)
> [2] tool: 33 degrees celcius.
> [3] assistant: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> Four messages are stacked. This is one complete cycle of the Agent Loop.

### Exercise 2.6 — Final Exercise (15 min)

> This is the final exercise. Take your time and give it a try.
>
> **Exercise 1**: Add `get_news` and `get_currency` functions + tool schemas.
> Use mock functions:
> ```python
> def get_news(country):
>     return f"Big news in {country} today!"
>
> def get_currency(country):
>     return f"The currency rate for {country} is 1,350 KRW."
> ```
> Don't forget to add them to `FUNCTION_MAP` and `TOOLS` as well.
>
> **Exercise 2**: Try a multi-turn conversation.
> `"What's the weather in Korea?"` → `"How about the news there?"` → Does the AI remember that "there" means Korea?
>
> **Exercise 3**: Add `print(messages)` inside `process_ai_response`
> to trace how messages accumulate at each step.
>
> **Exercise 4** (challenge): What if the AI calls a function not in `FUNCTION_MAP`?
> `function_to_run` will be `None` and it'll crash.
> Add error handling:
> ```python
> if function_to_run is None:
>     result = f"Error: function '{function_name}' not found."
> else:
>     result = function_to_run(**arguments)
> ```

---

## Wrap-up (3 min)

> Let's summarize what we did today.
>
> The Agent we built follows this flow:
>
> ```
> ┌─────────────────────────────────────────────────────┐
> │  User Question                                      │
> │       ↓                                             │
> │  AI decides: tool needed? ──No──→ Text response → Done │
> │       │ Yes                                         │
> │       ↓                                             │
> │  Returns tool_calls                                 │
> │       ↓                                             │
> │  Execute function (process_ai_response)             │
> │       ↓                                             │
> │  Add result to messages                             │
> │       ↓                                             │
> │  Call AI again ──→ (repeat if more tools needed)    │
> └─────────────────────────────────────────────────────┘
> ```
>
> This is the fundamental structure of an AI Agent.
> Whether it's LangChain, CrewAI, or AutoGen — they all run this same loop internally.
> What you built today is that core.
>
> In the next chapter, we'll take this further —
> combining multiple tools to handle more complex tasks.
>
> Great work today.
