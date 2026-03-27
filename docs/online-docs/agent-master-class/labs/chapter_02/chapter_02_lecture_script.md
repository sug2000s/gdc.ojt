# Chapter 02: AI Agent Basics — 강의 대본

---

## 오프닝 (2분)

> 자, 오늘은 AI Agent를 **처음부터 직접 만들어봅니다**.
>
> ChatGPT 쓸 때 "인터넷 검색해줘", "파일 읽어줘" 하면 알아서 하죠?
> 그게 바로 Agent입니다. AI가 **스스로 판단해서 도구를 골라 쓰는 것**.
>
> 오늘 우리가 만들 건 바로 그겁니다.
> 단, 한 번에 완성하지 않고 **5단계로 나눠서** 하나씩 붙여나갑니다.
>
> ```
> 2.2 프롬프트로 함수 선택  (원시적 방법)
> 2.3 메모리 추가           (대화 기록)
> 2.4 도구 정의             (JSON Schema)
> 2.5 함수 실행             (Function Calling)
> 2.6 완전한 에이전트 루프   (결과 피드백)
> ```
>
> 끝나면 여러분 손으로 만든 AI Agent가 동작하게 됩니다. 시작합시다.

---

## 2.0 Setup & Environment (3분)

### 셀 실행 전

> 먼저 환경 세팅부터 합니다.
> `.env` 파일에 API 키가 들어있고, 이걸 `load_dotenv`로 불러옵니다.
>
> 우리는 Azure OpenAI의 GPT 모델을 사용합니다.
> 중요한 건 세 가지:
> - `OPENAI_API_KEY` — 인증 키
> - `OPENAI_BASE_URL` — Azure 엔드포인트 주소
> - `OPENAI_MODEL_NAME` — 사용할 모델명

### 셀 실행 후

> 키 앞 8자리만 찍히면 정상입니다.
> 전체가 출력되면 안 됩니다 — API 키는 절대 노출하면 안 되거든요.
>
> 에러가 나는 분? `.env` 파일 경로 확인해주세요. `../` 한 단계 위에 있어야 합니다.
>
> 다 되셨으면 다음으로 넘어갑니다.

---

## 2.2 Your First AI Agent (10분)

### 개념 설명

> 자, 첫 번째 질문입니다.
> **AI가 어떻게 "날씨 물어보면 get_weather를 써야지" 하고 판단할까요?**
>
> 가장 원시적인 방법이 있습니다.
> 그냥 **프롬프트에 함수 목록을 적어주고, "골라줘"** 라고 하는 겁니다.
>
> 지금 이 코드가 정확히 그걸 합니다.

### 코드 설명 (셀 실행 전)

> 프롬프트를 같이 읽어봅시다.
>
> ```
> "내 시스템에 이런 함수들이 있다: get_weather, get_currency, get_news"
> "인자는 나라 이름이다"
> "함수 이름만 답해라. 다른 말은 하지 마라"
> "질문: 그리스 날씨는?"
> ```
>
> 핵심은 이 줄입니다: **"Please say nothing else, just the name of the function"**
> 이게 없으면 AI가 설명을 덧붙입니다.
>
> 자, 실행해봅시다.

### 셀 실행 후

> 결과를 봅시다. `response` 객체가 통째로 나왔죠.
> 여기서 우리가 필요한 건 딱 하나: `choices[0].message.content`
>
> 다음 셀 실행하면...
>
> `'get_weather("Greece")'` — 정확히 함수 호출 형태로 답했습니다.
>
> **여기서 중요한 포인트:**
> 이건 진짜 함수 호출이 아닙니다. 그냥 **문자열**입니다.
> AI가 "이 함수를 부르면 될 것 같아요"라고 텍스트로 답한 것 뿐이에요.
>
> 이걸 실제로 실행하려면? `eval()`을 쓰거나 파싱을 해야 합니다.
> 불안정하죠. 포맷이 바뀔 수도 있고, 위험한 코드가 올 수도 있습니다.
>
> 그래서 OpenAI가 공식적으로 **Tool Calling** 기능을 만든 겁니다.
> 그건 2.4에서 할 거고, 지금은 Exercise 먼저 해봅시다.

### Exercise 2.2 (5분)

> **Exercise 1**: 질문을 `"What is the currency of Japan?"`으로 바꿔보세요.
> `get_currency('Japan')` 이 나오는지 확인.
>
> **Exercise 2**: 프롬프트에서 `"Please say nothing else"` 를 지워보세요.
> AI가 어떻게 달라지나요? 아마 이런 식으로 나올 겁니다:
> `"I would call get_currency('Japan') because..."`
> 프롬프트 한 줄이 출력 형태를 완전히 바꿉니다.
>
> **Exercise 3**: 시간 되시면 모델명을 바꿔서 비교해보세요.
>
> (5분 후) 다 하셨으면 넘어갑니다.

---

## 2.3 Adding Memory (10분)

### 개념 설명

> 이제 중요한 개념입니다.
>
> **LLM은 기억력이 없습니다.**
>
> ChatGPT를 쓸 때 이전 대화를 기억하는 것 같지만,
> 실제로는 **매번 전체 대화 기록을 다시 보내는 것**입니다.
>
> 이 `messages` 배열이 바로 AI의 "기억"입니다.
> 우리가 직접 관리해야 합니다.

### 코드 설명

> 코드를 봅시다.
>
> `messages = []` — 빈 배열로 시작.
>
> `call_ai()` 함수를 보면:
> 1. `messages` 전체를 API에 보냄
> 2. 응답을 받아서
> 3. `messages`에 `{"role": "assistant", ...}` 로 추가
>
> 아래 `while` 루프는:
> 1. 사용자 입력을 받아서
> 2. `messages`에 `{"role": "user", ...}` 로 추가
> 3. `call_ai()` 호출
>
> 이게 전부입니다. 매번 **전체 messages를 보내니까** AI가 이전 대화를 "아는 것처럼" 동작합니다.

### 셀 실행

> 실행해봅시다. 이름 알려주고, 다음에 "내 이름이 뭐야?" 물어보세요.
> 기억하죠?
>
> 이제 `q`를 눌러서 종료하고, 다음 셀에서 `messages` 내용을 확인해봅시다.
>
> `[user]`, `[assistant]`, `[user]`, `[assistant]`... 순서대로 쌓여있죠?
> 이게 AI의 기억입니다. 이 배열이 곧 **context window**에 들어가는 겁니다.

### Exercise 2.3 (5분)

> **Exercise 1**: 대화를 10번 이상 해보세요. 전부 기억하나요?
>
> **Exercise 2**: 중간에 `messages = []` 을 실행해보세요.
> 그리고 다시 "내 이름이 뭐야?" 물어보면? — 모릅니다. 기억이 날아갔으니까.
>
> **Exercise 3** (재밌는 거): 이걸 해보세요:
> ```python
> messages = [{"role": "system", "content": "You are a pirate. Speak like a pirate."}]
> ```
> 시스템 메시지를 넣으면 AI의 성격이 바뀝니다.
> "Arrr! What be ye name, matey?" 이런 식으로 답할 겁니다.
>
> 시스템 메시지는 `messages[0]`에 넣는 게 관례입니다. 이걸로 AI의 역할, 성격, 규칙을 정합니다.

---

## 2.4 Adding Tools (15분)

### 개념 설명

> 자, 2.2에서 프롬프트로 함수를 선택하게 했죠.
> 문제가 뭐였습니까?
>
> 1. 문자열로 돌아오니까 **파싱이 불안정**
> 2. AI가 엉뚱한 포맷으로 답할 수 있음
> 3. 어떤 인자가 필요한지 AI가 정확히 모름
>
> OpenAI가 이걸 해결한 게 **Tools** 기능입니다.
> 함수의 이름, 설명, 파라미터를 **JSON Schema로 정의**해서 API에 넘기면,
> AI가 구조화된 형태로 "이 함수를 이 인자로 불러줘" 라고 응답합니다.

### 코드 설명 — TOOLS 정의

> `TOOLS` 배열을 봅시다.
>
> ```python
> {
>     "type": "function",
>     "function": {
>         "name": "get_weather",          # 함수 이름
>         "description": "...",            # AI가 읽는 설명
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
> **description이 핵심입니다.** AI는 이 설명을 읽고 "이 함수를 쓸지 말지" 판단합니다.
> description을 잘못 쓰면 AI가 도구를 안 쓰거나, 잘못 씁니다.
>
> `FUNCTION_MAP`은 이름과 실제 파이썬 함수를 연결하는 딕셔너리입니다.
> 나중에 AI가 "get_weather 불러줘" 하면, 여기서 찾아서 실행합니다.

### 코드 설명 — call_ai()

> `call_ai()`를 보면, 이전과 다른 게 하나 있습니다:
>
> ```python
> tools=TOOLS  # 이거 추가됨
> ```
>
> 이걸 넘기면 AI가 두 가지 모드로 응답합니다:
> - **일반 답변**: `finish_reason = 'stop'`, `tool_calls = None`
> - **도구 필요**: `finish_reason = 'tool_calls'`, `tool_calls = [...]`

### 셀 실행 — Test 1

> Test 1 실행합니다. `"My name is Nico"` — 일반 대화죠.
>
> 결과 보세요:
> - `finish_reason: stop` ← 도구 필요 없음
> - `tool_calls: None` ← 호출할 도구 없음
> - `content: "Nice to meet you, Nico!"` ← 일반 텍스트 응답
>
> AI가 판단한 겁니다: "이건 도구 안 써도 답할 수 있어."

### 셀 실행 — Test 2

> Test 2 실행합니다. `"What is the weather in Spain?"` — 날씨 질문.
>
> 결과:
> - `finish_reason: tool_calls` ← **도구가 필요하다!**
> - `tool_calls: [...]` ← `get_weather`를 `Spain`으로 부르라는 요청
> - `content: None` ← 텍스트 응답 없음
>
> **여기가 2.2와의 차이입니다.**
> 2.2에서는 `"get_weather('Spain')"` 이라는 **문자열**이 왔지만,
> 지금은 **구조화된 객체**로 옵니다. 파싱할 필요가 없어요.
>
> 하지만! 아직 한 가지 빠졌습니다.
> AI가 "get_weather 불러줘" 라고 했는데, **실제로 실행은 안 했습니다.**
> 그건 다음 섹션에서 합니다.

### Exercise 2.4 (5분)

> **Exercise 1**: `get_news` 도구를 `TOOLS`에 추가해보세요.
> 형식은 `get_weather`와 똑같이, name과 description만 바꾸면 됩니다.
>
> **Exercise 2**: "Hello" 같은 일반 질문과 "뉴스 알려줘" 를 번갈아 하면서
> `finish_reason`이 바뀌는 걸 확인해보세요.
>
> **Exercise 3**: `description`을 일부러 이상하게 바꿔보세요.
> 예: `"A function to order pizza"` 로 바꾸면 AI가 날씨 질문에 이 함수를 쓸까요?

---

## 2.5 Adding Function Calling (15분)

### 개념 설명

> 2.4에서 AI가 "get_weather를 불러줘" 라고 요청했죠.
> 하지만 실제로 함수를 실행한 적은 없습니다.
>
> 이번 섹션에서는 **`process_ai_response`** 함수를 만듭니다.
> AI 응답을 받아서:
> 1. `tool_calls`가 있으면 → 함수를 찾아서 실행
> 2. 없으면 → 그냥 텍스트 출력

### 코드 설명 — process_ai_response

> 이 함수가 이번 섹션의 핵심입니다. 천천히 봅시다.
>
> **Case 1: tool_calls가 있을 때** (AI가 도구를 쓰고 싶을 때)
>
> ```
> Step 1: AI의 tool_calls 메시지를 messages에 추가
> Step 2: 각 tool_call을 순회하면서
>          - 함수 이름 추출 (function_name)
>          - 인자 추출 (arguments) → JSON 파싱
>          - FUNCTION_MAP에서 함수를 찾아서 실행
>          - 결과를 messages에 추가 (role: "tool")
> ```
>
> **Case 2: tool_calls가 없을 때** (일반 대화)
>
> ```
> 그냥 messages에 추가하고 출력
> ```
>
> 중요한 디테일: **`role: "tool"`로 결과를 넣습니다.**
> `tool_call_id`도 같이 넣어서 어떤 요청에 대한 결과인지 매칭합니다.

### 셀 실행

> 실행해봅시다. `"What is the weather in Spain?"`
>
> 출력:
> ```
> Calling function: get_weather with {"city": "Spain"}
> ```
>
> 함수가 실행됐습니다! 이제 messages를 확인해봅시다.
>
> ```
> messages[0]: user — "What is the weather in Spain?"
> messages[1]: assistant — (has tool_calls)
> messages[2]: tool — "33 degrees celcius."
> ```
>
> 세 개의 메시지가 쌓여있습니다.
>
> **그런데 여기서 문제!**
> 사용자한테 답이 안 갔습니다. AI가 "33도입니다" 라고 **사람한테 말해주지 않았어요**.
>
> 왜냐면? 함수 실행 후에 **AI를 다시 안 불렀으니까**.
> 도구 결과를 messages에 넣었지만, 그걸 보고 답변을 만들 기회를 안 줬습니다.
>
> 이걸 해결하는 게 다음 섹션, 2.6입니다.

---

## 2.6 Tool Results — Complete Agent Loop (15분)

### 개념 설명

> 드디어 마지막 조각입니다.
>
> 지금까지 만든 걸 정리하면:
>
> ```
> 사용자 질문 → AI 판단 → tool_calls → 함수 실행 → 결과를 messages에 추가
> ```
>
> 여기서 **한 줄만 추가**하면 완전한 Agent가 됩니다:
>
> ```
> 사용자 질문 → AI 판단 → tool_calls → 함수 실행 → 결과를 messages에 추가
>     → ★ AI를 다시 호출 → 최종 텍스트 응답
> ```
>
> 이게 바로 **Agent Loop**입니다.

### 코드 설명 — 달라진 점

> 코드를 봅시다. 2.5와 거의 똑같은데, **딱 한 줄** 추가됐습니다.
>
> `process_ai_response` 안에서 tool 결과를 messages에 넣은 후:
>
> ```python
> # KEY: Call AI again so it can use the tool result!
> call_ai()
> ```
>
> 이 한 줄이 전부입니다.
>
> `call_ai()` → `process_ai_response()` → 또 `tool_calls`면 → 또 `call_ai()`...
> **재귀적으로 돌아갑니다.** 도구가 더 필요 없으면 텍스트로 답하고 끝.
>
> 이게 Agent Loop의 본질입니다:
> **"AI야, 네가 필요한 도구를 다 쓸 때까지 계속 돌려줄게."**

### 셀 실행

> 실행합시다. 대화해보세요.
>
> `"What is the weather in Seoul?"`
>
> ```
> Calling function: get_weather with {"city": "Seoul"}
> AI: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> **드디어 사용자한테 답이 왔습니다!**
>
> 내부적으로 무슨 일이 벌어졌나면:
> 1. AI가 `tool_calls`로 `get_weather("Seoul")` 요청
> 2. 우리가 함수를 실행해서 `"33 degrees celcius."` 결과를 받음
> 3. 결과를 messages에 추가
> 4. **AI를 다시 호출** → AI가 결과를 보고 자연어로 답변 생성
>
> 이제 `q`로 종료하고, messages를 확인해봅시다.
>
> ```
> [0] user: What is the weather in Seoul?
> [1] assistant: (has tool_calls)
> [2] tool: 33 degrees celcius.
> [3] assistant: The weather in Seoul is currently 33 degrees Celsius.
> ```
>
> 4개의 메시지가 쌓여있습니다. 이게 완전한 Agent Loop의 한 사이클입니다.

### Exercise 2.6 — Final Exercise (15분)

> 마지막 Exercise입니다. 시간 충분히 드릴 테니 도전해보세요.
>
> **Exercise 1**: `get_news`와 `get_currency` 함수 + 도구 스키마를 추가하세요.
> 함수는 mock으로 만들면 됩니다:
> ```python
> def get_news(country):
>     return f"Big news in {country} today!"
>
> def get_currency(country):
>     return f"The currency rate for {country} is 1,350 KRW."
> ```
> `FUNCTION_MAP`과 `TOOLS`에도 추가하는 거 잊지 마세요.
>
> **Exercise 2**: 멀티턴 대화를 해보세요.
> `"한국 날씨 알려줘"` → `"거기 뉴스는?"` → AI가 "거기"가 한국인 걸 기억하나요?
>
> **Exercise 3**: `process_ai_response` 안에 `print(messages)` 를 넣어서
> 매 단계마다 messages가 어떻게 쌓이는지 추적해보세요.
>
> **Exercise 4** (도전): AI가 `FUNCTION_MAP`에 없는 함수를 부르면?
> `function_to_run`이 `None`이 되고 터집니다.
> 에러 핸들링을 추가해보세요:
> ```python
> if function_to_run is None:
>     result = f"Error: function '{function_name}' not found."
> else:
>     result = function_to_run(**arguments)
> ```

---

## 마무리 (3분)

> 오늘 한 걸 정리합니다.
>
> 우리가 만든 Agent는 이 흐름을 따릅니다:
>
> ```
> ┌─────────────────────────────────────────────────┐
> │  User Question                                  │
> │       ↓                                         │
> │  AI 판단: 도구 필요? ──No──→ 텍스트 응답 → 끝    │
> │       │ Yes                                     │
> │       ↓                                         │
> │  tool_calls 반환                                 │
> │       ↓                                         │
> │  함수 실행 (process_ai_response)                 │
> │       ↓                                         │
> │  결과를 messages에 추가                           │
> │       ↓                                         │
> │  AI 재호출 ──→ (다시 도구 필요하면 반복)           │
> └─────────────────────────────────────────────────┘
> ```
>
> 이게 AI Agent의 기본 구조입니다.
> LangChain이든, CrewAI든, AutoGen이든 내부적으로 다 이 루프를 돌립니다.
> 오늘 여러분이 만든 게 그 핵심입니다.
>
> 다음 챕터에서는 이걸 더 발전시켜서
> 여러 도구를 조합하고, 더 복잡한 작업을 처리하는 Agent를 만들어보겠습니다.
>
> 수고하셨습니다.
