# Chapter 2: AI 에이전트 기초

---

## 챕터 개요

이 챕터에서는 OpenAI API를 활용하여 처음부터 AI 에이전트를 구축하는 과정을 단계별로 학습한다. 단순한 API 호출에서 시작하여, 대화 메모리를 추가하고, 외부 도구(Tool)를 정의하며, 함수 호출(Function Calling)을 통해 AI가 실제 함수를 실행할 수 있는 완전한 에이전트를 만드는 것까지 진행한다.

### 학습 목표

1. Python 개발 환경을 설정하고 OpenAI API 연동을 확인한다
2. OpenAI Chat Completions API의 기본 사용법을 익힌다
3. 대화 히스토리(메모리)를 관리하여 문맥을 유지하는 챗봇을 만든다
4. Tool 스키마를 정의하여 AI에게 사용 가능한 함수를 알려준다
5. Function Calling을 구현하여 AI가 선택한 함수를 실제로 실행한다
6. Tool 실행 결과를 AI에게 다시 전달하여 최종 응답을 생성한다

### 챕터 구조

| 섹션 | 주제 | 핵심 키워드 |
|------|------|------------|
| 2.0 | 프로젝트 셋업 | uv, Python 3.13, OpenAI SDK |
| 2.2 | 첫 번째 AI 에이전트 | Chat Completions API, 프롬프트 엔지니어링 |
| 2.3 | 메모리 추가 | 대화 히스토리, messages 배열, while 루프 |
| 2.4 | 도구 추가 | Tools 스키마, JSON Schema, FUNCTION_MAP |
| 2.5 | 함수 호출 추가 | Function Calling, tool_calls, process_ai_response |
| 2.6 | 도구 실행 결과 | Tool Results, 재귀 호출, 에이전트 루프 완성 |

---

## 2.0 프로젝트 셋업 (Setup)

### 주제 및 목표

AI 에이전트를 개발하기 위한 Python 프로젝트 환경을 구성한다. `uv` 패키지 매니저를 사용하여 프로젝트를 초기화하고, OpenAI Python SDK를 설치하며, Jupyter Notebook 환경에서 API 키가 정상적으로 로드되는지 확인한다.

### 핵심 개념 설명

#### uv 패키지 매니저

`uv`는 Rust로 작성된 차세대 Python 패키지 매니저로, 기존의 `pip`이나 `poetry`보다 훨씬 빠른 의존성 해결과 설치 속도를 제공한다. 이 프로젝트에서는 `uv`를 사용하여 가상환경을 생성하고 패키지를 관리한다.

#### 프로젝트 구조

프로젝트는 다음과 같은 구조로 구성된다:

```
my-first-agent/
├── .gitignore          # Git에서 추적하지 않을 파일 목록
├── .python-version     # Python 버전 지정 (3.13)
├── README.md           # 프로젝트 설명
├── main.ipynb          # 메인 노트북 (코드 작성 공간)
├── pyproject.toml      # 프로젝트 설정 및 의존성 정의
└── uv.lock             # 의존성 잠금 파일
```

### 코드 분석

#### pyproject.toml - 프로젝트 설정 파일

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

**핵심 포인트:**

- `requires-python = ">=3.13"`: Python 3.13 이상을 요구한다. 최신 Python 기능을 활용하기 위함이다.
- `dependencies`: 런타임 의존성으로 `openai>=1.98.0`을 지정한다. 이것이 OpenAI API와 통신하는 공식 Python SDK이다.
- `[dependency-groups] dev`: 개발 의존성으로 `ipykernel`을 포함한다. 이는 Jupyter Notebook에서 Python 코드를 실행하기 위해 필요하다.

#### .python-version

```
3.13
```

이 파일은 `uv`나 `pyenv` 같은 도구가 프로젝트에서 사용할 Python 버전을 자동으로 인식하게 해준다.

#### main.ipynb - API 키 확인

```python
import os

print(os.getenv("OPENAI_API_KEY"))
```

이 코드는 환경변수에서 OpenAI API 키가 올바르게 설정되어 있는지 확인한다. API 키는 보안상 코드에 직접 하드코딩하지 않고, `.env` 파일이나 시스템 환경변수를 통해 관리해야 한다.

#### .gitignore - 무시 파일 설정

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

**중요:** `.env` 파일이 `.gitignore`에 포함되어 있다. 이는 API 키와 같은 민감한 정보가 Git 저장소에 업로드되는 것을 방지한다. 보안 관점에서 매우 중요한 설정이다.

### 실습 포인트

1. `uv init my-first-agent` 명령으로 프로젝트를 초기화한다
2. `uv add openai` 명령으로 OpenAI SDK를 설치한다
3. `uv add --dev ipykernel` 명령으로 Jupyter 커널을 설치한다
4. `.env` 파일을 생성하고 `OPENAI_API_KEY=sk-...` 형태로 API 키를 저장한다
5. Jupyter Notebook을 실행하여 API 키가 정상적으로 출력되는지 확인한다

---

## 2.2 첫 번째 AI 에이전트 (Your First AI Agent)

### 주제 및 목표

OpenAI Chat Completions API를 사용하여 가장 기본적인 형태의 AI 에이전트를 만든다. 이 단계에서는 프롬프트 엔지니어링을 통해 AI에게 사용 가능한 함수 목록을 텍스트로 알려주고, AI가 적절한 함수를 선택하도록 유도하는 방식을 실험한다.

### 핵심 개념 설명

#### Chat Completions API

OpenAI의 Chat Completions API는 대화형 AI 모델과 상호작용하는 주요 인터페이스이다. 메시지를 보내면 AI가 응답을 생성하여 돌려준다. 각 메시지에는 `role`(역할)과 `content`(내용)가 포함된다.

역할(role)의 종류:
- `system`: AI의 행동 방식을 지시하는 시스템 메시지
- `user`: 사용자가 보내는 메시지
- `assistant`: AI가 생성한 응답 메시지
- `tool`: 도구 실행 결과를 전달하는 메시지 (이후 섹션에서 학습)

#### 프롬프트 엔지니어링을 통한 함수 선택

이 단계에서는 아직 OpenAI의 공식 Function Calling 기능을 사용하지 않는다. 대신 프롬프트(텍스트 지시문)를 통해 AI에게 "이런 함수들이 있으니 적절한 것을 골라 달라"고 요청하는 방식을 사용한다. 이것은 AI 에이전트의 가장 원시적인 형태이다.

### 코드 분석

#### 프롬프트 기반 함수 선택

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

**코드 분석:**

1. **`openai.OpenAI()`**: OpenAI 클라이언트를 생성한다. 이때 환경변수 `OPENAI_API_KEY`를 자동으로 읽어온다.
2. **`PROMPT`**: 여러 줄로 된 프롬프트를 정의한다. AI에게 사용 가능한 함수 목록(`get_weather`, `get_currency`, `get_news`)을 알려주고, 질문에 맞는 함수를 선택하도록 요청한다.
3. **`client.chat.completions.create()`**: API 호출을 수행한다. `model` 파라미터로 사용할 모델을 지정하고, `messages` 파라미터로 대화 내용을 전달한다.

#### 응답에서 메시지 추출

```python
message = response.choices[0].message.content
message
```

**출력 결과:**
```
"get_weather('Greece')"
```

**핵심 이해:**

- `response.choices[0]`: API 응답에서 첫 번째 선택지를 가져온다 (일반적으로 하나만 반환됨)
- `.message.content`: 해당 선택지의 메시지 내용(텍스트)을 추출한다
- AI는 프롬프트의 지시에 따라 `get_weather('Greece')`라는 함수 호출 형태의 텍스트를 반환했다

#### 이 접근법의 한계

이 방식은 동작하지만 여러 가지 문제점이 있다:

- AI의 응답이 항상 일관된 형식이라는 보장이 없다 (예: `"get_weather('Greece')"` vs `"I would call get_weather with Greece"`)
- 반환된 텍스트를 파싱하여 실제 함수를 호출하려면 추가적인 문자열 처리가 필요하다
- 함수의 파라미터 타입이나 필수 여부를 AI에게 정확히 전달하기 어렵다

이러한 한계를 해결하기 위해 OpenAI는 공식 **Function Calling** 기능을 제공하며, 이는 섹션 2.4 이후에서 학습한다.

### 실습 포인트

1. 프롬프트를 수정하여 다른 질문을 해보자 (예: "What is the currency of Japan?")
2. AI가 `get_currency('Japan')`을 반환하는지 확인한다
3. 프롬프트에서 "Please say nothing else" 부분을 제거하면 AI 응답이 어떻게 달라지는지 관찰한다
4. 다른 모델(예: `gpt-4o`)을 사용했을 때 응답의 차이를 비교해본다

---

## 2.3 메모리 추가 (Adding Memory)

### 주제 및 목표

이전 대화 내용을 기억하는 AI 챗봇을 만든다. `messages` 배열을 사용하여 대화 히스토리를 관리하고, 사용자와 AI 간의 연속적인 대화가 가능하도록 구현한다.

### 핵심 개념 설명

#### 대화 메모리의 원리

LLM(Large Language Model)은 기본적으로 **상태를 유지하지 않는(stateless)** 시스템이다. 매 API 호출은 독립적이며, 이전 대화 내용을 자동으로 기억하지 않는다. 따라서 대화의 문맥을 유지하려면, **이전의 모든 메시지를 매번 API에 함께 전송**해야 한다.

이것이 바로 `messages` 배열의 역할이다. 사용자가 새 메시지를 보낼 때마다:

1. 사용자의 메시지를 `messages` 배열에 추가한다
2. 전체 `messages` 배열을 API에 전송한다
3. AI의 응답을 다시 `messages` 배열에 추가한다
4. 다음 대화에서는 이 모든 히스토리가 함께 전송된다

```
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico!"},
    {"role": "user", "content": "What is my name?"},
    # AI는 위의 히스토리를 보고 "Your name is Nico."라고 답할 수 있다
]
```

### 코드 분석

#### 초기 설정

```python
import openai

client = openai.OpenAI()
messages = []
```

`messages`를 빈 배열로 초기화한다. 이 배열이 전체 대화 히스토리를 저장하는 **메모리** 역할을 한다.

#### AI 호출 함수 정의

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

**코드 분석:**

1. `messages` 전체를 API에 전달하여 대화의 문맥을 유지한다
2. AI의 응답(`message.content`)을 추출한다
3. 응답을 `{"role": "assistant", "content": message}` 형태로 `messages` 배열에 추가한다 -- 이것이 **메모리에 저장**하는 행위이다
4. 응답을 화면에 출력한다

#### 대화 루프

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

**코드 분석:**

1. `while True`: 무한 루프로 대화를 계속 이어간다
2. `input()`: 사용자로부터 메시지를 입력받는다
3. `"quit"` 또는 `"q"`를 입력하면 루프를 종료한다
4. 사용자 메시지를 `{"role": "user", "content": message}` 형태로 `messages`에 추가한 뒤 `call_ai()`를 호출한다

#### 실행 결과 (메모리 동작 확인)

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

**이 결과가 보여주는 것:**

- AI는 사용자의 이름("Nico")을 기억한다
- AI는 사용자의 출신지("Korea")를 기억한다
- AI는 첫 번째 질문이 무엇이었는지까지 기억한다
- 이 모든 것은 `messages` 배열에 이전 대화가 누적되어 있기 때문에 가능하다

### messages 배열의 구조 시각화

```
API 호출 #1:
messages = [
    {"role": "user", "content": "My name is Nico"}
]

API 호출 #2:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"}
]

API 호출 #3:
messages = [
    {"role": "user", "content": "My name is Nico"},
    {"role": "assistant", "content": "Nice to meet you, Nico! ..."},
    {"role": "user", "content": "What is my name?"},
    {"role": "assistant", "content": "Your name is Nico."},
    {"role": "user", "content": "I'm from Korea"}
]
```

매 호출마다 전체 히스토리가 전송되므로, 대화가 길어질수록 API 비용(토큰 사용량)이 증가한다는 점을 유의해야 한다.

### 실습 포인트

1. 긴 대화를 이어가며 AI가 얼마나 잘 기억하는지 테스트한다
2. `messages` 배열을 직접 출력해서 내부 구조를 확인한다
3. 대화 중간에 `messages = []`로 초기화하면 AI가 이전 대화를 잊어버리는 것을 확인한다
4. `system` 역할의 메시지를 추가하여 AI의 성격을 바꿔본다 (예: `{"role": "system", "content": "You are a pirate. Speak like a pirate."}`)

---

## 2.4 도구 추가 (Adding Tools)

### 주제 및 목표

OpenAI의 공식 **Tools** 기능을 사용하여 AI에게 사용 가능한 함수들을 구조적으로 알려준다. JSON Schema를 사용하여 함수의 이름, 설명, 파라미터를 정의하고, AI가 도구 호출을 결정하면 `finish_reason`이 `tool_calls`로 변경되는 것을 관찰한다.

### 핵심 개념 설명

#### 프롬프트 기반 vs Tools 기반 비교

섹션 2.2에서는 프롬프트 텍스트로 함수 목록을 알려주는 방식을 사용했다. 이는 불안정하고 파싱이 어려웠다. OpenAI의 **Tools** 기능은 이를 구조화된 JSON Schema로 대체한다:

| 구분 | 프롬프트 기반 (2.2) | Tools 기반 (2.4) |
|------|-------------------|-----------------|
| 함수 정의 방식 | 자연어 텍스트 | JSON Schema |
| 응답 형식 | 자유 텍스트 | 구조화된 tool_calls 객체 |
| 파라미터 정의 | 불명확 | 타입, 필수 여부 등 명확 |
| 파싱 난이도 | 높음 | 낮음 (SDK가 처리) |

#### FUNCTION_MAP 패턴

AI가 함수 이름을 반환하면, 해당 이름으로 실제 Python 함수를 찾아 실행해야 한다. 이를 위해 **함수 이름(문자열)을 함수 객체에 매핑하는 딕셔너리**를 사용한다:

```python
FUNCTION_MAP = {"get_weather": get_weather}
```

이 패턴은 AI가 반환한 문자열 `"get_weather"`를 통해 `get_weather` 함수를 동적으로 호출할 수 있게 해준다.

### 코드 분석

#### 함수 정의와 매핑

```python
def get_weather(city):
    return "33 degrees celcius."


FUNCTION_MAP = {"get_weather": get_weather}
```

- `get_weather`: 실제 날씨 정보를 반환하는 함수이다. 현재는 하드코딩된 값을 반환하지만, 실제 프로덕션에서는 날씨 API를 호출할 것이다.
- `FUNCTION_MAP`: 문자열 키를 함수 객체에 매핑한다. 나중에 AI가 `"get_weather"`라고 응답하면, `FUNCTION_MAP["get_weather"]`로 실제 함수를 찾는다.

#### Tools 스키마 정의 (핵심)

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

**스키마 구조 상세 분석:**

1. **`"type": "function"`**: 이 도구의 타입이 함수임을 명시한다.

2. **`"function"` 객체:**
   - `"name"`: 함수 이름. AI가 이 이름으로 함수 호출을 요청한다.
   - `"description"`: 함수에 대한 설명. AI가 어떤 상황에서 이 함수를 사용할지 판단하는 데 활용된다. **좋은 description을 작성하는 것이 매우 중요하다.**
   - `"parameters"`: JSON Schema 형식으로 파라미터를 정의한다.
     - `"type": "object"`: 파라미터가 객체 형태임을 나타낸다.
     - `"properties"`: 각 파라미터의 이름, 타입, 설명을 정의한다.
     - `"required"`: 필수 파라미터 목록을 배열로 지정한다.

3. **`TOOLS`는 배열이다.** 여러 개의 도구를 정의하여 AI에게 제공할 수 있다.

#### API 호출에 Tools 전달

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

**변경점:** `tools=TOOLS` 파라미터가 추가되었다. 이제 AI는 대화 중에 적절한 함수를 호출할 수 있다.

#### 실행 결과 분석

일반 대화의 경우:
```
User: my name is nico
AI: Nice to meet you, Nico! How can I assist you today?
```
- `finish_reason`이 `'stop'`이다 -- AI가 일반 텍스트로 응답했다.
- `tool_calls`가 `None`이다.

도구가 필요한 질문의 경우:
```
User: what is the weather in Spain
AI: None
```
- `finish_reason`이 `'tool_calls'`로 바뀌었다!
- `tool_calls` 배열에 호출할 함수 정보가 담겨 있다:
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
- `message.content`가 `None`이다 -- AI가 텍스트 대신 도구 호출을 선택했기 때문이다.

**이것이 핵심이다:** AI가 "날씨를 알려달라"는 요청에 직접 답하지 않고, "get_weather 함수를 city='Spain' 인자로 호출해달라"고 요청한 것이다. 하지만 아직 이 요청을 처리하는 코드가 없으므로 `AI: None`이 출력된다.

### 실습 포인트

1. `TOOLS`에 `get_news` 함수 스키마를 추가해본다
2. 도구가 필요한 질문과 필요 없는 질문을 번갈아 보내며 `finish_reason`의 변화를 관찰한다
3. `description`을 변경하여 AI의 도구 선택 행동이 어떻게 달라지는지 실험한다
4. `response` 객체를 상세히 출력하여 `tool_calls` 구조를 직접 확인한다

---

## 2.5 함수 호출 추가 (Adding Function Calling)

### 주제 및 목표

AI가 요청한 도구 호출을 실제로 처리하는 `process_ai_response` 함수를 구현한다. AI 응답에 `tool_calls`가 포함되어 있으면 해당 함수를 실행하고, 그렇지 않으면 일반 텍스트 응답으로 처리하는 분기 로직을 작성한다.

### 핵심 개념 설명

#### Function Calling의 전체 흐름

```
사용자 질문 → AI 판단 → tool_calls 응답 → 함수 실행 → 결과를 messages에 추가 → AI 재호출 → 최종 응답
```

이 섹션에서는 "함수 실행"과 "결과를 messages에 추가"까지를 구현한다. "AI 재호출"은 다음 섹션(2.6)에서 완성한다.

#### tool_calls 응답의 구조

AI가 도구를 사용하기로 결정하면, 응답의 `message` 객체에 다음 정보가 포함된다:

```python
message.tool_calls = [
    ChatCompletionMessageToolCall(
        id='call_yTID1R7DPur7eJMWlobM8tgu',    # 고유 ID
        function=Function(
            name='get_weather',                   # 호출할 함수 이름
            arguments='{"city":"Spain"}'           # JSON 문자열 형태의 인자
        ),
        type='function'
    )
]
```

주목할 점:
- `id`: 각 도구 호출에 고유한 ID가 부여된다. 나중에 결과를 전달할 때 이 ID로 어떤 호출의 결과인지 매칭한다.
- `arguments`: **JSON 문자열**이다. Python 딕셔너리가 아니므로 `json.loads()`로 파싱해야 한다.
- `tool_calls`는 **배열**이다. AI가 한 번에 여러 함수를 호출할 수도 있다.

### 코드 분석

#### import 추가

```python
import openai, json
```

`json` 모듈을 추가했다. AI가 반환하는 함수 인자가 JSON 문자열이므로 이를 파싱하기 위해 필요하다.

#### process_ai_response 함수 (핵심 로직)

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

**단계별 상세 분석:**

**1단계: 분기 판단**
```python
if message.tool_calls > 0:
```
AI 응답에 `tool_calls`가 있는지 확인한다. 있으면 도구 호출 로직을, 없으면 일반 텍스트 응답 로직을 실행한다.

> 참고: 이 조건문은 이후 2.6 섹션에서 `if message.tool_calls:`로 수정된다. Python에서 `None > 0`은 `TypeError`를 발생시킬 수 있기 때문이다. truthy/falsy 검사가 더 안전하다.

**2단계: assistant 메시지를 히스토리에 추가**
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

AI의 도구 호출 응답을 `messages` 배열에 기록한다. **이것이 매우 중요하다.** OpenAI API는 다음 호출에서 이 히스토리를 보고, 도구 호출이 이루어졌음을 이해한다. `content`가 `None`일 수 있으므로 `message.content or ""`로 빈 문자열을 기본값으로 설정한다.

**3단계: 각 도구 호출을 순회하며 실행**
```python
for tool_call in message.tool_calls:
    function_name = tool_call.function.name      # "get_weather"
    arguments = tool_call.function.arguments      # '{"city":"Spain"}'

    print(f"Calling function: {function_name} with {arguments}")

    try:
        arguments = json.loads(arguments)         # {"city": "Spain"}
    except json.JSONDecodeError:
        arguments = {}

    function_to_run = FUNCTION_MAP.get(function_name)  # get_weather 함수 객체

    result = function_to_run(**arguments)          # get_weather(city="Spain")
```

- `json.loads()`: JSON 문자열을 Python 딕셔너리로 변환한다
- `try/except`: JSON 파싱 실패에 대비한 방어적 코딩이다
- `FUNCTION_MAP.get()`: 문자열 이름으로 실제 함수를 찾는다
- `**arguments`: 딕셔너리를 키워드 인자로 언패킹한다. `{"city": "Spain"}`이 `city="Spain"`이 된다

**4단계: 도구 실행 결과를 히스토리에 추가**
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

- `"role": "tool"`: 이 메시지가 도구 실행 결과임을 나타낸다
- `"tool_call_id"`: 어떤 도구 호출에 대한 결과인지 매칭하는 ID이다. 이것이 빠지면 API 오류가 발생한다
- `"content"`: 함수의 실행 결과 (여기서는 "33 degrees celcius.")

#### ** 연산자(언패킹) 이해를 위한 보조 코드

커밋에는 `**` 연산자를 이해하기 위한 실험 코드가 포함되어 있다:

```python
a = '{"city": "Spain"}'

b = json.loads(a)    # b = {"city": "Spain"}

**b                   # 언패킹: city="Spain"

get_weather(city='Spain')
```

이 코드는 JSON 문자열이 어떻게 함수 호출 인자로 변환되는지를 보여주는 학습용 예제이다.

#### 간소화된 call_ai 함수

```python
def call_ai():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    process_ai_response(response.choices[0].message)
```

이전의 `call_ai`에서 응답 처리 로직을 `process_ai_response`로 분리하여 코드를 깔끔하게 정리했다.

### 실습 포인트

1. `process_ai_response` 함수 내에서 각 단계마다 `print(messages)`를 추가하여 히스토리 변화를 추적한다
2. `FUNCTION_MAP`에 없는 함수를 AI가 호출하려 할 때 어떤 에러가 발생하는지 확인한다 (힌트: `None(**arguments)`는 `TypeError`를 발생시킨다)
3. `get_currency` 함수와 스키마를 추가하여 여러 도구를 사용할 수 있도록 확장한다

---

## 2.6 도구 실행 결과 전달 (Tool Results)

### 주제 및 목표

도구 실행 결과를 AI에게 다시 전달하여, AI가 그 결과를 바탕으로 사용자에게 자연스러운 최종 응답을 생성하도록 한다. 이를 통해 **에이전트 루프**를 완성한다.

### 핵심 개념 설명

#### 에이전트 루프 (Agent Loop)

완전한 AI 에이전트는 다음과 같은 루프를 형성한다:

```
사용자 질문
    ↓
AI가 판단 ───→ 일반 응답이면 → 사용자에게 텍스트 출력 (루프 종료)
    ↓
도구 호출이 필요하면
    ↓
함수 실행 → 결과를 messages에 추가
    ↓
AI 재호출 (call_ai를 다시 호출)
    ↓
AI가 판단 ───→ 일반 응답이면 → 사용자에게 텍스트 출력 (루프 종료)
    ↓
또 도구 호출이 필요하면 → 다시 함수 실행... (반복)
```

이 루프의 핵심은 **도구 실행 후 AI를 다시 호출한다**는 점이다. AI는 도구 실행 결과를 받아서 사용자에게 적절한 형태로 가공하여 전달한다.

#### 이전 섹션(2.5)의 문제점

2.5에서는 도구를 실행하고 결과를 `messages`에 추가하는 것까지만 구현했다. 하지만 그 결과를 AI에게 다시 전달하지 않았기 때문에, AI가 최종 답변을 생성하지 못했다. 이번 섹션에서 이를 해결한다.

### 코드 분석

#### 수정된 process_ai_response (최종 버전)

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

**핵심 변경점 3가지:**

**1. 조건문 수정: `message.tool_calls > 0` -> `message.tool_calls`**

```python
# 이전 (2.5)
if message.tool_calls > 0:

# 이후 (2.6)
if message.tool_calls:
```

Python에서 `None > 0`은 `TypeError`를 발생시킬 수 있다. `tool_calls`가 `None`이면 falsy이고, 리스트가 있으면 truthy이므로 이 방식이 더 안전하다.

**2. 디버그 출력 추가**

```python
print(f"Ran {function_name} with args {arguments} for a result of {result}")
```

함수 실행 결과를 콘솔에 출력하여 디버깅을 돕는다.

**3. 도구 실행 후 AI 재호출 (가장 중요한 변경)**

```python
        # 모든 도구 실행이 끝난 후
        call_ai()
```

이 한 줄이 에이전트 루프를 완성한다. 모든 도구의 결과를 `messages`에 추가한 뒤, `call_ai()`를 다시 호출한다. 그러면 AI는 도구 실행 결과가 포함된 전체 히스토리를 받아서 최종 응답을 생성한다.

**호출 흐름을 추적해보면:**

```
call_ai()                          # 1차 호출
  → AI: tool_calls 반환
  → process_ai_response()
    → 도구 실행, 결과를 messages에 추가
    → call_ai()                    # 2차 호출 (재귀)
      → AI: 일반 텍스트 응답 반환
      → process_ai_response()
        → else 분기: 텍스트 출력
```

이것은 **상호 재귀 호출** 패턴이다: `call_ai` -> `process_ai_response` -> `call_ai` -> `process_ai_response` -> ...

#### 실행 결과 (완전한 에이전트 동작)

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

**동작 과정 분석:**

1. "What is the weather in Spain" 질문에 AI가 `get_weather(city="Spain")` 호출을 결정했다
2. 함수가 실행되어 "33 degrees celcius."를 반환했다
3. 이 결과가 `messages`에 추가되고 AI가 다시 호출되었다
4. AI는 원시 데이터 "33 degrees celcius."를 "The weather in Spain is 33 degrees Celsius. If you need more specific weather details..."라는 자연스러운 문장으로 변환하여 응답했다

#### 최종 messages 배열 확인

```python
messages
```

출력:
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

이 배열은 에이전트의 전체 동작 과정을 보여준다:
1. 일반 대화 (user -> assistant)
2. 도구 호출 요청 (assistant with tool_calls)
3. 도구 실행 결과 (tool)
4. 결과를 바탕으로 한 최종 응답 (assistant)

### 실습 포인트

1. `get_news`와 `get_currency` 함수를 추가하고, 복합 질문("What is the weather and news in Korea?")을 던져 여러 도구가 동시에 호출되는지 확인한다
2. 재귀 호출이 무한 루프에 빠지지 않는 이유를 생각해본다 (힌트: AI가 도구 결과를 받으면 일반 텍스트로 응답하므로 `else` 분기로 빠진다)
3. `messages` 배열을 출력하여 각 역할(user, assistant, tool)의 메시지가 어떻게 쌓이는지 시각적으로 확인한다
4. 도구 실행 결과를 일부러 에러 메시지로 바꿔서 AI가 어떻게 반응하는지 테스트한다

---

## 챕터 핵심 정리

### 1. AI 에이전트의 본질

AI 에이전트는 **LLM + 도구 + 루프**의 조합이다. LLM이 판단하고, 도구가 실행하며, 루프가 이를 반복적으로 연결한다.

### 2. messages 배열이 곧 메모리

LLM은 상태를 유지하지 않는다. `messages` 배열에 이전 대화를 누적하여 매번 전송하는 것이 "메모리"의 실체이다.

### 3. Tools 스키마는 AI와의 계약

JSON Schema로 도구를 정의하면, AI는 구조화된 형태로 도구 호출을 요청한다. `description`의 품질이 AI의 판단 정확도에 직접적인 영향을 미친다.

### 4. Function Calling의 핵심 흐름

```
사용자 질문 → AI 판단 → tool_calls 반환 → 함수 실행 → 결과를 messages에 추가 → AI 재호출 → 최종 응답
```

### 5. 에이전트 루프의 종료 조건

AI가 더 이상 도구 호출 없이 일반 텍스트로 응답하면 루프가 종료된다. 이것은 AI 스스로가 "더 이상 도구가 필요 없다"고 판단한 것이다.

### 6. 주요 데이터 흐름 요약

```python
# 사용자 메시지
{"role": "user", "content": "What is the weather in Spain?"}

# AI의 도구 호출 요청
{"role": "assistant", "content": "", "tool_calls": [...]}

# 도구 실행 결과
{"role": "tool", "tool_call_id": "call_xxx", "name": "get_weather", "content": "33 degrees"}

# AI의 최종 응답
{"role": "assistant", "content": "The weather in Spain is 33 degrees Celsius."}
```

---

## 실습 과제

### 과제 1: 다중 도구 에이전트 (기초)

`get_weather`, `get_news`, `get_currency` 세 가지 함수를 모두 구현하고 `TOOLS` 스키마를 정의하여, AI가 질문에 따라 적절한 도구를 선택하도록 만들어라.

**요구사항:**
- 각 함수는 하드코딩된 결과를 반환해도 된다
- `FUNCTION_MAP`에 세 함수를 모두 등록한다
- "What is the news in Japan?", "What is the currency of Brazil?" 같은 질문에 올바르게 응답하는지 확인한다

### 과제 2: System Prompt 추가 (기초)

`messages` 배열의 맨 앞에 `system` 역할의 메시지를 추가하여, AI의 성격을 변경해보아라.

**예시:**
```python
messages = [
    {"role": "system", "content": "You are a helpful weather assistant. Always respond in Korean."}
]
```

### 과제 3: 에러 처리 강화 (중급)

다음 상황에 대한 에러 처리를 추가하라:
- `FUNCTION_MAP`에 존재하지 않는 함수를 AI가 호출한 경우
- 함수 실행 중 예외가 발생한 경우
- `json.loads()`가 실패한 경우 (이미 기본적인 처리는 있지만, 에러 메시지를 AI에게 전달하도록 개선)

**힌트:** 에러가 발생하면 `"role": "tool"` 메시지의 `content`에 에러 메시지를 넣어서 AI에게 알려줄 수 있다.

### 과제 4: 대화 히스토리 관리 (중급)

대화가 길어지면 토큰 사용량이 증가한다. 다음 전략 중 하나를 구현하라:
- `messages` 배열의 길이가 일정 수를 초과하면 오래된 메시지를 제거한다 (단, `system` 메시지는 유지)
- 전체 토큰 수를 추정하여 제한을 설정한다

### 과제 5: 실제 API 연동 (심화)

`get_weather` 함수를 실제 날씨 API(예: OpenWeatherMap API)와 연동하여, 실시간 날씨 데이터를 반환하도록 구현하라. 함수의 반환값이 달라지면 AI의 응답도 그에 맞게 변하는 것을 확인한다.
