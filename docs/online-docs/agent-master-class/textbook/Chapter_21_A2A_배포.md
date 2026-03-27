# Chapter 21: AI 에이전트 배포 (Deployment)

## 챕터 개요

이번 챕터에서는 OpenAI Agents SDK로 구축한 AI 에이전트를 **실제 프로덕션 환경에 배포**하는 전체 과정을 학습합니다. 단순히 로컬에서 에이전트를 실행하는 것을 넘어, FastAPI 웹 프레임워크를 사용하여 REST API로 감싸고, OpenAI의 Conversations API를 활용한 대화 상태 관리, 동기/스트리밍 응답 처리, 그리고 최종적으로 Railway 클라우드 플랫폼에 배포하는 과정까지 다룹니다.

### 학습 목표

- FastAPI를 사용하여 AI 에이전트를 REST API로 래핑하는 방법 이해
- OpenAI Conversations API를 활용한 대화 상태(context) 관리 방법 습득
- 동기(Sync) 응답과 스트리밍(Streaming) 응답의 차이점 및 구현 방법 학습
- Railway 플랫폼을 이용한 클라우드 배포 실습

### 사용 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.13 | 프로그래밍 언어 |
| FastAPI | 0.118.3 | 웹 프레임워크 |
| OpenAI Agents SDK | 0.3.3 | AI 에이전트 프레임워크 |
| Uvicorn | 0.37.0 | ASGI 서버 |
| python-dotenv | 1.1.1 | 환경변수 관리 |
| Railway | - | 클라우드 배포 플랫폼 |

---

## 21.0 Introduction - 프로젝트 초기 설정

### 주제 및 목표

배포용 프로젝트의 기본 골격을 생성합니다. `uv`(Python 패키지 매니저)를 사용하여 새로운 Python 프로젝트를 초기화하고, 필요한 의존성(dependencies)을 설정합니다.

### 핵심 개념 설명

#### 프로젝트 구조

이번 챕터에서는 기존 마스터클래스 프로젝트와 별도로 `deployment/`라는 독립적인 디렉토리를 생성합니다. 이는 배포 가능한 독립 애플리케이션으로 설계하기 위함입니다.

```
deployment/
├── .python-version    # Python 버전 지정 (3.13)
├── README.md          # 프로젝트 설명
├── main.py            # 메인 애플리케이션 파일
└── pyproject.toml     # 프로젝트 메타데이터 및 의존성
```

#### pyproject.toml - 의존성 관리

`pyproject.toml`은 현대 Python 프로젝트의 표준 설정 파일입니다. 이 파일에서 프로젝트의 메타데이터와 의존성을 선언합니다.

```toml
[project]
name = "deployment"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi==0.118.3",
    "openai-agents==0.3.3",
    "python-dotenv==1.1.1",
    "uvicorn==0.37.0",
]
```

각 의존성의 역할:

- **`fastapi`**: 고성능 Python 웹 프레임워크. 자동 API 문서 생성, 타입 검증 등을 제공합니다.
- **`openai-agents`**: OpenAI의 공식 에이전트 SDK. Agent, Runner 등 에이전트 실행에 필요한 핵심 클래스를 제공합니다.
- **`python-dotenv`**: `.env` 파일에서 환경변수를 로드합니다. API 키 등 민감한 정보를 코드에서 분리하기 위해 사용합니다.
- **`uvicorn`**: ASGI 서버. FastAPI 애플리케이션을 실제로 HTTP 요청을 받을 수 있게 해주는 서버입니다.

#### 초기 main.py

```python
def main():
    print("Hello from deployment!")


if __name__ == "__main__":
    main()
```

이 시점에서 `main.py`는 아직 단순한 스켈레톤(skeleton) 코드입니다. 다음 섹션부터 본격적으로 FastAPI 애플리케이션으로 변환합니다.

### 실습 포인트

1. `uv init deployment` 명령으로 새 프로젝트를 생성해 보세요.
2. `uv add fastapi openai-agents python-dotenv uvicorn` 으로 의존성을 추가해 보세요.
3. `.python-version` 파일에 `3.13`이 설정되어 있는지 확인하세요.

---

## 21.1 Conversations API - 대화 관리 API 구축

### 주제 및 목표

FastAPI를 사용하여 AI 에이전트와의 대화(conversation)를 관리할 수 있는 REST API를 구축합니다. OpenAI의 **Conversations API**를 활용하여 대화 세션을 생성하고, 각 대화에 메시지를 추가할 수 있는 엔드포인트를 만듭니다.

### 핵심 개념 설명

#### OpenAI Conversations API란?

Conversations API는 OpenAI가 제공하는 대화 상태 관리 기능입니다. 기존에는 대화 이력(history)을 직접 관리해야 했지만, Conversations API를 사용하면 OpenAI 서버 측에서 대화 상태를 유지합니다.

핵심 흐름:
1. `client.conversations.create()`로 새 대화 세션을 생성하면 고유한 `conversation_id`를 받습니다.
2. 이후 에이전트 실행 시 이 `conversation_id`를 전달하면, OpenAI가 자동으로 이전 대화 맥락을 유지합니다.

이 방식의 장점:
- **서버리스 호환**: 서버에 상태를 저장할 필요가 없으므로, 서버가 재시작되어도 대화가 유지됩니다.
- **간편한 구현**: 대화 이력 배열을 직접 관리하는 복잡한 로직이 불필요합니다.
- **확장성**: 여러 서버 인스턴스에서 동일한 대화를 이어갈 수 있습니다.

#### FastAPI 애플리케이션 구조

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner
```

**주의할 점**: `load_dotenv()`가 `from agents import ...` 보다 **먼저** 호출됩니다. 이는 `agents` 모듈이 import될 때 환경변수(특히 `OPENAI_API_KEY`)를 참조하기 때문입니다. 순서가 바뀌면 API 키를 찾을 수 없어 에러가 발생합니다.

#### Agent 정의

```python
agent = Agent(
    name="Assistant",
    instructions="You help users with their questions."
)
```

에이전트는 모듈 레벨에서 한 번만 생성합니다. 요청마다 새로 생성할 필요가 없습니다. `instructions`는 에이전트의 시스템 프롬프트 역할을 합니다.

#### FastAPI 앱 및 OpenAI 클라이언트 초기화

```python
app = FastAPI()
client = AsyncOpenAI()
```

`AsyncOpenAI()`는 비동기(async) OpenAI 클라이언트입니다. FastAPI는 비동기 프레임워크이므로, 비동기 클라이언트를 사용하는 것이 성능상 적절합니다.

#### 대화 생성 엔드포인트

```python
class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }
```

이 코드의 핵심 포인트:

1. **Pydantic BaseModel**: `CreateConversationResponse`는 응답의 스키마를 정의합니다. FastAPI는 이를 자동으로 JSON으로 직렬화하고, Swagger 문서에도 반영합니다.
2. **`client.conversations.create()`**: OpenAI API를 호출하여 새 대화 세션을 생성합니다. 반환값의 `.id` 필드에 `conv_`로 시작하는 고유 ID가 담겨 있습니다.
3. **비동기 처리**: `await` 키워드를 사용하여 API 호출이 완료될 때까지 비동기적으로 대기합니다.

#### 메시지 엔드포인트 (스켈레톤)

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(conversation_id: str):
    pass
```

이 시점에서 메시지 엔드포인트는 아직 구현되지 않은 상태입니다. URL 경로에 `{conversation_id}`가 포함되어 있어, 특정 대화에 메시지를 보낼 수 있는 구조입니다.

### 코드 분석 - 전체 흐름

```
클라이언트                    FastAPI 서버               OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │                            │── conversations.create()─>│
   │                            │<── conversation 객체 ─────│
   │<── { conversation_id } ────│                          │
```

### 실습 포인트

1. `uvicorn main:app --reload` 명령으로 개발 서버를 시작해 보세요.
2. 브라우저에서 `http://127.0.0.1:8000/docs`에 접속하면 FastAPI의 자동 생성 Swagger 문서를 확인할 수 있습니다.
3. `POST /conversations` 엔드포인트를 호출하여 `conversation_id`가 정상적으로 반환되는지 확인하세요.

---

## 21.2 Sync Responses - 동기 응답 구현

### 주제 및 목표

대화에 메시지를 전송하고 에이전트의 응답을 **동기적으로** 받는 엔드포인트를 완성합니다. `Runner.run()`을 사용하여 에이전트를 실행하고, 전체 응답이 완성된 후 한 번에 클라이언트에게 전달합니다.

### 핵심 개념 설명

#### 동기 응답 vs 스트리밍 응답

| 특성 | 동기(Sync) 응답 | 스트리밍(Streaming) 응답 |
|------|-----------------|------------------------|
| 응답 방식 | 전체 응답 완성 후 한 번에 전송 | 토큰 단위로 실시간 전송 |
| 사용자 경험 | 응답까지 대기 시간 발생 | 즉시 텍스트가 나타남 |
| 구현 난이도 | 상대적으로 단순 | 이벤트 스트림 처리 필요 |
| 적합한 상황 | 백엔드 간 통신, 짧은 응답 | 사용자 대면 UI, 긴 응답 |

이번 섹션에서는 먼저 동기 응답을 구현합니다.

#### 요청/응답 모델 정의

```python
class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str
```

Pydantic `BaseModel`을 사용하여 입출력 스키마를 엄격하게 정의합니다.

- **`CreateMessageInput`**: 클라이언트가 보내는 요청 본문(body). `question` 필드에 사용자의 질문이 담깁니다.
- **`CreateMessageOutput`**: 서버가 반환하는 응답. `answer` 필드에 에이전트의 답변이 담깁니다.

FastAPI는 이 모델을 기반으로:
- 요청 본문의 JSON을 자동으로 파싱하고 타입 검증합니다.
- 잘못된 형식의 요청이 오면 422 Validation Error를 자동으로 반환합니다.

#### 메시지 처리 엔드포인트

```python
@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }
```

이 코드의 핵심 포인트:

1. **`conversation_id` 경로 매개변수**: URL에서 추출되어 함수의 인자로 전달됩니다. 이를 통해 어떤 대화에 메시지를 보낼지 식별합니다.
2. **`message_input` 본문 매개변수**: FastAPI가 요청 본문의 JSON을 자동으로 `CreateMessageInput` 객체로 변환합니다.
3. **`Runner.run()`**: 에이전트를 실행하는 핵심 메서드입니다.
   - `starting_agent`: 실행할 에이전트 객체
   - `input`: 사용자의 질문 텍스트
   - `conversation_id`: OpenAI Conversations API의 대화 ID. 이것이 **대화 맥락 유지의 핵심**입니다.
4. **`answer.final_output`**: `Runner.run()`의 반환값에서 에이전트의 최종 텍스트 출력을 추출합니다.

#### API 테스트 (api.http)

```http
POST http://127.0.0.1:8000/conversations

###

POST http://127.0.0.1:8000/conversations/conv_68ecdf11ff6081969cc4e8e9d126c015082054e6371dc260/message
Content-Type: application/json

{
    "question": "What is the first question i asked you?"
}
```

`api.http` 파일은 VS Code의 REST Client 확장 등에서 사용하는 HTTP 요청 테스트 파일입니다. `###`으로 요청을 구분하며, 각 요청을 개별적으로 실행할 수 있습니다.

위 테스트에서 "What is the first question i asked you?"라는 질문은 **대화 맥락 유지**를 검증하기 위한 것입니다. `conversation_id`를 통해 이전 대화 내용이 유지되므로, 에이전트가 이전에 받은 질문을 기억하고 답변할 수 있습니다.

### 코드 분석 - 전체 흐름

```
클라이언트                    FastAPI 서버               OpenAI API
   │                            │                          │
   │── POST /conversations ────>│                          │
   │<── { conversation_id } ────│                          │
   │                            │                          │
   │── POST /conversations/     │                          │
   │   {id}/message ───────────>│                          │
   │   { question: "..." }      │── Runner.run() ─────────>│
   │                            │   (conversation_id 포함)   │
   │                            │<── 완성된 응답 ────────────│
   │<── { answer: "..." } ──────│                          │
```

### 실습 포인트

1. 대화를 생성한 후, 반환된 `conversation_id`를 사용하여 여러 번 메시지를 보내 보세요.
2. "My name is [이름]"이라고 말한 후, "What is my name?"이라고 질문하여 대화 맥락이 유지되는지 확인하세요.
3. 서로 다른 `conversation_id`를 사용하면 대화가 독립적으로 유지되는지 검증해 보세요.

---

## 21.3 StreamingResponse - 스트리밍 응답 구현

### 주제 및 목표

에이전트의 응답을 **실시간 스트리밍**으로 전달하는 엔드포인트를 구현합니다. 이를 통해 사용자는 에이전트가 답변을 생성하는 과정을 실시간으로 확인할 수 있습니다. 두 가지 스트리밍 방식(텍스트만 / 전체 이벤트)을 모두 구현합니다.

### 핵심 개념 설명

#### 스트리밍 응답의 필요성

동기 응답(`Runner.run()`)은 전체 답변이 생성될 때까지 클라이언트가 기다려야 합니다. 긴 답변의 경우 수 초에서 수십 초가 걸릴 수 있어 사용자 경험이 좋지 않습니다.

스트리밍 응답(`Runner.run_streamed()`)은 답변이 생성되는 대로 토큰 단위로 클라이언트에 전송합니다. ChatGPT 웹 인터페이스에서 텍스트가 한 글자씩 나타나는 것과 동일한 원리입니다.

#### FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse
```

FastAPI의 `StreamingResponse`는 제너레이터(generator) 함수를 받아 데이터를 청크(chunk) 단위로 클라이언트에 전송합니다. HTTP 연결을 유지한 채 데이터를 점진적으로 보내는 방식입니다.

#### 방법 1: 텍스트 델타만 스트리밍

```python
@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")
```

이 코드를 단계별로 분석합니다:

**1단계 - `Runner.run_streamed()` 호출**:
```python
events = Runner.run_streamed(
    starting_agent=agent,
    input=message_input.question,
    conversation_id=conversation_id,
)
```
`Runner.run()` 대신 `Runner.run_streamed()`를 사용합니다. 이 메서드는 결과를 한 번에 반환하지 않고, 이벤트 스트림 객체를 반환합니다.

**2단계 - 이벤트 필터링**:
```python
async for event in events.stream_events():
    if (
        event.type == "raw_response_event"
        and event.data.type == "response.output_text.delta"
    ):
        yield event.data.delta
```

`stream_events()`는 다양한 종류의 이벤트를 생성합니다. 여기서는 두 가지 조건으로 필터링합니다:
- `event.type == "raw_response_event"`: OpenAI API에서 직접 전달되는 원시(raw) 이벤트
- `event.data.type == "response.output_text.delta"`: 텍스트 출력의 **변경분(delta)**에 해당하는 이벤트

`yield`는 Python의 비동기 제너레이터 문법입니다. 데이터를 하나씩 생성하여 `StreamingResponse`에 전달합니다.

**3단계 - StreamingResponse 반환**:
```python
return StreamingResponse(event_generator(), media_type="text/plain")
```
`media_type="text/plain"`으로 설정하여 순수 텍스트로 스트리밍합니다. 클라이언트는 연결을 유지하며 텍스트 청크를 순차적으로 수신합니다.

#### 방법 2: 전체 이벤트 스트리밍

```python
@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

이 엔드포인트는 텍스트 델타만이 아니라, **모든 raw_response_event**를 JSON 형식으로 스트리밍합니다.

핵심 차이점:
- 필터 조건이 `event.type == "raw_response_event"`만 남았습니다 (텍스트 델타 조건 제거).
- `event.data.to_json()`으로 이벤트 전체를 JSON 문자열로 변환합니다.
- 각 이벤트 뒤에 `\n`(줄바꿈)을 추가하여 클라이언트가 이벤트를 구분할 수 있게 합니다.

이 방식은 프론트엔드에서 더 세밀한 제어가 필요할 때 유용합니다. 예를 들어, 도구 호출 이벤트, 에이전트 전환 이벤트 등을 모두 수신하여 UI에 반영할 수 있습니다.

#### 두 방식의 비교

| 특성 | `/message-stream` | `/message-stream-all` |
|------|-------------------|----------------------|
| 전송 내용 | 텍스트 조각만 | 모든 이벤트 (JSON) |
| 데이터 형식 | 순수 텍스트 | JSON (줄바꿈 구분) |
| 데이터 양 | 적음 | 많음 |
| 적합한 용도 | 단순 채팅 UI | 고급 UI (도구 실행 표시 등) |

#### curl을 사용한 스트리밍 테스트

```bash
curl -N -X POST http://127.0.0.1:8000/conversations/{conv_id}/message-stream \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the size of the great wall of china?"}'
```

`curl`의 `-N` 플래그는 출력 버퍼링을 비활성화합니다. 이 옵션이 없으면 curl이 데이터를 버퍼에 모아두었다가 한 번에 출력하므로, 스트리밍 효과를 확인할 수 없습니다.

### 실습 포인트

1. `/message-stream` 엔드포인트를 `curl -N`으로 호출하여 텍스트가 실시간으로 출력되는 것을 확인하세요.
2. `/message-stream-all` 엔드포인트를 호출하여 어떤 종류의 이벤트들이 전달되는지 관찰하세요.
3. 동기 응답(`/message`)과 스트리밍 응답(`/message-stream`)의 체감 속도 차이를 비교해 보세요.
4. 스트리밍된 JSON 이벤트에서 `response.output_text.delta` 외에 어떤 이벤트 타입들이 있는지 분석해 보세요.

---

## 21.4 Deployment - Railway 클라우드 배포

### 주제 및 목표

완성된 AI 에이전트 API를 **Railway** 클라우드 플랫폼에 배포하여 인터넷에서 접근 가능한 실제 서비스로 만듭니다. 배포 설정 파일 작성, 환경변수 관리, 보안 설정을 다룹니다.

### 핵심 개념 설명

#### Railway란?

Railway는 개발자 친화적인 클라우드 배포 플랫폼입니다. Git 저장소를 연결하면 자동으로 빌드하고 배포해 주며, 환경변수 관리, 로그 확인, 도메인 설정 등을 쉽게 할 수 있습니다.

Railway의 장점:
- Git push 기반 자동 배포 (CI/CD)
- NIXPACKS를 사용한 자동 빌드 (Dockerfile 불필요)
- 무료 티어 제공
- 간편한 환경변수 관리

#### 헬스 체크 엔드포인트 추가

```python
@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }
```

루트 경로(`/`)에 간단한 GET 엔드포인트를 추가합니다. 이는 여러 목적으로 사용됩니다:
- **헬스 체크(Health Check)**: 서버가 정상적으로 동작하는지 확인하는 용도. Railway 등 클라우드 플랫폼이 주기적으로 이 엔드포인트를 호출하여 서비스 상태를 확인합니다.
- **빠른 확인**: 브라우저에서 배포된 URL에 접속했을 때 서비스가 동작하는지 즉시 확인할 수 있습니다.
- **동기 함수**: `async def`가 아닌 `def`로 정의했습니다. 외부 API 호출이 없으므로 비동기가 불필요합니다.

#### railway.json - 배포 설정

```json
{
    "$schema": "https://railway.app/railway.schema.json",
    "build": {
        "builder": "NIXPACKS"
    },
    "deploy": {
        "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT"
    }
}
```

각 설정 항목 분석:

- **`$schema`**: JSON 스키마 URL. IDE에서 자동완성과 유효성 검사를 지원합니다.
- **`build.builder: "NIXPACKS"`**: NIXPACKS는 소스 코드를 분석하여 자동으로 빌드 환경을 구성하는 도구입니다. `pyproject.toml`을 감지하면 Python 환경을 자동으로 설정하고, 의존성을 설치합니다. Dockerfile을 직접 작성할 필요가 없습니다.
- **`deploy.startCommand`**: 배포 후 애플리케이션을 실행하는 명령어입니다.
  - `uvicorn main:app`: `main.py` 파일의 `app` 객체를 ASGI 서버로 실행
  - `--host 0.0.0.0`: 모든 네트워크 인터페이스에서 접속 허용 (컨테이너 환경에서 필수)
  - `--port $PORT`: Railway가 동적으로 할당하는 포트를 사용 (`$PORT` 환경변수)

#### .gitignore - 보안 및 정리

```
.env
.venv
__pycache__
```

배포 시 Git 저장소에 포함되지 않아야 하는 파일들:
- **`.env`**: API 키 등 민감한 환경변수가 담긴 파일. **절대로** Git에 커밋하면 안 됩니다.
- **`.venv`**: Python 가상환경 디렉토리. 배포 환경에서 별도로 생성됩니다.
- **`__pycache__`**: Python 바이트코드 캐시. 불필요한 파일입니다.

#### 배포 후 URL 변경

```http
POST https://my-agent-deployment-production.up.railway.app/conversations
```

로컬 개발 시 `http://127.0.0.1:8000`이었던 URL이 Railway 배포 후 `https://my-agent-deployment-production.up.railway.app`으로 변경됩니다. Railway는 자동으로 HTTPS를 제공하며, 프로젝트 이름 기반의 하위 도메인을 할당합니다.

### 배포 절차 요약

```
1. Railway 계정 생성 및 프로젝트 생성
2. GitHub 저장소 연결
3. 환경변수 설정 (OPENAI_API_KEY)
4. railway.json 설정에 따라 자동 빌드 및 배포
5. 할당된 URL로 API 테스트
```

### 최종 프로젝트 구조

```
deployment/
├── .gitignore         # Git 제외 파일 목록
├── .python-version    # Python 3.13
├── .env               # 환경변수 (Git 제외)
├── README.md          # 프로젝트 설명
├── api.http           # API 테스트 파일
├── main.py            # 메인 애플리케이션 (FastAPI + Agent)
├── pyproject.toml     # 의존성 관리
├── railway.json       # Railway 배포 설정
└── uv.lock            # 의존성 잠금 파일
```

### 최종 main.py 전체 코드

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from agents import Agent, Runner


agent = Agent(
    name="Assistant",
    instructions="You help users with their questions.",
)

app = FastAPI()
client = AsyncOpenAI()


class CreateConversationResponse(BaseModel):
    conversation_id: str


@app.get("/")
def hello_world():
    return {
        "message": "hello world",
    }


@app.post("/conversations")
async def create_conversation() -> CreateConversationResponse:
    conversation = await client.conversations.create()
    return {
        "conversation_id": conversation.id,
    }


class CreateMessageInput(BaseModel):
    question: str


class CreateMessageOutput(BaseModel):
    answer: str


@app.post("/conversations/{conversation_id}/message")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    answer = await Runner.run(
        starting_agent=agent,
        input=message_input.question,
        conversation_id=conversation_id,
    )
    return {
        "answer": answer.final_output,
    }


@app.post("/conversations/{conversation_id}/message-stream")
async def create_message(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if (
                event.type == "raw_response_event"
                and event.data.type == "response.output_text.delta"
            ):
                yield event.data.delta

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/conversations/{conversation_id}/message-stream-all")
async def create_message_all(
    conversation_id: str, message_input: CreateMessageInput
) -> CreateMessageOutput:
    async def event_generator():
        events = Runner.run_streamed(
            starting_agent=agent,
            input=message_input.question,
            conversation_id=conversation_id,
        )
        async for event in events.stream_events():
            if event.type == "raw_response_event":
                yield f"{event.data.to_json()}\n"

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### 실습 포인트

1. Railway(https://railway.app)에 가입하고 새 프로젝트를 생성해 보세요.
2. GitHub 저장소를 연결하고, Railway 대시보드에서 `OPENAI_API_KEY` 환경변수를 설정하세요.
3. 배포된 URL로 `POST /conversations`를 호출하여 대화를 생성하고, 메시지를 보내 보세요.
4. 배포된 서비스의 로그를 Railway 대시보드에서 확인하며, 요청 처리 과정을 관찰하세요.

---

## 챕터 핵심 정리

### 1. 아키텍처 패턴
AI 에이전트를 프로덕션에 배포할 때는 **REST API로 래핑**하는 것이 표준적인 접근법입니다. FastAPI는 비동기 지원, 자동 문서 생성, 타입 검증 등을 제공하여 이 작업에 최적화되어 있습니다.

### 2. 대화 상태 관리
OpenAI의 **Conversations API**를 사용하면 서버 측에서 대화 이력을 관리할 필요가 없습니다. `conversation_id`만 전달하면 OpenAI가 자동으로 이전 맥락을 유지합니다. 이는 서버리스 환경이나 수평 확장(horizontal scaling) 시 큰 이점이 됩니다.

### 3. 동기 vs 스트리밍
- **`Runner.run()`**: 전체 응답을 기다려야 하지만 구현이 단순합니다. 백엔드 간 통신에 적합합니다.
- **`Runner.run_streamed()`**: 실시간 토큰 스트리밍을 지원합니다. 사용자 대면 UI에 필수적입니다.

### 4. 이벤트 필터링
스트리밍 시 `stream_events()`가 생성하는 다양한 이벤트 중, 용도에 맞는 이벤트만 필터링하는 것이 중요합니다:
- 텍스트만 필요: `raw_response_event` + `response.output_text.delta`
- 전체 이벤트 필요: `raw_response_event` 전체

### 5. 클라우드 배포
Railway + NIXPACKS 조합을 사용하면 Dockerfile 없이 `pyproject.toml`만으로 자동 빌드 및 배포가 가능합니다. 핵심 설정은 `railway.json`의 `startCommand`와 환경변수(`OPENAI_API_KEY`)입니다.

### 6. 보안
`.env` 파일은 반드시 `.gitignore`에 포함하여 Git 저장소에 커밋되지 않도록 합니다. 배포 환경에서는 플랫폼의 환경변수 관리 기능을 사용합니다.

---

## 실습 과제

### 과제 1: 기본 배포 (난이도: ★★☆☆☆)
챕터의 코드를 따라 입력하여 로컬에서 동작하는 AI 에이전트 API를 구축하세요.

**요구사항:**
- `POST /conversations` - 대화 생성
- `POST /conversations/{id}/message` - 동기 메시지 전송
- `POST /conversations/{id}/message-stream` - 스트리밍 메시지 전송
- 대화 맥락이 올바르게 유지되는지 검증

### 과제 2: 에이전트 커스터마이징 (난이도: ★★★☆☆)
기본 Assistant 에이전트를 특정 도메인에 특화된 에이전트로 변경하세요.

**예시:**
```python
agent = Agent(
    name="Korean Teacher",
    instructions="""당신은 한국어 교사입니다.
    사용자의 한국어 학습을 도와주세요.
    문법 오류가 있으면 교정해 주고, 자연스러운 표현을 제안하세요.""",
)
```

### 과제 3: 도구(Tool) 추가 (난이도: ★★★★☆)
에이전트에 도구(function tool)를 추가하여 외부 데이터를 활용할 수 있게 확장하세요.

**힌트:**
```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 반환합니다."""
    # 실제 날씨 API 연동 또는 더미 데이터 반환
    return f"{city}의 현재 날씨: 맑음, 22도"

agent = Agent(
    name="Weather Assistant",
    instructions="You help users check the weather.",
    tools=[get_weather],
)
```

스트리밍 엔드포인트(`/message-stream-all`)에서 도구 호출 이벤트가 어떻게 전달되는지 관찰하세요.

### 과제 4: Railway 배포 (난이도: ★★★★☆)
실제로 Railway에 배포하고, 배포된 URL로 API를 호출해 보세요.

**체크리스트:**
- [ ] Railway 프로젝트 생성
- [ ] GitHub 저장소 연결
- [ ] `OPENAI_API_KEY` 환경변수 설정
- [ ] 배포 완료 후 `/` 엔드포인트로 헬스 체크
- [ ] 대화 생성 및 메시지 전송 테스트
- [ ] 스트리밍 응답 테스트 (`curl -N` 사용)

### 과제 5: 프론트엔드 연동 (난이도: ★★★★★)
배포된 API에 연동되는 간단한 채팅 프론트엔드를 구현하세요.

**힌트:**
- `fetch()` API의 스트리밍 읽기:
```javascript
const response = await fetch(url, { method: 'POST', body: JSON.stringify({ question }), headers: { 'Content-Type': 'application/json' } });
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // UI에 텍스트 추가
}
```

---

> **다음 챕터 예고**: 다음 챕터에서는 AI 에이전트의 응답을 테스트하고 품질을 검증하는 방법을 학습합니다.
