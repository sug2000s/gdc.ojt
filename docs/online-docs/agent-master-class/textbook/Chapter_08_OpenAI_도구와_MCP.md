# Chapter 8: OpenAI 도구(Tools)와 MCP를 활용한 ChatGPT 클론 만들기

---

## 1. 챕터 개요

이번 챕터에서는 OpenAI의 Agents SDK가 제공하는 다양한 **내장 도구(Built-in Tools)**와 **MCP(Model Context Protocol)**를 활용하여 본격적인 ChatGPT 클론 애플리케이션을 구축한다. Streamlit을 UI 프레임워크로 사용하며, 단순한 텍스트 대화부터 시작하여 웹 검색, 파일 검색, 이미지 입력/생성, 코드 실행, 그리고 외부 MCP 서버 연동까지 단계적으로 기능을 확장해 나간다.

### 학습 목표

- Streamlit과 OpenAI Agents SDK를 연동하여 채팅 UI를 구현할 수 있다
- `SQLiteSession`을 활용한 대화 기록 영속화(persistence)를 이해한다
- `WebSearchTool`, `FileSearchTool`, `ImageGenerationTool`, `CodeInterpreterTool` 등 OpenAI 내장 도구의 사용법을 익힌다
- 멀티모달(이미지) 입력을 처리하는 방법을 학습한다
- `HostedMCPTool`과 `MCPServerStdio`를 통한 외부 도구 연동 패턴을 이해한다
- 스트리밍 이벤트를 활용한 실시간 UI 업데이트 기법을 습득한다

### 사용 기술 스택

| 기술 | 역할 |
|------|------|
| **Streamlit** | 웹 기반 채팅 UI 프레임워크 |
| **OpenAI Agents SDK** | Agent, Runner, 도구 관리 |
| **SQLiteSession** | 대화 기록 로컬 저장소 |
| **OpenAI API** | 파일 업로드, Vector Store 관리 |
| **MCP (Model Context Protocol)** | 외부 도구 서버 연동 프로토콜 |

### 프로젝트 구조

```
chatgpt-clone/
├── main.py                      # 메인 애플리케이션
├── chat-gpt-clone-memory.db     # SQLite 대화 기록 DB
├── facts.txt                    # File Search용 샘플 데이터
└── international.png            # 멀티모달 테스트용 이미지
```

---

## 2. 섹션별 상세 설명

---

### 8.0 Chat UI - Streamlit 채팅 인터페이스 구축

#### 주제 및 목표

Streamlit을 사용하여 기본적인 채팅 UI를 구축하고, OpenAI Agents SDK의 `Agent`와 `Runner`를 연동하여 스트리밍 응답을 실시간으로 표시하는 기초 구조를 만든다.

#### 핵심 개념 설명

**Streamlit의 `session_state`**는 웹 앱에서 상태를 유지하기 위한 핵심 메커니즘이다. Streamlit은 사용자 상호작용이 있을 때마다 전체 스크립트를 재실행하는 구조이므로, Agent나 Session 같은 객체를 매번 새로 생성하지 않도록 `session_state`에 저장해야 한다.

**`SQLiteSession`**은 OpenAI Agents SDK가 제공하는 대화 기록 영속화 도구로, SQLite 데이터베이스에 대화 내용을 자동으로 저장하고 불러온다. 이를 통해 페이지를 새로고침해도 이전 대화가 유지된다.

**`Runner.run_streamed()`**는 에이전트를 스트리밍 모드로 실행하여, 응답이 생성되는 동안 실시간으로 이벤트를 수신할 수 있게 한다.

#### 코드 분석

```python
import dotenv
dotenv.load_dotenv()

import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession

# Agent를 session_state에 저장하여 재실행 시에도 동일 인스턴스 유지
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.
        """,
    )
agent = st.session_state["agent"]

# SQLiteSession으로 대화 기록을 로컬 DB에 영속화
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",               # 세션 식별자
        "chat-gpt-clone-memory.db",   # SQLite DB 파일 경로
    )
session = st.session_state["session"]
```

위 코드에서 중요한 패턴은 `if "key" not in st.session_state` 가드이다. Streamlit은 모든 사용자 상호작용마다 `main.py` 전체를 재실행하므로, 이 가드가 없으면 Agent와 Session이 매번 새로 생성되어 이전 상태가 모두 사라진다.

```python
async def run_agent(message):
    stream = Runner.run_streamed(
        agent,
        message,
        session=session,  # 세션을 전달하여 대화 기록 자동 관리
    )

    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                with st.chat_message("ai"):
                    st.write_stream(event.data.delta)
```

`stream.stream_events()`는 비동기 이터레이터로, 응답 생성 과정에서 발생하는 다양한 이벤트를 하나씩 전달한다. `raw_response_event` 중에서 `response.output_text.delta` 타입인 이벤트가 실제 텍스트 조각(delta)을 담고 있다.

```python
# 채팅 입력 UI
prompt = st.chat_input("Write a message for your assistant")

if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    asyncio.run(run_agent(prompt))

# 사이드바: 메모리 초기화 및 디버깅
with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```

`st.chat_input()`은 Streamlit이 제공하는 채팅 전용 입력 위젯이고, `st.chat_message()`는 사용자/AI 메시지를 시각적으로 구분하여 표시하는 컨테이너이다. 사이드바에서는 세션 초기화 버튼과 현재 저장된 대화 항목을 디버깅용으로 표시한다.

#### 실습 포인트

- `streamlit run main.py`로 앱을 실행하고 대화를 주고받아 본다
- 브라우저를 새로고침한 뒤에도 사이드바에서 대화 기록이 유지되는지 확인한다
- `session.clear_session()`으로 대화를 초기화한 뒤 동작을 확인한다

---

### 8.1 Conversation History - 대화 기록 렌더링

#### 주제 및 목표

페이지 새로고침 시 이전 대화 기록을 화면에 복원하고, 스트리밍 응답의 표시 방식을 개선하여 텍스트가 점진적으로 나타나도록 한다.

#### 핵심 개념 설명

이전 섹션에서는 대화 기록이 DB에 저장되지만 페이지를 새로고침하면 화면에는 표시되지 않았다. **`paint_history()`** 함수를 추가하여, 앱이 로드될 때마다 SQLiteSession에서 저장된 메시지를 읽어와 화면에 다시 그려주는 기능을 구현한다.

또한 이전에는 각 delta마다 새로운 `st.write()`를 호출하여 메시지가 여러 줄로 중복 표시되는 문제가 있었다. 이를 **`st.empty()`** 플레이스홀더를 활용하여, 하나의 영역에 텍스트를 누적 업데이트하는 방식으로 개선한다.

#### 코드 분석

```python
async def paint_history():
    messages = await session.get_items()

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                if message["type"] == "message":
                    st.write(message["content"][0]["text"])

# 앱 로드 시 즉시 실행
asyncio.run(paint_history())
```

`session.get_items()`는 저장된 전체 대화 기록을 리스트로 반환한다. 각 메시지는 딕셔너리 형태로, `role` 필드가 `"user"` 또는 `"assistant"`인지에 따라 다른 구조를 가진다. 사용자 메시지는 `content`가 단순 문자열이고, AI 응답은 `content`가 리스트 형태(`[{"text": "..."}]`)이다.

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()  # 빈 플레이스홀더 생성
        response = ""
        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta        # 텍스트 누적
                    text_placeholder.write(response)    # 같은 위치에 업데이트
```

**`st.empty()`의 역할**: Streamlit에서 `st.empty()`는 나중에 내용을 채울 수 있는 빈 컨테이너를 만든다. 이 컨테이너에 `.write()`를 호출하면 이전 내용을 **대체(replace)**하므로, 스트리밍 텍스트가 한 곳에서 점진적으로 길어지는 자연스러운 효과를 얻을 수 있다.

#### 실습 포인트

- 여러 번 대화를 나눈 뒤 페이지를 새로고침하여 기록이 복원되는지 확인한다
- `st.empty()` 대신 `st.write()`를 직접 사용했을 때의 차이를 비교해 본다
- 사이드바의 `get_items()` 출력으로 메시지 딕셔너리 구조를 분석한다

---

### 8.2 Web Search Tool - 웹 검색 도구 추가

#### 주제 및 목표

에이전트에 `WebSearchTool`을 추가하여 실시간 웹 검색 기능을 부여하고, 검색 진행 상태를 UI에 실시간으로 표시하는 상태 관리 시스템을 구축한다.

#### 핵심 개념 설명

**`WebSearchTool`**은 OpenAI Agents SDK가 제공하는 내장 도구로, 에이전트가 자신의 학습 데이터에 없는 최신 정보를 웹에서 검색할 수 있게 한다. 에이전트의 `instructions`에 도구 사용 지침을 명시하여, 언제 웹 검색을 수행해야 하는지 안내하는 것이 중요하다.

**상태 컨테이너(`st.status`)**는 Streamlit이 제공하는 진행 상태 표시 위젯으로, 도구 실행 과정을 사용자에게 시각적으로 알려준다.

#### 코드 분석

```python
from agents import Agent, Runner, SQLiteSession, WebSearchTool

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data. Use this tool when the users asks about current
              or future events, when you think you don't know the answer, try
              searching for it in the web first.
        """,
        tools=[
            WebSearchTool(),  # 웹 검색 도구 등록
        ],
    )
```

에이전트의 `instructions`에 도구 사용 조건을 명시한다. "현재 또는 미래의 사건에 대해 물을 때", "답을 모를 때" 웹 검색을 먼저 시도하라는 지침이다. 이는 도구 선택의 정확도를 높이는 프롬프트 엔지니어링 기법이다.

```python
def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": (
            "🔍 Starting web search...",
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 Web search in progress...",
            "running",
        ),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)
```

`update_status()` 함수는 스트리밍 이벤트의 타입에 따라 UI의 상태 표시를 업데이트하는 **이벤트 디스패처** 역할을 한다. 웹 검색 관련 이벤트는 세 단계로 구분된다:

1. `in_progress` - 검색 시작
2. `searching` - 검색 진행 중
3. `completed` - 검색 완료

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)  # 상태 컨테이너
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(agent, message, session=session)

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)  # 상태 업데이트

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response)
```

대화 기록 복원 시에도 웹 검색 호출 기록을 표시한다:

```python
if "type" in message and message["type"] == "web_search_call":
    with st.chat_message("ai"):
        st.write("🔍 Searched the web...")
```

#### 실습 포인트

- "오늘 날씨 어때?" 같은 실시간 정보를 물어보고 웹 검색이 트리거되는지 확인한다
- 상태 컨테이너의 변화 과정을 관찰한다 (시작 -> 진행 중 -> 완료)
- 학습 데이터에 있는 일반 지식 질문을 했을 때는 웹 검색이 실행되지 않는지 확인한다

---

### 8.3 File Search Tool - 파일 검색 도구와 Vector Store

#### 주제 및 목표

`FileSearchTool`과 OpenAI의 Vector Store를 활용하여 사용자가 업로드한 파일의 내용을 검색할 수 있는 기능을 추가한다. 또한 Streamlit의 파일 업로드 기능을 통해 사용자가 직접 텍스트 파일을 업로드할 수 있도록 한다.

#### 핵심 개념 설명

**Vector Store**는 OpenAI가 호스팅하는 벡터 데이터베이스로, 텍스트 파일을 업로드하면 자동으로 임베딩(embedding)하여 의미 기반 검색이 가능하게 한다. `FileSearchTool`은 에이전트가 이 Vector Store에서 관련 정보를 검색할 수 있게 하는 도구이다.

**파일 업로드 워크플로우**는 두 단계로 구성된다:
1. `client.files.create()`로 OpenAI에 파일 업로드
2. `client.vector_stores.files.create()`로 Vector Store에 파일 연결

#### 코드 분석

```python
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool

client = OpenAI()

# Vector Store ID (OpenAI 대시보드 또는 API로 사전 생성)
VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: ...
            - File Search Tool: Use this tool when the user asks a question
              about facts related to themselves. Or when they ask questions
              about specific files.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],  # 검색 대상 Vector Store
                max_num_results=3,                    # 최대 검색 결과 수
            ),
        ],
    )
```

`FileSearchTool`은 `vector_store_ids`로 검색 대상 Vector Store를 지정하고, `max_num_results`로 반환할 최대 결과 수를 제한한다.

파일 업로드 및 Vector Store 연결 코드:

```python
# 채팅 입력에서 파일 첨부 활성화
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt"],
)

if prompt:
    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    # 1단계: OpenAI에 파일 업로드
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching file...")

                    # 2단계: Vector Store에 파일 연결
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded", state="complete")
```

이 프로젝트에서 사용하는 샘플 데이터 파일(`facts.txt`)은 가상의 투자 포트폴리오와 지출 내역을 담고 있다. 이 파일을 업로드하면 에이전트가 "내 Apple 주식은 몇 주?" 같은 개인 정보 관련 질문에 답할 수 있게 된다.

`$` 기호가 포함된 응답에서 Streamlit의 LaTeX 렌더링 문제를 방지하기 위해 `replace("$", "\$")`를 적용한 점도 주목할 만하다:

```python
st.write(message["content"][0]["text"].replace("$", "\$"))
```

#### 실습 포인트

- `facts.txt`를 업로드한 뒤 "내 포트폴리오 총 가치가 얼마야?" 같은 질문을 해 본다
- OpenAI 대시보드에서 Vector Store를 직접 생성하고 ID를 교체해 본다
- 파일 검색과 웹 검색이 각각 언제 트리거되는지 비교 관찰한다

---

### 8.4 Multi Modal Agent - 멀티모달 이미지 입력

#### 주제 및 목표

에이전트가 이미지를 입력으로 받아 분석할 수 있도록 멀티모달 기능을 추가한다. 사용자가 이미지를 업로드하면 Base64로 인코딩하여 세션에 저장하고, 에이전트가 이를 이해할 수 있도록 한다.

#### 핵심 개념 설명

**멀티모달(Multi-Modal)**이란 텍스트뿐 아니라 이미지, 오디오 등 여러 형태의 입력을 처리할 수 있는 능력을 말한다. OpenAI의 GPT-4 계열 모델은 이미지를 입력으로 받아 내용을 분석하고 설명할 수 있다.

이미지를 API에 전달하기 위해 **Base64 인코딩**을 사용한다. 이미지 바이트 데이터를 Base64 문자열로 변환한 뒤, `data:image/png;base64,...` 형태의 Data URI로 만들어 전달한다.

#### 코드 분석

```python
import base64

# 채팅 입력에서 이미지 파일도 허용
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=[
        "txt",
        "jpg",
        "jpeg",
        "png",
    ],
)
```

이미지 업로드 처리:

```python
elif file.type.startswith("image/"):
    with st.status("⏳ Uploading image...") as status:
        file_bytes = file.getvalue()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{file.type};base64,{base64_data}"

        # 이미지를 세션에 사용자 메시지로 저장
        asyncio.run(
            session.add_items(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": data_uri,
                            }
                        ],
                    }
                ]
            )
        )
        status.update(label="✅ Image uploaded", state="complete")
    with st.chat_message("human"):
        st.image(data_uri)
```

핵심은 `session.add_items()`를 통해 이미지를 대화 히스토리에 직접 추가하는 것이다. OpenAI API가 요구하는 형식인 `input_image` 타입과 `image_url` 필드를 사용한다. `detail: "auto"`는 모델이 이미지 해상도를 자동으로 결정하게 한다.

대화 기록 복원 시 이미지도 표시하도록 `paint_history()`를 수정:

```python
if message["role"] == "user":
    content = message["content"]
    if isinstance(content, str):
        st.write(content)           # 텍스트 메시지
    elif isinstance(content, list):
        for part in content:
            if "image_url" in part:
                st.image(part["image_url"])  # 이미지 메시지
```

사용자 메시지의 `content`가 문자열일 수도 있고(일반 텍스트), 리스트일 수도 있다(멀티모달). `isinstance()` 체크로 두 경우를 모두 처리한다.

#### 실습 포인트

- 차트나 그래프 이미지를 업로드하고 "이 이미지에 뭐가 보여?" 라고 질문한다
- 이미지를 업로드한 뒤 텍스트로 후속 질문을 해 보고, 에이전트가 이미지 맥락을 기억하는지 확인한다
- Base64 인코딩된 Data URI의 구조를 분석해 본다

---

### 8.5 Image Generation Tool - 이미지 생성 도구

#### 주제 및 목표

`ImageGenerationTool`을 추가하여 에이전트가 사용자의 요청에 따라 이미지를 생성할 수 있게 한다. 이미지 생성 과정의 중간 결과(partial image)를 실시간으로 표시하는 기법도 구현한다.

#### 핵심 개념 설명

**`ImageGenerationTool`**은 OpenAI의 이미지 생성 API(DALL-E)를 에이전트가 도구로 호출할 수 있게 래핑한 것이다. `partial_images` 설정을 통해 생성 중간에 저해상도 미리보기 이미지를 받아볼 수 있어, 사용자에게 생성 진행 상황을 시각적으로 전달할 수 있다.

#### 코드 분석

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
)

# 에이전트 도구 목록에 ImageGenerationTool 추가
ImageGenerationTool(
    tool_config={
        "type": "image_generation",
        "quality": "high",           # 고품질 이미지 생성
        "output_format": "jpeg",     # 출력 형식
        "partial_images": 1,         # 중간 미리보기 이미지 수
    }
),
```

`tool_config`의 주요 옵션:
- `quality`: `"high"` 또는 `"standard"`. 고품질은 더 정교하지만 생성 시간이 길다
- `output_format`: `"jpeg"` 또는 `"png"`
- `partial_images`: 생성 중 전달받을 중간 이미지 수. 1 이상으로 설정하면 프로그레시브 렌더링 효과를 얻을 수 있다

스트리밍에서 이미지 이벤트 처리:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        image_placeholder = st.empty()  # 이미지용 플레이스홀더 추가
        response = ""

        # ... 스트리밍 루프 ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # 이미지 생성 중간 결과 표시
                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)

                elif event.data.type == "response.completed":
                    image_placeholder.empty()
                    text_placeholder.empty()
```

중간 이미지(`partial_image`)는 Base64로 인코딩되어 전달되므로, `base64.b64decode()`로 디코딩한 뒤 `st.image()`로 표시한다.

대화 기록 복원 시 생성된 이미지도 표시:

```python
elif message_type == "image_generation_call":
    image = base64.b64decode(message["result"])
    with st.chat_message("ai"):
        st.image(image)
```

이미지 생성 관련 상태 메시지도 추가:

```python
"response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
"response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
```

#### 실습 포인트

- "고양이가 우주에서 피자를 먹는 그림을 그려줘" 같은 요청으로 이미지 생성을 테스트한다
- `partial_images` 값을 0과 1로 바꿔보면서 중간 미리보기 효과를 비교한다
- 생성된 이미지가 대화 기록에 저장되어 새로고침 후에도 표시되는지 확인한다

---

### 8.6 Code Interpreter Tool - 코드 실행 도구

#### 주제 및 목표

`CodeInterpreterTool`을 추가하여 에이전트가 Python 코드를 작성하고 실행하여 계산, 데이터 분석, 차트 생성 등을 수행할 수 있게 한다.

#### 핵심 개념 설명

**`CodeInterpreterTool`**은 OpenAI가 호스팅하는 샌드박스 환경에서 Python 코드를 실행할 수 있게 하는 도구이다. 에이전트가 코드를 작성하면 안전한 컨테이너에서 실행되고, 결과가 반환된다. 수학 계산, 데이터 분석, 시각화 등에 유용하다.

`container` 설정의 `"type": "auto"`는 OpenAI가 자동으로 적절한 실행 환경을 선택하게 한다.

#### 코드 분석

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool, CodeInterpreterTool,
)

CodeInterpreterTool(
    tool_config={
        "type": "code_interpreter",
        "container": {
            "type": "auto",      # 자동 컨테이너 선택
        },
    }
),
```

코드 실행 과정의 스트리밍 처리:

```python
async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        code_placeholder = st.empty()    # 코드 표시용 플레이스홀더
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""
        code_response = ""

        # 플레이스홀더를 session_state에 저장 (다음 실행 시 정리용)
        st.session_state["code_placeholder"] = code_placeholder
        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        # ... 스트리밍 루프 ...
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # 코드 작성 과정을 실시간으로 표시
                if event.data.type == "response.code_interpreter_call_code.delta":
                    code_response += event.data.delta
                    code_placeholder.code(code_response)  # 코드 블록으로 표시
```

`st.code()`는 구문 강조(syntax highlighting)가 적용된 코드 블록을 표시한다. 코드가 작성되는 과정을 실시간으로 볼 수 있어 사용자 경험이 향상된다.

다음 메시지 실행 시 이전 플레이스홀더를 정리하는 코드:

```python
if prompt:
    # 이전 플레이스홀더 정리
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()
```

이 정리 코드가 없으면 이전 응답의 스트리밍 플레이스홀더가 화면에 남아 `paint_history()`가 복원한 메시지와 중복 표시되는 문제가 발생한다.

코드 실행 관련 상태 메시지:

```python
"response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
"response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
"response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
```

#### 실습 포인트

- "피보나치 수열의 처음 20개 항을 계산해줘" 같은 코드 실행 요청을 해 본다
- "사인 함수 그래프를 그려줘" 같은 시각화 요청도 시도한다
- 코드가 실시간으로 작성되는 과정을 관찰한다

---

### 8.7 Hosted MCP Tool - 호스팅 MCP 도구 연동

#### 주제 및 목표

**HostedMCPTool**을 사용하여 외부 호스팅 MCP 서버(Context7)에 연결하고, 소프트웨어 프로젝트의 문서를 검색하는 기능을 추가한다.

#### 핵심 개념 설명

**MCP(Model Context Protocol)**는 AI 모델이 외부 도구 및 데이터 소스와 상호작용하기 위한 개방형 프로토콜이다. MCP를 통해 에이전트는 자체 내장 도구 외에도 제3자가 제공하는 다양한 기능을 활용할 수 있다.

**HostedMCPTool**은 인터넷에 공개된 MCP 서버에 HTTP로 연결하는 방식이다. 서버 URL만 알면 바로 사용할 수 있어 설정이 간단하다.

**Context7**은 소프트웨어 프로젝트의 최신 문서를 제공하는 MCP 서버로, 에이전트가 특정 라이브러리나 프레임워크의 공식 문서를 검색할 수 있게 한다.

#### 코드 분석

```python
from agents import (
    Agent, Runner, SQLiteSession,
    WebSearchTool, FileSearchTool, ImageGenerationTool,
    CodeInterpreterTool, HostedMCPTool,
)

HostedMCPTool(
    tool_config={
        "server_url": "https://mcp.context7.com/mcp",  # MCP 서버 URL
        "type": "mcp",
        "server_label": "Context7",                      # 표시용 레이블
        "server_description": "Use this to get the docs from software projects.",
        "require_approval": "never",                     # 자동 승인 (사용자 확인 불필요)
    }
),
```

`tool_config`의 주요 필드:
- `server_url`: MCP 서버의 엔드포인트 URL
- `server_label`: UI에 표시할 서버 이름
- `server_description`: 에이전트가 이 도구를 언제 사용할지 판단하는 데 쓰이는 설명
- `require_approval`: `"never"`로 설정하면 사용자 승인 없이 자동으로 도구를 호출한다

MCP 관련 대화 기록 복원:

```python
elif message_type == "mcp_list_tools":
    with st.chat_message("ai"):
        st.write(f"Listed {message['server_label']}'s tools")
elif message_type == "mcp_call":
    with st.chat_message("ai"):
        st.write(
            f"Called {message['server_label']}'s {message['name']} "
            f"with args {message['arguments']}"
        )
```

MCP 호출은 두 단계로 이루어진다:
1. **`mcp_list_tools`**: 서버에서 사용 가능한 도구 목록을 조회
2. **`mcp_call`**: 특정 도구를 인자와 함께 실제 호출

MCP 관련 상태 메시지:

```python
"response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
"response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
"response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
"response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
"response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
"response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
```

#### 실습 포인트

- "Streamlit의 st.chat_input 사용법을 알려줘" 같은 질문으로 Context7 MCP를 테스트한다
- MCP 호출 과정에서 `mcp_list_tools`와 `mcp_call`이 순서대로 발생하는지 관찰한다
- `require_approval`을 `"always"`로 바꿔보고 동작 차이를 확인한다

---

### 8.8 Local MCP Server - 로컬 MCP 서버 연동

#### 주제 및 목표

**`MCPServerStdio`**를 사용하여 로컬에서 실행되는 MCP 서버(Yahoo Finance)에 연결한다. 이를 통해 호스팅 MCP와 로컬 MCP의 차이를 이해하고, 에이전트 생성 구조를 비동기 컨텍스트 매니저 패턴으로 리팩토링한다.

#### 핵심 개념 설명

**`MCPServerStdio`**는 로컬 프로세스로 MCP 서버를 실행하고, 표준 입출력(stdin/stdout)을 통해 통신하는 방식이다. `uvx`(Python 패키지 실행기)를 사용하여 MCP 서버 패키지를 직접 실행한다.

**Hosted MCP vs Local MCP의 차이**:
| 특성 | Hosted MCP | Local MCP |
|------|-----------|-----------|
| 실행 위치 | 원격 서버 | 로컬 머신 |
| 연결 방식 | HTTP | stdin/stdout |
| 설정 방식 | URL만 지정 | 실행 명령어 지정 |
| 생명주기 | 항상 가용 | 프로세스 시작/종료 필요 |

로컬 MCP 서버는 `async with` 문(비동기 컨텍스트 매니저)으로 생명주기를 관리해야 한다. 이 때문에 에이전트 생성 위치가 `session_state` 초기화에서 `run_agent()` 함수 내부로 이동한다.

#### 코드 분석

```python
from agents.mcp.server import MCPServerStdio

async def run_agent(message):
    # 로컬 MCP 서버 정의
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",                    # 실행할 명령어
            "args": ["mcp-yahoo-finance"],       # 패키지 이름
        },
        cache_tools_list=True,  # 도구 목록 캐싱으로 성능 최적화
    )

    # 비동기 컨텍스트 매니저로 서버 생명주기 관리
    async with yfinance_server:

        # Agent를 컨텍스트 내부에서 생성 (MCP 서버가 활성 상태여야 하므로)
        agent = Agent(
            mcp_servers=[
                yfinance_server,       # 로컬 MCP 서버 연결
            ],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.
        ...
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID],
                    max_num_results=3,
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        # 이제 agent를 사용하여 스트리밍 실행
        with st.chat_message("ai"):
            # ... 기존 스트리밍 로직과 동일 ...
```

**구조적 변화의 핵심**: 이전 섹션까지는 Agent를 `st.session_state`에 한 번만 생성하여 재사용했다. 그러나 로컬 MCP 서버는 `async with` 블록 안에서만 유효하므로, Agent도 해당 블록 안에서 매번 새로 생성해야 한다. 이는 성능상 약간의 오버헤드가 있지만, 로컬 MCP 서버의 안정적인 생명주기 관리를 위해 필요한 트레이드오프이다.

`cache_tools_list=True`는 MCP 서버의 도구 목록을 캐싱하여, 매번 도구 목록을 조회하지 않아도 되게 한다. 도구 목록이 자주 변경되지 않는 서버에서 유용하다.

#### 실습 포인트

- "Apple 주식의 현재 가격을 알려줘" 같은 금융 관련 질문으로 Yahoo Finance MCP를 테스트한다
- `uvx mcp-yahoo-finance`를 터미널에서 직접 실행해보고 MCP 서버의 동작을 관찰한다
- `async with` 블록 밖에서 Agent를 생성하면 어떤 에러가 발생하는지 확인한다

---

### 8.9 Conclusions - 두 번째 로컬 MCP 서버 추가

#### 주제 및 목표

두 번째 로컬 MCP 서버(시간대 서버)를 추가하여, 여러 MCP 서버를 동시에 사용하는 패턴을 학습한다.

#### 핵심 개념 설명

Python의 `async with` 문은 콤마(`,`)로 구분하여 **여러 컨텍스트 매니저를 동시에** 관리할 수 있다. 이를 활용하면 여러 로컬 MCP 서버를 동시에 실행하고, 모두 같은 Agent에 연결할 수 있다.

#### 코드 분석

```python
async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-yahoo-finance"],
        },
        cache_tools_list=True,
    )

    # 두 번째 로컬 MCP 서버: 시간대 정보 제공
    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    # 두 서버를 동시에 컨텍스트 매니저로 관리
    async with yfinance_server, timezone_server:

        agent = Agent(
            mcp_servers=[
                yfinance_server,
                timezone_server,      # 두 번째 MCP 서버 추가
            ],
            name="ChatGPT Clone",
            # ... 나머지 설정 동일 ...
        )
```

`mcp-server-time`은 `--local-timezone` 인자로 기본 시간대를 지정할 수 있다. 에이전트는 이 서버를 통해 특정 시간대의 현재 시각이나 시간대 변환 등의 작업을 수행할 수 있다.

`async with yfinance_server, timezone_server:` 구문은 두 서버를 동시에 시작하고, 블록이 끝나면 둘 다 깔끔하게 종료한다. 하나의 에러가 발생해도 나머지 서버도 정상적으로 정리(cleanup)된다.

#### 실습 포인트

- "뉴욕의 현재 시간은?" 같은 질문으로 시간대 MCP를 테스트한다
- 한 번의 대화에서 여러 MCP 서버를 활용하는 복합 질문을 해 본다 (예: "Apple 주가와 현재 뉴욕 시간을 함께 알려줘")
- 새로운 MCP 서버 패키지를 찾아 추가해 본다

---

## 3. 챕터 핵심 정리

### 아키텍처 패턴

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │chat_input│  │chat_msg  │  │  sidebar (debug)   │   │
│  └────┬─────┘  └──────────┘  └────────────────────┘   │
│       │                                                 │
│       v                                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              run_agent()                         │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │  async with MCP_Server_1, MCP_Server_2  │    │   │
│  │  │  ┌───────────────────────────────────┐  │    │   │
│  │  │  │           Agent                   │  │    │   │
│  │  │  │  ┌──────────┐ ┌──────────────┐   │  │    │   │
│  │  │  │  │WebSearch │ │ FileSearch   │   │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │ImageGen  │ │CodeInterpreter│  │  │    │   │
│  │  │  │  ├──────────┤ ├──────────────┤   │  │    │   │
│  │  │  │  │HostedMCP │ │ Local MCP x2 │  │  │    │   │
│  │  │  │  └──────────┘ └──────────────┘   │  │    │   │
│  │  │  └───────────────────────────────────┘  │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         v                               │
│              ┌─────────────────┐                       │
│              │  SQLiteSession   │                       │
│              │  (대화 기록 저장) │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 도구별 요약

| 도구 | 용도 | 핵심 이벤트 타입 |
|------|------|-----------------|
| **WebSearchTool** | 실시간 웹 정보 검색 | `response.web_search_call.*` |
| **FileSearchTool** | Vector Store 기반 파일 내용 검색 | `response.file_search_call.*` |
| **ImageGenerationTool** | DALL-E 이미지 생성 | `response.image_generation_call.*` |
| **CodeInterpreterTool** | Python 코드 실행 | `response.code_interpreter_call*` |
| **HostedMCPTool** | 원격 MCP 서버 연동 | `response.mcp_call.*`, `response.mcp_list_tools.*` |
| **MCPServerStdio** | 로컬 MCP 서버 연동 | `response.mcp_call.*`, `response.mcp_list_tools.*` |

### 핵심 개념 정리

1. **`st.session_state`**: Streamlit의 상태 관리 메커니즘. 스크립트 재실행 간에 데이터를 유지한다.

2. **`st.empty()`**: 나중에 내용을 대체할 수 있는 플레이스홀더. 스트리밍 텍스트 표시에 필수적이다.

3. **`st.status()`**: 작업 진행 상태를 시각적으로 표시하는 위젯. `running`/`complete` 상태를 지원한다.

4. **`SQLiteSession`**: 대화 기록을 SQLite DB에 자동 저장/복원. `session` 파라미터로 Runner에 전달한다.

5. **스트리밍 이벤트 패턴**: `raw_response_event` > `event.data.type`으로 이벤트 종류를 판별하고, 각 도구별 이벤트에 맞는 UI 업데이트를 수행한다.

6. **`async with` 컨텍스트 매니저**: 로컬 MCP 서버의 생명주기(시작/종료)를 안전하게 관리한다. 여러 서버를 콤마로 연결하여 동시에 관리할 수 있다.

7. **Vector Store**: OpenAI가 호스팅하는 벡터 DB로, 파일을 업로드하면 자동으로 임베딩하여 의미 기반 검색을 가능하게 한다.

---

## 4. 실습 과제

### 과제 1: 기본 - 도구 사용 로깅 시스템

`update_status()` 함수를 확장하여, 모든 도구 호출을 시간과 함께 로그 파일에 기록하는 시스템을 만들어라. 사이드바에서 로그를 확인할 수 있도록 구현하라.

**힌트**:
- Python의 `logging` 모듈을 활용한다
- `st.sidebar`에 로그 내용을 표시한다
- 각 도구 호출의 시작 시간, 종료 시간, 소요 시간을 기록한다

### 과제 2: 중급 - 새로운 MCP 서버 추가

`mcp-server-fetch`(웹페이지 내용 가져오기) 또는 다른 MCP 서버 패키지를 찾아 프로젝트에 추가하라. 로컬 MCP 서버로 연결하고, 에이전트가 해당 도구를 적절히 사용할 수 있도록 `instructions`를 수정하라.

**힌트**:
- `uvx`로 실행 가능한 MCP 서버 패키지를 PyPI에서 검색한다
- `MCPServerStdio`의 `params`에 적절한 명령어와 인자를 설정한다
- `async with` 문에 새 서버를 추가하고, `mcp_servers` 리스트에도 포함한다

### 과제 3: 중급 - 다중 대화 세션 관리

현재는 하나의 대화 세션만 지원한다. 사이드바에서 여러 대화 세션을 생성하고 전환할 수 있는 기능을 구현하라.

**힌트**:
- `SQLiteSession`의 첫 번째 인자(세션 ID)를 동적으로 변경한다
- `st.sidebar.selectbox()`로 세션 목록을 표시한다
- 새 세션 생성 시 고유한 ID를 부여한다

### 과제 4: 고급 - 커스텀 도구 통합

`@function_tool` 데코레이터를 사용하여 커스텀 Python 함수를 도구로 만들고, 기존 내장 도구와 함께 에이전트에 등록하라. 예를 들어, 로컬 파일 시스템을 탐색하거나 간단한 계산을 수행하는 도구를 만들어 볼 수 있다.

**힌트**:
- 이전 챕터에서 학습한 `@function_tool` 패턴을 활용한다
- 도구의 docstring이 에이전트의 도구 선택에 영향을 미친다
- `instructions`에 새 도구의 사용 조건을 명시한다

### 과제 5: 고급 - 오디오 입력 멀티모달 확장

현재 이미지 입력만 지원하는 멀티모달 기능을 오디오 파일도 지원하도록 확장하라. OpenAI의 Whisper API를 활용하여 음성을 텍스트로 변환한 뒤 에이전트에 전달하는 방식으로 구현하라.

**힌트**:
- `st.chat_input`의 `file_type`에 오디오 형식을 추가한다
- `client.audio.transcriptions.create()`로 음성을 텍스트로 변환한다
- 변환된 텍스트를 에이전트에 일반 메시지로 전달한다

---

## 부록: 최종 완성 코드 (main.py)

아래는 Chapter 8의 모든 기능이 통합된 최종 `main.py`이다:

```python
import dotenv

dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    HostedMCPTool,
)
from agents.mcp.server import MCPServerStdio

client = OpenAI()

VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"


if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )
session = st.session_state["session"]


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)
            elif message_type == "code_interpreter_call":
                with st.chat_message("ai"):
                    st.code(message["code"])
            elif message_type == "mcp_list_tools":
                with st.chat_message("ai"):
                    st.write(f"Listed {message['server_label']}'s tools")
            elif message_type == "mcp_call":
                with st.chat_message("ai"):
                    st.write(
                        f"Called {message['server_label']}'s {message['name']} "
                        f"with args {message['arguments']}"
                    )


asyncio.run(paint_history())


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
        "response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
        "response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
        "response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
        "response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
        "response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
        "response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
        "response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
        "response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
        "response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):
    yfinance_server = MCPServerStdio(
        params={"command": "uvx", "args": ["mcp-yahoo-finance"]},
        cache_tools_list=True,
    )

    timezone_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=America/New_York"],
        }
    )

    async with yfinance_server, timezone_server:
        agent = Agent(
            mcp_servers=[yfinance_server, timezone_server],
            name="ChatGPT Clone",
            instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't
              in your training data.
            - File Search Tool: Use this tool when the user asks a question about
              facts related to themselves.
            - Code Interpreter Tool: Use this tool when you need to write and run
              code to answer the user's question.
        """,
            tools=[
                WebSearchTool(),
                FileSearchTool(
                    vector_store_ids=[VECTOR_STORE_ID], max_num_results=3
                ),
                ImageGenerationTool(
                    tool_config={
                        "type": "image_generation",
                        "quality": "high",
                        "output_format": "jpeg",
                        "partial_images": 1,
                    }
                ),
                CodeInterpreterTool(
                    tool_config={
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                    }
                ),
                HostedMCPTool(
                    tool_config={
                        "server_url": "https://mcp.context7.com/mcp",
                        "type": "mcp",
                        "server_label": "Context7",
                        "server_description": "Use this to get the docs from software projects.",
                        "require_approval": "never",
                    }
                ),
            ],
        )

        with st.chat_message("ai"):
            status_container = st.status("⏳", expanded=False)
            code_placeholder = st.empty()
            image_placeholder = st.empty()
            text_placeholder = st.empty()
            response = ""
            code_response = ""

            st.session_state["code_placeholder"] = code_placeholder
            st.session_state["image_placeholder"] = image_placeholder
            st.session_state["text_placeholder"] = text_placeholder

            stream = Runner.run_streamed(agent, message, session=session)

            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    update_status(status_container, event.data.type)

                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                    if event.data.type == "response.code_interpreter_call_code.delta":
                        code_response += event.data.delta
                        code_placeholder.code(code_response)

                    elif event.data.type == "response.image_generation_call.partial_image":
                        image = base64.b64decode(event.data.partial_image_b64)
                        image_placeholder.image(image)


prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
)

if prompt:
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()), purpose="user_data"
                    )
                    status.update(label="⏳ Attaching file...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID, file_id=uploaded_file.id
                    )
                    status.update(label="✅ File uploaded", state="complete")
        elif file.type.startswith("image/"):
            with st.status("⏳ Uploading image...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")
            with st.chat_message("human"):
                st.image(data_uri)

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
```
