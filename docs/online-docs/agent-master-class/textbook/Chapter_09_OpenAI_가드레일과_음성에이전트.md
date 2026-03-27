# Chapter 9: OpenAI Agents SDK - 가드레일, 핸드오프, 음성 에이전트

---

## 챕터 개요

이 챕터에서는 OpenAI Agents SDK(`openai-agents`)를 활용하여 실전 수준의 **고객 지원 에이전트 시스템**을 단계적으로 구축한다. 단순한 챗봇을 넘어, 컨텍스트 관리, 동적 지시문, 입출력 가드레일, 에이전트 간 핸드오프, 라이프사이클 훅, 그리고 음성 에이전트까지 포괄하는 종합 프로젝트를 완성한다.

### 학습 목표

| 섹션 | 주제 | 핵심 키워드 |
|------|------|-------------|
| 9.0 | 프로젝트 소개 및 기본 구조 | Streamlit, SQLiteSession, Runner |
| 9.1 | 컨텍스트 관리 | RunContextWrapper, Pydantic 모델 |
| 9.2 | 동적 지시문 | Dynamic Instructions, 함수 기반 프롬프트 |
| 9.3 | 입력 가드레일 | Input Guardrail, Tripwire |
| 9.4 | 에이전트 핸드오프 | Handoff, 전문 에이전트 라우팅 |
| 9.5 | 핸드오프 UI | agent_updated_stream_event, 실시간 전환 표시 |
| 9.6 | 훅(Hooks) | AgentHooks, 도구 사용 로깅 |
| 9.7 | 출력 가드레일 | Output Guardrail, 응답 검증 |
| 9.8 | 음성 에이전트 I | AudioInput, WAV 변환 |
| 9.9 | 음성 에이전트 II | VoicePipeline, VoiceWorkflowBase, sounddevice |

### 프로젝트 구조 (최종)

```
customer-support-agent/
├── main.py                      # Streamlit 메인 애플리케이션
├── models.py                    # Pydantic 데이터 모델
├── tools.py                     # 에이전트 도구 함수들
├── output_guardrails.py         # 출력 가드레일 정의
├── workflow.py                  # 음성 에이전트 커스텀 워크플로우
├── my_agents/
│   ├── triage_agent.py          # 트리아지(분류) 에이전트
│   ├── technical_agent.py       # 기술 지원 에이전트
│   ├── billing_agent.py         # 결제 지원 에이전트
│   ├── order_agent.py           # 주문 관리 에이전트
│   └── account_agent.py         # 계정 관리 에이전트
├── pyproject.toml               # 프로젝트 의존성
└── customer-support-memory.db   # SQLite 세션 저장소
```

---

## 9.0 프로젝트 소개 및 기본 구조

### 주제 및 목표

OpenAI Agents SDK와 Streamlit을 결합하여 고객 지원 챗봇의 기본 골격을 만든다. 대화 이력을 SQLite에 저장하고 스트리밍 응답을 실시간으로 화면에 표시하는 구조를 설정한다.

### 핵심 개념 설명

**OpenAI Agents SDK**는 에이전트 기반 애플리케이션 개발을 위한 프레임워크로, `Runner`를 통해 에이전트를 실행하고, `SQLiteSession`으로 대화 이력을 영구 저장할 수 있다. Streamlit은 빠른 프로토타이핑을 위한 웹 UI 프레임워크로, `st.chat_message`와 `st.chat_input`을 활용하여 채팅 인터페이스를 쉽게 구현할 수 있다.

### 코드 분석

**프로젝트 의존성 설정 (`pyproject.toml`)**

```toml
[project]
name = "customer-support-agent"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "openai-agents[voice]>=0.2.8",
    "python-dotenv>=1.1.1",
    "streamlit>=1.48.1",
]
```

- `openai-agents[voice]`: 음성 에이전트 기능을 포함한 OpenAI Agents SDK 패키지
- `python-dotenv`: `.env` 파일에서 API 키 등 환경변수를 로드
- `streamlit`: 웹 UI 프레임워크

**메인 애플리케이션 (`main.py`)**

```python
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import streamlit as st
from agents import Runner, SQLiteSession

client = OpenAI()

# SQLite 기반 세션 관리
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]
```

**핵심 포인트:**
- `SQLiteSession`은 두 개의 인자를 받는다: 세션 이름(`"chat-history"`)과 데이터베이스 파일 경로(`"customer-support-memory.db"`)
- `st.session_state`를 사용하여 Streamlit의 리렌더링 사이에도 세션 객체가 유지되도록 한다

**대화 이력 표시 함수:**

```python
async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        st.write(message["content"][0]["text"].replace("$", "\$"))

asyncio.run(paint_history())
```

- `session.get_items()`로 저장된 모든 메시지를 가져온다
- `$` 기호를 `\$`로 이스케이프하는 이유는 Streamlit이 LaTeX 수식으로 해석하는 것을 방지하기 위함이다

**스트리밍 에이전트 실행:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))
```

- `Runner.run_streamed()`는 에이전트를 스트리밍 모드로 실행하여 토큰이 생성될 때마다 이벤트를 발생시킨다
- `raw_response_event` 타입에서 `response.output_text.delta`를 감지하여 실시간으로 텍스트를 누적 표시한다
- `st.empty()`는 나중에 내용을 업데이트할 수 있는 플레이스홀더를 생성한다

### 실습 포인트

1. `uv` 패키지 매니저를 사용하여 프로젝트를 초기화하고 의존성을 설치해 본다
2. `streamlit run main.py`로 앱을 실행하고 기본 채팅 인터페이스를 확인한다
3. SQLite DB 파일을 열어 대화 이력이 어떤 형태로 저장되는지 살펴본다

---

## 9.1 컨텍스트 관리 (Context Management)

### 주제 및 목표

에이전트 실행 시 **사용자 컨텍스트 정보**를 전달하는 방법을 학습한다. Pydantic 모델로 타입 안전한 컨텍스트를 정의하고, `RunContextWrapper`를 통해 도구 함수 내에서 이 정보에 접근하는 패턴을 익힌다.

### 핵심 개념 설명

**컨텍스트(Context)**란 에이전트가 실행되는 동안 참조할 수 있는 외부 정보를 의미한다. 예를 들어, 현재 로그인한 사용자의 ID, 이름, 구독 등급 등이 해당한다. 이 정보는 에이전트의 프롬프트나 도구 함수에서 활용된다.

`RunContextWrapper`는 제네릭 타입으로, `RunContextWrapper[UserAccountContext]`와 같이 컨텍스트 타입을 명시하면 IDE의 자동완성과 타입 검사를 지원받을 수 있다.

### 코드 분석

**컨텍스트 모델 정의 (`models.py`)**

```python
from pydantic import BaseModel

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"  # premium, enterprise
```

- Pydantic `BaseModel`을 상속하여 데이터 검증과 직렬화를 자동으로 처리한다
- `tier` 필드에 기본값 `"basic"`을 설정하여 선택적 필드로 만들었다

**도구 함수에서 컨텍스트 사용 (`main.py`)**

```python
from agents import Runner, SQLiteSession, function_tool, RunContextWrapper
from models import UserAccountContext

@function_tool
def get_user_tier(wrapper: RunContextWrapper[UserAccountContext]):
    return (
        f"The user {wrapper.context.customer_id} has a {wrapper.context.tier} account."
    )
```

- `@function_tool` 데코레이터는 일반 Python 함수를 에이전트가 호출할 수 있는 도구로 변환한다
- `wrapper.context`를 통해 실행 시 전달된 `UserAccountContext` 인스턴스에 접근한다

**컨텍스트 생성 및 전달:**

```python
user_account_ctx = UserAccountContext(
    customer_id=1,
    name="nico",
    tier="basic",
)

# Runner 실행 시 context 전달
stream = Runner.run_streamed(
    agent,
    message,
    session=session,
    context=user_account_ctx,  # 컨텍스트 주입
)
```

- `Runner.run_streamed()`의 `context` 파라미터로 컨텍스트 객체를 전달한다
- 이 컨텍스트는 에이전트의 모든 도구 함수와 지시문에서 접근 가능하다

### 실습 포인트

1. `UserAccountContext`에 `phone_number`, `preferred_language` 등 새로운 필드를 추가해 본다
2. 컨텍스트 정보를 활용하는 새로운 `@function_tool`을 만들어 본다
3. `tier` 값에 따라 다른 응답을 반환하는 도구를 구현해 본다

---

## 9.2 동적 지시문 (Dynamic Instructions)

### 주제 및 목표

에이전트의 지시문(instructions)을 **정적 문자열**이 아닌 **함수**로 정의하여, 실행 시점의 컨텍스트에 따라 동적으로 변하는 프롬프트를 생성하는 방법을 학습한다.

### 핵심 개념 설명

일반적으로 에이전트의 `instructions`는 고정된 문자열이다. 그러나 사용자의 이름, 등급, 이메일 등 런타임 정보를 프롬프트에 포함하려면 함수 형태의 동적 지시문이 필요하다. 이 함수는 `RunContextWrapper`와 `Agent` 객체를 인자로 받아 문자열을 반환한다.

### 코드 분석

**트리아지 에이전트 생성 (`my_agents/triage_agent.py`)**

```python
from agents import Agent, RunContextWrapper
from models import UserAccountContext

def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a customer support agent. You ONLY help customers with their
    questions about their User Account, Billing, Orders, or Technical Support.
    You call customers by their name.

    The customer's name is {wrapper.context.name}.
    The customer's email is {wrapper.context.email}.
    The customer's tier is {wrapper.context.tier}.

    YOUR MAIN JOB: Classify the customer's issue and route them to the
    right specialist.

    ISSUE CLASSIFICATION GUIDE:

    TECHNICAL SUPPORT - Route here for:
    - Product not working, errors, bugs
    - App crashes, loading issues, performance problems
    ...

    BILLING SUPPORT - Route here for:
    - Payment issues, failed charges, refunds
    ...

    ORDER MANAGEMENT - Route here for:
    - Order status, shipping, delivery questions
    ...

    ACCOUNT MANAGEMENT - Route here for:
    - Login problems, password resets, account access
    ...

    SPECIAL HANDLING:
    - Premium/Enterprise customers: Mention their priority status when routing
    - Multiple issues: Handle the most urgent first, note others for follow-up
    - Unclear issues: Ask 1-2 clarifying questions before routing
    """

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,  # 함수를 직접 전달
)
```

**핵심 포인트:**
- `instructions` 파라미터에 문자열 대신 **함수 참조**를 전달한다 (괄호 없이 `dynamic_triage_agent_instructions`)
- 함수 시그니처는 `(wrapper: RunContextWrapper[T], agent: Agent[T]) -> str` 형태여야 한다
- f-string을 사용하여 `wrapper.context.name`, `wrapper.context.tier` 등 컨텍스트 값을 프롬프트에 삽입한다
- 트리아지(Triage) 에이전트는 고객 문의를 분류하여 적절한 전문 에이전트로 라우팅하는 역할을 한다

**main.py의 변경사항:**

이 커밋에서 이전에 `main.py`에 있던 `get_user_tier` 도구 함수가 제거되었다. 컨텍스트 정보 접근을 도구가 아닌 동적 지시문에서 직접 처리하는 방식으로 전환한 것이다.

### 실습 포인트

1. 동적 지시문에서 `wrapper.context.tier` 값에 따라 다른 톤(격식체/비격식체)으로 응답하도록 수정해 본다
2. 현재 시간(`datetime.now()`)을 지시문에 포함하여 시간대에 따른 인사말을 추가해 본다
3. 에이전트 객체(`agent`)의 속성을 활용하는 동적 지시문을 작성해 본다

---

## 9.3 입력 가드레일 (Input Guardrails)

### 주제 및 목표

사용자 입력이 에이전트의 업무 범위를 벗어나는지 **자동으로 검사**하는 입력 가드레일을 구현한다. 별도의 "가드레일 에이전트"가 입력을 분석하고, 부적절한 요청이면 대화를 차단하는 패턴을 학습한다.

### 핵심 개념 설명

**입력 가드레일(Input Guardrail)**은 에이전트가 사용자 입력을 처리하기 전에 실행되는 검증 단계이다. 이 과정에서 별도의 작은 에이전트(가드레일 에이전트)가 입력을 분석하여 "오프 토픽(off-topic)" 여부를 판단한다. 부적절한 입력이 감지되면 **Tripwire**가 발동되어 `InputGuardrailTripwireTriggered` 예외를 발생시킨다.

이 패턴의 장점은:
- 메인 에이전트의 지시문을 복잡하게 만들 필요가 없다
- 가드레일 검사가 **비동기적으로 병렬 실행**되어 성능 저하가 최소화된다
- 검증 로직을 별도 모듈로 분리하여 재사용과 테스트가 용이하다

### 코드 분석

**가드레일 출력 모델 (`models.py`)**

```python
from pydantic import BaseModel
from typing import Optional

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"
    email: Optional[str] = None

class InputGuardRailOutput(BaseModel):
    is_off_topic: bool
    reason: str
```

- `InputGuardRailOutput`은 가드레일 에이전트의 구조화된 출력 형식이다
- `is_off_topic`: 요청이 업무 범위를 벗어나는지 여부
- `reason`: 판단 근거 (디버깅 및 로깅용)

**가드레일 에이전트 및 데코레이터 (`my_agents/triage_agent.py`)**

```python
from agents import (
    Agent, RunContextWrapper, input_guardrail,
    Runner, GuardrailFunctionOutput,
)
from models import UserAccountContext, InputGuardRailOutput

# 1. 가드레일 전용 에이전트 정의
input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="""
    Ensure the user's request specifically pertains to User Account details,
    Billing inquiries, Order information, or Technical Support issues, and
    is not off-topic. If the request is off-topic, return a reason for the
    tripwire. You can make small conversation with the user, specially at
    the beginning of the conversation, but don't help with requests that
    are not related to User Account details, Billing inquiries, Order
    information, or Technical Support issues.
    """,
    output_type=InputGuardRailOutput,  # 구조화된 출력 강제
)

# 2. 가드레일 함수 정의
@input_guardrail
async def off_topic_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
    input: str,
):
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=wrapper.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_off_topic,
    )

# 3. 트리아지 에이전트에 가드레일 연결
triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[
        off_topic_guardrail,
    ],
)
```

**동작 순서:**
1. 사용자가 메시지를 보낸다
2. `off_topic_guardrail` 함수가 실행된다
3. 내부에서 `input_guardrail_agent`가 메시지를 분석한다
4. `InputGuardRailOutput` 형태로 결과를 반환한다
5. `is_off_topic`이 `True`이면 `tripwire_triggered=True`로 설정된다
6. Tripwire가 발동되면 `InputGuardrailTripwireTriggered` 예외가 발생한다

**예외 처리 (`main.py`)**

```python
from agents import Runner, SQLiteSession, InputGuardrailTripwireTriggered

async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                triage_agent,
                message,
                session=session,
                context=user_account_ctx,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))
        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

- `try/except`로 `InputGuardrailTripwireTriggered` 예외를 잡아서 사용자에게 적절한 메시지를 표시한다

### 실습 포인트

1. 가드레일 에이전트의 지시문을 수정하여 더 엄격한/느슨한 필터링을 적용해 본다
2. "오늘 날씨 어때?"와 같은 오프 토픽 메시지와 "비밀번호를 변경하고 싶어요"와 같은 정상 메시지로 가드레일을 테스트해 본다
3. `reason` 필드를 UI에 표시하여 왜 차단되었는지 알려주는 기능을 추가해 본다

---

## 9.4 에이전트 핸드오프 (Handoffs)

### 주제 및 목표

트리아지 에이전트가 고객 문의를 분류한 후, 적절한 **전문 에이전트**에게 대화를 넘기는(handoff) 멀티 에이전트 구조를 구현한다. 4개의 전문 에이전트(기술 지원, 결제, 주문, 계정)를 생성하고 핸드오프 메커니즘을 설정한다.

### 핵심 개념 설명

**핸드오프(Handoff)**는 하나의 에이전트가 대화 제어권을 다른 에이전트에게 넘기는 것이다. 이는 실제 콜센터에서 상담원이 전문 부서로 전화를 돌리는 것과 유사하다.

OpenAI Agents SDK에서 핸드오프는 다음 요소로 구성된다:
- `handoff()` 함수: 핸드오프 설정을 정의
- `on_handoff`: 핸드오프 발생 시 실행되는 콜백 함수
- `input_type`: 핸드오프 시 전달되는 데이터의 스키마
- `input_filter`: 핸드오프 시 이전 에이전트의 도구 호출 기록을 정리하는 필터

### 코드 분석

**핸드오프 데이터 모델 (`models.py`)**

```python
class HandoffData(BaseModel):
    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
```

이 모델은 핸드오프 시 트리아지 에이전트가 전문 에이전트에게 전달하는 메타데이터를 정의한다.

**전문 에이전트 예시 - 기술 지원 (`my_agents/technical_agent.py`)**

```python
from agents import Agent, RunContextWrapper
from models import UserAccountContext

def dynamic_technical_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a Technical Support specialist helping {wrapper.context.name}.
    Customer tier: {wrapper.context.tier}
    {"(Premium Support)" if wrapper.context.tier != "basic" else ""}

    YOUR ROLE: Solve technical issues with our products and services.

    TECHNICAL SUPPORT PROCESS:
    1. Gather specific details about the technical issue
    2. Ask for error messages, steps to reproduce, system info
    3. Provide step-by-step troubleshooting solutions
    4. Test solutions with the customer
    5. Escalate to engineering if needed

    {"PREMIUM PRIORITY: Offer direct escalation to senior engineers
    if standard solutions don't work." if wrapper.context.tier != "basic" else ""}
    """

technical_agent = Agent(
    name="Technical Support Agent",
    instructions=dynamic_technical_agent_instructions,
)
```

- 모든 전문 에이전트는 동일한 패턴을 따른다: 동적 지시문 + `Agent` 생성
- `wrapper.context.tier`에 따라 프리미엄 고객에게 추가 혜택을 안내한다

**결제 에이전트 (`my_agents/billing_agent.py`)**

```python
def dynamic_billing_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    You are a Billing Support specialist helping {wrapper.context.name}.
    Customer tier: {wrapper.context.tier}
    {"(Premium Billing Support)" if wrapper.context.tier != "basic" else ""}

    YOUR ROLE: Resolve billing, payment, and subscription issues.
    ...
    {"PREMIUM BENEFITS: Fast-track refund processing and flexible
    payment options available." if wrapper.context.tier != "basic" else ""}
    """

billing_agent = Agent(
    name="Billing Support Agent",
    instructions=dynamic_billing_agent_instructions,
)
```

**핸드오프 설정 (`my_agents/triage_agent.py`)**

```python
import streamlit as st
from agents import (
    Agent, RunContextWrapper, input_guardrail, Runner,
    GuardrailFunctionOutput, handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from models import UserAccountContext, InputGuardRailOutput, HandoffData
from my_agents.account_agent import account_agent
from my_agents.technical_agent import technical_agent
from my_agents.order_agent import order_agent
from my_agents.billing_agent import billing_agent

# 핸드오프 콜백: 핸드오프 발생 시 사이드바에 정보 표시
def handle_handoff(
    wrapper: RunContextWrapper[UserAccountContext],
    input_data: HandoffData,
):
    with st.sidebar:
        st.write(f"""
            Handing off to {input_data.to_agent_name}
            Reason: {input_data.reason}
            Issue Type: {input_data.issue_type}
            Description: {input_data.issue_description}
        """)

# 핸드오프 팩토리 함수
def make_handoff(agent):
    return handoff(
        agent=agent,
        on_handoff=handle_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[off_topic_guardrail],
    handoffs=[
        make_handoff(technical_agent),
        make_handoff(billing_agent),
        make_handoff(account_agent),
        make_handoff(order_agent),
    ],
)
```

**핵심 포인트:**
- `RECOMMENDED_PROMPT_PREFIX`: OpenAI가 제공하는 핸드오프 관련 권장 프롬프트 접두어로, 에이전트에게 핸드오프 방법을 알려준다
- `handoff_filters.remove_all_tools`: 핸드오프 시 이전 에이전트의 도구 호출 이력을 제거하여 새 에이전트가 깨끗한 상태에서 시작하도록 한다
- `make_handoff()` 팩토리 함수로 중복 코드를 줄인다
- 핸드오프는 에이전트를 `as_tool()` 메서드로 도구화하는 방식으로도 구현할 수 있다 (주석 처리된 코드 참고)

### 실습 포인트

1. 새로운 전문 에이전트(예: "반품 전문 에이전트")를 추가하고 핸드오프를 연결해 본다
2. `input_filter`를 `handoff_filters.remove_all_tools` 대신 커스텀 필터로 교체해 본다
3. `as_tool()` 방식과 `handoff()` 방식의 차이를 실험해 본다

---

## 9.5 핸드오프 UI (Handoff UI)

### 주제 및 목표

에이전트 간 핸드오프가 발생했을 때 UI에 **실시간으로 전환 상태를 표시**하고, 현재 활성 에이전트를 추적하여 후속 메시지가 올바른 에이전트에게 전달되도록 한다.

### 핵심 개념 설명

스트리밍 이벤트 중 `agent_updated_stream_event`는 에이전트가 변경될 때 발생한다. 이를 감지하여 UI에 전환 메시지를 표시하고, `st.session_state`에 현재 에이전트를 저장하면 사용자의 다음 메시지가 올바른 전문 에이전트에게 전달된다.

### 코드 분석

**에이전트 상태 추적 (`main.py`)**

```python
# 현재 활성 에이전트를 세션 상태에 저장
if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent
```

**스트리밍 이벤트에서 핸드오프 감지:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                st.session_state["agent"],  # 현재 활성 에이전트 사용
                message,
                session=session,
                context=user_account_ctx,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                # 에이전트 전환 이벤트 감지
                elif event.type == "agent_updated_stream_event":
                    if st.session_state["agent"].name != event.new_agent.name:
                        st.write(
                            f"Transfered from "
                            f"{st.session_state['agent'].name} to "
                            f"{event.new_agent.name}"
                        )
                        # 현재 에이전트를 새 에이전트로 업데이트
                        st.session_state["agent"] = event.new_agent
                        # 새 에이전트의 응답을 위한 플레이스홀더 초기화
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder
                        response = ""

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

**핵심 포인트:**
- `agent_updated_stream_event`: 에이전트가 변경되었을 때 발생하는 스트리밍 이벤트
- `event.new_agent`: 새로 활성화된 에이전트 객체
- 에이전트가 변경되면 `response`를 초기화하고 새로운 `text_placeholder`를 생성하여, 새 에이전트의 응답이 처음부터 표시되도록 한다
- `st.session_state["agent"]`를 업데이트하여 다음 사용자 메시지가 새 에이전트에게 전달되도록 한다

### 실습 포인트

1. 핸드오프 시 전환 메시지의 스타일을 개선해 본다 (예: 구분선 추가)
2. 사이드바에 현재 활성 에이전트 이름을 항상 표시하는 기능을 추가해 본다
3. "트리아지로 돌아가기" 버튼을 만들어 수동으로 에이전트를 초기화하는 기능을 구현해 본다

---

## 9.6 훅 (Hooks)

### 주제 및 목표

에이전트의 **라이프사이클 이벤트**(시작, 종료, 도구 실행, 핸드오프)에 커스텀 로직을 삽입하는 **AgentHooks**를 구현한다. 또한 각 전문 에이전트에 실제 업무 도구들을 추가한다.

### 핵심 개념 설명

**Hooks**는 에이전트의 실행 과정에서 특정 시점에 자동으로 호출되는 콜백 함수들의 모음이다. `AgentHooks` 클래스를 상속하여 다음 메서드를 오버라이드할 수 있다:

| 메서드 | 호출 시점 |
|--------|-----------|
| `on_start` | 에이전트 실행 시작 |
| `on_end` | 에이전트 실행 완료 |
| `on_tool_start` | 도구 함수 호출 직전 |
| `on_tool_end` | 도구 함수 실행 완료 |
| `on_handoff` | 다른 에이전트로 핸드오프 발생 |

### 코드 분석

**도구 함수 예시 (`tools.py`)**

이 커밋에서 441줄에 달하는 `tools.py`가 추가되었다. 각 전문 에이전트를 위한 도구 함수들이 카테고리별로 정리되어 있다.

```python
import streamlit as st
from agents import function_tool, AgentHooks, Agent, Tool, RunContextWrapper
from models import UserAccountContext
import random
from datetime import datetime, timedelta

# === 기술 지원 도구 ===

@function_tool
def run_diagnostic_check(
    context: UserAccountContext, product_name: str, issue_description: str
) -> str:
    """
    Run a diagnostic check on the customer's product to identify potential issues.
    """
    diagnostics = [
        "Server connectivity: Normal",
        "API endpoints: Responsive",
        "Cache memory: 85% full (recommend clearing)",
        "Database connections: Stable",
        "Last update: 7 days ago (update available)",
    ]
    return f"Diagnostic results for {product_name}:\n" + "\n".join(diagnostics)

@function_tool
def escalate_to_engineering(
    context: UserAccountContext, issue_summary: str, priority: str = "medium"
) -> str:
    """Escalate a technical issue to the engineering team."""
    ticket_id = f"ENG-{random.randint(10000, 99999)}"
    return f"""
Issue escalated to Engineering Team
Ticket ID: {ticket_id}
Priority: {priority.upper()}
Summary: {issue_summary}
Expected response: {2 if context.is_premium_customer() else 4} hours
    """.strip()
```

**도구 함수 설계 패턴:**
- `context: UserAccountContext`를 첫 번째 인자로 받아 사용자 정보에 접근한다
- docstring이 에이전트에게 도구의 용도를 설명하는 역할을 한다
- `Args` 섹션의 설명도 에이전트가 참조한다
- 프리미엄 고객 여부에 따라 다른 처리를 한다 (예: 응답 시간 차등)

```python
# === 결제 지원 도구 ===

@function_tool
def process_refund_request(
    context: UserAccountContext, refund_amount: float, reason: str
) -> str:
    """Process a refund request for the customer."""
    processing_days = 3 if context.is_premium_customer() else 5
    refund_id = f"REF-{random.randint(100000, 999999)}"
    return f"""
Refund request processed
Refund ID: {refund_id}
Amount: ${refund_amount}
Processing time: {processing_days} business days
    """.strip()

# === 주문 관리 도구 ===

@function_tool
def lookup_order_status(context: UserAccountContext, order_number: str) -> str:
    """Look up the current status and details of an order."""
    statuses = ["processing", "shipped", "in_transit", "delivered"]
    current_status = random.choice(statuses)
    tracking_number = f"1Z{random.randint(100000, 999999)}"
    estimated_delivery = datetime.now() + timedelta(days=random.randint(1, 5))
    return f"""
Order Status: {order_number}
Status: {current_status.title()}
Tracking: {tracking_number}
Estimated delivery: {estimated_delivery.strftime('%B %d, %Y')}
    """.strip()

# === 계정 관리 도구 ===

@function_tool
def reset_user_password(context: UserAccountContext, email: str) -> str:
    """Send password reset instructions to the customer's email."""
    reset_token = f"RST-{random.randint(100000, 999999)}"
    return f"""
Password reset initiated
Reset link sent to: {email}
Reset token: {reset_token}
Link expires in: 1 hour
    """.strip()
```

**AgentHooks 구현:**

```python
class AgentToolUsageLoggingHooks(AgentHooks):

    async def on_tool_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** starting tool: `{tool.name}`")

    async def on_tool_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        tool: Tool,
        result: str,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** used tool: `{tool.name}`")
            st.code(result)

    async def on_handoff(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        source: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"Handoff: **{source.name}** -> **{agent.name}**")

    async def on_start(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** activated")

    async def on_end(
        self,
        context: RunContextWrapper[UserAccountContext],
        agent: Agent[UserAccountContext],
        output,
    ):
        with st.sidebar:
            st.write(f"**{agent.name}** completed")
```

**에이전트에 도구와 훅 연결 (예: `my_agents/account_agent.py`)**

```python
from tools import (
    reset_user_password,
    enable_two_factor_auth,
    update_account_email,
    deactivate_account,
    export_account_data,
    AgentToolUsageLoggingHooks,
)

account_agent = Agent(
    name="Account Management Agent",
    instructions=dynamic_account_agent_instructions,
    tools=[
        reset_user_password,
        enable_two_factor_auth,
        update_account_email,
        deactivate_account,
        export_account_data,
    ],
    hooks=AgentToolUsageLoggingHooks(),
)
```

**각 전문 에이전트에 할당된 도구 목록:**

| 에이전트 | 도구 |
|----------|------|
| Technical | `run_diagnostic_check`, `provide_troubleshooting_steps`, `escalate_to_engineering` |
| Billing | `lookup_billing_history`, `process_refund_request`, `update_payment_method`, `apply_billing_credit` |
| Order | `lookup_order_status`, `initiate_return_process`, `schedule_redelivery`, `expedite_shipping` |
| Account | `reset_user_password`, `enable_two_factor_auth`, `update_account_email`, `deactivate_account`, `export_account_data` |

### 실습 포인트

1. `on_tool_start`에서 도구 실행 시작 시간을 기록하고, `on_tool_end`에서 소요 시간을 계산하여 표시해 본다
2. 특정 도구(예: `deactivate_account`)가 호출될 때 확인 메시지를 표시하는 훅을 추가해 본다
3. 로그 파일에 도구 사용 이력을 저장하는 훅을 구현해 본다

---

## 9.7 출력 가드레일 (Output Guardrails)

### 주제 및 목표

에이전트의 **응답**이 해당 에이전트의 업무 범위를 벗어나는 내용을 포함하는지 검증하는 출력 가드레일을 구현한다. 입력 가드레일과 대칭적인 구조를 가지되, 에이전트의 최종 출력을 검증한다는 점이 다르다.

### 핵심 개념 설명

**출력 가드레일(Output Guardrail)**은 에이전트가 응답을 생성한 후, 그 응답이 적절한지 검증하는 단계이다. 예를 들어, 기술 지원 에이전트가 결제 정보나 계정 관리 정보를 포함한 응답을 생성했다면, 이는 자신의 영역을 벗어난 것이므로 차단해야 한다.

`OutputGuardrailTripwireTriggered` 예외가 발생하면, 이미 스트리밍으로 표시된 텍스트를 제거하고 대체 메시지를 표시한다.

### 코드 분석

**출력 가드레일 모델 (`models.py`)**

```python
class TechnicalOutputGuardRailOutput(BaseModel):
    contains_off_topic: bool
    contains_billing_data: bool
    contains_account_data: bool
    reason: str
```

- 여러 종류의 부적절한 내용을 개별적으로 검사한다
- 입력 가드레일보다 세분화된 검증 기준을 가진다

**출력 가드레일 정의 (`output_guardrails.py`)**

```python
from agents import (
    Agent, output_guardrail, Runner,
    RunContextWrapper, GuardrailFunctionOutput,
)
from models import TechnicalOutputGuardRailOutput, UserAccountContext

# 출력 검증 전용 에이전트
technical_output_guardrail_agent = Agent(
    name="Technical Support Guardrail",
    instructions="""
    Analyze the technical support response to check if it
    inappropriately contains:

    - Billing information (payments, refunds, charges, subscriptions)
    - Order information (shipping, tracking, delivery, returns)
    - Account management info (passwords, email changes, account settings)

    Technical agents should ONLY provide technical troubleshooting,
    diagnostics, and product support.
    Return true for any field that contains inappropriate content.
    """,
    output_type=TechnicalOutputGuardRailOutput,
)

@output_guardrail
async def technical_output_guardrail(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        technical_output_guardrail_agent,
        output,
        context=wrapper.context,
    )

    validation = result.final_output

    # 세 가지 검증 기준 중 하나라도 위반하면 tripwire 발동
    triggered = (
        validation.contains_off_topic
        or validation.contains_billing_data
        or validation.contains_account_data
    )

    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=triggered,
    )
```

**입력 가드레일과 출력 가드레일 비교:**

| 항목 | 입력 가드레일 | 출력 가드레일 |
|------|--------------|--------------|
| 데코레이터 | `@input_guardrail` | `@output_guardrail` |
| 검사 대상 | 사용자의 메시지 | 에이전트의 응답 |
| 세 번째 인자 | `input: str` | `output: str` |
| 예외 타입 | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
| 적용 위치 | `input_guardrails=[]` | `output_guardrails=[]` |

**에이전트에 출력 가드레일 연결 (`my_agents/technical_agent.py`)**

```python
from output_guardrails import technical_output_guardrail

technical_agent = Agent(
    name="Technical Support Agent",
    instructions=dynamic_technical_agent_instructions,
    tools=[
        run_diagnostic_check,
        provide_troubleshooting_steps,
        escalate_to_engineering,
    ],
    hooks=AgentToolUsageLoggingHooks(),
    output_guardrails=[
        technical_output_guardrail,
    ],
)
```

**예외 처리 (`main.py`)**

```python
from agents import (
    Runner, SQLiteSession,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

# run_agent 함수 내부:
except OutputGuardrailTripwireTriggered:
    st.write("Cant show you that answer.")
    st.session_state["text_placeholder"].empty()  # 이미 표시된 텍스트 제거
```

- `text_placeholder.empty()`로 스트리밍 도중 이미 표시된 부적절한 응답을 화면에서 제거한다

### 실습 포인트

1. 결제 에이전트에도 출력 가드레일을 추가하여 기술 정보가 포함되지 않도록 해 본다
2. 출력 가드레일 에이전트의 지시문에 허용/금지 키워드 목록을 추가해 본다
3. 가드레일 발동 시 `reason`을 로그에 기록하는 기능을 구현해 본다

---

## 9.8 음성 에이전트 I (Voice Agent I)

### 주제 및 목표

텍스트 기반 채팅 인터페이스를 **음성 입력** 인터페이스로 전환한다. Streamlit의 `st.audio_input`으로 음성을 녹음하고, WAV 형식의 오디오를 NumPy 배열로 변환하여 OpenAI Agents SDK의 `AudioInput`으로 전달하는 파이프라인을 구축한다.

### 핵심 개념 설명

음성 에이전트는 다음 단계로 동작한다:
1. **음성 녹음**: Streamlit의 `st.audio_input` 위젯으로 브라우저에서 직접 녹음
2. **오디오 변환**: WAV 파일을 NumPy `int16` 배열로 변환
3. **AudioInput 생성**: 변환된 배열을 `AudioInput(buffer=array)` 형태로 래핑
4. **에이전트 실행**: 음성 데이터를 에이전트에 전달

### 코드 분석

**새 의존성 추가 (`pyproject.toml`)**

```toml
dependencies = [
    "numpy>=2.3.2",
    "openai-agents[voice]>=0.2.8",
    "python-dotenv>=1.1.1",
    "sounddevice>=0.5.2",
    "streamlit>=1.48.1",
]
```

- `numpy`: 오디오 데이터를 배열로 처리
- `sounddevice`: 오디오 출력(재생)용

**오디오 변환 함수 (`main.py`)**

```python
from agents.voice import AudioInput
import numpy as np
import wave, io

def convert_audio(audio_input):
    # Streamlit 오디오 입력을 바이트로 변환
    audio_data = audio_input.getvalue()

    # WAV 파일로 파싱하여 프레임 추출
    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        audio_frames = wav_file.readframes(-1)  # -1은 모든 프레임

    # NumPy int16 배열로 변환
    return np.frombuffer(
        audio_frames,
        dtype=np.int16,
    )
```

**동작 원리:**
1. `audio_input.getvalue()`: Streamlit의 `UploadedFile` 객체에서 바이너리 데이터를 추출
2. `wave.open(io.BytesIO(...))`: 바이트 데이터를 메모리 상의 WAV 파일로 열기
3. `wav_file.readframes(-1)`: 모든 오디오 프레임(raw PCM 데이터)을 읽기
4. `np.frombuffer(..., dtype=np.int16)`: PCM 데이터를 16비트 정수 배열로 변환

**음성 입력 UI 전환:**

```python
# 텍스트 입력 대신 음성 입력 사용
audio_input = st.audio_input(
    "Record your message",
)

if audio_input:
    with st.chat_message("human"):
        st.audio(audio_input)  # 녹음된 오디오를 재생 가능하게 표시
    asyncio.run(run_agent(audio_input))
```

**에이전트 실행 함수 변경:**

```python
async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)
            # 다음 섹션에서 VoicePipeline으로 완성

            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=user_account_ctx,
            )
        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**이 커밋에서 제거된 것들:**
- `paint_history()` 함수 (대화 이력 표시) - 음성 에이전트에서는 이전 대화를 텍스트로 표시하는 것이 부자연스러우므로 제거
- 텍스트 기반 스트리밍 이벤트 처리 로직 - 다음 섹션에서 음성 파이프라인으로 대체

### 실습 포인트

1. WAV 파일의 구조(헤더, 채널 수, 샘플 레이트)를 `wave` 모듈로 확인해 본다
2. 오디오 배열의 shape과 dtype을 출력하여 데이터 형태를 이해한다
3. `st.audio(audio_input)`과 직접 변환한 배열을 비교하여 변환이 올바른지 검증한다

---

## 9.9 음성 에이전트 II (Voice Agent II)

### 주제 및 목표

`VoicePipeline`과 커스텀 `VoiceWorkflowBase`를 구현하여 **음성 입력 -> 텍스트 변환 -> 에이전트 처리 -> 음성 출력** 전체 파이프라인을 완성한다. `sounddevice`를 사용한 실시간 음성 출력까지 구현한다.

### 핵심 개념 설명

**VoicePipeline**은 OpenAI Agents SDK가 제공하는 음성 처리 파이프라인으로, 다음 단계를 자동으로 처리한다:
1. **STT (Speech-to-Text)**: 오디오를 텍스트로 변환
2. **워크플로우 실행**: 변환된 텍스트를 에이전트에 전달하여 응답 생성
3. **TTS (Text-to-Speech)**: 에이전트 응답을 음성으로 변환

**VoiceWorkflowBase**를 상속하여 커스텀 워크플로우를 정의하면, 에이전트 실행 로직을 자유롭게 커스터마이징할 수 있다.

### 코드 분석

**커스텀 워크플로우 (`workflow.py`)**

```python
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
from agents import Runner
import streamlit as st

class CustomWorkflow(VoiceWorkflowBase):

    def __init__(self, context):
        self.context = context

    async def run(self, transcription):
        # STT로 변환된 텍스트(transcription)를 받아서 에이전트 실행
        result = Runner.run_streamed(
            st.session_state["agent"],
            transcription,
            session=st.session_state["session"],
            context=self.context,
        )

        # 에이전트 응답을 텍스트 청크 단위로 스트리밍
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # 핸드오프가 발생했을 수 있으므로 마지막 활성 에이전트를 업데이트
        st.session_state["agent"] = result.last_agent
```

**핵심 포인트:**
- `VoiceWorkflowBase`를 상속하고 `run()` 메서드를 구현한다
- `run()` 메서드는 **async generator**로 정의된다 (`yield` 사용)
- `transcription`: 음성이 텍스트로 변환된 결과 (STT 결과)
- `VoiceWorkflowHelper.stream_text_from()`: `Runner` 결과에서 텍스트 청크를 추출하는 유틸리티
- `result.last_agent`: 핸드오프가 발생했을 경우 마지막으로 활성화된 에이전트를 반환한다. 이를 `session_state`에 저장하여 다음 음성 입력 시 올바른 에이전트가 사용되도록 한다

**VoicePipeline 통합 (`main.py`)**

```python
from agents.voice import AudioInput, VoicePipeline
from workflow import CustomWorkflow
import sounddevice as sd

async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            # 1. 오디오 변환
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)

            # 2. 커스텀 워크플로우 생성
            workflow = CustomWorkflow(context=user_account_ctx)

            # 3. 음성 파이프라인 생성 및 실행
            pipeline = VoicePipeline(workflow=workflow)

            status_container.update(label="Running workflow", state="running")

            result = await pipeline.run(audio)

            # 4. 오디오 출력 스트림 설정
            player = sd.OutputStream(
                samplerate=24000,  # 24kHz 샘플 레이트
                channels=1,        # 모노 오디오
                dtype=np.int16,    # 16비트 정수
            )
            player.start()

            status_container.update(state="complete")

            # 5. 음성 응답 실시간 재생
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**음성 파이프라인 동작 순서:**
1. `convert_audio()`: WAV -> NumPy 배열
2. `AudioInput(buffer=audio_array)`: 배열 -> AudioInput 객체
3. `CustomWorkflow(context=...)`: 컨텍스트를 포함한 워크플로우 생성
4. `VoicePipeline(workflow=workflow)`: 파이프라인 생성
5. `pipeline.run(audio)`: STT -> 에이전트 실행 -> TTS (비동기)
6. `result.stream()`: TTS 결과를 청크 단위로 스트리밍
7. `player.write(event.data)`: 각 오디오 청크를 스피커로 출력

**트리아지 에이전트 수정 (`my_agents/triage_agent.py`)**

```python
def dynamic_triage_agent_instructions(
    wrapper: RunContextWrapper[UserAccountContext],
    agent: Agent[UserAccountContext],
):
    return f"""
    SPEAK TO THE USER IN ENGLISH

    {RECOMMENDED_PROMPT_PREFIX}

    You are a customer support agent...
    """

triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    # input_guardrails=[
    #     # off_topic_guardrail,
    # ],
    handoffs=[
        make_handoff(technical_agent),
        make_handoff(billing_agent),
        make_handoff(account_agent),
        make_handoff(order_agent),
    ],
)
```

**변경 사항:**
- `"SPEAK TO THE USER IN ENGLISH"`을 지시문 맨 앞에 추가하여 TTS 출력 언어를 명시
- 입력 가드레일이 주석 처리됨 - 음성 입력의 경우 STT 변환 결과가 가드레일 에이전트에 부적합할 수 있기 때문

### 실습 포인트

1. `samplerate`를 변경하여 음질 차이를 확인해 본다 (16000, 24000, 48000)
2. `CustomWorkflow.run()`에서 `transcription` 값을 로깅하여 STT 변환 품질을 확인해 본다
3. TTS 응답을 파일로 저장하여 나중에 재생할 수 있는 기능을 추가해 본다
4. 음성 에이전트에서 핸드오프가 정상적으로 동작하는지 테스트해 본다

---

## 챕터 핵심 정리

### 1. 아키텍처 패턴

이 챕터에서 구현한 시스템은 다음과 같은 계층적 멀티 에이전트 아키텍처를 따른다:

```
사용자 입력
    |
    v
[입력 가드레일] -- 부적절 --> 차단 메시지
    |
    v (통과)
[트리아지 에이전트] -- 분류 --> 핸드오프
    |           |           |           |
    v           v           v           v
[기술지원]  [결제지원]  [주문관리]  [계정관리]
   |           |           |           |
   v           v           v           v
[출력 가드레일] -- 부적절 --> 차단 메시지
    |
    v (통과)
사용자에게 응답
```

### 2. 핵심 SDK 컴포넌트 요약

| 컴포넌트 | 역할 | 사용 섹션 |
|----------|------|-----------|
| `Agent` | 에이전트 정의 (지시문, 도구, 가드레일) | 전체 |
| `Runner.run_streamed()` | 스트리밍 에이전트 실행 | 전체 |
| `SQLiteSession` | 대화 이력 영구 저장 | 9.0 |
| `RunContextWrapper` | 실행 컨텍스트 접근 | 9.1+ |
| `@function_tool` | 도구 함수 정의 | 9.1, 9.6 |
| `@input_guardrail` | 입력 검증 | 9.3 |
| `@output_guardrail` | 출력 검증 | 9.7 |
| `handoff()` | 에이전트 간 전환 | 9.4 |
| `AgentHooks` | 라이프사이클 콜백 | 9.6 |
| `VoicePipeline` | 음성 처리 파이프라인 | 9.9 |
| `VoiceWorkflowBase` | 커스텀 음성 워크플로우 | 9.9 |

### 3. 가드레일 설계 원칙

- **입력 가드레일**: 에이전트 처리 전에 부적절한 요청을 차단한다 (비용 절감, 보안)
- **출력 가드레일**: 에이전트 응답 후에 부적절한 내용을 차단한다 (품질 보장, 데이터 유출 방지)
- 가드레일 에이전트는 별도의 경량 에이전트로 분리하여 관심사를 분리한다
- `output_type`에 Pydantic 모델을 지정하여 구조화된 판단 결과를 강제한다

### 4. 핸드오프 설계 원칙

- 각 전문 에이전트는 명확한 책임 영역을 가진다
- `handoff_filters.remove_all_tools`로 이전 에이전트의 도구 이력을 정리한다
- `on_handoff` 콜백으로 핸드오프 메타데이터를 로깅한다
- `result.last_agent` 또는 `agent_updated_stream_event`로 현재 활성 에이전트를 추적한다

---

## 실습 과제

### 과제 1: 새로운 전문 에이전트 추가 (난이도: 중)

**목표**: 환불 전문 에이전트를 추가하라.

**요구사항**:
- `my_agents/refund_agent.py` 파일을 생성한다
- 동적 지시문을 정의하고, 프리미엄 고객에게 추가 혜택을 안내한다
- 환불 관련 도구 함수 2개 이상을 `tools.py`에 추가한다
- `AgentToolUsageLoggingHooks`를 연결한다
- 트리아지 에이전트의 핸드오프 목록에 추가한다
- 트리아지 에이전트의 분류 가이드에 환불 관련 항목을 추가한다

### 과제 2: 출력 가드레일 확장 (난이도: 중)

**목표**: 모든 전문 에이전트에 출력 가드레일을 추가하라.

**요구사항**:
- 결제 에이전트: 기술 지원 정보가 포함되지 않도록 검증
- 주문 에이전트: 결제 또는 계정 정보가 포함되지 않도록 검증
- 계정 에이전트: 주문 또는 결제 정보가 포함되지 않도록 검증
- 각 가드레일에 대한 Pydantic 출력 모델을 `models.py`에 정의한다

### 과제 3: 커스텀 훅 시스템 (난이도: 상)

**목표**: 에이전트 사용 통계를 수집하는 고급 훅 시스템을 구현하라.

**요구사항**:
- 각 에이전트의 호출 횟수, 평균 응답 시간, 도구 사용 빈도를 추적한다
- 통계 데이터를 사이드바에 실시간으로 표시한다
- 세션이 초기화되면 통계도 초기화된다
- `on_start`에서 시작 시간을 기록하고, `on_end`에서 소요 시간을 계산한다

### 과제 4: 양방향 음성 에이전트 (난이도: 상)

**목표**: 연속적인 음성 대화가 가능한 에이전트를 구현하라.

**요구사항**:
- 음성 응답 재생이 끝나면 자동으로 다음 녹음을 시작한다
- 대화 중 핸드오프가 발생하면 어떤 에이전트로 전환되었는지 음성으로 안내한다
- 대화 기록을 텍스트로 사이드바에 표시한다 (STT 결과와 에이전트 응답 모두)
- "대화 종료" 명령어를 음성으로 인식하여 세션을 종료하는 기능을 추가한다

### 과제 5: 에이전트 간 복귀 메커니즘 (난이도: 상)

**목표**: 전문 에이전트에서 트리아지 에이전트로 돌아가는 메커니즘을 구현하라.

**요구사항**:
- 각 전문 에이전트에 "트리아지로 복귀" 핸드오프를 추가한다
- 전문 에이전트가 자신의 업무 범위를 벗어나는 요청을 받으면 자동으로 트리아지로 복귀한다
- 복귀 시 현재까지의 대화 요약을 핸드오프 데이터에 포함한다
- UI에서 복귀 이벤트를 시각적으로 표시한다
