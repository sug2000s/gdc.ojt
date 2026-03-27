# Chapter 9: OpenAI Agents SDK - Guardrails, Handoff và Voice Agent

---

## Tổng quan chương

Trong chương này, chúng ta sử dụng OpenAI Agents SDK (`openai-agents`) để xây dựng từng bước **hệ thống agent hỗ trợ khách hàng** ở mức độ thực chiến. Vượt xa chatbot đơn giản, chúng ta hoàn thành dự án tổng hợp bao gồm quản lý context, chỉ dẫn động, guardrail đầu vào/đầu ra, handoff giữa các agent, lifecycle hook, và voice agent.

### Mục tiêu học tập

| Phần | Chủ đề | Từ khóa chính |
|------|--------|---------------|
| 9.0 | Giới thiệu dự án và cấu trúc cơ bản | Streamlit, SQLiteSession, Runner |
| 9.1 | Quản lý context | RunContextWrapper, Pydantic model |
| 9.2 | Chỉ dẫn động | Dynamic Instructions, prompt dựa trên hàm |
| 9.3 | Guardrail đầu vào | Input Guardrail, Tripwire |
| 9.4 | Handoff agent | Handoff, routing agent chuyên biệt |
| 9.5 | UI Handoff | agent_updated_stream_event, hiển thị chuyển đổi thời gian thực |
| 9.6 | Hook | AgentHooks, logging sử dụng công cụ |
| 9.7 | Guardrail đầu ra | Output Guardrail, xác minh phản hồi |
| 9.8 | Voice Agent I | AudioInput, chuyển đổi WAV |
| 9.9 | Voice Agent II | VoicePipeline, VoiceWorkflowBase, sounddevice |

### Cấu trúc dự án (cuối cùng)

```
customer-support-agent/
├── main.py                      # Ứng dụng chính Streamlit
├── models.py                    # Mô hình dữ liệu Pydantic
├── tools.py                     # Các hàm công cụ agent
├── output_guardrails.py         # Định nghĩa guardrail đầu ra
├── workflow.py                  # Workflow tùy chỉnh voice agent
├── my_agents/
│   ├── triage_agent.py          # Agent phân loại (triage)
│   ├── technical_agent.py       # Agent hỗ trợ kỹ thuật
│   ├── billing_agent.py         # Agent hỗ trợ thanh toán
│   ├── order_agent.py           # Agent quản lý đơn hàng
│   └── account_agent.py         # Agent quản lý tài khoản
├── pyproject.toml               # Dependency dự án
└── customer-support-memory.db   # Kho lưu trữ session SQLite
```

---

## 9.0 Giới thiệu dự án và cấu trúc cơ bản

### Chủ đề và mục tiêu

Kết hợp OpenAI Agents SDK và Streamlit để tạo khung cơ bản cho chatbot hỗ trợ khách hàng. Thiết lập cấu trúc lưu lịch sử hội thoại vào SQLite và hiển thị streaming response theo thời gian thực trên màn hình.

### Giải thích khái niệm cốt lõi

**OpenAI Agents SDK** là framework phát triển ứng dụng dựa trên agent, cho phép thực thi agent thông qua `Runner` và lưu trữ vĩnh viễn lịch sử hội thoại bằng `SQLiteSession`. Streamlit là framework web UI cho prototyping nhanh, giúp dễ dàng triển khai giao diện chat bằng `st.chat_message` và `st.chat_input`.

### Phân tích code

**Thiết lập dependency dự án (`pyproject.toml`)**

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

- `openai-agents[voice]`: Gói OpenAI Agents SDK bao gồm tính năng voice agent
- `python-dotenv`: Tải biến môi trường như API key từ file `.env`
- `streamlit`: Framework web UI

**Ứng dụng chính (`main.py`)**

```python
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import streamlit as st
from agents import Runner, SQLiteSession

client = OpenAI()

# Quản lý session dựa trên SQLite
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]
```

**Điểm cốt lõi:**
- `SQLiteSession` nhận hai đối số: tên session (`"chat-history"`) và đường dẫn file database (`"customer-support-memory.db"`)
- Sử dụng `st.session_state` để duy trì đối tượng session giữa các lần re-render của Streamlit

**Hàm hiển thị lịch sử hội thoại:**

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

- Lấy tất cả tin nhắn đã lưu bằng `session.get_items()`
- Lý do escape ký hiệu `$` thành `\$` là để ngăn Streamlit diễn giải thành công thức LaTeX

**Thực thi agent streaming:**

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

- `Runner.run_streamed()` chạy agent ở chế độ streaming, phát sinh sự kiện mỗi khi token được tạo
- Phát hiện `response.output_text.delta` từ loại `raw_response_event` để hiển thị tích lũy văn bản theo thời gian thực
- `st.empty()` tạo placeholder có thể cập nhật nội dung sau

### Điểm thực hành

1. Khởi tạo dự án và cài đặt dependency bằng trình quản lý gói `uv`
2. Chạy ứng dụng bằng `streamlit run main.py` và kiểm tra giao diện chat cơ bản
3. Mở file SQLite DB và xem lịch sử hội thoại được lưu ở định dạng nào

---

## 9.1 Quản lý Context (Context Management)

### Chủ đề và mục tiêu

Học cách truyền **thông tin context người dùng** khi thực thi agent. Định nghĩa context an toàn về kiểu bằng mô hình Pydantic và nắm vững mẫu truy cập thông tin này bên trong hàm công cụ thông qua `RunContextWrapper`.

### Giải thích khái niệm cốt lõi

**Context** là thông tin bên ngoài mà agent có thể tham chiếu trong quá trình thực thi. Ví dụ: ID, tên, cấp độ đăng ký của người dùng đang đăng nhập. Thông tin này được sử dụng trong prompt hoặc hàm công cụ của agent.

`RunContextWrapper` là kiểu generic, khi chỉ định kiểu context như `RunContextWrapper[UserAccountContext]`, bạn nhận được hỗ trợ tự động hoàn thành và kiểm tra kiểu từ IDE.

### Phân tích code

**Định nghĩa mô hình context (`models.py`)**

```python
from pydantic import BaseModel

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"  # premium, enterprise
```

- Kế thừa Pydantic `BaseModel` để tự động xử lý xác thực dữ liệu và serialization
- Đặt giá trị mặc định `"basic"` cho trường `tier` để biến nó thành trường tùy chọn

**Sử dụng context trong hàm công cụ (`main.py`)**

```python
from agents import Runner, SQLiteSession, function_tool, RunContextWrapper
from models import UserAccountContext

@function_tool
def get_user_tier(wrapper: RunContextWrapper[UserAccountContext]):
    return (
        f"The user {wrapper.context.customer_id} has a {wrapper.context.tier} account."
    )
```

- Decorator `@function_tool` chuyển đổi hàm Python thông thường thành công cụ mà agent có thể gọi
- Truy cập instance `UserAccountContext` được truyền khi thực thi thông qua `wrapper.context`

**Tạo và truyền context:**

```python
user_account_ctx = UserAccountContext(
    customer_id=1,
    name="nico",
    tier="basic",
)

# Truyền context khi thực thi Runner
stream = Runner.run_streamed(
    agent,
    message,
    session=session,
    context=user_account_ctx,  # Inject context
)
```

- Truyền đối tượng context qua tham số `context` của `Runner.run_streamed()`
- Context này có thể truy cập từ tất cả hàm công cụ và chỉ dẫn của agent

### Điểm thực hành

1. Thêm trường mới như `phone_number`, `preferred_language` vào `UserAccountContext`
2. Tạo `@function_tool` mới sử dụng thông tin context
3. Triển khai công cụ trả về response khác nhau tùy theo giá trị `tier`

---

## 9.2 Chỉ dẫn Động (Dynamic Instructions)

### Chủ đề và mục tiêu

Học cách định nghĩa chỉ dẫn (instructions) của agent không phải là **chuỗi tĩnh** mà là **hàm**, tạo prompt thay đổi động tùy theo context tại thời điểm thực thi.

### Giải thích khái niệm cốt lõi

Thông thường `instructions` của agent là chuỗi cố định. Tuy nhiên, để bao gồm thông tin runtime như tên, cấp độ, email của người dùng trong prompt, cần chỉ dẫn động dạng hàm. Hàm này nhận đối tượng `RunContextWrapper` và `Agent` làm tham số và trả về chuỗi.

### Phân tích code

**Tạo agent phân loại (`my_agents/triage_agent.py`)**

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
    instructions=dynamic_triage_agent_instructions,  # Truyền trực tiếp hàm
)
```

**Điểm cốt lõi:**
- Truyền **tham chiếu hàm** cho tham số `instructions` (không có ngoặc đơn: `dynamic_triage_agent_instructions`)
- Chữ ký hàm phải có dạng `(wrapper: RunContextWrapper[T], agent: Agent[T]) -> str`
- Sử dụng f-string để chèn giá trị context như `wrapper.context.name`, `wrapper.context.tier` vào prompt
- Agent phân loại (Triage) đóng vai trò phân loại yêu cầu khách hàng và routing đến agent chuyên biệt phù hợp

**Thay đổi trong main.py:**

Trong commit này, hàm công cụ `get_user_tier` trước đó trong `main.py` đã bị xóa. Chuyển sang phương thức xử lý truy cập thông tin context trực tiếp trong chỉ dẫn động thay vì qua công cụ.

### Điểm thực hành

1. Sửa đổi chỉ dẫn động để response bằng giọng điệu khác (trang trọng/thân mật) tùy theo giá trị `wrapper.context.tier`
2. Thêm thời gian hiện tại (`datetime.now()`) vào chỉ dẫn để thêm lời chào theo múi giờ
3. Viết chỉ dẫn động sử dụng thuộc tính của đối tượng agent (`agent`)

---

## 9.3 Guardrail Đầu vào (Input Guardrails)

### Chủ đề và mục tiêu

Triển khai guardrail đầu vào **tự động kiểm tra** xem đầu vào người dùng có nằm ngoài phạm vi công việc của agent hay không. Học mẫu trong đó một "agent guardrail" riêng biệt phân tích đầu vào và chặn hội thoại nếu yêu cầu không phù hợp.

### Giải thích khái niệm cốt lõi

**Input Guardrail** là bước xác minh được thực thi trước khi agent xử lý đầu vào người dùng. Trong quá trình này, một agent nhỏ riêng biệt (agent guardrail) phân tích đầu vào để đánh giá xem có "lệch chủ đề (off-topic)" hay không. Khi phát hiện đầu vào không phù hợp, **Tripwire** được kích hoạt và phát sinh ngoại lệ `InputGuardrailTripwireTriggered`.

Ưu điểm của mẫu này:
- Không cần làm phức tạp chỉ dẫn của agent chính
- Kiểm tra guardrail được **thực thi song song bất đồng bộ** nên giảm thiểu suy giảm hiệu suất
- Tách logic xác minh thành module riêng giúp dễ tái sử dụng và kiểm thử

### Phân tích code

**Mô hình đầu ra guardrail (`models.py`)**

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

- `InputGuardRailOutput` là định dạng đầu ra có cấu trúc của agent guardrail
- `is_off_topic`: Yêu cầu có nằm ngoài phạm vi công việc hay không
- `reason`: Căn cứ đánh giá (dùng cho debug và logging)

**Agent guardrail và decorator (`my_agents/triage_agent.py`)**

```python
from agents import (
    Agent, RunContextWrapper, input_guardrail,
    Runner, GuardrailFunctionOutput,
)
from models import UserAccountContext, InputGuardRailOutput

# 1. Định nghĩa agent chuyên dụng cho guardrail
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
    output_type=InputGuardRailOutput,  # Ép buộc đầu ra có cấu trúc
)

# 2. Định nghĩa hàm guardrail
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

# 3. Kết nối guardrail với agent phân loại
triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_triage_agent_instructions,
    input_guardrails=[
        off_topic_guardrail,
    ],
)
```

**Trình tự hoạt động:**
1. Người dùng gửi tin nhắn
2. Hàm `off_topic_guardrail` được thực thi
3. Bên trong, `input_guardrail_agent` phân tích tin nhắn
4. Trả về kết quả dạng `InputGuardRailOutput`
5. Nếu `is_off_topic` là `True`, `tripwire_triggered=True` được thiết lập
6. Khi Tripwire được kích hoạt, ngoại lệ `InputGuardrailTripwireTriggered` phát sinh

**Xử lý ngoại lệ (`main.py`)**

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

- Bắt ngoại lệ `InputGuardrailTripwireTriggered` bằng `try/except` và hiển thị tin nhắn phù hợp cho người dùng

### Điểm thực hành

1. Sửa đổi chỉ dẫn của agent guardrail để áp dụng lọc nghiêm ngặt hơn/lỏng hơn
2. Kiểm thử guardrail bằng tin nhắn off-topic như "Thời tiết hôm nay thế nào?" và tin nhắn bình thường như "Tôi muốn đổi mật khẩu"
3. Thêm tính năng hiển thị trường `reason` trên UI để cho biết tại sao bị chặn

---

## 9.4 Handoff Agent (Chuyển giao Agent)

### Chủ đề và mục tiêu

Triển khai cấu trúc multi-agent trong đó agent phân loại phân loại yêu cầu khách hàng rồi chuyển giao (handoff) cho **agent chuyên biệt** phù hợp. Tạo 4 agent chuyên biệt (hỗ trợ kỹ thuật, thanh toán, đơn hàng, tài khoản) và thiết lập cơ chế handoff.

### Giải thích khái niệm cốt lõi

**Handoff** là việc một agent chuyển quyền kiểm soát hội thoại cho agent khác. Tương tự như nhân viên tổng đài chuyển cuộc gọi đến bộ phận chuyên môn.

Trong OpenAI Agents SDK, handoff bao gồm các yếu tố sau:
- Hàm `handoff()`: Định nghĩa cấu hình handoff
- `on_handoff`: Hàm callback được thực thi khi handoff xảy ra
- `input_type`: Schema dữ liệu được truyền khi handoff
- `input_filter`: Bộ lọc dọn dẹp lịch sử tool call của agent trước khi handoff

### Phân tích code

**Mô hình dữ liệu handoff (`models.py`)**

```python
class HandoffData(BaseModel):
    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
```

Mô hình này định nghĩa metadata mà agent phân loại truyền cho agent chuyên biệt khi handoff.

**Ví dụ agent chuyên biệt - Hỗ trợ kỹ thuật (`my_agents/technical_agent.py`)**

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

- Tất cả agent chuyên biệt tuân theo cùng một mẫu: chỉ dẫn động + tạo `Agent`
- Hướng dẫn quyền lợi bổ sung cho khách hàng premium tùy theo `wrapper.context.tier`

**Agent thanh toán (`my_agents/billing_agent.py`)**

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

**Thiết lập handoff (`my_agents/triage_agent.py`)**

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

# Callback handoff: Hiển thị thông tin trong sidebar khi handoff xảy ra
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

# Hàm factory handoff
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

**Điểm cốt lõi:**
- `RECOMMENDED_PROMPT_PREFIX`: Tiền tố prompt khuyến nghị về handoff do OpenAI cung cấp, hướng dẫn agent cách thực hiện handoff
- `handoff_filters.remove_all_tools`: Xóa lịch sử tool call của agent trước đó khi handoff để agent mới bắt đầu từ trạng thái sạch
- Hàm factory `make_handoff()` giảm code trùng lặp
- Handoff cũng có thể được triển khai bằng cách tool hóa agent với phương thức `as_tool()` (xem code đã comment)

### Điểm thực hành

1. Thêm agent chuyên biệt mới (ví dụ: "agent chuyên hoàn trả") và kết nối handoff
2. Thay thế `input_filter` bằng bộ lọc tùy chỉnh thay vì `handoff_filters.remove_all_tools`
3. Thử nghiệm sự khác biệt giữa phương thức `as_tool()` và `handoff()`

---

## 9.5 UI Handoff (Handoff UI)

### Chủ đề và mục tiêu

Khi handoff giữa các agent xảy ra, **hiển thị trạng thái chuyển đổi theo thời gian thực trên UI** và theo dõi agent đang hoạt động hiện tại để tin nhắn tiếp theo được chuyển đến agent đúng.

### Giải thích khái niệm cốt lõi

Trong các sự kiện streaming, `agent_updated_stream_event` phát sinh khi agent thay đổi. Phát hiện sự kiện này để hiển thị tin nhắn chuyển đổi trên UI và lưu agent hiện tại vào `st.session_state` để tin nhắn tiếp theo của người dùng được chuyển đến agent chuyên biệt đúng.

### Phân tích code

**Theo dõi trạng thái agent (`main.py`)**

```python
# Lưu agent đang hoạt động hiện tại vào trạng thái session
if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent
```

**Phát hiện handoff trong sự kiện streaming:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                st.session_state["agent"],  # Sử dụng agent đang hoạt động hiện tại
                message,
                session=session,
                context=user_account_ctx,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\$"))

                # Phát hiện sự kiện chuyển đổi agent
                elif event.type == "agent_updated_stream_event":
                    if st.session_state["agent"].name != event.new_agent.name:
                        st.write(
                            f"Transfered from "
                            f"{st.session_state['agent'].name} to "
                            f"{event.new_agent.name}"
                        )
                        # Cập nhật agent hiện tại thành agent mới
                        st.session_state["agent"] = event.new_agent
                        # Khởi tạo placeholder cho response của agent mới
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder
                        response = ""

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

**Điểm cốt lõi:**
- `agent_updated_stream_event`: Sự kiện streaming phát sinh khi agent thay đổi
- `event.new_agent`: Đối tượng agent mới được kích hoạt
- Khi agent thay đổi, khởi tạo `response` và tạo `text_placeholder` mới để response của agent mới hiển thị từ đầu
- Cập nhật `st.session_state["agent"]` để tin nhắn người dùng tiếp theo được chuyển đến agent mới

### Điểm thực hành

1. Cải thiện style tin nhắn chuyển đổi khi handoff (ví dụ: thêm đường phân cách)
2. Thêm tính năng luôn hiển thị tên agent đang hoạt động hiện tại trong sidebar
3. Tạo nút "Quay lại triage" để khởi tạo lại agent thủ công

---

## 9.6 Hook

### Chủ đề và mục tiêu

Triển khai **AgentHooks** để chèn logic tùy chỉnh vào các **sự kiện lifecycle** của agent (bắt đầu, kết thúc, thực thi công cụ, handoff). Đồng thời thêm các công cụ nghiệp vụ thực tế cho mỗi agent chuyên biệt.

### Giải thích khái niệm cốt lõi

**Hooks** là tập hợp các hàm callback được tự động gọi tại thời điểm cụ thể trong quá trình thực thi agent. Kế thừa lớp `AgentHooks` và override các phương thức sau:

| Phương thức | Thời điểm gọi |
|-------------|---------------|
| `on_start` | Bắt đầu thực thi agent |
| `on_end` | Hoàn thành thực thi agent |
| `on_tool_start` | Ngay trước khi gọi hàm công cụ |
| `on_tool_end` | Hoàn thành thực thi hàm công cụ |
| `on_handoff` | Handoff sang agent khác xảy ra |

### Phân tích code

**Ví dụ hàm công cụ (`tools.py`)**

Trong commit này, `tools.py` với 441 dòng được thêm vào. Các hàm công cụ cho mỗi agent chuyên biệt được tổ chức theo danh mục.

```python
import streamlit as st
from agents import function_tool, AgentHooks, Agent, Tool, RunContextWrapper
from models import UserAccountContext
import random
from datetime import datetime, timedelta

# === Công cụ hỗ trợ kỹ thuật ===

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

**Mẫu thiết kế hàm công cụ:**
- Nhận `context: UserAccountContext` làm đối số đầu tiên để truy cập thông tin người dùng
- Docstring đóng vai trò giải thích mục đích công cụ cho agent
- Mô tả trong phần `Args` cũng được agent tham khảo
- Xử lý khác nhau tùy theo khách hàng premium hay không (ví dụ: thời gian phản hồi khác nhau)

```python
# === Công cụ hỗ trợ thanh toán ===

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

# === Công cụ quản lý đơn hàng ===

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

# === Công cụ quản lý tài khoản ===

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

**Triển khai AgentHooks:**

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

**Kết nối công cụ và hook với agent (ví dụ: `my_agents/account_agent.py`)**

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

**Danh sách công cụ được gán cho mỗi agent chuyên biệt:**

| Agent | Công cụ |
|-------|---------|
| Technical | `run_diagnostic_check`, `provide_troubleshooting_steps`, `escalate_to_engineering` |
| Billing | `lookup_billing_history`, `process_refund_request`, `update_payment_method`, `apply_billing_credit` |
| Order | `lookup_order_status`, `initiate_return_process`, `schedule_redelivery`, `expedite_shipping` |
| Account | `reset_user_password`, `enable_two_factor_auth`, `update_account_email`, `deactivate_account`, `export_account_data` |

### Điểm thực hành

1. Ghi lại thời gian bắt đầu thực thi công cụ trong `on_tool_start` và tính thời gian thực hiện trong `on_tool_end` để hiển thị
2. Thêm hook hiển thị tin nhắn xác nhận khi công cụ cụ thể (ví dụ: `deactivate_account`) được gọi
3. Triển khai hook lưu lịch sử sử dụng công cụ vào file log

---

## 9.7 Guardrail Đầu ra (Output Guardrails)

### Chủ đề và mục tiêu

Triển khai guardrail đầu ra để xác minh xem **response** của agent có chứa nội dung nằm ngoài phạm vi công việc của agent đó hay không. Có cấu trúc đối xứng với guardrail đầu vào, nhưng khác ở chỗ xác minh đầu ra cuối cùng của agent.

### Giải thích khái niệm cốt lõi

**Output Guardrail** là bước xác minh sau khi agent tạo response, kiểm tra xem response đó có phù hợp hay không. Ví dụ, nếu agent hỗ trợ kỹ thuật tạo response chứa thông tin thanh toán hoặc quản lý tài khoản, đó là vi phạm phạm vi và cần bị chặn.

Khi ngoại lệ `OutputGuardrailTripwireTriggered` phát sinh, văn bản đã hiển thị qua streaming sẽ bị xóa và thay bằng tin nhắn thay thế.

### Phân tích code

**Mô hình guardrail đầu ra (`models.py`)**

```python
class TechnicalOutputGuardRailOutput(BaseModel):
    contains_off_topic: bool
    contains_billing_data: bool
    contains_account_data: bool
    reason: str
```

- Kiểm tra riêng từng loại nội dung không phù hợp
- Tiêu chí xác minh chi tiết hơn so với guardrail đầu vào

**Định nghĩa guardrail đầu ra (`output_guardrails.py`)**

```python
from agents import (
    Agent, output_guardrail, Runner,
    RunContextWrapper, GuardrailFunctionOutput,
)
from models import TechnicalOutputGuardRailOutput, UserAccountContext

# Agent chuyên dụng cho xác minh đầu ra
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

    # Kích hoạt tripwire nếu vi phạm một trong ba tiêu chí xác minh
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

**So sánh guardrail đầu vào và đầu ra:**

| Hạng mục | Guardrail đầu vào | Guardrail đầu ra |
|----------|-------------------|-------------------|
| Decorator | `@input_guardrail` | `@output_guardrail` |
| Đối tượng kiểm tra | Tin nhắn người dùng | Response agent |
| Đối số thứ ba | `input: str` | `output: str` |
| Kiểu ngoại lệ | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
| Vị trí áp dụng | `input_guardrails=[]` | `output_guardrails=[]` |

**Kết nối guardrail đầu ra với agent (`my_agents/technical_agent.py`)**

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

**Xử lý ngoại lệ (`main.py`)**

```python
from agents import (
    Runner, SQLiteSession,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

# Bên trong hàm run_agent:
except OutputGuardrailTripwireTriggered:
    st.write("Cant show you that answer.")
    st.session_state["text_placeholder"].empty()  # Xóa văn bản đã hiển thị
```

- `text_placeholder.empty()` xóa response không phù hợp đã hiển thị trên màn hình trong quá trình streaming

### Điểm thực hành

1. Thêm guardrail đầu ra cho agent thanh toán để đảm bảo không chứa thông tin kỹ thuật
2. Thêm danh sách từ khóa cho phép/cấm vào chỉ dẫn của agent guardrail đầu ra
3. Triển khai tính năng ghi `reason` vào log khi guardrail được kích hoạt

---

## 9.8 Voice Agent I

### Chủ đề và mục tiêu

Chuyển đổi giao diện chat dựa trên văn bản sang giao diện **nhập liệu bằng giọng nói**. Xây dựng pipeline ghi âm giọng nói bằng `st.audio_input` của Streamlit, chuyển đổi audio WAV thành mảng NumPy, và truyền dưới dạng `AudioInput` cho OpenAI Agents SDK.

### Giải thích khái niệm cốt lõi

Voice agent hoạt động theo các bước sau:
1. **Ghi âm**: Ghi âm trực tiếp từ trình duyệt bằng widget `st.audio_input` của Streamlit
2. **Chuyển đổi audio**: Chuyển file WAV thành mảng NumPy `int16`
3. **Tạo AudioInput**: Wrap mảng đã chuyển đổi dưới dạng `AudioInput(buffer=array)`
4. **Thực thi agent**: Truyền dữ liệu giọng nói cho agent

### Phân tích code

**Thêm dependency mới (`pyproject.toml`)**

```toml
dependencies = [
    "numpy>=2.3.2",
    "openai-agents[voice]>=0.2.8",
    "python-dotenv>=1.1.1",
    "sounddevice>=0.5.2",
    "streamlit>=1.48.1",
]
```

- `numpy`: Xử lý dữ liệu audio dưới dạng mảng
- `sounddevice`: Dùng cho output (phát) audio

**Hàm chuyển đổi audio (`main.py`)**

```python
from agents.voice import AudioInput
import numpy as np
import wave, io

def convert_audio(audio_input):
    # Chuyển audio input Streamlit thành bytes
    audio_data = audio_input.getvalue()

    # Parse thành file WAV và trích xuất frames
    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        audio_frames = wav_file.readframes(-1)  # -1 là tất cả frames

    # Chuyển đổi thành mảng NumPy int16
    return np.frombuffer(
        audio_frames,
        dtype=np.int16,
    )
```

**Nguyên lý hoạt động:**
1. `audio_input.getvalue()`: Trích xuất dữ liệu nhị phân từ đối tượng `UploadedFile` của Streamlit
2. `wave.open(io.BytesIO(...))`: Mở dữ liệu bytes thành file WAV trong bộ nhớ
3. `wav_file.readframes(-1)`: Đọc tất cả audio frames (dữ liệu PCM thô)
4. `np.frombuffer(..., dtype=np.int16)`: Chuyển đổi dữ liệu PCM thành mảng số nguyên 16-bit

**Chuyển đổi UI nhập giọng nói:**

```python
# Sử dụng nhập giọng nói thay vì nhập văn bản
audio_input = st.audio_input(
    "Record your message",
)

if audio_input:
    with st.chat_message("human"):
        st.audio(audio_input)  # Hiển thị audio đã ghi có thể phát lại
    asyncio.run(run_agent(audio_input))
```

### Điểm thực hành

1. Kiểm tra cấu trúc file WAV (header, số kênh, sample rate) bằng module `wave`
2. In shape và dtype của mảng audio để hiểu dạng dữ liệu
3. So sánh `st.audio(audio_input)` với mảng đã chuyển đổi trực tiếp để xác minh chuyển đổi đúng

---

## 9.9 Voice Agent II

### Chủ đề và mục tiêu

Triển khai `VoicePipeline` và `VoiceWorkflowBase` tùy chỉnh để hoàn thành toàn bộ pipeline **nhập giọng nói -> chuyển đổi văn bản -> xử lý agent -> xuất giọng nói**. Triển khai cả xuất giọng nói thời gian thực sử dụng `sounddevice`.

### Giải thích khái niệm cốt lõi

**VoicePipeline** là pipeline xử lý giọng nói do OpenAI Agents SDK cung cấp, tự động xử lý các bước sau:
1. **STT (Speech-to-Text)**: Chuyển audio thành văn bản
2. **Thực thi workflow**: Truyền văn bản đã chuyển đổi cho agent để tạo response
3. **TTS (Text-to-Speech)**: Chuyển response agent thành giọng nói

Kế thừa **VoiceWorkflowBase** để định nghĩa workflow tùy chỉnh, cho phép tự do tùy chỉnh logic thực thi agent.

### Phân tích code

**Workflow tùy chỉnh (`workflow.py`)**

```python
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
from agents import Runner
import streamlit as st

class CustomWorkflow(VoiceWorkflowBase):

    def __init__(self, context):
        self.context = context

    async def run(self, transcription):
        # Nhận văn bản đã chuyển đổi từ STT (transcription) và thực thi agent
        result = Runner.run_streamed(
            st.session_state["agent"],
            transcription,
            session=st.session_state["session"],
            context=self.context,
        )

        # Streaming response agent theo đơn vị text chunk
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Cập nhật agent hoạt động cuối cùng vì handoff có thể đã xảy ra
        st.session_state["agent"] = result.last_agent
```

**Điểm cốt lõi:**
- Kế thừa `VoiceWorkflowBase` và triển khai phương thức `run()`
- Phương thức `run()` được định nghĩa là **async generator** (sử dụng `yield`)
- `transcription`: Kết quả chuyển đổi giọng nói thành văn bản (kết quả STT)
- `VoiceWorkflowHelper.stream_text_from()`: Tiện ích trích xuất text chunk từ kết quả `Runner`
- `result.last_agent`: Trả về agent cuối cùng được kích hoạt nếu handoff đã xảy ra. Lưu vào `session_state` để agent đúng được sử dụng cho input giọng nói tiếp theo

**Tích hợp VoicePipeline (`main.py`)**

```python
from agents.voice import AudioInput, VoicePipeline
from workflow import CustomWorkflow
import sounddevice as sd

async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            # 1. Chuyển đổi audio
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)

            # 2. Tạo workflow tùy chỉnh
            workflow = CustomWorkflow(context=user_account_ctx)

            # 3. Tạo và thực thi voice pipeline
            pipeline = VoicePipeline(workflow=workflow)

            status_container.update(label="Running workflow", state="running")

            result = await pipeline.run(audio)

            # 4. Thiết lập audio output stream
            player = sd.OutputStream(
                samplerate=24000,  # Sample rate 24kHz
                channels=1,        # Audio mono
                dtype=np.int16,    # Số nguyên 16-bit
            )
            player.start()

            status_container.update(state="complete")

            # 5. Phát voice response thời gian thực
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**Thứ tự hoạt động voice pipeline:**
1. `convert_audio()`: WAV -> mảng NumPy
2. `AudioInput(buffer=audio_array)`: Mảng -> đối tượng AudioInput
3. `CustomWorkflow(context=...)`: Tạo workflow bao gồm context
4. `VoicePipeline(workflow=workflow)`: Tạo pipeline
5. `pipeline.run(audio)`: STT -> thực thi agent -> TTS (bất đồng bộ)
6. `result.stream()`: Streaming kết quả TTS theo đơn vị chunk
7. `player.write(event.data)`: Xuất mỗi audio chunk ra loa

**Sửa đổi agent phân loại (`my_agents/triage_agent.py`)**

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

**Thay đổi:**
- Thêm `"SPEAK TO THE USER IN ENGLISH"` ở đầu chỉ dẫn để chỉ định rõ ngôn ngữ TTS output
- Guardrail đầu vào bị comment -- vì kết quả chuyển đổi STT từ nhập giọng nói có thể không phù hợp cho agent guardrail

### Điểm thực hành

1. Thay đổi `samplerate` để kiểm tra sự khác biệt chất lượng âm thanh (16000, 24000, 48000)
2. Log giá trị `transcription` trong `CustomWorkflow.run()` để kiểm tra chất lượng chuyển đổi STT
3. Thêm tính năng lưu TTS response thành file để phát lại sau
4. Kiểm tra handoff có hoạt động bình thường trong voice agent không

---

## Tổng kết cốt lõi chương

### 1. Mẫu kiến trúc

Hệ thống triển khai trong chương này tuân theo kiến trúc multi-agent phân cấp sau:

```
Đầu vào người dùng
    |
    v
[Guardrail đầu vào] -- Không phù hợp --> Tin nhắn chặn
    |
    v (Thông qua)
[Agent phân loại] -- Phân loại --> Handoff
    |           |           |           |
    v           v           v           v
[Hỗ trợ KT]  [Thanh toán]  [Đơn hàng]  [Tài khoản]
   |           |           |           |
   v           v           v           v
[Guardrail đầu ra] -- Không phù hợp --> Tin nhắn chặn
    |
    v (Thông qua)
Response cho người dùng
```

### 2. Tóm tắt thành phần SDK cốt lõi

| Thành phần | Vai trò | Phần sử dụng |
|------------|---------|--------------|
| `Agent` | Định nghĩa agent (chỉ dẫn, công cụ, guardrail) | Toàn bộ |
| `Runner.run_streamed()` | Thực thi agent streaming | Toàn bộ |
| `SQLiteSession` | Lưu trữ vĩnh viễn lịch sử hội thoại | 9.0 |
| `RunContextWrapper` | Truy cập context thực thi | 9.1+ |
| `@function_tool` | Định nghĩa hàm công cụ | 9.1, 9.6 |
| `@input_guardrail` | Xác minh đầu vào | 9.3 |
| `@output_guardrail` | Xác minh đầu ra | 9.7 |
| `handoff()` | Chuyển đổi giữa agent | 9.4 |
| `AgentHooks` | Callback lifecycle | 9.6 |
| `VoicePipeline` | Pipeline xử lý giọng nói | 9.9 |
| `VoiceWorkflowBase` | Workflow giọng nói tùy chỉnh | 9.9 |

### 3. Nguyên tắc thiết kế Guardrail

- **Guardrail đầu vào**: Chặn yêu cầu không phù hợp trước khi agent xử lý (tiết kiệm chi phí, bảo mật)
- **Guardrail đầu ra**: Chặn nội dung không phù hợp sau khi agent response (đảm bảo chất lượng, ngăn rò rỉ dữ liệu)
- Tách agent guardrail thành agent nhẹ riêng biệt để phân tách mối quan tâm
- Chỉ định mô hình Pydantic cho `output_type` để ép buộc kết quả đánh giá có cấu trúc

### 4. Nguyên tắc thiết kế Handoff

- Mỗi agent chuyên biệt có phạm vi trách nhiệm rõ ràng
- Dọn dẹp lịch sử công cụ của agent trước bằng `handoff_filters.remove_all_tools`
- Log metadata handoff bằng callback `on_handoff`
- Theo dõi agent đang hoạt động hiện tại bằng `result.last_agent` hoặc `agent_updated_stream_event`

---

## Bài tập thực hành

### Bài 1: Thêm Agent chuyên biệt mới (Độ khó: Trung bình)

**Mục tiêu**: Thêm agent chuyên hoàn trả.

**Yêu cầu**:
- Tạo file `my_agents/refund_agent.py`
- Định nghĩa chỉ dẫn động và hướng dẫn quyền lợi bổ sung cho khách hàng premium
- Thêm ít nhất 2 hàm công cụ liên quan đến hoàn trả vào `tools.py`
- Kết nối `AgentToolUsageLoggingHooks`
- Thêm vào danh sách handoff của agent phân loại
- Thêm mục liên quan đến hoàn trả vào hướng dẫn phân loại của agent phân loại

### Bài 2: Mở rộng Guardrail Đầu ra (Độ khó: Trung bình)

**Mục tiêu**: Thêm guardrail đầu ra cho tất cả agent chuyên biệt.

**Yêu cầu**:
- Agent thanh toán: Xác minh không chứa thông tin hỗ trợ kỹ thuật
- Agent đơn hàng: Xác minh không chứa thông tin thanh toán hoặc tài khoản
- Agent tài khoản: Xác minh không chứa thông tin đơn hàng hoặc thanh toán
- Định nghĩa mô hình Pydantic output cho mỗi guardrail trong `models.py`

### Bài 3: Hệ thống Hook tùy chỉnh (Độ khó: Cao)

**Mục tiêu**: Triển khai hệ thống hook nâng cao thu thập thống kê sử dụng agent.

**Yêu cầu**:
- Theo dõi số lần gọi, thời gian response trung bình, tần suất sử dụng công cụ của mỗi agent
- Hiển thị dữ liệu thống kê theo thời gian thực trong sidebar
- Khởi tạo lại thống kê khi session được khởi tạo lại
- Ghi lại thời gian bắt đầu trong `on_start` và tính thời gian thực hiện trong `on_end`

### Bài 4: Voice Agent hai chiều (Độ khó: Cao)

**Mục tiêu**: Triển khai agent cho phép hội thoại giọng nói liên tục.

**Yêu cầu**:
- Tự động bắt đầu ghi âm tiếp theo khi phát voice response kết thúc
- Hướng dẫn bằng giọng nói agent nào đã chuyển đổi khi handoff xảy ra trong hội thoại
- Hiển thị lịch sử hội thoại dưới dạng văn bản trong sidebar (cả kết quả STT và response agent)
- Thêm tính năng nhận diện lệnh "kết thúc hội thoại" bằng giọng nói để kết thúc session

### Bài 5: Cơ chế quay lại giữa các Agent (Độ khó: Cao)

**Mục tiêu**: Triển khai cơ chế quay lại từ agent chuyên biệt về agent phân loại.

**Yêu cầu**:
- Thêm handoff "quay lại triage" cho mỗi agent chuyên biệt
- Tự động quay lại triage khi agent chuyên biệt nhận yêu cầu ngoài phạm vi công việc
- Bao gồm tóm tắt hội thoại hiện tại trong dữ liệu handoff khi quay lại
- Hiển thị sự kiện quay lại trực quan trên UI
