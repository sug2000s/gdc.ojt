# Chương 9: OpenAI Agents SDK - Guardrails, Handoffs và Voice Agents

---

## Tổng quan chương

Trong chương này, chúng ta sử dụng OpenAI Agents SDK (`openai-agents`) để từng bước xây dựng **hệ thống agent hỗ trợ khách hàng** ở cấp độ sản xuất. Vượt ra ngoài chatbot đơn giản, chúng ta hoàn thành một dự án toàn diện bao gồm quản lý ngữ cảnh, chỉ thị động, guardrails đầu vào/đầu ra, handoff giữa các agent, hooks vòng đời và voice agent.

### Mục tiêu học tập

| Phần | Chủ đề | Từ khóa chính |
|------|------|-------------|
| 9.0 | Giới thiệu dự án và cấu trúc cơ bản | Streamlit, SQLiteSession, Runner |
| 9.1 | Quản lý ngữ cảnh | RunContextWrapper, Pydantic Model |
| 9.2 | Chỉ thị động | Dynamic Instructions, Prompt dựa trên hàm |
| 9.3 | Input Guardrails | Input Guardrail, Tripwire |
| 9.4 | Agent Handoffs | Handoff, Định tuyến agent chuyên biệt |
| 9.5 | Handoff UI | agent_updated_stream_event, Hiển thị chuyển đổi thời gian thực |
| 9.6 | Hooks | AgentHooks, Ghi nhật ký sử dụng công cụ |
| 9.7 | Output Guardrails | Output Guardrail, Xác thực phản hồi |
| 9.8 | Voice Agent I | AudioInput, Chuyển đổi WAV |
| 9.9 | Voice Agent II | VoicePipeline, VoiceWorkflowBase, sounddevice |

### Cấu trúc dự án (Cuối cùng)

```
customer-support-agent/
├── main.py                      # Ứng dụng chính Streamlit
├── models.py                    # Mô hình dữ liệu Pydantic
├── tools.py                     # Các hàm công cụ agent
├── output_guardrails.py         # Định nghĩa output guardrail
├── workflow.py                  # Workflow tùy chỉnh voice agent
├── my_agents/
│   ├── triage_agent.py          # Agent phân loại (triage)
│   ├── technical_agent.py       # Agent hỗ trợ kỹ thuật
│   ├── billing_agent.py         # Agent hỗ trợ thanh toán
│   ├── order_agent.py           # Agent quản lý đơn hàng
│   └── account_agent.py         # Agent quản lý tài khoản
├── pyproject.toml               # Phụ thuộc dự án
└── customer-support-memory.db   # Kho lưu trữ phiên SQLite
```

---

## 9.0 Giới thiệu dự án và cấu trúc cơ bản

### Chủ đề và mục tiêu

Kết hợp OpenAI Agents SDK với Streamlit để tạo khung cơ bản của chatbot hỗ trợ khách hàng. Thiết lập cấu trúc lưu trữ lịch sử hội thoại trong SQLite và hiển thị phản hồi streaming theo thời gian thực.

### Khái niệm chính

**OpenAI Agents SDK** là framework phát triển ứng dụng dựa trên agent. Nó chạy agent thông qua `Runner` và có thể lưu trữ vĩnh viễn lịch sử hội thoại bằng `SQLiteSession`. Streamlit là framework giao diện web cho prototyping nhanh, cho phép triển khai dễ dàng giao diện chat bằng `st.chat_message` và `st.chat_input`.

### Phân tích mã

**Thiết lập phụ thuộc dự án (`pyproject.toml`)**

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
- `python-dotenv`: Tải biến môi trường như API key từ tệp `.env`
- `streamlit`: Framework giao diện web

**Ứng dụng chính (`main.py`)**

```python
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import streamlit as st
from agents import Runner, SQLiteSession

client = OpenAI()

# Quản lý phiên dựa trên SQLite
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]
```

**Điểm chính:**
- `SQLiteSession` nhận hai tham số: tên phiên (`"chat-history"`) và đường dẫn tệp cơ sở dữ liệu (`"customer-support-memory.db"`)
- `st.session_state` được sử dụng để đảm bảo đối tượng phiên được duy trì qua các lần re-render của Streamlit

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

- `session.get_items()` lấy tất cả tin nhắn đã lưu
- Lý do escape ký hiệu `$` bằng `\$` là để ngăn Streamlit hiểu nó như công thức LaTeX

**Thực thi streaming agent:**

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

- `Runner.run_streamed()` chạy agent ở chế độ streaming, tạo sự kiện khi mỗi token được sinh ra
- Trong loại `raw_response_event`, phát hiện `response.output_text.delta` để hiển thị văn bản thời gian thực bằng cách tích lũy
- `st.empty()` tạo placeholder có thể cập nhật nội dung sau

### Điểm thực hành

1. Khởi tạo dự án bằng trình quản lý gói `uv` và cài đặt phụ thuộc
2. Chạy ứng dụng bằng `streamlit run main.py` và xác minh giao diện chat cơ bản
3. Mở tệp DB SQLite và kiểm tra cách lịch sử hội thoại được lưu trữ

---

## 9.1 Quản lý ngữ cảnh (Context Management)

### Chủ đề và mục tiêu

Học cách truyền **thông tin ngữ cảnh người dùng** khi thực thi agent. Định nghĩa ngữ cảnh an toàn kiểu với Pydantic model và học mẫu truy cập thông tin này trong các hàm công cụ thông qua `RunContextWrapper`.

### Khái niệm chính

**Ngữ cảnh (Context)** là thông tin bên ngoài mà agent có thể tham chiếu trong quá trình thực thi. Ví dụ: ID, tên, cấp đăng ký của người dùng đang đăng nhập. Thông tin này được sử dụng trong prompt và hàm công cụ của agent.

`RunContextWrapper` là kiểu generic. Bằng cách chỉ định kiểu ngữ cảnh như `RunContextWrapper[UserAccountContext]`, bạn có thể nhận hỗ trợ tự động hoàn thành IDE và kiểm tra kiểu.

### Phân tích mã

**Định nghĩa mô hình ngữ cảnh (`models.py`)**

```python
from pydantic import BaseModel

class UserAccountContext(BaseModel):
    customer_id: int
    name: str
    tier: str = "basic"  # premium, enterprise
```

- Kế thừa `BaseModel` của Pydantic để tự động xử lý xác thực dữ liệu và tuần tự hóa
- Trường `tier` có giá trị mặc định `"basic"`, biến nó thành trường tùy chọn

**Sử dụng ngữ cảnh trong hàm công cụ (`main.py`)**

```python
from agents import Runner, SQLiteSession, function_tool, RunContextWrapper
from models import UserAccountContext

@function_tool
def get_user_tier(wrapper: RunContextWrapper[UserAccountContext]):
    return (
        f"The user {wrapper.context.customer_id} has a {wrapper.context.tier} account."
    )
```

- Decorator `@function_tool` chuyển đổi hàm Python thường thành công cụ mà agent có thể gọi
- Truy cập instance `UserAccountContext` được truyền lúc chạy thông qua `wrapper.context`

**Tạo và truyền ngữ cảnh:**

```python
user_account_ctx = UserAccountContext(
    customer_id=1,
    name="nico",
    tier="basic",
)

# Truyền ngữ cảnh khi chạy Runner
stream = Runner.run_streamed(
    agent,
    message,
    session=session,
    context=user_account_ctx,  # Tiêm ngữ cảnh
)
```

- Truyền đối tượng ngữ cảnh qua tham số `context` của `Runner.run_streamed()`
- Ngữ cảnh này có thể truy cập từ tất cả hàm công cụ và chỉ thị của agent

### Điểm thực hành

1. Thêm trường mới như `phone_number`, `preferred_language` vào `UserAccountContext`
2. Tạo `@function_tool` mới sử dụng thông tin ngữ cảnh
3. Triển khai công cụ trả về phản hồi khác nhau dựa trên giá trị `tier`

---

## 9.2 Chỉ thị động (Dynamic Instructions)

### Chủ đề và mục tiêu

Học cách định nghĩa chỉ thị (instructions) của agent không phải là **chuỗi tĩnh** mà là **hàm**, tạo prompt thay đổi động dựa trên ngữ cảnh tại thời điểm thực thi.

### Khái niệm chính

Thông thường, `instructions` của agent là chuỗi cố định. Tuy nhiên, để đưa thông tin runtime như tên, cấp độ, email của người dùng vào prompt, cần chỉ thị động dạng hàm. Hàm này nhận đối tượng `RunContextWrapper` và `Agent` làm tham số và trả về chuỗi.

### Phân tích mã

**Tạo Triage Agent (`my_agents/triage_agent.py`)**

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
    instructions=dynamic_triage_agent_instructions,  # Truyền hàm trực tiếp
)
```

**Điểm chính:**
- **Tham chiếu hàm** được truyền cho tham số `instructions` thay vì chuỗi (không có ngoặc: `dynamic_triage_agent_instructions`)
- Chữ ký hàm phải là `(wrapper: RunContextWrapper[T], agent: Agent[T]) -> str`
- f-string được sử dụng để chèn giá trị ngữ cảnh như `wrapper.context.name`, `wrapper.context.tier` vào prompt
- Agent Triage có vai trò phân loại yêu cầu của khách hàng và định tuyến đến agent chuyên biệt phù hợp

**Thay đổi trong main.py:**

Trong commit này, hàm công cụ `get_user_tier` trước đây trong `main.py` đã bị xóa. Cách tiếp cận đã chuyển từ truy cập thông tin ngữ cảnh qua công cụ sang xử lý trực tiếp trong chỉ thị động.

### Điểm thực hành

1. Sửa đổi chỉ thị động để phản hồi với giọng điệu khác nhau (trang trọng/thân mật) dựa trên giá trị `wrapper.context.tier`
2. Đưa thời gian hiện tại (`datetime.now()`) vào chỉ thị để thêm lời chào dựa trên thời gian
3. Viết chỉ thị động sử dụng thuộc tính của đối tượng agent (`agent`)

---

## 9.3 Input Guardrails

### Chủ đề và mục tiêu

Triển khai input guardrails **tự động kiểm tra** xem đầu vào của người dùng có nằm ngoài phạm vi công việc của agent không. Học mẫu trong đó "guardrail agent" riêng biệt phân tích đầu vào và chặn hội thoại nếu yêu cầu không phù hợp.

### Khái niệm chính

**Input Guardrail** là bước xác thực chạy trước khi agent xử lý đầu vào người dùng. Trong quá trình này, một agent nhỏ riêng biệt (guardrail agent) phân tích đầu vào để xác định xem nó có "lạc đề (off-topic)" không. Nếu phát hiện đầu vào không phù hợp, **Tripwire** kích hoạt ngoại lệ `InputGuardrailTripwireTriggered`.

Ưu điểm của mẫu này:
- Không cần làm phức tạp chỉ thị của agent chính
- Kiểm tra guardrail được **thực thi bất đồng bộ song song**, giảm thiểu suy giảm hiệu suất
- Logic xác thực được tách thành module riêng để tái sử dụng và kiểm thử dễ dàng

### Phân tích mã

**Mô hình đầu ra Guardrail (`models.py`)**

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

- `InputGuardRailOutput` là định dạng đầu ra có cấu trúc của guardrail agent
- `is_off_topic`: Yêu cầu có nằm ngoài phạm vi công việc không
- `reason`: Cơ sở phán đoán (để gỡ lỗi và ghi nhật ký)

**Guardrail Agent và Decorator (`my_agents/triage_agent.py`)**

```python
from agents import (
    Agent, RunContextWrapper, input_guardrail,
    Runner, GuardrailFunctionOutput,
)
from models import UserAccountContext, InputGuardRailOutput

# 1. Định nghĩa agent guardrail chuyên dụng
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
    output_type=InputGuardRailOutput,  # Bắt buộc đầu ra có cấu trúc
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

# 3. Kết nối guardrail với triage agent
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
2. Hàm `off_topic_guardrail` thực thi
3. Bên trong, `input_guardrail_agent` phân tích tin nhắn
4. Trả về kết quả ở dạng `InputGuardRailOutput`
5. Nếu `is_off_topic` là `True`, `tripwire_triggered=True` được đặt
6. Khi Tripwire kích hoạt, ngoại lệ `InputGuardrailTripwireTriggered` được phát sinh

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

- `try/except` bắt ngoại lệ `InputGuardrailTripwireTriggered` và hiển thị thông báo phù hợp cho người dùng

### Điểm thực hành

1. Sửa đổi chỉ thị của guardrail agent để áp dụng bộ lọc nghiêm ngặt hơn/lỏng hơn
2. Kiểm thử guardrail với tin nhắn lạc đề như "Thời tiết hôm nay thế nào?" và tin nhắn hợp lệ như "Tôi muốn đổi mật khẩu"
3. Thêm chức năng hiển thị trường `reason` trên giao diện để giải thích lý do bị chặn

---

## 9.4 Agent Handoffs

### Chủ đề và mục tiêu

Triển khai cấu trúc đa agent trong đó triage agent phân loại yêu cầu khách hàng và chuyển giao (handoff) hội thoại cho **agent chuyên biệt** phù hợp. Tạo 4 agent chuyên biệt (hỗ trợ kỹ thuật, thanh toán, đơn hàng, tài khoản) và thiết lập cơ chế handoff.

### Khái niệm chính

**Handoff** là khi một agent chuyển quyền kiểm soát hội thoại cho agent khác. Tương tự như nhân viên tổng đài chuyển cuộc gọi đến bộ phận chuyên biệt.

Trong OpenAI Agents SDK, handoff bao gồm các yếu tố sau:
- Hàm `handoff()`: Định nghĩa cấu hình handoff
- `on_handoff`: Hàm callback thực thi khi handoff xảy ra
- `input_type`: Schema của dữ liệu được truyền trong handoff
- `input_filter`: Bộ lọc dọn dẹp lịch sử gọi công cụ của agent trước đó

### Phân tích mã

**Mô hình dữ liệu Handoff (`models.py`)**

```python
class HandoffData(BaseModel):
    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
```

Mô hình này định nghĩa metadata mà triage agent truyền cho agent chuyên biệt trong handoff.

**Ví dụ Agent chuyên biệt - Hỗ trợ kỹ thuật (`my_agents/technical_agent.py`)**

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

- Tất cả agent chuyên biệt tuân theo cùng mẫu: chỉ thị động + tạo `Agent`
- Quyền lợi bổ sung được thông báo cho khách hàng cao cấp dựa trên `wrapper.context.tier`

**Cấu hình Handoff (`my_agents/triage_agent.py`)**

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

# Callback handoff: Hiển thị thông tin handoff trong thanh bên
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

**Điểm chính:**
- `RECOMMENDED_PROMPT_PREFIX`: Tiền tố prompt khuyến nghị do OpenAI cung cấp cho handoff, hướng dẫn agent cách thực hiện handoff
- `handoff_filters.remove_all_tools`: Xóa lịch sử gọi công cụ của agent trước đó trong handoff để agent mới bắt đầu với trạng thái sạch
- Hàm factory `make_handoff()` giảm mã trùng lặp
- Handoff cũng có thể được triển khai bằng cách chuyển đổi agent thành công cụ sử dụng phương thức `as_tool()`

### Điểm thực hành

1. Thêm agent chuyên biệt mới (ví dụ: "Agent chuyên hoàn trả") và kết nối handoff
2. Thay thế `input_filter` bằng bộ lọc tùy chỉnh thay vì `handoff_filters.remove_all_tools`
3. Thử nghiệm sự khác biệt giữa cách tiếp cận `as_tool()` và `handoff()`

---

## 9.5 Handoff UI

### Chủ đề và mục tiêu

Hiển thị **trạng thái chuyển đổi thời gian thực trên giao diện** khi handoff giữa các agent xảy ra, và theo dõi agent đang hoạt động hiện tại để tin nhắn tiếp theo được gửi đến đúng agent.

### Khái niệm chính

Trong các sự kiện streaming, `agent_updated_stream_event` được phát khi agent thay đổi. Bằng cách phát hiện điều này và hiển thị thông báo chuyển đổi trên giao diện đồng thời lưu agent hiện tại vào `st.session_state`, tin nhắn tiếp theo của người dùng được gửi đến đúng agent chuyên biệt.

### Phân tích mã

**Theo dõi trạng thái Agent (`main.py`)**

```python
# Lưu agent đang hoạt động hiện tại trong session state
if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent
```

**Phát hiện Handoff trong sự kiện Streaming:**

```python
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""
        st.session_state["text_placeholder"] = text_placeholder
        try:
            stream = Runner.run_streamed(
                st.session_state["agent"],  # Sử dụng agent đang hoạt động
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
                        # Cập nhật agent hiện tại sang agent mới
                        st.session_state["agent"] = event.new_agent
                        # Khởi tạo placeholder cho phản hồi của agent mới
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder
                        response = ""

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
```

**Điểm chính:**
- `agent_updated_stream_event`: Sự kiện streaming phát khi agent thay đổi
- `event.new_agent`: Đối tượng agent mới được kích hoạt
- Khi agent thay đổi, `response` được đặt lại và `text_placeholder` mới được tạo để phản hồi của agent mới hiển thị từ đầu
- Cập nhật `st.session_state["agent"]` đảm bảo tin nhắn tiếp theo được gửi đến agent mới

### Điểm thực hành

1. Cải thiện kiểu thông báo chuyển đổi trong handoff (ví dụ: thêm đường phân cách)
2. Thêm chức năng luôn hiển thị tên agent đang hoạt động trong thanh bên
3. Tạo nút "Quay lại Triage" để đặt lại agent thủ công

---

## 9.6 Hooks

### Chủ đề và mục tiêu

Triển khai **AgentHooks** chèn logic tùy chỉnh vào **sự kiện vòng đời** của agent (bắt đầu, kết thúc, thực thi công cụ, handoff). Đồng thời thêm công cụ nghiệp vụ thực tế cho mỗi agent chuyên biệt.

### Khái niệm chính

**Hooks** là tập hợp các hàm callback được gọi tự động tại các thời điểm cụ thể trong quá trình thực thi agent. Bằng cách kế thừa lớp `AgentHooks`, bạn có thể ghi đè các phương thức sau:

| Phương thức | Thời điểm kích hoạt |
|--------|-----------|
| `on_start` | Bắt đầu thực thi agent |
| `on_end` | Hoàn thành thực thi agent |
| `on_tool_start` | Ngay trước khi hàm công cụ được gọi |
| `on_tool_end` | Sau khi hàm công cụ hoàn thành |
| `on_handoff` | Khi handoff đến agent khác xảy ra |

### Phân tích mã

**Ví dụ hàm công cụ (`tools.py`)**

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

**Công cụ được gán cho mỗi Agent chuyên biệt:**

| Agent | Công cụ |
|----------|------|
| Technical | `run_diagnostic_check`, `provide_troubleshooting_steps`, `escalate_to_engineering` |
| Billing | `lookup_billing_history`, `process_refund_request`, `update_payment_method`, `apply_billing_credit` |
| Order | `lookup_order_status`, `initiate_return_process`, `schedule_redelivery`, `expedite_shipping` |
| Account | `reset_user_password`, `enable_two_factor_auth`, `update_account_email`, `deactivate_account`, `export_account_data` |

### Điểm thực hành

1. Ghi thời gian bắt đầu thực thi công cụ trong `on_tool_start` và tính thời gian đã trôi qua trong `on_tool_end`
2. Thêm hook hiển thị thông báo xác nhận khi công cụ cụ thể (ví dụ: `deactivate_account`) được gọi
3. Triển khai hook lưu lịch sử sử dụng công cụ vào tệp nhật ký

---

## 9.7 Output Guardrails

### Chủ đề và mục tiêu

Triển khai output guardrails xác minh xem **phản hồi** của agent có chứa nội dung ngoài phạm vi công việc của agent đó không. Mặc dù có cấu trúc đối xứng với input guardrails, điểm khác biệt là nó xác minh đầu ra cuối cùng của agent.

### Khái niệm chính

**Output Guardrail** là bước xác minh xem phản hồi của agent có phù hợp không sau khi nó đã được tạo. Ví dụ, nếu agent hỗ trợ kỹ thuật tạo phản hồi chứa thông tin thanh toán hoặc quản lý tài khoản, điều này nằm ngoài lĩnh vực của nó và cần bị chặn.

Khi ngoại lệ `OutputGuardrailTripwireTriggered` được phát sinh, văn bản đã hiển thị qua streaming bị xóa và thay bằng thông báo thay thế.

### Phân tích mã

**Mô hình Output Guardrail (`models.py`)**

```python
class TechnicalOutputGuardRailOutput(BaseModel):
    contains_off_topic: bool
    contains_billing_data: bool
    contains_account_data: bool
    reason: str
```

**Định nghĩa Output Guardrail (`output_guardrails.py`)**

```python
from agents import (
    Agent, output_guardrail, Runner,
    RunContextWrapper, GuardrailFunctionOutput,
)
from models import TechnicalOutputGuardRailOutput, UserAccountContext

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

**So sánh Input Guardrails và Output Guardrails:**

| Mục | Input Guardrail | Output Guardrail |
|------|--------------|--------------|
| Decorator | `@input_guardrail` | `@output_guardrail` |
| Đối tượng kiểm tra | Tin nhắn người dùng | Phản hồi agent |
| Tham số thứ ba | `input: str` | `output: str` |
| Loại ngoại lệ | `InputGuardrailTripwireTriggered` | `OutputGuardrailTripwireTriggered` |
| Vị trí áp dụng | `input_guardrails=[]` | `output_guardrails=[]` |

### Điểm thực hành

1. Thêm output guardrails cho billing agent để xác minh thông tin kỹ thuật không bị bao gồm
2. Thêm danh sách từ khóa cho phép/cấm vào chỉ thị của output guardrail agent
3. Triển khai chức năng ghi nhật ký `reason` khi guardrail kích hoạt

---

## 9.8 Voice Agent I

### Chủ đề và mục tiêu

Chuyển đổi giao diện chat văn bản sang giao diện **nhập liệu bằng giọng nói**. Xây dựng pipeline ghi âm bằng `st.audio_input` của Streamlit, chuyển đổi âm thanh WAV thành mảng NumPy và truyền cho `AudioInput` của OpenAI Agents SDK.

### Khái niệm chính

Voice agent hoạt động theo các giai đoạn sau:
1. **Ghi âm**: Ghi trực tiếp trong trình duyệt bằng widget `st.audio_input` của Streamlit
2. **Chuyển đổi âm thanh**: Chuyển đổi tệp WAV thành mảng NumPy `int16`
3. **Tạo AudioInput**: Bọc mảng đã chuyển đổi ở dạng `AudioInput(buffer=array)`
4. **Thực thi Agent**: Truyền dữ liệu giọng nói cho agent

### Phân tích mã

**Hàm chuyển đổi âm thanh (`main.py`)**

```python
from agents.voice import AudioInput
import numpy as np
import wave, io

def convert_audio(audio_input):
    # Chuyển đổi đầu vào âm thanh Streamlit thành bytes
    audio_data = audio_input.getvalue()

    # Phân tích như tệp WAV và trích xuất frames
    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        audio_frames = wav_file.readframes(-1)  # -1 nghĩa là tất cả frames

    # Chuyển đổi thành mảng NumPy int16
    return np.frombuffer(
        audio_frames,
        dtype=np.int16,
    )
```

**Cách hoạt động:**
1. `audio_input.getvalue()`: Trích xuất dữ liệu nhị phân từ đối tượng `UploadedFile` của Streamlit
2. `wave.open(io.BytesIO(...))`: Mở dữ liệu byte dưới dạng tệp WAV trong bộ nhớ
3. `wav_file.readframes(-1)`: Đọc tất cả audio frames (dữ liệu PCM thô)
4. `np.frombuffer(..., dtype=np.int16)`: Chuyển đổi dữ liệu PCM thành mảng số nguyên 16-bit

### Điểm thực hành

1. Kiểm tra cấu trúc tệp WAV (header, số kênh, sample rate) bằng module `wave`
2. In shape và dtype của mảng âm thanh để hiểu định dạng dữ liệu
3. So sánh `st.audio(audio_input)` với mảng đã chuyển đổi trực tiếp để xác minh chuyển đổi đúng

---

## 9.9 Voice Agent II

### Chủ đề và mục tiêu

Triển khai `VoicePipeline` và `VoiceWorkflowBase` tùy chỉnh để hoàn thành toàn bộ pipeline: **nhập giọng nói -> chuyển đổi văn bản -> xử lý agent -> xuất giọng nói**. Cũng triển khai xuất giọng nói thời gian thực bằng `sounddevice`.

### Khái niệm chính

**VoicePipeline** là pipeline xử lý giọng nói do OpenAI Agents SDK cung cấp, tự động xử lý các giai đoạn sau:
1. **STT (Speech-to-Text)**: Chuyển đổi âm thanh thành văn bản
2. **Thực thi Workflow**: Truyền văn bản đã chuyển đổi cho agent để tạo phản hồi
3. **TTS (Text-to-Speech)**: Chuyển đổi phản hồi agent thành giọng nói

Bằng cách kế thừa **VoiceWorkflowBase** và định nghĩa workflow tùy chỉnh, bạn có thể tự do tùy chỉnh logic thực thi agent.

### Phân tích mã

**Workflow tùy chỉnh (`workflow.py`)**

```python
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
from agents import Runner
import streamlit as st

class CustomWorkflow(VoiceWorkflowBase):

    def __init__(self, context):
        self.context = context

    async def run(self, transcription):
        # Nhận văn bản đã chuyển đổi bởi STT (transcription) và chạy agent
        result = Runner.run_streamed(
            st.session_state["agent"],
            transcription,
            session=st.session_state["session"],
            context=self.context,
        )

        # Stream phản hồi agent theo từng đoạn văn bản
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # Cập nhật agent hoạt động cuối cùng vì handoff có thể đã xảy ra
        st.session_state["agent"] = result.last_agent
```

**Tích hợp VoicePipeline (`main.py`)**

```python
from agents.voice import AudioInput, VoicePipeline
from workflow import CustomWorkflow
import sounddevice as sd

async def run_agent(audio_input):
    with st.chat_message("ai"):
        status_container = st.status("Processing voice message...")
        try:
            # 1. Chuyển đổi âm thanh
            audio_array = convert_audio(audio_input)
            audio = AudioInput(buffer=audio_array)

            # 2. Tạo workflow tùy chỉnh
            workflow = CustomWorkflow(context=user_account_ctx)

            # 3. Tạo và chạy voice pipeline
            pipeline = VoicePipeline(workflow=workflow)

            status_container.update(label="Running workflow", state="running")

            result = await pipeline.run(audio)

            # 4. Thiết lập luồng xuất âm thanh
            player = sd.OutputStream(
                samplerate=24000,  # Sample rate 24kHz
                channels=1,        # Âm thanh mono
                dtype=np.int16,    # Số nguyên 16-bit
            )
            player.start()

            status_container.update(state="complete")

            # 5. Phát phản hồi giọng nói thời gian thực
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)

        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
```

**Trình tự hoạt động Voice Pipeline:**
1. `convert_audio()`: WAV -> Mảng NumPy
2. `AudioInput(buffer=audio_array)`: Mảng -> Đối tượng AudioInput
3. `CustomWorkflow(context=...)`: Tạo workflow với ngữ cảnh
4. `VoicePipeline(workflow=workflow)`: Tạo pipeline
5. `pipeline.run(audio)`: STT -> Thực thi Agent -> TTS (bất đồng bộ)
6. `result.stream()`: Stream kết quả TTS theo từng đoạn
7. `player.write(event.data)`: Xuất mỗi đoạn âm thanh ra loa

### Điểm thực hành

1. Thay đổi `samplerate` và kiểm tra sự khác biệt chất lượng âm thanh (16000, 24000, 48000)
2. Ghi nhật ký giá trị `transcription` trong `CustomWorkflow.run()` để kiểm tra chất lượng chuyển đổi STT
3. Thêm chức năng lưu phản hồi TTS vào tệp để phát lại sau
4. Kiểm thử xem handoff có hoạt động chính xác trong voice agent không

---

## Tóm tắt trọng điểm chương

### 1. Mẫu kiến trúc

```
Đầu vào người dùng
    |
    v
[Input Guardrail] -- Không phù hợp --> Thông báo chặn
    |
    v (Đạt)
[Triage Agent] -- Phân loại --> Handoff
    |           |           |           |
    v           v           v           v
[Kỹ thuật]  [Thanh toán]  [Đơn hàng]  [Tài khoản]
   |           |           |           |
   v           v           v           v
[Output Guardrail] -- Không phù hợp --> Thông báo chặn
    |
    v (Đạt)
Phản hồi cho người dùng
```

### 2. Tóm tắt thành phần SDK chính

| Thành phần | Vai trò | Phần sử dụng |
|----------|------|-----------|
| `Agent` | Định nghĩa agent (chỉ thị, công cụ, guardrails) | Tất cả |
| `Runner.run_streamed()` | Thực thi agent streaming | Tất cả |
| `SQLiteSession` | Lưu trữ vĩnh viễn lịch sử hội thoại | 9.0 |
| `RunContextWrapper` | Truy cập ngữ cảnh thực thi | 9.1+ |
| `@function_tool` | Định nghĩa hàm công cụ | 9.1, 9.6 |
| `@input_guardrail` | Xác thực đầu vào | 9.3 |
| `@output_guardrail` | Xác thực đầu ra | 9.7 |
| `handoff()` | Chuyển đổi giữa các agent | 9.4 |
| `AgentHooks` | Callback vòng đời | 9.6 |
| `VoicePipeline` | Pipeline xử lý giọng nói | 9.9 |
| `VoiceWorkflowBase` | Workflow giọng nói tùy chỉnh | 9.9 |

### 3. Nguyên tắc thiết kế Guardrail

- **Input Guardrails**: Chặn yêu cầu không phù hợp trước khi agent xử lý (tiết kiệm chi phí, bảo mật)
- **Output Guardrails**: Chặn nội dung không phù hợp sau phản hồi agent (đảm bảo chất lượng, ngăn rò rỉ dữ liệu)
- Guardrail agent được tách thành agent nhẹ độc lập để duy trì tách biệt mối quan tâm
- Chỉ định Pydantic model trong `output_type` bắt buộc kết quả phán đoán có cấu trúc

### 4. Nguyên tắc thiết kế Handoff

- Mỗi agent chuyên biệt có lĩnh vực trách nhiệm được xác định rõ ràng
- `handoff_filters.remove_all_tools` dọn dẹp lịch sử công cụ của agent trước đó
- Callback `on_handoff` ghi nhật ký metadata handoff
- `result.last_agent` hoặc `agent_updated_stream_event` theo dõi agent đang hoạt động

---

## Bài tập thực hành

### Bài tập 1: Thêm Agent chuyên biệt mới (Độ khó: Trung bình)

**Mục tiêu**: Thêm agent chuyên hoàn trả.

**Yêu cầu**:
- Tạo tệp `my_agents/refund_agent.py`
- Định nghĩa chỉ thị động và thông báo quyền lợi bổ sung cho khách hàng cao cấp
- Thêm 2 hàm công cụ liên quan đến hoàn trả trở lên vào `tools.py`
- Kết nối `AgentToolUsageLoggingHooks`
- Thêm vào danh sách handoff của triage agent
- Thêm mục liên quan đến hoàn trả vào hướng dẫn phân loại của triage agent

### Bài tập 2: Mở rộng Output Guardrails (Độ khó: Trung bình)

**Mục tiêu**: Thêm output guardrails cho tất cả agent chuyên biệt.

**Yêu cầu**:
- Billing agent: Xác minh thông tin kỹ thuật không bị bao gồm
- Order agent: Xác minh thông tin thanh toán hoặc tài khoản không bị bao gồm
- Account agent: Xác minh thông tin đơn hàng hoặc thanh toán không bị bao gồm
- Định nghĩa Pydantic output model cho mỗi guardrail trong `models.py`

### Bài tập 3: Hệ thống Hook tùy chỉnh (Độ khó: Cao)

**Mục tiêu**: Triển khai hệ thống hook nâng cao thu thập thống kê sử dụng agent.

**Yêu cầu**:
- Theo dõi số lần gọi, thời gian phản hồi trung bình và tần suất sử dụng công cụ của mỗi agent
- Hiển thị dữ liệu thống kê trong thanh bên thời gian thực
- Đặt lại thống kê khi phiên được khởi tạo
- Ghi thời gian bắt đầu trong `on_start` và tính thời gian đã trôi qua trong `on_end`

### Bài tập 4: Voice Agent hai chiều (Độ khó: Cao)

**Mục tiêu**: Triển khai agent cho phép hội thoại giọng nói liên tục.

**Yêu cầu**:
- Tự động bắt đầu ghi âm tiếp theo khi phát xong phản hồi giọng nói
- Thông báo bằng giọng nói agent nào được chuyển đến khi handoff xảy ra
- Hiển thị lịch sử hội thoại dạng văn bản trong thanh bên (cả kết quả STT và phản hồi agent)
- Thêm chức năng nhận dạng "kết thúc hội thoại" bằng giọng nói để chấm dứt phiên

### Bài tập 5: Cơ chế quay lại Agent (Độ khó: Cao)

**Mục tiêu**: Triển khai cơ chế quay lại từ agent chuyên biệt về triage agent.

**Yêu cầu**:
- Thêm handoff "quay lại triage" cho mỗi agent chuyên biệt
- Tự động quay lại triage khi agent chuyên biệt nhận yêu cầu ngoài phạm vi
- Bao gồm tóm tắt hội thoại đến thời điểm hiện tại trong dữ liệu handoff khi quay lại
- Hiển thị trực quan sự kiện quay lại trên giao diện
