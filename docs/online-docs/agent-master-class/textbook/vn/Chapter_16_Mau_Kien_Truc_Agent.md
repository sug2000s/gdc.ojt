# Chapter 16: Mẫu kiến trúc Workflow (Workflow Architecture Patterns)

---

## 1. Tổng quan chương

Trong chương này, chúng ta sẽ học các **mẫu kiến trúc workflow** cốt lõi có thể sử dụng khi xây dựng hệ thống AI agent. Chúng ta sẽ thực hành nhiều cách kết hợp các lời gọi LLM một cách có hệ thống bằng LangGraph, với mục tiêu hiểu rõ mỗi mẫu phù hợp với tình huống nào.

### Các mẫu kiến trúc được đề cập

| Phần | Mẫu | Khái niệm cốt lõi |
|------|------|-----------|
| 16.0 | Introduction | Cấu hình môi trường dự án |
| 16.1 | Prompt Chaining | Chuỗi gọi LLM tuần tự |
| 16.2 | Prompt Chaining Gate | Phân nhánh có điều kiện (gate) |
| 16.3 | Routing | Định tuyến động dựa trên đầu vào |
| 16.4 | Parallelization | Thực thi song song và tổng hợp kết quả |
| 16.5 | Orchestrator-Workers | Phân phối tác vụ động (Map-Reduce) |

### Công nghệ sử dụng

- **Python 3.13**
- **LangGraph 0.6.6** -- Framework xây dựng đồ thị workflow
- **LangChain 0.3.27** -- Lớp tích hợp LLM
- **OpenAI GPT-4o** -- Mô hình LLM chính
- **Pydantic** -- Định nghĩa đầu ra có cấu trúc (Structured Output)

---

## 2. Giải thích chi tiết từng phần

---

### 16.0 Introduction -- Cấu hình môi trường dự án

#### Chủ đề và mục tiêu

Tạo dự án mới `workflow-architectures` và thiết lập môi trường phát triển cho các thí nghiệm workflow dựa trên LangGraph.

#### Giải thích khái niệm cốt lõi

Trong phần này, chúng ta sử dụng trình quản lý gói `uv` để khởi tạo dự án Python. Tất cả dependency được khai báo trong `pyproject.toml`, và Jupyter Notebook (`main.ipynb`) được sử dụng làm môi trường thực thi.

#### Phân tích code

**Dependency dự án (`pyproject.toml`):**

```toml
[project]
name = "workflow-architectures"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "grandalf==0.8",
    "langchain[openai]==0.3.27",
    "langgraph==0.6.6",
    "langgraph-checkpoint-sqlite==2.0.11",
    "langgraph-cli[inmem]==0.4.0",
    "python-dotenv==1.1.1",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

Giải thích các dependency chính:
- **`langgraph`**: Thư viện cốt lõi để xây dựng đồ thị workflow dựa trên trạng thái. Định nghĩa luồng gọi LLM bằng node và edge.
- **`langchain[openai]`**: Cung cấp tích hợp với model OpenAI. Có thể khởi tạo nhiều model khác nhau bằng `init_chat_model()`.
- **`grandalf`**: Thư viện trực quan hóa đồ thị.
- **`ipykernel`**: Dependency phát triển để sử dụng kernel của môi trường ảo trong Jupyter Notebook.

#### Điểm thực hành

1. Làm quen với cách tạo dự án bằng `uv init` và thêm dependency bằng `uv add`.
2. Hiểu mẫu cố định phiên bản Python bằng file `.python-version`.
3. Cần thiết lập `OPENAI_API_KEY` trong file `.env` để gọi LLM hoạt động bình thường.

---

### 16.1 Prompt Chaining Architecture -- Chuỗi prompt tuần tự

#### Chủ đề và mục tiêu

Triển khai mẫu workflow cơ bản nhất -- **Prompt Chaining**. Kết nối nhiều lời gọi LLM **theo thứ tự tuần tự**, tạo ra pipeline trong đó đầu ra của bước trước trở thành đầu vào của bước sau.

#### Giải thích khái niệm cốt lõi

**Prompt Chaining** là mẫu phân tách tác vụ phức tạp thành nhiều bước nhỏ. Mỗi bước được xử lý bởi một lời gọi LLM, kết quả được lưu vào State và truyền sang bước tiếp theo.

Trong ví dụ này, quy trình tạo công thức nấu ăn được chia thành 3 bước:
1. **Liệt kê nguyên liệu** (list_ingredients) -- Tạo danh sách nguyên liệu cần thiết cho món ăn
2. **Viết công thức** (create_recipe) -- Tạo cách nấu dựa trên nguyên liệu
3. **Mô tả bày biện** (describe_plating) -- Mô tả cách bày biện dựa trên công thức

```
START --> list_ingredients --> create_recipe --> describe_plating --> END
```

Cốt lõi của mẫu này là **mỗi bước phụ thuộc vào kết quả của bước trước**. Không biết nguyên liệu thì không thể viết công thức, không biết công thức thì không thể mô tả bày biện.

#### Phân tích code

**Bước 1: Định nghĩa State và mô hình dữ liệu**

```python
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

llm = init_chat_model("openai:gpt-4o")
```

`init_chat_model()` là hàm khởi tạo model phổ quát do LangChain cung cấp. Chỉ định provider và tên model theo định dạng `"openai:gpt-4o"`.

```python
class State(TypedDict):
    dish: str
    ingredients: list[dict]
    recipe_steps: str
    plating_instructions: str
```

`State` là **đối tượng trạng thái** được chia sẻ trong toàn bộ workflow. Sử dụng `TypedDict` để chỉ định rõ kiểu của mỗi trường. Trong LangGraph, tất cả node đều đọc và cập nhật State này.

```python
class Ingredient(BaseModel):
    name: str
    quantity: str
    unit: str

class IngredientsOutput(BaseModel):
    ingredients: List[Ingredient]
```

Sử dụng Pydantic `BaseModel` để định nghĩa **đầu ra có cấu trúc (Structured Output)**. Có thể buộc LLM trả về JSON theo schema đã định thay vì văn bản tự do.

**Bước 2: Định nghĩa hàm node**

```python
def list_ingredients(state: State):
    structured_llm = llm.with_structured_output(IngredientsOutput)
    response = structured_llm.invoke(
        f"List 5-8 ingredients needed to make {state['dish']}"
    )
    return {
        "ingredients": response.ingredients,
    }
```

`with_structured_output(IngredientsOutput)` thiết lập tự động phân tích phản hồi LLM thành model Pydantic `IngredientsOutput`. Nhờ đó, LLM trả về dữ liệu có cấu trúc dạng `{"ingredients": [{"name": "Chickpeas", "quantity": "1", "unit": "cup"}, ...]}`.

Mỗi hàm node bắt buộc phải **trả về dictionary**. Các cặp key-value trong dictionary trả về sẽ được cập nhật vào State.

```python
def create_recipe(state: State):
    response = llm.invoke(
        f"Write a step by step cooking instruction for {state['dish']}, "
        f"using these ingredients {state['ingredients']}",
    )
    return {
        "recipe_steps": response.content,
    }

def describe_plating(state: State):
    response = llm.invoke(
        f"Describe how to beautifully plate this dish {state['dish']} "
        f"based on this recipe {state['recipe_steps']}"
    )
    return {
        "plating_instructions": response.content,
    }
```

`create_recipe` tham chiếu `state['ingredients']`, và `describe_plating` tham chiếu `state['recipe_steps']`. Đây chính là **cốt lõi của chaining** -- đầu ra của bước trước được bao gồm trong prompt của bước sau.

**Bước 3: Xây dựng và thực thi đồ thị**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_edge("list_ingredients", "create_recipe")
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

Tạo graph builder bằng `StateGraph(State)`, đăng ký node bằng `add_node()`, rồi định nghĩa kết nối giữa các node bằng `add_edge()`. `START` và `END` là các node đặc biệt do LangGraph cung cấp, biểu thị điểm bắt đầu và kết thúc của đồ thị.

```python
graph.invoke({"dish": "hummus"})
```

Truyền State khởi tạo vào `graph.invoke()` để thực thi đồ thị. Chỉ cần cung cấp giá trị `dish`, các trường còn lại sẽ được mỗi node lần lượt điền vào.

#### Điểm thực hành

1. Kiểm tra giá trị trả về của `graph.invoke()` để quan sát cách mỗi trường được điền.
2. Thay đổi giá trị `dish` để so sánh kết quả cho các món ăn khác nhau.
3. Hiểu sự khác biệt giữa node sử dụng `with_structured_output()` và node sử dụng `invoke()` thông thường.

---

### 16.2 Prompt Chaining Gate -- Gate có điều kiện

#### Chủ đề và mục tiêu

Thêm **phân nhánh có điều kiện (Gate)** vào Prompt Chaining, triển khai mẫu **thực thi lại** bước trước nếu không đáp ứng điều kiện nhất định.

#### Giải thích khái niệm cốt lõi

Trong ứng dụng thực tế, đầu ra của LLM không phải lúc nào cũng đáp ứng kỳ vọng. **Gate** đóng vai trò cổng kiểm tra chất lượng, kiểm tra xem đầu ra LLM có đáp ứng tiêu chí nhất định hay không. Nếu không đáp ứng, bước đó sẽ được thực thi lại để có kết quả tốt hơn.

Trong ví dụ này, chỉ được tiến sang bước tiếp theo khi số nguyên liệu nằm trong khoảng 3-8:

```
START --> list_ingredients --[gate]--> create_recipe --> describe_plating --> END
                ^                          |
                |    (khi không đạt)        |
                +----------<---------------+
```

#### Phân tích code

**Định nghĩa hàm gate:**

```python
def gate(state: State):
    ingredients = state["ingredients"]

    if len(ingredients) > 8 or len(ingredients) < 3:
        return False

    return True
```

Hàm gate nhận State và trả về `True` hoặc `False`. Đường đi tiếp theo của đồ thị được quyết định dựa trên giá trị trả về này.

- **True**: Số nguyên liệu trong khoảng 3-8 -- tiến sang `create_recipe`
- **False**: Ngoài khoảng -- thực thi lại `list_ingredients` (thử lại)

**Thiết lập edge có điều kiện:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("list_ingredients", list_ingredients)
graph_builder.add_node("create_recipe", create_recipe)
graph_builder.add_node("describe_plating", describe_plating)

graph_builder.add_edge(START, "list_ingredients")
graph_builder.add_conditional_edges(
    "list_ingredients",
    gate,
    {
        True: "create_recipe",
        False: "list_ingredients",
    },
)
graph_builder.add_edge("create_recipe", "describe_plating")
graph_builder.add_edge("describe_plating", END)

graph = graph_builder.compile()
```

Điểm cốt lõi là phương thức `add_conditional_edges()`:
- **Tham số thứ nhất**: Node xuất phát (`"list_ingredients"`)
- **Tham số thứ hai**: Hàm phán đoán điều kiện (`gate`)
- **Tham số thứ ba**: Dictionary ánh xạ giá trị trả về với node đích

Khi `gate` trả về `False`, quay lại `"list_ingredients"`, tạo thành vòng lặp thử lại cho đến khi điều kiện được đáp ứng.

#### Kịch bản ứng dụng mẫu Gate

| Kịch bản | Điều kiện Gate |
|---------|-----------|
| Sinh code | Code được tạo có vượt qua kiểm tra cú pháp không? |
| Dịch thuật | Ngôn ngữ kết quả dịch có chính xác không? |
| Trích xuất dữ liệu | Tất cả trường bắt buộc đã được điền chưa? |
| Tóm tắt | Độ dài bản tóm tắt có phù hợp không? |

#### Điểm thực hành

1. Thay đổi điều kiện trong hàm `gate` để xem số lần thử lại thay đổi như thế nào.
2. Suy nghĩ cách thêm số lần thử lại tối đa để ngăn vòng lặp vô hạn (ví dụ: thêm trường `retry_count` vào State).
3. Cũng có thể phân nhánh theo nhiều đường khác nhau thay vì chỉ `True`/`False` đơn giản.

---

### 16.3 Routing Architecture -- Định tuyến động

#### Chủ đề và mục tiêu

Triển khai mẫu **Routing** phân nhánh sang **các đường xử lý khác nhau** tùy theo đặc tính của đầu vào. LLM phân loại đầu vào, và dựa trên kết quả phân loại chọn model hoặc logic xử lý phù hợp.

#### Giải thích khái niệm cốt lõi

Mẫu Routing dựa trên ý tưởng "không cần áp dụng cùng một cách cho mọi tác vụ". Sử dụng model đắt tiền cho câu hỏi dễ là lãng phí, và sử dụng model yếu cho câu hỏi khó thì chất lượng giảm.

Trong ví dụ này, độ khó của câu hỏi được tự động đánh giá, sau đó chọn model phù hợp với độ khó để tạo phản hồi:

```
                    +--> dumb_node (GPT-3.5) ---+
                    |                           |
START --> assess_difficulty --> average_node (GPT-4o) --> END
                    |                           |
                    +--> smart_node (GPT-5) ----+
```

#### Phân tích code

**Khởi tạo model:**

```python
llm = init_chat_model("openai:gpt-4o")

dumb_llm = init_chat_model("openai:gpt-3.5-turbo")
average_llm = init_chat_model("openai:gpt-4o")
smart_llm = init_chat_model("openai:gpt-5-2025-08-07")
```

Chuẩn bị ba LLM với các mức năng lực khác nhau. Trong production thực tế, chiến lược này thường được sử dụng để cân bằng giữa chi phí và hiệu suất.

**Định nghĩa state và schema:**

```python
class State(TypedDict):
    question: str
    difficulty: str
    answer: str
    model_used: str

class DifficultyResponse(BaseModel):
    difficulty_level: Literal["easy", "medium", "hard"]
```

`DifficultyResponse` sử dụng kiểu `Literal` để buộc LLM chỉ được chọn một trong `"easy"`, `"medium"`, `"hard"`. Đây là ưu điểm mạnh mẽ của Structured Output -- có thể giới hạn phản hồi LLM thành dạng có thể điều khiển lập trình.

**Node đánh giá độ khó và định tuyến:**

```python
def assess_difficulty(state: State):
    structured_llm = llm.with_structured_output(DifficultyResponse)

    response = structured_llm.invoke(
        f"""
        Assess the difficulty of this question
        Question: {state["question"]}

        - EASY: Simple facts, basic definitions, yes/no answers
        - MEDIUM: Requires explanation, comparison, analysis
        - HARD: Complex reasoning, multiple steps, deep expertise.
        """
    )

    difficulty_level = response.difficulty_level

    if difficulty_level == "easy":
        goto = "dumb_node"
    elif difficulty_level == "medium":
        goto = "average_node"
    elif difficulty_level == "hard":
        goto = "smart_node"

    return Command(
        goto=goto,
        update={
            "difficulty": difficulty_level,
        },
    )
```

Hàm này có hai khái niệm quan trọng:

1. **Đối tượng `Command`**: `Command` của LangGraph **thực hiện đồng thời cập nhật trạng thái và định tuyến**. Chỉ định node tiếp theo bằng `goto` và cập nhật State bằng `update`. Khác với `add_conditional_edges()` trước đó, có thể quyết định định tuyến trực tiếp bên trong hàm node.

2. **Prompt đánh giá độ khó**: Trình bày rõ tiêu chí cho mỗi mức độ khó để hướng dẫn LLM phân loại nhất quán.

**Các node xử lý:**

```python
def dumb_node(state: State):
    response = dumb_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-3.5",
    }

def average_node(state: State):
    response = average_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-4o",
    }

def smart_node(state: State):
    response = smart_llm.invoke(state["question"])
    return {
        "answer": response.content,
        "model_used": "gpt-o3",
    }
```

Mỗi node trả lời câu hỏi bằng LLM được gán cho nó. Có thể theo dõi model nào đã được sử dụng thông qua trường `model_used`.

**Xây dựng đồ thị:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("dumb_node", dumb_node)
graph_builder.add_node("average_node", average_node)
graph_builder.add_node("smart_node", smart_node)
graph_builder.add_node(
    "assess_difficulty",
    assess_difficulty,
    destinations=(
        "dumb_node",
        "average_node",
        "smart_node",
    ),
)

graph_builder.add_edge(START, "assess_difficulty")
graph_builder.add_edge("dumb_node", END)
graph_builder.add_edge("average_node", END)
graph_builder.add_edge("smart_node", END)

graph = graph_builder.compile()
```

Node trả về `Command` cần thêm tham số `destinations`. Điều này cho LangGraph biết node này có thể định tuyến đến những node nào. Được sử dụng cho trực quan hóa và xác thực đồ thị.

**Thực thi:**

```python
graph.invoke({"question": "Investment potential of Uranium in 2026"})
```

Câu hỏi này yêu cầu phân tích phức tạp nên sẽ được phân loại là `"hard"` và định tuyến đến `smart_node` (GPT-5).

#### Điểm thực hành

1. Nhập câu hỏi với các mức độ khó khác nhau để kiểm tra định tuyến có hoạt động đúng không.
2. Kiểm tra trường `model_used` để xác nhận model nào thực sự được chọn.
3. Ngoài việc chọn model, hãy thiết kế kịch bản định tuyến đến các template prompt hoặc tool khác nhau.

---

### 16.4 Parallelization Architecture -- Thực thi song song

#### Chủ đề và mục tiêu

Triển khai mẫu **Parallelization** -- **thực thi đồng thời nhiều lời gọi LLM song song**, sau đó khi tất cả kết quả đã sẵn sàng thì **tổng hợp (aggregation)**.

#### Giải thích khái niệm cốt lõi

Thực thi tuần tự đơn giản nhưng không hiệu quả khi có nhiều tác vụ độc lập với nhau. Ví dụ, nếu thực hiện tuần tự tóm tắt, phân tích cảm xúc, trích xuất điểm chính và đưa ra khuyến nghị từ tài liệu, sẽ mất gấp 4 lần thời gian. Các tác vụ này không phụ thuộc lẫn nhau nên có thể thực thi đồng thời.

Trong ví dụ này, bài phát biểu của Chủ tịch Fed được phân tích đồng thời từ 4 góc độ:

```
            +--> get_summary --------+
            |                        |
            +--> get_sentiment ------+
START ----->|                        +--> get_final_analysis --> END
            +--> get_key_points -----+
            |                        |
            +--> get_recommendation -+
```

4 node phân tích được **thực thi đồng thời**, và khi tất cả hoàn thành, `get_final_analysis` thực hiện phân tích tổng hợp.

#### Phân tích code

**Định nghĩa state:**

```python
class State(TypedDict):
    document: str
    summary: str
    sentiment: str
    key_points: str
    recommendation: str
    final_analysis: str
```

Mỗi trường mà các node song song sẽ điền được định nghĩa riêng biệt. Các trường này độc lập với nhau nên cập nhật đồng thời không gây xung đột.

**Các hàm node song song:**

```python
def get_summary(state: State):
    response = llm.invoke(
        f"Write a 3-sentence summary of this document {state['document']}"
    )
    return {"summary": response.content}

def get_sentiment(state: State):
    response = llm.invoke(
        f"Analyse the sentiment and tone of this document {state['document']}"
    )
    return {"sentiment": response.content}

def get_key_points(state: State):
    response = llm.invoke(
        f"List the 5 most important points of this document {state['document']}"
    )
    return {"key_points": response.content}

def get_recommendation(state: State):
    response = llm.invoke(
        f"Based on the document, list 3 recommended next steps {state['document']}"
    )
    return {"recommendation": response.content}
```

Cả 4 hàm đều đọc cùng `state["document"]` nhưng lưu kết quả vào các trường khác nhau. Đây là lý do thực thi song song khả thi -- đọc được chia sẻ, ghi được tách biệt.

**Node tổng hợp:**

```python
def get_final_analysis(state: State):
    response = llm.invoke(
        f"""
    Give me an analysis of the following report

    DOCUMENT ANALYSIS REPORT
    ========================

    EXECUTIVE SUMMARY:
    {state['summary']}

    SENTIMENT ANALYSIS:
    {state['sentiment']}

    KEY POINTS:
    {state.get("key_points", "")}

    RECOMMENDATIONS:
    {state.get('recommendation', "N/A")}
    """
    )
    return {"final_analysis": response.content}
```

Node tổng hợp kết hợp tất cả trường mà các node song song đã điền để tạo phân tích cuối cùng. Node này chỉ được thực thi **sau khi tất cả node song song đã hoàn thành**.

**Xây dựng đồ thị -- Cốt lõi của edge song song:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("get_summary", get_summary)
graph_builder.add_node("get_sentiment", get_sentiment)
graph_builder.add_node("get_key_points", get_key_points)
graph_builder.add_node("get_recommendation", get_recommendation)
graph_builder.add_node("get_final_analysis", get_final_analysis)

# Kết nối từ START đến 4 node đồng thời = thực thi song song!
graph_builder.add_edge(START, "get_summary")
graph_builder.add_edge(START, "get_sentiment")
graph_builder.add_edge(START, "get_key_points")
graph_builder.add_edge(START, "get_recommendation")

# Kết nối cả 4 node đến get_final_analysis = thực thi sau khi tất cả hoàn thành
graph_builder.add_edge("get_summary", "get_final_analysis")
graph_builder.add_edge("get_sentiment", "get_final_analysis")
graph_builder.add_edge("get_key_points", "get_final_analysis")
graph_builder.add_edge("get_recommendation", "get_final_analysis")

graph_builder.add_edge("get_final_analysis", END)

graph = graph_builder.compile()
```

Cách triển khai thực thi song song trong LangGraph rất trực quan:
- **Kết nối edge từ `START` đến nhiều node** thì các node đó được thực thi đồng thời.
- **Kết nối edge từ nhiều node đến một node** thì sẽ chờ tất cả node trước hoàn thành rồi mới thực thi (join/barrier).

**Thực thi streaming:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

Sử dụng `graph.stream()` để nhận kết quả mỗi khi node hoàn thành. `stream_mode="updates"` chỉ stream cập nhật State của mỗi node. Kết quả của các node song song được xuất theo thứ tự hoàn thành, cho phép theo dõi thời gian thực tác vụ nào hoàn thành trước.

#### Điểm thực hành

1. So sánh tổng thời gian giữa thực thi tuần tự và song song.
2. Dùng `stream_mode="updates"` để quan sát thứ tự kết quả đến -- phân tích nào hoàn thành nhanh nhất?
3. Tăng hoặc giảm số node song song để đo lường sự thay đổi hiệu suất.
4. Sử dụng tài liệu thực tế (bài phát biểu Fed) để đánh giá chất lượng phân tích.

---

### 16.5 Orchestrator-Workers Architecture -- Orchestrator-Worker

#### Chủ đề và mục tiêu

Triển khai mẫu **Orchestrator-Workers (Map-Reduce)** -- **tạo worker động theo đầu vào** để phân phối tác vụ, thu thập kết quả từ tất cả worker và tạo kết quả cuối cùng.

#### Giải thích khái niệm cốt lõi

Parallelization ở phần 16.4 có **số node cố định** (luôn là 4 node phân tích). Nhưng trong thực tế, số tác vụ song song thường cần thay đổi tùy theo đầu vào. Ví dụ:
- Tài liệu có 3 đoạn thì cần 3 worker tóm tắt
- Tài liệu có 20 đoạn thì cần 20 worker tóm tắt

Trong mẫu này, **orchestrator (dispatcher)** phân tích đầu vào và tạo động số worker cần thiết. Triển khai bằng API `Send` của LangGraph.

```
            +--> summarize_p (đoạn 0) --+
            |                           |
            +--> summarize_p (đoạn 1) --+
START ----->|                           +--> final_summary --> END
            +--> summarize_p (đoạn 2) --+
            |                           |
            +--> summarize_p (đoạn N) --+
```

Số worker (N) được xác định tại thời điểm thực thi dựa trên số đoạn của tài liệu.

#### Phân tích code

**Import mới:**

```python
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.types import Send
from operator import add
```

- **`Send`**: Đối tượng chỉ thị thực thi bằng cách truyền tham số cụ thể đến node cụ thể
- **`Annotated` và `add`**: Được sử dụng để định nghĩa **reducer** cho trường danh sách

**Định nghĩa state -- Reducer Annotated:**

```python
class State(TypedDict):
    document: str
    final_summary: str
    summaries: Annotated[list[dict], add]
```

`Annotated[list[dict], add]` là khái niệm cốt lõi của LangGraph -- **reducer**. Khi nhiều worker đồng thời trả về giá trị cho trường `summaries`, hành vi mặc định là ghi đè bằng giá trị cuối cùng. Nhưng khi chỉ định reducer `add`, **kết quả của tất cả worker được tích lũy vào danh sách**.

Ví dụ, nếu worker A trả về `{"summaries": [item_a]}` và worker B trả về `{"summaries": [item_b]}`, thì `summaries` cuối cùng sẽ là `[item_a, item_b]`. Vì `operator.add` thực hiện phép toán `+` trên danh sách (nối).

**Node worker:**

```python
def summarize_p(args):
    paragraph = args["paragraph"]
    index = args["index"]
    response = llm.invoke(
        f"Write a 3-sentence summary for this paragraph: {paragraph}",
    )
    return {
        "summaries": [
            {
                "summary": response.content,
                "index": index,
            }
        ],
    }
```

Lưu ý:
- Tham số của hàm này là `args` chứ không phải `state`. Nhận tham số tùy chỉnh được truyền qua `Send`.
- Lưu kèm `index` để có thể khôi phục thứ tự đoạn sau này.
- Giá trị trả về `summaries` được bọc trong danh sách -- để sử dụng cùng reducer `add`.

**Hàm orchestrator (dispatcher):**

```python
def dispatch_summarizers(state: State):
    chunks = state["document"].split("\n\n")
    return [
        Send("summarize_p", {"paragraph": chunk, "index": index})
        for index, chunk in enumerate(chunks)
    ]
```

Đây là cốt lõi của mẫu Orchestrator-Workers:

1. Tách tài liệu bằng `"\n\n"` (dòng trống) để tạo danh sách đoạn.
2. Tạo đối tượng `Send("summarize_p", {...})` cho mỗi đoạn.
3. Tham số thứ nhất của `Send` là tên node cần thực thi, tham số thứ hai là dữ liệu truyền cho node đó.
4. Khi trả về danh sách đối tượng `Send`, LangGraph **thực thi song song các node đó đồng thời**.

Nếu tài liệu có 15 đoạn thì 15 instance `summarize_p` được thực thi đồng thời. Đây là điểm khác biệt lớn nhất so với song song hóa tĩnh ở phần 16.4.

**Node tổng hợp cuối cùng:**

```python
def final_summary(state: State):
    response = llm.invoke(
        f"Using the following summaries, give me a final one {state['summaries']}"
    )
    return {
        "final_summary": response.content,
    }
```

Khi tóm tắt của tất cả worker đã được tập hợp vào danh sách `summaries`, node `final_summary` tổng hợp chúng để tạo bản tóm tắt cuối cùng.

**Xây dựng đồ thị:**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("summarize_p", summarize_p)
graph_builder.add_node("final_summary", final_summary)

graph_builder.add_conditional_edges(
    START,
    dispatch_summarizers,
    ["summarize_p"],
)

graph_builder.add_edge("summarize_p", "final_summary")
graph_builder.add_edge("final_summary", END)

graph = graph_builder.compile()
```

Sử dụng `add_conditional_edges()` nhưng mục đích ở đây khác với gate ở phần 16.2:
- **Tham số thứ ba `["summarize_p"]`**: Danh sách node đích có thể. Khi `dispatch_summarizers` trả về các đối tượng `Send`, các node tương ứng được tạo động.
- LangGraph chờ tất cả `Send` hoàn thành rồi mới tiếp tục edge tiếp theo (`"summarize_p"` --> `"final_summary"`).

**Thực thi:**

```python
with open("fed_transcript.md", "r", encoding="utf-8") as file:
    document = file.read()

for chunk in graph.stream(
    {"document": document},
    stream_mode="updates",
):
    print(chunk, "\n")
```

Khi thực thi streaming, kết quả tóm tắt của mỗi đoạn được xuất ra khi hoàn thành. Có thể xác nhận rằng số worker được tự động điều chỉnh theo số đoạn.

#### Điểm thực hành

1. Thử nghiệm số đối tượng `Send` mà `dispatch_summarizers` tạo ra với các tài liệu khác nhau.
2. Xem điều gì xảy ra khi loại bỏ reducer `Annotated[list[dict], add]`.
3. Thêm logic hậu xử lý sắp xếp kết quả theo thứ tự ban đầu bằng trường `index`.
4. Áp dụng cùng mẫu cho các tác vụ khác (dịch thuật, trích xuất từ khóa, v.v.).

---

## 3. Tổng kết chương

### Bảng so sánh mẫu kiến trúc

| Mẫu | Cách thực thi | Số node | Tình huống phù hợp | API LangGraph |
|------|----------|---------|------------|---------------|
| **Prompt Chaining** | Tuần tự | Cố định | Tác vụ có phụ thuộc giữa các bước | `add_edge()` |
| **Prompt Chaining + Gate** | Tuần tự + thử lại | Cố định | Tác vụ cần kiểm tra chất lượng | `add_conditional_edges()` |
| **Routing** | Phân nhánh | Cố định | Tác vụ cần xử lý khác nhau theo đặc tính đầu vào | `Command(goto=...)` |
| **Parallelization** | Song song | Cố định | Thực hiện đồng thời nhiều phân tích độc lập | Nhiều `add_edge(START, ...)` |
| **Orchestrator-Workers** | Song song động | Thay đổi | Trường hợp số tác vụ thay đổi theo kích thước đầu vào | `Send()` + `Annotated[..., add]` |

### Tóm tắt khái niệm cốt lõi LangGraph

1. **StateGraph**: Thành phần cơ bản của đồ thị dựa trên trạng thái. Các node chia sẻ trạng thái được định nghĩa bằng `TypedDict`.
2. **add_edge()**: Định nghĩa đường đi cố định giữa các node.
3. **add_conditional_edges()**: Xác định node tiếp theo một cách động dựa trên giá trị trả về của hàm.
4. **Command**: Thực hiện đồng thời định tuyến và cập nhật trạng thái bên trong hàm node.
5. **Send**: Truyền tham số tùy chỉnh đến node cụ thể để triển khai thực thi song song động.
6. **Annotated + Reducer**: Định nghĩa chiến lược hợp nhất khi nhiều node thêm giá trị vào cùng một trường.
7. **with_structured_output()**: Cấu trúc hóa đầu ra LLM thành model Pydantic để sử dụng lập trình.

---
