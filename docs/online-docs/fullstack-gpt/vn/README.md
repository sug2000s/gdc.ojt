# Giáo trình FullStack GPT

Giáo trình phát triển ứng dụng GPT full-stack sử dụng LangChain và OpenAI.

## Mục lục

| Chương | Tiêu đề | Nội dung chính |
|--------|---------|----------------|
| [Chapter 01](chapter_01.md) | LLMs and Chat Models | Cơ bản LLM, ChatOpenAI, Prompt Template, LCEL |
| [Chapter 02](chapter_02.md) | Prompts | FewShot, ExampleSelector, Bộ nhớ đệm, Tuần tự hóa |
| [Chapter 03](chapter_03.md) | Memory | Buffer, Summary, KG Memory, LCEL Memory |
| [Chapter 04](chapter_04.md) | RAG | Document Loader, Embedding, Vector Store, Stuff/MapReduce Chain |
| [Chapter 05](chapter_05.md) | Streamlit | Streamlit UI, Tải tệp lên, Chat, Streaming |
| [Chapter 06](chapter_06.md) | Alternative Providers | HuggingFace, GPT4All, Ollama |
| [Chapter 07](chapter_07.md) | QuizGPT | Wikipedia, Tạo câu đố, Function Calling |
| [Chapter 08](chapter_08.md) | SiteGPT | Web Scraping, SitemapLoader, Map Re-Rank |
| [Chapter 09](chapter_09.md) | MeetingGPT | Trích xuất âm thanh, Whisper, Refine Chain |
| [Chapter 10](chapter_10.md) | Agents | ReAct Agent, Tools, SQLDatabaseToolkit |
| [Chapter 11](chapter_11.md) | FastAPI & GPT Actions | FastAPI, Pinecone, OAuth |
| [Chapter 12](chapter_12.md) | Assistants API | OpenAI Assistants, Threads, RAG Assistant |
| [Chapter 13](chapter_13.md) | Cloud Providers | AWS Bedrock, Azure OpenAI |
| [Chapter 14](chapter_14.md) | CrewAI | Multi-Agent, Crew, Tasks, Custom Tools |

## Đối tượng độc giả

- Lập trình viên có kiến thức cơ bản về Python
- Người quan tâm đến phát triển ứng dụng LLM
- Người muốn học framework LangChain

## Kho mã nguồn

Mã nguồn thực hành của giáo trình này có thể xem tại kho [fullstack-gpt-3](https://github.com/sug2000s/full-stack-gpt).
Mỗi commit tương ứng với một bước học tập.

## Thiết lập môi trường

```bash
# Clone kho mã nguồn
git clone https://github.com/sug2000s/full-stack-gpt.git
cd full-stack-gpt

# Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Thiết lập tệp .env
cp .env.sample .env
# Mở tệp .env và nhập API key
```
