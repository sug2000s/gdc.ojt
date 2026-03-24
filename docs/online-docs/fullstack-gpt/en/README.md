# FullStack GPT Textbook

A textbook for developing full-stack GPT applications using LangChain and OpenAI.

## Table of Contents

| Chapter | Title | Key Topics |
|---------|-------|----------|
| [Chapter 01](chapter_01.md) | LLMs and Chat Models | LLM basics, ChatOpenAI, Prompt Templates, LCEL |
| [Chapter 02](chapter_02.md) | Prompts | FewShot, ExampleSelector, Caching, Serialization |
| [Chapter 03](chapter_03.md) | Memory | Buffer, Summary, KG Memory, LCEL Memory |
| [Chapter 04](chapter_04.md) | RAG | Document Loaders, Embeddings, Vector Stores, Stuff/MapReduce Chains |
| [Chapter 05](chapter_05.md) | Streamlit | Streamlit UI, File Upload, Chat, Streaming |
| [Chapter 06](chapter_06.md) | Alternative Providers | HuggingFace, GPT4All, Ollama |
| [Chapter 07](chapter_07.md) | QuizGPT | Wikipedia, Quiz Generation, Function Calling |
| [Chapter 08](chapter_08.md) | SiteGPT | Web Scraping, SitemapLoader, Map Re-Rank |
| [Chapter 09](chapter_09.md) | MeetingGPT | Audio Extraction, Whisper, Refine Chain |
| [Chapter 10](chapter_10.md) | Agents | ReAct Agent, Tools, SQLDatabaseToolkit |
| [Chapter 11](chapter_11.md) | FastAPI & GPT Actions | FastAPI, Pinecone, OAuth |
| [Chapter 12](chapter_12.md) | Assistants API | OpenAI Assistants, Threads, RAG Assistant |
| [Chapter 13](chapter_13.md) | Cloud Providers | AWS Bedrock, Azure OpenAI |
| [Chapter 14](chapter_14.md) | CrewAI | Multi-Agent, Crew, Tasks, Custom Tools |

## Target Audience

- Developers with basic Python knowledge
- Those interested in developing LLM applications
- Those who want to learn the LangChain framework

## Code Repository

The hands-on code for this textbook can be found in the [fullstack-gpt-3](https://github.com/sug2000s/full-stack-gpt) repository.
Each commit corresponds to one learning step.

## Environment Setup

```bash
# Clone the code repository
git clone https://github.com/sug2000s/full-stack-gpt.git
cd full-stack-gpt

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Configure the .env file
cp .env.sample .env
# Open the .env file and enter your API keys
```
