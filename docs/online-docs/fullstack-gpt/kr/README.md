# FullStack GPT 교재

LangChain과 OpenAI를 활용한 풀스택 GPT 애플리케이션 개발 교재입니다.

## 목차

| 챕터 | 제목 | 주요 내용 |
|------|------|----------|
| [Chapter 01](chapter_01.md) | LLMs and Chat Models | LLM 기초, ChatOpenAI, 프롬프트 템플릿, LCEL |
| [Chapter 02](chapter_02.md) | Prompts | FewShot, ExampleSelector, 캐싱, 직렬화 |
| [Chapter 03](chapter_03.md) | Memory | Buffer, Summary, KG 메모리, LCEL 메모리 |
| [Chapter 04](chapter_04.md) | RAG | 문서 로더, 임베딩, 벡터스토어, Stuff/MapReduce 체인 |
| [Chapter 05](chapter_05.md) | Streamlit | Streamlit UI, 파일 업로드, 채팅, 스트리밍 |
| [Chapter 06](chapter_06.md) | Alternative Providers | HuggingFace, GPT4All, Ollama |
| [Chapter 07](chapter_07.md) | QuizGPT | Wikipedia, 퀴즈 생성, Function Calling |
| [Chapter 08](chapter_08.md) | SiteGPT | 웹 스크래핑, SitemapLoader, Map Re-Rank |
| [Chapter 09](chapter_09.md) | MeetingGPT | 오디오 추출, Whisper, Refine Chain |
| [Chapter 10](chapter_10.md) | Agents | ReAct Agent, Tools, SQLDatabaseToolkit |
| [Chapter 11](chapter_11.md) | FastAPI & GPT Actions | FastAPI, Pinecone, OAuth |
| [Chapter 12](chapter_12.md) | Assistants API | OpenAI Assistants, Threads, RAG Assistant |
| [Chapter 13](chapter_13.md) | Cloud Providers | AWS Bedrock, Azure OpenAI |
| [Chapter 14](chapter_14.md) | CrewAI | Multi-Agent, Crew, Tasks, Custom Tools |

## 대상 독자

- Python 기초 문법을 아는 개발자
- LLM 애플리케이션 개발에 관심 있는 분
- LangChain 프레임워크를 배우고 싶은 분

## 코드 저장소

이 교재의 실습 코드는 [fullstack-gpt-3](https://github.com/sug2000s/full-stack-gpt) 저장소에서 확인할 수 있습니다.
각 커밋이 하나의 학습 단계에 대응됩니다.

## 환경 설정

```bash
# 코드 저장소 클론
git clone https://github.com/sug2000s/full-stack-gpt.git
cd full-stack-gpt

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# .env 파일 설정
cp .env.sample .env
# .env 파일을 열어서 API 키 입력
```
