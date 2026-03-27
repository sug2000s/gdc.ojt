# Chapter 15: LangGraph 유튜브 썸네일 자동 생성 프로젝트

---

## 1. 챕터 개요

이번 챕터에서는 **LangGraph**를 활용하여 유튜브 영상으로부터 자동으로 썸네일을 생성하는 완전한 파이프라인을 구축한다. 이 프로젝트는 단순한 이미지 생성을 넘어서, 영상의 오디오 추출부터 음성 인식(Transcription), 텍스트 요약, AI 이미지 생성, 그리고 사람의 피드백을 반영한 최종 고화질 썸네일 생성까지 전체 워크플로우를 그래프 기반으로 설계한다.

### 프로젝트의 핵심 학습 목표

| 섹션 | 주제 | 핵심 기술 |
|------|------|-----------|
| 15.0 | 프로젝트 소개 및 환경 설정 | uv, pyproject.toml, 의존성 관리 |
| 15.1 | 오디오 추출 및 음성 인식 | ffmpeg, OpenAI Whisper API, StateGraph |
| 15.2 | 병렬 요약 노드 | Send API, Map-Reduce 패턴, Annotated 리듀서 |
| 15.3 | 썸네일 스케치 생성 노드 | GPT Image API, 병렬 이미지 생성 |
| 15.4 | 휴먼 피드백 루프 | interrupt, Command, InMemorySaver |
| 15.5 | HD 썸네일 생성 및 배포 | LangGraph CLI, langgraph.json, 프로덕션 배포 |

### 전체 그래프 흐름도

```
[START]
   |
   v
[extract_audio] -----> ffmpeg로 mp4에서 mp3 추출
   |
   v
[transcribe_audio] --> OpenAI Whisper로 음성 -> 텍스트 변환
   |
   v (conditional: dispatch_summarizers)
[summarize_chunk] x N --> 텍스트를 청크로 나누어 병렬 요약
   |
   v
[mega_summary] --------> 모든 청크 요약을 하나의 최종 요약으로 통합
   |
   v (conditional: dispatch_artists)
[generate_thumbnails] x 5 --> 5개의 서로 다른 썸네일 스케치 병렬 생성
   |
   v
[human_feedback] ------> interrupt로 사용자 피드백 수집
   |
   v
[generate_hd_thumbnail] -> 피드백 반영한 고화질 최종 썸네일 생성
   |
   v
[END]
```

---

## 2. 섹션별 상세 설명

---

### 15.0 프로젝트 소개 및 환경 설정

**커밋:** `c8e3d16` "15.0 Introduction"

#### 주제 및 목표

프로젝트의 기초를 세우는 단계이다. Python 패키지 관리 도구인 **uv**를 사용하여 프로젝트를 초기화하고, 필요한 모든 의존성을 설정한다. Jupyter Notebook 환경에서 개발을 시작할 수 있도록 `ipykernel`도 개발 의존성으로 포함한다.

#### 핵심 개념 설명

**uv 패키지 매니저**: Python 생태계에서 `pip`와 `venv`를 대체하는 현대적인 패키지 매니저이다. Rust로 작성되어 매우 빠르며, `pyproject.toml` 기반으로 프로젝트를 관리한다. `uv.lock` 파일을 통해 정확한 의존성 버전을 고정(lock)하여 재현 가능한 환경을 보장한다.

**pyproject.toml**: Python 프로젝트의 표준 설정 파일이다. 프로젝트 메타데이터와 의존성을 선언적으로 정의한다.

#### 코드 분석

```toml
[project]
name = "youtube-thumbnail-maker"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.7",
    "langgraph-cli[inmem]>=0.4.2",
    "langsmith>=0.4.28",
    "openai[aiohttp]>=1.107.3",
    "python-dotenv>=1.1.1",
    "tiktoken>=0.11.0",
]

[dependency-groups]
dev = [
    "ipykernel>=6.30.1",
]
```

**의존성 분석:**

- **`langchain[openai]`**: LangChain 프레임워크의 OpenAI 통합. `[openai]`는 extras로, OpenAI 관련 추가 패키지를 함께 설치한다.
- **`langgraph`**: LangGraph 코어 라이브러리. 그래프 기반 에이전트 워크플로우를 구축하는 핵심 도구이다.
- **`langgraph-cli[inmem]`**: LangGraph CLI 도구. `[inmem]`은 인메모리 체크포인터 지원을 포함한다. 로컬 개발 서버를 실행할 때 사용한다.
- **`langsmith`**: LangSmith 관찰성(observability) 플랫폼. 그래프 실행을 모니터링하고 디버깅할 수 있다.
- **`openai[aiohttp]`**: OpenAI Python SDK. `[aiohttp]`는 비동기 HTTP 지원을 추가한다. Whisper API와 이미지 생성 API에 사용된다.
- **`python-dotenv`**: `.env` 파일에서 환경 변수(API 키 등)를 로드하는 유틸리티이다.
- **`tiktoken`**: OpenAI의 토크나이저. 텍스트의 토큰 수를 계산할 때 사용한다.
- **`ipykernel`**: Jupyter Notebook에서 Python 커널을 사용할 수 있게 해주는 개발 의존성이다.

#### 실습 포인트

1. `uv init youtube-thumbnail-maker` 명령으로 새 프로젝트를 생성해 본다.
2. `uv add langchain[openai] langgraph openai[aiohttp]` 등으로 의존성을 추가해 본다.
3. `.env` 파일을 생성하고 `OPENAI_API_KEY`를 설정한다.
4. `uv run jupyter notebook`으로 Jupyter 환경을 실행해 본다.

---

### 15.1 오디오 추출 및 음성 인식

**커밋:** `4133ae6` "15.1 Audio Extraction and Transcription"

#### 주제 및 목표

유튜브 영상 파일(mp4)에서 오디오(mp3)를 추출하고, OpenAI의 Whisper 모델을 사용하여 음성을 텍스트로 변환하는 LangGraph 그래프를 구축한다. 이 단계에서 **StateGraph의 기본 구조**와 **노드(Node) 및 엣지(Edge) 개념**을 학습한다.

#### 핵심 개념 설명

**StateGraph**: LangGraph의 핵심 클래스로, 상태(State)를 중심으로 노드 간 데이터를 전달하는 그래프를 정의한다. 각 노드는 상태를 입력받아 상태의 일부를 업데이트하여 반환한다.

**TypedDict를 이용한 State 정의**: Python의 `TypedDict`를 사용하여 그래프 전체에서 공유되는 상태의 스키마를 타입 안전하게 정의한다. 이는 그래프의 "데이터 계약(data contract)"이라 할 수 있다.

**ffmpeg**: 오디오/비디오 처리를 위한 강력한 명령줄 도구이다. `subprocess.run()`을 통해 Python에서 외부 프로세스로 실행한다.

**OpenAI Whisper API**: 음성 인식(Speech-to-Text) 모델이다. 다양한 언어를 지원하며, `prompt` 파라미터를 통해 도메인 특화 용어의 인식 정확도를 높일 수 있다.

#### 코드 분석

**1단계: 상태(State) 정의**

```python
from langgraph.graph import END, START, StateGraph
from typing import TypedDict
import subprocess
from openai import OpenAI


class State(TypedDict):
    video_file: str       # 입력 영상 파일 경로
    audio_file: str       # 추출된 오디오 파일 경로
    transcription: str    # 음성 인식 결과 텍스트
```

`State`는 그래프 전체에서 공유되는 데이터 구조이다. 각 노드는 이 State의 일부 또는 전체를 읽고, 업데이트할 부분만 딕셔너리로 반환한다. 반환된 값은 기존 State에 **병합(merge)** 된다.

**2단계: 오디오 추출 노드**

```python
def extract_audio(state: State):
    output_file = state["video_file"].replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file,
    ]
    subprocess.run(command)
    return {
        "audio_file": output_file,
    }
```

이 노드에서 주목할 부분:

- **`-filter:a "atempo=2.0"`**: 오디오 재생 속도를 2배로 가속한다. Whisper API는 업로드 파일 크기 제한(25MB)이 있으므로, 오디오를 빠르게 만들어 파일 크기를 줄이는 실용적인 트릭이다. 음성 인식 품질에는 큰 영향을 주지 않는다.
- **`-y`**: 출력 파일이 이미 존재할 경우 덮어쓰기를 자동 허용하는 플래그이다.
- **반환값**: `{"audio_file": output_file}`만 반환하여 State의 `audio_file` 필드만 업데이트한다.

**3단계: 음성 인식 노드**

```python
def transcribe_audio(state: State):
    client = OpenAI()
    with open(state["audio_file"], "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en",
            prompt="Netherlands, Rotterdam, Amsterdam, The Hage",
        )
        return {
            "transcription": transcription,
        }
```

- **`model="whisper-1"`**: OpenAI의 Whisper 음성 인식 모델을 사용한다.
- **`response_format="text"`**: 단순 텍스트 형식으로 결과를 반환받는다. `"json"`, `"verbose_json"`, `"srt"`, `"vtt"` 등의 형식도 지원된다.
- **`language="en"`**: 언어를 영어로 지정하여 인식 정확도를 높인다.
- **`prompt` 파라미터**: 이것은 매우 유용한 기능이다. 영상에 등장할 것으로 예상되는 고유명사(도시 이름 등)를 미리 제공하여 Whisper가 해당 단어를 정확히 인식할 수 있도록 돕는다.

**4단계: 그래프 구성 및 실행**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_edge("transcribe_audio", END)

graph = graph_builder.compile()
```

LangGraph의 그래프 구성은 세 단계로 이루어진다:

1. **노드 추가** (`add_node`): 실행할 함수를 이름과 함께 등록한다.
2. **엣지 추가** (`add_edge`): 노드 간의 실행 순서(방향)를 정의한다. `START`와 `END`는 그래프의 시작과 끝을 나타내는 특수 노드이다.
3. **컴파일** (`compile`): 그래프를 실행 가능한 형태로 변환한다.

```python
graph.invoke({"video_file": "netherlands.mp4"})
```

`invoke()`에 초기 상태를 전달하여 그래프를 실행한다. 그래프는 `START` -> `extract_audio` -> `transcribe_audio` -> `END` 순서로 실행되며, 최종 상태가 반환된다.

#### 실습 포인트

1. `ffmpeg`가 시스템에 설치되어 있는지 확인한다 (`ffmpeg -version`).
2. 짧은 테스트 영상(1~2분)으로 먼저 파이프라인을 테스트해 본다.
3. `atempo` 값을 변경(1.5, 2.0, 3.0)하며 음성 인식 품질 차이를 비교해 본다.
4. `prompt` 파라미터에 다른 고유명사를 넣어보고 인식 정확도 변화를 관찰한다.

---

### 15.2 병렬 요약 노드 (Summarizer Nodes)

**커밋:** `a664f18` "15.2 Summarizer Nodes"

#### 주제 및 목표

긴 트랜스크립션 텍스트를 여러 청크(chunk)로 나누고, 각 청크를 **병렬로** 요약하는 **Map-Reduce 패턴**을 구현한다. LangGraph의 `Send` API와 `Annotated` 리듀서를 사용하여 동적 병렬 처리를 학습한다.

#### 핵심 개념 설명

**Map-Reduce 패턴**: 대규모 데이터를 처리하는 고전적인 분산 컴퓨팅 패턴이다.
- **Map 단계**: 데이터를 여러 조각으로 나누어 각각 독립적으로 처리한다 (병렬 요약).
- **Reduce 단계**: 개별 결과를 하나로 합친다 (다음 섹션의 mega_summary).

**Send API**: LangGraph에서 동적 병렬 처리를 가능하게 하는 핵심 메커니즘이다. `Send("노드이름", 데이터)`를 리스트로 반환하면, 해당 노드가 각 데이터에 대해 병렬로 실행된다.

**Annotated 리듀서**: `Annotated[list[str], operator.add]`와 같이 타입에 리듀서 함수를 지정한다. 여러 노드가 동일한 State 필드에 값을 반환할 때, 값을 어떻게 결합할지를 정의한다. `operator.add`는 리스트를 연결(concatenate)한다.

#### 코드 분석

**State 확장: Annotated 리듀서**

```python
from langgraph.types import Send
from typing_extensions import Annotated
import operator
import textwrap
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")


class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]  # 리듀서 추가!
```

`summaries` 필드에 `Annotated[list[str], operator.add]`를 사용한 것이 핵심이다. 이것이 없으면 여러 병렬 노드가 동시에 `summaries`에 값을 쓸 때 마지막 값만 남게 된다. `operator.add` 리듀서를 지정하면, 각 노드의 반환값이 기존 리스트에 **누적**된다.

> **주의**: `typing.Annotated`가 아니라 `typing_extensions.Annotated`를 사용해야 한다. `typing.Annotated`는 `Annotated(list[str], operator.add)`처럼 함수 호출 문법을 사용하면 `TypeError: Cannot instantiate typing.Annotated` 에러가 발생한다. 반드시 `Annotated[list[str], operator.add]`와 같이 대괄호 문법을 사용해야 한다.

**디스패처 함수: 동적 병렬 분기**

```python
def dispatch_summarizers(state: State):
    transcription = state["transcription"]
    chunks = []
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append({"id": i + 1, "chunk": chunk})
    return [Send("summarize_chunk", chunk) for chunk in chunks]
```

이 함수의 동작 원리:

1. **텍스트 분할**: `textwrap.wrap(transcription, 500)`은 긴 텍스트를 최대 500자 단위로 나눈다. 단어 경계를 존중하므로 단어가 중간에 잘리지 않는다.
2. **청크 구성**: 각 청크에 고유 ID를 부여하여 순서를 추적할 수 있게 한다.
3. **Send 리스트 반환**: `Send("summarize_chunk", chunk)`의 리스트를 반환하면, LangGraph가 `summarize_chunk` 노드를 각 청크에 대해 **병렬로** 실행한다.

이 함수는 `add_conditional_edges`의 라우터로 사용된다. 일반적인 조건부 엣지는 다음 노드 이름(문자열)을 반환하지만, `Send` 객체를 반환하면 동적 병렬 분기가 된다.

**요약 노드**

```python
def summarize_chunk(chunk):
    chunk_id = chunk["id"]
    chunk = chunk["chunk"]

    response = llm.invoke(
        f"""
        Please summarize the following text.

        Text: {chunk}
        """
    )
    summary = f"[Chunk {chunk_id}] {response.content}"
    return {
        "summaries": [summary],
    }
```

주목할 점:

- **파라미터가 `state`가 아니다**: `Send`로 호출되는 노드는 일반적인 State가 아니라, `Send`에서 전달한 데이터를 직접 파라미터로 받는다. 여기서는 `{"id": ..., "chunk": ...}` 딕셔너리가 된다.
- **리스트로 감싸서 반환**: `"summaries": [summary]`처럼 반드시 리스트로 감싸야 한다. `operator.add` 리듀서는 리스트끼리 연결하므로, 단일 값도 리스트로 감싸야 정상 동작한다.
- **`[Chunk N]` 접두사**: 각 요약에 청크 ID를 붙여 추후 순서를 파악할 수 있게 한다.

**그래프 구성: 조건부 엣지**

```python
graph_builder.add_node("summarize_chunk", summarize_chunk)

graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", END)
```

`add_conditional_edges`의 세 번째 인자 `["summarize_chunk"]`는 이 조건부 엣지에서 도달할 수 있는 노드 목록을 명시한다. LangGraph가 그래프를 검증할 때 이 정보를 사용한다.

#### 실습 포인트

1. `textwrap.wrap`의 청크 크기(500)를 변경하며 요약 품질 변화를 관찰한다.
2. `Send` 대신 순차적 처리(for 루프)로 구현한 뒤 실행 시간을 비교한다.
3. 다른 리듀서를 실험해 본다. 예를 들어 `operator.add` 대신 커스텀 함수를 사용할 수 있다.
4. LLM 모델을 `gpt-4o-mini`에서 `gpt-4o`로 변경하며 요약 품질을 비교한다.

---

### 15.3 썸네일 스케치 생성 노드 (Thumbnail Sketcher Nodes)

**커밋:** `13e2688` "15.3 Thumbnail Sketcher Nodes"

#### 주제 및 목표

이전 섹션의 개별 요약들을 하나의 종합 요약(`mega_summary`)으로 통합한 뒤, 그 요약을 바탕으로 5개의 서로 다른 썸네일 스케치를 **병렬로** 생성한다. OpenAI의 이미지 생성 API(`gpt-image-1`)를 활용하며, Map-Reduce 패턴을 이미지 생성에 적용한다.

#### 핵심 개념 설명

**Mega Summary (종합 요약)**: Map-Reduce의 Reduce 단계이다. 여러 청크 요약을 하나로 합쳐 전체 영상의 핵심 내용을 담은 하나의 요약을 만든다. 이 요약이 썸네일의 기반이 된다.

**OpenAI Image Generation API**: `gpt-image-1` 모델을 사용하여 텍스트 프롬프트로부터 이미지를 생성한다. `quality` 파라미터로 생성 품질(`low`/`medium`/`high`)을, `moderation` 파라미터로 콘텐츠 필터링 수준을 제어한다.

**이중 Map-Reduce**: 이 프로젝트에서는 Map-Reduce를 **두 번** 사용한다.
1. 첫 번째: 텍스트 청크 -> 병렬 요약 -> 종합 요약
2. 두 번째: 종합 요약 -> 병렬 썸네일 생성 -> 사용자 선택

#### 코드 분석

**State 확장**

```python
class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]    # 새로 추가
    thumbnail_sketches: Annotated[list[str], operator.add]   # 새로 추가
    final_summary: str                                        # 새로 추가
```

새로 추가된 필드:
- **`thumbnail_prompts`**: 각 썸네일 생성에 사용된 프롬프트를 저장한다. 나중에 사용자가 선택한 썸네일의 프롬프트를 재사용하기 위함이다.
- **`thumbnail_sketches`**: 생성된 썸네일 이미지 파일 경로를 저장한다.
- **`final_summary`**: 종합 요약 텍스트를 저장한다.

**종합 요약 노드 (mega_summary)**

```python
def mega_summary(state: State):
    all_summaries = "\n".join(state["summaries"])

    prompt = f"""
        You are given multiple summaries of different chunks from a video transcription.

        Please create a comprehensive final summary that combines all the key points.

        Individual summaries:

        {all_summaries}
    """

    response = llm.invoke(prompt)

    return {
        "final_summary": response.content,
    }
```

이 노드는 `summarize_chunk`들이 모두 완료된 후 실행된다. `state["summaries"]`에는 `operator.add` 리듀서에 의해 모든 청크 요약이 누적되어 있다. 이를 하나의 프롬프트로 합쳐서 LLM에게 종합 요약을 요청한다.

**아티스트 디스패처: 두 번째 병렬 분기**

```python
def dispatch_artists(state: State):
    return [
        Send(
            "generate_thumbnails",
            {
                "id": i,
                "summary": state["final_summary"],
            },
        )
        for i in [1, 2, 3, 4, 5]
    ]
```

`dispatch_summarizers`와 동일한 패턴이지만, 이번에는 5개의 고정된 작업을 생성한다. 각 작업은 동일한 `final_summary`를 받지만, 서로 다른 `id`를 가진다. LLM의 확률적 특성 덕분에 같은 요약에서도 매번 다른 시각적 컨셉이 생성된다.

**썸네일 생성 노드**

```python
def generate_thumbnails(args):
    concept_id = args["id"]
    summary = args["summary"]

    prompt = f"""
    Based on this video summary, create a detailed visual prompt for a YouTube thumbnail.

    Create a detailed prompt for generating a thumbnail image that would attract viewers. Include:
        - Main visual elements
        - Color scheme
        - Text overlay suggestions
        - Overall composition

    Summary: {summary}
    """

    response = llm.invoke(prompt)
    thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    filename = f"thumbnail_{concept_id}.jpg"

    with open(filename, "wb") as file:
        file.write(image_bytes)

    return {
        "thumbnail_prompts": [thumbnail_prompt],
        "thumbnail_sketches": [filename],
    }
```

이 노드는 두 단계로 동작한다:

1. **프롬프트 생성**: LLM(`gpt-4o-mini`)에게 영상 요약을 기반으로 시각적 프롬프트를 생성하게 한다. 주요 시각 요소, 색상 구성, 텍스트 오버레이, 전체 구도를 포함하도록 지시한다.
2. **이미지 생성**: 생성된 프롬프트를 `gpt-image-1` 모델에 전달하여 실제 이미지를 생성한다.

주요 파라미터:
- **`quality="low"`**: 스케치 단계이므로 저화질로 빠르게 생성한다. 비용과 시간을 절약하기 위함이다.
- **`moderation="low"`**: 콘텐츠 필터링을 느슨하게 설정하여 창의적인 결과를 허용한다.
- **`size="auto"`**: 이미지 크기를 모델이 자동으로 결정하게 한다.
- **`b64_json`**: 이미지가 Base64 인코딩된 JSON으로 반환된다. 이를 디코딩하여 JPG 파일로 저장한다.

**완성된 그래프 구성**

```python
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", END)

graph = graph_builder.compile()
```

그래프의 흐름:
1. `summarize_chunk` -> `mega_summary`: 모든 병렬 요약이 완료되면 종합 요약 노드로 진행한다.
2. `mega_summary` -> `dispatch_artists` -> `generate_thumbnails` x 5: 종합 요약이 완료되면 5개의 썸네일을 병렬 생성한다.

#### 실습 포인트

1. 썸네일 생성 개수(현재 5개)를 변경해 본다.
2. 이미지 생성 프롬프트를 수정하여 특정 스타일(일러스트, 사진, 미니멀 등)을 요청해 본다.
3. `quality` 파라미터를 `"medium"`이나 `"high"`로 변경하며 품질과 생성 시간을 비교한다.
4. 생성된 5개의 썸네일을 비교하여 프롬프트의 확률적 특성을 확인한다.

---

### 15.4 휴먼 피드백 루프 (Human Feedback)

**커밋:** `910cdef` "15.4 Human Feedback"

#### 주제 및 목표

AI가 생성한 5개의 썸네일 스케치 중 사용자가 하나를 선택하고, 추가 피드백을 제공할 수 있는 **Human-in-the-Loop** 패턴을 구현한다. LangGraph의 `interrupt` 함수와 `Command` 클래스, 그리고 `InMemorySaver` 체크포인터를 사용한다.

#### 핵심 개념 설명

**Human-in-the-Loop (HITL)**: AI 워크플로우 중간에 사람의 판단이나 입력을 삽입하는 패턴이다. 완전 자동화가 아니라, 중요한 결정 지점에서 사람의 검토와 승인을 받는다. 이는 AI의 출력 품질을 높이고, 사용자의 의도를 정확히 반영하는 데 필수적이다.

**interrupt()**: LangGraph에서 그래프 실행을 **일시 중지**하는 함수이다. 이 함수가 호출되면 그래프 실행이 멈추고, 현재 상태가 체크포인터에 저장된다. 사용자의 응답을 받은 후 `Command(resume=response)`로 실행을 재개한다.

**Command**: 중단된 그래프에 사용자의 응답을 전달하고 실행을 재개하는 클래스이다.

**InMemorySaver**: 그래프의 상태를 메모리에 저장하는 체크포인터이다. `interrupt`가 동작하려면 반드시 체크포인터가 필요하다. 그래프를 중단했다가 재개할 때, 이전 상태를 어딘가에 저장하고 있어야 하기 때문이다.

**thread_id**: 동일한 대화/세션을 식별하는 고유 ID이다. 체크포인터는 `thread_id`를 키로 사용하여 상태를 저장하고 복원한다.

#### 코드 분석

**새로운 임포트 및 설정**

```python
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
```

**State 확장**

```python
class State(TypedDict):
    # ... 기존 필드들 ...
    user_feedback: str    # 사용자의 수정 피드백
    chosen_prompt: str    # 사용자가 선택한 썸네일의 프롬프트
```

**휴먼 피드백 노드**

```python
def human_feedback(state: State):
    answer = interrupt(
        {
            "chosen_thumbnail": "Which thumbnail do you like the most?",
            "feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
        }
    )
    user_feedback = answer["user_feedback"]
    chosen_prompt = answer["chosen_prompt"]
    return {
        "user_feedback": user_feedback,
        "chosen_prompt": state["thumbnail_prompts"][chosen_prompt - 1],
    }
```

`interrupt()` 함수의 동작:

1. `interrupt()`의 인자로 전달된 딕셔너리는 사용자에게 보여줄 **질문/안내 메시지**이다. 이 딕셔너리는 그래프의 반환값에 포함되어 클라이언트(UI)에서 표시할 수 있다.
2. 그래프 실행이 여기서 **멈춘다**. 현재까지의 모든 상태(5개의 썸네일 이미지 경로, 프롬프트 등)가 체크포인터에 저장된다.
3. 사용자가 응답을 제공하면 `interrupt()`가 해당 응답을 반환하고 실행이 계속된다.

반환값에서 `state["thumbnail_prompts"][chosen_prompt - 1]`을 통해, 사용자가 선택한 번호(1~5)에 해당하는 원본 프롬프트를 가져온다. 이 프롬프트는 HD 썸네일 생성의 기반이 된다.

**HD 썸네일 생성 노드**

```python
def generate_hd_thumbnail(state: State):
    chosen_prompt = state["chosen_prompt"]
    user_feedback = state["user_feedback"]

    prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail
    prompt and create an enhanced version that incorporates the user's specific feedback.

    ORIGINAL PROMPT:
    {chosen_prompt}

    USER FEEDBACK TO INCORPORATE:
    {user_feedback}

    Create an enhanced prompt that:
        1. Maintains the core concept from the original prompt
        2. Specifically addresses and implements the user's feedback requests
        3. Adds professional YouTube thumbnail specifications:
            - High contrast and bold visual elements
            - Clear focal points that draw the eye
            - Professional lighting and composition
            - Optimal text placement and readability with generous padding from edges
            - Colors that pop and grab attention
            - Elements that work well at small thumbnail sizes
            - IMPORTANT: Always ensure adequate white space/padding between any text
              and the image borders
    """

    response = llm.invoke(prompt)
    final_thumbnail_prompt = response.content

    client = OpenAI()

    result = client.images.generate(
        model="gpt-image-1",
        prompt=final_thumbnail_prompt,
        quality="high",           # 고화질로 변경!
        moderation="low",
        size="auto",
    )

    image_bytes = base64.b64decode(result.data[0].b64_json)

    with open("thumbnail_final.jpg", "wb") as file:
        file.write(image_bytes)
```

이 노드의 설계 포인트:

1. **프롬프트 증강**: 사용자가 선택한 원본 프롬프트에 피드백을 반영하고, 전문적인 유튜브 썸네일 사양(고대비, 시선 집중 포인트, 텍스트 여백 등)을 추가한다.
2. **`quality="high"`**: 스케치 단계의 `"low"`와 달리 최종 결과물이므로 고화질로 생성한다.
3. **고정 파일명**: `thumbnail_final.jpg`로 저장하여 최종 결과물임을 명시한다.

**그래프 컴파일 (체크포인터 추가)**

```python
graph = graph_builder.compile(checkpointer=memory)
```

`checkpointer=memory`를 전달하여 그래프에 상태 저장 기능을 활성화한다. 이것이 없으면 `interrupt()`가 동작하지 않는다.

**실행: 2단계 호출**

```python
# 1단계: 그래프 실행 시작 (human_feedback에서 중단됨)
config = {
    "configurable": {
        "thread_id": "1",
    },
}

graph.invoke(
    {"video_file": "netherlands.mp4"},
    config=config,
)

# 2단계: 사용자 응답과 함께 재개
response = {
    "user_feedback": "Make sure the fella is smiling, remove any mention of audible, "
                     "or any logo, and give it a photo realistic, 3d style.",
    "chosen_prompt": 1,
}

graph.invoke(
    Command(resume=response),
    config=config,
)
```

첫 번째 `invoke`는 `extract_audio` -> `transcribe_audio` -> `summarize_chunk` x N -> `mega_summary` -> `generate_thumbnails` x 5 -> `human_feedback`까지 실행하고 중단된다. 사용자는 생성된 5개의 썸네일을 확인한 뒤, 원하는 번호와 수정 피드백을 제공한다.

두 번째 `invoke`에서 `Command(resume=response)`를 전달하면, `interrupt()`가 `response`를 반환하고 `human_feedback` 노드가 완료된다. 이어서 `generate_hd_thumbnail`이 실행되어 최종 고화질 썸네일이 생성된다.

> **중요**: 두 번의 `invoke` 호출에서 **동일한 `config`**(동일한 `thread_id`)를 사용해야 한다. 체크포인터가 `thread_id`를 키로 상태를 관리하기 때문이다.

#### 실습 포인트

1. 다양한 피드백을 시도해 본다 ("더 밝게", "텍스트 제거", "일러스트 스타일로" 등).
2. `thread_id`를 변경하며 별도의 세션을 실행해 본다.
3. `interrupt()`에 전달하는 메시지를 수정하여 사용자에게 더 상세한 안내를 제공해 본다.
4. 체크포인터를 `SqliteSaver`로 변경하여 영구 저장을 실험해 본다.

---

### 15.5 HD 썸네일 생성 및 프로덕션 배포

**커밋:** `257d1b8` "15.5 HD Thumbnail Generation"

#### 주제 및 목표

Jupyter Notebook에서 개발한 코드를 **프로덕션 배포 가능한 형태**로 변환한다. `graph.py` 독립 모듈을 생성하고, `langgraph.json` 설정 파일을 통해 **LangGraph CLI**로 서비스할 수 있도록 구성한다.

#### 핵심 개념 설명

**LangGraph CLI (langgraph-cli)**: LangGraph 그래프를 로컬 서버로 실행하거나 클라우드에 배포할 수 있게 해주는 CLI 도구이다. `langgraph dev` 명령으로 로컬 개발 서버를 실행하면, REST API를 통해 그래프와 상호작용할 수 있다.

**langgraph.json**: LangGraph CLI의 설정 파일이다. 어떤 파일에 어떤 그래프가 정의되어 있는지, 의존성은 무엇인지, 환경 변수는 어디서 로드할지를 정의한다.

**Notebook에서 모듈로의 전환**: 개발 단계에서는 Jupyter Notebook이 편리하지만, 배포 시에는 `.py` 파일로 변환해야 한다. 이 과정에서 코드 구조를 정리하고 모듈화한다.

#### 코드 분석

**langgraph.json 설정 파일**

```json
{
    "dependencies": [
        "graph.py"
    ],
    "graphs": {
        "mr_thumbs": "./graph.py:graph"
    },
    "env": ".env"
}
```

- **`dependencies`**: 프로젝트의 의존성 파일을 지정한다. 여기서는 `graph.py` 자체를 의존성으로 지정한다. 일반적으로 `pyproject.toml`이 있으면 자동으로 의존성이 설치된다.
- **`graphs`**: 서비스할 그래프를 이름-경로 쌍으로 정의한다.
  - `"mr_thumbs"`: 그래프의 서비스 이름이다. API 엔드포인트에서 이 이름으로 접근한다.
  - `"./graph.py:graph"`: 파일 경로와 변수명을 콜론으로 구분하여 지정한다. `graph.py` 파일의 `graph` 변수를 서비스한다.
- **`env`**: 환경 변수를 로드할 `.env` 파일의 경로이다.

**graph.py: 완성된 프로덕션 코드**

```python
graph = graph_builder.compile(name="mr_thumbs")
```

Notebook 버전과의 차이점:
- `checkpointer=memory`가 제거되었다. LangGraph CLI가 자동으로 체크포인터를 관리하기 때문이다.
- `name="mr_thumbs"`로 그래프에 이름을 부여하여 식별성을 높인다.

**graph.py의 전체 구조**는 다음과 같다:

```python
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
import subprocess
from openai import OpenAI
import textwrap
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
import operator
import base64

llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
    video_file: str
    audio_file: str
    transcription: str
    summaries: Annotated[list[str], operator.add]
    thumbnail_prompts: Annotated[list[str], operator.add]
    thumbnail_sketches: Annotated[list[str], operator.add]
    final_summary: str
    user_feedback: str
    chosen_prompt: str

# --- 노드 함수들 (extract_audio, transcribe_audio, ...) ---

# --- 그래프 구성 ---
graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", END)

graph = graph_builder.compile(name="mr_thumbs")
```

**LangGraph CLI 실행 방법**

```bash
# 개발 서버 실행
langgraph dev

# 또는 인메모리 모드로 실행
langgraph dev --no-browser
```

실행하면 로컬 서버가 시작되고, LangGraph Studio UI에서 그래프를 시각적으로 확인하고, 실행하고, `interrupt` 지점에서 피드백을 제공할 수 있다.

#### 실습 포인트

1. `langgraph dev` 명령으로 로컬 서버를 실행하고 LangGraph Studio에서 그래프를 확인한다.
2. Studio UI에서 `video_file`을 입력하고 그래프를 실행해 본다.
3. `interrupt` 지점에서 Studio UI를 통해 피드백을 제공하고 결과를 확인한다.
4. 그래프에 새로운 노드(예: 워터마크 추가, 이미지 리사이징)를 추가해 본다.

---

## 3. 챕터 핵심 정리

### LangGraph 핵심 패턴

| 패턴 | 설명 | 사용 사례 |
|------|------|-----------|
| **StateGraph** | 상태 중심의 그래프 워크플로우 | 모든 LangGraph 프로젝트의 기본 |
| **Send (Map-Reduce)** | 동적 병렬 분기 및 결과 수집 | 텍스트 청크 병렬 요약, 다중 이미지 생성 |
| **Annotated 리듀서** | 병렬 노드의 결과를 하나로 합치는 전략 | `operator.add`로 리스트 누적 |
| **interrupt / Command** | Human-in-the-Loop 패턴 | 사용자 선택, 피드백 수집 |
| **Checkpointer** | 그래프 상태의 영구 저장 | interrupt 지원, 세션 관리 |
| **conditional_edges** | 조건부 라우팅 | Send 기반 동적 분기 |

### 사용된 외부 도구 및 API

| 도구/API | 용도 |
|----------|------|
| **ffmpeg** | 비디오에서 오디오 추출, 속도 조절 |
| **OpenAI Whisper** (`whisper-1`) | 음성 -> 텍스트 변환 |
| **OpenAI Chat** (`gpt-4o-mini`) | 텍스트 요약, 프롬프트 생성 |
| **OpenAI Image** (`gpt-image-1`) | 텍스트 -> 이미지 생성 |
| **LangGraph CLI** | 그래프 서비스 배포 |

### 아키텍처 설계 원칙

1. **점진적 복잡성 증가**: 단순한 2개 노드 그래프에서 시작하여 7개 노드의 복잡한 그래프로 점진적으로 확장했다.
2. **비용 효율적 설계**: 스케치 단계에서는 `quality="low"`를 사용하고, 최종 결과물에만 `quality="high"`를 적용하여 API 비용을 최적화했다.
3. **모듈화**: 각 노드가 하나의 명확한 책임을 가지도록 설계하여, 개별 노드의 수정이 다른 노드에 영향을 주지 않도록 했다.
4. **Human-in-the-Loop**: 완전 자동화가 아닌, 핵심 결정 지점에서 사용자의 판단을 반영하는 현실적인 워크플로우를 구현했다.

---

## 4. 실습 과제

### 과제 1: 기본 - 다국어 지원 추가 (난이도: ★★☆☆☆)

`transcribe_audio` 노드를 수정하여, State에 `language` 필드를 추가하고, 사용자가 영상의 언어를 지정할 수 있게 만들어라. 한국어(`"ko"`), 일본어(`"ja"`) 등 다양한 언어로 테스트해 보라.

```python
class State(TypedDict):
    video_file: str
    language: str  # 추가
    # ...
```

### 과제 2: 중급 - 요약 품질 개선 (난이도: ★★★☆☆)

현재 `textwrap.wrap`은 단순히 글자 수 기준으로 텍스트를 자른다. `tiktoken`을 사용하여 토큰 수 기준으로 청크를 나누도록 `dispatch_summarizers`를 개선하라. 또한, 각 청크 간에 약간의 오버랩(overlap)을 두어 문맥 손실을 줄여보라.

### 과제 3: 중급 - 썸네일 스타일 선택 (난이도: ★★★☆☆)

`human_feedback` 노드 **이전에** 추가 `interrupt`를 넣어, 사용자가 원하는 썸네일 스타일(사진 리얼리즘, 일러스트, 미니멀, 3D 렌더링 등)을 먼저 선택할 수 있게 만들어라. 선택된 스타일을 `generate_thumbnails`의 프롬프트에 반영하라.

### 과제 4: 고급 - 반복적 피드백 루프 (난이도: ★★★★☆)

현재는 피드백을 한 번만 받을 수 있다. 사용자가 최종 HD 썸네일에 만족하지 않을 경우, 추가 피드백을 받아 다시 생성할 수 있는 **반복적 피드백 루프**를 구현하라. 사용자가 "완료"라고 응답할 때까지 `generate_hd_thumbnail` -> `interrupt` -> `generate_hd_thumbnail` 사이클을 반복하도록 조건부 엣지를 설계하라.

### 과제 5: 고급 - 전체 파이프라인 확장 (난이도: ★★★★★)

다음 기능들을 추가하여 파이프라인을 확장하라:

1. **유튜브 URL 입력**: `video_file` 대신 유튜브 URL을 받아, `yt-dlp` 등으로 자동 다운로드하는 노드를 추가하라.
2. **A/B 테스트 모드**: 최종 썸네일을 2개 생성하여 A/B 테스트용으로 제공하라.
3. **이미지 후처리**: Pillow 라이브러리를 사용하여 생성된 썸네일에 채널 로고 워터마크를 자동으로 추가하는 노드를 만들어라.
4. **결과 저장**: 생성된 모든 중간 결과물(요약, 프롬프트, 이미지)을 JSON 파일로 정리하여 저장하는 노드를 추가하라.

---

## 부록: 주요 API 레퍼런스

### LangGraph StateGraph API

```python
from langgraph.graph import END, START, StateGraph

# 그래프 생성
graph_builder = StateGraph(State)

# 노드 추가
graph_builder.add_node("이름", 함수)

# 엣지 추가 (순차)
graph_builder.add_edge("시작노드", "끝노드")

# 조건부 엣지 (라우터 함수 사용)
graph_builder.add_conditional_edges("시작노드", 라우터함수, ["가능한노드1", "가능한노드2"])

# 컴파일
graph = graph_builder.compile(checkpointer=체크포인터, name="그래프이름")

# 실행
result = graph.invoke(초기상태, config={"configurable": {"thread_id": "1"}})
```

### LangGraph Send API

```python
from langgraph.types import Send

# 디스패처 함수에서 Send 리스트 반환
def dispatcher(state: State):
    return [Send("노드이름", 데이터) for 데이터 in 데이터리스트]
```

### LangGraph interrupt / Command

```python
from langgraph.types import interrupt, Command

# 노드 내에서 실행 중단
def my_node(state: State):
    answer = interrupt({"질문": "사용자에게 보여줄 메시지"})
    # answer는 Command(resume=응답)에서 전달한 값
    return {"field": answer["key"]}

# 실행 재개
graph.invoke(Command(resume=응답), config=config)
```

### OpenAI Image Generation API

```python
from openai import OpenAI
import base64

client = OpenAI()
result = client.images.generate(
    model="gpt-image-1",
    prompt="이미지 설명",
    quality="low" | "medium" | "high",
    moderation="low" | "auto",
    size="auto" | "1024x1024" | "1792x1024",
)
image_bytes = base64.b64decode(result.data[0].b64_json)
```
