# Week 1 실습 평가 / Week 1 Homework Evaluation / Đánh giá bài tập Tuần 1

> 평가일: 2026-03-27 / Evaluation Date: 2026-03-27 / Ngày đánh giá: 2026-03-27

---

## 학생 목록 / Students / Danh sách sinh viên

| 학생 / Student | Git Author | 제출 커밋 / Commits |
|---|---|---|
| **Le Dang Cuong** | cuongledang_aau929 / Le Dang Cuong | `d77d19c` (chap1+2), `c685778` (chap4 fix), `a9f8c36` (FastAPI), `03aeb53` (chap3-5), `151f296`~`ff34e10` (DocumentGPT) |
| **Tran Nu Khoi Nguyen** | nguyentnk2_aau929 / Tran Nu Khoi Nguyen | `d293071` (FastAPI), `8a0b8f1` (chap1,2), `1731acf` (full homework upload) |

---

## 1. Le Dang Cuong (cuongld)

### 1.1 제출 현황 / Submission Status / Tình trạng nộp bài

| 과제 / Assignment | 상태 / Status | 비고 / Notes |
|---|---|---|
| Notebook chap1 (LangChain 기초) | ✅ 완료 | 번역 체인 + 퀴즈 생성 체인 |
| Notebook chap2 (Few-Shot & 캐싱) | ✅ 완료 | Few-shot 번역 + SQLite 캐시 |
| Notebook chap3 (메모리) | ✅ 완료 | 3가지 메모리 비교 + Python 튜터 챗봇 |
| Notebook chap4 (RAG) | ✅ 완료 | Stuff vs MapReduce 비교 + 자동 평가 |
| Notebook chap5 | ❌ 미제출 | `load_dotenv()`만 존재 |
| DocumentGPT (Streamlit) | ✅ 완료 | 모델 선택, 스트리밍, 에러 처리 포함 |
| PrivateGPT ~ InvestorGPT | ❌ 미구현 | 빈 stub 또는 title만 존재 |
| Bulletin Board API (FastAPI) | ✅ 완료 | 인증 + CRUD 완전 구현 |

### 1.2 상세 평가 / Detailed Evaluation / Đánh giá chi tiết

#### Notebooks (chap1~4) — 9/10

**강점 / Strengths:**
- LCEL 파이프 문법을 정확하게 사용하며 체인 합성(linear, fan-out-then-merge) 패턴을 이해
- chap3: 10턴 대화로 Buffer/Window/Summary 메모리를 비교하는 **실험 설계**가 뛰어남. Window(k=3)가 초기 정보를 잃는 것을 실제로 증명 (buffer=256 tokens, window=93, summary=243)
- chap4: Stuff vs MapReduce를 토큰 사용량과 4가지 기준(정확성, 완전성, 명확성, 근거성)으로 **LLM-as-judge 자동 평가** 구현 — 과제 범위를 초과하는 우수한 시도
- Few-shot 예제를 격식체/캐주얼/문학체로 직접 설계하여 톤/레지스터 이해도를 보여줌

**개선점 / Areas for Improvement:**
- MapReduce 체인에서 `StrOutputParser` 누락으로 raw `AIMessage` 출력
- chap5 미제출

#### DocumentGPT (Streamlit) — 8/10

**강점:**
- `ChatCallbackHandler`로 스트리밍 구현, `@st.cache_resource`로 임베딩 캐시
- Azure OpenAI 배포명 매핑 (`MODEL_DEPLOYMENT_ENV` dict) — 실무 수준의 패턴
- .txt/.pdf/.docx 다중 파일 형식 지원, 사이드바 모델 선택기

**개선점:**
- PrivateGPT ~ InvestorGPT 5개 페이지 미구현 (전체 Streamlit 과제의 ~83% 미완성)

#### Bulletin Board API — 9/10

**강점:**
- 깔끔한 레이어드 아키텍처: routers / models / schemas / core (security, database, errors, config)
- `PostUpdate`에서 `model_dump(exclude_unset=True)` 사용 — Pydantic v2의 미묘한 차이(`exclude_none`과 다름)를 정확히 이해
- 소유권 검증(author_id != current_user.id), bcrypt 해싱, JWT 만료, 요청 로깅 미들웨어
- `pydantic_settings`로 설정 관리, `AliasChoices`로 DB URL 유연성 확보

**개선점:**
- `secret_key = "change-me"` 기본값 — 운영 환경에서 위험
- `@app.on_event("startup")` deprecated (lifespan context manager 권장)

### 1.3 종합 평가 / Overall Assessment

| 항목 / Category | 점수 / Score |
|---|---|
| Notebooks (chap1-4) | 9/10 |
| Notebook chap5 | 0/10 |
| DocumentGPT | 8/10 |
| PrivateGPT ~ InvestorGPT | 0/10 |
| Bulletin Board API | 9/10 |
| **종합 / Overall** | **B+** |

**총평:** 제출한 과제의 품질은 우수하며, 특히 실험 설계와 정량적 비교 분석에서 독립적 사고력이 돋보임. Bulletin Board API는 production-ready 수준의 구조. 그러나 Streamlit 페이지 5개와 chap5 노트북 미제출이 전체 완성도를 낮춤.

---

## 2. Tran Nu Khoi Nguyen (nguyen)

### 2.1 제출 현황 / Submission Status / Tình trạng nộp bài

| 과제 / Assignment | 상태 / Status | 비고 / Notes |
|---|---|---|
| Notebook chap1 (LangChain 기초) | ✅ 완료 | 번역 체인 (1.1) + 퀴즈 생성 (1.2) |
| Notebook chap2 (Few-Shot & 캐싱) | ✅ 완료 | Few-shot (2.1) + SQLite 캐시 (2.2) |
| Notebook chap3 (메모리) | ✅ 완료 | 3가지 메모리 비교 (3.1) + 튜터 챗봇 (3.2) |
| Notebook chap4 (RAG) | ✅ 완료 | FAISS RAG (4.1) + Stuff vs MapReduce (4.2) |
| Streamlit chap5 | ✅ 완료 | 기본 채팅 UI (task1) + 모델 선택 (task2) |
| Day2 퀴즈 (1.1~3.2) | ✅ 완료 | 6개 노트북 제출 |
| Bulletin Board API (FastAPI) | ✅ 완료 | 인증 + CRUD 구현 |

### 2.2 상세 평가 / Detailed Evaluation / Đánh giá chi tiết

#### Notebooks (chap1~4) — 8.5/10

**강점 / Strengths:**
- chap3.1 메모리 비교가 우수: Buffer(257 tokens) vs Window(k=3, 초기 대화 망각) vs Summary(198 tokens) — 정량적 비교로 메모리 효율 트레이드오프를 증명
- chap4.2에서 Stuff(643 tokens) vs MapReduce(935 tokens) 토큰 비교 분석
- `CacheBackedEmbeddings`와 `LocalFileStore` 사용으로 임베딩 캐시 구현
- RunnableParallel/dict 기반 체이닝 패턴을 정확히 이해 (`{"summary": summary_chain} | quiz_chain`)
- 커스텀 파서 클래스 (`OutputParser`, `QuizOutputParser`) 직접 구현

**개선점:**
- chap1.2의 summary_prompt에서 system message가 빈 문자열 `""` — 불필요
- chap3.2에서 `result.content` 대신 raw `result` 출력하여 메타데이터 포함

#### Streamlit (chap5) — 8/10

**강점:**
- task1: session_state 관리, 채팅 기록, 사이드바 clear 버튼 구현
- task2: `get_llm(model_name)` 팩토리 패턴으로 모델 선택기 구현, `.env` 활용

**개선점:**
- LLM 호출 실패 시 에러 처리 없음 (`try/except` + `st.error()` 권장)
- 매 메시지마다 새 `ChatOpenAI` 인스턴스 생성 — 비효율적

#### Bulletin Board API — 8/10

**강점:**
- FastAPI 표준 패턴 준수: `OAuth2PasswordRequestForm`, `OAuth2PasswordBearer`
- 인증/인가 분리, update/delete 모두 소유권 검증 적용
- Pydantic 스키마 분리 (Create/Update/Response), `EmailStr` 검증, `Field` 제약조건
- 베트남어 코드 주석으로 학습 과정의 자체 이해도 확인 가능

**개선점:**
- **N+1 쿼리 문제**: `post_router.py`에서 게시글 목록 조회 시 각 게시글마다 작성자를 개별 쿼리 (`db.query(User).filter(...)` per post) — SQLAlchemy `relationship()` + `joinedload` 사용 권장
- `PostResponse` 수동 구성이 모든 엔드포인트에서 반복 — 헬퍼 함수로 추출 필요
- `CORS allow_origins=["*"]` + `allow_credentials=True` 조합은 브라우저 CORS 스펙 위반
- `datetime.utcnow()` deprecated (Python 3.12+) — `datetime.now(timezone.utc)` 권장

### 2.3 종합 평가 / Overall Assessment

| 항목 / Category | 점수 / Score |
|---|---|
| Notebooks (chap1-4) | 8.5/10 |
| Streamlit (chap5) | 8/10 |
| Day2 퀴즈 | 8/10 |
| Bulletin Board API | 8/10 |
| **종합 / Overall** | **A-** |

**총평:** 모든 과제를 빠짐없이 제출한 유일한 학생. 각 챕터별로 파일을 체계적으로 분리(1.1/1.2, 2.1/2.2...)하여 학습 과정을 명확히 보여줌. 메모리 비교, 토큰 분석 등에서 독립적 사고 확인. N+1 쿼리 패턴과 에러 처리가 개선 포인트.

---

## 비교 분석 / Comparative Analysis / Phân tích so sánh

### 완성도 비교 / Completeness

| 과제 / Assignment | Cuong | Nguyen |
|---|---|---|
| Notebook chap1 | ✅ | ✅ |
| Notebook chap2 | ✅ | ✅ |
| Notebook chap3 | ✅ | ✅ |
| Notebook chap4 | ✅ | ✅ |
| Notebook/Streamlit chap5 | ❌ | ✅ |
| Day2 퀴즈 | — | ✅ |
| DocumentGPT | ✅ | — |
| Bulletin Board API | ✅ | ✅ |
| **완성률 / Completion Rate** | **~60%** | **~100%** |

### 코드 품질 비교 / Code Quality

| 항목 / Criteria | Cuong | Nguyen |
|---|---|---|
| 아키텍처 설계 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 에러 처리 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 보안 의식 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 독립적 사고 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 과제 완성도 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 핵심 차이점 / Key Differences

1. **Cuong**: 제출한 코드의 **품질과 깊이**가 뛰어남 (LLM-as-judge 평가, `exclude_unset` 패턴, 에러 헬퍼 모듈). 그러나 과제 완성도가 낮아 전체 평가에 영향.
2. **Nguyen**: **모든 과제를 완료**했으며, 각 파일을 체계적으로 분리. N+1 쿼리 같은 성능 이슈가 있지만 전체적인 학습 태도와 성실성이 우수.

---

## 종합 성적 / Final Grades / Điểm tổng kết

| 학생 / Student | 종합 / Grade | 한줄 평 / Summary |
|---|---|---|
| **Le Dang Cuong** | **B+** | 제출 코드의 품질은 최상급이나, 미제출 과제가 많아 감점. 깊이 > 넓이 |
| **Tran Nu Khoi Nguyen** | **A-** | 전 과제 완료, 체계적 학습, 견실한 코드. 성능 최적화 학습 필요. 넓이 + 깊이 균형 |

---

## 공통 개선 권장사항 / Shared Recommendations / Khuyến nghị chung

1. **SQLAlchemy Relationships 학습**: `relationship()` + `joinedload()`로 N+1 문제 해결
2. **Secret 관리**: 하드코딩된 기본값 대신 환경변수 필수화 또는 시작 시 검증
3. **FastAPI Lifespan**: deprecated `on_event` 대신 lifespan context manager 사용
4. **에러 처리**: Streamlit 앱에서 LLM 호출 실패 시 graceful error 표시
5. **Python 3.12+ 호환**: `datetime.utcnow()` → `datetime.now(timezone.utc)`
