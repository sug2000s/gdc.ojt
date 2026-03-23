# 서비스 카탈로그 - 전체 소스 경로 정리

## 1. assetization_mobile (모바일 프론트엔드)

**경로**: `/Users/ryu/assetization_mobile`
**스택**: React 19 + TypeScript + Vite + Zustand + Tailwind CSS

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `src/api/apis.ts` | 1480 | 모든 API 통합 (30+ 엔드포인트) |
| `src/main.tsx` | - | 앱 진입점 |
| `src/router/Router.tsx` | - | 라우팅 (/, /search, /elastic) |
| `src/pages/search/SearchPage.tsx` | 37 | 메인 검색/채팅 페이지 |
| `src/pages/landing/LandingPage.tsx` | - | 인증 랜딩 |
| `src/store/chatStore.ts` | 178 | 채팅 상태관리 (TOKEN_LIMIT=32000) |
| `src/store/userStore.ts` | - | 사용자 정보 |
| `src/model/chat.ts` | 130 | 채팅 타입 정의 |
| `src/plugins/nativePlugins.ts` | - | 모바일 네이티브 브릿지 |
| `src/i18n.ts` | - | 한국어 i18n |
| `mock/server.js` | 29822 | Express Mock 서버 |
| `vite.config.ts` | 38 | 빌드 설정 (base: /react_assetization/) |

### 주요 디렉토리
- `src/store/` - 17개 Zustand 스토어
- `src/pages/search/components/` - 45+ UI 컴포넌트
- `src/components/` - 공용 컴포넌트 (alert, markdown, icon)
- `src/locales/ko/` - 한국어 번역

---

## 2. assetization_authsso (PC 웹 프론트엔드 - Legacy)

**경로**: `/Users/ryu/assetization_authsso`
**스택**: Java 11 + Spring Boot 2.6.6 + JSP + jQuery

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `src/main/java/.../SearchController.java` | 276 | 검색 API 컨트롤러 |
| `src/main/java/.../MobileSearchController.java` | 155 | 모바일 검색 |
| `src/main/java/.../SearchService.java` | 117 | 검색 비즈니스 로직 |
| `src/main/java/.../SSOFilter.java` | 231 | SSO 인증 필터 |
| `src/main/java/.../LoginController.java` | 277 | 로그인 처리 |
| `src/main/java/.../SecurityConfig.java` | 167 | Spring Security 설정 |
| `src/main/java/.../SessionCreateInterceptor.java` | 223 | 세션 생성 인터셉터 |
| `src/main/java/.../UserMapper.xml` | 142 | MyBatis 사용자 쿼리 |
| `src/main/resources/application.yml` | 406 | 멀티 프로파일 설정 |
| `pom.xml` | 265 | Maven 의존성 |

### 프론트엔드 JavaScript (8852줄)
- `assetization_ai_search_summerize_new.js` (2184줄) - SSE 스트리밍 검색
- `assetization_ai_lnb.js` (885줄) - 사이드바 UI
- `assetization_ai_collection.js` (854줄) - 컬렉션 관리
- `assetization_ai_search_external.js` (753줄) - 외부(GPT) 검색
- `assetization_search_cache.js` (545줄) - LocalStorage 캐시

### 내장 Python 게이트웨이
- `assetization_auth_gateway/` - FastAPI 기반 인증 게이트웨이 (assetization_auth의 원형)

---

## 3. assetization_auth (인증 게이트웨이)

**경로**: `/Users/ryu/assetization_auth`
**스택**: Python 3.13 + FastAPI + Redis + MSSQL

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `app/main.py` | 163 | FastAPI 진입점, SPA 서빙 |
| `app/config.py` | 246 | 환경 설정 관리 |
| `app/api/system_routes.py` | 372 | SSO 콜백, 로그인, agent-auth |
| `app/api/search_routes.py` | 113 | 세션 체크, ERP 연동 |
| `app/api/mobile_routes.py` | 171 | EMS 모바일 인증 |
| `app/middleware/sso_middleware.py` | 73 | SSO 인증 필터 |
| `app/middleware/session_middleware.py` | 94 | Redis 세션 생성/복원 |
| `app/middleware/client_check.py` | 86 | IP/토큰 검증 |
| `app/middleware/logging_middleware.py` | 42 | Kafka 접근 로깅 |
| `app/models/user.py` | 161 | UserModel, MobileUserModel |
| `app/services/user_service.py` | 116 | CommonDB 사용자 조회 |
| `app/services/mobile_service.py` | 95 | EMS 토큰 인증 |
| `app/utils/crypto.py` | 78 | AES/CBC 암복호화 |

### 미들웨어 실행 순서
1. LoggingMiddleware (최외곽) → Kafka 로깅
2. ClientCheckMiddleware → IP/토큰 검증
3. SSOMiddleware → SSO 인증
4. SessionMiddleware (최내곽) → Redis 세션

---

## 4. assetization_orchestrator (AI 채팅/검색 엔진)

**경로**: `/Users/ryu/assetization_orchestrator`
**스택**: Python 3.13 + FastAPI + LangGraph + LangChain + Azure OpenAI

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `app/main.py` | 182 | FastAPI 진입점, Alembic 마이그레이션 |
| `app/features/orchestrator/service.py` | 1000+ | **핵심** LangGraph V2 ReAct, 파일 업로드 |
| `app/features/orchestrator/router.py` | 713 | 11개 API (chat, context, files) |
| `app/features/orchestrator/langchain_extension/summarize_callback_handler.py` | 400+ | 토큰 스트리밍, ES 로깅 |
| `app/features/search/router.py` | 85 | 검색 전용 엔드포인트 |
| `app/features/es_logs/router.py` | 162 | ES 로그 조회 |
| `app/security/auth_filter.py` | 132 | SSO 세션 검증 |
| `app/storage/db/schema.py` | 61 | chat_files, tag, collection 테이블 |
| `app/util/keylook_api.py` | 150+ | KeyLook HTTP 클라이언트 |
| `app/util/llm_connection_pool.py` | 62 | LLM 라운드로빈 선택 |
| `app/util/file_extractor.py` | 100+ | PDF/DOCX/XLSX/이미지 추출 |
| `app/util/mcp_risk_client.py` | 80+ | MCP 리스크 서비스 호출 |

### LangGraph V2 ReAct 구조
```
START → agent_node → tools_condition ┐
                                      ├→ tools_node → agent_node (loop)
                                      └→ END
Tools: search_keylook(), search_risk()
Checkpointer: PostgreSQL (AsyncPostgresSaver)
```

---

## 5. assetization_mcp (리스크 분석 MCP 서버)

**경로**: `/Users/ryu/assetization_mcp`
**스택**: Python 3.13 + FastAPI + FastMCP + LangGraph + Azure OpenAI

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `app.py` | 1402 | FastAPI 메인 앱 |
| `multi_server_client.py` | 961 | MCP 클라이언트 오케스트레이션 |
| `improved_state_orchestrator.py` | 729 | LangGraph 워크플로우 |
| `orchestrator_core.py` | 362 | 코어 유틸리티 |
| `session_manager.py` | 393 | Redis 세션 관리 |
| `session_middleware.py` | 255 | 세션 미들웨어 |
| `auth_filter.py` | 237 | 인증 필터 |
| `config.py` | 308 | 환경 설정 |
| `kafka_logging_helper.py` | 190 | Kafka 로깅 |

### 6개 MCP 서버

| 서버 | 파일 | 줄수 | 역할 |
|------|------|------|------|
| Intent Analyzer | `servers/intent_analyzer_server.py` | 515 | 쿼리 의도 분석 |
| Elasticsearch | `servers/elasticsearch_server.py` | 703 | 벡터/키워드 검색 |
| Keylook | `servers/keylook_server.py` | 614 | 임베딩 생성 |
| MSSQL | `servers/mssql_server.py` | 823 | 프로젝트/리스크 DB 조회 |
| PMS | `servers/pms_server.py` | 334 | PMS 첨부파일 조회 |
| Orchestrator | `servers/orchestrator_server.py` | 429 | 메인 오케스트레이터 |

### LangGraph 워크플로우 단계
```
INTENT_ANALYSIS → EMBEDDING → VECTOR_SEARCH → DIRECT_QUERY
→ METADATA_FETCH → RESPONSE_GENERATION → ERROR_HANDLING
```

---

## 6. assetization_datacenter (데이터 관리)

**경로**: `/Users/ryu/assetization_datacenter`
**스택**: Python 3.9 + FastAPI + SQLAlchemy + PostgreSQL + Redshift

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `app/main.py` | 139 | FastAPI 진입점 |
| `app/features/batch/service.py` | 657 | 배치 작업 실행 |
| `app/features/collection/service.py` | 540 | 컬렉션 CRUD |
| `app/features/tag/service.py` | 470 | 태그 CRUD |
| `app/features/history/service.py` | 432 | 채팅 히스토리 (Redshift) |
| `app/features/indexer/service.py` | 402 | 인덱싱 작업 |
| `app/util/keylook_api.py` | 412 | KeyLook REST 클라이언트 |
| `app/util/query/query_manager.py` | 518 | ES 인덱스 관리 |
| `app/storage/db/schema.py` | 198 | 14개 ORM 모델 |

### 7개 기능 모듈

| 모듈 | 경로 prefix | 역할 |
|------|------------|------|
| indexer | `/apis/data/indexer` | 문서 인덱싱/삭제/메타 업데이트 |
| analytics | `/apis/data/analytics` | 검색 통계, 유사 쿼리 |
| feedback | `/apis/data/feedback` | 피드백 (Good/Bad) |
| history | `/apis/data/history` | 채팅 히스토리 (Redshift 연동) |
| tag | `/apis/data/tag` | 태그 관리 |
| collection | `/apis/data/collection` | 컬렉션 관리 (9개 엔드포인트) |
| batch | `/apis/data/batch` | 배치 작업 |

---

## 7. assetization_api (레거시 API)

**경로**: `/Users/ryu/assetization_api`
**스택**: Python 3.13 + FastAPI + LangChain + LangGraph

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `app/main.py` | 81 | FastAPI 진입점 |
| `app/services/weekly_service.py` | 758 | 주간보고 요약 (LangGraph map-reduce) |
| `app/api/endpoints/assetization_router.py` | 359 | 통합 검색 API |
| `app/api/endpoints/weekly_router.py` | 114 | 주간보고 엔드포인트 |
| `app/api/endpoints/prm_service_router.py` | 200 | PRM 리스크 평가 |
| `app/api/endpoints/pa2_service_router.py` | 136 | PA2.0 서비스 |
| `app/core/auth.py` | 227 | API 키 + 직원 인증 |
| `app/core/config.py` | 77 | 환경 설정 |

### API 라우트 (`/apis/legacy/`)
- `/weekly/*` - 주간보고 요약/초안 생성 (SSE 스트리밍)
- `/assetization/search` - 통합 검색 (KeyLook 연동)
- `/prm/fetch-data` - PM/SA 리스크 평가
- `/pa2/search` - PA2.0 검색

---

## 8. keylook-officeplus (하이브리드 검색 엔진)

**경로**: `/Users/ryu/keylook-officeplus`
**스택**: Python 3.9+ + FastAPI + FlagEmbedding + Elasticsearch 8.17

### 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `src/keylook/main.py` | 979 | FastAPI 앱, 17개 엔드포인트 |
| `src/keylook/core/indexer.py` | 1470 | 문서 CRUD, 청킹, 임베딩 |
| `src/keylook/core/es_connector.py` | 1179 | ES 쿼리 빌딩, 하이브리드 검색 |
| `src/keylook/core/retriever.py` | 361 | 검색 오케스트레이션 |
| `src/keylook/core/model_manager.py` | 323 | BGE-M3 모델 로딩 |
| `src/keylook/core/chunker.py` | 322 | Mecab 한국어 청킹 |
| `src/keylook/core/reranker.py` | 200 | Cross-encoder 리랭킹 |
| `src/keylook/core/project_manager.py` | 250 | 프로젝트 CRUD |
| `src/keylook/schemas.py` | 584 | 21개 Pydantic 모델 |
| `src/keylook/config.py` | 117 | 환경 설정 |

### 검색 파이프라인
```
쿼리 → BGE-M3 임베딩 → ES 하이브리드 쿼리
  BM25 (weight=3) + Sparse (weight=1) + Dense (weight=200)
  → 결과 병합 (sum/rescore/RRF)
  → [선택] Cross-encoder 리랭킹
  → 결과 반환
```

### 모델
- **Dense**: BGE-M3 (1024차원, cosine)
- **Sparse**: BERT-ATT (토큰별 가중치)
- **Reranker**: BGE-Reranker-v2-m3

---

## 9. keylook_script (Docker 래퍼)

**경로**: `/Users/ryu/keylook_script`
**내용**: `run.sh` (13줄) - keylook-officeplus Docker 실행 스크립트

---

## 전체 코드 규모

| 서비스 | 언어 | 코드 줄수 (약) |
|--------|------|---------------|
| assetization_mobile | TypeScript/TSX | 164파일, ~10,000줄 |
| assetization_authsso | Java + JS | ~12,000줄 |
| assetization_auth | Python | ~2,000줄 |
| assetization_orchestrator | Python | ~5,500줄 |
| assetization_mcp | Python | ~8,700줄 |
| assetization_datacenter | Python | ~8,400줄 |
| assetization_api | Python | ~2,100줄 |
| keylook-officeplus | Python | ~7,300줄 |
| **합계** | | **~56,000줄** |
ㅇ