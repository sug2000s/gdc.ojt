# 서비스 카탈로그 / Service Catalog / Danh mục dịch vụ - 전체 소스 경로 / Source Paths / Đường dẫn mã nguồn

## 1. assetization_mobile (프론트엔드 / Frontend / Giao diện)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_mobile`
**스택 / Stack / Công nghệ**: React 19 + TypeScript + Vite + Zustand + Tailwind CSS

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `src/api/apis.ts` | 1480 | 모든 API 통합 / All API integration / Tích hợp tất cả API (30+) |
| `src/main.tsx` | - | 앱 진입점 / App entry / Điểm vào ứng dụng |
| `src/router/Router.tsx` | - | 라우팅 / Routing / Định tuyến (/, /search, /elastic) |
| `src/pages/search/SearchPage.tsx` | 37 | 메인 검색/채팅 / Main search/chat / Trang tìm kiếm/chat chính |
| `src/pages/landing/LandingPage.tsx` | - | 인증 랜딩 / Auth landing / Trang xác thực |
| `src/store/chatStore.ts` | 178 | 채팅 상태관리 / Chat state / Quản lý trạng thái chat (TOKEN_LIMIT=32000) |
| `src/store/userStore.ts` | - | 사용자 정보 / User info / Thông tin người dùng |
| `src/model/chat.ts` | 130 | 채팅 타입 정의 / Chat types / Định nghĩa kiểu chat |
| `src/plugins/nativePlugins.ts` | - | 모바일 브릿지 / Mobile bridge / Cầu nối di động |
| `src/i18n.ts` | - | 한국어 i18n / Korean i18n / i18n tiếng Hàn |
| `mock/server.js` | 29822 | Express Mock 서버 / Mock server / Server giả lập |
| `vite.config.ts` | 38 | 빌드 설정 / Build config / Cấu hình build (base: /react_assetization/) |

### 주요 디렉토리 / Key Directories / Thư mục chính
- `src/store/` - 17개 Zustand 스토어 / 17 Zustand stores / 17 store Zustand
- `src/pages/search/components/` - 45+ UI 컴포넌트 / UI components / Thành phần UI
- `src/components/` - 공용 컴포넌트 / Shared components / Thành phần dùng chung (alert, markdown, icon)
- `src/locales/ko/` - 한국어 번역 / Korean translations / Bản dịch tiếng Hàn

---

## 2. assetization_auth (인증 게이트웨이 / Auth Gateway / Cổng xác thực)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_auth`
**스택 / Stack / Công nghệ**: Python 3.13 + FastAPI + Redis + MSSQL

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `app/main.py` | 163 | FastAPI 진입점, SPA 서빙 / Entry point, SPA serving / Điểm vào, phục vụ SPA |
| `app/config.py` | 246 | 환경 설정 관리 / Config management / Quản lý cấu hình |
| `app/api/system_routes.py` | 372 | SSO 콜백, 로그인 / SSO callback, login / SSO callback, đăng nhập |
| `app/api/search_routes.py` | 113 | 세션 체크, ERP 연동 / Session check, ERP / Kiểm tra phiên, ERP |
| `app/api/mobile_routes.py` | 171 | EMS 모바일 인증 / EMS mobile auth / Xác thực di động EMS |
| `app/middleware/sso_middleware.py` | 73 | SSO 인증 필터 / SSO auth filter / Bộ lọc xác thực SSO |
| `app/middleware/session_middleware.py` | 94 | Redis 세션 생성/복원 / Redis session create/restore / Tạo/khôi phục phiên Redis |
| `app/middleware/client_check.py` | 86 | IP/토큰 검증 / IP/token validation / Xác minh IP/token |
| `app/middleware/logging_middleware.py` | 42 | Kafka 접근 로깅 / Kafka access logging / Ghi log truy cập Kafka |
| `app/models/user.py` | 161 | UserModel, MobileUserModel |
| `app/services/user_service.py` | 116 | CommonDB 사용자 조회 / User lookup / Tra cứu người dùng |
| `app/services/mobile_service.py` | 95 | EMS 토큰 인증 / EMS token auth / Xác thực token EMS |
| `app/utils/crypto.py` | 78 | AES/CBC 암복호화 / Encryption/Decryption / Mã hóa/Giải mã |

### 미들웨어 실행 순서 / Middleware Order / Thứ tự middleware
1. LoggingMiddleware (최외곽 / outermost / ngoài cùng) → Kafka 로깅 / logging / ghi log
2. ClientCheckMiddleware → IP/토큰 검증 / validation / xác minh
3. SSOMiddleware → SSO 인증 / auth / xác thực
4. SessionMiddleware (최내곽 / innermost / trong cùng) → Redis 세션 / session / phiên

---

## 3. assetization_orchestrator (AI 채팅/검색 엔진 / AI Chat/Search Engine / Công cụ AI Chat/Tìm kiếm)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_orchestrator`
**스택 / Stack / Công nghệ**: Python 3.13 + FastAPI + LangGraph + LangChain + Azure OpenAI

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `app/main.py` | 182 | FastAPI 진입점 / Entry point / Điểm vào, Alembic migration |
| `app/features/orchestrator/service.py` | 1000+ | **핵심 / Core / Lõi** LangGraph V2 ReAct, 파일 업로드 / file upload / tải file |
| `app/features/orchestrator/router.py` | 713 | 11개 API / 11 APIs / 11 API (chat, context, files) |
| `app/features/orchestrator/langchain_extension/summarize_callback_handler.py` | 400+ | 토큰 스트리밍, ES 로깅 / Token streaming, ES logging / Streaming token, ghi log ES |
| `app/features/search/router.py` | 85 | 검색 전용 / Search only / Chỉ tìm kiếm |
| `app/features/es_logs/router.py` | 162 | ES 로그 조회 / ES log query / Truy vấn log ES |
| `app/security/auth_filter.py` | 132 | SSO 세션 검증 / SSO session validation / Xác minh phiên SSO |
| `app/storage/db/schema.py` | 61 | chat_files, tag, collection 테이블 / tables / bảng |
| `app/util/keylook_api.py` | 150+ | KeyLook HTTP 클라이언트 / client / máy khách |
| `app/util/llm_connection_pool.py` | 62 | LLM 라운드로빈 / Round-robin / Chọn luân phiên LLM |
| `app/util/file_extractor.py` | 100+ | PDF/DOCX/XLSX/이미지 추출 / extraction / trích xuất |
| `app/util/mcp_risk_client.py` | 80+ | MCP 리스크 서비스 호출 / Risk service call / Gọi dịch vụ rủi ro MCP |

### LangGraph V2 ReAct 구조 / Structure / Cấu trúc
```
START → agent_node → tools_condition ┐
                                      ├→ tools_node → agent_node (loop)
                                      └→ END
Tools: search_keylook(), search_risk()
Checkpointer: PostgreSQL (AsyncPostgresSaver)
```

---

## 4. assetization_mcp (리스크 분석 MCP / Risk Analysis MCP / Phân tích rủi ro MCP)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_mcp`
**스택 / Stack / Công nghệ**: Python 3.13 + FastAPI + FastMCP + LangGraph + Azure OpenAI

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `app.py` | 1402 | FastAPI 메인 앱 / Main app / Ứng dụng chính |
| `multi_server_client.py` | 961 | MCP 클라이언트 오케스트레이션 / Client orchestration / Điều phối client MCP |
| `improved_state_orchestrator.py` | 729 | LangGraph 워크플로우 / Workflow / Luồng công việc |
| `orchestrator_core.py` | 362 | 코어 유틸리티 / Core utilities / Tiện ích lõi |
| `session_manager.py` | 393 | Redis 세션 관리 / Session management / Quản lý phiên |
| `session_middleware.py` | 255 | 세션 미들웨어 / Session middleware / Middleware phiên |
| `auth_filter.py` | 237 | 인증 필터 / Auth filter / Bộ lọc xác thực |
| `config.py` | 308 | 환경 설정 / Config / Cấu hình |
| `kafka_logging_helper.py` | 190 | Kafka 로깅 / Logging / Ghi log |

### 6개 MCP 서버 / 6 MCP Servers / 6 Server MCP

| 서버 / Server / Server | 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|------|
| Intent Analyzer | `servers/intent_analyzer_server.py` | 515 | 쿼리 의도 분석 / Query intent analysis / Phân tích ý định truy vấn |
| Elasticsearch | `servers/elasticsearch_server.py` | 703 | 벡터/키워드 검색 / Vector/keyword search / Tìm kiếm vector/từ khóa |
| Keylook | `servers/keylook_server.py` | 614 | 임베딩 생성 / Embedding generation / Tạo embedding |
| MSSQL | `servers/mssql_server.py` | 823 | 프로젝트/리스크 DB 조회 / Project/risk DB query / Truy vấn DB dự án/rủi ro |
| PMS | `servers/pms_server.py` | 334 | PMS 첨부파일 조회 / PMS attachment query / Truy vấn đính kèm PMS |
| Orchestrator | `servers/orchestrator_server.py` | 429 | 메인 오케스트레이터 / Main orchestrator / Điều phối chính |

### LangGraph 워크플로우 / Workflow / Luồng công việc
```
INTENT_ANALYSIS → EMBEDDING → VECTOR_SEARCH → DIRECT_QUERY
→ METADATA_FETCH → RESPONSE_GENERATION → ERROR_HANDLING
```

---

## 5. assetization_datacenter (데이터 관리 / Data Management / Quản lý dữ liệu)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_datacenter`
**스택 / Stack / Công nghệ**: Python 3.9 + FastAPI + SQLAlchemy + PostgreSQL + Redshift

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `app/main.py` | 139 | FastAPI 진입점 / Entry point / Điểm vào |
| `app/features/batch/service.py` | 657 | 배치 작업 / Batch jobs / Tác vụ batch |
| `app/features/collection/service.py` | 540 | 컬렉션 CRUD / Collection CRUD / CRUD bộ sưu tập |
| `app/features/tag/service.py` | 470 | 태그 CRUD / Tag CRUD / CRUD tag |
| `app/features/history/service.py` | 432 | 채팅 히스토리 / Chat history / Lịch sử chat (Redshift) |
| `app/features/indexer/service.py` | 402 | 인덱싱 작업 / Indexing jobs / Tác vụ lập chỉ mục |
| `app/util/keylook_api.py` | 412 | KeyLook REST 클라이언트 / Client / Máy khách |
| `app/util/query/query_manager.py` | 518 | ES 인덱스 관리 / ES index management / Quản lý chỉ mục ES |
| `app/storage/db/schema.py` | 198 | 14개 ORM 모델 / 14 ORM models / 14 mô hình ORM |

### 7개 기능 모듈 / 7 Feature Modules / 7 Module chức năng

| 모듈 / Module / Module | 경로 / Path prefix / Tiền tố | 역할 / Role / Vai trò |
|------|------------|------|
| indexer | `/apis/data/indexer` | 문서 인덱싱/삭제 / Doc indexing/delete / Lập chỉ mục/xóa tài liệu |
| analytics | `/apis/data/analytics` | 검색 통계 / Search stats / Thống kê tìm kiếm |
| feedback | `/apis/data/feedback` | 피드백 / Feedback / Phản hồi (Good/Bad) |
| history | `/apis/data/history` | 채팅 히스토리 / Chat history / Lịch sử chat (Redshift) |
| tag | `/apis/data/tag` | 태그 관리 / Tag management / Quản lý tag |
| collection | `/apis/data/collection` | 컬렉션 관리 / Collection management / Quản lý bộ sưu tập (9 endpoints) |
| batch | `/apis/data/batch` | 배치 작업 / Batch jobs / Tác vụ batch |

---

## 6. assetization_api (레거시 API / Legacy API / API cũ)

**경로 / Path / Đường dẫn**: `/Users/ryu/assetization_api`
**스택 / Stack / Công nghệ**: Python 3.13 + FastAPI + LangChain + LangGraph

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `app/main.py` | 81 | FastAPI 진입점 / Entry point / Điểm vào |
| `app/services/weekly_service.py` | 758 | 주간보고 요약 / Weekly report summary / Tóm tắt báo cáo tuần (LangGraph map-reduce) |
| `app/api/endpoints/assetization_router.py` | 359 | 통합 검색 API / Unified search API / API tìm kiếm hợp nhất |
| `app/api/endpoints/weekly_router.py` | 114 | 주간보고 엔드포인트 / Weekly report endpoints / Endpoint báo cáo tuần |
| `app/api/endpoints/prm_service_router.py` | 200 | PRM 리스크 평가 / PRM risk assessment / Đánh giá rủi ro PRM |
| `app/api/endpoints/pa2_service_router.py` | 136 | PA2.0 서비스 / PA2.0 service / Dịch vụ PA2.0 |
| `app/core/auth.py` | 227 | API 키 + 직원 인증 / API key + employee auth / Xác thực API key + nhân viên |
| `app/core/config.py` | 77 | 환경 설정 / Config / Cấu hình |

### API 라우트 / Routes / Tuyến (`/apis/legacy/`)
- `/weekly/*` - 주간보고 요약/초안 / Weekly report summary/draft / Tóm tắt/bản nháp báo cáo tuần (SSE)
- `/assetization/search` - 통합 검색 / Unified search / Tìm kiếm hợp nhất (KeyLook)
- `/prm/fetch-data` - PM/SA 리스크 평가 / Risk assessment / Đánh giá rủi ro
- `/pa2/search` - PA2.0 검색 / Search / Tìm kiếm

---

## 7. keylook-officeplus (하이브리드 검색 엔진 / Hybrid Search Engine / Công cụ tìm kiếm lai)

**경로 / Path / Đường dẫn**: `/Users/ryu/keylook-officeplus`
**스택 / Stack / Công nghệ**: Python 3.9+ + FastAPI + FlagEmbedding + Elasticsearch 8.17

### 핵심 파일 / Key Files / File chính

| 파일 / File / File | 줄수 / Lines / Dòng | 역할 / Role / Vai trò |
|------|------|------|
| `src/keylook/main.py` | 979 | FastAPI 앱 / App / Ứng dụng, 17 endpoints |
| `src/keylook/core/indexer.py` | 1470 | 문서 CRUD, 청킹, 임베딩 / Doc CRUD, chunking, embedding / CRUD tài liệu, chia đoạn, embedding |
| `src/keylook/core/es_connector.py` | 1179 | ES 쿼리 빌딩 / ES query building / Xây dựng truy vấn ES |
| `src/keylook/core/retriever.py` | 361 | 검색 오케스트레이션 / Search orchestration / Điều phối tìm kiếm |
| `src/keylook/core/model_manager.py` | 323 | BGE-M3 모델 로딩 / Model loading / Tải mô hình |
| `src/keylook/core/chunker.py` | 322 | Mecab 한국어 청킹 / Korean chunking / Chia đoạn tiếng Hàn |
| `src/keylook/core/reranker.py` | 200 | Cross-encoder 리랭킹 / Reranking / Xếp hạng lại |
| `src/keylook/core/project_manager.py` | 250 | 프로젝트 CRUD / Project CRUD / CRUD dự án |
| `src/keylook/schemas.py` | 584 | 21개 Pydantic 모델 / 21 Pydantic models / 21 mô hình Pydantic |
| `src/keylook/config.py` | 117 | 환경 설정 / Config / Cấu hình |

### 검색 파이프라인 / Search Pipeline / Pipeline tìm kiếm
```
쿼리/Query/Truy vấn → BGE-M3 임베딩/Embedding → ES 하이브리드 쿼리/Hybrid query/Truy vấn lai
  BM25 (weight=3) + Sparse (weight=1) + Dense (weight=200)
  → 결과 병합/Result merge/Gộp kết quả (sum/rescore/RRF)
  → [선택/Optional/Tùy chọn] Cross-encoder 리랭킹/Reranking/Xếp hạng lại
  → 결과 반환/Return results/Trả kết quả
```

### 모델 / Models / Mô hình
- **Dense**: BGE-M3 (1024차원/dimensions/chiều, cosine)
- **Sparse**: BERT-ATT (토큰별 가중치 / per-token weights / trọng số theo token)
- **Reranker**: BGE-Reranker-v2-m3

---

## 8. keylook_script (Docker 래퍼 / Docker Wrapper / Trình bọc Docker)

**경로 / Path / Đường dẫn**: `/Users/ryu/keylook_script`
**내용 / Content / Nội dung**: `run.sh` (13줄/lines/dòng) - keylook-officeplus Docker 실행 / execution / thực thi

---

## 전체 코드 규모 / Total Code Size / Quy mô code tổng thể

| 서비스 / Service / Dịch vụ | 언어 / Language / Ngôn ngữ | 줄수 (약) / Lines (approx.) / Dòng (khoảng) |
|--------|------|---------------|
| assetization_mobile | TypeScript/TSX | 164 파일/files, ~10,000 |
| assetization_auth | Python | ~2,000 |
| assetization_orchestrator | Python | ~5,500 |
| assetization_mcp | Python | ~8,700 |
| assetization_datacenter | Python | ~8,400 |
| assetization_api | Python | ~2,100 |
| keylook-officeplus | Python | ~7,300 |
| **합계 / Total / Tổng** | | **~44,000** |
