# 전체 시스템 맵 / System Map / Bản đồ hệ thống

## 서비스 구성 / Service Architecture / Kiến trúc dịch vụ (8개 레포 / 8 repos / 8 repo)

```
                  [사용자 / User / Người dùng]
                          │
                  [PC/모바일 브라우저 / Browser / Trình duyệt]
                          │
               ┌──────────┴──────────┐
               │ assetization_mobile │
               │ (React 19 + Vite)   │
               │ 프론트엔드/Frontend │
               │ /Giao diện         │
               └──────────┬──────────┘
                          │
               ┌──────────┴──────────┐
               │ assetization_auth   │
               │ (Python/FastAPI)    │
               │ 인증/Auth/Xác thực  │
               └──────────┬──────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
┌──────┴───────┐  ┌───────┴──────┐  ┌───────┴───────┐
│ orchestrator │  │ asset_api    │  │ assetization_ │
│ (FastAPI+    │  │ (FastAPI)    │  │ mcp           │
│  LangGraph)  │  │ Legacy API   │  │ (FastAPI+MCP  │
│ 채팅/검색    │  │              │  │  +LangGraph)  │
│ Chat/Search  │  │              │  │ 리스크/Risk   │
│ Trò chuyện   │  │              │  │ Rủi ro        │
└──────┬───────┘  └───────┬──────┘  └───────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
               ┌──────────┴──────────┐
               │ keylook-officeplus  │
               │ (Python/FastAPI)    │
               │ 하이브리드 검색     │
               │ Hybrid Search      │
               │ Tìm kiếm lai       │
               └──────────┬──────────┘
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
  [Elasticsearch]   [PostgreSQL]        [Redis]
       │                  │                  │
  [MSSQL]           [Redshift]          [Kafka]
       │                                     │
  [AWS S3]                          [Azure OpenAI]
```

## 서비스 요약 / Service Summary / Tóm tắt dịch vụ

| # | 레포 / Repo / Repo | 스택 / Stack / Công nghệ | 역할 / Role / Vai trò | 포트 / Port / Cổng |
|---|-----------|---------------|------|------|
| 1 | `assetization_mobile` | React 19 + Vite + TS | 프론트엔드 / Frontend / Giao diện | 5173 |
| 2 | `assetization_auth` | Python 3.13 + FastAPI | 인증 게이트웨이 / Auth Gateway / Cổng xác thực (SSO/EMS) | 9090/8080 |
| 3 | `assetization_orchestrator` | Python 3.13 + FastAPI + LangGraph | AI 채팅/검색 / AI Chat/Search / Trò chuyện/Tìm kiếm AI | 80 |
| 4 | `assetization_mcp` | Python 3.13 + FastAPI + MCP | 리스크 분석 / Risk Analysis / Phân tích rủi ro (MCP) | 8080/9090 |
| 5 | `assetization_datacenter` | Python 3.9 + FastAPI | 히스토리/태그/컬렉션 / History/Tag/Collection / Lịch sử/Tag/Bộ sưu tập | 80 |
| 6 | `assetization_api` | Python 3.13 + FastAPI | 레거시 API / Legacy API / API cũ | 8000 |
| 7 | `keylook-officeplus` | Python 3.9+ + FastAPI | 하이브리드 검색 / Hybrid Search / Tìm kiếm lai | 8001-8006 |
| 8 | `keylook_script` | Shell | Docker 래퍼 / Docker Wrapper / Trình bọc Docker | - |

## 인프라 구성 / Infrastructure / Hạ tầng

| 인프라 / Infra / Hạ tầng | 용도 / Purpose / Mục đích | 비고 / Note / Ghi chú |
|--------|------|------|
| **Elasticsearch 8.x** | 문서/벡터 검색, 로깅 / Doc/Vector search, Logging / Tìm kiếm văn bản/vector, Ghi log | BM25 + Dense + Sparse |
| **PostgreSQL** | 채팅 히스토리, 체크포인터 / Chat history, Checkpointer / Lịch sử chat, Checkpointer | asyncpg |
| **Redis** | 세션 관리, 캐시 / Session, Cache / Phiên, Bộ nhớ đệm | Cluster/Standalone |
| **MSSQL** | 사용자 마스터, 리스크 / User master, Risk / Quản lý user, Rủi ro | CommonDB, RMDB |
| **Amazon Redshift** | 검색 로그 분석 / Search log analytics / Phân tích log tìm kiếm | Data Warehouse |
| **AWS S3** | 문서 저장 / Document storage / Lưu trữ tài liệu | LocalStack (로컬/local/nội bộ) |
| **Kafka** | 이벤트 스트리밍 / Event streaming / Truyền sự kiện | aiokafka |
| **Azure OpenAI** | LLM (GPT-4o/4.1) | 이중 엔드포인트 / Dual endpoint / Endpoint kép HA |

## 핵심 데이터 흐름 / Core Data Flows / Luồng dữ liệu chính

### 1. 인증 플로우 / Auth Flow / Luồng xác thực
```
사용자/User/Người dùng → SSO(OktaPlus) → assetization_auth → Redis(ASSET_SID)
  → 세션 생성 / Session created / Tạo phiên

모바일/Mobile/Di động → EMS Token → assetization_auth → AES 복호화/Decrypt/Giải mã
  → Redis → 세션 생성 / Session created / Tạo phiên
```

### 2. 검색/채팅 플로우 / Search/Chat Flow / Luồng tìm kiếm/trò chuyện
```
사용자 쿼리 / User query / Truy vấn người dùng
  → orchestrator (chat_summarize)
  → LangGraph ReAct Agent
    → search_keylook tool → keylook-officeplus → Elasticsearch
    → search_risk tool → assetization_mcp → MSSQL(RMDB)
  → Azure OpenAI (GPT) → SSE 스트리밍 응답 / SSE streaming / Phản hồi SSE
  → Elasticsearch 로깅 / Logging / Ghi log
```

### 3. 문서 인덱싱 플로우 / Document Indexing Flow / Luồng lập chỉ mục tài liệu
```
S3 문서/Documents/Tài liệu → datacenter(add_indexing) → keylook-officeplus
  → Mecab 한국어 토큰화 / Korean tokenization / Phân tách tiếng Hàn
  → BGE-M3 임베딩 생성 / Embedding generation / Tạo embedding
  → Elasticsearch 인덱싱 / Indexing / Lập chỉ mục (passages + orgdoc)
```
