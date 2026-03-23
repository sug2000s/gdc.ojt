# Assetization System - 전체 시스템 맵

## 서비스 구성 (9개 레포지토리)

```
                        [사용자]
                          │
                    ┌─────┴─────┐
                    │           │
              [PC/모바일 브라우저]
                        │
               ┌────────┴────────┐
               │ assetization_   │
               │ mobile          │
               │ (React/Vite)    │
               │ 프론트엔드      │
               └────────┬────────┘
                        │
              ┌─────────┴─────────┐
              │ assetization_auth │
              │ (Python/FastAPI)  │
              │ 인증 게이트웨이   │
              └─────────┬─────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────┴────────┐ ┌────┴─────┐ ┌───────┴────────┐
│ assetization_  │ │ asset_   │ │ assetization_  │
│ orchestrator   │ │ api      │ │ mcp            │
│ (FastAPI+      │ │ (FastAPI)│ │ (FastAPI+      │
│  LangGraph)    │ │ Legacy   │ │  MCP+LangGraph)│
│ 채팅/검색 엔진 │ │ API      │ │ 리스크 분석    │
└───────┬────────┘ └────┬─────┘ └───────┬────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
              ┌─────────┴─────────┐
              │ keylook-officeplus│
              │ (Python/FastAPI)  │
              │ 하이브리드 검색   │
              │ 임베딩/리랭킹     │
              └─────────┬─────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   [Elasticsearch] [PostgreSQL]    [Redis]
        │               │               │
   [MSSQL]         [Redshift]      [Kafka]
        │                               │
   [AWS S3]                    [Azure OpenAI]
```

## 서비스 요약

| # | 레포지토리 | 언어/프레임워크 | 역할 | 포트 |
|---|-----------|---------------|------|------|
| 1 | `assetization_mobile` | React 19 + Vite + TS | 프론트엔드 | 5173 |
| 2 | `assetization_auth` | Python 3.13 + FastAPI | 인증 게이트웨이 (SSO/EMS) | 9090/8080 |
| 3 | `assetization_orchestrator` | Python 3.13 + FastAPI + LangGraph | AI 채팅/검색 오케스트레이션 | 80 |
| 4 | `assetization_mcp` | Python 3.13 + FastAPI + MCP | 리스크 분석 (MCP 서버) | 8080/9090 |
| 5 | `assetization_datacenter` | Python 3.9 + FastAPI | 히스토리/태그/컬렉션/배치 | 80 |
| 6 | `assetization_api` | Python 3.13 + FastAPI | 레거시 API (주간보고 등) | 8000 |
| 7 | `keylook-officeplus` | Python 3.9+ + FastAPI | 하이브리드 검색 엔진 | 8001-8006 |
| 8 | `keylook_script` | Shell | KeyLook Docker 래퍼 | - |

## 인프라 구성

| 인프라 | 용도 | 비고 |
|--------|------|------|
| **Elasticsearch 8.x** | 문서 검색, 벡터 검색, 로깅 | BM25 + Dense + Sparse |
| **PostgreSQL** | 채팅 히스토리, LangGraph 체크포인터 | asyncpg |
| **Redis** | 세션 관리, 캐시 | Cluster/Standalone |
| **MSSQL** | 사용자 마스터(CommonDB), 리스크(RMDB) | pyodbc/pymssql |
| **Amazon Redshift** | 검색 로그 분석 | Data Warehouse |
| **AWS S3** | 문서 저장, 인덱싱 데이터 | LocalStack (로컬) |
| **Kafka** | 이벤트 스트리밍, 접근 로그 | aiokafka |
| **Azure OpenAI** | LLM (GPT-4o/4.1) | 이중 엔드포인트 HA |

## 핵심 데이터 흐름

### 1. 인증 플로우
```
사용자 → SSO(OktaPlus) → assetization_auth → Redis(ASSET_SID) → 세션 생성
모바일 → EMS Token → assetization_auth → AES 복호화 → Redis → 세션 생성
```

### 2. 검색/채팅 플로우
```
사용자 쿼리 → orchestrator(chat_summarize)
  → LangGraph ReAct Agent
    → search_keylook tool → keylook-officeplus → Elasticsearch
    → search_risk tool → assetization_mcp → MSSQL(RMDB)
  → Azure OpenAI (GPT) → SSE 스트리밍 응답
  → Elasticsearch 로깅
```

### 3. 문서 인덱싱 플로우
```
S3 문서 → datacenter(add_indexing) → keylook-officeplus
  → Mecab 한국어 토큰화
  → BGE-M3 임베딩 생성
  → Elasticsearch 인덱싱 (passages + orgdoc)
```
