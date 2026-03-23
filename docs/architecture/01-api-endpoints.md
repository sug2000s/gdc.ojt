# 전체 API 엔드포인트 맵

## 1. assetization_auth (인증)

### System Routes (prefix 없음)
| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 |
| GET | `/okta` | SSO 콜백 |
| GET | `/login` | 개발용 로그인 페이지 |
| GET | `/doLogin` | 개발용 수동 로그인 |
| GET | `/logout` | 로그아웃 |
| GET | `/tokencheck` | PC→모바일 토큰 생성 |
| GET | `/aisearch` | SSO 콜백 대안 |
| GET | `/agent-auth` | 데스크톱 Agent SSO |
| GET | `/agent-auth-complete/{state}` | Agent 세션 콜백 |

### Search/Auth Routes (API_PREFIX)
| Method | Path | 설명 |
|--------|------|------|
| GET | `/sessioninfo` | 현재 사용자 정보 |
| POST | `/auth/sessioncheck` | 세션 유효성 확인 |
| GET | `/tokencheck` | 모바일 세션 토큰 생성 |
| POST | `/search/getErpAssetAttachInfo` | ERP 첨부 presigned URL |

### Mobile Routes (`/mobile`)
| Method | Path | 설명 |
|--------|------|------|
| GET | `/mobile/sessioncheck` | EMS 토큰 인증 |
| GET | `/mobile/tokencheck` | AES 세션 토큰 검증 |
| GET | `/mobile/getErpAssetAttachInfo` | 모바일 ERP 첨부 |

---

## 2. assetization_orchestrator (`/apis/orche`)

### 채팅/검색
| Method | Path | 응답 | 설명 |
|--------|------|------|------|
| GET | `/health` | JSON | 헬스체크 |
| POST | `/chat_context` | JSON | 새 context_id 생성 (UUID) |
| POST | `/chat_fetch` | JSON | 쿼리 Redis 캐시 (step 1) |
| POST | `/chat_summarize_fetch` | JSON | 요약 요청 캐시 (step 1) |
| POST | `/chat_gpt_fetch` | JSON | GPT 요청 캐시 (step 1) |
| GET | `/chat_summarize` | SSE | **V2 LangGraph ReAct** (직접) |
| GET | `/chat_summarize_es` | SSE | 캐시된 쿼리 + V2 실행 (step 2) |
| GET | `/chat_gpt` | SSE | 검색 없는 GPT 응답 |
| GET | `/chat_gpt_es` | SSE | 캐시된 GPT 실행 (step 2) |
| GET | `/chat_es_summarize` | SSE | V1 Legacy 검색+요약 |
| GET | `/chat_prompt` | SSE | 커스텀 프롬프트 V2 ReAct |
| POST | `/context_reset` | JSON | 컨텍스트 리셋 |

### 파일 업로드
| Method | Path | 설명 |
|--------|------|------|
| POST | `/chat_upload` | 파일 업로드 (텍스트/PDF/이미지) |
| GET | `/chat_files` | 업로드된 파일 목록 |
| DELETE | `/chat_files/{file_id}` | 파일 삭제 |

### 검색 전용
| Method | Path | 설명 |
|--------|------|------|
| POST | `/search` | KeyLook 검색 (LLM 없이) |
| GET | `/search_url` | 문서 다운로드 URL |
| GET | `/related_chunks` | 관련 문서 청크 |

### 체크포인터
| Method | Path | 설명 |
|--------|------|------|
| POST | `/checkpointer/init` | PostgreSQL 체크포인터 초기화 |
| GET | `/checkpointer/status` | 체크포인터 상태 |

### ES 로그 (`/apis/orche/es_logs`)
| Method | Path | 설명 |
|--------|------|------|
| GET | `/search_rerank_log` | 검색 리랭크 로그 |
| GET | `/llm_prompt_log` | LLM 프롬프트 로그 |
| GET | `/search_rerank_log/{txn_id}` | 트랜잭션별 상세 |
| GET | `/llm_prompt_log/{txn_id}` | LLM 로그 상세 |

---

## 3. assetization_mcp (`/apis/mcp`)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 상세 헬스체크 (서비스별 상태) |
| POST | `/orchestrator/query` | **메인 쿼리** (LangGraph 오케스트레이터) |
| GET | `/mcptool/sse` | SSE 스트리밍 |
| POST | `/mcptool/messages` | 메시징 |

---

## 4. assetization_datacenter (`/apis/data`)

### 인덱서 (`/indexer`)
| Method | Path | 인증 | 설명 |
|--------|------|------|------|
| POST | `/del_indexing` | X-DATA-INDEXING 헤더 | 문서 삭제 |
| POST | `/add_indexing` | X-DATA-INDEXING 헤더 | 문서 추가 |
| POST | `/meta_update` | X-DATA-INDEXING 헤더 | 권한 메타 업데이트 |
| GET | `/request_status` | Bearer 토큰 | 비동기 인덱싱 상태 |

### 분석 (`/analytics`)
| Method | Path | 설명 |
|--------|------|------|
| GET | `/ranking_top_search` | 월별 인기 검색어 |
| POST | `/similar_query` | 유사 쿼리 검색 |
| POST | `/es_search_query` | ES 직접 쿼리 |

### 피드백 (`/feedback`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/add_feedback` | 피드백 저장 (G/B) |
| POST | `/del_feedback` | 피드백 삭제 |

### 히스토리 (`/history`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/recent_history_new` | 최근 히스토리 (Redshift) |
| POST | `/history_detail_new` | 히스토리 상세 |
| POST | `/del_history` | 히스토리 삭제 |

### 태그 (`/tag`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/search_chat_tag` | 채팅별 태그 조회 |
| POST | `/add_chat_tag` | 태그 추가 |
| POST | `/search_tag` | 사용자 태그 검색 |
| POST | `/search_tag_chat` | 태그별 채팅 조회 |
| POST | `/del_tag` | 태그 삭제 |

### 컬렉션 (`/collection`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/search_collection` | 컬렉션 목록 |
| POST | `/add_collection` | 컬렉션 생성 |
| POST | `/add_chat_collection` | 채팅-컬렉션 연결 |
| POST | `/search_collection_list` | 전체 컬렉션 (카운트 포함) |
| POST | `/search_collection_chat` | 컬렉션 내 채팅 |
| POST | `/update_collection` | 컬렉션 이름 변경 |
| POST | `/update_collection_chat` | 채팅 이동 |
| POST | `/del_collection` | 컬렉션 삭제 |
| POST | `/del_collection_chat` | 채팅-컬렉션 해제 |
| POST | `/add_chat_seq_info` | 채팅 메타 저장 |

### 배치 (`/batch`)
| Method | Path | 설명 |
|--------|------|------|
| POST | `/run_usage_batch` | 수동 배치 실행 |

---

## 5. assetization_api (`/apis/legacy`)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/weekly/summarize3` | 단일 보고서 요약 (SSE) |
| GET | `/weekly/get-multiple-summaries` | 복수 보고서 요약 (SSE) |
| GET | `/weekly/get-multiple-summaries-by-user` | 사용자별 요약 (SSE) |
| GET | `/weekly/get-single-report` | 단일 보고서 데이터 |
| GET | `/weekly/get-draft-report` | 보고서 초안 생성 (SSE) |
| POST | `/pa2/fetch-data` | PA2.0 데이터 |
| POST | `/pa2/search` | PA2.0 검색 |
| POST | `/prm/fetch-data` | PRM 리스크 평가 |
| POST | `/assetization/search` | 통합 검색 (인증 필수) |

---

## 6. keylook-officeplus (`/keylook-plus`)

### 검색
| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 (모델 정보 포함) |
| POST | `/search` | 하이브리드 검색 (BM25+Sparse+Dense) |
| POST | `/rerank` | 독립 리랭킹 |
| POST | `/embedding` | 임베딩 생성 |

### 문서 관리
| Method | Path | 설명 |
|--------|------|------|
| POST | `/documents` | ID로 문서 조회 |
| GET | `/documents/{project}/{doc_id}` | 문서+청크 조회 |
| POST | `/document/create` | 단건/배치 인덱싱 |
| POST | `/document/batch_create` | S3 배치 인덱싱 |
| DELETE | `/document/delete` | 단건 삭제 |
| DELETE | `/document/batch_delete` | 배치 삭제 |
| PUT | `/document/update` | 메타 업데이트 |
| PUT | `/document/batch_update` | 배치 메타 업데이트 |

### 프로젝트 관리
| Method | Path | 설명 |
|--------|------|------|
| GET | `/projects` | 프로젝트 목록 |
| POST | `/project/create` | 프로젝트 생성 |
| DELETE | `/project/delete` | 프로젝트 삭제 |

### 태스크
| Method | Path | 설명 |
|--------|------|------|
| GET | `/task/status/{task_id}` | 태스크 상태 |
| POST | `/task/cancel/{task_id}` | 태스크 취소 |
