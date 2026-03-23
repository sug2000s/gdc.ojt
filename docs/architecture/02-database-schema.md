# 데이터베이스 스키마 정리

## 1. PostgreSQL (assetization DB)

### chat_files (orchestrator)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| file_id | String(36) PK | UUID |
| thread_id | String FK | context_id (인덱스) |
| file_name | String(500) | 파일명 |
| mime_type | String(200) | MIME 타입 |
| size | Integer | 파일 크기 |
| type | String(20) | text / image / image_pages |
| content | Text | 추출된 텍스트 (최대 5000자) |
| base64_data | Text | 이미지 base64 |
| pages_json | JSONB | 스캔 PDF 페이지 목록 |
| emp_no | String(50) | 업로드 사번 |
| uploaded_at | DateTime | 업로드 시간 |

### chat_feedback (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 |
| user_name | String | 사용자명 |
| status | String | G(Good) / B(Bad) |
| contents | Text | 피드백 내용 |
| created/updated | DateTime | 타임스탬프 |

### tag_master (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| tag_id | Integer PK | 자동 증가 |
| tag_name | String(50) | 태그명 |
| created/updated | DateTime | 타임스탬프 |

### chat_tag_info (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 |
| tag_id | Integer PK | 태그 ID |
| user_id | String | 사번 |
| context_id | String | 컨텍스트 ID |
| created/updated | DateTime | 타임스탬프 |

### user_collection (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | String PK | 사번 |
| collection_seq | Integer PK | 컬렉션 번호 |
| collection_name | String | 컬렉션명 |
| created/updated | DateTime | 타임스탬프 |

### chat_collection_info (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 |
| collection_seq | Integer PK | 컬렉션 번호 |
| user_id | String | 사번 |
| context_id | String | 컨텍스트 ID |
| query | String | 검색 쿼리 |
| display_no | Integer | 정렬 순서 |

### chat_seq_info (datacenter)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 |
| context_id | String | 컨텍스트 ID |
| query | String | 검색 쿼리 |
| target_system | String | 대상 시스템 |
| chat_response | Text | 응답 내용 |
| ref_documents | Text | 참조 문서 |
| end_time | String | 종료 시간 |

### chat_context_del / chat_seq_del (datacenter)
- 삭제된 컨텍스트/채팅 추적 테이블

### keylook_project (datacenter/orchestrator)
| 컬럼 | 타입 | 설명 |
|------|------|------|
| target_system | String PK | 시스템 식별자 |
| delete_flag | String | N(활성) / Y(삭제) |

### batch_log / index_log / statistics_search_usage (datacenter)
- 배치 실행 로그, 인덱싱 로그, 검색 통계

### LangGraph Checkpoint Tables (orchestrator)
- langgraph-checkpoint-postgres 자동 생성
- thread_id 기반 상태 저장

---

## 2. MSSQL - CommonDB

### LCMAT_EMP (직원 마스터)
| 컬럼 | 설명 |
|------|------|
| EMP_NO | 사번 (PK) |
| EMP_NM / EMP_NM_GLOBAL / EMP_NM_LOCAL | 이름 |
| DEPT_CD / DEPT_NM | 부서 코드/명 |
| CORP_CD / CORP_NM | 법인 코드/명 |
| TITLE_NM | 직위명 |
| EMAIL_ID | 이메일 |
| JC_NM | 직무 분류 |
| LANGUAGE | 언어 |

### LCACT_USER_USRGRP (사용자 그룹 매핑)
- 관리자 권한 확인용 (USRGRP_ID 기반)

### LCMAT_CORP (법인 마스터)
### LCMAT_DEPT (부서 마스터 - L1~L4 계층)
### LCMAT_HOLIDAY (휴일 달력)

---

## 3. MSSQL - RMDB (리스크 관리)

### rm_user.RMEMT_PM_ESTIMATION
- PM 리스크 평가 레코드

### rm_user.RMEMT_SA_ESTIMATION
- SA 리스크 평가 레코드

### rm_user.RMBIT_PM_PROJECT
- PM 프로젝트 매핑

---

## 4. MSSQL - WINDB (주간보고)

### ep_user.WINWRT_REPORT
| 컬럼 | 설명 |
|------|------|
| REPORT_ID | 보고서 ID (PK) |
| REPORT_TITLE | 제목 |
| THIS_WEEK_CONTENT | 금주 내용 (HTML) |
| REPORT_YEAR/MONTH/WEEK | 보고 기간 |
| REG_EMP_NO | 작성자 사번 |

### ep_user.WINWRT_REPORT_USER
- 보고서-사용자 매핑

---

## 5. Amazon Redshift

### searchlogging
| 컬럼 (kafka_value.*) | 설명 |
|---------------------|------|
| context_id | 대화 컨텍스트 |
| chat_seq | 채팅 순번 |
| is_context_start | 컨텍스트 시작 플래그 |
| query | 검색 쿼리 |
| ref_documents | 참조 문서 |
| chat_response | 시스템 응답 |
| target_system | 대상 시스템 |
| user_id | 사번 |
| start_time | 시작 시간 |

---

## 6. Redis

### 세션 저장
| 키 | TTL | 값 |
|----|-----|-----|
| `{ASSET_SID}` | 86400s (1일) | UserModel JSON (camelCase) |
| `{context_id}` | 1800s (30분) | emp_no |
| `chat_query:{context_id}` | 30s | 쿼리 JSON |
| `{llm_name}` | 60s | 'lock' (LLM 쿼터) |
| `mcp_rag:session:{session_id}` | 설정 가능 | 대화 히스토리 JSON |

---

## 7. Elasticsearch

### 문서 인덱스 (`{project_name}`)
- `passage_id`, `document_id` - 식별자
- `title` - nori 분석기
- `context` - nori 분석기
- `dense` - dense_vector (1024차원, cosine)
- `sparse.*` - rank_feature (토큰별)
- `meta.*` - 동적 메타 필드

### 원본 문서 인덱스 (`{project_name}_orgdoc`)
- `document_id`, `title`, `context`, `meta.*`

### 프로젝트 메타 (`projects_meta`)
- 프로젝트별 설정, 모델 정보, 검색 파라미터

### 로깅 인덱스
- `search_rerank_log` - 검색/리랭크 로그
- `llm_prompt_log` - LLM 프롬프트 로그
- `keylook-api-logs-*` - API 요청 로그
- `keylook-batch-logs-*` - 배치 작업 로그
- `keylook-model-logs-*` - 모델 성능 로그
