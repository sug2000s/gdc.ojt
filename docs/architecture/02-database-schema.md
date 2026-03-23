# 데이터베이스 스키마 / Database Schema / Lược đồ cơ sở dữ liệu

## 1. PostgreSQL (assetization DB)

### chat_files (orchestrator)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| file_id | String(36) PK | UUID |
| thread_id | String FK | context_id (인덱스 / indexed / có chỉ mục) |
| file_name | String(500) | 파일명 / File name / Tên file |
| mime_type | String(200) | MIME 타입 / MIME type / Kiểu MIME |
| size | Integer | 파일 크기 / File size / Kích thước file |
| type | String(20) | text / image / image_pages |
| content | Text | 추출된 텍스트 / Extracted text / Văn bản trích xuất (최대/max/tối đa 5000자/chars/ký tự) |
| base64_data | Text | 이미지 / Image / Ảnh base64 |
| pages_json | JSONB | 스캔 PDF 페이지 목록 / Scanned PDF pages / Danh sách trang PDF quét |
| emp_no | String(50) | 업로드 사번 / Uploader emp no / Mã nhân viên tải lên |
| uploaded_at | DateTime | 업로드 시간 / Upload time / Thời gian tải |

### chat_feedback (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 / Chat identifier / Mã định danh chat |
| user_name | String | 사용자명 / User name / Tên người dùng |
| status | String | G(Good/좋음/Tốt) / B(Bad/나쁨/Xấu) |
| contents | Text | 피드백 내용 / Feedback content / Nội dung phản hồi |
| created/updated | DateTime | 타임스탬프 / Timestamp / Dấu thời gian |

### tag_master (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| tag_id | Integer PK | 자동 증가 / Auto increment / Tự tăng |
| tag_name | String(50) | 태그명 / Tag name / Tên tag |
| created/updated | DateTime | 타임스탬프 / Timestamp / Dấu thời gian |

### chat_tag_info (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 / Chat ID / Mã chat |
| tag_id | Integer PK | 태그 ID / Tag ID / ID tag |
| user_id | String | 사번 / Emp no / Mã nhân viên |
| context_id | String | 컨텍스트 ID / Context ID / ID ngữ cảnh |
| created/updated | DateTime | 타임스탬프 / Timestamp / Dấu thời gian |

### user_collection (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| user_id | String PK | 사번 / Emp no / Mã nhân viên |
| collection_seq | Integer PK | 컬렉션 번호 / Collection number / Số bộ sưu tập |
| collection_name | String | 컬렉션명 / Collection name / Tên bộ sưu tập |
| created/updated | DateTime | 타임스탬프 / Timestamp / Dấu thời gian |

### chat_collection_info (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 / Chat ID / Mã chat |
| collection_seq | Integer PK | 컬렉션 번호 / Collection number / Số bộ sưu tập |
| user_id | String | 사번 / Emp no / Mã nhân viên |
| context_id | String | 컨텍스트 ID / Context ID / ID ngữ cảnh |
| query | String | 검색 쿼리 / Search query / Truy vấn tìm kiếm |
| display_no | Integer | 정렬 순서 / Sort order / Thứ tự sắp xếp |

### chat_seq_info (datacenter)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| chat_seq | String PK | 채팅 식별자 / Chat ID / Mã chat |
| context_id | String | 컨텍스트 ID / Context ID / ID ngữ cảnh |
| query | String | 검색 쿼리 / Search query / Truy vấn tìm kiếm |
| target_system | String | 대상 시스템 / Target system / Hệ thống đích |
| chat_response | Text | 응답 내용 / Response content / Nội dung phản hồi |
| ref_documents | Text | 참조 문서 / Referenced docs / Tài liệu tham chiếu |
| end_time | String | 종료 시간 / End time / Thời gian kết thúc |

### chat_context_del / chat_seq_del (datacenter)
- 삭제된 컨텍스트/채팅 추적 / Deleted context/chat tracking / Theo dõi context/chat đã xóa

### keylook_project (datacenter/orchestrator)
| 컬럼 / Column / Cột | 타입 / Type / Kiểu | 설명 / Description / Mô tả |
|------|------|------|
| target_system | String PK | 시스템 식별자 / System ID / Mã hệ thống |
| delete_flag | String | N(활성/Active/Hoạt động) / Y(삭제/Deleted/Đã xóa) |

### batch_log / index_log / statistics_search_usage (datacenter)
- 배치 로그, 인덱싱 로그, 검색 통계 / Batch log, Indexing log, Search stats / Log batch, Log lập chỉ mục, Thống kê tìm kiếm

### LangGraph Checkpoint Tables (orchestrator)
- langgraph-checkpoint-postgres 자동 생성 / Auto-created / Tự động tạo
- thread_id 기반 상태 저장 / State storage by thread_id / Lưu trạng thái theo thread_id

---

## 2. MSSQL - CommonDB

### LCMAT_EMP (직원 마스터 / Employee Master / Quản lý nhân viên)
| 컬럼 / Column / Cột | 설명 / Description / Mô tả |
|------|------|
| EMP_NO | 사번 / Employee number / Mã nhân viên (PK) |
| EMP_NM / EMP_NM_GLOBAL / EMP_NM_LOCAL | 이름 / Name / Tên |
| DEPT_CD / DEPT_NM | 부서 코드/명 / Dept code/name / Mã/tên phòng ban |
| CORP_CD / CORP_NM | 법인 코드/명 / Corp code/name / Mã/tên pháp nhân |
| TITLE_NM | 직위명 / Title / Chức danh |
| EMAIL_ID | 이메일 / Email / Email |
| JC_NM | 직무 분류 / Job classification / Phân loại công việc |
| LANGUAGE | 언어 / Language / Ngôn ngữ |

### LCACT_USER_USRGRP (사용자 그룹 매핑 / User Group Mapping / Ánh xạ nhóm người dùng)
- 관리자 권한 확인용 / Admin permission check / Kiểm tra quyền admin (USRGRP_ID)

### LCMAT_CORP (법인 마스터 / Corp Master / Quản lý pháp nhân)
### LCMAT_DEPT (부서 마스터 / Dept Master / Quản lý phòng ban - L1~L4 계층/hierarchy/phân cấp)
### LCMAT_HOLIDAY (휴일 달력 / Holiday Calendar / Lịch ngày nghỉ)

---

## 3. MSSQL - RMDB (리스크 관리 / Risk Management / Quản lý rủi ro)

### rm_user.RMEMT_PM_ESTIMATION
- PM 리스크 평가 레코드 / PM risk assessment records / Bản ghi đánh giá rủi ro PM

### rm_user.RMEMT_SA_ESTIMATION
- SA 리스크 평가 레코드 / SA risk assessment records / Bản ghi đánh giá rủi ro SA

### rm_user.RMBIT_PM_PROJECT
- PM 프로젝트 매핑 / PM project mapping / Ánh xạ dự án PM

---

## 4. MSSQL - WINDB (주간보고 / Weekly Report / Báo cáo tuần)

### ep_user.WINWRT_REPORT
| 컬럼 / Column / Cột | 설명 / Description / Mô tả |
|------|------|
| REPORT_ID | 보고서 ID / Report ID / ID báo cáo (PK) |
| REPORT_TITLE | 제목 / Title / Tiêu đề |
| THIS_WEEK_CONTENT | 금주 내용 / This week content / Nội dung tuần này (HTML) |
| REPORT_YEAR/MONTH/WEEK | 보고 기간 / Report period / Kỳ báo cáo |
| REG_EMP_NO | 작성자 사번 / Author emp no / Mã nhân viên tác giả |

### ep_user.WINWRT_REPORT_USER
- 보고서-사용자 매핑 / Report-user mapping / Ánh xạ báo cáo-người dùng

---

## 5. Amazon Redshift

### searchlogging
| 컬럼 / Column / Cột (kafka_value.*) | 설명 / Description / Mô tả |
|---------------------|------|
| context_id | 대화 컨텍스트 / Chat context / Ngữ cảnh hội thoại |
| chat_seq | 채팅 순번 / Chat sequence / Số thứ tự chat |
| is_context_start | 컨텍스트 시작 플래그 / Context start flag / Cờ bắt đầu ngữ cảnh |
| query | 검색 쿼리 / Search query / Truy vấn tìm kiếm |
| ref_documents | 참조 문서 / Referenced docs / Tài liệu tham chiếu |
| chat_response | 시스템 응답 / System response / Phản hồi hệ thống |
| target_system | 대상 시스템 / Target system / Hệ thống đích |
| user_id | 사번 / Emp no / Mã nhân viên |
| start_time | 시작 시간 / Start time / Thời gian bắt đầu |

---

## 6. Redis

### 세션 저장 / Session Storage / Lưu trữ phiên
| 키 / Key / Khóa | TTL | 값 / Value / Giá trị |
|----|-----|-----|
| `{ASSET_SID}` | 86400s (1일/day/ngày) | UserModel JSON (camelCase) |
| `{context_id}` | 1800s (30분/min/phút) | emp_no |
| `chat_query:{context_id}` | 30s | 쿼리 JSON / Query JSON / JSON truy vấn |
| `{llm_name}` | 60s | 'lock' (LLM 쿼터/quota/hạn mức) |
| `mcp_rag:session:{session_id}` | 설정 가능/configurable/tùy chỉnh | 대화 히스토리 JSON / Chat history JSON / JSON lịch sử chat |

---

## 7. Elasticsearch

### 문서 인덱스 / Document Index / Chỉ mục tài liệu (`{project_name}`)
- `passage_id`, `document_id` - 식별자 / Identifiers / Mã định danh
- `title` - nori 분석기 / analyzer / bộ phân tích
- `context` - nori 분석기 / analyzer / bộ phân tích
- `dense` - dense_vector (1024차원/dimensions/chiều, cosine)
- `sparse.*` - rank_feature (토큰별/per-token/theo token)
- `meta.*` - 동적 메타 필드 / Dynamic meta fields / Trường meta động

### 원본 문서 인덱스 / Original Document Index / Chỉ mục tài liệu gốc (`{project_name}_orgdoc`)
- `document_id`, `title`, `context`, `meta.*`

### 프로젝트 메타 / Project Meta / Meta dự án (`projects_meta`)
- 프로젝트별 설정 / Per-project config / Cấu hình theo dự án, 모델 정보 / model info / thông tin mô hình, 검색 파라미터 / search params / tham số tìm kiếm

### 로깅 인덱스 / Logging Indices / Chỉ mục log
- `search_rerank_log` - 검색/리랭크 로그 / Search/rerank log / Log tìm kiếm/xếp hạng lại
- `llm_prompt_log` - LLM 프롬프트 로그 / LLM prompt log / Log prompt LLM
- `keylook-api-logs-*` - API 요청 로그 / API request log / Log yêu cầu API
- `keylook-batch-logs-*` - 배치 작업 로그 / Batch job log / Log tác vụ batch
- `keylook-model-logs-*` - 모델 성능 로그 / Model performance log / Log hiệu suất mô hình
