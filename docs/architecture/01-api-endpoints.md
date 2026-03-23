# 전체 API 엔드포인트 맵 / API Endpoint Map / Bản đồ API Endpoint

## 1. assetization_auth (인증 / Auth / Xác thực)

### System Routes (prefix 없음 / no prefix / không có prefix)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/health` | 헬스체크 / Health check / Kiểm tra sức khỏe |
| GET | `/okta` | SSO 콜백 / SSO callback / SSO callback |
| GET | `/login` | 개발용 로그인 페이지 / Dev login page / Trang đăng nhập dev |
| GET | `/doLogin` | 개발용 수동 로그인 / Dev manual login / Đăng nhập thủ công dev |
| GET | `/logout` | 로그아웃 / Logout / Đăng xuất |
| GET | `/tokencheck` | PC→모바일 토큰 생성 / PC→Mobile token / Tạo token PC→Di động |
| GET | `/aisearch` | SSO 콜백 대안 / SSO callback alt / SSO callback thay thế |
| GET | `/agent-auth` | 데스크톱 Agent SSO / Desktop Agent SSO / SSO Agent desktop |
| GET | `/agent-auth-complete/{state}` | Agent 세션 콜백 / Agent session callback / Callback phiên Agent |

### Search/Auth Routes (API_PREFIX)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/sessioninfo` | 현재 사용자 정보 / Current user info / Thông tin người dùng hiện tại |
| POST | `/auth/sessioncheck` | 세션 유효성 확인 / Session validation / Xác minh phiên |
| GET | `/tokencheck` | 모바일 세션 토큰 생성 / Mobile session token / Tạo token phiên di động |
| POST | `/search/getErpAssetAttachInfo` | ERP 첨부 presigned URL / ERP attachment URL / URL đính kèm ERP |

### Mobile Routes (`/mobile`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/mobile/sessioncheck` | EMS 토큰 인증 / EMS token auth / Xác thực token EMS |
| GET | `/mobile/tokencheck` | AES 세션 토큰 검증 / AES session token verify / Xác minh token phiên AES |
| GET | `/mobile/getErpAssetAttachInfo` | 모바일 ERP 첨부 / Mobile ERP attachment / Đính kèm ERP di động |

---

## 2. assetization_orchestrator (`/apis/orche`)

### 채팅/검색 / Chat/Search / Trò chuyện/Tìm kiếm
| Method | Path | 응답 / Resp | 설명 / Description / Mô tả |
|--------|------|------|------|
| GET | `/health` | JSON | 헬스체크 / Health check / Kiểm tra sức khỏe |
| POST | `/chat_context` | JSON | 새 context_id 생성 / New context_id / Tạo context_id mới (UUID) |
| POST | `/chat_fetch` | JSON | 쿼리 Redis 캐시 / Query Redis cache / Cache truy vấn Redis (step 1) |
| POST | `/chat_summarize_fetch` | JSON | 요약 요청 캐시 / Summary request cache / Cache yêu cầu tóm tắt (step 1) |
| POST | `/chat_gpt_fetch` | JSON | GPT 요청 캐시 / GPT request cache / Cache yêu cầu GPT (step 1) |
| GET | `/chat_summarize` | SSE | **V2 LangGraph ReAct** (직접 / direct / trực tiếp) |
| GET | `/chat_summarize_es` | SSE | 캐시된 쿼리 + V2 실행 / Cached query + V2 exec / Truy vấn cache + V2 (step 2) |
| GET | `/chat_gpt` | SSE | 검색 없는 GPT / GPT without search / GPT không tìm kiếm |
| GET | `/chat_gpt_es` | SSE | 캐시된 GPT 실행 / Cached GPT exec / GPT cache (step 2) |
| GET | `/chat_es_summarize` | SSE | V1 Legacy 검색+요약 / Search+Summary / Tìm kiếm+Tóm tắt |
| GET | `/chat_prompt` | SSE | 커스텀 프롬프트 / Custom prompt / Prompt tùy chỉnh V2 ReAct |
| POST | `/context_reset` | JSON | 컨텍스트 리셋 / Context reset / Đặt lại context |

### 파일 업로드 / File Upload / Tải file
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/chat_upload` | 파일 업로드 / File upload / Tải file (텍스트/text/PDF/이미지/image/ảnh) |
| GET | `/chat_files` | 업로드된 파일 목록 / Uploaded file list / Danh sách file đã tải |
| DELETE | `/chat_files/{file_id}` | 파일 삭제 / Delete file / Xóa file |

### 검색 전용 / Search Only / Chỉ tìm kiếm
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/search` | KeyLook 검색 / KeyLook search / Tìm kiếm KeyLook (LLM 없이/without/không có) |
| GET | `/search_url` | 문서 다운로드 URL / Doc download URL / URL tải tài liệu |
| GET | `/related_chunks` | 관련 문서 청크 / Related doc chunks / Đoạn tài liệu liên quan |

### 체크포인터 / Checkpointer / Checkpointer
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/checkpointer/init` | PostgreSQL 체크포인터 초기화 / Init checkpointer / Khởi tạo checkpointer |
| GET | `/checkpointer/status` | 체크포인터 상태 / Checkpointer status / Trạng thái checkpointer |

### ES 로그 / ES Logs / Log ES (`/apis/orche/es_logs`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/search_rerank_log` | 검색 리랭크 로그 / Search rerank log / Log xếp hạng lại tìm kiếm |
| GET | `/llm_prompt_log` | LLM 프롬프트 로그 / LLM prompt log / Log prompt LLM |
| GET | `/search_rerank_log/{txn_id}` | 트랜잭션별 상세 / Detail by txn / Chi tiết theo giao dịch |
| GET | `/llm_prompt_log/{txn_id}` | LLM 로그 상세 / LLM log detail / Chi tiết log LLM |

---

## 3. assetization_mcp (`/apis/mcp`)

| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/health` | 상세 헬스체크 / Detailed health check / Kiểm tra sức khỏe chi tiết |
| POST | `/orchestrator/query` | **메인 쿼리 / Main query / Truy vấn chính** (LangGraph) |
| GET | `/mcptool/sse` | SSE 스트리밍 / SSE streaming / Truyền SSE |
| POST | `/mcptool/messages` | 메시징 / Messaging / Nhắn tin |

---

## 4. assetization_datacenter (`/apis/data`)

### 인덱서 / Indexer / Lập chỉ mục (`/indexer`)
| Method | Path | 인증 / Auth / Xác thực | 설명 / Description / Mô tả |
|--------|------|------|------|
| POST | `/del_indexing` | X-DATA-INDEXING | 문서 삭제 / Delete docs / Xóa tài liệu |
| POST | `/add_indexing` | X-DATA-INDEXING | 문서 추가 / Add docs / Thêm tài liệu |
| POST | `/meta_update` | X-DATA-INDEXING | 권한 메타 업데이트 / Auth meta update / Cập nhật meta quyền |
| GET | `/request_status` | Bearer | 비동기 인덱싱 상태 / Async indexing status / Trạng thái lập chỉ mục bất đồng bộ |

### 분석 / Analytics / Phân tích (`/analytics`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/ranking_top_search` | 월별 인기 검색어 / Monthly top queries / Từ khóa phổ biến theo tháng |
| POST | `/similar_query` | 유사 쿼리 검색 / Similar query search / Tìm truy vấn tương tự |
| POST | `/es_search_query` | ES 직접 쿼리 / Direct ES query / Truy vấn ES trực tiếp |

### 피드백 / Feedback / Phản hồi (`/feedback`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/add_feedback` | 피드백 저장 / Save feedback / Lưu phản hồi (G/B) |
| POST | `/del_feedback` | 피드백 삭제 / Delete feedback / Xóa phản hồi |

### 히스토리 / History / Lịch sử (`/history`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/recent_history_new` | 최근 히스토리 / Recent history / Lịch sử gần đây (Redshift) |
| POST | `/history_detail_new` | 히스토리 상세 / History detail / Chi tiết lịch sử |
| POST | `/del_history` | 히스토리 삭제 / Delete history / Xóa lịch sử |

### 태그 / Tag / Tag (`/tag`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/search_chat_tag` | 채팅별 태그 조회 / Tags by chat / Tag theo chat |
| POST | `/add_chat_tag` | 태그 추가 / Add tag / Thêm tag |
| POST | `/search_tag` | 사용자 태그 검색 / User tag search / Tìm tag người dùng |
| POST | `/search_tag_chat` | 태그별 채팅 조회 / Chats by tag / Chat theo tag |
| POST | `/del_tag` | 태그 삭제 / Delete tag / Xóa tag |

### 컬렉션 / Collection / Bộ sưu tập (`/collection`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/search_collection` | 컬렉션 목록 / Collection list / Danh sách bộ sưu tập |
| POST | `/add_collection` | 컬렉션 생성 / Create collection / Tạo bộ sưu tập |
| POST | `/add_chat_collection` | 채팅-컬렉션 연결 / Link chat-collection / Liên kết chat-bộ sưu tập |
| POST | `/search_collection_list` | 전체 컬렉션 / All collections / Tất cả bộ sưu tập (카운트/count/đếm) |
| POST | `/search_collection_chat` | 컬렉션 내 채팅 / Chats in collection / Chat trong bộ sưu tập |
| POST | `/update_collection` | 컬렉션 이름 변경 / Rename collection / Đổi tên bộ sưu tập |
| POST | `/update_collection_chat` | 채팅 이동 / Move chat / Di chuyển chat |
| POST | `/del_collection` | 컬렉션 삭제 / Delete collection / Xóa bộ sưu tập |
| POST | `/del_collection_chat` | 채팅-컬렉션 해제 / Unlink chat / Hủy liên kết chat |
| POST | `/add_chat_seq_info` | 채팅 메타 저장 / Save chat meta / Lưu meta chat |

### 배치 / Batch / Batch (`/batch`)
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/run_usage_batch` | 수동 배치 실행 / Manual batch run / Chạy batch thủ công |

---

## 5. assetization_api (`/apis/legacy`)

| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/weekly/summarize3` | 단일 보고서 요약 / Single report summary / Tóm tắt một báo cáo (SSE) |
| GET | `/weekly/get-multiple-summaries` | 복수 보고서 요약 / Multiple report summaries / Tóm tắt nhiều báo cáo (SSE) |
| GET | `/weekly/get-multiple-summaries-by-user` | 사용자별 요약 / Summaries by user / Tóm tắt theo người dùng (SSE) |
| GET | `/weekly/get-single-report` | 단일 보고서 데이터 / Single report data / Dữ liệu một báo cáo |
| GET | `/weekly/get-draft-report` | 보고서 초안 생성 / Draft report generation / Tạo bản nháp báo cáo (SSE) |
| POST | `/pa2/fetch-data` | PA2.0 데이터 / PA2.0 data / Dữ liệu PA2.0 |
| POST | `/pa2/search` | PA2.0 검색 / PA2.0 search / Tìm kiếm PA2.0 |
| POST | `/prm/fetch-data` | PRM 리스크 평가 / PRM risk assessment / Đánh giá rủi ro PRM |
| POST | `/assetization/search` | 통합 검색 / Unified search / Tìm kiếm hợp nhất (인증 필수/auth required/cần xác thực) |

---

## 6. keylook-officeplus (`/keylook-plus`)

### 검색 / Search / Tìm kiếm
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/health` | 헬스체크 / Health check / Kiểm tra sức khỏe (모델 정보/model info/thông tin mô hình) |
| POST | `/search` | 하이브리드 검색 / Hybrid search / Tìm kiếm lai (BM25+Sparse+Dense) |
| POST | `/rerank` | 독립 리랭킹 / Standalone reranking / Xếp hạng lại độc lập |
| POST | `/embedding` | 임베딩 생성 / Embedding generation / Tạo embedding |

### 문서 관리 / Document Management / Quản lý tài liệu
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| POST | `/documents` | ID로 문서 조회 / Get docs by ID / Lấy tài liệu theo ID |
| GET | `/documents/{project}/{doc_id}` | 문서+청크 조회 / Doc+chunks / Tài liệu+đoạn |
| POST | `/document/create` | 단건/배치 인덱싱 / Single/batch indexing / Lập chỉ mục đơn/hàng loạt |
| POST | `/document/batch_create` | S3 배치 인덱싱 / S3 batch indexing / Lập chỉ mục hàng loạt S3 |
| DELETE | `/document/delete` | 단건 삭제 / Single delete / Xóa đơn |
| DELETE | `/document/batch_delete` | 배치 삭제 / Batch delete / Xóa hàng loạt |
| PUT | `/document/update` | 메타 업데이트 / Meta update / Cập nhật meta |
| PUT | `/document/batch_update` | 배치 메타 업데이트 / Batch meta update / Cập nhật meta hàng loạt |

### 프로젝트 관리 / Project Management / Quản lý dự án
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/projects` | 프로젝트 목록 / Project list / Danh sách dự án |
| POST | `/project/create` | 프로젝트 생성 / Create project / Tạo dự án |
| DELETE | `/project/delete` | 프로젝트 삭제 / Delete project / Xóa dự án |

### 태스크 / Task / Tác vụ
| Method | Path | 설명 / Description / Mô tả |
|--------|------|------|
| GET | `/task/status/{task_id}` | 태스크 상태 / Task status / Trạng thái tác vụ |
| POST | `/task/cancel/{task_id}` | 태스크 취소 / Cancel task / Hủy tác vụ |
