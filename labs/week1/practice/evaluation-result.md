# FastAPI 게시판 과제 평가 결과 / Evaluation Result

> 평가일 / Date: 2026-03-23

---

## cuongld 평가 / cuongld Evaluation

### 채점표 / Score Sheet

| Step | 항목 / Item | 배점 | 득점 | 비고 / Note |
|------|------------|------|------|-------------|
| 1 | `/health` → 200 | 5 | **5** | `main.py:27` 정상 구현 |
| 1 | `/docs` Swagger 접근 | 5 | **5** | `FastAPI(title="Bulletin Board API")` |
| 2 | 앱 실행 시 테이블 자동 생성 | 5 | **5** | `@app.on_event("startup")`에서 `create_all()` |
| 2 | Pydantic 스키마 정의 | 5 | **5** | v2 스타일 `ConfigDict(from_attributes=True)` 사용 |
| 2 | User-Post FK 관계 설정 | 5 | **5** | `relationship()` + `back_populates` 양방향 |
| 3 | `POST /api/auth/signup` | 5 | **5** | 정상 동작, `status_code=201` |
| 3 | 중복 가입 → 409 | 5 | **5** | `username \| email` OR 조건 필터 |
| 3 | `POST /api/auth/login` → JWT | 10 | **10** | `python-jose` + `HS256` |
| 3 | 잘못된 비밀번호 → 401 | 5 | **5** | `verify_password` 실패 시 401 |
| 3 | `get_current_user` 의존성 | 5 | **5** | `Depends(oauth2_scheme)` + DB 조회 |
| 4 | `POST /api/posts/` 작성 | 5 | **5** | `current_user.id`를 `author_id`로 설정 |
| 4 | `GET /api/posts/` 목록 | 5 | **5** | `order_by(desc)` + `skip/limit` |
| 4 | `GET /api/posts/{id}` 상세 | 5 | **5** | `author_username` 포함 |
| 4 | `PUT /api/posts/{id}` 수정 | 5 | **5** | `model_dump(exclude_unset=True)` 사용 |
| 4 | `DELETE /api/posts/{id}` 삭제 | 5 | **5** | `Response(status_code=204)` |
| 4 | 에러 코드 401/403/404 | 10 | **10** | 모두 정확히 구현 |
| 5 | 통일된 에러 응답 포맷 | 5 | **5** | `{"code":"...", "message":"..."}` + `raise_api_error()` 헬퍼 |
| 5 | 요청 로깅 미들웨어 | 5 | **5** | `@app.middleware("http")` + `time.time()` |
| | **합계 / Total** | **100** | **100** | |

### 등급: A (우수 / Excellent)

### 강점 / Strengths
- **완성도**: 모든 요구사항 100% 충족
- **설정 관리**: `pydantic_settings`로 환경변수 관리 체계화, `AliasChoices`까지 활용
- **에러 처리**: `errors.py` 모듈 분리, `RequestValidationError` 커스텀 핸들러까지 구현
- **코드 구조**: `_to_post_response()` 헬퍼로 응답 변환 로직 통일
- **ORM 활용**: `relationship` + `cascade="all, delete-orphan"` 적용

### 개선점 / Areas for Improvement
- `@app.on_event("startup")`은 deprecated 예정 → `lifespan` context manager 권장
- `bcrypt` 직접 사용보다 `passlib.CryptContext`가 향후 알고리즘 교체 시 유연
- `_to_post_response()`를 쓸 거면 `response_model` + `from_attributes` 자동 변환도 고려

---

## nguyen 평가 / nguyen Evaluation

### 채점표 / Score Sheet

| Step | 항목 / Item | 배점 | 득점 | 비고 / Note |
|------|------------|------|------|-------------|
| 1 | `/health` → 200 | 5 | **5** | `main.py:27` 정상 구현 |
| 1 | `/docs` Swagger 접근 | 5 | **5** | `FastAPI()` 기본 설정 |
| 2 | 앱 실행 시 테이블 자동 생성 | 5 | **5** | import 시점에 `create_all()` (동작은 함) |
| 2 | Pydantic 스키마 정의 | 5 | **5** | 필드 정의 완료 (v1 스타일 Config) |
| 2 | User-Post FK 관계 설정 | 5 | **3** | FK는 있으나 `relationship()` 미사용 (-2) |
| 3 | `POST /api/auth/signup` | 5 | **5** | 정상 동작, `status_code=201` |
| 3 | 중복 가입 → 409 | 5 | **5** | `username \| email` OR 조건 필터 |
| 3 | `POST /api/auth/login` → JWT | 10 | **10** | `python-jose` + `HS256` |
| 3 | 잘못된 비밀번호 → 401 | 5 | **5** | `verify_password` 실패 시 401 |
| 3 | `get_current_user` 의존성 | 5 | **5** | `Depends(oauth2_scheme)` + DB 조회 |
| 4 | `POST /api/posts/` 작성 | 5 | **5** | `current_user.id`를 `author_id`로 설정 |
| 4 | `GET /api/posts/` 목록 | 5 | **5** | `order_by(desc)` + `skip/limit` |
| 4 | `GET /api/posts/{id}` 상세 | 5 | **5** | author를 별도 쿼리로 조회 (동작은 함) |
| 4 | `PUT /api/posts/{id}` 수정 | 5 | **5** | 작성자 확인 후 수정 |
| 4 | `DELETE /api/posts/{id}` 삭제 | 5 | **5** | `Response(status_code=204)` |
| 4 | 에러 코드 401/403/404 | 10 | **10** | 상태 코드는 모두 정확 |
| 5 | 통일된 에러 응답 포맷 | 5 | **0** | `detail="문자열"` 사용, `{"code":..., "message":...}` 포맷 미적용 |
| 5 | 요청 로깅 미들웨어 | 5 | **0** | **미구현** |
| | **합계 / Total** | **100** | **88** | |

### 등급: B (합격 / Pass)

### 강점 / Strengths
- **핵심 기능 완성**: 인증 + CRUD 전체 동작
- **과제 힌트 활용**: `passlib.CryptContext` 등 힌트 코드를 적절히 적용
- **에러 상태 코드**: 401/403/404 분기 처리 정확

### 개선점 / Areas for Improvement

| 우선순위 | 항목 | 설명 | 감점 |
|---------|------|------|------|
| **높음** | 에러 응답 포맷 | `detail="문자열"` → `detail={"code":"NOT_FOUND", "message":"..."}` 변경 필요 | -5 |
| **높음** | 로깅 미들웨어 | Step 5 요구사항 미구현. `@app.middleware("http")` 추가 필요 | -5 |
| **중간** | ORM relationship | `relationship()` 미사용으로 N+1 문제 발생 (게시글 100개 → 쿼리 101번) | -2 |
| 낮음 | 설정 관리 | `config.py` 비어있음, SECRET_KEY 등 하드코딩. `pydantic_settings` 권장 |  |
| 낮음 | Pydantic 버전 | `class Config` (v1) → `model_config = ConfigDict(...)` (v2) 권장 |  |
| 낮음 | 테이블명 | `"user"` (단수) 사용 → 관례상 `"users"` (복수) 권장 |  |

---

## 종합 비교 / Overall Comparison

| 항목 / Criteria | cuongld | nguyen |
|----------------|---------|--------|
| **점수 / Score** | **100 / 100** | **88 / 100** |
| **등급 / Grade** | **A** | **B** |
| Step 1 (초기화) | 10/10 | 10/10 |
| Step 2 (모델/스키마) | 15/15 | 13/15 |
| Step 3 (인증) | 30/30 | 30/30 |
| Step 4 (CRUD) | 35/35 | 35/35 |
| Step 5 (에러/로깅) | 10/10 | 0/10 |
| 코드 품질 | 높음 (모듈화, 헬퍼, 관계 설정) | 보통 (기능 동작은 하나 구조 개선 필요) |

### 총평 / Summary

**cuongld**: 과제 요구사항을 100% 충족하며 코드 구조화 수준이 높음. `pydantic_settings`, 에러 헬퍼 함수, ORM relationship 등 **실무 수준의 패턴** 적용. 다만 이 수준의 완성도는 AI 도구 활용 가능성도 있으므로, **즉석 질문(code-review-questions.md Q1~Q5 cuongld 전용)** 으로 실제 이해도 확인 필요.

**nguyen**: 핵심 기능(인증 + CRUD)은 모두 동작하여 합격 기준(70점) 충족. Step 5(에러 포맷 통일, 로깅 미들웨어)가 미구현된 점이 아쉬움. `relationship()` 미사용으로 N+1 문제가 있으나 기능적으로는 정상 동작. **즉석 질문(nguyen 전용 Q1~Q5)** 으로 기본 개념 이해 확인 후, Step 5 보완 과제 부여 권장.
