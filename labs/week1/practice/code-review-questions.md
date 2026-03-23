# FastAPI 게시판 과제 - 코드 리뷰 & 즉석 질문 / Code Review & Oral Questions / Danh gia code & Cau hoi truc tiep

> **목적 / Purpose / Muc dich**: AI를 사용했더라도 코드를 **이해하고 설명할 수 있는지** 평가
> Even if AI was used, evaluate whether the student can **understand and explain** the code
> Danh gia xem hoc vien co the **hieu va giai thich** code hay khong, ke ca khi dung AI

---

## Part 1: 제출 소스 코드 비교 분석 / Submitted Code Comparison / So sanh code da nop

### cuongld 소스 특징 / cuongld's Code Characteristics / Dac diem code cuongld

| 항목 / Item | 내용 / Detail |
|-------------|--------------|
| 프로젝트 구조 | `config.py`에 `pydantic_settings` 사용, `errors.py` 유틸 모듈 별도 분리 |
| 비밀번호 해싱 | `bcrypt` 직접 사용 (`_bcrypt.hashpw`, `_bcrypt.checkpw`) |
| JWT | `python-jose` + `settings` 객체에서 키/알고리즘 참조 |
| 에러 처리 | `raise_api_error()` 헬퍼 함수로 통일된 `{"code":..., "message":...}` 포맷 |
| 로깅 미들웨어 | `@app.middleware("http")` + `time.time()` 으로 구현 |
| Validation 에러 | `RequestValidationError` 커스텀 핸들러 등록 |
| ORM 관계 | `relationship()` + `back_populates` 양방향 설정 |
| 응답 변환 | `_to_post_response()` 헬퍼 함수 사용 |
| Pydantic | v2 스타일 (`model_config = ConfigDict(from_attributes=True)`) |
| 테이블 생성 | `@app.on_event("startup")` 이벤트에서 수행 |

### nguyen 소스 특징 / nguyen's Code Characteristics / Dac diem code nguyen

| 항목 / Item | 내용 / Detail |
|-------------|--------------|
| 프로젝트 구조 | `config.py` 비어있음, 설정값 하드코딩 |
| 비밀번호 해싱 | `passlib.context.CryptContext` 사용 (과제 힌트 그대로) |
| JWT | `python-jose` + `os.getenv("SECRET_KEY", "changeme")` |
| 에러 처리 | 에러 포맷 통일 안 됨 (단순 `detail="..."` 문자열) |
| 로깅 미들웨어 | 미구현 |
| ORM 관계 | `relationship()` 미사용 → 매번 author를 별도 쿼리 |
| 응답 변환 | 각 라우터에서 수동으로 `PostResponse(...)` 생성 |
| Pydantic | v1 스타일 (`class Config: from_attributes = True`) |
| 테이블 생성 | 모듈 로드 시점에 `Base.metadata.create_all()` (startup 이벤트 없이) |
| 라우터 prefix | 라우터 내부에서 `prefix="/api/auth"` 직접 지정 (main.py에서 prefix 없음) |

### 주요 차이점 요약 / Key Differences / Khac biet chinh

| 관점 / Aspect | cuongld | nguyen |
|---------------|---------|--------|
| 에러 응답 포맷 | 통일 (`{"code":..., "message":...}`) | 불일치 (문자열 detail) |
| 로깅 미들웨어 (Step 5) | 구현 완료 | 미구현 |
| DB→응답 변환 | 헬퍼 함수 1개 | 각 엔드포인트에서 반복 |
| author 조회 (N+1 문제) | `relationship`으로 해결 | 매번 `db.query(User)` 별도 실행 |
| 설정 관리 | `pydantic_settings` 환경변수 관리 | 하드코딩 + `os.getenv` |

---

## Part 2: 핵심 질문 문항 / Core Oral Questions / Cau hoi phong van cot loi

> 면접관 가이드: 코드를 화면에 띄워놓고 "이 줄"을 가리키며 질문하세요
> Interviewer guide: Display the code on screen and point at "this line" while asking
> Huong dan: Hien thi code tren man hinh va chi vao "dong nay" khi hoi

---

### Q1. `Depends()` 의존성 주입 이해 / Understanding Dependency Injection

**코드를 가리키며 / Point at code:**
```python
def create_post(
    payload: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
)
```

**질문 / Question / Cau hoi:**
- "여기서 `Depends(get_db)`가 하는 일이 뭐예요? `db = get_db()`로 직접 호출하면 안 되나요?"
- "What does `Depends(get_db)` do here? Why not just call `db = get_db()` directly?"
- "`Depends(get_db)` lam gi? Tai sao khong goi truc tiep `db = get_db()`?"

**기대 답변 / Expected answer:**
> FastAPI가 요청마다 자동으로 `get_db()`를 호출하고, yield로 세션을 넘긴 뒤 finally에서 close해줌. 직접 호출하면 yield generator가 제대로 동작 안 하고 세션 정리가 안 됨

**Follow-up:**
- "`get_db()` 안의 `yield`를 `return`으로 바꾸면 어떻게 될까요?"
- "What happens if you change `yield` to `return` in `get_db()`?"

---

### Q2. 비밀번호 해싱 이해 / Password Hashing Understanding

**cuongld에게 / For cuongld:**
```python
# cuongld: bcrypt 직접 사용
return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()
```

**nguyen에게 / For nguyen:**
```python
# nguyen: passlib 사용
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
return pwd_context.hash(password)
```

**질문 / Question / Cau hoi:**
- "bcrypt가 뭐예요? 왜 SHA-256이나 MD5 대신 bcrypt를 쓰나요?"
- "What is bcrypt? Why use it instead of SHA-256 or MD5?"
- "Bcrypt la gi? Tai sao dung bcrypt thay vi SHA-256 hay MD5?"

**기대 답변 / Expected answer:**
> bcrypt는 의도적으로 느린 해시 함수. salt가 자동 포함됨. MD5/SHA-256은 빠르기 때문에 brute-force 공격에 취약

**Follow-up:**
- "`.encode()`는 왜 해요? (cuongld) / `deprecated='auto'`는 뭐예요? (nguyen)"
- "DB에 저장된 해시값에서 원래 비밀번호를 복원할 수 있나요?"

---

### Q3. JWT 토큰 흐름 이해 / JWT Token Flow

**코드를 가리키며:**
```python
payload = {"sub": username, "exp": expire}
return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
```

**질문 / Question / Cau hoi:**
- "JWT 토큰에 `sub`랑 `exp`가 뭐예요? 사용자 ID 대신 username을 넣은 이유는?"
- "What are `sub` and `exp` in a JWT? Why use username instead of user ID?"
- "`sub` va `exp` trong JWT la gi? Tai sao dung username thay vi user ID?"

**기대 답변 / Expected answer:**
> `sub` = subject (토큰 주체), `exp` = expiration (만료). JWT는 서버에서 디코딩해서 사용자 식별. username은 unique key라서 사용 가능

**Follow-up:**
- "이 토큰을 누군가 가로채면 어떻게 되나요? 어떻게 방어하죠?"
- "SECRET_KEY가 유출되면 무슨 일이 벌어져요?"
- "세션 방식이랑 JWT 방식의 차이가 뭐예요?"

---

### Q4. ORM relationship vs N+1 문제 / ORM Relationship vs N+1 Problem

**nguyen 코드를 가리키며:**
```python
# nguyen: 게시글마다 author를 별도 쿼리
for post in posts:
    author = db.query(User).filter(User.id == post.author_id).first()
    result.append(PostResponse(..., author_username=author.username))
```

**cuongld 코드를 가리키며:**
```python
# cuongld: relationship으로 해결
author = relationship("User", back_populates="posts")
# 사용: post.author.username
```

**질문 / Question / Cau hoi:**
- "(nguyen) 게시글이 100개면 쿼리가 몇 번 실행될까요? 이걸 뭐라고 부르죠?"
- "(nguyen) If there are 100 posts, how many queries run? What is this problem called?"
- "(cuongld) `relationship()`에서 `back_populates`를 빼면 어떻게 되나요?"

**기대 답변 / Expected answer:**
> nguyen: 1(목록) + 100(author) = 101번 → N+1 문제. `relationship` + `joinedload`로 해결 가능
> cuongld: `back_populates` 없으면 역방향 탐색(user.posts) 불가, 단방향만 동작

**Follow-up:**
- "이걸 해결하는 방법을 아는 대로 말해보세요 (eager loading, joinedload 등)"

---

### Q5. 에러 처리 설계 / Error Handling Design

**cuongld에게:**
```python
raise HTTPException(
    status_code=404,
    detail={"code": "NOT_FOUND", "message": "Post not found"}
)
```

**nguyen에게:**
```python
raise HTTPException(status_code=404, detail="Post not found")
```

**질문 / Question / Cau hoi:**
- "(nguyen) 과제에서 요구한 에러 포맷이랑 지금 코드가 다른데, 왜 이렇게 했어요?"
- "(cuongld) 에러 응답을 `{"code":..., "message":...}`로 통일한 이유가 뭐예요? 프론트엔드에서 어떻게 활용해요?"

**기대 답변 / Expected answer:**
> 프론트엔드에서 에러 코드(NOT_FOUND, DUPLICATE 등)로 분기 처리 가능. 문자열만 보내면 언어가 바뀌거나 메시지가 변경될 때 프론트가 깨짐

---

### Q6. async vs sync 이해 / async vs sync Understanding

**nguyen 코드를 가리키며:**
```python
# nguyen: 모든 라우터가 sync
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
```

**cuongld 코드를 가리키며:**
```python
# cuongld: middleware는 async, 라우터는 sync
@app.middleware("http")
async def request_logger(request, call_next):
```

**질문 / Question / Cau hoi:**
- "`def`랑 `async def`의 차이가 뭐예요? 라우터를 `async def`로 바꾸면 뭐가 달라져요?"
- "What's the difference between `def` and `async def`? What changes if you make the router `async def`?"
- "Su khac nhau giua `def` va `async def`? Thay doi gi neu dung `async def` cho router?"

**기대 답변 / Expected answer:**
> `async def`는 이벤트 루프에서 실행, I/O 대기 중 다른 요청 처리 가능. 단 SQLAlchemy의 기본 세션은 동기식이라 `async def` + 동기 DB 호출은 오히려 블로킹됨. 진짜 async하려면 `asyncpg` 같은 async 드라이버 필요

---

### Q7. CORS 미들웨어 / CORS Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**질문 / Question / Cau hoi:**
- "CORS가 뭐예요? 왜 필요한 거죠?"
- "What is CORS and why is it needed?"
- "CORS la gi? Tai sao can dung?"

**기대 답변 / Expected answer:**
> 브라우저가 다른 도메인으로 API 호출할 때 보안 정책에 의해 차단됨. CORS 설정으로 어떤 도메인에서 접근 가능한지 서버가 허용. `allow_origins=["*"]`는 개발용이고 프로덕션에서는 특정 도메인만 허용해야 함

**Follow-up:**
- "`allow_origins=['*']`를 프로덕션에서 쓰면 어떤 위험이 있나요?"
- "`allow_credentials=True`인데 origin이 `*`면 실제로 어떻게 동작해요?"

---

### Q8. 테이블 생성 시점 / Table Creation Timing

**cuongld:**
```python
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
```

**nguyen:**
```python
# main.py 최상단 (import 직후)
Base.metadata.create_all(bind=engine)
```

**질문 / Question / Cau hoi:**
- "테이블 생성을 startup 이벤트에서 하는 거랑 import 시점에 하는 거랑 차이가 뭐예요?"
- "What's the difference between creating tables in startup event vs at import time?"

**기대 답변 / Expected answer:**
> import 시점: 모듈이 import될 때 즉시 실행, 테스트 시 DB가 원치 않게 생성될 수 있음. startup 이벤트: 앱이 실제로 시작될 때만 실행되어 더 제어 가능

**Follow-up:**
- "프로덕션에서는 `create_all()` 대신 뭘 써야 할까요?" (→ Alembic 마이그레이션)

---

### Q9. 디버깅 상황 시뮬레이션 / Debugging Scenario / Mo phong debug

**시나리오를 주고 해결 요청:**

> "서버를 실행했는데 이런 에러가 났어요. 원인이 뭘까요?"
> ```
> sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: users
> ```

**기대 답변 / Expected answer:**
> 모델을 import하기 전에 `create_all()`이 호출됨. `Base.metadata`에 모델이 등록되지 않은 상태에서 테이블 생성 시도

**시나리오 2:**
> "로그인은 되는데 게시글 작성 시 401이 나와요. 토큰도 제대로 보내는데요."
> ```
> Authorization: Bearer eyJ...
> ```

**기대 답변 / Expected answer:**
> 토큰이 만료됐거나, SECRET_KEY가 토큰 발급 시점과 검증 시점에 다르거나, `tokenUrl` 설정이 잘못됐을 수 있음. `jwt.decode()`에서 exception이 발생하는지 로그 확인

---

### Q10. 요구사항 즉석 추가 테스트 / Live Requirement Addition / Them yeu cau truc tiep

> 30분 시간을 주고 아래 중 하나를 추가 구현하게 하세요:

**Option A - 쉬움:**
> "비밀번호 확인 필드를 추가하고, 회원가입 시 password와 password_confirm이 일치하지 않으면 422를 반환하세요"

```python
# 기대 결과
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)
    password_confirm: str = Field(min_length=6)

    @model_validator(mode="after")
    def check_passwords_match(self):
        if self.password != self.password_confirm:
            raise ValueError("Passwords do not match")
        return self
```

**Option B - 중간:**
> "게시글 목록 조회에 `?search=keyword` 파라미터를 추가해서 제목+내용 검색을 구현하세요"

```python
# 기대 결과
@router.get("/")
def list_posts(skip: int = 0, limit: int = 20, search: str = None, db: Session = Depends(get_db)):
    query = db.query(Post)
    if search:
        query = query.filter(
            (Post.title.contains(search)) | (Post.content.contains(search))
        )
    posts = query.order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
```

**Option C - 어려움:**
> "게시글에 댓글 기능을 추가하세요 (모델 + 스키마 + CRUD 엔드포인트)"

---

## Part 3: 채점 가이드 / Grading Guide / Huong dan cham diem

### 질문 응답 평가 기준 / Oral Answer Rubric

| 등급 / Grade | 기준 / Criteria | 설명 / Description |
|-------------|----------------|-------------------|
| **A** (완벽 이해) | 정확한 답변 + 추가 설명 가능 | 핵심 개념을 이해하고 trade-off까지 설명 |
| **B** (기본 이해) | 대략적으로 맞는 답변 | 핵심은 알지만 깊은 설명은 부족 |
| **C** (표면적 이해) | 부분적으로 맞는 답변 | 단어는 알지만 동작 원리 설명 못함 |
| **D** (이해 부족) | 틀리거나 답변 불가 | 본인 코드를 설명하지 못함 |

### AI 사용 의심 지표 / AI Usage Indicators / Dau hieu dung AI

| 지표 / Indicator | 설명 / Description |
|-----------------|-------------------|
| 코드 품질 vs 설명 능력 괴리 | 코드는 깔끔한데 기본 질문에 답 못함 |
| 사용 라이브러리 미숙지 | `bcrypt` 직접 썼는데 `.encode()`가 뭔지 모름 |
| 에러 핸들링 패턴 불일치 | 일부는 정교하고 일부는 누락 → 복붙 가능성 |
| 설정 패턴 불일치 | `pydantic_settings` 썼는데 왜 썼는지 모름 |
| 응답이 너무 교과서적 | 실제 경험 없이 정의만 외운 느낌 |

### 추천 평가 배분 / Recommended Score Distribution

| 항목 / Item | 배점 / Points |
|------------|-------------|
| 코드 제출 (기능 동작) | 60점 |
| 즉석 질문 응답 (Q1-Q9 중 5개) | 25점 (5점 x 5) |
| 즉석 요구사항 추가 구현 (Q10) | 15점 |
| **합계 / Total** | **100점** |

> **핵심**: 코드만으로 60점. 나머지 40점은 **이해도**에서 나옴.
> AI를 써서 60점을 받을 수는 있지만, 설명 못 하면 최대 60점에서 멈춤.
> **Key**: Code alone is worth 60. The remaining 40 comes from **understanding**.
> AI can get you 60, but without explanation ability, that's your ceiling.

---

## Part 4: 학생별 맞춤 질문 / Personalized Questions / Cau hoi rieng cho tung hoc vien

### cuongld 전용 질문 (고급)

코드 완성도가 높으므로 **설계 의도와 대안**을 물어볼 것:

1. "`errors.py`에 `raise_api_error()` 헬퍼를 만들었는데, 커스텀 Exception 클래스를 만드는 방식과 비교하면 장단점이 뭐예요?"
2. "`@app.on_event('startup')`은 FastAPI에서 deprecated 예정인데, 대안을 알아요?" (→ `lifespan`)
3. "`_to_post_response()` 대신 `response_model`이 자동으로 변환하게 할 수는 없나요?" (→ `from_attributes` + computed field)
4. "`allow_credentials=True`인데 `allow_origins=['*']`면 브라우저에서 실제로 credential이 전송되나요?"
5. "Config에 `AliasChoices`를 쓴 이유는? 환경변수 이름이 여러 개인 상황이 실무에서 어떻게 발생해요?"

### nguyen 전용 질문 (기본)

미구현/미흡 부분이 있으므로 **기본 개념 확인** 위주:

1. "에러 응답을 `detail='Post not found'`로 했는데, 과제에서 요구한 포맷과 다릅니다. 어떻게 수정해야 하죠?"
2. "로깅 미들웨어를 구현 안 했는데, `@app.middleware('http')`가 뭔지 알아요? 어떻게 만들 건지 말로 설명해보세요"
3. "`config.py`가 비어있는데, SECRET_KEY를 `os.getenv`로 가져오고 있어요. `.env` 파일은 어디에 놓아야 해요?"
4. "`relationship()`을 안 쓰고 매번 `db.query(User)`로 author를 가져오는데, 이게 성능에 어떤 영향이 있을까요?"
5. "Pydantic v1 스타일의 `class Config`를 쓰고 있는데, v2에서는 어떻게 바뀌었는지 알아요?"
