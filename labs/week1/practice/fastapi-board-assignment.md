# FastAPI 게시판 과제 / FastAPI Bulletin Board Assignment / Bai tap FastAPI Bang tin

---

## 개요 / Overview / Tong quan

FastAPI + SQLite를 사용하여 간단한 게시판 API 서버를 만드세요.
Build a simple bulletin board API server using FastAPI + SQLite.
Xay dung mot API server bang tin don gian su dung FastAPI + SQLite.

- 제한 시간 / Time limit / Thoi gian: **4~6시간 / 4-6 hours / 4-6 gio**
- 합격 기준 / Pass criteria / Tieu chi dat: **70점 이상 / 70+ points / 70+ diem** (100점 만점 / out of 100 / tren 100)
- 제공 파일 / Provided files / File duoc cung cap: `requirements.txt`, `.env.example`

---

## 환경 설정 / Environment Setup / Cai dat moi truong

```bash
# 1. 프로젝트 폴더 생성 / Create project folder / Tao thu muc du an
mkdir bulletin-board && cd bulletin-board

# 2. 가상환경 생성 / Create virtual environment / Tao moi truong ao
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치 / Install dependencies / Cai dat thu vien
pip install -r requirements.txt

# 4. 환경변수 설정 / Setup environment / Cai dat bien moi truong
cp .env.example .env
# SECRET_KEY를 랜덤 문자열로 변경 / Change SECRET_KEY to a random string / Doi SECRET_KEY thanh chuoi ngau nhien
```

---

## 권장 프로젝트 구조 / Recommended Structure / Cau truc khuyen nghi

```
bulletin-board/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 / App entry / Ung dung FastAPI
│   ├── config.py            # 설정 / Settings / Cau hinh
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py      # DB 연결 / DB connection / Ket noi DB
│   │   └── security.py      # JWT 인증 / JWT auth / Xac thuc JWT
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py          # User 모델 / User model / Model User
│   │   └── post.py          # Post 모델 / Post model / Model Post
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py          # User 스키마 / User schema / Schema User
│   │   └── post.py          # Post 스키마 / Post schema / Schema Post
│   └── api/
│       ├── __init__.py
│       ├── auth_router.py   # 인증 API / Auth API / API xac thuc
│       └── post_router.py   # 게시글 API / Post API / API bai viet
├── requirements.txt
├── .env
└── .env.example
```

> 이 구조는 권장사항입니다. 자유롭게 구성해도 됩니다.
> This structure is a recommendation. Feel free to organize as you wish.
> Day la cau truc khuyen nghi. Ban co the to chuc theo cach cua ban.

---

## Step 1: 프로젝트 초기화 / Project Setup / Khoi tao du an (10점 / 10pts / 10 diem)

### 요구사항 / Requirements / Yeu cau

1. FastAPI 앱을 생성하세요 / Create a FastAPI app / Tao ung dung FastAPI
2. CORS 미들웨어를 추가하세요 (모든 origin 허용) / Add CORS middleware (allow all origins) / Them CORS middleware (cho phep tat ca origin)
3. 아래 엔드포인트를 구현하세요 / Implement this endpoint / Trien khai endpoint sau:

| Method | Path | Response | Status |
|--------|------|----------|--------|
| GET | `/health` | `{"status": "ok"}` | 200 |

4. uvicorn으로 실행하세요 / Run with uvicorn / Chay bang uvicorn:
```bash
uvicorn app.main:app --reload --port 8000
```

### 검증 / Verification / Kiem tra

```bash
# 이 두 요청이 성공해야 합니다 / Both requests must succeed / Ca hai yeu cau phai thanh cong
curl http://localhost:8000/health
# => {"status": "ok"}

curl http://localhost:8000/docs
# => Swagger UI 페이지 / Swagger UI page / Trang Swagger UI
```

<details>
<summary>힌트 / Hint / Goi y</summary>

- `from fastapi import FastAPI`
- `from fastapi.middleware.cors import CORSMiddleware`
- `app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)`

</details>

---

## Step 2: DB 모델 & 스키마 / DB Models & Schemas / Model DB & Schema (15점 / 15pts / 15 diem)

### 요구사항 / Requirements / Yeu cau

#### 2-1. SQLite 연결 / SQLite Connection / Ket noi SQLite

- SQLAlchemy로 SQLite에 연결하세요 / Connect to SQLite with SQLAlchemy / Ket noi SQLite bang SQLAlchemy
- DB URL: `sqlite:///./bulletin.db`
- `get_db()` 의존성 함수를 만드세요 / Create `get_db()` dependency / Tao ham dependency `get_db()`
- 앱 시작 시 테이블이 자동 생성되어야 합니다 / Tables must auto-create on startup / Bang phai tu dong tao khi khoi dong

#### 2-2. User 테이블 / User Table / Bang User

| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | Primary Key, Auto Increment |
| username | String(50) | Unique, Not Null, Index |
| email | String(100) | Unique, Not Null |
| hashed_password | String(200) | Not Null |
| created_at | DateTime | Default: now |
| is_active | Boolean | Default: True |

#### 2-3. Post 테이블 / Post Table / Bang Post

| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | Primary Key, Auto Increment |
| title | String(200) | Not Null |
| content | Text | Not Null |
| author_id | Integer | Foreign Key → users.id, Not Null |
| created_at | DateTime | Default: now |
| updated_at | DateTime | Nullable, onupdate: now |

#### 2-4. Pydantic 스키마 / Pydantic Schemas / Schema Pydantic

**User 스키마:**

| Schema | Fields |
|--------|--------|
| UserCreate | username (3~50자), email (이메일 형식), password (6자 이상) |
| UserResponse | id, username, email, created_at, is_active |

**Post 스키마:**

| Schema | Fields |
|--------|--------|
| PostCreate | title (1~200자), content (1자 이상) |
| PostUpdate | title (optional), content (optional) |
| PostResponse | id, title, content, author_id, author_username, created_at, updated_at |

### 검증 / Verification / Kiem tra

```bash
# 앱 실행 후 bulletin.db 파일이 생성되어야 합니다
# After running the app, bulletin.db file must be created
# Sau khi chay app, file bulletin.db phai duoc tao

ls bulletin.db
# => bulletin.db

# Swagger에서 스키마 확인 / Check schemas in Swagger / Kiem tra schema tren Swagger
curl http://localhost:8000/openapi.json | python -m json.tool | grep -E "UserCreate|PostCreate"
```

<details>
<summary>힌트 / Hint / Goi y</summary>

- `from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey`
- `from sqlalchemy.ext.declarative import declarative_base`
- `from sqlalchemy.orm import sessionmaker, relationship`
- `from pydantic import BaseModel, Field, EmailStr`
- Pydantic v2에서는 `model_config = ConfigDict(from_attributes=True)` 사용

</details>

---

## Step 3: 회원가입 & 로그인 / Signup & Login / Dang ky & Dang nhap (30점 / 30pts / 30 diem)

### 요구사항 / Requirements / Yeu cau

#### 3-1. 비밀번호 해싱 / Password Hashing / Ma hoa mat khau

- bcrypt로 비밀번호를 해싱하세요 / Hash passwords with bcrypt / Ma hoa mat khau bang bcrypt
- 평문 비밀번호를 절대 DB에 저장하지 마세요 / Never store plain passwords / Khong bao gio luu mat khau dang plain text

#### 3-2. JWT 토큰 / JWT Token / Token JWT

- `python-jose`로 JWT 토큰을 생성/검증하세요 / Create/verify JWT with python-jose / Tao/xac thuc JWT bang python-jose
- 토큰에 `sub` (username) 클레임을 포함하세요 / Include `sub` (username) claim / Bao gom claim `sub` (username)
- 만료 시간을 설정하세요 / Set expiration time / Dat thoi gian het han

#### 3-3. 인증 의존성 / Auth Dependency / Dependency xac thuc

- `get_current_user` 함수를 만드세요 / Create `get_current_user` function / Tao ham `get_current_user`
- `Authorization: Bearer <token>` 헤더에서 토큰을 추출하세요 / Extract token from Bearer header / Lay token tu header Bearer
- 이 함수는 다음 Step에서 게시글 API 보호에 사용됩니다 / This will protect post APIs in next step / Ham nay se bao ve API bai viet o buoc tiep

#### 3-4. API 엔드포인트 / API Endpoints / Endpoint API

| Method | Path | Auth | Request Body | Response | Status |
|--------|------|------|-------------|----------|--------|
| POST | `/api/auth/signup` | No | `{"username":"...", "email":"...", "password":"..."}` | UserResponse | 201 |
| POST | `/api/auth/login` | No | Form: `username`, `password` (OAuth2PasswordRequestForm) | `{"access_token":"...", "token_type":"bearer"}` | 200 |

#### 에러 처리 / Error Handling / Xu ly loi

| 상황 / Case / Truong hop | Status | Message |
|--------------------------|--------|---------|
| 중복 username 또는 email / Duplicate username or email / Trung username hoac email | 409 | Already exists |
| 잘못된 비밀번호 / Wrong password / Sai mat khau | 401 | Invalid credentials |
| 존재하지 않는 사용자 / User not found / Khong tim thay user | 401 | Invalid credentials |

### 검증 / Verification / Kiem tra

```bash
# 1. 회원가입 / Signup / Dang ky
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@test.com","password":"test123"}'
# => 201, {"id":1, "username":"testuser", ...}

# 2. 중복 가입 / Duplicate signup / Dang ky trung
curl -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@test.com","password":"test123"}'
# => 409

# 3. 로그인 / Login / Dang nhap
curl -X POST http://localhost:8000/api/auth/login \
  -d "username=testuser&password=test123"
# => 200, {"access_token":"eyJ...", "token_type":"bearer"}

# 4. 잘못된 비밀번호 / Wrong password / Sai mat khau
curl -X POST http://localhost:8000/api/auth/login \
  -d "username=testuser&password=wrong"
# => 401
```

<details>
<summary>힌트 / Hint / Goi y</summary>

- `from passlib.context import CryptContext` → `pwd_context = CryptContext(schemes=["bcrypt"])`
- `from jose import jwt` → `jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm="HS256")`
- `from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm`
- `oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")`
- `get_current_user`는 `Depends(oauth2_scheme)`로 토큰을 받아 디코딩

</details>

---

## Step 4: 게시글 CRUD / Post CRUD / CRUD Bai viet (35점 / 35pts / 35 diem)

### 요구사항 / Requirements / Yeu cau

| Method | Path | Auth | Description | Status |
|--------|------|------|-------------|--------|
| POST | `/api/posts/` | Yes | 게시글 작성 / Create post / Tao bai viet | 201 |
| GET | `/api/posts/` | No | 목록 조회 / List posts / Danh sach bai viet | 200 |
| GET | `/api/posts/{id}` | No | 상세 조회 / Get post detail / Chi tiet bai viet | 200 |
| PUT | `/api/posts/{id}` | Yes | 수정 (작성자만) / Update (author only) / Sua (chi tac gia) | 200 |
| DELETE | `/api/posts/{id}` | Yes | 삭제 (작성자만) / Delete (author only) / Xoa (chi tac gia) | 204 |

#### 상세 규칙 / Detail Rules / Quy tac chi tiet

**POST /api/posts/** (게시글 작성 / Create / Tao)
- 로그인 필수 / Login required / Can dang nhap
- `author_id`는 현재 로그인한 사용자의 ID / `author_id` is current user's ID / `author_id` la ID cua user hien tai

**GET /api/posts/** (목록 조회 / List / Danh sach)
- 로그인 불필요 / No login required / Khong can dang nhap
- Query parameters: `skip` (default: 0), `limit` (default: 20)
- 최신 글이 먼저 / Newest first / Bai moi nhat truoc

**GET /api/posts/{id}** (상세 조회 / Detail / Chi tiet)
- 로그인 불필요 / No login required / Khong can dang nhap
- 작성자 username 포함 / Include author username / Bao gom username tac gia

**PUT /api/posts/{id}** (수정 / Update / Sua)
- 로그인 필수 / Login required / Can dang nhap
- 작성자 본인만 수정 가능 / Only author can update / Chi tac gia moi duoc sua
- 비작성자 → 403 Forbidden

**DELETE /api/posts/{id}** (삭제 / Delete / Xoa)
- 로그인 필수 / Login required / Can dang nhap
- 작성자 본인만 삭제 가능 / Only author can delete / Chi tac gia moi duoc xoa
- 비작성자 → 403 Forbidden

#### 에러 처리 / Error Handling / Xu ly loi

| 상황 / Case / Truong hop | Status |
|--------------------------|--------|
| 미인증 요청 / Unauthenticated / Chua xac thuc | 401 |
| 비작성자 수정/삭제 시도 / Non-author edit/delete / Khong phai tac gia sua/xoa | 403 |
| 존재하지 않는 게시글 / Post not found / Khong tim thay bai viet | 404 |

### 검증 / Verification / Kiem tra

```bash
# 먼저 로그인하여 토큰 획득 / First login to get token / Truoc tien dang nhap de lay token
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -d "username=testuser&password=test123" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# 1. 게시글 작성 / Create post / Tao bai viet
curl -X POST http://localhost:8000/api/posts/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"First Post","content":"Hello World"}'
# => 201

# 2. 목록 조회 / List posts / Danh sach
curl http://localhost:8000/api/posts/
# => 200, [{"id":1, "title":"First Post", ...}]

# 3. 상세 조회 / Get detail / Chi tiet
curl http://localhost:8000/api/posts/1
# => 200, {"id":1, "title":"First Post", "author_username":"testuser", ...}

# 4. 수정 / Update / Sua
curl -X PUT http://localhost:8000/api/posts/1 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated Post"}'
# => 200

# 5. 다른 사용자로 삭제 시도 / Try delete as another user / Thu xoa bang user khac
# (다른 사용자로 가입+로그인 후 시도 / Signup+login as another user first / Dang ky+dang nhap user khac truoc)
# => 403

# 6. 삭제 / Delete / Xoa
curl -X DELETE http://localhost:8000/api/posts/1 \
  -H "Authorization: Bearer $TOKEN"
# => 204
```

<details>
<summary>힌트 / Hint / Goi y</summary>

- Router에서 `Depends(get_current_user)`로 현재 사용자를 주입
- `db.query(Post).filter(Post.id == post_id).first()` → 없으면 404
- `if post.author_id != current_user.id` → 403
- 목록 조회: `db.query(Post).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()`
- 삭제 후 `Response(status_code=204)` 반환 (body 없음)

</details>

---

## Step 5: 에러 처리 & 응답 표준화 / Error Handling & Response Standardization / Xu ly loi & Chuan hoa phan hoi (10점 / 10pts / 10 diem)

### 요구사항 / Requirements / Yeu cau

#### 5-1. 통일된 에러 응답 포맷 / Unified Error Response / Dinh dang loi thong nhat

모든 에러 응답이 다음 포맷을 따르도록 하세요:
All error responses should follow this format:
Tat ca phan hoi loi phai theo dinh dang nay:

```json
{
  "detail": {
    "code": "NOT_FOUND",
    "message": "Post not found"
  }
}
```

에러 코드 목록 / Error codes / Danh sach ma loi:

| Code | 의미 / Meaning / Y nghia | HTTP Status |
|------|--------------------------|-------------|
| `DUPLICATE` | 중복 / Duplicate / Trung lap | 409 |
| `UNAUTHORIZED` | 인증 실패 / Auth failed / Xac thuc that bai | 401 |
| `FORBIDDEN` | 권한 없음 / No permission / Khong co quyen | 403 |
| `NOT_FOUND` | 없음 / Not found / Khong tim thay | 404 |
| `VALIDATION_ERROR` | 입력 오류 / Input error / Loi nhap lieu | 422 |

#### 5-2. 요청 로깅 미들웨어 / Request Logging Middleware / Middleware ghi log

모든 요청에 대해 다음을 로깅하세요:
Log the following for every request:
Ghi log cac thong tin sau cho moi yeu cau:

```
2024-03-23 09:00:00 | POST | /api/auth/signup | 201 | 45ms
```

포맷: `{timestamp} | {method} | {path} | {status_code} | {duration}`

### 검증 / Verification / Kiem tra

```bash
# 없는 게시글 조회 / Get non-existent post / Lay bai viet khong ton tai
curl http://localhost:8000/api/posts/999
# => 404, {"detail": {"code": "NOT_FOUND", "message": "Post not found"}}

# 서버 터미널에서 로그 확인 / Check logs in server terminal / Kiem tra log tren terminal server
# => 2024-03-23 09:00:00 | GET | /api/posts/999 | 404 | 12ms
```

<details>
<summary>힌트 / Hint / Goi y</summary>

- `from fastapi import HTTPException` → `raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "..."})`
- 미들웨어: `@app.middleware("http")` 또는 `BaseHTTPMiddleware` 상속
- `import time` → `start = time.time()` ... `duration = (time.time() - start) * 1000`

</details>

---

## 채점 기준 / Grading Criteria / Tieu chi cham diem

| Step | 항목 / Item / Hang muc | 점수 / Points / Diem |
|------|------------------------|----------------------|
| 1 | `/health` → 200 | 5 |
| 1 | `/docs` (Swagger) 접근 가능 | 5 |
| 2 | 앱 실행 시 테이블 자동 생성 | 5 |
| 2 | Pydantic 스키마 정의 완료 | 5 |
| 2 | User-Post 관계 설정 (FK) | 5 |
| 3 | `POST /api/auth/signup` 정상 동작 | 5 |
| 3 | 중복 가입 → 409 | 5 |
| 3 | `POST /api/auth/login` → JWT 토큰 반환 | 10 |
| 3 | 잘못된 비밀번호 → 401 | 5 |
| 3 | `get_current_user` 의존성 동작 | 5 |
| 4 | `POST /api/posts/` 게시글 작성 | 5 |
| 4 | `GET /api/posts/` 목록 조회 | 5 |
| 4 | `GET /api/posts/{id}` 상세 조회 | 5 |
| 4 | `PUT /api/posts/{id}` 수정 (작성자만) | 5 |
| 4 | `DELETE /api/posts/{id}` 삭제 (작성자만) | 5 |
| 4 | 미인증 → 401, 비작성자 → 403, 없는 글 → 404 | 10 |
| 5 | 통일된 에러 응답 포맷 | 5 |
| 5 | 요청 로깅 미들웨어 | 5 |
| | **합계 / Total / Tong** | **100** |

### 등급 / Grade / Xep hang

| 점수 / Score / Diem | 등급 / Grade / Xep hang | 평가 / Evaluation / Danh gia |
|---------------------|------------------------|------------------------------|
| 90-100 | A | 우수 / Excellent / Xuat sac |
| 70-89 | B (합격 / Pass / Dat) | 양호 / Good / Tot |
| 50-69 | C | 보통 / Average / Trung binh |
| 0-49 | D | 재도전 필요 / Retry needed / Can thu lai |

---

이름 / Name / Ten: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

날짜 / Date / Ngay: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

점수 / Score / Diem: \_\_\_\_\_\_ / 100
