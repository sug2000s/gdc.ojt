# FastAPI 게시판 과제 해설 / Assignment Explained / Giai thich bai tap

> 강사용 / For instructors / Danh cho giang vien
> 과제 완료 후 배포하세요 / Distribute after assignment deadline / Phat sau khi het han bai tap

---

## Step 1 해설: 프로젝트 초기화 / Project Setup / Khoi tao du an

### 핵심 코드 / Key Code / Code chinh

**app/main.py**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # 테이블 생성 / Create tables / Tao bang
    yield

app = FastAPI(title="Bulletin Board API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Router 등록은 Step 3, 4에서 추가
# from app.api.auth_router import router as auth_router
# from app.api.post_router import router as post_router
# app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
# app.include_router(post_router, prefix="/api/posts", tags=["posts"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### 실제 서비스 매핑 / Real Service Mapping / Anh xa dich vu thuc

- `assetization_auth/app/main.py`와 동일한 패턴: lifespan으로 초기화, CORS 설정, router include
- `assetization_api/app/main.py`: `app.include_router(router, prefix=..., tags=[...])`

### 흔한 실수 / Common Mistakes / Loi thuong gap

- `lifespan` 대신 deprecated된 `@app.on_event("startup")` 사용
- CORS에서 `allow_methods`나 `allow_headers` 빠뜨림
- `uvicorn app.main:app`에서 경로 틀림

---

## Step 2 해설: DB 모델 & 스키마 / Models & Schemas / Model & Schema

### 핵심 코드 / Key Code / Code chinh

**app/config.py**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY: str = "changeme"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    DATABASE_URL: str = "sqlite:///./bulletin.db"

    class Config:
        env_file = ".env"

settings = Settings()
```

**app/core/database.py**
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite 전용
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app.models import user, post  # 모델 import하여 Base에 등록
    Base.metadata.create_all(bind=engine)
```

**app/models/user.py**
```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime, timezone
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
```

**app/models/post.py**
```python
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.core.database import Base

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=True, onupdate=lambda: datetime.now(timezone.utc))

    author = relationship("User")
```

**app/schemas/user.py**
```python
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from datetime import datetime

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

**app/schemas/post.py**
```python
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional

class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)

class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    author_id: int
    author_username: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
```

### 실제 서비스 매핑 / Real Service Mapping / Anh xa dich vu thuc

- `assetization_auth/app/core/database.py`: 동일한 `get_db()` 패턴 (실제로는 MSSQL + Redis)
- `assetization_auth/app/models/user.py`: UserModel과 유사한 구조
- `assetization_api/app/api/endpoints/pa2_service_router.py`: Pydantic `BaseModel` + `Field()` 패턴

### 흔한 실수 / Common Mistakes / Loi thuong gap

- SQLite에서 `check_same_thread=False` 빠뜨림 → 멀티스레드 에러
- `init_db()`에서 모델을 import하지 않아 테이블이 안 생김
- Pydantic v2에서 `orm_mode = True` 대신 `from_attributes=True` 사용해야 함
- `ForeignKey("users.id")`에서 테이블명 대소문자 틀림

---

## Step 3 해설: 회원가입 & 로그인 / Signup & Login / Dang ky & Dang nhap

### 핵심 코드 / Key Code / Code chinh

**app/core/security.py**
```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.config import settings
from app.core.database import get_db
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
```

**app/api/auth_router.py**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, TokenResponse

router = APIRouter()

@router.post("/signup", response_model=UserResponse, status_code=201)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    # 중복 체크
    existing = db.query(User).filter(
        (User.username == user_in.username) | (User.email == user_in.email)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username or email already exists")

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
```

### 실제 서비스 매핑 / Real Service Mapping / Anh xa dich vu thuc

- `assetization_auth/app/utils/auth.py`: SSO 토큰 검증 → 여기서는 JWT로 단순화
- `assetization_auth/app/middleware/session_middleware.py`: 세션 기반 인증 → JWT로 대체
- `assetization_api/app/core/auth.py`: `verify_api_key()` 패턴 → `get_current_user()` 패턴
- 실제 서비스는 SSO + Redis 세션이지만, 핵심 흐름(인증 → 사용자 식별 → 요청 보호)은 동일

### 흔한 실수 / Common Mistakes / Loi thuong gap

- `OAuth2PasswordRequestForm`은 JSON이 아니라 form-data (`application/x-www-form-urlencoded`)
- `tokenUrl`이 실제 로그인 경로와 다르면 Swagger에서 인증 안 됨
- `jwt.decode`에서 `algorithms`를 리스트로 안 넘김
- 로그인 실패 시 "사용자 없음"과 "비밀번호 틀림"을 구분하면 보안 취약점 → 둘 다 401

---

## Step 4 해설: 게시글 CRUD / Post CRUD / CRUD Bai viet

### 핵심 코드 / Key Code / Code chinh

**app/api/post_router.py**
```python
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.post import Post
from app.schemas.post import PostCreate, PostUpdate, PostResponse

router = APIRouter()

def post_to_response(post: Post) -> dict:
    return {
        "id": post.id,
        "title": post.title,
        "content": post.content,
        "author_id": post.author_id,
        "author_username": post.author.username,
        "created_at": post.created_at,
        "updated_at": post.updated_at,
    }

@router.post("/", status_code=201)
def create_post(
    post_in: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = Post(
        title=post_in.title,
        content=post_in.content,
        author_id=current_user.id,
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    return post_to_response(post)

@router.get("/")
def list_posts(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    posts = db.query(Post).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    return [post_to_response(p) for p in posts]

@router.get("/{post_id}")
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post_to_response(post)

@router.put("/{post_id}")
def update_post(
    post_id: int,
    post_in: PostUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if post_in.title is not None:
        post.title = post_in.title
    if post_in.content is not None:
        post.content = post_in.content
    db.commit()
    db.refresh(post)
    return post_to_response(post)

@router.delete("/{post_id}", status_code=204)
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    db.delete(post)
    db.commit()
    return Response(status_code=204)
```

**main.py에 라우터 등록 추가:**
```python
from app.api.auth_router import router as auth_router
from app.api.post_router import router as post_router

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(post_router, prefix="/api/posts", tags=["posts"])
```

### 실제 서비스 매핑 / Real Service Mapping / Anh xa dich vu thuc

- `assetization_api/app/api/endpoints/`: 동일한 라우터 기반 CRUD 패턴
- `assetization_datacenter/app/features/feedback/router.py`: `Depends()` 인증 가드 패턴
- 실제 서비스에서도 `작성자 == 현재 사용자` 체크로 권한 제어

### 흔한 실수 / Common Mistakes / Loi thuong gap

- `relationship("User")`를 선언했지만 쿼리 시 lazy loading으로 N+1 문제 발생
- DELETE에서 `Response(status_code=204)` 대신 `return None` → 기본 200 반환
- `PostUpdate`에서 모든 필드가 Optional인데 빈 바디 `{}` 처리 안 함
- 목록에서 `order_by` 없으면 순서 보장 안 됨

---

## Step 5 해설: 에러 처리 & 로깅 / Error Handling & Logging / Xu ly loi & Ghi log

### 핵심 코드 / Key Code / Code chinh

**에러 응답 표준화 (main.py에 추가):**
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time

# 에러 응답 표준화
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND", 409: "DUPLICATE"}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": {
                "code": code_map.get(exc.status_code, "ERROR"),
                "message": exc.detail if isinstance(exc.detail, str) else exc.detail.get("message", str(exc.detail)),
            }
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "VALIDATION_ERROR",
                "message": str(exc.errors()),
            }
        },
    )

# 요청 로깅 미들웨어
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {request.method} | {request.url.path} | {response.status_code} | {duration}ms")
    return response
```

### 실제 서비스 매핑 / Real Service Mapping / Anh xa dich vu thuc

- `assetization_api/app/core/errors.py`: `AuthError` + `create_error_response()` 패턴
- `assetization_auth/app/middleware/logging_middleware.py`: Kafka로 로그 전송하는 미들웨어
- 실제 서비스는 Kafka + ELK 스택으로 로깅하지만, 핵심 패턴(미들웨어에서 요청 시작/종료 측정)은 동일

### 흔한 실수 / Common Mistakes / Loi thuong gap

- 미들웨어 등록 순서: 로깅 미들웨어는 CORS 다음에 등록해야 함
- `exception_handler`가 `HTTPException`의 서브클래스까지 잡으려면 별도 처리 필요
- `time.time()`은 정밀도가 낮을 수 있음 → `time.perf_counter()` 권장

---

## 전체 아키텍처 정리 / Architecture Summary / Tom tat kien truc

```
Client (curl / browser)
    │
    ▼
[Logging Middleware] ← 요청 시간 측정 / Request timing / Do thoi gian
    │
    ▼
[CORS Middleware] ← Cross-Origin 허용 / Allow CORS / Cho phep CORS
    │
    ▼
[Router] ─── /health ──────────────────→ 200 {"status": "ok"}
    │
    ├── /api/auth/signup ──────────────→ UserCreate → hash → DB → UserResponse
    ├── /api/auth/login ───────────────→ verify → JWT token
    │
    ├── /api/posts/ (GET) ─────────────→ DB query → PostResponse[]
    ├── /api/posts/{id} (GET) ─────────→ DB query → PostResponse
    ├── /api/posts/ (POST) ──[auth]───→ create → DB → PostResponse
    ├── /api/posts/{id} (PUT) ─[auth]─→ owner check → update → PostResponse
    └── /api/posts/{id} (DELETE) [auth]→ owner check → delete → 204
```

## 이 과제에서 배운 패턴과 실제 서비스 연결 / Patterns Learned vs Real Services / Cac pattern hoc duoc va dich vu thuc

| 이 과제 / This Assignment / Bai tap nay | 실제 Assetization 서비스 / Real Service / Dich vu thuc |
|----------------------------------------|-------------------------------------------------------|
| SQLite + SQLAlchemy | MSSQL + SQLAlchemy / Redis |
| JWT (python-jose) | SSO + Redis Session |
| `get_current_user` Depends | SSO Middleware + `request.state.user` |
| FastAPI Router + prefix | 동일 패턴 / Same pattern / Cung pattern |
| Pydantic BaseModel | 동일 패턴 / Same pattern / Cung pattern |
| `@app.middleware("http")` | LoggingMiddleware (Kafka) |
| HTTPException handler | AuthError + global handler |
| `get_db()` Depends | 동일 패턴 / Same pattern / Cung pattern |
