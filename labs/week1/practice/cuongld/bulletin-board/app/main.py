import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.auth_router import router as auth_router
from app.core.database import Base, engine
from app.api.post_router import router as post_router
from app.models import post, user

app = FastAPI(title="Bulletin Board API")

# Assignment requires allowing all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, __):
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "VALIDATION_ERROR",
                "message": "Validation error",
            }
        },
    )


@app.middleware("http")
async def request_logger(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{timestamp} | {request.method} | {request.url.path} | "
        f"{response.status_code} | {duration_ms}ms"
    )
    return response


@app.on_event("startup")
def on_startup() -> None:
    # Import models before create_all so SQLAlchemy can register all tables.
    _ = (user, post)
    Base.metadata.create_all(bind=engine)


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(post_router, prefix="/api/posts", tags=["posts"])
