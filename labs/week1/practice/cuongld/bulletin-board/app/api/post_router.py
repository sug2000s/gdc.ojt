from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.database import get_db
from app.core.errors import raise_api_error
from app.core.security import get_current_user
from app.models.post import Post
from app.models.user import User
from app.schemas.post import PostCreate, PostResponse, PostUpdate

router = APIRouter()


def _to_post_response(post: Post) -> PostResponse:
    return PostResponse(
        id=post.id,
        title=post.title,
        content=post.content,
        author_id=post.author_id,
        author_username=post.author.username if post.author else None,
        created_at=post.created_at,
        updated_at=post.updated_at,
    )


@router.post("/", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
def create_post(
    payload: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PostResponse:
    post = Post(title=payload.title, content=payload.content, author_id=current_user.id)
    db.add(post)
    db.commit()
    db.refresh(post)
    return _to_post_response(post)


@router.get("/", response_model=list[PostResponse])
def list_posts(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)) -> list[PostResponse]:
    posts = db.query(Post).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    return [_to_post_response(post) for post in posts]


@router.get("/{post_id}", response_model=PostResponse)
def get_post(post_id: int, db: Session = Depends(get_db)) -> PostResponse:
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise_api_error(status.HTTP_404_NOT_FOUND, "NOT_FOUND", "Post not found")
    return _to_post_response(post)


@router.put("/{post_id}", response_model=PostResponse)
def update_post(
    post_id: int,
    payload: PostUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PostResponse:
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise_api_error(status.HTTP_404_NOT_FOUND, "NOT_FOUND", "Post not found")
    if post.author_id != current_user.id:
        raise_api_error(status.HTTP_403_FORBIDDEN, "FORBIDDEN", "Forbidden")

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(post, field, value)

    db.commit()
    db.refresh(post)
    return _to_post_response(post)


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise_api_error(status.HTTP_404_NOT_FOUND, "NOT_FOUND", "Post not found")
    if post.author_id != current_user.id:
        raise_api_error(status.HTTP_403_FORBIDDEN, "FORBIDDEN", "Forbidden")

    db.delete(post)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
