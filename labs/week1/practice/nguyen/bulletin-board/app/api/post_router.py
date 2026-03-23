from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List
from app.schemas.post import PostCreate, PostUpdate, PostResponse
from app.models.post import Post
from app.models.user import User
from app.core.database import get_db
from app.core.security import get_current_user

router = APIRouter(prefix="/api/posts", tags=["posts"])

@router.post("/", response_model=PostResponse, status_code=201)
def create_post(
    post_in: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    post = Post(
        title=post_in.title,
        content=post_in.content,
        author_id=current_user.id
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    
    response = PostResponse(
        id=post.id,
        title=post.title,
        content=post.content,
        author_id=post.author_id,
        author_username=current_user.username,
        created_at=post.created_at,
        updated_at=post.updated_at
    )
    return response

@router.get("/", response_model=List[PostResponse])
def list_posts(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    posts = db.query(Post).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for post in posts:
        author = db.query(User).filter(User.id == post.author_id).first()
        result.append(PostResponse(
            id=post.id,
            title=post.title,
            content=post.content,
            author_id=post.author_id,
            author_username=author.username if author else "Unknown",
            created_at=post.created_at,
            updated_at=post.updated_at
        ))
    return result

@router.get("/{post_id}", response_model=PostResponse)
def get_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    author = db.query(User).filter(User.id == post.author_id).first()
    return PostResponse(
        id=post.id,
        title=post.title,
        content=post.content,
        author_id=post.author_id,
        author_username=author.username if author else "Unknown",
        created_at=post.created_at,
        updated_at=post.updated_at
    )

@router.put("/{post_id}", response_model=PostResponse)
def update_post(
    post_id: int,
    post_in: PostUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if post_in.title is not None:
        post.title = post_in.title
    if post_in.content is not None:
        post.content = post_in.content
    
    db.commit()
    db.refresh(post)
    
    return PostResponse(
        id=post.id,
        title=post.title,
        content=post.content,
        author_id=post.author_id,
        author_username=current_user.username,
        created_at=post.created_at,
        updated_at=post.updated_at
    )

@router.delete("/{post_id}", status_code=204)
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    db.delete(post)
    db.commit()
    return Response(status_code=204)
