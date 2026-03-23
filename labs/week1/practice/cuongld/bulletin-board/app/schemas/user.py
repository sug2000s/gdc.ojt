from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime | None = None
    is_active: bool

    model_config = ConfigDict(from_attributes=True)
