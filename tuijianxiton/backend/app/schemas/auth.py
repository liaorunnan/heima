from pydantic import BaseModel, Field
from typing import Optional


class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")


class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, description="密码")
    role: str = Field(..., pattern="^(customer|merchant)$", description="角色")


class UserLogin(UserBase):
    """用户登录模型"""
    password: str = Field(..., description="密码")


class User(UserBase):
    """用户响应模型"""
    id: int
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """令牌数据模型"""
    user_id: Optional[int] = None
    username: Optional[str] = None
    role: Optional[str] = None