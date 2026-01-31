from typing import Optional
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.auth import UserCreate, UserLogin
from app.utils.security import verify_password, get_password_hash


def create_user(db: Session, user_create: UserCreate) -> User:
    """创建用户"""
    # 检查用户名是否已存在
    existing_user = db.query(User).filter(User.username == user_create.username).first()
    if existing_user:
        raise ValueError("用户名已存在")

    # 创建新用户
    hashed_password = get_password_hash(user_create.password)
    db_user = User(
        username=user_create.username,
        password_hash=hashed_password,
        role=user_create.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, user_login: UserLogin, expected_role: Optional[str] = None) -> Optional[User]:
    """认证用户"""
    # 查找用户
    user = db.query(User).filter(User.username == user_login.username).first()
    if not user:
        return None
    # 验证密码
    if not verify_password(user_login.password, user.password_hash):
        return None
    # 验证角色
    if expected_role and user.role != expected_role:
        return None
    return user


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """根据ID获取用户"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """根据用户名获取用户"""
    return db.query(User).filter(User.username == username).first()