from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.auth import User, UserCreate, Token
from app.services.auth_service import create_user, authenticate_user
from app.utils.security import create_access_token
from app.schemas.auth import TokenData
from jose import JWTError, jwt
from app.utils.security import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/api/auth", tags=["认证"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id))
    except JWTError:
        raise credentials_exception
    from app.services.auth_service import get_user_by_id
    user = get_user_by_id(db, user_id=token_data.user_id)
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=User)
def register(user_create: UserCreate, db: Session = Depends(get_db)):
    """用户注册"""
    try:
        user = create_user(db=db, user_create=user_create)
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


from pydantic import BaseModel


class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str
    password: str
    expected_role: Optional[str] = None


@router.post("/login", response_model=Token)
def login(login_request: LoginRequest, db: Session = Depends(get_db)):
    """用户登录"""
    from app.schemas.auth import UserLogin
    login_data = UserLogin(username=login_request.username, password=login_request.password)
    user = authenticate_user(db=db, user_login=login_data, expected_role=login_request.expected_role)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误，或角色不匹配",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/form", response_model=Token)
def login_form(user_login: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """用户登录（表单格式）"""
    from app.schemas.auth import UserLogin
    login_data = UserLogin(username=user_login.username, password=user_login.password)
    user = authenticate_user(db=db, user_login=login_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}