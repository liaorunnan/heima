from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.schemas.chat import ChatRoom, ChatRoomCreate, ChatMessage, ChatMessageCreate, ChatRoomWithMessages
from app.services.chat_service import (
    create_chat_room, get_chat_room_by_id, get_chat_rooms_by_user_id,
    create_chat_message, get_chat_messages_by_room_id
)
from app.api.auth import get_current_user
from app.models.user import User


class TargetUsername(BaseModel):
    """目标用户名模型"""
    target_username: str

router = APIRouter(prefix="/api/chat", tags=["聊天"])


@router.post("/rooms", response_model=ChatRoom)
def create_room(chat_room_create: ChatRoomCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """创建聊天室"""
    # 验证用户是否有权限创建聊天室
    if current_user.role == "customer" and current_user.id != chat_room_create.customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限创建此聊天室"
        )
    if current_user.role == "merchant" and current_user.id != chat_room_create.merchant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限创建此聊天室"
        )

    try:
        chat_room = create_chat_room(db=db, chat_room_create=chat_room_create)
        return chat_room
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/rooms/by-username", response_model=ChatRoom)
def create_room_by_username(target: TargetUsername, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """通过用户名创建聊天室"""
    # 查找目标用户
    target_user = db.query(User).filter(User.username == target.target_username).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="目标用户不存在"
        )
    
    # 验证角色匹配
    if current_user.role == "customer" and target_user.role != "merchant":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="客户只能与商家创建聊天室"
        )
    if current_user.role == "merchant" and target_user.role != "customer":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="商家只能与客户创建聊天室"
        )
    
    # 构建聊天室创建数据
    if current_user.role == "customer":
        chat_room_create = ChatRoomCreate(
            customer_id=current_user.id,
            merchant_id=target_user.id
        )
    else:  # merchant
        chat_room_create = ChatRoomCreate(
            customer_id=target_user.id,
            merchant_id=current_user.id
        )
    
    try:
        chat_room = create_chat_room(db=db, chat_room_create=chat_room_create)
        return chat_room
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/rooms", response_model=List[ChatRoom])
def get_rooms(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取用户的聊天室列表"""
    chat_rooms = get_chat_rooms_by_user_id(db=db, user_id=current_user.id, role=current_user.role)
    return chat_rooms


@router.get("/rooms/{room_id}", response_model=ChatRoomWithMessages)
def get_room(room_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取聊天室详情"""
    chat_room = get_chat_room_by_id(db=db, chat_room_id=room_id)
    if not chat_room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="聊天室不存在"
        )

    # 验证用户是否有权限访问此聊天室
    if current_user.id != chat_room.customer_id and current_user.id != chat_room.merchant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限访问此聊天室"
        )

    # 获取聊天室消息
    messages = get_chat_messages_by_room_id(db=db, chat_room_id=room_id)
    chat_room.messages = messages
    return chat_room


@router.post("/rooms/{room_id}/messages", response_model=ChatMessage)
def create_message(room_id: int, message_create: ChatMessageCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """发送消息"""
    try:
        message = create_chat_message(
            db=db,
            chat_room_id=room_id,
            sender_id=current_user.id,
            message_create=message_create
        )
        return message
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/rooms/{room_id}/messages", response_model=List[ChatMessage])
def get_messages(room_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取聊天室消息"""
    # 验证用户是否有权限访问此聊天室
    chat_room = get_chat_room_by_id(db=db, chat_room_id=room_id)
    if not chat_room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="聊天室不存在"
        )

    if current_user.id != chat_room.customer_id and current_user.id != chat_room.merchant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限访问此聊天室"
        )

    messages = get_chat_messages_by_room_id(db=db, chat_room_id=room_id)
    return messages