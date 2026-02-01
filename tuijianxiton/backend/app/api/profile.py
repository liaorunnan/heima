from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.database import get_db
from app.services.profile_service import (
    process_chat_room_profile,
    get_chat_room_tags,
    get_chat_room_profile
)

router = APIRouter(
    prefix="/profile",
    tags=["profile"],
    responses={404: {"description": "Not found"}},
)


@router.post("/process/chat-room/{chat_room_id}")
async def process_profile(chat_room_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """处理聊天室画像，包括聊天记录打标和画像构建
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        处理结果，包含标签和画像信息
    """
    try:
        result = process_chat_room_profile(chat_room_id, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理聊天室画像失败: {str(e)}")


@router.get("/tags/chat-room/{chat_room_id}")
async def get_tags(chat_room_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """获取聊天室标签
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        聊天室标签列表
    """
    try:
        tags = get_chat_room_tags(chat_room_id, db)
        return tags
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天室标签失败: {str(e)}")


@router.get("/chat-room/{chat_room_id}")
async def get_profile(chat_room_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """获取聊天室画像
    
    Args:
        chat_room_id: 聊天室ID
        db: 数据库会话
        
    Returns:
        聊天室画像信息
    """
    try:
        profile = get_chat_room_profile(chat_room_id, db)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取聊天室画像失败: {str(e)}")
