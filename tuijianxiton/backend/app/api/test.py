from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(
    prefix="/test",
    tags=["test"],
    responses={404: {"description": "Not found"}},
)


@router.get("/hello/{chat_room_id}")
async def hello(chat_room_id: int) -> Dict[str, Any]:
    """测试端点
    
    Args:
        chat_room_id: 聊天室ID
        
    Returns:
        测试结果
    """
    return {
        "chat_room_id": chat_room_id,
        "message": "Hello, World!"
    }
