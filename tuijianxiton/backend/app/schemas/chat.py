from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessageBase(BaseModel):
    """聊天消息基础模型"""
    content: str = Field(..., description="消息内容")
    message_type: str = Field(..., pattern="^(text|product|image|video|audio|file)$", description="消息类型")
    product_id: Optional[int] = Field(None, description="商品ID")


class ChatMessageCreate(ChatMessageBase):
    """聊天消息创建模型"""
    pass


class ChatMessage(ChatMessageBase):
    """聊天消息响应模型"""
    id: int
    chat_room_id: int
    sender_id: int
    sender_role: str = Field(..., description="发送者角色")
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRoomBase(BaseModel):
    """聊天室基础模型"""
    customer_id: int = Field(..., description="客户ID")
    merchant_id: int = Field(..., description="商家ID")


class ChatRoomCreate(ChatRoomBase):
    """聊天室创建模型"""
    pass


class ChatRoom(ChatRoomBase):
    """聊天室响应模型"""
    id: int
    created_at: datetime
    last_message: Optional[str] = Field(None, description="最后一条消息")
    last_message_time: Optional[datetime] = Field(None, description="最后一条消息时间")

    class Config:
        from_attributes = True


class ChatRoomWithMessages(ChatRoom):
    """带消息的聊天室响应模型"""
    messages: List[ChatMessage] = Field(default_factory=list, description="消息列表")