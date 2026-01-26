from typing import List, Optional, Dict, Any
from datetime import datetime
from typing_extensions import TypedDict


class Message(TypedDict):
    """单条消息模型
    
    Args:
        content: 消息内容
        timestamp: 消息发送时间
        sender: 消息发送者（'user'或'agent'）
    """
    content: str
    timestamp: datetime
    sender: str


class MessageGroup(TypedDict):
    """消息组模型
    
    Args:
        messages: 消息组包含的消息列表
        timestamp: 消息组的时间戳（取第一条消息的时间）
        count: 消息组中的消息数量
    """
    messages: List[Message]
    timestamp: datetime
    count: int


class Session(TypedDict):
    """会话模型
    
    Args:
        session_id: 会话唯一标识
        start_time: 会话开始时间
        end_time: 会话结束时间（可选，未结束的会话为None）
        initial_score: 会话初始分数
        current_score: 会话当前分数
        message_groups: 会话包含的消息组列表
        last_semantic_recalculation: 距离上次语义重算的消息组数
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    initial_score: float
    current_score: float
    message_groups: List[MessageGroup]
    last_semantic_recalculation: int


class KeyEvent(TypedDict):
    """关键事件模型
    
    Args:
        event_type: 事件类型
        timestamp: 事件发生时间
        details: 事件详细信息
    """
    event_type: str
    timestamp: datetime
    details: Dict[str, Any]
