from typing import Optional, List
from datetime import datetime, timedelta
from uuid import uuid4
from ..models.data_models import Session, MessageGroup
from ..utils.score_utils import get_time_difference_in_hours


class SessionManager:
    """管理逻辑会话的创建、切分和分数继承"""
    
    # 会话切分阈值：用户静默时间 > 12小时
    SESSION_SPLIT_THRESHOLD_HOURS = 12
    
    # 分数继承衰减系数
    SCORE_INHERITANCE_DECAY = 0.4
    
    def create_new_session(self, last_session_final_score: float = 0.0) -> Session:
        """创建新会话，处理分数继承
        
        Args:
            last_session_final_score: 上一个会话的最终分数
            
        Returns:
            新创建的会话对象
        """
        # 计算新会话初始分 = 上一轮最终分 × 衰减系数
        initial_score = last_session_final_score * self.SCORE_INHERITANCE_DECAY
        
        return {
            "session_id": str(uuid4()),
            "start_time": datetime.now(),
            "end_time": None,
            "initial_score": initial_score,
            "current_score": initial_score,
            "message_groups": [],
            "last_semantic_recalculation": 0
        }
    
    def should_split_session(self, last_message_time: datetime, current_time: datetime) -> bool:
        """判断是否需要切分会话
        
        Args:
            last_message_time: 上一条消息的时间
            current_time: 当前时间
            
        Returns:
            是否需要切分会话（True/False）
        """
        time_diff_hours = get_time_difference_in_hours(last_message_time, current_time)
        return time_diff_hours > self.SESSION_SPLIT_THRESHOLD_HOURS
    
    def end_session(self, session: Session) -> Session:
        """结束会话
        
        Args:
            session: 要结束的会话对象
            
        Returns:
            结束后的会话对象
        """
        session["end_time"] = datetime.now()
        return session
    
    def add_message_group_to_session(self, session: Session, message_group: MessageGroup) -> Session:
        """向会话添加消息组
        
        Args:
            session: 会话对象
            message_group: 要添加的消息组
            
        Returns:
            更新后的会话对象
        """
        session["message_groups"].append(message_group)
        session["last_semantic_recalculation"] += 1
        return session
    
    def update_session_score(self, session: Session, new_score: float) -> Session:
        """更新会话分数
        
        Args:
            session: 会话对象
            new_score: 新的分数
            
        Returns:
            更新后的会话对象
        """
        session["current_score"] = new_score
        return session
    
    def reset_semantic_recalculation_counter(self, session: Session) -> Session:
        """重置语义重算计数器
        
        Args:
            session: 会话对象
            
        Returns:
            更新后的会话对象
        """
        session["last_semantic_recalculation"] = 0
        return session
