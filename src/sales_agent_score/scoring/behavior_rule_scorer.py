from typing import List, Optional
from datetime import datetime, timedelta
from ..models.data_models import MessageGroup, Session
from ..utils.score_utils import get_time_difference_in_minutes


class BehaviorRuleScorer:
    """处理行为规则修正（权重30%）"""
    
    def __init__(self):
        # 秒回率规则：用户回复时间 < 2分钟，+2分/次，上限20分
        self.reply_speed_rule = {
            "threshold_minutes": 2.0,
            "score_per_time": 2.0,
            "max_score": 20.0
        }
        
        # 报价静默规则：Agent发送含数字/价格的消息后，用户>30分钟未回，-10分/次
        self.quote_silence_rule = {
            "threshold_minutes": 30.0,
            "penalty_score": -10.0
        }
    
    def calculate_adjustment(self, message_group: MessageGroup, session: Session) -> float:
        """计算行为规则修正分
        
        Args:
            message_group: 当前消息组
            session: 当前会话对象
            
        Returns:
            行为规则修正分
        """
        adjustment_score = 0.0
        
        # 计算秒回率分数
        adjustment_score += self.calculate_reply_speed_score(message_group, session)
        
        # 计算报价静默分数
        adjustment_score += self.calculate_quote_silence_score(message_group, session)
        
        return adjustment_score
    
    def calculate_reply_speed_score(self, message_group: MessageGroup, session: Session) -> float:
        """计算秒回率分数
        
        Args:
            message_group: 当前消息组
            session: 当前会话对象
            
        Returns:
            秒回率分数
        """
        # 只有用户消息需要计算秒回率
        if message_group["messages"][0]["sender"] != "user":
            return 0.0
        
        # 检查是否有前一个消息组
        message_groups = session["message_groups"]
        if not message_groups:
            return 0.0
        
        # 获取上一个消息组（应该是Agent消息）
        last_message_group = message_groups[-1]
        if last_message_group["messages"][0]["sender"] != "agent":
            return 0.0
        
        # 计算回复时间差（分钟）
        reply_time_minutes = get_time_difference_in_minutes(
            last_message_group["timestamp"],
            message_group["timestamp"]
        )
        
        # 检查是否符合秒回条件
        if reply_time_minutes < self.reply_speed_rule["threshold_minutes"]:
            return self.reply_speed_rule["score_per_time"]
        
        return 0.0
    
    def calculate_quote_silence_score(self, message_group: MessageGroup, session: Session) -> float:
        """计算报价静默分数
        
        Args:
            message_group: 当前消息组
            session: 当前会话对象
            
        Returns:
            报价静默分数
        """
        # 检查Agent是否发送了含价格的消息
        quote_message_time = self._find_last_quote_message_time(session["message_groups"])
        if not quote_message_time:
            return 0.0
        
        # 计算静默时间（分钟）
        silence_duration_minutes = get_time_difference_in_minutes(
            quote_message_time,
            message_group["timestamp"]
        )
        
        # 检查是否符合报价静默条件
        if silence_duration_minutes > self.quote_silence_rule["threshold_minutes"]:
            return self.quote_silence_rule["penalty_score"]
        
        return 0.0
    
    def _find_last_quote_message_time(self, message_groups: List[MessageGroup]) -> Optional[datetime]:
        """查找最后一条含价格的Agent消息时间
        
        Args:
            message_groups: 消息组列表
            
        Returns:
            最后一条含价格的Agent消息时间，如没有则返回None
        """
        # 倒序遍历消息组
        for message_group in reversed(message_groups):
            # 遍历消息组中的消息
            for message in message_group["messages"]:
                # 只处理Agent消息
                if message["sender"] != "agent":
                    continue
                
                # 检查是否包含价格（简单实现：包含数字）
                if any(char.isdigit() for char in message["content"]):
                    return message["timestamp"]
        
        return None
