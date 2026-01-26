from typing import List, Optional
from datetime import datetime, timedelta
from ..models.data_models import KeyEvent, MessageGroup


class KeyEventScorer:
    """处理关键事件评分（权重20%）"""
    
    def __init__(self):
        # 关键事件评分规则
        self.event_rules = {
            "点击商品卡片": 5.0,
            "询问价格": 15.0,
            "触发知识库未覆盖": 10.0,
            "主动索要支付链接": 30.0,
            "点击支付/结账链接": 40.0
        }
    
    def calculate_event_score(self, key_event: KeyEvent) -> float:
        """计算关键事件分数
        
        Args:
            key_event: 关键事件对象
            
        Returns:
            关键事件分数
        """
        event_type = key_event["event_type"]
        return self.event_rules.get(event_type, 0.0)
    
    def check_and_score_message_events(self, message_group: MessageGroup) -> float:
        """检查消息内容中的关键事件并计算分数
        
        Args:
            message_group: 消息组对象
            
        Returns:
            消息中关键事件的分数
        """
        event_score = 0.0
        
        # 遍历消息组中的所有消息
        for message in message_group["messages"]:
            # 只处理用户消息
            if message["sender"] != "user":
                continue
            
            content = message["content"]
            
            # 检查询问价格
            if self._has_price_inquiry(content):
                event_score += self.event_rules["询问价格"]
            
            # 检查主动索要支付链接
            if self._has_payment_link_request(content):
                event_score += self.event_rules["主动索要支付链接"]
        
        return event_score
    
    def check_and_score_system_events(self, system_events: List[KeyEvent]) -> float:
        """检查系统事件中的关键事件并计算分数
        
        Args:
            system_events: 系统事件列表
            
        Returns:
            系统事件中关键事件的分数
        """
        event_score = 0.0
        
        # 遍历所有系统事件
        for event in system_events:
            event_score += self.calculate_event_score(event)
        
        return event_score
    
    def _has_price_inquiry(self, content: str) -> bool:
        """判断是否为询问价格
        
        Args:
            content: 消息内容
            
        Returns:
            是否为询问价格
        """
        price_keywords = ["多少钱", "价格", "报价", "售价", "费用"]
        return any(keyword in content for keyword in price_keywords)
    
    def _has_payment_link_request(self, content: str) -> bool:
        """判断是否为主动索要支付链接
        
        Args:
            content: 消息内容
            
        Returns:
            是否为主动索要支付链接
        """
        payment_link_keywords = ["支付链接", "付款链接", "结账链接", "购买链接"]
        return any(keyword in content for keyword in payment_link_keywords)
