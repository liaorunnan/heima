from typing import List, Optional
from datetime import datetime, timedelta
from src.sales_agent_score.models.data_models import MessageGroup, Session, KeyEvent
from src.sales_agent_score.utils.score_utils import calculate_weighted_score
from src.sales_agent_score.scoring.ai_semantic_scorer import AISemanticScorer
from src.sales_agent_score.scoring.behavior_rule_scorer import BehaviorRuleScorer
from src.sales_agent_score.scoring.key_event_scorer import KeyEventScorer


class ScoreCalculator:
    """核心评分引擎，整合AI语义评分、行为规则修正和关键事件突变"""
    
    def __init__(self):
        # 初始化各个评分器
        self.semantic_scorer = AISemanticScorer()
        self.behavior_scorer = BehaviorRuleScorer()
        self.key_event_scorer = KeyEventScorer()
        
        # AI语义重算触发条件：累计3个消息组
        self.SEMANTIC_RECALCULATION_THRESHOLD = 3
    
    def calculate_total_score(self, message_group: MessageGroup, session: Session, system_events: List[KeyEvent] = None) -> float:
        """计算总意愿分
        
        Args:
            message_group: 当前消息组
            session: 当前会话对象
            system_events: 系统事件列表（可选）
            
        Returns:
            总意愿分（0-100）
        """
        if system_events is None:
            system_events = []
        
        # 检查是否需要重新计算AI语义分
        recalculate_semantic = self.should_recalculate_semantic(session)
        
        # 获取AI语义分
        semantic_score = self._get_semantic_score(session, recalculate_semantic)
        
        # 计算行为规则修正分
        behavior_score = self.behavior_scorer.calculate_adjustment(message_group, session)
        
        # 计算关键事件分
        event_score = self._calculate_event_score(message_group, system_events)
        
        # 计算加权总分
        total_score = calculate_weighted_score(semantic_score, behavior_score, event_score)
        
        return total_score
    
    def should_recalculate_semantic(self, session: Session) -> bool:
        """判断是否需要重新计算AI语义分
        
        Args:
            session: 当前会话对象
            
        Returns:
            是否需要重新计算AI语义分
        """
        # 检查是否达到累计消息组阈值
        return session["last_semantic_recalculation"] >= self.SEMANTIC_RECALCULATION_THRESHOLD
    
    def _get_semantic_score(self, session: Session, recalculate: bool) -> float:
        """获取AI语义分
        
        Args:
            session: 当前会话对象
            recalculate: 是否需要重新计算
            
        Returns:
            AI语义分
        """
        if recalculate:
            # 重新计算AI语义分
            return self.semantic_scorer.score(session["message_groups"])
        else:
            # 使用当前会话分数作为基础语义分
            return session["current_score"]
    
    def _calculate_event_score(self, message_group: MessageGroup, system_events: List[KeyEvent]) -> float:
        """计算关键事件分
        
        Args:
            message_group: 当前消息组
            system_events: 系统事件列表
            
        Returns:
            关键事件分
        """
        event_score = 0.0
        
        # 检查消息内容中的关键事件
        event_score += self.key_event_scorer.check_and_score_message_events(message_group)
        
        # 检查系统事件中的关键事件
        event_score += self.key_event_scorer.check_and_score_system_events(system_events)
        
        return event_score
    
    def update_session_with_new_score(self, session: Session, new_score: float, recalculate_semantic: bool) -> Session:
        """使用新分数更新会话
        
        Args:
            session: 当前会话对象
            new_score: 新计算的分数
            recalculate_semantic: 是否重新计算了AI语义分
            
        Returns:
            更新后的会话对象
        """
        # 更新会话当前分数
        session["current_score"] = new_score
        
        # 如果重新计算了AI语义分，重置计数器
        if recalculate_semantic:
            session["last_semantic_recalculation"] = 0
        
        return session
