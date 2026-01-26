import math
from typing import List
from datetime import datetime, timedelta


def normalize_score(score: float) -> float:
    """使用log计算将分数归一化到0-100范围
    
    Args:
        score: 原始分数
        
    Returns:
        归一化后的分数（0-100）
    """
    # 使用log1p将分数映射到0-100，避免超过100
    return min(100.0, 100 * math.log1p(max(0, score)) / math.log1p(100))


def calculate_weighted_score(semantic_score: float, behavior_score: float, event_score: float) -> float:
    """计算加权总分
    
    Args:
        semantic_score: AI语义分（权重50%）
        behavior_score: 行为规则修正分（权重30%）
        event_score: 关键事件分（权重20%）
        
    Returns:
        加权总分（0-100）
    """
    weighted_semantic = semantic_score * 0.5
    weighted_behavior = behavior_score * 0.3
    weighted_event = event_score * 0.2
    
    total_score = weighted_semantic + weighted_behavior + weighted_event
    return normalize_score(total_score)


def get_time_difference_in_hours(time1: datetime, time2: datetime) -> float:
    """计算两个时间之间的小时差
    
    Args:
        time1: 第一个时间
        time2: 第二个时间
        
    Returns:
        小时差
    """
    return abs((time2 - time1).total_seconds() / 3600)


def get_time_difference_in_minutes(time1: datetime, time2: datetime) -> float:
    """计算两个时间之间的分钟差
    
    Args:
        time1: 第一个时间
        time2: 第二个时间
        
    Returns:
        分钟差
    """
    return abs((time2 - time1).total_seconds() / 60)


def get_time_difference_in_seconds(time1: datetime, time2: datetime) -> float:
    """计算两个时间之间的秒差
    
    Args:
        time1: 第一个时间
        time2: 第二个时间
        
    Returns:
        秒差
    """
    return abs((time2 - time1).total_seconds())
