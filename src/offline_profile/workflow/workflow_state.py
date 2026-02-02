from typing import TypedDict, Optional, List, Dict, Any
import datetime


class OfflineProfileState(TypedDict, total=False):
    """离线用户画像工作流状态"""
    # 基本信息
    user_id: str  # 用户 ID
    start_time: datetime.datetime  # 开始时间
    end_time: datetime.datetime  # 结束时间
    standard_tags: List[Dict[str, Any]]  # 结束时间
    
    # 处理数据
    chat_data: Dict[str, Any]  # 处理后的聊天数据
    summaries: List[Dict[str, Any]]  # 对话摘要
    extracted_tags: List[Dict[str, Any]]  # 提取的标签
    extracted_data: Dict[str, Any]  # 提取的结构化数据 (global_traits, intents, instructions)
    
    # 画像信息
    existing_profile: Dict[str, Any]  # 现有用户画像
    new_profile: Dict[str, Any]  # 新生成的用户画像
    
    # 处理状态
    status: str  # 处理状态：pending, processing, success, error, no_summaries, no_tags
    error_message: str  # 错误信息
    
    # 处理结果
    tag_count: int  # 标签数量
    updated_at: str  # 更新时间
