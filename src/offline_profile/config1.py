import os
import logging
from typing import Dict, Any

from src.utils.logger_utils import setup_logger

logger = setup_logger('offline_profile')
# 性能配置
PERFORMANCE_CONFIG: Dict[str, Any] = {
    'enable_caching': True,  # 启用缓存
    'cache_ttl_seconds': 3600,  # 缓存过期时间（秒）
    'batch_size': 100,  # 批处理大小
    'max_workers': 4,  # 最大工作线程数
}

# 日志配置
LOG_CONFIG: Dict[str, Any] = {
    'enable_performance_logs': True,  # 启用性能日志
    'enable_detailed_logs': False,  # 启用详细日志
    'log_file_size': 10 * 1024 * 1024,  # 日志文件大小（10MB）
    'log_backup_count': 5,  # 日志文件备份数量
}


# 时间衰减配置
DECAY_CONFIG: Dict[str, Any] = {
    'short_term': 0.7,  # 短期兴趣衰减系数
    'long_term': 0.9,   # 长期兴趣衰减系数
    'default': 0.9,     # 默认衰减系数
    'min_score': 0.1,   # 最小权重阈值，低于此值的标签将被删除
    'max_score': 1.0,   # 最大权重阈值
}

# 意图分级得分配置
INTENT_SCORE_MAP: Dict[str, float] = {
    'L0': 0.1,  # 提及
    'L1': 0.3,  # 咨询
    'L2': 0.5,  # 偏好
    'L3': 0.8,  # 紧迫
    'L4': -1.0, # 拒绝
}

# 存储配置
STORAGE_CONFIG: Dict[str, Any] = {
    'profile_dir': os.path.join(os.path.dirname(__file__), 'profiles'),
    'max_tags': 50,  # 每个用户最多保留的标签数量
    'history_days': 30,  # 保留历史数据的天数
}

# LLM 配置
LLM_CONFIG: Dict[str, Any] = {
    'model_name': 'default',  # 使用默认模型
    'temperature': 0.3,  # 低温度，减少随机性
    'max_tokens': 2000,  # 最大生成 token 数
}

# 数据处理配置
DATA_PROCESSING_CONFIG: Dict[str, Any] = {
    'session_timeout_minutes': 30,  # 会话超时时间（分钟）
    'max_chat_history_days': 30,  # 处理最近30天的聊天记录
    'batch_size': 100,  # 批处理大小
}

# 敏感信息正则表达式
PII_REGEX: Dict[str, str] = {
    'phone': r'1[3-9]\d{9}',
    'id_card': r'[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]',
    'address': r'[省市自治区]{1,3}[市州盟]{1,3}[区县市旗]{1,3}[乡镇街道]{1,3}[村组社区]{1,3}',
}

# 确保存储目录存在
os.makedirs(STORAGE_CONFIG['profile_dir'], exist_ok=True)
