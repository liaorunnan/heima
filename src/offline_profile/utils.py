import re
import json
import datetime
from typing import Dict, List, Any, Optional

from src.workflow.offline_profile.config1 import PII_REGEX


def format_datetime(dt: datetime.datetime) -> str:
    """格式化日期时间为字符串
    
    Args:
        dt: 日期时间对象
        
    Returns:
        格式化后的字符串，格式为 "YYYY-MM-DD HH:MM:SS"
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime(dt_str: str) -> datetime.datetime:
    """解析日期时间字符串
    
    Args:
        dt_str: 日期时间字符串，格式为 "YYYY-MM-DD HH:MM:SS"
        
    Returns:
        日期时间对象
    """
    return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def get_yesterday() -> datetime.datetime:
    """获取昨天的日期
    
    Returns:
        昨天的日期时间对象，时间为 00:00:00
    """
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    return today - datetime.timedelta(days=1)


def remove_pii(text: str) -> str:
    """去除文本中的敏感信息
    
    Args:
        text: 原始文本
        
    Returns:
        去除敏感信息后的文本
    """
    # 替换手机号
    text = re.sub(PII_REGEX['phone'], '[PHONE]', text)
    # 替换身份证号
    text = re.sub(PII_REGEX['id_card'], '[ID_CARD]', text)
    # 替换地址
    text = re.sub(PII_REGEX['address'], '[ADDRESS]', text)
    return text


def clean_text(text: str) -> str:
    """清洗文本
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    text = text.strip()
    return text


def load_json(file_path: str) -> Dict[str, Any]:
    """加载 JSON 文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON 数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """保存数据到 JSON 文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        
    Returns:
        合并后的字典
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def calculate_time_diff(start_time: datetime.datetime, end_time: datetime.datetime) -> int:
    """计算时间差（分钟）
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        时间差（分钟）
    """
    diff = end_time - start_time
    return int(diff.total_seconds() / 60)


def is_valid_tag(tag: Dict[str, Any]) -> bool:
    """验证标签是否有效
    
    Args:
        tag: 标签字典
        
    Returns:
        是否有效
    """
    required_fields = ['id', 'name']
    for field in required_fields:
        if field not in tag:
            return False
    return True


def normalize_tag_name(tag_name: str) -> str:
    """标准化标签名称
    
    Args:
        tag_name: 原始标签名称
        
    Returns:
        标准化后的标签名称
    """
    # 转为小写
    tag_name = tag_name.lower()
    # 去除多余的空格
    tag_name = re.sub(r'\s+', ' ', tag_name)
    # 去除首尾空格
    tag_name = tag_name.strip()
    return tag_name


def batch_process(items: List[Any], batch_size: int) -> List[List[Any]]:
    """批量处理数据
    
    Args:
        items: 数据列表
        batch_size: 批次大小
        
    Returns:
        批次列表
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i+batch_size])
    return batches
