from typing import Dict, Any, List
import datetime
from src.workflow.offline_profile.user_persona import UserPersonaSystem
from src.workflow.offline_profile.utils import format_datetime
from src.workflow.offline_profile.config1 import logger

def fuse_profiles(existing_profile: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """融合用户画像
    
    Args:
        existing_profile: 现有用户画像
        extracted_data: 新提取的结构化数据 (global_traits, intents, instructions)
        
    Returns:
        融合后的用户画像
    """
    logger.info(f"开始融合画像，用户ID: {existing_profile.get('user_id')}")
    
    try:
        system = UserPersonaSystem()
        
        # 加载现有画像
        system.load_from_profile(existing_profile)
        
        # 更新画像
        if extracted_data:
            system.update_persona(extracted_data)
        
        # 获取最终画像
        final_profile = system.get_final_persona()
        
        # 添加元数据
        final_profile['user_id'] = existing_profile.get('user_id', '')
        final_profile['updated_at'] = format_datetime(datetime.datetime.now())
        
        logger.info("画像融合完成")
        return final_profile
        
    except Exception as e:
        logger.error(f"画像融合失败: {e}")
        # 出错时返回原有画像，避免数据丢失
        return existing_profile

def create_empty_profile(user_id: str) -> Dict[str, Any]:
    """创建空的用户画像
    
    Args:
        user_id: 用户 ID
        
    Returns:
        空的用户画像
    """
    return {
        'user_id': user_id,
        'global_traits': {},
        'category_intents': {},
        'recent_instructions': [],
        'updated_at': format_datetime(datetime.datetime.now())
    }
