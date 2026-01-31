import os
import json
from typing import Dict, List, Any, Optional

from src.workflow.offline_profile.config1 import STORAGE_CONFIG, logger
from src.workflow.offline_profile.utils import load_json, save_json


class Storage:
    """存储类"""
    
    def __init__(self):
        self.profile_dir = STORAGE_CONFIG['profile_dir']
        # 确保存储目录存在
        os.makedirs(self.profile_dir, exist_ok=True)
    
    def get_profile_path(self, user_id: str) -> str:
        """获取用户画像文件路径
        
        Args:
            user_id: 用户 ID
            
        Returns:
            文件路径
        """
        return os.path.join(self.profile_dir, f"{user_id}.json")
    
    def save_profile(self, profile: Dict[str, Any]) -> bool:
        """保存用户画像
        
        Args:
            profile: 用户画像
            
        Returns:
            是否保存成功
        """
        try:
            user_id = profile.get('user_id')
            if not user_id:
                logger.error("用户画像缺少 user_id")
                return False
            
            file_path = self.get_profile_path(user_id)
            save_json(profile, file_path)
            logger.info(f"保存用户画像成功: {user_id}")
            return True
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")
            return False
    
    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """加载用户画像
        
        Args:
            user_id: 用户 ID
            
        Returns:
            用户画像，如果不存在返回 None
        """
        try:
            file_path = self.get_profile_path(user_id)
            if os.path.exists(file_path):
                profile = load_json(file_path)
                logger.info(f"加载用户画像成功: {user_id}")
                return profile
            else:
                logger.info(f"用户画像不存在: {user_id}")
                return None
        except Exception as e:
            logger.error(f"加载用户画像失败: {e}")
            return None
    
    def delete_profile(self, user_id: str) -> bool:
        """删除用户画像
        
        Args:
            user_id: 用户 ID
            
        Returns:
            是否删除成功
        """
        try:
            file_path = self.get_profile_path(user_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"删除用户画像成功: {user_id}")
                return True
            else:
                logger.info(f"用户画像不存在: {user_id}")
                return False
        except Exception as e:
            logger.error(f"删除用户画像失败: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """列出所有用户画像
        
        Returns:
            用户 ID 列表
        """
        try:
            profiles = []
            for file_name in os.listdir(self.profile_dir):
                if file_name.endswith('.json'):
                    user_id = file_name[:-5]  # 移除 .json 后缀
                    profiles.append(user_id)
            logger.info(f"获取用户画像列表成功，数量: {len(profiles)}")
            return profiles
        except Exception as e:
            logger.error(f"获取用户画像列表失败: {e}")
            return []
    
    def get_profile_count(self) -> int:
        """获取用户画像数量
        
        Returns:
            用户画像数量
        """
        return len(self.list_profiles())
    
    def cleanup_old_profiles(self, days: int = 30) -> int:
        """清理旧的用户画像
        
        Args:
            days: 保留天数
            
        Returns:
            清理的数量
        """
        try:
            import datetime
            
            cleanup_count = 0
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            for file_name in os.listdir(self.profile_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.profile_dir, file_name)
                    # 获取文件修改时间
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if mtime < cutoff_date:
                        os.remove(file_path)
                        cleanup_count += 1
            
            logger.info(f"清理旧用户画像成功，数量: {cleanup_count}")
            return cleanup_count
        except Exception as e:
            logger.error(f"清理旧用户画像失败: {e}")
            return 0


# 创建全局存储实例
storage = Storage()


def save_user_profile(profile: Dict[str, Any]) -> bool:
    """保存用户画像
    
    Args:
        profile: 用户画像
        
    Returns:
        是否保存成功
    """
    return storage.save_profile(profile)


def load_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """加载用户画像
    
    Args:
        user_id: 用户 ID
        
    Returns:
        用户画像，如果不存在返回 None
    """
    return storage.load_profile(user_id)


def delete_user_profile(user_id: str) -> bool:
    """删除用户画像
    
    Args:
        user_id: 用户 ID
        
    Returns:
        是否删除成功
    """
    return storage.delete_profile(user_id)


def list_user_profiles() -> List[str]:
    """列出所有用户画像
    
    Returns:
        用户 ID 列表
    """
    return storage.list_profiles()


def get_user_profile_count() -> int:
    """获取用户画像数量
    
    Returns:
        用户画像数量
    """
    return storage.get_profile_count()


def cleanup_old_user_profiles(days: int = 30) -> int:
    """清理旧的用户画像
    
    Args:
        days: 保留天数
        
    Returns:
        清理的数量
    """
    return storage.cleanup_old_profiles(days)
