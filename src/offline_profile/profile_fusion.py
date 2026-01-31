import datetime
from typing import Dict, List, Any, Optional

from src.workflow.offline_profile.config1 import DECAY_CONFIG, STORAGE_CONFIG, logger
from src.workflow.offline_profile.utils import format_datetime


class ProfileFusionEngine:
    """画像融合引擎"""
    
    def __init__(self):
        self.decay_config = DECAY_CONFIG
        self.storage_config = STORAGE_CONFIG
    
    def apply_time_decay(self, tags: Dict[str, Any]) -> Dict[str, Any]:
        """应用时间衰减
        
        Args:
            tags: 旧标签字典，格式为 {tag_id: {score: float, last_update: str, ...}}
            
        Returns:
            应用衰减后的标签字典
        """
        today = datetime.datetime.now()
        decayed_tags = {}
        
        for tag_id, tag_info in tags.items():
            # 计算距离上次更新的天数
            last_update_str = tag_info.get('last_update', format_datetime(today))
            last_update = datetime.datetime.strptime(last_update_str, '%Y-%m-%d %H:%M:%S')
            days_diff = (today - last_update).days
            
            # 应用衰减
            decay_factor = self.decay_config['default']
            if days_diff > 0:
                # 根据标签类型选择不同的衰减系数
                # 这里简化处理，实际应用中可以根据标签类型进行区分
                decay_factor = decay_factor ** days_diff
                
            new_score = tag_info['score'] * decay_factor
            
            # 如果得分低于阈值，不保留该标签
            if new_score >= self.decay_config['min_score']:
                decayed_tags[tag_id] = {
                    'score': new_score,
                    'last_update': tag_info['last_update'],
                    'name': tag_info.get('name', ''),
                    'description': tag_info.get('description', '')
                }
        
        return decayed_tags
    
    def resolve_conflicts(self, tags: Dict[str, Any], new_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解决标签冲突
        
        Args:
            tags: 现有标签字典
            new_tags: 新提取的标签列表
            
        Returns:
            解决冲突后的标签字典
        """
        # 首先处理新标签中的拒绝标签 (L4)
        rejected_tags = set()
        for tag in new_tags:
            if tag.get('intent_level') == 'L4':
                rejected_tags.add(str(tag['id']))
        
        # 移除被拒绝的标签
        filtered_tags = {tag_id: tag_info for tag_id, tag_info in tags.items() if tag_id not in rejected_tags}
        
        # 处理新标签中的其他标签
        for tag in new_tags:
            if tag.get('intent_level') != 'L4':
                tag_id = str(tag['id'])
                # 如果标签已存在，更新权重
                if tag_id in filtered_tags:
                    # 计算新权重
                    current_score = filtered_tags[tag_id]['score']
                    new_score = min(current_score + tag['score'], self.decay_config['max_score'])
                    filtered_tags[tag_id] = {
                        'score': new_score,
                        'last_update': format_datetime(datetime.datetime.now()),
                        'name': tag.get('name', filtered_tags[tag_id].get('name', '')),
                        'description': tag.get('description', filtered_tags[tag_id].get('description', ''))
                    }
                else:
                    # 新增标签
                    filtered_tags[tag_id] = {
                        'score': tag['score'],
                        'last_update': format_datetime(datetime.datetime.now()),
                        'name': tag.get('name', ''),
                        'description': tag.get('description', '')
                    }
        
        return filtered_tags
    
    def merge_tags(self, existing_tags: Dict[str, Any], new_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并标签
        
        Args:
            existing_tags: 现有标签字典
            new_tags: 新提取的标签列表
            
        Returns:
            合并后的标签字典
        """
        # 首先应用时间衰减
        decayed_tags = self.apply_time_decay(existing_tags)
        
        # 然后解决冲突并合并新标签
        merged_tags = self.resolve_conflicts(decayed_tags, new_tags)
        
        # 按权重排序，保留前 N 个标签
        sorted_tags = sorted(merged_tags.items(), key=lambda x: x[1]['score'], reverse=True)
        top_tags = dict(sorted_tags[:self.storage_config['max_tags']])

        print(top_tags)
        
        logger.info(f"合并后标签数量: {len(top_tags)}")
        return top_tags
    
    def calculate_tag_weight(self, tag: Dict[str, Any], days_since_last_update: int = 0) -> float:
        """计算标签权重
        
        Args:
            tag: 标签字典
            days_since_last_update: 距离上次更新的天数
            
        Returns:
            计算后的权重
        """
        # 基础得分
        base_score = tag.get('score', 0.1)
        
        # 应用时间衰减
        decay_factor = self.decay_config['default'] ** days_since_last_update
        
        # 计算最终得分
        final_score = base_score * decay_factor
        
        # 确保得分在合理范围内
        final_score = max(final_score, 0.0)
        final_score = min(final_score, self.decay_config['max_score'])
        
        return final_score
    
    def generate_user_profile(self, existing_profile: Dict[str, Any], new_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成用户画像
        
        Args:
            existing_profile: 现有用户画像
            new_tags: 新提取的标签列表
            
        Returns:
            新的用户画像
        """
        # 获取现有标签
        existing_tags = existing_profile.get('tags', {})
        
        # 合并标签
        merged_tags = self.merge_tags(existing_tags, new_tags)
        
        # 生成新的用户画像
        new_profile = {
            'user_id': existing_profile.get('user_id', ''),
            'tags': merged_tags,
            'updated_at': format_datetime(datetime.datetime.now()),
            'tag_count': len(merged_tags)
        }
        
        logger.info(f"生成用户画像完成，标签数量: {len(merged_tags)}")
        return new_profile


def fuse_profiles(existing_profile: Dict[str, Any], new_tags: List[Dict[str, Any]]) -> Dict[str, Any]:
    """融合用户画像
    
    Args:
        existing_profile: 现有用户画像
        new_tags: 新提取的标签列表
        
    Returns:
        融合后的用户画像
    """
    engine = ProfileFusionEngine()
    return engine.generate_user_profile(existing_profile, new_tags)


def create_empty_profile(user_id: str) -> Dict[str, Any]:
    """创建空的用户画像
    
    Args:
        user_id: 用户 ID
        
    Returns:
        空的用户画像
    """
    return {
        'user_id': user_id,
        'tags': {},
        'updated_at': format_datetime(datetime.datetime.now()),
        'tag_count': 0
    }


def get_top_tags(profile: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """获取用户画像中的 top 标签
    
    Args:
        profile: 用户画像
        limit: 返回的标签数量限制
        
    Returns:
        排序后的标签列表
    """
    tags = profile.get('tags', {})
    sorted_tags = sorted(tags.items(), key=lambda x: x[1]['score'], reverse=True)
    
    top_tags = []
    for tag_id, tag_info in sorted_tags[:limit]:
        top_tags.append({
            'id': tag_id,
            'name': tag_info.get('name', ''),
            'score': tag_info.get('score', 0.0),
            'last_update': tag_info.get('last_update', '')
        })
    
    return top_tags
