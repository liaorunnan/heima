from typing import Dict, Any,List
import datetime


from src.workflow.offline_profile.data_etl import process_user_chat_data
from src.workflow.offline_profile.tagging import analyze_user_intent
from src.workflow.offline_profile.profile_fusion import fuse_profiles, create_empty_profile
from src.workflow.offline_profile.storage import save_user_profile, load_user_profile
from src.workflow.offline_profile.config1 import logger
from src.workflow.offline_profile.workflow.workflow_state import OfflineProfileState


def process_chat_data_node(state: OfflineProfileState) -> Dict[str, Any]:
    """数据处理节点
    
    Args:
        state: 工作流状态
        
    Returns:
        更新后的状态
    """
    logger.info(f"[数据处理节点] 开始处理用户: {state['user_id']}")
    
    try:
        # 获取状态中的基本信息
        user_id = state['user_id']
     
        start_time = state['start_time']
        end_time = state['end_time']
        
        # 处理聊天数据
        chat_data = process_user_chat_data(user_id, 0, start_time, end_time)


        logger.info(f"[数据处理节点] 用户 {user_id} 的聊天数据: {chat_data}")

        
        
        # 获取对话摘要
        summaries = chat_data.get('summaries', [])
        
        # 更新状态
        updated_state = {
            'chat_data': chat_data,
            'summaries': summaries,
            'status': 'processing'
        }
        
        # 检查是否有摘要
        if not summaries:
            updated_state['status'] = 'no_summaries'
            updated_state['error_message'] = '没有有效的聊天摘要'
            logger.info(f"[数据处理节点] 用户 {user_id} 没有有效的聊天摘要")
        else:
            logger.info(f"[数据处理节点] 成功处理用户 {user_id} 的聊天数据，生成 {len(summaries)} 个摘要")
        
        return updated_state
    
    except Exception as e:
        error_msg = f"数据处理失败: {str(e)}"
        logger.error(f"[数据处理节点] {error_msg}")
        return {
            'status': 'error',
            'error_message': error_msg
        }


def generate_tags_node(state: OfflineProfileState) -> Dict[str, Any]:
    """标签生成节点
    
    Args:
        state: 工作流状态
        
    Returns:
        更新后的状态
    """
    # 检查是否有摘要
    if state.get('status') == 'no_summaries':
        logger.info("[标签生成节点] 跳过，因为没有有效的聊天摘要")
        return {}
    # 获取标准标签库
    standard_tags = state.get('standard_tags')
    user_id = state.get('user_id')
    logger.info(f"[标签生成节点] 开始为用户{user_id} 生成标签")
    
    try:
        # 获取对话摘要
        summaries = state.get('summaries', [])

        # 分析用户意图，提取标签
        extracted_data = analyze_user_intent(summaries, standard_tags=standard_tags, user_id=user_id)
        
        # 更新状态
        updated_state = {
            'extracted_data': extracted_data,
            'status': 'processing'
        }
        
        # 检查是否有标签数据
        # extracted_data 包含 global_traits, intents, instructions 等 keys
        has_data = any(v for k, v in extracted_data.items() if v)
        
        if not has_data:
            updated_state['status'] = 'no_tags'
            updated_state['error_message'] = '没有提取到有效标签数据'
            logger.info(f"[标签生成节点] 用户 {state['user_id']} 没有提取到有效标签数据")
        else:
            logger.info(f"[标签生成节点] 成功为用户 {state['user_id']} 提取标签数据")
        
        return updated_state
    
    except Exception as e:
        error_msg = f"标签生成失败: {str(e)}"
        logger.error(f"[标签生成节点] {error_msg}")
        return {
            'status': 'error',
            'error_message': error_msg
        }



def fuse_profile_node(state: OfflineProfileState) -> Dict[str, Any]:
    """画像融合节点
    
    Args:
        state: 工作流状态
        
    Returns:
        更新后的状态
    """
    # 检查是否有标签
    if state.get('status') in ['no_summaries', 'no_tags']:
        logger.info("[画像融合节点] 跳过，因为没有有效的标签")
        return {}
    
    logger.info(f"[画像融合节点] 开始为用户 {state['user_id']} 融合画像")
    
    try:
        # 获取用户 ID 和提取的标签
        user_id = state['user_id']
        extracted_data = state.get('extracted_data', {})

        
        # 加载现有画像
        existing_profile = load_user_profile(user_id)
        if not existing_profile:
            existing_profile = create_empty_profile(user_id)
            logger.info(f"[画像融合节点] 为用户 {user_id} 创建空的用户画像")
        else:
            logger.info(f"[画像融合节点] 加载用户 {user_id} 的现有画像")
        
        # 融合画像
        new_profile = fuse_profiles(existing_profile, extracted_data)
        
        # 计算简单的统计信息
        global_count = len(new_profile.get('global_traits', {}))
        intent_count = sum(len(tags) for tags in new_profile.get('category_intents', {}).values())
        total_count = global_count + intent_count

        # 更新状态
        updated_state = {
            'existing_profile': existing_profile,
            'new_profile': new_profile,
            'tag_count': total_count,
            'updated_at': new_profile.get('updated_at'),
            'status': 'processing'
        }
        
        logger.info(f"[画像融合节点] 成功为用户 {user_id} 融合画像，生成 {total_count} 个标签点")
        return updated_state
    
    except Exception as e:
        error_msg = f"画像融合失败: {str(e)}"
        logger.error(f"[画像融合节点] {error_msg}")
        return {
            'status': 'error',
            'error_message': error_msg
        }


def save_profile_node(state: OfflineProfileState) -> Dict[str, Any]:
    """存储节点
    
    Args:
        state: 工作流状态
        
    Returns:
        更新后的状态
    """
    # 检查是否有新画像
    if state.get('status') in ['no_summaries', 'no_tags', 'error']:
        logger.info("[存储节点] 跳过，因为处理未完成")
        return {}
    
    logger.info(f"[存储节点] 开始保存用户 {state['user_id']} 的画像")
    
    try:
        # 获取新画像
        new_profile = state.get('new_profile')
        
        # 保存画像
        save_result = save_user_profile(new_profile)
        
        # 更新状态
        if save_result:
            updated_state = {
                'status': 'success',
                'error_message': ''
            }
            logger.info(f"[存储节点] 成功保存用户 {state['user_id']} 的画像")
        else:
            updated_state = {
                'status': 'error',
                'error_message': '画像保存失败'
            }
            logger.error(f"[存储节点] 用户 {state['user_id']} 的画像保存失败")
        
        return updated_state
    
    except Exception as e:
        error_msg = f"画像保存失败: {str(e)}"
        logger.error(f"[存储节点] {error_msg}")
        return {
            'status': 'error',
            'error_message': error_msg
        }

if __name__ == '__main__':
    generate_tags_node()
