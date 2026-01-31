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

        # summaries = [{'session_id': 'session_0', 'start_time': '2026-01-29 10:21:53', 'end_time': '2026-01-29 10:22:34', 'summary': '用户想了解适用于小型团队的基础版产品的价格。'}, {'session_id': 'session_1', 'start_time': '2026-01-29 11:16:31', 'end_time': '2026-01-29 11:18:09', 'summary': '用户询问了基础版的价格及包含的服务内容，得知基础版价格为每月199元，包括5个 用户许可。'}]
        
        

        # standard_tags = [{'id': 10, 'name': 'lllllllllllllllllll', 'description': 'lllllllllllllllllll', 'parent': 'lllllllllllllllllll', 'alias': 'None'}, {'id': 9, 'name': '标签名标签名标签名标签 名标签名标签名标签名标签名标签名标签名标签名标签名', 'description': '标签名标签名标签名标签名标签名', 'parent': '标签名标签名标签名标签名标签名', 'alias': 'None'}, {'id': 8, 'name': '测试', 'description': '123456789', 'parent': 'test', 'alias': 'None'}, {'id': 7, 'name': 'test', 'description': '测试测试测试', 'parent': '测试', 'alias': 'None'}, {'id': 5, 'name': '标签3', 'description': '3', 'parent': '3', 'alias': 'None'}, {'id': 4, 'name': '标签2', 'description': '2', 'parent': '2', 'alias': 'None'}, {'id': 1, 'name': '标签1', 'description': '1', 'parent': '1', 'alias': 'None'},{'id': 11, 'name': '性别 - 男', 'description': '用户自述为男，或提及“我女朋友/老婆/兄弟”等具有明显男性画像的语义。', 'parent': None, 'alias': '男'}]

        
        # 分析用户意图，提取标签
        extracted_tags = analyze_user_intent(summaries, standard_tags=standard_tags,user_id=user_id)
        
        # 更新状态
        updated_state = {
            'extracted_tags': extracted_tags,
            'status': 'processing'
        }
        
        # 检查是否有标签
        if not extracted_tags:
            updated_state['status'] = 'no_tags'
            updated_state['error_message'] = '没有提取到有效标签'
            logger.info(f"[标签生成节点] 用户 {state['user_id']} 没有提取到有效标签")
        else:
            logger.info(f"[标签生成节点] 成功为用户 {state['user_id']} 提取 {len(extracted_tags)} 个标签")
        
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
        extracted_tags = state.get('extracted_tags', [])

        
        # 加载现有画像
        existing_profile = load_user_profile(user_id)
        if not existing_profile:
            existing_profile = create_empty_profile(user_id)
            logger.info(f"[画像融合节点] 为用户 {user_id} 创建空的用户画像")
        else:
            logger.info(f"[画像融合节点] 加载用户 {user_id} 的现有画像")
        
        # 融合画像
        new_profile = fuse_profiles(existing_profile, extracted_tags)
        
        # 更新状态
        updated_state = {
            'existing_profile': existing_profile,
            'new_profile': new_profile,
            'tag_count': len(new_profile.get('tags', {})),
            'updated_at': new_profile.get('updated_at'),
            'status': 'processing'
        }
        
        logger.info(f"[画像融合节点] 成功为用户 {user_id} 融合画像，生成 {len(new_profile.get('tags', {}))} 个标签")
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