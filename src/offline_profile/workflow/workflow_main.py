from typing import Dict, List, Any,Optional
import datetime
import time
import threading

from langgraph.graph import StateGraph

from src.workflow.offline_profile.storage import cleanup_old_user_profiles
from src.workflow.offline_profile.config1 import logger, DATA_PROCESSING_CONFIG, PERFORMANCE_CONFIG, LOG_CONFIG
from src.workflow.offline_profile.workflow.workflow_graph import create_offline_profile_workflow
from src.workflow.offline_profile.workflow.workflow_state import OfflineProfileState
from src.tools.porstgreDB_tools import get_chatuser_list






# 任务执行状态管理（参照 scorer.py 的实现方式）
_label_task_running = False
_label_task_lock = threading.Lock()
_LABEL_JOB_ID = "hourly_label_task"
_label_cache_lock = threading.Lock()
_label_cache: Dict[str, Dict[str, Any]] = {}
_LABEL_CACHE_TTL = 10 * 60  # 10 分钟缓存


def get_valid_labels(app_id: str) -> List[Dict[str, Any]]:
    """获取指定应用的有效系统标签列表，带缓存机制"""
    # 强制使用 "aiwa-admin" 作为 app_id 获取标签定义
    admin_app_id = "aiwa-admin"
    
    now = time.time()
    with _label_cache_lock:
        cached = _label_cache.get(admin_app_id)
        if cached and cached.get("expires_at", 0) > now:
            return cached.get("labels", [])

    try:

        flattened = get_tags_list()

            
    except Exception as exc:
        logger.error("获取系统标签失败: %s", exc, exc_info=True)
        return []


    logger.info(f"标签数量: {len(flattened)}")

    with _label_cache_lock:
        _label_cache[admin_app_id] = {"labels": flattened, "expires_at": now + _LABEL_CACHE_TTL}

    return flattened



# 编译工作流
_workflow = create_offline_profile_workflow()
compiled_workflow = _workflow.compile()




def run_offline_profile_workflow(state: OfflineProfileState) -> Dict[str, Any]:
    """执行离线用户画像工作流
    
    Args:
        state: 初始状态
        
    Returns:
        执行结果
    """
    start_time = time.time()
    user_id = state.get('user_id', '0')

    
    logger.info(f"[工作流] 开始执行离线用户画像工作流，用户: {user_id}")
    
    try:

        
        # 执行工作流
        result = compiled_workflow.invoke(state)
        
        # 提取执行结果
        execution_result = {
            'user_id': result.get('user_id'),
            'status': result.get('status', 'error'),
            'error_message': result.get('error_message', ''),
            'tag_count': result.get('tag_count', 0),
            'updated_at': result.get('updated_at', '')
        }
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录性能日志
        if LOG_CONFIG.get('enable_performance_logs', True):
            logger.info(f"[工作流] 执行完成，用户: {user_id}, 状态: {execution_result['status']}, "
                        f"标签数量: {execution_result['tag_count']}, 执行时间: {execution_time:.2f}秒")
        else:
            logger.info(f"[工作流] 执行完成，结果: {execution_result}")
        
        
        return execution_result
    
    except Exception as e:
        error_msg = f"工作流执行失败: {str(e)}"
        execution_time = time.time() - start_time
        
        if LOG_CONFIG.get('enable_performance_logs', True):
            logger.error(f"[工作流] 执行失败，用户: {user_id}, 错误: {error_msg}, 执行时间: {execution_time:.2f}秒")
        else:
            logger.error(f"[工作流] {error_msg}")
        
        return {
            'user_id': user_id,
            'status': 'error',
            'error_message': error_msg,
            'tag_count': 0,
            'updated_at': ''
        }



def process_single_user(user_id: str, start_time: datetime.datetime, end_time: datetime.datetime, standard_tags: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """处理单个用户
    
    Args:
        user_id: 用户 ID
        start_time: 开始时间
        end_time: 结束时间
        standard_tags: 标准标签库（可选）
        
    Returns:
        处理结果
    """

    # 创建初始状态
    initial_state: OfflineProfileState = {
        'user_id': user_id,
        'start_time': start_time,
        'end_time': end_time,
        'standard_tags': standard_tags,
        'status': 'pending'
    }
    
    # 执行工作流
    return run_offline_profile_workflow(initial_state)


def process_batch(start_time: datetime.datetime, end_time: datetime.datetime,app_id: str = 'aiwa-tenant-server_2') -> Dict[str, Any]:
    """批处理多个用户
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        批处理结果
    """
    logger.info(f"[批处理] 开始执行批处理，时间范围: {start_time} 到 {end_time}")
    
    results = []
    success_count = 0
    error_count = 0

    # 获取标准标签库（只获取一次，避免重复调用）
    try:
        from src.tools.porstgreDB_tools import get_tags_list
        
        standard_tags = get_tags_list(app_id)

        logger.info(f"[批处理] 成功获取标准标签库，标签数量: {len(standard_tags)}")
    except Exception as e:
        logger.error(f"[批处理] 获取标准标签库失败: {e}")
        standard_tags = None

    # users_to_process = get_chatuser_list(start_time, end_time)

    users_to_process = [
        {'id':166}
    ]

    
    
    for user_id in users_to_process:
        result = process_single_user(user_id['id'], start_time, end_time, standard_tags=standard_tags)
        results.append(result)
        
        if result['status'] == 'success':
            success_count += 1
        else:
            error_count += 1
    
    # 清理旧的用户画像
    cleanup_count = cleanup_old_user_profiles(days=DATA_PROCESSING_CONFIG['max_chat_history_days'])
    
    batch_result = {
        'total_users': len(users_to_process),
        'success_count': success_count,
        'error_count': error_count,
        'cleanup_count': cleanup_count,
        'results': results
    }
    
    logger.info(f"[批处理] 批处理完成: {batch_result}")
    return batch_result


def get_workflow() -> StateGraph:
    """获取编译后的工作流
    
    Returns:
        编译后的工作流
    """
    return compiled_workflow
