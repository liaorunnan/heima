import datetime
import argparse
from typing import Dict, List, Any

from src.workflow.offline_profile.config1 import logger, DATA_PROCESSING_CONFIG
from src.workflow.offline_profile.workflow.workflow_main import process_batch as workflow_process_batch
from src.workflow.offline_profile.utils import get_yesterday


def main(target_date: str = None):
    """
    离线用户画像批处理入口函数
    
    Args:
        target_date (str, optional): 指定处理日期，格式为 'YYYY-MM-DD'。
                                     如果不传或为 None，默认处理昨天的数据。
    """
    # 1. 确定处理日期
    if target_date:
        try:
            # 尝试解析传入的日期字符串
            process_date = datetime.datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f'传入的日期格式错误: {target_date}，将回退使用昨天的日期')
            process_date = get_yesterday()
    else:
        # 如果没有传入参数，默认获取昨天
        process_date = get_yesterday()
    
    # 2. 计算时间范围 (当天的 00:00:00 到 23:59:59)
    start_time = process_date.replace(hour=0, minute=0, second=0)
    end_time = process_date.replace(hour=23, minute=59, second=59)
    
    logger.info(f"=== 开始执行离线批处理 ===")
    logger.info(f"处理日期: {process_date.date()}")
    logger.info(f"时间范围: {start_time} 到 {end_time}")


    
    # 3. 仅执行批处理逻辑
    try:
        result = workflow_process_batch(start_time, end_time)
        logger.info(f"批处理执行完毕，结果: {result}")
        return result
    except Exception as e:
        logger.error(f"批处理执行过程中发生异常: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    main()

