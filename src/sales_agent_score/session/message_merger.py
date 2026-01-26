from typing import List
from datetime import datetime, timedelta
from ..models.data_models import Message, MessageGroup
from ..utils.score_utils import get_time_difference_in_seconds


class MessageMerger:
    """实现消息归并逻辑，将60秒内的连续消息合并为消息组"""
    
    # 消息归并阈值：60秒内连续发送的消息视为一个消息组
    MESSAGE_MERGE_THRESHOLD_SECONDS = 60
    
    def merge_messages(self, messages: List[Message]) -> List[MessageGroup]:
        """将连续消息合并为消息组
        
        Args:
            messages: 原始消息列表
            
        Returns:
            合并后的消息组列表
        """
        if not messages:
            return []
        
        # 按照时间排序消息
        sorted_messages = sorted(messages, key=lambda msg: msg["timestamp"])
        
        message_groups = []
        current_group = [sorted_messages[0]]
        
        for message in sorted_messages[1:]:
            # 获取当前消息与上一条消息的时间差
            time_diff_seconds = get_time_difference_in_seconds(
                current_group[-1]["timestamp"],
                message["timestamp"]
            )
            
            if time_diff_seconds <= self.MESSAGE_MERGE_THRESHOLD_SECONDS:
                # 60秒内的消息，合并到当前消息组
                current_group.append(message)
            else:
                # 超过60秒，创建新的消息组
                message_groups.append(self._create_message_group(current_group))
                current_group = [message]
        
        # 添加最后一个消息组
        if current_group:
            message_groups.append(self._create_message_group(current_group))
        
        return message_groups
    
    def _create_message_group(self, messages: List[Message]) -> MessageGroup:
        """创建消息组
        
        Args:
            messages: 消息组包含的消息列表
            
        Returns:
            消息组对象
        """
        return {
            "messages": messages,
            "timestamp": messages[0]["timestamp"],
            "count": len(messages)
        }
    
    def add_message_to_group(self, message: Message, message_groups: List[MessageGroup]) -> List[MessageGroup]:
        """将单条消息添加到合适的消息组
        
        Args:
            message: 要添加的消息
            message_groups: 现有的消息组列表
            
        Returns:
            更新后的消息组列表
        """
        if not message_groups:
            # 没有现有消息组，创建新的
            return [self._create_message_group([message])]
        
        # 检查是否可以添加到最后一个消息组
        last_group = message_groups[-1]
        time_diff_seconds = get_time_difference_in_seconds(
            last_group["timestamp"],
            message["timestamp"]
        )
        
        if time_diff_seconds <= self.MESSAGE_MERGE_THRESHOLD_SECONDS:
            # 可以添加到最后一个消息组
            updated_group = {
                "messages": last_group["messages"] + [message],
                "timestamp": last_group["timestamp"],
                "count": last_group["count"] + 1
            }
            return message_groups[:-1] + [updated_group]
        else:
            # 创建新的消息组
            return message_groups + [self._create_message_group([message])]
