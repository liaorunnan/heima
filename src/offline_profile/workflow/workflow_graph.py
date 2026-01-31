from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.workflow.offline_profile.workflow.workflow_state import OfflineProfileState
from src.workflow.offline_profile.workflow.workflow_nodes import (
    process_chat_data_node,
    generate_tags_node,
    fuse_profile_node,
    save_profile_node
)



def create_offline_profile_workflow() -> StateGraph:
    """创建离线用户画像工作流图
    
    Returns:
        工作流图
    """
    # 创建状态图
    workflow = StateGraph(OfflineProfileState)
    
    # 添加节点
    workflow.add_node("process_chat_data", process_chat_data_node)
    workflow.add_node("generate_tags", generate_tags_node)
    workflow.add_node("fuse_profile", fuse_profile_node)
    workflow.add_node("save_profile", save_profile_node)
    
    # 定义边
    workflow.set_entry_point("process_chat_data")
    
    # 数据处理节点 -> 标签生成节点
    workflow.add_edge("process_chat_data", "generate_tags")
    
    # 标签生成节点 -> 画像融合节点
    workflow.add_edge("generate_tags", "fuse_profile")
    
    # 画像融合节点 -> 存储节点
    workflow.add_edge("fuse_profile", "save_profile")
    
    # 存储节点 -> 结束
    workflow.add_edge("save_profile", END)
    
    return workflow
