
"""
SqliteSaver 实现记忆功能，共有两种存储方式：
    一种是类似于`MemorySaver`将`checkpointer`存储在内存中
    另外一种是存储在`sqlite`数据库中

此脚本使用第一种存储方式，将`checkpointer`存储在 **内存** 中，仅是测试存储功能，没有接入图

"""

from langgraph.checkpoint.sqlite import SqliteSaver

# 创建一个内存中的检查点
# memory = SqliteSaver.from_conn_string(":memory:")

checkpoint_data = {
    "thread_id": "muyu123",  
    "thread_ts": "2024-10-30T07:23:38.656547+00:00", 
    "checkpoint": {
        "id": "1ef968fe-1eb4-6049-bfff", 
    },
    "metadata": {"timestamp": "2024-10-30T07:23:38.656547+00:00"}
}

with SqliteSaver.from_conn_string(":memory:") as memory:
    # 保存检查点，包括时间戳
    saved_config = memory.put(
        config={"configurable": {"thread_id": checkpoint_data["thread_id"], "thread_ts": checkpoint_data["thread_ts"], "checkpoint_ns": ""}},
        checkpoint=checkpoint_data["checkpoint"],
        metadata=checkpoint_data["metadata"],
        new_versions= {"writes": {"key": "value"}}
    )

    # 检索检查点的数据
    config = {"configurable": {"thread_id": checkpoint_data["thread_id"]}}

    # 获取给定 thread_id 的所有检查点
    checkpoints = list(memory.list(config))
    for checkpoint in checkpoints:
        print(checkpoint)
        print(len(checkpoints))
print("------------------------------------------------")
print(saved_config)









