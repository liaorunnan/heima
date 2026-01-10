
"""
SqliteSaver 实现记忆功能，共有两种存储方式：
    一种是类似于`MemorySaver`将`checkpointer`存储在内存中
    另外一种是存储在`sqlite`数据库中

此脚本使用第二种存储方式，将`checkpointer`存储在 **`sqlite`数据库** 中，仅是测试存储功能，没有接入图

"""
import sqlite3
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

with SqliteSaver.from_conn_string("checkpoints20241101.sqlite") as memory:
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
        
print("------------------------------------------------")
print(saved_config)

print("===============================================================================")
print("===============================================================================")


# 建立数据库连接
conn = sqlite3.connect("checkpoints20241101.sqlite")

# 创建一个游标对象来执行你的SQL查询
cursor = conn.cursor()

# 查询数据库中所有表的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# 获取查询结果
tables = cursor.fetchall()

# 打印所有表名
for table in tables:
    print(table)

print("*******************************************************************************")
print("*******************************************************************************")

# 从检查点表中检索所有数据
cursor.execute(f"SELECT * FROM checkpoints;")
all_data = cursor.fetchall()

# 打印检查点表中的所有数据
print("Data in the 'checkpoints' table:")
for row in all_data:
    print(row)






