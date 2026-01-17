#!/usr/bin/env python3
# 测试核心记忆功能
from workflow_db_yiwang_weight import CoreMemory
from rag.embedding import get_embedding
import datetime

# 初始化CoreMemory实例
core_memory_db = CoreMemory()

# 测试数据
thread_id = "test_thread_3"
content = "用户名叫张三，测试核心记忆功能"
vector = get_embedding(content).tolist()

print(f"测试开始：thread_id={thread_id}, content={content}")

# 测试更新核心记忆
print("\n1. 测试更新核心记忆...")
core_memory_db.update_by_thread_id(thread_id, content, vector)
print("更新成功")

# 测试查询核心记忆
print(f"\n2. 测试查询核心记忆 (thread_id={thread_id})...")
results = core_memory_db.query_by_thread_id(thread_id)

if results:
    print(f"查询成功！共找到 {len(results)} 条记录")
    for result in results:
        print(f"  ID: {result.id}")
        print(f"  Thread ID: {result.thread_id}")
        print(f"  Content: {result.content}")
        print(f"  Last Update: {result.last_update}")
else:
    print("查询失败！未找到核心记忆")

# 测试另一个thread_id
print(f"\n3. 测试查询不存在的thread_id...")
empty_results = core_memory_db.query_by_thread_id("non_existent_thread")
print(f"查询结果：{empty_results}")

print("\n测试完成！")
