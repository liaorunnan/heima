"""
长期记忆和Store（仓库）
测试简单的 InMemoryStore 存储记忆，未接入图

"""

from langgraph.store.memory import InMemoryStore
import uuid


in_memory_store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")



memory_id = str(uuid.uuid4())
memory = {"user" : "你好，我叫木羽"}
in_memory_store.put(namespace_for_memory, memory_id,  memory)

memories = in_memory_store.search(namespace_for_memory)

print(memories[-1].dict())







