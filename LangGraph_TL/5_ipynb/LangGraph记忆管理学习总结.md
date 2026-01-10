# LangGraph 记忆管理学习总结

本文档总结了 10 个脚本中关于 LangGraph 记忆管理的核心知识点，涵盖短期记忆、长期记忆、不同存储方式以及同步/异步实现。

---

## 📚 目录

1. [脚本概览](#1-脚本概览)
2. [核心概念](#2-核心概念)
3. [短期记忆实现](#3-短期记忆实现)
4. [长期记忆实现](#4-长期记忆实现)
5. [存储方式对比](#5-存储方式对比)
6. [上下文管理](#6-上下文管理)
7. [实战案例](#7-实战案例)
8. [最佳实践](#8-最佳实践)

---

## 1. 脚本概览

| 脚本 | 行数 | 核心功能 | 记忆类型 | 存储位置 | 同步/异步 |
|------|------|----------|----------|----------|-----------|
| **test_36.py** | 94 | MemorySaver 基础使用 | 短期记忆 | 内存 | 同步 |
| **test_37.py** | 53 | SqliteSaver 测试（未接图） | 短期记忆 | 内存 | 同步 |
| **test_38.py** | 82 | SqliteSaver 测试（未接图） | 短期记忆 | SQLite | 同步 |
| **test_39.py** | 87 | SqliteSaver + with 语句 | 短期记忆 | 内存 | 同步 |
| **test_40.py** | 89 | SqliteSaver + ExitStack | 短期记忆 | 内存 | 同步 |
| **test_41.py** | 123 | AsyncSqliteSaver + AsyncExitStack | 短期记忆 | 内存 | 异步 |
| **test_42.py** | 93 | SqliteSaver + ExitStack | 短期记忆 | SQLite | 同步 |
| **test_43.py** | 123 | AsyncSqliteSaver + AsyncExitStack | 短期记忆 | SQLite | 异步 |
| **test_44.py** | 32 | InMemoryStore 测试（未接图） | 长期记忆 | 内存 | 同步 |
| **test_45.py** | 151 | InMemoryStore 接入图 | 长期记忆 | 内存 | 异步 |

---

## 2. 核心概念

### 2.1 短期记忆 vs 长期记忆

| 特性 | 短期记忆 (Checkpointer) | 长期记忆 (Store) |
|------|------------------------|------------------|
| **用途** | 保存对话历史（会话上下文） | 保存用户信息、知识库 |
| **存储内容** | 消息列表（messages） | 结构化数据（任意 JSON） |
| **标识符** | thread_id | namespace + key |
| **实现方式** | MemorySaver, SqliteSaver | InMemoryStore, RedisStore |
| **生命周期** | 单次会话或跨会话 | 跨会话、持久化 |
| **典型场景** | 多轮对话 | 用户画像、偏好设置 |

### 2.2 关键标识符

#### thread_id（线程ID）
- **作用**：标识一次对话会话
- **特点**：相同 thread_id 可以访问同一个会话的历史消息
- **示例**：`{"configurable": {"thread_id": "1"}}`

#### user_id（用户ID）
- **作用**：标识用户身份
- **特点**：用于长期记忆中区分不同用户
- **示例**：`{"configurable": {"user_id": "6"}}`

#### namespace（命名空间）
- **作用**：组织和隔离数据
- **格式**：元组形式，如 `("memories", user_id)`
- **特点**：支持层级结构

**推荐命名规则**：
```python
# user_id：数字，例如 1、2、3
# thread_id：用户id + 数字，例如 1_10、1_11、2_10、2_11
config = {"configurable": {"thread_id": "6_10", "user_id": "6"}}
```

---

## 3. 短期记忆实现

### 3.1 MemorySaver - 最简单的记忆方式 (test_36.py)

**特点**：将检查点存储在内存中，程序结束后数据消失。

#### 核心代码：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# 创建记忆实例
memory = MemorySaver()

# 构建图
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# 编译时添加 checkpointer
graph = builder.compile(checkpointer=memory)

# 使用时指定 thread_id
config = {"configurable": {"thread_id": "1"}}
graph.stream({"messages": ["你好，我叫木羽"]}, config)
graph.stream({"messages": ["请问我叫什么？"]}, config)  # 能记住上一轮对话
```

#### 使用场景：
- ✅ 开发和测试阶段
- ✅ 单次运行的对话应用
- ❌ 不适合生产环境（无持久化）

---

### 3.2 SqliteSaver - 两种存储模式

SqliteSaver 提供了两种存储方式：

| 模式 | 连接字符串 | 持久化 | 适用场景 |
|------|-----------|--------|----------|
| **内存模式** | `:memory:` | ❌ | 测试、临时会话 |
| **数据库模式** | `"filename.sqlite"` | ✅ | 生产环境、持久化需求 |

#### 3.2.1 内存模式 - 未接入图 (test_37.py)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpoint_data = {
    "thread_id": "muyu123",  
    "thread_ts": "2024-10-30T07:23:38.656547+00:00", 
    "checkpoint": {"id": "1ef968fe-1eb4-6049-bfff"},
    "metadata": {"timestamp": "2024-10-30T07:23:38.656547+00:00"}
}

# 使用 with 语句管理上下文
with SqliteSaver.from_conn_string(":memory:") as memory:
    # 保存检查点
    saved_config = memory.put(
        config={"configurable": {"thread_id": checkpoint_data["thread_id"]}},
        checkpoint=checkpoint_data["checkpoint"],
        metadata=checkpoint_data["metadata"],
        new_versions={"writes": {"key": "value"}}
    )
    
    # 检索检查点
    config = {"configurable": {"thread_id": checkpoint_data["thread_id"]}}
    checkpoints = list(memory.list(config))
    for checkpoint in checkpoints:
        print(checkpoint)
```

#### 3.2.2 数据库模式 - 未接入图 (test_38.py)

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# 存储到 SQLite 文件
with SqliteSaver.from_conn_string("checkpoints20241101.sqlite") as memory:
    saved_config = memory.put(
        config={"configurable": {"thread_id": "muyu123"}},
        checkpoint=checkpoint_data["checkpoint"],
        metadata=checkpoint_data["metadata"],
        new_versions={"writes": {"key": "value"}}
    )

# 查看数据库表结构
conn = sqlite3.connect("checkpoints20241101.sqlite")
cursor = conn.cursor()

# 查询所有表名
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(tables)

# 查询检查点数据
cursor.execute("SELECT * FROM checkpoints;")
all_data = cursor.fetchall()
for row in all_data:
    print(row)
```

**数据库表结构**：
- `checkpoints` 表：存储检查点数据
- 字段包括：thread_id, thread_ts, checkpoint, metadata 等

---

### 3.3 with 语句的局限性 (test_39.py)

**问题**：使用 `with` 语句时，脱离上下文环境后记忆会丢失。

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# ❌ 问题：with 语句结束后，checkpointer 被关闭
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "1"}}
    graph.stream({"messages": ["你好，我叫木羽"]}, config)
    graph.stream({"messages": ["请问我叫什么？"]}, config)

# with 语句外，graph 无法再使用 checkpointer（已关闭）
```

**解决方案**：使用 `ExitStack` 或 `AsyncExitStack`

---

### 3.4 ExitStack - 同步版本 (test_40.py, test_42.py)

**特点**：使记忆不再局限于 `with` 语句块中。

#### 内存模式 (test_40.py)：

```python
from contextlib import ExitStack
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# 创建 ExitStack 实例
stack = ExitStack()

# 进入上下文
checkpointer = stack.enter_context(
    SqliteSaver.from_conn_string(":memory:")
)

# 创建图
graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

# 使用图（不受 with 限制）
config = {"configurable": {"thread_id": "1"}}
graph.stream({"messages": ["你好，我叫木羽"]}, config)
graph.stream({"messages": ["请问我叫什么？"]}, config)

# 手动关闭资源
stack.close()
```

#### 数据库模式 (test_42.py)：

```python
stack = ExitStack()

# 存储到 SQLite 文件
checkpointer = stack.enter_context(
    SqliteSaver.from_conn_string("checkpoints20241101.sqlite")
)

graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.stream({"messages": ["你好，我叫木羽"]}, config)

# 再次运行时，即使不执行上一轮对话，也能从数据库中获取记忆
graph.stream({"messages": ["请问我叫什么？"]}, config)

stack.close()
```

**关键优势**：
- ✅ 使用数据库模式时，程序重启后记忆依然存在
- ✅ 跨会话持久化

---

### 3.5 AsyncExitStack - 异步版本 (test_41.py, test_43.py)

**特点**：异步版本，支持高并发场景。

#### 内存模式 (test_41.py)：

```python
import asyncio
from contextlib import AsyncExitStack
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    stack = AsyncExitStack()
    
    try:
        # 异步进入上下文
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(":memory:")
        )
        
        # 创建图
        graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "24"}}
        
        # 异步流式输出
        async for chunk in graph.astream(
            {"messages": ["帮我查一下北京的天气"]}, 
            config, 
            stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()
        
        # 记忆测试
        async for chunk in graph.astream(
            {"messages": ["我刚才问了你什么问题"]}, 
            config, 
            stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()
        
    finally:
        await stack.aclose()  # 异步关闭资源

asyncio.run(main())
```

#### 数据库模式 (test_43.py)：

```python
async def main():
    stack = AsyncExitStack()
    
    try:
        # 存储到 SQLite 文件
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string("checkpoints20241101.sqlite")
        )
        
        graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        
        config = {"configurable": {"thread_id": "24"}}
        
        # 使用 astream_events 实现逐字输出
        async for event in graph.astream_events(
            {"messages": ["请你非常详细的介绍一下你自己"]}, 
            config, 
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
        
    finally:
        await stack.aclose()

asyncio.run(main())
```

**关键区别**：

| 特性 | ExitStack (同步) | AsyncExitStack (异步) |
|------|------------------|----------------------|
| 导入模块 | `SqliteSaver` | `AsyncSqliteSaver` |
| 进入上下文 | `stack.enter_context()` | `await stack.enter_async_context()` |
| 关闭资源 | `stack.close()` | `await stack.aclose()` |
| 适用场景 | 简单脚本 | 高并发、Web 应用 |

---

## 4. 长期记忆实现

### 4.1 InMemoryStore - 基础使用 (test_44.py)

**特点**：存储结构化数据，支持按 namespace 组织。

```python
from langgraph.store.memory import InMemoryStore
import uuid

# 创建存储实例
in_memory_store = InMemoryStore()

# 定义命名空间
user_id = "1"
namespace_for_memory = (user_id, "memories")

# 存储记忆
memory_id = str(uuid.uuid4())
memory = {"user": "你好，我叫木羽"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

# 检索记忆
memories = in_memory_store.search(namespace_for_memory)
print(memories[-1].dict())
```

**核心方法**：
- `put(namespace, key, value)`: 存储数据
- `search(namespace)`: 检索数据
- `get(namespace, key)`: 获取特定数据
- `delete(namespace, key)`: 删除数据

---

### 4.2 InMemoryStore 接入图 (test_45.py)

**特点**：将长期记忆集成到 LangGraph 工作流中，实现跨会话的用户记忆。

#### 核心实现：

```python
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# 创建存储实例
in_memory_store = InMemoryStore()
memory = MemorySaver()  # 短期记忆（对话历史）

# 定义节点，访问长期记忆
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 获取用户ID
    user_id = config["configurable"]["user_id"]
    
    # 定义命名空间
    namespace = ("memories", user_id)
    
    # 检索用户的长期记忆
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])
    
    # 存储新记忆（用户输入）
    last_message = state["messages"][-1]
    store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})
    
    # 使用记忆作为上下文
    system_msg = f"Answer the user's question in context: {info}"
    response = llm.invoke(
        [{"type": "system", "content": system_msg}] + state["messages"]
    )
    
    # 存储新记忆（AI 回复）
    store.put(namespace, str(uuid.uuid4()), {"data": response.content})
    
    return {"messages": response}

# 构建图
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# 编译图（同时添加 checkpointer 和 store）
graph = builder.compile(checkpointer=memory, store=in_memory_store)
```

#### 使用示例：

```python
async def main():
    # 用户6，线程10
    config = {"configurable": {"thread_id": "6_10", "user_id": "6"}}
    async for chunk in graph.astream(
        {"messages": ["你好，我是木羽"]}, 
        config, 
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    
    # 用户6，线程11（不同线程，但同一用户）
    config = {"configurable": {"thread_id": "6_11", "user_id": "6"}}
    async for chunk in graph.astream(
        {"messages": ["你知道我叫什么吗？"]}, 
        config, 
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()  # ✅ 能记住，因为同一用户
    
    # 用户8，线程10（不同用户）
    config = {"configurable": {"thread_id": "8_10", "user_id": "8"}}
    async for chunk in graph.astream(
        {"messages": ["你知道我叫什么吗？"]}, 
        config, 
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()  # ❌ 不知道，因为不同用户
    
    # 查看用户6的所有记忆
    for memory in in_memory_store.search(("memories", "6")):
        print(memory.value)

asyncio.run(main())
```

#### 记忆隔离机制：

| 场景 | thread_id | user_id | 短期记忆（对话历史） | 长期记忆（用户信息） |
|------|-----------|---------|---------------------|---------------------|
| 同用户，同线程 | 6_10 | 6 | ✅ 共享 | ✅ 共享 |
| 同用户，不同线程 | 6_11 | 6 | ❌ 隔离 | ✅ 共享 |
| 不同用户，同线程 | 8_10 | 8 | ⚠️ 共享（不推荐） | ❌ 隔离 |
| 不同用户，不同线程 | 8_11 | 8 | ❌ 隔离 | ❌ 隔离 |

**关键发现**：
- ✅ **长期记忆按 user_id 隔离**：同一用户在不同线程中可以访问相同的长期记忆
- ⚠️ **短期记忆按 thread_id 隔离**：MemorySaver 按 thread_id 存储消息历史，不同用户如果使用相同 thread_id 会共享对话历史（应避免）

---

## 5. 存储方式对比

### 5.1 短期记忆存储方式

| 存储方式 | 持久化 | 性能 | 适用场景 | 代码示例 |
|----------|--------|------|----------|----------|
| **MemorySaver** | ❌ | ⚡⚡⚡ | 开发测试 | `MemorySaver()` |
| **SqliteSaver (:memory:)** | ❌ | ⚡⚡ | 测试、临时会话 | `SqliteSaver.from_conn_string(":memory:")` |
| **SqliteSaver (文件)** | ✅ | ⚡ | 生产环境、小规模 | `SqliteSaver.from_conn_string("db.sqlite")` |
| **PostgresSaver** | ✅ | ⚡⚡ | 生产环境、大规模 | `PostgresSaver.from_conn_string(...)` |

### 5.2 长期记忆存储方式

| 存储方式 | 持久化 | 性能 | 适用场景 |
|----------|--------|------|----------|
| **InMemoryStore** | ❌ | ⚡⚡⚡ | 开发测试 |
| **RedisStore** | ✅ | ⚡⚡⚡ | 生产环境、高并发 |
| **PostgresStore** | ✅ | ⚡⚡ | 生产环境、复杂查询 |
| **FileStore** | ✅ | ⚡ | 单机部署 |

---

## 6. 上下文管理

### 6.1 为什么需要上下文管理？

**问题场景**：
```python
# ❌ 使用 with 语句时的问题
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
    # 在 with 内可以使用

# ❌ 离开 with 后，checkpointer 被关闭，graph 无法继续使用
```

### 6.2 三种上下文管理方式对比

| 方式 | 生命周期 | 灵活性 | 适用场景 |
|------|----------|--------|----------|
| **with 语句** | 代码块内 | ❌ 受限 | 简单的单次操作 |
| **ExitStack** | 手动控制 | ✅ 灵活 | 同步场景、需要长期持有 |
| **AsyncExitStack** | 手动控制 | ✅ 灵活 | 异步场景、Web 应用 |

#### 6.2.1 ExitStack 示例

```python
from contextlib import ExitStack

# 创建栈
stack = ExitStack()

# 进入多个上下文
checkpointer = stack.enter_context(SqliteSaver.from_conn_string(":memory:"))
file = stack.enter_context(open("log.txt", "w"))

# 使用资源
graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
file.write("Log message")

# 手动关闭所有资源
stack.close()
```

#### 6.2.2 AsyncExitStack 示例

```python
from contextlib import AsyncExitStack

async def main():
    stack = AsyncExitStack()
    
    try:
        # 异步进入上下文
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(":memory:")
        )
        
        # 使用资源
        graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        
        # 执行异步操作
        async for chunk in graph.astream(...):
            pass
    
    finally:
        # 确保资源被释放
        await stack.aclose()

asyncio.run(main())
```

---

## 7. 实战案例

### 7.1 多轮对话记忆 (test_36.py)

**场景**：用户自我介绍后，AI 能在后续对话中记住用户名字。

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

# 第一轮：自我介绍
graph.stream({"messages": ["你好，我叫木羽"]}, config)

# 第二轮：测试记忆
graph.stream({"messages": ["请问我叫什么？"]}, config)
# 输出：你叫木羽
```

### 7.2 跨会话持久化记忆 (test_42.py)

**场景**：程序重启后，依然能记住之前的对话。

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from contextlib import ExitStack

stack = ExitStack()
checkpointer = stack.enter_context(
    SqliteSaver.from_conn_string("checkpoints.sqlite")  # 存储到文件
)

graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

# 第一次运行
graph.stream({"messages": ["你好，我叫木羽"]}, config)
stack.close()

# 程序重启...

# 第二次运行（重新加载数据库）
stack = ExitStack()
checkpointer = stack.enter_context(
    SqliteSaver.from_conn_string("checkpoints.sqlite")
)
graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

# ✅ 依然能记住
graph.stream({"messages": ["请问我叫什么？"]}, config)
stack.close()
```

### 7.3 多用户记忆隔离 (test_45.py)

**场景**：为不同用户维护独立的长期记忆。

```python
from langgraph.store.memory import InMemoryStore

in_memory_store = InMemoryStore()
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, store=in_memory_store)

# 用户1的对话
config1 = {"configurable": {"thread_id": "1_10", "user_id": "1"}}
graph.stream({"messages": ["你好，我是张三"]}, config1)

# 用户2的对话
config2 = {"configurable": {"thread_id": "2_10", "user_id": "2"}}
graph.stream({"messages": ["你好，我是李四"]}, config2)

# 用户1询问（能记住自己的名字）
graph.stream({"messages": ["我叫什么？"]}, config1)  # 输出：张三

# 用户2询问（能记住自己的名字）
graph.stream({"messages": ["我叫什么？"]}, config2)  # 输出：李四
```

### 7.4 异步高并发场景 (test_41.py)

**场景**：Web 应用中处理多个并发请求。

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from contextlib import AsyncExitStack

async def handle_request(user_message, thread_id):
    """处理单个用户请求"""
    stack = AsyncExitStack()
    
    try:
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
        )
        
        graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        response = ""
        async for chunk in graph.astream(
            {"messages": [user_message]}, 
            config, 
            stream_mode="values"
        ):
            response = chunk["messages"][-1].content
        
        return response
    
    finally:
        await stack.aclose()

# 并发处理多个请求
async def main():
    tasks = [
        handle_request("北京天气怎么样？", "user1_thread1"),
        handle_request("上海天气怎么样？", "user2_thread1"),
        handle_request("深圳天气怎么样？", "user3_thread1"),
    ]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

asyncio.run(main())
```

---

## 8. 最佳实践

### 8.1 记忆类型选择指南

| 需求 | 推荐方案 | 理由 |
|------|----------|------|
| 多轮对话上下文 | MemorySaver + thread_id | 简单、快速 |
| 跨会话持久化 | SqliteSaver (文件模式) | 轻量级持久化 |
| 用户个性化信息 | InMemoryStore + user_id | 结构化存储 |
| 高并发 Web 应用 | AsyncSqliteSaver + RedisStore | 异步、高性能 |
| 企业级应用 | PostgresSaver + PostgresStore | 可靠、可扩展 |

### 8.2 命名规范

```python
# ✅ 推荐的命名规范
user_id = "6"                                    # 用户唯一标识
thread_id = f"{user_id}_10"                      # 线程ID = 用户ID + 会话编号
namespace = ("memories", user_id)                # 命名空间
config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": user_id
    }
}
```

### 8.3 资源管理

#### 同步场景：

```python
from contextlib import ExitStack

def create_agent_with_memory():
    stack = ExitStack()
    checkpointer = stack.enter_context(
        SqliteSaver.from_conn_string("checkpoints.sqlite")
    )
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
    return graph, stack

# 使用
graph, stack = create_agent_with_memory()
try:
    # 使用 graph
    pass
finally:
    stack.close()  # 确保资源释放
```

#### 异步场景：

```python
from contextlib import AsyncExitStack

async def create_agent_with_memory():
    stack = AsyncExitStack()
    checkpointer = await stack.enter_async_context(
        AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
    )
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
    return graph, stack

# 使用
async def main():
    graph, stack = await create_agent_with_memory()
    try:
        # 使用 graph
        async for chunk in graph.astream(...):
            pass
    finally:
        await stack.aclose()  # 确保资源释放

asyncio.run(main())
```

### 8.4 错误处理

```python
async def safe_agent_call(graph, message, config):
    """安全的 Agent 调用，带错误处理"""
    try:
        response = ""
        async for chunk in graph.astream(
            {"messages": [message]}, 
            config, 
            stream_mode="values"
        ):
            response = chunk["messages"][-1].content
        return response
    except Exception as e:
        print(f"Agent 调用失败: {e}")
        return None
```

### 8.5 性能优化

#### 1. 使用连接池（生产环境）

```python
# SQLite 不支持连接池，生产环境使用 PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost:5432/dbname",
    pool_size=10  # 连接池大小
)
```

#### 2. 定期清理旧记忆

```python
# 清理30天前的检查点
from datetime import datetime, timedelta

def cleanup_old_checkpoints(checkpointer, days=30):
    cutoff_date = datetime.now() - timedelta(days=days)
    # 实现清理逻辑
    pass
```

#### 3. 记忆限制

```python
def call_model_with_limit(state, config, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    
    # 只取最近10条记忆
    memories = store.search(namespace)
    recent_memories = memories[-10:]  # 限制记忆数量
    
    info = "\n".join([d.value["data"] for d in recent_memories])
    # ... 其他逻辑
```

### 8.6 生产环境配置

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# 根据环境选择存储方式
if os.getenv("ENVIRONMENT") == "production":
    # 生产环境：使用 PostgreSQL
    CHECKPOINT_URI = os.getenv("POSTGRES_URI")
    STORE_TYPE = "redis"
else:
    # 开发环境：使用 SQLite
    CHECKPOINT_URI = ":memory:"
    STORE_TYPE = "inmemory"
```

---

## 9. 常见问题

### 9.1 为什么使用 ExitStack？

**问题**：`with` 语句在代码块结束后会自动关闭资源，导致 Agent 无法继续使用。

**解决**：使用 `ExitStack` 手动管理资源生命周期。

### 9.2 短期记忆和长期记忆如何配合？

```python
# 同时使用短期记忆和长期记忆
checkpointer = MemorySaver()           # 短期记忆：对话历史
store = InMemoryStore()                # 长期记忆：用户信息

graph = builder.compile(
    checkpointer=checkpointer,         # 保存对话上下文
    store=store                        # 保存用户画像
)

config = {
    "configurable": {
        "thread_id": "1_10",           # 标识对话会话
        "user_id": "1"                 # 标识用户身份
    }
}
```

### 9.3 如何查看 SQLite 数据库内容？

```python
import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# 查看检查点数据
cursor.execute("SELECT * FROM checkpoints;")
for row in cursor.fetchall():
    print(row)

conn.close()
```

### 9.4 同步 vs 异步如何选择？

| 场景 | 推荐方式 | 理由 |
|------|----------|------|
| 命令行脚本 | 同步 (ExitStack) | 简单直观 |
| FastAPI/Flask | 异步 (AsyncExitStack) | 高并发性能 |
| Jupyter Notebook | 同步 | 交互式环境 |
| 批量处理 | 异步 | 并发处理多个任务 |

### 9.5 如何避免记忆污染？

**问题**：不同用户使用相同 thread_id 会共享对话历史。

**解决**：使用 `{user_id}_{session_id}` 格式的 thread_id。

```python
# ✅ 正确做法
def generate_thread_id(user_id, session_id):
    return f"{user_id}_{session_id}"

config = {
    "configurable": {
        "thread_id": generate_thread_id("6", "10"),  # "6_10"
        "user_id": "6"
    }
}
```

---

## 10. 技术栈总结

### 10.1 核心模块

| 模块 | 功能 | 导入路径 |
|------|------|----------|
| **MemorySaver** | 内存中的短期记忆 | `langgraph.checkpoint.memory` |
| **SqliteSaver** | SQLite 短期记忆（同步） | `langgraph.checkpoint.sqlite` |
| **AsyncSqliteSaver** | SQLite 短期记忆（异步） | `langgraph.checkpoint.sqlite.aio` |
| **InMemoryStore** | 内存中的长期记忆 | `langgraph.store.memory` |
| **ExitStack** | 同步上下文管理 | `contextlib` |
| **AsyncExitStack** | 异步上下文管理 | `contextlib` |

### 10.2 数据结构

```python
# Checkpointer 配置
config = {
    "configurable": {
        "thread_id": "1_10",      # 必需：会话标识
        "thread_ts": "...",       # 可选：时间戳
        "checkpoint_ns": ""       # 可选：命名空间
    }
}

# Store 命名空间
namespace = (
    "memories",    # 第一级：数据类型
    "user_6",      # 第二级：用户标识
    "preferences"  # 第三级：子分类（可选）
)

# Store 数据格式
memory_data = {
    "data": "用户输入或AI回复",
    "timestamp": "2024-11-28T...",
    "metadata": {...}
}
```

---

## 11. 学习路径建议

### 第一阶段：理解基础概念
1. 运行 `test_36.py` - 理解 MemorySaver 的基本用法
2. 运行 `test_44.py` - 理解 InMemoryStore 的基本用法
3. 理解 thread_id 和 user_id 的区别

### 第二阶段：掌握 SqliteSaver
4. 运行 `test_37.py` 和 `test_38.py` - 理解内存模式和数据库模式
5. 运行 `test_39.py` - 理解 with 语句的局限性
6. 运行 `test_40.py` 和 `test_42.py` - 掌握 ExitStack

### 第三阶段：学习异步编程
7. 运行 `test_41.py` 和 `test_43.py` - 掌握 AsyncExitStack
8. 对比同步和异步的差异

### 第四阶段：实战应用
9. 运行 `test_45.py` - 理解长期记忆在实际应用中的使用
10. 理解多用户、多会话的记忆隔离机制

---

## 12. 扩展阅读

- [LangGraph 官方文档 - Checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph 官方文档 - Store](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [SQLite 官方文档](https://www.sqlite.org/docs.html)
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)
- [contextlib 文档](https://docs.python.org/3/library/contextlib.html)

---

## 13. 总结

本系列脚本全面介绍了 LangGraph 的记忆管理系统：

### 核心要点

✅ **短期记忆（Checkpointer）**
- 用于保存对话历史
- 按 thread_id 隔离
- 支持内存和数据库两种模式

✅ **长期记忆（Store）**
- 用于保存用户信息和知识库
- 按 namespace 组织
- 支持结构化数据存储

✅ **上下文管理**
- ExitStack：同步场景
- AsyncExitStack：异步场景
- 解决 with 语句的局限性

✅ **持久化方案**
- 开发测试：InMemory
- 小规模生产：SQLite
- 大规模生产：PostgreSQL + Redis

✅ **记忆隔离**
- thread_id：会话级隔离
- user_id：用户级隔离
- namespace：数据组织和隔离

掌握这些知识后，你可以构建具有完整记忆能力的 LLM 应用，实现多轮对话、个性化服务和跨会话持久化！

---

**时间**: 2025年11月28日  
**版本**: v1.0  
**作者**: AI Assistant

