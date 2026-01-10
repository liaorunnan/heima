# LangGraph 学习笔记文档

> 本文档总结自 `2_ipynb` 文件夹中的 7 个脚本（test_08 ~ test_14），帮助快速掌握 LangGraph 的核心概念和使用方法。

---

## 📚 目录

1. [核心概念总览](#核心概念总览)
2. [test_08: 基础图构建与可视化](#test_08-基础图构建与可视化)
3. [test_09: TypedDict 状态定义](#test_09-typeddict-状态定义)
4. [test_10: Reducer 与状态拼接](#test_10-reducer-与状态拼接)
5. [test_11: 与 LLM 集成使用](#test_11-与-llm-集成使用)
6. [test_12: MessageGraph 快速构建对话](#test_12-messagegraph-快速构建对话)
7. [test_13: StateGraph + add_messages](#test_13-stategraph--add_messages)
8. [test_14: LangSmith 追踪集成](#test_14-langsmith-追踪集成)
9. [关键 API 速查表](#关键-api-速查表)
10. [最佳实践建议](#最佳实践建议)

---

## 核心概念总览

| 概念 | 说明 |
|------|------|
| **StateGraph** | LangGraph 的核心类，用于构建状态图 |
| **State** | 图中流动的数据结构，通常用 TypedDict 定义 |
| **Node** | 图中的节点，执行具体操作的函数 |
| **Edge** | 连接节点的边，定义执行顺序 |
| **Reducer** | 定义状态如何更新（覆盖/拼接） |
| **START/END** | 特殊节点，标记图的起点和终点 |

---

## test_08: 基础图构建与可视化

### 🎯 核心知识点

- 使用 `StateGraph(dict)` 构建灵活的图，不固定输入输出格式
- 图的基本操作：添加节点、添加边、编译、调用
- 图的可视化并保存为 PNG 图片

### 📝 代码示例

```python
from langgraph.graph import StateGraph, START, END

# 1. 创建图实例（使用 dict 灵活定义状态）
builder = StateGraph(dict)

# 2. 定义节点函数
def addition(state):
    return {"x": state["x"] + 1}

def subtraction(state):
    return {"y": state["x"] - 2}

# 3. 添加节点
builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

# 4. 添加边（定义执行顺序）
builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

# 5. 编译并运行
graph = builder.compile()
result = graph.invoke({"x": 10})  # 输出: {'x': 11, 'y': 9}
```

### 🖼️ 可视化方法

```python
# 保存为 PNG 图片
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
```

### 💡 重点理解

- `builder.schema` - 查看图的输入输出模式
- `builder.edges` - 查看图的所有边
- `builder.nodes` - 查看图的所有节点

---

## test_09: TypedDict 状态定义

### 🎯 核心知识点

- 使用 `TypedDict` 标准化输入输出格式
- **默认行为**：状态更新是**覆盖操作**

### 📝 代码示例

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

# 定义状态结构
class State(TypedDict):
    x: int
    y: int

# 使用 TypedDict 创建图
builder = StateGraph(State)
```

### ⚠️ 重要提示

```
在 LangGraph 中，如果没有显式指定 Reducer，
则对状态中某个键的所有更新都执行的是【覆盖操作】。
```

---

## test_10: Reducer 与状态拼接

### 🎯 核心知识点

- 使用 `Annotated` + `operator.add` 实现**拼接操作**而非覆盖
- Reducer 允许增量式更新状态

### 📝 代码示例

```python
import operator
from typing import Annotated, List
from typing_extensions import TypedDict

# 使用 Annotated 指定 Reducer
class State(TypedDict):
    messages: Annotated[List[str], operator.add]  # 使用 add 拼接

def addition(state):
    msg = state['messages'][-1]
    response = {"x": msg["x"] + 1}
    return {"messages": [response]}  # 返回列表，会被拼接到现有列表中
```

### 🔄 覆盖 vs 拼接

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| 覆盖（默认） | 新值替换旧值 | 单一值状态 |
| 拼接（`operator.add`） | 新值追加到列表 | 消息历史、日志记录 |

### 💡 为什么需要 Reducer？

> 没有 Reducer 时，状态更新是覆盖式的。有了 Reducer，你可以实现增量式更新，
> 这对于构建复杂的、多节点协作的工作流非常重要。

---

## test_11: 与 LLM 集成使用

### 🎯 核心知识点

- 使用 `init_chat_model` 初始化大语言模型
- 结合 `SystemMessage`、`HumanMessage`、`AIMessage` 管理对话
- 构建多节点处理流程

### 📝 代码示例

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

# 初始化 LLM
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)

# 调用 LLM
def chat_with_model(state):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}
```

### 🔗 多节点流程示例

```python
# 节点1: 与模型对话
builder.add_node("chat_with_model", chat_with_model)
# 节点2: 数据提取转换
builder.add_node("convert_messages", convert_messages)

# 设置起点和边
builder.set_entry_point("chat_with_model")
builder.add_edge("chat_with_model", "convert_messages")
builder.add_edge("convert_messages", END)
```

---

## test_12: MessageGraph 快速构建对话

### 🎯 核心知识点

- `MessageGraph` 是 `StateGraph` 的子类
- 默认使用 `add_messages` reducer（比 `operator.add` 更智能）
- 适合快速构建对话应用

### 📝 代码示例

```python
from langgraph.graph.message import MessageGraph

builder = MessageGraph()

# 添加节点（直接返回消息元组列表）
builder.add_node("chatbot", lambda state: [("assistant", "你好！")])

# 设置起点和终点
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")

graph = builder.compile()

# 调用（直接传入消息列表）
result = graph.invoke([("user", "你好")])
```

### 📊 MessageGraph vs StateGraph + operator.add

| 特性 | MessageGraph | StateGraph + operator.add |
|------|--------------|---------------------------|
| 消息处理 | 智能合并（通过 ID 更新） | 简单追加 |
| 状态结构 | 固定为消息列表 | 可自定义 |
| 适用场景 | 快速原型、标准对话 | 复杂自定义状态 |
| 代码量 | 更少 | 更多但更灵活 |

### 💡 选择建议

```
使用 MessageGraph 当：
    ✅ 构建标准的对话应用
    ✅ 需要快速原型开发
    ✅ 不需要复杂的自定义状态

使用 StateGraph + operator.add 当：
    ✅ 需要完全自定义状态结构
    ✅ 有其他非消息状态字段
    ✅ 需要更精细的控制
```

---

## test_13: StateGraph + add_messages

### 🎯 核心知识点

- `add_messages` 函数可智能管理消息（追加或通过 ID 更新）
- 与 `MessageGraph` 功能等效，但更灵活
- 支持流式输出 (`stream`)

### 📝 add_messages 用法

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# 不同 ID → 追加
msgs1 = [HumanMessage(content="你好。", id="1")]
msgs2 = [AIMessage(content="你好，很高兴认识你。", id="2")]
msgs = add_messages(msgs1, msgs2)  # 结果包含两条消息

# 相同 ID → 更新替换
msgs1 = [HumanMessage(content="你好。", id="1")]
msgs2 = [HumanMessage(content="你好呀。", id="1")]  # 相同 ID
msgs = add_messages(msgs1, msgs2)  # 结果只有一条消息（被更新）
```

### 📝 使用 add_messages 作为 Reducer

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 使用 add_messages 作为 reducer

graph_builder = StateGraph(State)
```

### 🔄 流式输出

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("模型回复:", value["messages"][-1].content)
```

---

## test_14: LangSmith 追踪集成

### 🎯 核心知识点

- LangSmith 用于追踪和监控 LLM 应用
- 只需设置环境变量即可启用

### 📝 环境变量配置

在 `.env` 文件中添加：

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
```

### 💡 使用说明

```
什么也不用设置，只设置环境变量即可！
LangSmith 会自动追踪所有的 LLM 调用、图执行等操作。
```

---

## 关键 API 速查表

### StateGraph 常用方法

| 方法 | 说明 | 示例 |
|------|------|------|
| `StateGraph(schema)` | 创建图实例 | `StateGraph(State)` |
| `add_node(name, fn)` | 添加节点 | `builder.add_node("chat", chat_fn)` |
| `add_edge(from, to)` | 添加边 | `builder.add_edge(START, "chat")` |
| `set_entry_point(name)` | 设置入口点 | `builder.set_entry_point("chat")` |
| `set_finish_point(name)` | 设置结束点 | `builder.set_finish_point("output")` |
| `compile()` | 编译图 | `graph = builder.compile()` |

### 编译后的图方法

| 方法 | 说明 | 示例 |
|------|------|------|
| `invoke(state)` | 同步执行 | `graph.invoke({"x": 1})` |
| `stream(state)` | 流式执行 | `for event in graph.stream(state):` |
| `get_graph()` | 获取图结构 | `graph.get_graph(xray=True)` |

### 消息类型

| 类型 | 说明 | 导入 |
|------|------|------|
| `HumanMessage` | 用户消息 | `from langchain_core.messages import HumanMessage` |
| `AIMessage` | AI 回复 | `from langchain_core.messages import AIMessage` |
| `SystemMessage` | 系统提示 | `from langchain_core.messages import SystemMessage` |

---

## 最佳实践建议

### 1️⃣ 状态设计

```python
# ✅ 推荐：使用 TypedDict 明确定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

# ❌ 避免：使用裸 dict（除非需要极高灵活性）
builder = StateGraph(dict)
```

### 2️⃣ Reducer 选择

```python
# 消息历史 → 使用 add_messages
messages: Annotated[list, add_messages]

# 简单列表追加 → 使用 operator.add
items: Annotated[List[str], operator.add]

# 单一值 → 不需要 Reducer（默认覆盖）
count: int
```

### 3️⃣ 调试技巧

```python
# 在节点函数中打印状态
def my_node(state):
    print("当前状态:", state)
    return {"result": process(state)}

# 使用 xray=True 可视化内部状态
graph.get_graph(xray=True).draw_mermaid_png()
```

### 4️⃣ 环境变量管理

```python
from dotenv import load_dotenv
load_dotenv(override=True)  # override=True 强制重新加载

# 在 .env 文件中统一管理：
# DEEPSEEK_API_KEY=xxx
# DEEPSEEK_URL=xxx
# LANGCHAIN_TRACING_V2=true
```

---

## 学习路径建议

```
test_08 (基础图构建)
    ↓
test_09 (TypedDict 状态)
    ↓
test_10 (Reducer 概念)
    ↓
test_11 (LLM 集成)
    ↓
test_12 (MessageGraph)  ←→  test_13 (StateGraph + add_messages)
    ↓
test_14 (LangSmith 监控)
```

---
