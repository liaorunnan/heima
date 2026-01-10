# LangGraph 入门学习笔记

> 本文档总结了 `1_ipynb` 文件夹中 7 个脚本的核心知识点，帮助你快速掌握从 LangChain 基础到 LangGraph 图构建的完整流程。

---

## 📚 目录

1. [test_01.py - LangChain 基础回顾](#test_01---langchain-基础回顾)
2. [test_02.py - TypedDict 类型定义](#test_02---typeddict-类型定义)
3. [test_03.py - StateGraph 基础结构](#test_03---stategraph-基础结构)
4. [test_04.py - 节点间状态传递](#test_04---节点间状态传递)
5. [test_05.py - LangGraph + LLM 集成](#test_05---langgraph--llm-集成)
6. [test_06.py - 多节点图与翻译链](#test_06---多节点图与翻译链)
7. [test_07.py - init_chat_model 统一接口](#test_07---init_chat_model-统一接口)

---

## test_01 - LangChain 基础回顾

### 📌 核心概念
使用 LangChain 的标准 `ChatOpenAI` 接口调用 DeepSeek 模型。

### 🔧 关键代码

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 创建 LLM 实例（兼容 OpenAI 接口的模型）
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{input}"),
])

# 链式调用 (LCEL)
chain = prompt | llm
response = chain.invoke({
    "input_language": "English",
    "output_language": "Chinese",
    "input": "I love programming.",
})
```

### 💡 知识点
- **ChatOpenAI**: LangChain 提供的 OpenAI 兼容接口，可用于调用其他 API（如 DeepSeek）
- **ChatPromptTemplate**: 创建结构化的提示模板
- **LCEL (LangChain Expression Language)**: 使用 `|` 管道符将 prompt 和 llm 串联

---

## test_02 - TypedDict 类型定义

### 📌 核心概念
Python 的 `TypedDict` 用于定义具有明确类型的字典结构，这是 LangGraph 状态管理的基础。

### 🔧 关键代码

```python
from typing import TypedDict

class Contact(TypedDict):
    name: str
    email: str
    phone: str

def send_email(contact: Contact) -> None:
    print(f"Sending email to {contact['name']} at {contact['email']}")

# 创建符合类型定义的字典
contact_info: Contact = {
    'name': 'thy',
    'email': 'thy@example.com',
    'phone': '123-456-7890'
}
```

### 💡 知识点
- **TypedDict**: 为字典添加类型提示，提供更好的 IDE 支持和代码检查
- 在 LangGraph 中用于定义 **State（状态）** 的数据结构

---

## test_03 - StateGraph 基础结构

### 📌 核心概念
使用 `StateGraph` 定义一个简单的图结构，这是 LangGraph 的核心组件。

### 🔧 关键代码

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 1️⃣ 定义状态模式
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(InputState, OutputState):
    pass  # 合并输入输出状态

# 2️⃣ 定义节点函数
def agent_node(state: InputState):
    print("我是一个AI Agent。")
    return  # 不返回任何内容

def action_node(state: InputState):
    print("我现在是一个执行者。")
    return {"answer": "我现在执行成功了"}

# 3️⃣ 构建图
builder = StateGraph(OverallState, input=InputState, output=OutputState)

builder.add_node("agent_node", agent_node)
builder.add_node("action_node", action_node)

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "action_node")
builder.add_edge("action_node", END)

# 4️⃣ 编译并执行
graph = builder.compile()
result = graph.invoke({"question": "你好"})
```

### 💡 知识点
| 组件 | 说明 |
|------|------|
| `StateGraph` | 图的构建器 |
| `InputState` | 定义输入数据结构 |
| `OutputState` | 定义输出数据结构 |
| `OverallState` | 合并输入输出（通过继承） |
| `add_node()` | 添加节点 |
| `add_edge()` | 添加边（定义执行顺序） |
| `START / END` | 特殊节点，表示图的起点和终点 |
| `compile()` | 编译图为可执行对象 |
| `invoke()` | 执行图 |

### 🔄 执行流程
```
START → agent_node → action_node → END
```

---

## test_04 - 节点间状态传递

### 📌 核心概念
演示节点如何读取 state 并返回更新后的数据。

### 🔧 关键代码

```python
def agent_node(state: InputState):
    print("我是一个AI Agent。")
    return {"question": state["question"]}  # 返回状态

def action_node(state: InputState):
    step = state["question"]  # 读取上一节点的状态
    return {"answer": f"我接收到的问题是：{step}，读取成功了！"}
```

### 💡 知识点
- **状态传递**: 节点通过返回字典来更新状态
- **状态读取**: 节点通过 `state["key"]` 读取当前状态
- 节点返回的字典会 **合并** 到当前状态中

---

## test_05 - LangGraph + LLM 集成

### 📌 核心概念
将 DeepSeek LLM 集成到 LangGraph 图中，构建真正的 AI 问答系统。

### 🔧 关键代码

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

def llm_node(state: InputState):
    messages = [
        ("system", "你是一位乐于助人的智能小助理"),
        ("human", state["question"])
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

# 图结构
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)
```

### 🔄 执行流程
```
START → llm_node (调用 DeepSeek) → END
```

---

## test_06 - 多节点图与翻译链

### 📌 核心概念
创建包含多个 LLM 节点的图：先回答问题，再翻译成法语。

### 🔧 关键代码

```python
from typing_extensions import TypedDict, Optional

class InputState(TypedDict):
    question: str
    llm_answer: Optional[str]  # 可选字段，初始为 None

class OutputState(TypedDict):
    answer: str

# 第一个节点：回答问题
def llm_node(state: InputState):
    messages = [
        ("system", "你是一位乐于助人的智能小助理"),
        ("human", state["question"])
    ]
    response = llm.invoke(messages)
    return {"llm_answer": response.content}  # 保存到中间状态

# 第二个节点：翻译成法语
def action_node(state: InputState):
    messages = [
        ("system", "无论你接收到什么语言的文本，请翻译成法语"),
        ("human", state["llm_answer"])  # 读取上一节点的输出
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

# 边的连接
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "action_node")
builder.add_edge("action_node", END)
```

### 💡 知识点
- **Optional[str]**: 表示字段可以是 `str` 或 `None`，适用于中间状态
- **多节点链**: 多个节点可以依次处理数据，形成处理链

### 🔄 执行流程
```
START → llm_node (AI回答) → action_node (翻译成法语) → END
```

---

## test_07 - init_chat_model 统一接口

### 📌 核心概念
使用 LangChain 的统一接口 `init_chat_model` 替代特定的 `ChatOpenAI`，实现更灵活的模型切换。

### 🔧 关键代码

```python
from langchain.chat_models import init_chat_model

# 使用统一接口创建模型
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",  # 可选，留空时自动推断
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)
```

### 📝 LangChain Messages 的三种格式

LangChain 支持三种消息格式，灵活选择：

#### 格式一：标准格式（推荐）

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="你是一位乐于助人的智能小助理"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮助你的？"),
]
```

#### 格式二：字典格式（OpenAI 风格）

```python
messages = [
    {"role": "system", "content": "你是一位乐于助人的智能小助理"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
]
```

#### 格式三：元组格式（简洁）

```python
messages = [
    ("system", "你是一位乐于助人的智能小助理"),
    ("human", "你好"),  # 注意：这里用 "human" 而不是 "user"
]
```

### 💡 知识点对比

| 特性 | `ChatOpenAI` | `init_chat_model` |
|------|--------------|-------------------|
| 用途 | OpenAI 兼容接口 | 通用模型接口 |
| 切换模型 | 需修改代码 | 只需改参数 |
| 参数名 | `openai_api_key` | `api_key` |
| 推荐度 | 旧版写法 | ✅ 新版推荐 |

---

## 🎯 学习路线总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                          学习路线图                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  test_01: LangChain 基础      test_02: TypedDict 类型               │
│      │                             │                                │
│      └─────────────┬───────────────┘                                │
│                    ↓                                                │
│            test_03: StateGraph 基础                                  │
│                    │                                                │
│                    ↓                                                │
│            test_04: 状态传递                                         │
│                    │                                                │
│                    ↓                                                │
│            test_05: LLM 集成                                         │
│                    │                                                │
│                    ↓                                                │
│            test_06: 多节点图                                         │
│                    │                                                │
│                    ↓                                                │
│            test_07: 统一接口 + Messages 格式                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 核心模式速查

### StateGraph 标准模板

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 1. 定义状态
class State(TypedDict):
    input: str
    output: str

# 2. 定义节点
def my_node(state: State):
    return {"output": "处理结果"}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)

# 4. 编译执行
graph = builder.compile()
result = graph.invoke({"input": "测试"})
```

---

## 📦 依赖包

```bash
pip install langchain langchain-openai langgraph python-dotenv
```

---

## 🌐 环境变量配置

在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_URL=https://api.deepseek.com
```

---

*文档生成时间：2025年11月28日*

