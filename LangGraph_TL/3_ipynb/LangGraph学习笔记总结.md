# LangGraph 学习笔记总结

本文档总结了 `3_ipynb` 文件夹中的 14 个脚本（test_15.py ~ test_28.py）的核心知识点，帮助您快速掌握 LangGraph 的关键概念和实践方法。

---

## 📚 目录

1. [条件边（Conditional Edges）](#1-条件边conditional-edges)
2. [提示工程（Prompt Engineering）](#2-提示工程prompt-engineering)
3. [结构化输出（Structured Output）](#3-结构化输出structured-output)
4. [工具节点（Tool Node）](#4-工具节点tool-node)
5. [完整Agent实现](#5-完整agent实现)

---

## 1. 条件边（Conditional Edges）

### 1.1 简单条件边（test_15.py）

条件边允许根据状态动态决定下一步执行哪个节点。

```python
from langgraph.graph import START, StateGraph, END

# 定义路由函数，根据状态返回下一个节点名称
def routing_function(state):
    if state["x"] == 10:
        return "node_b"
    else:
        return "node_c"

builder = StateGraph(dict)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

# 添加条件边
builder.add_conditional_edges("node_a", routing_function)

graph = builder.compile()
```

**核心要点：**
- 路由函数接收 `state` 参数
- 返回值是**目标节点的名称字符串**

### 1.2 使用 path_map 的条件边（test_16.py）

当路由函数返回的不是节点名称时，使用 `path_map` 进行映射。

```python
def routing_function(state):
    if state["x"] == 10:
        return True  # 返回布尔值
    else:
        return False

# path_map 将返回值映射到节点名称
builder.add_conditional_edges(
    "node_a", 
    routing_function, 
    path_map={True: "node_b", False: "node_c"}
)

builder.add_edge("node_b", END)
builder.add_edge("node_c", END)
```

**核心要点：**
- `path_map` 是一个字典，将路由函数返回值映射到节点名称
- 适用于返回布尔值、枚举值等非字符串的情况

---

## 2. 提示工程（Prompt Engineering）

### 2.1 ChatPromptTemplate 基础用法（test_17.py）

使用 LangChain 的 `ChatPromptTemplate` 创建提示词模板：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query. Wrap the output in `json`"),
    ("human", "{query}"),
])

# 使用管道符连接
chain = prompt | llm
ans = chain.invoke({"query": "用户输入..."})
print(ans.content)
```

### 2.2 从 LLM 输出中提取 JSON（test_18.py）

使用正则表达式从 LLM 返回的 Markdown JSON 块中提取数据：

```python
import re
import json
from langchain_core.messages import AIMessage

def extract_json(message: AIMessage) -> list[dict]:
    """从 ```json ``` 标签中提取 JSON"""
    text = message.content
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    return [json.loads(match.strip()) for match in matches]

# 在链中使用
chain = prompt | llm | extract_json
```

---

## 3. 结构化输出（Structured Output）

LangChain 提供了三种方式实现结构化输出，都使用 `with_structured_output()` 方法。

### 3.1 使用 Pydantic（test_19.py）⭐推荐

```python
from pydantic import BaseModel, Field
from typing import Optional

class UserInfo(BaseModel):
    """提取用户信息的模型"""
    name: str = Field(description="用户姓名")
    age: Optional[int] = Field(description="用户年龄")
    email: str = Field(description="邮箱地址")
    phone: Optional[str] = Field(description="电话号码")

structured_llm = llm.with_structured_output(UserInfo)
result = structured_llm.invoke("我叫木羽，今年28岁...")

# 可以使用 isinstance 进行类型检查
if isinstance(result, UserInfo):
    print("成功提取用户信息")
```

**优点：**
- 支持类型验证和数据校验
- 可以使用 `isinstance()` 进行类型检查
- IDE 友好，有代码提示

### 3.2 使用 TypedDict（test_20.py）

```python
from typing import Optional
from typing_extensions import Annotated, TypedDict

class UserInfo(TypedDict):
    """提取用户信息"""
    name: Annotated[str, ..., "用户姓名"]
    age: Annotated[Optional[int], None, "用户年龄"]
    email: Annotated[str, ..., "邮箱地址"]
    phone: Annotated[Optional[str], None, "电话号码"]

structured_llm = llm.with_structured_output(UserInfo)
```

**注意：** TypedDict 不支持 `isinstance()` 检查

### 3.3 使用 JSON Schema（test_21.py）

```python
json_schema = {
    "title": "user_info",
    "description": "Extracted user information",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "用户姓名"},
        "age": {"type": "integer", "description": "用户年龄", "default": None},
        "email": {"type": "string", "description": "邮箱地址"},
        "phone": {"type": "string", "description": "电话号码", "default": None},
    },
    "required": ["name", "email"],
}

structured_llm = llm.with_structured_output(json_schema)
```

### 3.4 联合类型输出 - Union（test_22.py）

根据输入内容智能选择不同的输出格式：

```python
from typing import Union

class UserInfo(BaseModel):
    """用户信息，用于数据库存储"""
    name: str = Field(description="用户姓名")
    age: Optional[int] = Field(description="用户年龄")
    email: str = Field(description="邮箱地址")
    phone: Optional[str] = Field(description="电话号码")

class ConversationalResponse(BaseModel):
    """对话响应，用于普通聊天"""
    response: str = Field(description="回复内容")

class FinalResponse(BaseModel):
    """最终响应，可以是用户信息或普通对话"""
    final_output: Union[UserInfo, ConversationalResponse]

structured_llm = llm.with_structured_output(FinalResponse)

# 测试：普通对话
result1 = structured_llm.invoke("你好")  # → ConversationalResponse

# 测试：提取用户信息
result2 = structured_llm.invoke("我叫木羽，今年28岁...")  # → UserInfo
```

---

## 4. 工具节点（Tool Node）

### 4.1 定义工具（test_24.py）

使用 `@tool` 装饰器定义工具：

```python
from langchain_core.tools import tool

@tool
def fetch_real_time_info(query):
    """Get real-time Internet information"""
    # 实现网络搜索逻辑
    url = "https://google.serper.dev/search"
    # ... 请求代码 ...
    return result

# 查看工具信息
print(f"工具名称: {fetch_real_time_info.name}")
print(f"工具描述: {fetch_real_time_info.description}")
print(f"工具参数: {fetch_real_time_info.args}")
```

### 4.2 ToolNode 工具执行节点（test_24.py, test_25.py）

`ToolNode` 负责执行工具调用：

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# 创建工具节点
tools = [fetch_real_time_info, get_weather]
tool_node = ToolNode(tools)

# 构造工具调用消息
message = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "fetch_real_time_info",
            "args": {"query": "最新新闻"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

# 执行工具
result = tool_node.invoke({"messages": [message]})
```

**核心概念：**
- `ToolNode` 只负责**执行**工具，不负责决定调用哪个工具
- 输入是包含 `tool_calls` 的 AIMessage
- 输出是工具执行结果

### 4.3 LLM 绑定工具（test_26.py）

使用 `bind_tools()` 让 LLM 自动决定调用哪个工具：

```python
from langgraph.prebuilt import ToolNode

tools = [fetch_real_time_info, get_weather]
tool_node = ToolNode(tools)

# 将工具绑定到模型
model_with_tools = llm.bind_tools(tools)

# LLM 会自动判断是否需要调用工具
result = model_with_tools.invoke("北京现在多少度？")
print(result.tool_calls)  # 包含工具调用信息

# 使用 ToolNode 执行工具
tool_result = tool_node.invoke({"messages": [result]})
```

### 4.4 使用 args_schema 定义工具参数（test_27.py, test_28.py）

为工具定义更精确的参数 Schema：

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(description="搜索查询语句")

class WeatherLoc(BaseModel):
    location: str = Field(description="城市名称")

@tool(args_schema=SearchQuery)
def fetch_real_time_info(query):
    """获取实时网络信息"""
    # ...

@tool(args_schema=WeatherLoc)
def get_weather(location):
    """获取天气信息"""
    # ...
```

---

## 5. 完整 Agent 实现

### 5.1 结构化输出 + 条件分支（test_23.py）

实现根据输出类型执行不同操作（数据库插入 vs 普通回答）：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, HumanMessage

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# 节点函数
def chat_with_model(state):
    """生成结构化输出"""
    messages = state['messages']
    structured_llm = llm.with_structured_output(FinalResponse)
    response = structured_llm.invoke(messages)
    return {"messages": [response]}

def final_answer(state):
    """普通回答节点"""
    response = state['messages'][-1].final_output.response
    return {"messages": [response]}

def insert_db(state):
    """数据库插入节点"""
    output = state['messages'][-1].final_output
    # 执行数据库插入...
    return {"messages": ["数据已存储"]}

# 路由函数
def generate_branch(state: AgentState):
    output = state['messages'][-1].final_output
    if isinstance(output, UserInfo):
        return True  # 走数据库分支
    elif isinstance(output, ConversationalResponse):
        return False  # 走普通回答分支

# 构建图
graph = StateGraph(AgentState)
graph.add_node("chat_with_model", chat_with_model)
graph.add_node("final_answer", final_answer)
graph.add_node("insert_db", insert_db)

graph.set_entry_point("chat_with_model")
graph.add_conditional_edges(
    "chat_with_model",
    generate_branch,
    {True: "insert_db", False: "final_answer"}
)
graph.set_finish_point("final_answer")
graph.set_finish_point("insert_db")

graph = graph.compile()
```

**架构图：**
```
START → chat_with_model → [条件判断] → insert_db → END
                              ↓
                       final_answer → END
```

### 5.2 Tool Calling Agent - 工具调用结束即完成（test_27.py）

支持多种工具（搜索、天气、数据库插入）+ 普通对话：

```python
# 定义输出类型的联合
class FinalResponse(BaseModel):
    final_output: Union[ConversationalResponse, SearchQuery, WeatherLoc, UserInfo]

def execute_function(state):
    """根据结构化输出执行对应工具"""
    final_output = state['messages'][-1].final_output
    
    # @tool 装饰的函数使用 .invoke(dict) 方法调用
    if isinstance(final_output, SearchQuery):
        result = fetch_real_time_info.invoke({"query": final_output.query})
        return {"messages": [result]}
    
    elif isinstance(final_output, WeatherLoc):
        result = get_weather.invoke({"location": final_output.location})
        return {"messages": [result]}
    
    elif isinstance(final_output, UserInfo):
        result = insert_db.invoke({
            "name": final_output.name,
            "age": final_output.age,
            "email": final_output.email,
            "phone": final_output.phone
        })
        return result
```

**架构图：**
```
START → chat_with_model → [条件判断] → execute_function → END
                              ↓
                       final_answer → END
```

### 5.3 Tool Calling Agent - 带自然语言总结（test_28.py）⭐最佳实践

工具执行后，再用 LLM 生成自然语言回复：

```python
from langchain_core.messages import SystemMessage, ToolMessage

# 使用 bind_tools 方式
tools = [insert_db, fetch_real_time_info, get_weather]
llm = llm.bind_tools(tools)

def execute_function(state: AgentState):
    """执行工具调用"""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    tools_dict = {t.name: t for t in tools}
    
    for t in tool_calls:
        if t['name'] not in tools_dict:
            result = "bad tool name, retry"
        else:
            result = tools_dict[t['name']].invoke(t['args'])
        results.append(ToolMessage(
            tool_call_id=t['id'], 
            name=t['name'], 
            content=str(result)
        ))
    return {'messages': results}

SYSTEM_PROMPT = """
请根据获取的信息，生成专业的中文回复。
"""

def natural_response(state):
    """将工具结果转换为自然语言"""
    messages = state['messages'][-1]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=messages.content)]
    response = llm.invoke(messages)
    return {"messages": [response]}

def exists_function_calling(state: AgentState):
    """判断是否需要调用工具"""
    result = state['messages'][-1]
    return len(result.tool_calls) > 0

# 构建图
graph = StateGraph(AgentState)
graph.add_node("chat_with_model", chat_with_model)
graph.add_node("execute_function", execute_function)
graph.add_node("final_answer", final_answer)
graph.add_node("natural_response", natural_response)

graph.set_entry_point("chat_with_model")
graph.add_conditional_edges(
    "chat_with_model",
    exists_function_calling,
    {True: "execute_function", False: "final_answer"}
)
graph.add_edge("execute_function", "natural_response")
graph.add_edge("final_answer", "natural_response")
graph.set_finish_point("natural_response")

graph = graph.compile()
```

**架构图：**
```
START → chat_with_model → [是否调用工具?] → execute_function → natural_response → END
                              ↓
                       final_answer → natural_response → END
```

---

## 📋 核心概念速查表

| 概念 | 说明 | 相关文件 |
|------|------|----------|
| `StateGraph` | 状态图构建器 | 所有文件 |
| `add_conditional_edges` | 添加条件边 | test_15, test_16 |
| `path_map` | 条件边返回值映射 | test_16 |
| `ChatPromptTemplate` | 提示词模板 | test_17, test_18 |
| `with_structured_output` | 结构化输出 | test_19-22 |
| `Pydantic BaseModel` | 定义输出结构 | test_19, test_22-28 |
| `TypedDict` | 定义输出结构（备选） | test_20 |
| `Union` | 联合类型输出 | test_22, test_23, test_27 |
| `@tool` | 工具装饰器 | test_24-28 |
| `ToolNode` | 工具执行节点 | test_24-26 |
| `bind_tools` | 绑定工具到LLM | test_26, test_28 |
| `args_schema` | 工具参数Schema | test_27, test_28 |
| `ToolMessage` | 工具调用结果消息 | test_28 |
| `set_entry_point` | 设置入口节点 | 所有图文件 |
| `set_finish_point` | 设置终止节点 | test_23, test_27, test_28 |

---

## 🔧 常用代码模板

### 初始化 LLM

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)
```

### 定义状态类型

```python
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
```

### 可视化图

```python
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
```

### 调用图

```python
from langchain_core.messages import HumanMessage

query = "用户输入"
input_message = {"messages": [HumanMessage(content=query)]}
result = graph.invoke(input_message)
print(result["messages"][-1])
```

---

## 📈 学习路径建议

1. **基础阶段**：test_15 → test_16（理解条件边）
2. **提示工程**：test_17 → test_18（LLM 输出处理）
3. **结构化输出**：test_19 → test_20 → test_21 → test_22（掌握三种方式）
4. **工具调用**：test_24 → test_25 → test_26（理解 ToolNode）
5. **实战整合**：test_23 → test_27 → test_28（完整 Agent 实现）

---

## 📝 注意事项

1. **环境变量**：确保 `.env` 文件中配置了 `DEEPSEEK_API_KEY`、`DEEPSEEK_URL`、`google_serper_KEY` 等
2. **数据库**：test_23、test_27、test_28 需要配置 MySQL 连接
3. **@tool 函数调用**：使用 `.invoke(dict)` 方法，不能直接传关键字参数
4. **TypedDict vs Pydantic**：推荐使用 Pydantic，支持类型检查和更好的 IDE 提示

---

*时间：2025年11月28日*

