# LangGraph 流式输出学习总结

本文档总结了 7 个脚本中关于 LangGraph 和 LangChain 流式输出的核心知识点。

---

## 📚 目录

1. [基础知识](#1-基础知识)
2. [脚本概览](#2-脚本概览)
3. [核心知识点详解](#3-核心知识点详解)
4. [实战案例](#4-实战案例)
5. [最佳实践](#5-最佳实践)

---

## 1. 基础知识

### 1.1 环境配置

所有脚本都使用了以下基础配置：

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 初始化模型
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)
```

### 1.2 核心依赖库

- **langchain** / **langchain_core**: LLM 框架
- **langgraph**: 构建 Agent 工作流
- **sqlalchemy**: 数据库 ORM 操作
- **requests**: HTTP 请求（API 调用）
- **pydantic**: 数据验证和模型定义

---

## 2. 脚本概览

| 脚本 | 行数 | 核心功能 | 关键知识点 |
|------|------|----------|------------|
| **test_29.py** | 45 | OpenWeather API 测试 | API 调用基础、JSON 解析 |
| **test_30.py** | 242 | LangGraph Agent 基础 | create_react_agent、工具集成、非流式输出 |
| **test_31.py** | 43 | LangChain 流式输出 | astream 方法、chunk 累加 |
| **test_32.py** | 266 | LangGraph 同步流式输出 | values/updates 模式 |
| **test_33.py** | 245 | LangGraph 异步流式输出 | 异步 astream |
| **test_34.py** | 245 | messages 模式流式输出 | 增量 token 处理 |
| **test_35.py** | 254 | astream_events 事件流 | 事件驱动流式输出 |

---

## 3. 核心知识点详解

### 3.1 OpenWeather API 集成 (test_29.py)

**功能**：查询城市天气信息

**代码要点**：
```python
def get_weather(loc):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,               # 城市名称（英文）
        "appid": "YOUR_API_KEY",
        "units": "metric",      # 摄氏度
        "lang": "zh_cn"         # 简体中文
    }
    response = requests.get(url, params=params)
    return json.dumps(response.json())
```

**注意事项**：
- 中国城市需使用英文名称（如 Beijing, Shanghai）
- API Key 需要从 OpenWeather 网站注册获取
- 返回值为 JSON 字符串格式

---

### 3.2 LangGraph Agent 基础 (test_30.py)

#### 3.2.1 工具定义

使用 `@tool` 装饰器定义工具，并使用 Pydantic 模型定义参数：

```python
class WeatherLoc(BaseModel):
    location: str = Field(description="城市名称")

@tool(args_schema=WeatherLoc)
def get_weather(location):
    """查询当前天气"""
    # 实现代码
```

#### 3.2.2 数据库集成

使用 SQLAlchemy 定义数据模型：

```python
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class Weather(Base):
    __tablename__ = 'weather_11'
    city_id = Column(Integer, primary_key=True)
    city_name = Column(String(50))
    temperature = Column(Float)
    # ... 其他字段
```

**数据库连接**：
```python
DATABASE_URI = 'mysql+pymysql://user:password@host:3306/database?charset=utf8mb4'
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
```

#### 3.2.3 创建 Agent

```python
from langgraph.prebuilt import create_react_agent

tools = [fetch_real_time_info, get_weather, 
         insert_weather_to_db, query_weather_from_db]
graph = create_react_agent(llm, tools=tools)
```

#### 3.2.4 可视化

```python
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_30.png", "wb") as f:
    f.write(png_bytes)
```

#### 3.2.5 非流式调用

```python
response = graph.invoke({
    "messages": ["北京今天的天气怎么样？"]
})
print(response["messages"][-1].content)
```

---

### 3.3 LangChain 流式输出基础 (test_31.py)

**异步流式输出**：

```python
async def stream_function():
    chunks = []
    async for chunk in llm.astream("你好，请你详细的介绍一下你自己。"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)
    
    # chunk 可以累加
    combined = chunks[0] + chunks[1] + chunks[2]
    print(combined)

asyncio.run(stream_function())
```

**关键点**：
- 使用 `astream()` 方法进行异步流式输出
- 每个 chunk 包含部分响应内容
- chunk 对象支持 `+` 操作符进行累加
- 需要 `asyncio.run()` 来执行异步函数

---

### 3.4 LangGraph 流式输出模式

LangGraph 提供了 5 种流式输出模式：

| 模式 | 返回内容 | 使用场景 |
|------|----------|----------|
| **values** | 每个步骤后的完整状态 | 需要完整上下文 |
| **updates** | 每个节点的增量更新 | 按节点处理 |
| **messages** | 增量 token 流 | 实时文本输出 |
| **debug** | 详细调试信息 | 调试程序 |
| **custom** | 自定义流 | 高级定制 |

---

### 3.5 同步流式输出 (test_32.py)

#### 3.5.1 values 模式

**特点**：返回每个步骤后的完整状态

```python
def print_stream(stream):
    for sub_stream in stream:
        # sub_stream 是字典，包含 messages 字段
        message = sub_stream["messages"][-1]
        message.pretty_print()

input_message = {"messages": ["你好，南京现在的天气怎么样？"]}
print_stream(graph.stream(input_message, stream_mode="values"))
```

**输出结构**：
```python
{
    "messages": [HumanMessage(...), AIMessage(...), ToolMessage(...)]
}
```

#### 3.5.2 updates 模式

**特点**：返回每个节点的增量更新

```python
def print_stream_updates(stream):
    for sub_stream in stream:
        # sub_stream 结构: {节点名称: {messages: [消息]}}
        for node_name, node_data in sub_stream.items():
            print(f"--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()

print_stream_updates(graph.stream(input_message, stream_mode="updates"))
```

**输出结构**：
```python
{
    "agent": {"messages": [AIMessage(...)]},
    "tools": {"messages": [ToolMessage(...)]}
}
```

**节点类型**：
- `agent`: LLM 的决策或响应
- `tools`: 工具执行结果

---

### 3.6 异步流式输出 (test_33.py)

#### 3.6.1 异步 values 模式

```python
async def stream_function():
    async for chunk in graph.astream(
        input={"messages": ["你好，成都的天气怎么样？"]}, 
        stream_mode="values"
    ):
        message = chunk["messages"][-1].pretty_print()

asyncio.run(stream_function())
```

#### 3.6.2 异步 updates 模式

```python
async def stream_function_2():
    inputs = {"messages": [("human", "你好，乌鲁木齐的天气怎么样？")]}
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"接收到的更新节点: '{node}'")
            message = values["messages"][0]
            message.pretty_print()

asyncio.run(stream_function_2())
```

**同步 vs 异步**：
- **同步** (`stream`): 阻塞式，适合简单脚本
- **异步** (`astream`): 非阻塞，适合高并发场景

---

### 3.7 messages 模式流式输出 (test_34.py)

**特点**：记录每个消息中的增量 token，实现逐字输出

```python
async def stream_function():
    async for msg, metadata in graph.astream(
        {"messages": ["你好，帮我查询一下数据库中北京的天气数据"]}, 
        stream_mode="messages"
    ):
        # 只输出非 HumanMessage 的内容
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="|", flush=True)

        # 处理 AIMessageChunk
        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg  # 累加 chunk
            
            # 输出工具调用信息
            if msg.tool_call_chunks:
                print(gathered.tool_calls)

asyncio.run(stream_function())
```

**适用场景**：
- 实时显示 LLM 生成的文本
- 类似 ChatGPT 的打字机效果
- 需要逐 token 处理的场景

---

### 3.8 astream_events 事件流 (test_35.py)

**特点**：事件驱动的流式输出，提供更细粒度的控制

#### 3.8.1 基础用法

```python
async def stream_function():
    async for event in graph.astream_events(
        {"messages": ["北京的天气怎么样"]}, 
        version="v2"
    ):
        kind = event["event"]
        print(f"{kind}: {event['name']}----------------{event['data']}")

asyncio.run(stream_function())
```

#### 3.8.2 提取 AIMessageChunk

```python
async def stream_function():
    async for event in graph.astream_events(
        {"messages": ["北京的天气怎么样"]}, 
        version="v2"
    ):
        kind = event["event"]
        
        # 过滤聊天模型流事件
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            
            # 输出文本内容
            if chunk.content:
                print(chunk.content, end="", flush=True)
            
            # 输出工具调用信息
            elif chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    if tool_call.get('name'):
                        print(f"\n[调用工具: {tool_call['name']}]\n")

asyncio.run(stream_function())
```

**事件类型**：
- `on_chat_model_stream`: 聊天模型输出流
- `on_tool_start`: 工具开始执行
- `on_tool_end`: 工具执行结束
- 其他更多事件类型...

**优势**：
- 精细控制每个事件的处理
- 可以区分模型响应和工具调用
- 适合复杂的 UI 交互场景

---

## 4. 实战案例

### 4.1 多城市天气查询与存储

**需求**：查询多个城市天气并存储到数据库

```python
response = graph.invoke({
    "messages": ["帮我查一下北京、上海、哈尔滨三个城市的天气，并存储到数据库"]
})
```

**Agent 执行流程**：
1. 解析用户意图（需要查询 3 个城市）
2. 并行调用 `get_weather` 工具 3 次
3. 提取天气数据中的关键字段
4. 调用 `insert_weather_to_db` 工具 3 次存储数据
5. 返回执行结果

### 4.2 数据库天气对比分析

**需求**：从数据库读取天气数据并进行对比分析

```python
response = graph.invoke({
    "messages": ["帮我分析一下数据库中北京和哈尔滨城市天气的信息，做详细对比"]
})
```

**Agent 执行流程**：
1. 调用 `query_weather_from_db` 查询北京天气
2. 调用 `query_weather_from_db` 查询哈尔滨天气
3. LLM 对比分析温度、天气状况等数据
4. 生成详细的对比报告和出行建议

### 4.3 实时信息检索

**需求**：获取最新的互联网信息

```python
response = graph.invoke({
    "messages": ["你知道 Claude 3.5 发布的 computer use 吗？请用中文回复"]
})
```

**Agent 执行流程**：
1. 识别需要实时信息
2. 调用 `fetch_real_time_info` 工具搜索
3. 解析搜索结果
4. 生成中文回答

---

## 5. 最佳实践

### 5.1 流式输出选择指南

| 场景 | 推荐模式 | 理由 |
|------|----------|------|
| 简单问答 | `values` | 完整上下文，易于调试 |
| UI 实时显示 | `messages` | 逐字输出，用户体验好 |
| 调试工具调用 | `updates` | 清晰区分 agent 和 tools |
| 复杂事件处理 | `astream_events` | 精细控制每个事件 |
| 性能要求高 | 异步 (`astream`) | 非阻塞，高并发 |

### 5.2 工具设计原则

1. **明确的文档字符串**：描述工具功能、参数、返回值
2. **使用 Pydantic 模型**：定义清晰的参数结构
3. **错误处理**：捕获异常并返回友好的错误信息
4. **返回标准格式**：统一返回 JSON 字符串或字典

### 5.3 数据库操作最佳实践

```python
@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_name: str):
    session = Session()
    try:
        # 执行查询
        weather_data = session.query(Weather).filter(
            Weather.city_name == city_name
        ).first()
        
        if weather_data:
            return {
                "city_name": weather_data.city_name,
                "temperature": weather_data.temperature,
                # ... 其他字段
            }
        else:
            return {"messages": [f"未找到城市 '{city_name}' 的天气信息。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()  # 确保关闭会话
```

**关键点**：
- 使用 `try-except-finally` 确保资源释放
- 每次操作创建新的 session
- 使用 `merge()` 实现插入或更新
- 发生错误时 `rollback()`

### 5.4 流式输出性能优化

1. **使用异步**：对于 I/O 密集型操作
2. **批量处理**：减少网络往返次数
3. **合理的 flush**：`print(..., end="", flush=True)`
4. **避免过度打印**：只输出必要信息

### 5.5 消息类型总结

| 消息类型 | 用途 | 来源 |
|----------|------|------|
| `HumanMessage` | 用户输入 | 用户 |
| `AIMessage` | 模型响应 | LLM |
| `ToolMessage` | 工具执行结果 | 工具 |
| `AIMessageChunk` | 模型流式输出片段 | LLM (流式) |

---

## 6. 常见问题

### 6.1 为什么使用英文城市名称？

OpenWeather API 只支持英文城市名称。对于中国城市：
- 北京 → Beijing
- 上海 → Shanghai
- 哈尔滨 → Harbin

### 6.2 如何处理 API Key？

使用环境变量：
```python
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv("OPENWEATHER_API_KEY")
```

`.env` 文件示例：
```
OPENWEATHER_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_URL=https://api.deepseek.com
```

### 6.3 流式输出没有实时显示？

确保：
1. 使用 `flush=True` 参数
2. 不要使用 `print()` 的缓冲
3. 在 Jupyter Notebook 中可能需要 `IPython.display`

### 6.4 数据库连接失败？

检查：
1. 数据库服务是否运行
2. 连接字符串是否正确
3. 用户权限是否足够
4. 防火墙设置

---

## 7. 技术栈总结

### 7.1 核心框架

- **LangChain**: LLM 应用开发框架
- **LangGraph**: 构建多步骤 Agent 工作流
- **SQLAlchemy**: Python SQL 工具包和 ORM

### 7.2 外部服务

- **OpenWeather API**: 天气数据服务
- **Serper API**: Google 搜索 API 代理
- **DeepSeek**: 大语言模型服务
- **MySQL**: 关系型数据库

### 7.3 Python 标准库

- **asyncio**: 异步 I/O
- **requests**: HTTP 库
- **json**: JSON 处理
- **os**: 操作系统接口

---

## 8. 学习路径建议

1. **第一步**：运行 `test_29.py` 理解 API 调用基础
2. **第二步**：运行 `test_30.py` 理解 Agent 的构建和工具集成
3. **第三步**：运行 `test_31.py` 理解基础流式输出
4. **第四步**：运行 `test_32.py` 和 `test_33.py` 对比同步/异步流式输出
5. **第五步**：运行 `test_34.py` 理解 messages 模式的实时输出
6. **第六步**：运行 `test_35.py` 掌握事件驱动的高级流式输出

---

## 9. 扩展阅读

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [OpenWeather API 文档](https://openweathermap.org/api)
- [SQLAlchemy 官方文档](https://www.sqlalchemy.org/)
- [Pydantic 官方文档](https://docs.pydantic.dev/)

---

## 10. 总结

本系列脚本从基础的 API 调用到复杂的 Agent 流式输出，全面展示了 LangGraph 的核心功能：

✅ **工具集成**：如何定义和使用工具  
✅ **数据库操作**：SQLAlchemy ORM 的最佳实践  
✅ **流式输出**：5 种不同模式的适用场景  
✅ **异步编程**：提升并发性能  
✅ **事件驱动**：精细控制 Agent 执行流程  

掌握这些知识点后，你可以构建功能强大的 LLM 应用，实现复杂的多步骤推理、工具调用和实时交互。

---

**文档生成时间**: 2025年11月28日  
**版本**: v1.0  
**作者**: AI Assistant

