"""
实现了 3个工具 + 普通回答，共4种回答方式：
    搜索、天气、插入数据库、普通回答

Tool Calling Agent的完整实现案例，最终输出是工具调用完成后就结束了，并且最后接一个自然语言输出的节点

图可视化

"""

import os
import requests
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from typing import Optional, Union
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
import operator
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv(override=True)




"""   数据库相关的   """
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker 
from sqlalchemy.orm import sessionmaker
# 创建基类
Base = declarative_base()
# 定义 UserInfo 模型
class User(Base):
    __tablename__ = 'users' # 要存入的表的名称
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    age = Column(Integer)
    email = Column(String(100))
    phone = Column(String(15))
# 数据库连接 URI，这里要替换成自己的Mysql 连接信息，以下是各个字段的对应解释：
# root：MySQL 数据库的用户名。
# snowball950123：MySQL 数据库的密码。
# 192.168.110.131：MySQL 服务器的 IP 地址。
# langgraph_agent：要连接的数据库的名称。
# charset=utf8mb4：设置数据库的字符集为 utf8mb4，支持更广泛的 Unicode 字符
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/langgraph_agent?charset=utf8mb4'   
engine = create_engine(DATABASE_URI, echo=True)
# 如果表不存在，则创建表
Base.metadata.create_all(engine)
# 创建会话
Session = sessionmaker(bind=engine)
session = Session()



"""   ===========工具相关的===========   """
class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")
class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")
class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email address of the user")
    phone: Optional[str] = Field(description="The phone number of the user")

@tool(args_schema = SearchQuery)
def fetch_real_time_info(query):
    """Get real-time Internet information"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
      "q": query,
      "num": 1,
    })
    headers = {
      'X-API-KEY': os.getenv("google_serper_KEY"),
      'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    data = json.loads(response.text)  # 将返回的JSON字符串转换为字典
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False)  # 返回'organic'部分的JSON字符串
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False)  # 如果没有'organic'键，返回错误信息

@tool(args_schema = WeatherLoc)
def get_weather(location):
    """Call to get the current weather."""
    if location.lower() in ["beijing","北京"]:
        return "北京的温度是16度，天气晴朗。"
    elif location.lower() in ["shanghai","上海"]:
        return "上海的温度是20度，部分多云。"
    else:
        return "不好意思，并未查询到具体的天气信息。"

@tool(args_schema = UserInfo)
def insert_db(name, age, email, phone):
    """Insert user information into the database, The required parameters are name, age, email, phone"""
    session = Session()  # 确保为每次操作创建新的会话
    try:
        # 创建用户实例
        user = User(name=name, age=age, email=email, phone=phone)
        # 添加到会话
        session.add(user)
        # 提交事务
        session.commit()
        return {"messages": [f"数据已成功存储至Mysql数据库。"]}
    except Exception as e:
        session.rollback()  # 出错时回滚
        return {"messages": [f"数据存储失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话

print(f'''
name: {fetch_real_time_info.name}
description: {fetch_real_time_info.description}
arguments: {fetch_real_time_info.args}
''')
print("****************************1*****************************")




"""   ===========模型相关的===========   """
# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

tools = [insert_db, fetch_real_time_info, get_weather]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def chat_with_model(state):
    """generate structured output"""
    messages = state['messages']
    response = llm.invoke(messages)  # 这里可以不使用格式化输出
    return {"messages": [response]}

def execute_function(state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    tools = [insert_db, fetch_real_time_info, get_weather]
    tools = {t.name: t for t in tools}
    for t in tool_calls:
        if not t['name'] in tools:     
            result = "bad tool name, retry" 
        else:
            result = tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

def final_answer(state):
    """generate natural language responses"""
    messages = state['messages'][-1]
    return {"messages": [messages]}


# 请你基于现在得到的信息，进行总结，生成专业的回复，注意，请用中文回复
SYSTEM_PROMPT = """
Please summarize the information obtained so far and generate a professional response. Note, please reply in Chinese.
"""

def natural_response(state):
    """generate final language responses"""
    messages = state['messages'][-1]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=messages.content)]
    response = llm.invoke(messages)
    return {"messages": [response]}


def exists_function_calling(state: AgentState):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0



graph = StateGraph(AgentState)

graph.add_node("chat_with_model", chat_with_model)
graph.add_node("execute_function", execute_function)
graph.add_node("final_answer", final_answer)
graph.add_node("natural_response", natural_response)

# 设置图的启动节点
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


# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_28.png", "wb") as f:
    f.write(png_bytes)


messages = [HumanMessage(content="你好，请你介绍一下你自己")]
result = graph.invoke({"messages": messages})
print("结果一：\n", result["messages"][-1].content)
print("****************************2*****************************")

messages = [HumanMessage(content="Cloud3.5的最新新闻")]
result = graph.invoke({"messages": messages})
print("结果二：\n", result["messages"][-1].content)
print("****************************3*****************************")

messages = [HumanMessage(content="我是田灵，28岁，电话是133232，有问题随时联系，邮箱是873@qq.com，存入数据库，不要废话，直接行动")]
result = graph.invoke({"messages": messages})
print("结果三：\n", result["messages"][-1].content)
print("****************************4*****************************")



