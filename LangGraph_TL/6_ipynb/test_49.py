

import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv(override=True)


"""===================数据库初始化相关初始化操作================="""
# 创建基类
Base = declarative_base()
# 定义 WeatherInfo 模型
class Weather(Base):
    __tablename__ = 'weather_333'
    city_id = Column(Integer, primary_key=True)  # 城市ID
    city_name = Column(String(50))                # 城市名称
    main_weather = Column(String(50))             # 主要天气状况
    description = Column(String(100))              # 描述
    temperature = Column(Float)                    # 温度
    feels_like = Column(Float)                    # 体感温度
    temp_min = Column(Float)                      # 最低温度
    temp_max = Column(Float)                      # 最高温度
# 数据库连接 URI
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/langgraph_agent?charset=utf8mb4'     # 这里要替换成自己的数据库连接串
engine = create_engine(DATABASE_URI)
# 如果表不存在，则创建表
Base.metadata.create_all(engine)
# 创建会话
Session = sessionmaker(bind=engine)



"""===================工具定义================="""
class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")
class WeatherInfo(BaseModel):
    """Extracted weather information for a specific city."""
    city_id: int = Field(..., description="The unique identifier for the city")
    city_name: str = Field(..., description="The name of the city")
    main_weather: str = Field(..., description="The main weather condition")
    description: str = Field(..., description="A detailed description of the weather")
    temperature: float = Field(..., description="Current temperature in Celsius")
    feels_like: float = Field(..., description="Feels-like temperature in Celsius")
    temp_min: float = Field(..., description="Minimum temperature in Celsius")
    temp_max: float = Field(..., description="Maximum temperature in Celsius")
class QueryWeatherSchema(BaseModel):
    """Schema for querying weather information by city name."""
    city_name: str = Field(..., description="The name of the city to query weather information")
class DeleteWeatherSchema(BaseModel):
    """Schema for deleting weather information by city name."""
    city_name: str = Field(..., description="The name of the city to delete weather information")
    
@tool(args_schema = WeatherLoc)
def get_weather(location):
    """
    Function to query current weather.
    :param loc: Required parameter, of type string, representing the specific city name for the weather query. \
    Note that for cities in China, the corresponding English city name should be used. For example, to query the weather for Beijing, \
    the loc parameter should be input as 'Beijing'.
    :return: The result of the OpenWeather API query for current weather, with the specific URL request address being: https://api.openweathermap.org/data/2.5/weather. \
    The return type is a JSON-formatted object after parsing, represented as a string, containing all important weather information.
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"
    # Step 2.设置查询参数
    params = {
        "q": location,               
        "appid": os.getenv("OPENWEATHER_API_KEY"),    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }
    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

@tool(args_schema=WeatherInfo)
def insert_weather_to_db(city_id, city_name, main_weather, description, temperature, feels_like, temp_min, temp_max):
    """Insert weather information into the database."""
    session = Session()  # 确保为每次操作创建新的会话
    try:
        # 创建天气实例
        weather = Weather(
            city_id=city_id,
            city_name=city_name,
            main_weather=main_weather,
            description=description,
            temperature=temperature,
            feels_like=feels_like,
            temp_min=temp_min,
            temp_max=temp_max
        )
        # 添加到会话
        session.add(weather)
        # 提交事务
        session.commit()
        return {"messages": [f"天气数据已成功存储至Mysql数据库。"]}
    except Exception as e:
        session.rollback()  # 出错时回滚
        return {"messages": [f"数据存储失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话

@tool(args_schema=QueryWeatherSchema)
def query_weather_from_db(city_name: str):
    """Query weather information from the database by city name."""
    session = Session()
    try:
        # 查询天气数据
        weather_data = session.query(Weather).filter(Weather.city_name == city_name).first()
        print(weather_data)
        if weather_data:
            return {
                "city_id": weather_data.city_id,
                "city_name": weather_data.city_name,
                "main_weather": weather_data.main_weather,
                "description": weather_data.description,
                "temperature": weather_data.temperature,
                "feels_like": weather_data.feels_like,
                "temp_min": weather_data.temp_min,
                "temp_max": weather_data.temp_max
            }
        else:
            return {"messages": [f"未找到城市 '{city_name}' 的天气信息。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话

@tool(args_schema=DeleteWeatherSchema)
def delete_weather_from_db(city_name: str):
    """Delete weather information from the database by city name."""
    session = Session()
    try:
        # 查询要删除的天气数据
        weather_data = session.query(Weather).filter(Weather.city_name == city_name).first()
        if weather_data:
            # 删除记录
            session.delete(weather_data)
            session.commit()
            return {"messages": [f"城市 '{city_name}' 的天气信息已成功删除。"]}
        else:
            return {"messages": [f"未找到城市 '{city_name}' 的天气信息。"]}
    except Exception as e:
        session.rollback()  # 出错时回滚
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话

tools = [get_weather, insert_weather_to_db, query_weather_from_db, delete_weather_from_db]
tool_node = ToolNode(tools)


"""===================模型实例化================="""
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )
llm = llm.bind_tools(tools)

"""===================构建图================="""
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    elif last_message.tool_calls[0]["name"] == "delete_weather_from_db":
        return "2"
    else:
        return "1"

def risk_tool(state):
    new_messages = []
    tool_calls = state["messages"][-1].tool_calls
    
    # tools =  [get_weather, insert_weather_to_db, query_weather_from_db, delete_weather_from_db]
    tools =  [delete_weather_from_db]
    tools = {t.name: t for t in tools}
    
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        new_messages.append(
            {
                "role": "tool",
                "name": tool_call["name"],
                "content": result,
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": new_messages}

workflow = StateGraph(MessagesState)
workflow.add_node("call_model", call_model)
workflow.add_node("tool_node", tool_node)
workflow.add_node("risk_tool", risk_tool)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "1": "tool_node",
        "2":"risk_tool",
        "end": END,
    },
)

workflow.add_edge("tool_node", "call_model")
workflow.add_edge("risk_tool", "call_model")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["risk_tool"])



# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_49.png", "wb") as f:
    f.write(png_bytes)

"""
print("*********************** 第一步 ***********************")
config = {"configurable": {"thread_id": "99"}} 
for chunk in graph.stream({"messages": "北京的天气怎么样？"}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("*********************** 第二步 ***********************")
config = {"configurable": {"thread_id": "99"}} 
for chunk in graph.stream({"messages": "将你知道的北京的天气存入数据库"}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第三步 ***********************")
config = {"configurable": {"thread_id": "99"}}
for chunk in graph.stream({"messages": "帮我同时查一下上海、杭州的天气，比较哪个城市更适合现在出游。"}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第四步（上） ***********************")
config = {"configurable": {"thread_id": "99"}}
for chunk in graph.stream({"messages": "帮我删除数据库中北京的天气数据"}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
state = graph.get_state(config)
print(state.next)
print(state.tasks)
print(state.values)

print("\n\n\n")
print("*********************** 第四步（下） ***********************")
for chunk in graph.stream(None, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

"""


print("\n\n\n")
print("*********************** 第五步 ***********************")
config = {"configurable": {"thread_id": "10"}}
for chunk in graph.stream({"messages": "查询郑州的天气，存入数据库，然后再从数据库中删除郑州的天气数据"}, config, stream_mode="values"):
    state = graph.get_state(config)

    # print(state.next)
    # print(state.tasks)

    # 检查是否有任务，如果没有则结束循环
    chunk["messages"][-1].pretty_print()
    if not state.tasks:
        # print("所有任务都已完成。")

        break
    
    if state.tasks[0].name == 'risk_tool':
        while True:
            user_input = input("是否允许执行删除操作？请输入'是'或'否'：")
            if user_input in ["是", "否"]:
                break
            else:
                print("输入错误，请输入'是'或'否'。")
            
        if user_input == "是":
            graph.update_state(config=config, values=chunk)
            for event in graph.stream(None, config, stream_mode="values"):
                event["messages"][-1].pretty_print()
        elif user_input == "否":
            state = graph.get_state(config)
            tool_call_id = state.values["messages"][-1].tool_calls[0]["id"]
            print(tool_call_id)

            #我们现在需要构造一个替换工具调用。把参数改为“xxsd”，请注意，我们可以更改任意数量的参数或工具名称-它必须是一个有效的
            new_message = {
                "role": "tool",
                # 这是得到的用户不允许操作的反馈
                "content": "管理员不允许执行该操作！",
                "name": "delete_weather_from_db",
                "tool_call_id": tool_call_id,
            }
            graph.update_state(config, {"messages": [new_message]}, as_node="risk_tool",)
            for event in graph.stream(None, config, stream_mode="values"):
                event["messages"][-1].pretty_print()




