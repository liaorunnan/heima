
"""
和上一个脚本类似，只是这个是异步情况下，测试 langgraph 流式输出的方法

使用 langgraph 的 create_react_agent 来创建一个 Agent，
    Agent 包含 4 个工具：
        搜索、天气、插入数据库、查询数据库
    Agent 的输入是用户的问题，输出是问题答案
    Agent 的图可视化
    Agent 的执行

详细分析了 langgraph 流式输出的方法
这里测试了两种模式，参数分别是values和updates
    values：返回add后的列表，即每个步骤之后流式传输状态的完整值。输出就是字典，其中一个messages字段，其中是一个列表，列表中的原始是HumanMessage，AIMessage，ToolMessage对象
    updates：返回每个步骤之后流式传输状态的完整值。输出的是：{节点类型（agent/tools）：{messages字段：[列表]}}，列表中的原始是HumanMessage，AIMessage，ToolMessage对象(列表中仅有一个元素，只有3者之一)
    debug：输出更详细，主要用于调试程序                     （该脚本未测试）
    messages：记录每个`messages`中的增量`token`            （该脚本未测试）
    custom：自定义流，通过`LangGraph 的 StreamWriter`方法  （该脚本未测试）


可视化图
"""



import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
import json
from dotenv import load_dotenv
import asyncio
load_dotenv(override=True)


"""================数据库相关初始化操作================="""
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
# 创建基类
Base = declarative_base()
# 定义 WeatherInfo 模型
class Weather(Base):
    __tablename__ = 'weather_11'
    city_id = Column(Integer, primary_key=True)    # 城市ID
    city_name = Column(String(50))                 # 城市名称
    main_weather = Column(String(50))              # 主要天气状况
    description = Column(String(100))              # 描述
    temperature = Column(Float)                    # 温度
    feels_like = Column(Float)                     # 体感温度
    temp_min = Column(Float)                       # 最低温度
    temp_max = Column(Float)                       # 最高温度
# 数据库连接 URI，这里要替换成自己的Mysql 连接信息，以下是各个字段的对应解释：
# root：MySQL 数据库的用户名。
# snowball950123：MySQL 数据库的密码。
# 192.168.110.131：MySQL 服务器的 IP 地址。
# langgraph_agent：要连接的数据库的名称。
# charset=utf8mb4：设置数据库的字符集为 utf8mb4，支持更广泛的 Unicode 字符
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/langgraph_agent?charset=utf8mb4'   
engine = create_engine(DATABASE_URI)
# 如果表不存在，则创建表
Base.metadata.create_all(engine)
# 创建会话
Session = sessionmaker(bind=engine)


class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")



"""================工具相关================="""
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

class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")

@tool(args_schema=WeatherLoc)
def get_weather(location):
    """
    Function to query current weather.
    :param location: Required parameter, of type string, representing the specific city name for the weather query. \
    Note that for cities in China, the corresponding English city name should be used. For example, to query the weather for Beijing, \
    the location parameter should be input as 'Beijing'.
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
        # 使用 merge 方法来插入或更新（如果已有记录则更新）
        session.merge(weather)
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

@tool(args_schema = SearchQuery)
def fetch_real_time_info(query):
    """Get real-time Internet information"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
      "q": query,
      "num": 1,
    })
    headers = {
      'X-API-KEY': '0bb32cfbd7a4d934daf4a6c3d6603b043ce9cffd',
      'Content-Type': 'application/json'
    }  
    response = requests.post(url, headers=headers, data=payload)
    data = json.loads(response.text)  # 将返回的JSON字符串转换为字典
    if 'organic' in data:
        return json.dumps(data['organic'],  ensure_ascii=False)  # 返回'organic'部分的JSON字符串
    else:
        return json.dumps({"error": "No organic results found"},  ensure_ascii=False)  # 如果没有'organic'键，返回错误信息


tools = [fetch_real_time_info, get_weather, insert_weather_to_db, query_weather_from_db]



"""================模型相关================="""
# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


"""================图相关================="""
from langgraph.prebuilt import create_react_agent
graph = create_react_agent(llm, tools=tools)


# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_08.png", "wb") as f:
    f.write(png_bytes)



print("******************** test01：参数设置为 values ********************")
async def stream_function():
    async for chunk in graph.astream(input={"messages": ["你好，成都的天气怎么样？"]}, stream_mode="values"):
        message = chunk["messages"][-1].pretty_print()
        # print(message)

asyncio.run(stream_function())

print("\n")
print("\n")
print("******************** test02：参数设置为 updates ********************")
async def stream_function_2():
    inputs = {"messages": [("human", "你好，乌鲁木齐的天气怎么样？")]}
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node, values in chunk.items():
            print(f"接收到的更新节点: '{node}'")
            message = values["messages"][0]
            message.pretty_print()
            print("\n\n")

asyncio.run(stream_function_2())


