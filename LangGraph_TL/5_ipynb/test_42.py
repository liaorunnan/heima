
"""
使用 SqliteSaver 实现记忆功能，将`checkpointer`存储在  **`sqlite`数据库**  中，并接入图，实现对话记忆功能
但是使用with语句时脱离了上下文的环境，记忆就没了

这里使用ExitStack上下文管理器，来管理SqliteSaver的上下文环境，不再使用with语句
使得记忆不再局限在 with 中

本脚本是同步版本


"""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
import json
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)
from contextlib import ExitStack



stack = ExitStack()
checkpointer = stack.enter_context(SqliteSaver.from_conn_string("checkpoints20241101.sqlite"))


class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")

@tool(args_schema=WeatherLoc)
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


tools = [get_weather]


# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream({"messages": ["你好，我叫木羽"]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# 当线程ID为1时，再次询问，会从记忆中获取答案；
# 当第二次运行此脚本，即使不执行上面那一次询问，也会从数据库中得到我的名字
for chunk in graph.stream({"messages": ["请问我叫什么？"]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


stack.close()  # 关闭所有已注册的资源



