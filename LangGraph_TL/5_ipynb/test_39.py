"""
使用 SqliteSaver 实现记忆功能，将`checkpointer`存储在  **内存**  中，并接入图，实现对话记忆功能
但是脱离了上下文的环境，记忆就没了

"""


from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
import json
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)


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

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display



with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)

    # 可视化
    png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph_10.png", "wb") as f:
        f.write(png_bytes)
    
    config = {"configurable": {"thread_id": "1"}}

    for chunk in graph.stream({"messages": ["你好，我叫木羽"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
     
    for chunk in graph.stream({"messages": ["请问我叫什么？"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()



