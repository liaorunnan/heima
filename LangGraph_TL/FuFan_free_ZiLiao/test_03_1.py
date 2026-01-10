
"""
预构建智能体 API 就是 create_react_agent 的使用

"""


import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv 
load_dotenv(override=True)
import requests,json

class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")

@tool(args_schema = WeatherQuery)
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": os.getenv("OPENWEATHER_API_KEY"),    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)





from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# 封装外部函数列表
tools = [get_weather]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  
agent = create_react_agent(model=model, tools=tools)

response = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍下你自己。"}]})

print("=================================================    1    ===============================================================")
print(response)

print("=================================================    2    ===============================================================")
print(response["messages"])


print("=================================================    3    ===============================================================")
print(response["messages"][-1].content)

print("=================================================    4    ===============================================================")
print(response["messages"][-1])


response = agent.invoke({"messages": [{"role": "user", "content": "请问北京今天天气如何？"}]})
print("=================================================    5    ===============================================================")
print(response["messages"])

print("=================================================    6    ===============================================================")
print(response["messages"][-1].content)

print("=================================================    7    ===============================================================")
print(response["messages"][0])  # 用户输入 HumanMessage
print(response["messages"][1])  # 模型输出 AIMessage （ Function calling 的结果 ）
print(response["messages"][2])  # 工具输出 ToolMessage
print(response["messages"][3])  # 模型输出 AIMessage


print("=================================================    8    ===============================================================")









