"""
使用 SqliteSaver 实现记忆功能，将`checkpointer`存储在  **`sqlite`数据库**  中，并接入图，实现对话记忆功能
但是使用with语句时脱离了上下文的环境，记忆就没了

这里使用AsyncExitStack上下文管理器，来管理AsyncSqliteSaver的上下文环境，不再使用with语句
使得记忆不再局限在 with 中

本脚本是***异步***版本
"""

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

# 导入异步版本的包
import asyncio
from contextlib import AsyncExitStack
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


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


# 将所有异步操作包装在一个异步函数中
async def main():
    # 创建 AsyncExitStack 实例
    stack = AsyncExitStack()
    
    try:
        # 异步进入上下文
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string("checkpoints20241101.sqlite")
        )
        
        # 创建图
        graph = create_react_agent(llm, tools=tools, checkpointer=checkpointer)
        
        # 配置线程ID
        config = {"configurable": {"thread_id": "24"}}
        
        # 第一个查询：天气
        print("=" * 50)
        print("查询1: 帮我查一下北京的天气")
        print("=" * 50)
        async for chunk in graph.astream({"messages": ["帮我查一下北京的天气"]}, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()
        
        # 第二个查询：记忆测试
        print("\n" + "=" * 50)
        print("查询2: 我刚才问了你什么问题")
        print("=" * 50)
        async for chunk in graph.astream({"messages": ["我刚才问了你什么问题"]}, config, stream_mode="values"):
            chunk["messages"][-1].pretty_print()
        
        # 第三个查询：事件流式输出
        print("\n" + "=" * 50)
        print("查询3: 请你非常详细的介绍一下你自己")
        print("=" * 50)
        async for event in graph.astream_events({"messages": ["请你非常详细的介绍一下你自己"]}, config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
        print()  # 最后换行
        
    finally:
        # 确保资源被正确关闭
        await stack.aclose()


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())