"""
复杂代理架构中如何添加动态断点，

"""

import os
import json
from langchain_core.tools import tool
from typing import Union, Optional
from pydantic import BaseModel, Field
import requests
from langgraph.prebuilt import ToolNode
import getpass
import os
import json
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv(override=True)



class WeatherLoc(BaseModel):
    location: str = Field(description="The location name of the city")

class SearchQuery(BaseModel):
    query: str = Field(description="Questions for networking queries")


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


tools = [get_weather, fetch_real_time_info]

tool_node = ToolNode(tools)



if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

llm = llm.bind_tools(tools)


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果没有 工具调用，则输出至最终节点
    if not last_message.tool_calls:
        return "end"
    # 如果还有子任务需要继续执行工具调用的话，则继续等待执行
    else:
        return "continue"

def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)


workflow.add_edge(START, "agent")

# 添加条件边 
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_edge("action", "agent")


memory = MemorySaver()

graph = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_48.png", "wb") as f:
    f.write(png_bytes)


print("*********************** 第一次：查询北京的天气 ***********************")
config = {"configurable": {"thread_id": "4"}}
for sub_stream in graph.stream({"messages": "请帮我查一下北京的天气"}, config, stream_mode="updates"):
        # 遍历所有节点的更新
        for node_name, node_data in sub_stream.items():
            print(f"\n--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()
print("\n")
print("-" * 20)
for sub_stream in graph.stream(None, config, stream_mode="updates"):
        # 遍历所有节点的更新
        for node_name, node_data in sub_stream.items():
            print(f"\n--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()


print("\n\n\n")
print("*********************** 第二次：浏览器查询 ***********************")
config = {"configurable": {"thread_id": "4"}}
for sub_stream in graph.stream({"messages": "最近 OpenAI 有哪些大动作？"}, config, stream_mode="updates"):
        # 遍历所有节点的更新
        for node_name, node_data in sub_stream.items():
            print(f"\n--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()
print("\n")
print("-" * 20)
for sub_stream in graph.stream(None, config, stream_mode="updates"):
        # 遍历所有节点的更新
        for node_name, node_data in sub_stream.items():
            print(f"\n--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()
print("\n")
print("-" * 20)
for sub_stream in graph.stream(None, config, stream_mode="updates"):
        # 遍历所有节点的更新
        for node_name, node_data in sub_stream.items():
            print(f"\n--- {node_name.upper()} 节点 ---")
            if "messages" in node_data:
                for message in node_data["messages"]:
                    message.pretty_print()



