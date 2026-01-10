"""
使用带两个工具（ google.serper 的 search 工具和 get_weather 工具 ）的 llm 的回答用户问题，单个节点

使用 model_with_tools = llm.bind_tools(tools) 来给 llm 加工具。invoke返回的是调用工具的字典列表
（此时工具还没用执行，再使用 ToolNode 来创建工具节点，并执行工具）


"""

import os
import requests
import json
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv(override=True)


@tool
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

@tool
def get_weather(location):
    """Call to get the current weather."""
    if location.lower() in ["beijing","北京"]:
        return "北京的温度是16度，天气晴朗。"
    elif location.lower() in ["shanghai","上海"]:
        return "上海的温度是20度，部分多云。"
    else:
        return "不好意思，并未查询到具体的天气信息。"


from langgraph.prebuilt import ToolNode
tools = [fetch_real_time_info, get_weather]
tool_node = ToolNode(tools)


# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


model_with_tools = llm.bind_tools(tools)

print("model_with_tools.kwargs结果:", model_with_tools.kwargs)
print("****************************1*****************************")

result_1 = model_with_tools.invoke("Cloud3.5的最新新闻，中文")
print("result_1结果:", result_1)
print("\n")
print("result_1.tool_calls结果:", result_1.tool_calls)
print("****************************2*****************************")

result_2 = model_with_tools.invoke("北京现在多少度呀？")
print("result_2结果:", result_2)
print("\n")
print("result_2.tool_calls结果:", result_2.tool_calls)
print("****************************3*****************************")


print("****************************开始执行工具*****************************")
tool_result_1 = tool_node.invoke({"messages": [model_with_tools.invoke("Cloud3.5的最新新闻，要中文")]})
print("tool_result结果:", tool_result_1)
print("****************************4*****************************")

tool_result_2 = tool_node.invoke({"messages": [model_with_tools.invoke("北京现在多少度呀")]})
print("tool_result结果:", tool_result_2)
print("****************************5*****************************")




