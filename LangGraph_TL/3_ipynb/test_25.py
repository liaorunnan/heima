
"""
测试 langgraph 的执行工具节点 “ tool_node = ToolNode(tools) ”，
测试两个工具的调用，调用 google.serper 的 search 工具和 get_weather 工具

"""

import requests
import json
from langchain_core.tools import tool
import os
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


# 使用 @tool 装饰器后，可使用下面字段查看工具详情
print(f'''
name: {fetch_real_time_info.name}
description: {fetch_real_time_info.description}
arguments: {fetch_real_time_info.args}
''')
print("****************************1*****************************")

from langgraph.prebuilt import ToolNode

tools = [fetch_real_time_info, get_weather]
tool_node = ToolNode(tools)

print("工具节点：", tool_node)
print("****************************2*****************************")


from langchain_core.messages import AIMessage

message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "fetch_real_time_info",
            "args": {"query": "Cloud3.5的最新新闻"},
            "id": "tool_call_id",
            "type": "tool_call",
        },
        {
            "name": "get_weather",
            "args": {"location": "beijing"},
            "id": "tool_call_id_2",
            "type": "tool_call",
        },
    ],
)

tool_result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]})
print("工具调用结果：", tool_result)
print("****************************3*****************************")



