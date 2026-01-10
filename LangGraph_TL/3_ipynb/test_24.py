"""
测试 langgraph 的工具节点，调用 google.serper 的 search 工具

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

# 使用 @tool 装饰器后，可使用下面字段查看工具详情
print(f'''
name: {fetch_real_time_info.name}
description: {fetch_real_time_info.description}
arguments: {fetch_real_time_info.args}
''')
print("****************************1*****************************")

from langgraph.prebuilt import ToolNode

tools = [fetch_real_time_info]
tool_node = ToolNode(tools)

print("工具执行节点：", tool_node)   # 只负责 action ，需要传入参数，返回工具执行结果
print("****************************2*****************************")


from langchain_core.messages import AIMessage

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "fetch_real_time_info",
            "args": {"query": "Cloud3.5的最新新闻，中文"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

tool_result = tool_node.invoke({"messages": [message_with_single_tool_call]})
print("工具调用结果：", tool_result)
print("****************************3*****************************")


