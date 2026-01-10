
"""
reducer的使用;
    为什么需要Reducer
    没有reducer时，状态更新是覆盖式的。有了reducer，你可以实现增量式更新，这对于构建复杂的、多节点协作的工作流非常重要。

使用 operator.add 来拼接消息

"""


import os
from dotenv import load_dotenv

import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph,  END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


# 定义图的状态模式
class State(TypedDict):
    messages: Annotated[List[str], operator.add]

# 创建图的实例
builder = StateGraph(State)

def chat_with_model(state):
    print(state)
    print("-----------------")
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

def convert_messages(state):
    # "您是一位数据提取专家，负责从文本中检索关键信息。请为所提供的文本提取相关信息，并以 JSON 格式输出。概述所提取的关键数据点。"
    EXTRACTION_PROMPT = """
    You are a data extraction specialist tasked with retrieving key information from a text.
    Extract such information for the provided text and output it in JSON format. Outline the key data points extracted.
    """
    print(state)
    print("-----------------")
    messages = state['messages']
    messages = messages[-1] 

    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=state['messages'][-1].content)
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# 添加节点
builder.add_node("chat_with_model", chat_with_model)
builder.add_node("convert_messages", convert_messages)

# 设置启动点
builder.set_entry_point("chat_with_model")

# 添加边
builder.add_edge("chat_with_model", "convert_messages")
builder.add_edge("convert_messages", END)

# 编译图
graph = builder.compile()


# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_01.png", "wb") as f:
    f.write(png_bytes)


query="你好，请你介绍一下你自己"
input_message = {"messages": [HumanMessage(content=query)]}

result = graph.invoke(input_message)

print("****************************************************************")
print(result)



