"""
使用MessageGraph来构建图，而不是StateGraph中的operator.add
    operator.add：  add reducer 会将新消息追加到现有消息列表中，当节点返回新的 messages 时，不会替换整个列表，而是合并
    MessageGraph：默认使用 add_messages reducer（比 operator.add 更智能），add_messages 可以处理消息的更新（通过 ID）而不仅是追加，适合快速构建对话应用



使用 MessageGraph 当：
    构建标准的对话应用
    需要快速原型开发
    不需要复杂的自定义状态
使用 StateGraph + operator.add 当：
    需要完全自定义状态结构
    有其他非消息状态字段
    需要更精细的控制

"""






import operator
from typing import Annotated, TypedDict, List
from langgraph.graph.message import MessageGraph
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)



llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

# def chatbot(state):
#     print(state)
#     return {"messages": [llm.invoke(state["messages"])]}


builder = MessageGraph()

builder.add_node("chatbot", lambda state: [("assistant", "你好，最帅气的人！")])

builder.set_entry_point("chatbot")

builder.set_finish_point("chatbot")

graph = builder.compile()


result = graph.invoke([("user", "你好，请你介绍一下你自己.")])

print("****************************************************************")
print(result)
















