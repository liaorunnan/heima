"""
前两个脚本讨论了 MessageGraph 和 StateGraph + add 来管理 messages  。
MessageGraph 是 StateGraph 的一个子类，它默认使用 add_messages reducer

使用 MessageGraph 或 StateGraph + add_messages 来更新 messages 是一样的，

本脚本 使用 StateGraph + add_messages 来管理 messages 。


"""

import os
from langchain_core.messages import AIMessage, HumanMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)



msgs1 = [HumanMessage(content="你好。", id="1")]
msgs2 = [AIMessage(content="你好，很高兴认识你。", id="2")]
msgs = add_messages(msgs1, msgs2)
print("****************************************************************")
print(msgs)


msgs1 = [HumanMessage(content="你好。", id="1")]
msgs2 = [HumanMessage(content="你好呀。", id="1")]
msgs = add_messages(msgs1, msgs2)
print("****************************************************************")
print(msgs)



class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)



llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

def chatbot(state: State):
    # print(state)
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_01.png", "wb") as f:
    f.write(png_bytes)



def stream_graph_updates(user_input: str):  
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("模型回复:", value["messages"][-1].content)


while True:
    try:
        #print("1111111111")
        user_input = input("用户提问: ")
        if user_input.lower() in ["退出"]:
            print("下次再见！")
            break

        stream_graph_updates(user_input)
    except:
        break






