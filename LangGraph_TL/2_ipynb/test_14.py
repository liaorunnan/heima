"""
langsmith 的使用
什么也不用设置，只设置环境变量即可，在 .env 中

"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import getpass
from langchain.chat_models import init_chat_model
# from langchain_core.prompts import ChatPromptTemplate
# from langsmith import traceable
# from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
load_dotenv(override=True)


# 设置环境变量
# os.environ["LANGCHAIN_TRACING_V2"] = ""
# os.environ["LANGCHAIN_API_KEY"] = ""

# # 验证环境变量是否设置成功
# print(os.getenv("LANGCHAIN_TRACING_V2"))
# print(os.getenv("LANGCHAIN_API_KEY"))

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


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


def stream_graph_updates(user_input: str):  
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("模型回复:", value["messages"][-1].content)


while True:
    try:
        user_input = input("用户提问: ")
        if user_input.lower() in ["退出"]:
            print("下次再见！")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available  
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


