
"""
langgraph 使用 deepseek 构建一个简单的问答图，这里使用的是旧版langchain的 ChatOpenAI来定义deepseek模型的
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph import START, END

load_dotenv(override=True)

# 定义输入的模式
class InputState(TypedDict):
    question: str

# 定义输出的模式
class OutputState(TypedDict):
    answer: str

# 将 InputState 和 OutputState 这两个 TypedDict 类型合并成一个更全面的字典类型。
class OverallState(InputState, OutputState):
    pass


llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        openai_api_base="https://api.deepseek.com"
    )

def llm_node(state: InputState):
    messages = [
        ("system","你是一位乐于助人的智能小助理",),
        ("human", state["question"])
    ]
    
    response = llm.invoke(messages) 

    return {"answer": response.content}



# 明确指定它的输入和输出数据的结构或模式
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# 添加节点
builder.add_node("llm_node", llm_node)

# 添加边
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

# 编译图
graph = builder.compile()


result = graph.invoke({"question":"你好，我用来测试"})
print("****************************************************************")
print(result)









