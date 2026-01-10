"""

langgraph 使用 deepseek / openai 构建一个简单的问答图，并使用 action_node 进行翻译
这里使用的是旧版langchain的 ChatOpenAI来定义deepseek模型的

"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Optional
from langgraph.graph import START, END

load_dotenv(override=True)

# 定义输入的模式
class InputState(TypedDict):
    question: str
    llm_answer: Optional[str]  # 表示 answer 可以是 str 类型，也可以是 None

# 定义输出的模式
class OutputState(TypedDict):
    answer: str

# 将 InputState 和 OutputState 这两个 TypedDict 类型合并成一个更全面的字典类型。
class OverallState(InputState, OutputState):
    pass


"""
# openai 的模型
llm = ChatOpenAI(model="gpt-4o",temperature=0,)
"""

# 使用deepseek的话，要把key和url放进去，否则会报错，使用openai的话就不用，可能是默认参数
llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        openai_api_base="https://api.deepseek.com",
        temperature=0,
    )

def llm_node(state: InputState):
    messages = [
        ("system","你是一位乐于助人的智能小助理",),
        ("human", state["question"])
    ]
    
    response = llm.invoke(messages) 

    return {"llm_answer": response.content}

def action_node(state: InputState):
    messages = [
        ("system","无论你接收到什么语言的文本，请翻译成法语",),
        ("human", state["llm_answer"])
    ]
    
    response = llm.invoke(messages) 

    return {"answer": response.content}


# 明确指定它的输入和输出数据的结构或模式
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# 添加节点
builder.add_node("llm_node", llm_node)
builder.add_node("action_node", action_node)

# 添加便
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "action_node")
builder.add_edge("action_node", END)

# 编译图
graph = builder.compile()

final_answer = graph.invoke({"question":"你好，请简单介绍一下你自己"})

print(final_answer["answer"])















