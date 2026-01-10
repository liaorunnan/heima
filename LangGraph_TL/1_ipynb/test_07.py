
"""
不再使用的是旧版langchain的 ChatOpenAI来定义deepseek模型的，而是使用langchain的统一接口 from langchain.chat_models import init_chat_model 来创建模型

在6的基础上，使用langchain的统一接口 from langchain.chat_models import init_chat_model 来创建模型，

验证了 langchain 中的 messages 的3种格式（标准格式，字典格式，元组格式），但标准的openai接口只支持一种格式（字典格式）

"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Optional
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
llm = init_chat_model(model="gpt-4o",temperature=0,)
"""

# 使用deepseek的话，要把key和url放进去，否则会报错，使用openai的话就不用，可能是默认参数
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

def llm_node(state: InputState):
    
    # langchain 支持的 messages 的格式 一：标准格式
    messages_1 = [
    # SystemMessage(content="你是一位乐于助人的智能小助理"),
    # HumanMessage(content="写个Python函数"),
    # AIMessage(content="def hello():\n    return 'Hello'"),
    # HumanMessage(content="加上文档字符串"),
    # AIMessage(content="def hello():\n    \"\"\"打招呼函数\"\"\"\n    return 'Hello'"),
    SystemMessage(content="你是一位乐于助人的智能小助理"),
    HumanMessage(content=state["question"]),
    ]


    
    # langchain 支持的 messages 的格式 二，字典格式
    messages_2 = [
    # {"role": "system", "content": "你是一个有帮助的助手"},
    # {"role": "user", "content": "你好"},
    # {"role": "assistant", "content": "你好！我很高兴为你服务。"},
    # {"role": "user", "content": "今天天气如何？"}
    {"role": "system", "content": "你是一位乐于助人的智能小助理"},
    {"role": "user", "content": state["question"]},
    ]

    
    # langchain 支持的 messages 的格式 三，元组格式
    messages_3 = [
        # ("system", "你是翻译助手"),
        # ("human", "Hello")  # human ⚠️
        ("system", "你是一位乐于助人的智能小助理"),
        ("human", state["question"]),
    ]


    response = llm.invoke(messages_3) 

    return {"llm_answer": response.content}

def action_node(state: InputState):
    messages = [
        # ("system","无论你接收到什么语言的文本，请翻译成法语",),
        # ("human", state["llm_answer"])
        {"role": "system", "content": "无论你接收到什么语言的文本，请翻译成法语"},
        {"role": "user", "content": state["llm_answer"]}
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




