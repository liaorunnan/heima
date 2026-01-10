"""
原理：直接断在设置断点的位置，然后直接图不再执行，后续如果再继续，就要重新运行整个图




"""


import getpass
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)


from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
import json
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage


llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


# 定义状态模式
class State(TypedDict):
    user_input: str
    model_response: str
    user_approval: str

# 定义用于大模型交互的节点
def call_model(state):
    messages = state["user_input"]
    if '删除' in state["user_input"]:
        state["user_approval"] = f"用户输入的指令是:{state['user_input']}, 请人工确认是否执行！"
    else:
        response = llm.invoke(messages)
        state["user_approval"] = "直接运行！"
        state["model_response"] = response
    return state

# 定义人工介入的 breakpoint 内部的执行逻辑
def execute_users(state):
    if state["user_approval"] == "是":
        response = "您的删除请求已经获得管理员的批准并成功执行。如果您有其他问题或需要进一步的帮助，请随时联系我们。"
        return {"model_response":AIMessage(response)}
    elif state["user_approval"] == "否":
        response = "对不起，您当前的请求是高风险操作，管理员不允许执行！"
        return {"model_response":AIMessage(response)}    
    else:
        return state

# 定义翻译节点
def translate_message(state: State):
    system_prompt = """
    Please translate the received text in any language into English as output
    """
    messages = state['model_response']
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages.content)]
    response = llm.invoke(messages)
    return {"model_response": response}

# 构建状态图
builder = StateGraph(State)

# 向图中添加节点
builder.add_node("call_model", call_model)
builder.add_node("execute_users", execute_users)
builder.add_node("translate_message", translate_message)

# 构建边
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "execute_users")
builder.add_edge("execute_users", "translate_message")
builder.add_edge("translate_message", END)

# 设置 checkpointer，使用内存存储
memory = MemorySaver()

# 在编译图的时候，添加短期记忆，并使用interrupt_before参数 设置 在 execute_users 节点之前中止图的运行，等待人工审核
graph = builder.compile(checkpointer=memory, interrupt_before=["execute_users"])

# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_46.png", "wb") as f:
    f.write(png_bytes)



# 创建一个函数来封装对话逻辑
def run_dialogue(graph, config, all_chunks=[]):
    while True:
        # 接收用户输入
        user_input = input("请输入您的消息（输入'退出'结束对话）：")
        if user_input.lower() == '退出':
            break
        
        # 运行图，直至到断点的节点
        for chunk in graph.stream({"user_input": user_input}, config, stream_mode="values"):
            all_chunks.append(chunk)
        
        # 处理可能的审批请求
        last_chunk = all_chunks[-1]
        if last_chunk["user_approval"] ==  f"用户输入的指令是:{last_chunk['user_input']}, 请人工确认是否执行！":
            user_approval = input(f"当前用户的输入是：{last_chunk['user_input']}, 请人工确认是否执行！请回复 是/否。")
            graph.update_state(config, {"user_approval": user_approval})

        # 继续执行图
        for chunk in graph.stream(None, config, stream_mode="values"):
            all_chunks.append(chunk)
        
        # 显示最终模型的响应
        print("人工智能助理：", all_chunks[-1]["model_response"].content)



# 初始化配置和状态存储
config = {"configurable": {"thread_id": "2"}}

# 使用该函数运行对话
run_dialogue(graph, config)


