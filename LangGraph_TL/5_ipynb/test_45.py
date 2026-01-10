"""
长期记忆和Store（仓库）
 InMemoryStore 存储记忆，接入图

这里：
    同一用户，不同线程，记忆相同
    不同用户，相同线程，记忆会有共享，原因是：MemorySaver 按 thread_id 存储消息历史
    不同用户，不同线程，记忆不同

这里为方便管理，命名规则可使用：
user_id ：  数字，            例如 1、2、3、...
thread_id ：用户id + 数字，   例如 1_10、1_11、1_12、2_10、2_11、2_12、...

"""


import getpass
import os
import uuid
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)


in_memory_store = InMemoryStore()
memory = MemorySaver()

# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


# 定义状态模式
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 定义对话节点， 访问记忆并在模型调用中使用它们。
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 获取用户id
    user_id = config["configurable"]["user_id"]

    # 定义命名空间
    namespace = ("memories", user_id)

    # 根据用户id检索记忆
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])

    # # 存储记忆
    last_message = state["messages"][-1]
    store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})

    system_msg = f"Answer the user's question in context: {info}"

    response = llm.invoke(
        [{"type": "system", "content": system_msg}] + state["messages"]
    )

    # 存储记忆
    store.put(namespace, str(uuid.uuid4()), {"data": response.content})
    return {"messages": response}

# 构建状态图
builder = StateGraph(State)

# 向图中添加节点
builder.add_node("call_model", call_model)

# 构建边
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

# 编译图
graph = builder.compile(checkpointer=memory, store=in_memory_store)

# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_45.png", "wb") as f:
    f.write(png_bytes)


# 将所有异步操作包装在一个异步函数中
async def main():
    print("******************** 用户6，线程10 ********************")
    config = {"configurable": {"thread_id": "6_10", "user_id": "6"}}  # ✅ 修正
    async for chunk in graph.astream({"messages": ["你好，我是木羽"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print("\n")

    print("******************** 用户6，线程10 ********************")
    config = {"configurable": {"thread_id": "6_10", "user_id": "6"}}  # ✅ 修正
    async for chunk in graph.astream({"messages": ["你知道我叫什么吗？"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print("\n")

    print("******************** 用户6，线程11 ********************")
    config = {"configurable": {"thread_id": "6_11", "user_id": "6"}}  # ✅ 修正
    async for chunk in graph.astream({"messages": ["你知道我叫什么吗？"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print("\n")

    print("******************** 用户8，线程10 ********************")
    config = {"configurable": {"thread_id": "8_10", "user_id": "8"}}  # ✅ 修正
    async for chunk in graph.astream({"messages": ["你知道我叫什么吗？"]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print("\n")

    print("******************** 用户6的记忆 ********************")
    for memory in in_memory_store.search(("memories", "6")):
        print(memory.value)
    print("\n")

    print("******************** 用户8的记忆 ********************")
    for memory in in_memory_store.search(("memories", "8")):
        print(memory.value)
    print("\n")

        
    


# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())









