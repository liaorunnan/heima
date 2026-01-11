from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.types import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from datetime import datetime



from llm_input import model
from conf import settings


from sqlmodel import Field, SQLModel, Session, create_engine

# 创建数据库引擎和连接 - 使用SQLite作为示例（实际使用时可替换为MySQL）
engine = create_engine(settings.url(), echo=True)  




class Memory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    data: str
    type: str = "system"
    time: datetime = Field(default_factory=datetime.now)
    thread_id: str = Field(default="")
# 创建表
SQLModel.metadata.create_all(engine)



class EmailAgentState(TypedDict):
    name: str
    messages: Annotated[list, add_messages]  # 添加自动消息累积机制
    response: str = ""
    thread_id: str = Field(default="")



def Insert_memory(lastest_memory: str, type: str = "system", thread_id: str = ""):
    memory = Memory(data=lastest_memory, type=type, thread_id=thread_id)
    with Session(engine) as session:
        session.add(memory)
        session.commit()


def Remember_name(State: EmailAgentState):
    # 记录线程
    print(State)
    
    print(f"记住了用户姓名: {State.get('name', '未知')}")
    
    # 获取最新的用户消息
    if State['messages']:
        latest_message = State['messages'][-1]
        Insert_memory(lastest_memory=latest_message.content, type="user", thread_id=State["thread_id"])
        print(f"问题: {latest_message.content}")
    
    # 构建完整的消息链
    message = [SystemMessage(content=f"你是一个聊天助手")] + State['messages']
    
    # 调用模型
    llm_input = model.invoke(message)
    State["response"] = llm_input.content
    Insert_memory(lastest_memory=llm_input.content, type="system", thread_id=State["thread_id"])
    print(f"回复: {llm_input.content}")
    print("------------------------------------")
    
    # 返回更新后的状态
    return {"messages": [AIMessage(content=llm_input.content)], "response": llm_input.content}


# 创建图
workflow = StateGraph(EmailAgentState)

workflow.add_node(Remember_name.__name__, Remember_name)


workflow.add_edge(START, Remember_name.__name__)
workflow.add_edge(Remember_name.__name__, END)


config1 = RunnableConfig(configurable={
    "thread_id": "1"
})

config2 = RunnableConfig(configurable={
    "thread_id": "2"
})

# 使用检查点进行持久化编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)



for event in app.stream({"name": "张三", "messages": [HumanMessage(content="你好，我叫张三，你叫什么名字？")], "thread_id": config1["configurable"]["thread_id"]}, config=config1):
    print(event)

for event in app.stream({"name": "李四", "messages": [HumanMessage(content="你好，我叫李四，你叫什么名字？")], "thread_id": config2["configurable"]["thread_id"]}, config=config2):
    print(event)
    
for event in app.stream({"messages": [HumanMessage(content="你好，你知道我是谁吗？")], "thread_id": config1["configurable"]["thread_id"]}, config=config1):
    print(event)    
