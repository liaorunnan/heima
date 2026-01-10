"""

在上述代码的基础上，添加一个节点，用于将用户信息插入到数据库中。
插入数据库和正常回答 两个分支进行。
可视化图

"""




import os
from typing import Union, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# 定义数据库插入的用户信息模型
class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email address of the user")
    phone: Optional[str] = Field(description="The phone number of the user")

# 定义正常生成模型回复的模型
class ConversationalResponse(BaseModel):
    """Respond to the user's query in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")


# 定义最终响应模型，可以是用户信息或一般响应
class FinalResponse(BaseModel):
    final_output: Union[UserInfo, ConversationalResponse]


# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )



"""   数据库相关的   """
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker 
from sqlalchemy.orm import sessionmaker
# 创建基类
Base = declarative_base()
# 定义 UserInfo 模型
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    age = Column(Integer)
    email = Column(String(100))
    phone = Column(String(15))
# 数据库连接 URI，这里要替换成自己的Mysql 连接信息，以下是各个字段的对应解释：
# root：MySQL 数据库的用户名。
# snowball950123：MySQL 数据库的密码。
# 192.168.110.131：MySQL 服务器的 IP 地址。
# langgraph_agent：要连接的数据库的名称。
# charset=utf8mb4：设置数据库的字符集为 utf8mb4，支持更广泛的 Unicode 字符
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/langgraph_agent?charset=utf8mb4'   
engine = create_engine(DATABASE_URI, echo=True)
# 如果表不存在，则创建表
Base.metadata.create_all(engine)
# 创建会话
Session = sessionmaker(bind=engine)
session = Session()



def chat_with_model(state):
    """generate structured output"""
    print(state)
    print("-----------------")
    messages = state['messages']
    structured_llm = llm.with_structured_output(FinalResponse)
    response = structured_llm.invoke(messages)
    return {"messages": [response]}

def final_answer(state):
    """generate natural language responses"""
    print(state)
    print("-----------------")
    messages = state['messages'][-1]
    response = messages.final_output.response
    return {"messages": [response]}

def insert_db(state):
    """Insert user information into the database"""
    session = Session()  # 确保为每次操作创建新的会话
    try:
        result = state['messages'][-1]
        output = result.final_output
        # 创建用户实例
        user = User(name=output.name, age=output.age, email=output.email, phone=output.phone)
        # 添加到会话
        session.add(user)
        # 提交事务
        session.commit()
        return {"messages": [f"数据已成功存储至Mysql数据库。"]}
    except Exception as e:
        session.rollback()  # 出错时回滚
        return {"messages": [f"数据存储失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def generate_branch(state: AgentState):
    result = state['messages'][-1]
    output = result.final_output

    if isinstance(output, UserInfo):
        return True
    elif isinstance(output, ConversationalResponse):
        return False


graph = StateGraph(AgentState)

# 添加三个节点
graph.add_node("chat_with_model", chat_with_model)
graph.add_node("final_answer", final_answer)
graph.add_node("insert_db", insert_db)

# 设置图的启动节点
graph.set_entry_point("chat_with_model")

# 设置条件边
graph.add_conditional_edges(
    "chat_with_model",
    generate_branch,
    {True: "insert_db", False: "final_answer"}
    )

# 设置终止节点
graph.set_finish_point("final_answer")
graph.set_finish_point("insert_db")

# 编译图
graph = graph.compile()


# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_23.png", "wb") as f:
    f.write(png_bytes)


query="我叫木羽，今年28岁，邮箱地址是snow@gmial.com，电话是1323521313"
input_message = {"messages": [HumanMessage(content=query)]}

result = graph.invoke(input_message)
print("*******************************1*********************************")
print(result)
print("\n"*3)

print("*******************************2*********************************")
print(result["messages"][-1])
print("\n"*3)


query="你好，请你介绍一下你自己"
input_message = {"messages": [HumanMessage(content=query)]}

result = graph.invoke(input_message)
print("*******************************3*********************************")
print(result)
print("\n"*3)

print("*******************************4*********************************")
print(result["messages"][-1])
print("\n"*3)







