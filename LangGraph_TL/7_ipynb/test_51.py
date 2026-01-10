"""
多代理协作系统 - 数据库管理 + 代码生成
===============================================
本脚本实现了一个基于 LangGraph 的多代理协作系统，主要包含：
1. 数据库管理代理 (db_manager)：负责增删改查销售数据
2. 代码生成代理 (code_generator)：负责执行 Python 代码生成图表
3. 工具节点 (call_tool)：执行具体的工具调用

系统架构：
- 两个代理通过状态图进行协作
- 支持工具调用和代理间通信
- 使用条件边实现智能路由
"""


import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)


print("=======================创建 llm 引擎=======================")
db_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )
# print(db_llm.invoke("你好,测试连通性。").content)

coder_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )
# print(coder_llm.invoke("帮我写一个使用Python实现的贪吃蛇的游戏代码").content)


print("=======================数据库相关配置=======================")
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from faker import Faker
import random

# 创建基类
Base = declarative_base()

# 定义模型
class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('product_information.product_id'))
    employee_id = Column(Integer)  # 示例简化，未创建员工表
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))
    sale_date = Column(String(50))
    quantity = Column(Integer)
    amount = Column(Float)
    discount = Column(Float)

class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)
    customer_name = Column(String(50))
    contact_info = Column(String(50))
    region = Column(String(50))
    customer_type = Column(String(50))

class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)
    product_name = Column(String(50))
    category = Column(String(50))
    unit_price = Column(Float)
    stock_level = Column(Integer)

class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)
    competitor_name = Column(String(50))
    region = Column(String(50))
    market_share = Column(Float)

# # 数据库连接和表创建
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/hhh?charset=utf8mb4'     # 这里要替换成自己的数据库连接串
engine = create_engine(DATABASE_URI, echo=True)
Base.metadata.create_all(engine)

print("--------------------------向数据库插入模拟数据-----------------------------")
# 插入模拟数据
Session = sessionmaker(bind=engine)
session = Session()

fake = Faker()

# 生成客户信息
for _ in range(50):  # 生成50个客户
    customer = CustomerInformation(
        customer_name=fake.name(),
        contact_info=fake.phone_number(),
        region=fake.state(),  # 地区
        customer_type=random.choice(['Retail', 'Wholesale'])  # 零售、批发
    )
    session.add(customer)

# 生成产品信息
for _ in range(20):  # 生成20种产品
    product = ProductInformation(
        product_name=fake.word(),
        category=random.choice(['Electronics', 'Clothing', 'Furniture', 'Food', 'Toys']),  # 电子设备，衣服，家具，食品，玩具
        unit_price=random.uniform(10.0, 1000.0),
        stock_level=random.randint(10, 100)  # 库存
    )
    session.add(product)

# 生成竞争对手信息
for _ in range(10):  # 生成10个竞争对手
    competitor = CompetitorAnalysis(
        competitor_name=fake.company(),
        region=fake.state(),
        market_share=random.uniform(0.01, 0.2)  # 市场占有率
    )
    session.add(competitor)

# 提交事务
session.commit()

# 生成销售数据，假设有100条销售记录
for _ in range(100):
    sale = SalesData(
        product_id=random.randint(1, 20),
        employee_id=random.randint(1, 10),  # 员工ID范围
        customer_id=random.randint(1, 50),
        sale_date=fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d'),
        quantity=random.randint(1, 10),
        amount=random.uniform(50.0, 5000.0),
        discount=random.uniform(0.0, 0.15)
    )
    session.add(sale)
session.commit()
# 关闭会话
session.close()



print("=======================工具相关=======================")
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Union, Optional

class AddSaleSchema(BaseModel):
    product_id: int
    employee_id: int
    customer_id: int
    sale_date: str
    quantity: int
    amount: float
    discount: float

class DeleteSaleSchema(BaseModel):
    sales_id: int

class UpdateSaleSchema(BaseModel):
    sales_id: int
    quantity: int
    amount: float

class QuerySalesSchema(BaseModel):
    sales_id: int

# 1. 添加销售数据：
@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """Add sale record to the database."""
    session = Session()
    try:
        new_sale = SalesData(
            product_id=product_id,
            employee_id=employee_id,
            customer_id=customer_id,
            sale_date=sale_date,
            quantity=quantity,
            amount=amount,
            discount=discount
        )
        session.add(new_sale)
        session.commit()
        return {"messages": ["销售记录添加成功。"]}
    except Exception as e:
        return {"messages": [f"添加失败，错误原因：{e}"]}
    finally:
        session.close()

# 2. 删除销售数据
@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """Delete sale record from the database."""
    session = Session()
    try:
        sale_to_delete = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_delete:
            session.delete(sale_to_delete)
            session.commit()
            return {"messages": ["销售记录删除成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()

# 3. 修改销售数据
@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """Update sale record in the database."""
    session = Session()
    try:
        sale_to_update = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_update:
            sale_to_update.quantity = quantity
            sale_to_update.amount = amount
            session.commit()
            return {"messages": ["销售记录更新成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"更新失败，错误原因：{e}"]}
    finally:
        session.close()

# 4. 查询销售数据
@tool(args_schema=QuerySalesSchema)
def query_sales(sales_id):
    """Query sale record from the database."""
    session = Session()
    try:
        sale_data = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_data:
            return {
                "sales_id": sale_data.sales_id,
                "product_id": sale_data.product_id,
                "employee_id": sale_data.employee_id,
                "customer_id": sale_data.customer_id,
                "sale_date": sale_data.sale_date,
                "quantity": sale_data.quantity,
                "amount": sale_data.amount,
                "discount": sale_data.discount
            }
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}。"]}
    except Exception as e:
        return {"messages": [f"查询失败，错误原因：{e}"]}
    finally:
        session.close()

from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import json
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

from langgraph.prebuilt import ToolNode

# 定义工具列表
tools = [add_sale, delete_sale, update_sale, query_sales, python_repl]
tool_executor = ToolNode(tools)


print("=======================agent 构建=======================")
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

# 数据库管理员
db_agent = create_agent(
    db_llm,
    [add_sale, delete_sale, update_sale, query_sales],
    system_message="You should provide accurate data for the code_generator to use.  and source code shouldn't be the final answer",
)

# 程序员
code_agent = create_agent(
    coder_llm,
    [python_repl],
    system_message="Run python code to display diagrams or output execution results",
)


import functools
from langchain_core.messages import AIMessage

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # 将代理输出转换为适合附加到全局状态的格式
    if isinstance(result, ToolMessage):
        pass
    else:
        # 创建一个 AIMessage 类的新实例，其中包含 result 对象的所有数据（除了 type 和 name），并且设置新实例的 name 属性为特定的值 name。
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # 跟踪发件人，这样我们就知道下一个要传给谁。
        "sender": name,
    }

# 使用 functools 的 partial 冻结 agent_node 的 agent 和 name 参数，创建了两个新函数 db_node 和 code_node，这两个新函数只需要传入 state 参数即可调用
db_node = functools.partial(agent_node, agent=db_agent, name="db_manager")
code_node = functools.partial(agent_node, agent=code_agent, name="code_generator")


print("=======================定义路由=======================")
# 任何一个代理都可以决定结束
from typing import Literal
from langgraph.graph import END
def router(state):
    # 这是一个路由
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # 前一个代理正在调用一个工具
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # 任何Agent都决定工作完成
        return END
    return "continue"


print("=======================定义状态和图=======================")
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

from langgraph.graph import END, StateGraph

# 初始化一个状态图
workflow = StateGraph(AgentState)

# 将Agent作为节点进行添加
workflow.add_node("db_manager", db_node)
workflow.add_node("code_generator", code_node)
workflow.add_node("call_tool", tool_executor)

# 通过条件边 构建 子代理之间的通信
workflow.add_conditional_edges(
    "db_manager",
    router,
    {"continue": "code_generator", "call_tool": "call_tool", END: END},
)

workflow.add_conditional_edges(
    "code_generator",
    router,
    {"continue": "db_manager", "call_tool": "call_tool",END: END},
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "db_manager": "db_manager",
        "code_generator": "code_generator",
    },
)

# 设置 db_manager 为初始节点
workflow.set_entry_point("db_manager")

# 编译图
graph = workflow.compile()


print("=======================可视化图=======================")
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_51.png", "wb") as f:
    f.write(png_bytes)


print("=======================功能测试=======================")
print("******************** test01 ********************")
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="根据sales_id使用折线图显示前5名销售的销售总额")]}, 
#     {"recursion_limit": 50}, 
#     stream_mode='values'):
#     print(chunk)

# print("******************** test02 ********************")
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="帮我删除销售id 是 5 的这名销售信息")]}, 
#     {"recursion_limit": 20}, 
#     stream_mode='values'):
#     print(chunk)

# print("******************** test03 ********************")
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="帮我找出数据库中sales_id为18的销售额是多少")]}, 
#     {"recursion_limit": 20}, 
#     stream_mode='values'):
#     print(chunk)


print("******************** test04 ********************")
for chunk in graph.stream(
    {"messages": [HumanMessage(content="帮我根据前10个销售记录id，生成对应的销售额柱状图，不要给我搞模拟的假数据")]}, 
    {"recursion_limit": 20}, 
    stream_mode='values'):
    chunk["messages"][-1].pretty_print()


