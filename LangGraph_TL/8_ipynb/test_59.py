"""
基于 Supervisor 架构实现多代理系统

"""




import matplotlib
# 使用默认交互式后端，plt.show() 可以正常弹出窗口显示

import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from typing import Literal
from faker import Faker
import random
from dotenv import load_dotenv
load_dotenv(override=True)




print("=======================创建 llm 引擎=======================")
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )





print("=======================数据库相关配置=======================")
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

# 数据库连接和表创建
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/no_09?charset=utf8mb4'     # 这里要替换成自己的数据库连接串
engine = create_engine(DATABASE_URI)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()



print("--------------------------向数据库插入/更新模拟数据-----------------------------")
# 按 ID 插入或更新模拟数据（使用 merge：ID存在则更新，不存在则插入）
Session = sessionmaker(bind=engine)
session = Session()

fake = Faker()

# 生成客户信息（ID: 1-50）
for i in range(1, 51):
    customer = CustomerInformation(
        customer_id=i,
        customer_name=fake.name(),
        contact_info=fake.phone_number(),
        region=fake.state(),  # 地区
        customer_type=random.choice(['Retail', 'Wholesale'])  # 零售、批发
    )
    session.merge(customer)  # 按 ID 更新或插入

# 生成产品信息（ID: 1-20）
for i in range(1, 21):
    product = ProductInformation(
        product_id=i,
        product_name=fake.word(),
        category=random.choice(['Electronics', 'Clothing', 'Furniture', 'Food', 'Toys']),  # 电子设备，衣服，家具，食品，玩具
        unit_price=random.uniform(10.0, 1000.0),
        stock_level=random.randint(10, 100)  # 库存
    )
    session.merge(product)  # 按 ID 更新或插入

# 生成竞争对手信息（ID: 1-10）
for i in range(1, 11):
    competitor = CompetitorAnalysis(
        competitor_id=i,
        competitor_name=fake.company(),
        region=fake.state(),
        market_share=random.uniform(0.01, 0.2)  # 市场占有率
    )
    session.merge(competitor)  # 按 ID 更新或插入

# 提交事务
session.commit()

# 生成销售数据（ID: 1-100）
for i in range(1, 101):
    sale = SalesData(
        sales_id=i,
        product_id=random.randint(1, 20),
        employee_id=random.randint(1, 10),  # 员工ID范围
        customer_id=random.randint(1, 50),
        sale_date=fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d'),
        quantity=random.randint(1, 10),
        amount=random.uniform(50.0, 5000.0),
        discount=random.uniform(0.0, 0.15)
    )
    session.merge(sale)  # 按 ID 更新或插入
session.commit()
# 关闭会话
session.close()




print("=======================工具相关=======================")
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



repl = PythonREPL()

# matplotlib 初始化代码：设置中文字体（使用交互式后端，plt.show() 可以正常显示）
MATPLOTLIB_INIT_CODE = """
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
"""

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        # 先执行 matplotlib 初始化，再执行用户代码
        full_code = MATPLOTLIB_INIT_CODE + "\n" + code
        result = repl.run(full_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str




print("=======================agent 构建=======================")
db_agent = create_react_agent(
    llm, 
    tools=[add_sale, delete_sale, update_sale, query_sales], 
    state_modifier="You use to perform database operations while should provide accurate data for the code_generator to use"
)


code_agent = create_react_agent(
    llm, 
    tools=[python_repl], 
    state_modifier="Run python code to display diagrams or output execution results"
)


class AgentState(MessagesState):
    next: str

members = ["chat", "coder", "sqler"]
options = members + ["FINISH"]


def db_node(state: AgentState):
    result = db_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="sqler")
        ]
    }
def code_node(state: AgentState):
    result = code_agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
    }
def chat(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="chatbot")]
    return {"messages": final_response}



print("=======================路由相关=======================")
class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal["chat", "coder", "sqler", "FINISH"]


print("=======================主管相关=======================")
def supervisor(state: AgentState):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: Run python code to display diagrams or output execution results.\n"
        "- sqler: Perform database operations (add, delete, update, query sales data).\n\n"
        "Given the following user request, respond with the worker to act next."
        " Each worker will perform a task and respond with their results and status.\n\n"
        "CRITICAL RULES - You MUST follow these:\n"
        "1. If a worker responds saying they 'cannot', 'unable to', '无法', or lack the ability to complete the task, you MUST respond with FINISH immediately.\n"
        "2. NEVER route to the same worker twice in a row. If the last response was from 'sqler', do NOT choose 'sqler' again.\n"
        "3. When the user's request has been answered or cannot be fulfilled, respond with FINISH.\n"
        "4. Analyze the conversation history carefully before making a decision.\n\n"
        "You must respond with a JSON object with a single key 'next' containing one of: chat, coder, sqler, FINISH"
    )

    messages = [{"role": "system", "content": system_prompt},] + state["messages"]

    response = llm.with_structured_output(Router, method="json_mode").invoke(messages)
    
    # 防御性检查
    if response is None:
        print("Warning: Structured output returned None, defaulting to chat")
        return {"next": "chat"}
    
    next_ = response.next  # pydantic 模型使用属性访问
    if next_ == "FINISH":
        next_ = END
    return {"next": next_}


print("=======================状态图构建=======================")

builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("chat", chat)
builder.add_node("coder", code_node)
builder.add_node("sqler", db_node)

for member in members:
    # 每个子代理在完成工作后总是向主管“汇报”
    builder.add_edge(member, "supervisor")

builder.add_conditional_edges("supervisor", lambda state: state["next"])

builder.add_edge(START, "supervisor")

graph = builder.compile()


print("=======================可视化图=======================")
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_59.png", "wb") as f:
    f.write(png_bytes)




print("\n\n\n")
print("*********************** 第1次测试 ***********************")
for chunk in graph.stream({"messages": "帮我查询前3个销售记录的具体信息"}, stream_mode="values"):
    print(chunk)
    print("----------------------------------------------------------\n")

print("\n\n\n")
print("*********************** 第2次测试 ***********************")
for chunk in graph.stream({"messages": "帮我根据前10名的销售记录，id为1-10，可以逐个查询或批量查询，生成对应的销售额柱状图"}, stream_mode="values"):
    print(chunk)
    print("----------------------------------------------------------\n")


print("\n\n\n")
print("*********************** 第3次测试 ***********************")
for chunk in graph.stream({"messages": "你好，请你介绍一下你自己"}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


print("\n\n\n")
print("*********************** 第4次测试 ***********************")
for chunk in graph.stream({"messages": "帮我删除 第 33条销售数据"}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()









