
"""
注释版本


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
# 加载环境变量，override=True 表示覆盖已存在的环境变量
load_dotenv(override=True)


print("=======================创建 llm 引擎=======================")
"""
创建两个独立的 LLM 实例
-------------------------------
为什么需要两个 LLM？
- db_llm: 专门用于数据库管理代理，处理数据查询和操作
- coder_llm: 专门用于代码生成代理，处理 Python 代码执行

参数说明：
- model: 模型名称
- model_provider: 模型提供商（可选，系统会自动推断）
- api_key: API 密钥，从环境变量中读取
- base_url: API 基础 URL
- temperature: 温度参数（0表示确定性输出，1表示更随机）
"""
db_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", 
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,  # 设置为0以获得稳定的数据库查询结果
    )
# 测试连通性：print(db_llm.invoke("你好,测试连通性。").content)

coder_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,  # 设置为0以获得稳定的代码生成结果
    )
# 测试代码生成：print(coder_llm.invoke("帮我写一个使用Python实现的贪吃蛇的游戏代码").content)


print("=======================数据库相关配置=======================")
"""
数据库模型定义和数据生成
=======================================
使用 SQLAlchemy ORM 定义销售管理系统的数据模型
包含4个主要表：
1. sales_data: 销售数据表（主表）
2. customer_information: 客户信息表
3. product_information: 产品信息表
4. competitor_analysis: 竞争对手分析表
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from faker import Faker
import random

# 创建基类 - 所有模型类都需要继承自这个基类
Base = declarative_base()

# 定义模型
"""
SalesData: 销售数据表
------------------------
存储每一笔销售交易的详细信息
主键: sales_id
外键: product_id (关联到产品表), customer_id (关联到客户表)
"""
class SalesData(Base):
    __tablename__ = 'sales_data'
    sales_id = Column(Integer, primary_key=True)  # 销售ID（主键，自增）
    product_id = Column(Integer, ForeignKey('product_information.product_id'))  # 产品ID（外键）
    employee_id = Column(Integer)  # 员工ID（简化处理，未创建员工表）
    customer_id = Column(Integer, ForeignKey('customer_information.customer_id'))  # 客户ID（外键）
    sale_date = Column(String(50))  # 销售日期
    quantity = Column(Integer)  # 销售数量
    amount = Column(Float)  # 销售金额
    discount = Column(Float)  # 折扣

"""
CustomerInformation: 客户信息表
--------------------------------
存储客户的基本信息和分类
"""
class CustomerInformation(Base):
    __tablename__ = 'customer_information'
    customer_id = Column(Integer, primary_key=True)  # 客户ID（主键）
    customer_name = Column(String(50))  # 客户姓名
    contact_info = Column(String(50))  # 联系方式
    region = Column(String(50))  # 所在地区
    customer_type = Column(String(50))  # 客户类型（零售/批发）

"""
ProductInformation: 产品信息表
-------------------------------
存储产品的详细信息和库存
"""
class ProductInformation(Base):
    __tablename__ = 'product_information'
    product_id = Column(Integer, primary_key=True)  # 产品ID（主键）
    product_name = Column(String(50))  # 产品名称
    category = Column(String(50))  # 产品分类
    unit_price = Column(Float)  # 单价
    stock_level = Column(Integer)  # 库存水平

"""
CompetitorAnalysis: 竞争对手分析表
------------------------------------
存储竞争对手的市场信息
"""
class CompetitorAnalysis(Base):
    __tablename__ = 'competitor_analysis'
    competitor_id = Column(Integer, primary_key=True)  # 竞争对手ID（主键）
    competitor_name = Column(String(50))  # 竞争对手名称
    region = Column(String(50))  # 所在地区
    market_share = Column(Float)  # 市场占有率

# 数据库连接和表创建
# 格式：mysql+pymysql://用户名:密码@主机:端口/数据库名?字符集
DATABASE_URI = 'mysql+pymysql://root:thy382324@127.0.0.1:3306/hhh?charset=utf8mb4'
# echo=True：打印所有 SQL 语句（用于调试）
engine = create_engine(DATABASE_URI, echo=True)
# 创建所有定义的表（如果表已存在则跳过）
Base.metadata.create_all(engine)

print("--------------------------向数据库插入模拟数据-----------------------------")
"""
使用 Faker 生成模拟数据
==========================
这部分代码用于生成测试数据，模拟真实的销售系统环境

生成数据量：
- 50 个客户
- 20 种产品
- 10 个竞争对手
- 100 条销售记录

Faker 库：可以生成各种类型的假数据（姓名、电话、地址等）
"""
# 创建会话工厂
Session = sessionmaker(bind=engine)
session = Session()  # 创建会话实例，用于与数据库交互

# 初始化 Faker 对象
fake = Faker()

# 生成客户信息
print("生成 50 个客户...")
for _ in range(50):  # 循环50次
    customer = CustomerInformation(
        customer_name=fake.name(),  # 生成随机姓名
        contact_info=fake.phone_number(),  # 生成随机电话号码
        region=fake.state(),  # 生成随机州/地区
        customer_type=random.choice(['Retail', 'Wholesale'])  # 随机选择客户类型：零售或批发
    )
    session.add(customer)  # 将客户对象添加到会话

# 生成产品信息
print("生成 20 种产品...")
for _ in range(20):
    product = ProductInformation(
        product_name=fake.word(),  # 生成随机单词作为产品名
        category=random.choice(['Electronics', 'Clothing', 'Furniture', 'Food', 'Toys']),  # 随机选择产品分类
        unit_price=random.uniform(10.0, 1000.0),  # 生成10到1000之间的随机价格
        stock_level=random.randint(10, 100)  # 生成10到100之间的随机库存
    )
    session.add(product)

# 生成竞争对手信息
print("生成 10 个竞争对手...")
for _ in range(10):
    competitor = CompetitorAnalysis(
        competitor_name=fake.company(),  # 生成随机公司名称
        region=fake.state(),  # 生成随机州/地区
        market_share=random.uniform(0.01, 0.2)  # 生成1%到20%的随机市场占有率
    )
    session.add(competitor)

# 提交事务 - 将客户、产品和竞争对手信息写入数据库
# 注意：必须先提交这些数据，才能在销售数据中引用它们的 ID
session.commit()

# 生成销售数据
print("生成 100 条销售记录...")
for _ in range(100):
    sale = SalesData(
        product_id=random.randint(1, 20),  # 随机选择产品ID（1-20）
        employee_id=random.randint(1, 10),  # 随机选择员工ID（1-10）
        customer_id=random.randint(1, 50),  # 随机选择客户ID（1-50）
        sale_date=fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d'),  # 生成过去一年内的随机日期
        quantity=random.randint(1, 10),  # 随机销售数量（1-10）
        amount=random.uniform(50.0, 5000.0),  # 随机销售金额（50-5000）
        discount=random.uniform(0.0, 0.15)  # 随机折扣（0-15%）
    )
    session.add(sale)

# 提交销售数据
session.commit()

# 关闭会话，释放数据库连接
session.close()
print("模拟数据生成完成！")



print("=======================工具相关=======================")
"""
定义工具函数 - Agent 可调用的功能
==========================================
工具是 Agent 与外部系统交互的接口
本系统包含5个工具：
1. add_sale: 添加销售记录
2. delete_sale: 删除销售记录
3. update_sale: 更新销售记录
4. query_sales: 查询销售记录
5. python_repl: 执行 Python 代码

使用 Pydantic 定义工具的输入模式，确保类型安全
"""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Union, Optional

# 定义工具的输入模式（Schema）
# 这些模式用于验证传入参数的类型和结构

class AddSaleSchema(BaseModel):
    """添加销售记录的参数模式"""
    product_id: int  # 产品ID
    employee_id: int  # 员工ID
    customer_id: int  # 客户ID
    sale_date: str  # 销售日期
    quantity: int  # 数量
    amount: float  # 金额
    discount: float  # 折扣

class DeleteSaleSchema(BaseModel):
    """删除销售记录的参数模式"""
    sales_id: int  # 要删除的销售记录ID

class UpdateSaleSchema(BaseModel):
    """更新销售记录的参数模式"""
    sales_id: int  # 要更新的销售记录ID
    quantity: int  # 新的数量
    amount: float  # 新的金额

class QuerySalesSchema(BaseModel):
    """查询销售记录的参数模式"""
    sales_id: int  # 要查询的销售记录ID

# 1. 添加销售数据工具
# @tool 装饰器将函数转换为 LangChain 工具
# args_schema 指定输入参数的验证模式
@tool(args_schema=AddSaleSchema)
def add_sale(product_id, employee_id, customer_id, sale_date, quantity, amount, discount):
    """
    Add sale record to the database.
    
    添加新的销售记录到数据库
    参数：产品ID、员工ID、客户ID、销售日期、数量、金额、折扣
    返回：操作结果消息
    """
    session = Session()  # 创建新的数据库会话
    try:
        # 创建新的销售记录对象
        new_sale = SalesData(
            product_id=product_id,
            employee_id=employee_id,
            customer_id=customer_id,
            sale_date=sale_date,
            quantity=quantity,
            amount=amount,
            discount=discount
        )
        session.add(new_sale)  # 添加到会话
        session.commit()  # 提交事务
        return {"messages": ["销售记录添加成功。"]}
    except Exception as e:
        # 捕获并返回错误信息
        return {"messages": [f"添加失败，错误原因：{e}"]}
    finally:
        # 无论成功或失败，都要关闭会话
        session.close()

# 2. 删除销售数据工具
@tool(args_schema=DeleteSaleSchema)
def delete_sale(sales_id):
    """
    Delete sale record from the database.
    
    从数据库中删除指定的销售记录
    参数：销售记录ID
    返回：操作结果消息
    """
    session = Session()
    try:
        # 查询要删除的记录
        sale_to_delete = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_delete:
            session.delete(sale_to_delete)  # 删除记录
            session.commit()
            return {"messages": ["销售记录删除成功。"]}
        else:
            return {"messages": [f"未找到销售记录ID：{sales_id}"]}
    except Exception as e:
        return {"messages": [f"删除失败，错误原因：{e}"]}
    finally:
        session.close()

# 3. 修改销售数据工具
@tool(args_schema=UpdateSaleSchema)
def update_sale(sales_id, quantity, amount):
    """
    Update sale record in the database.
    
    更新数据库中的销售记录（仅更新数量和金额）
    参数：销售记录ID、新数量、新金额
    返回：操作结果消息
    """
    session = Session()
    try:
        # 查询要更新的记录
        sale_to_update = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_to_update:
            # 更新字段
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

# 4. 查询销售数据工具
@tool(args_schema=QuerySalesSchema)
def query_sales(sales_id):
    """
    Query sale record from the database.
    
    从数据库中查询指定的销售记录
    参数：销售记录ID
    返回：销售记录的详细信息（字典格式）
    """
    session = Session()
    try:
        # 查询销售记录
        sale_data = session.query(SalesData).filter(SalesData.sales_id == sales_id).first()
        if sale_data:
            # 返回所有字段的字典
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


# 5. Python REPL 工具（代码执行器）
"""
Python REPL (Read-Eval-Print Loop) 工具
------------------------------------------
这是一个强大的工具，允许 Agent 执行任意 Python 代码
主要用途：
- 数据分析
- 生成图表
- 数学计算
- 数据处理

安全提示：在生产环境中使用时需要特别注意安全性
"""
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import json

# 创建 Python REPL 实例
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    
    执行 Python 代码工具
    ---------------------
    功能：执行传入的 Python 代码字符串
    注意事项：
    - 如果要查看输出，必须使用 print() 函数
    - 代码在隔离的环境中执行
    - 可以使用常见的数据分析库（numpy, pandas, matplotlib 等）
    """
    try:
        # 执行代码并捕获输出
        result = repl.run(code)
    except BaseException as e:
        # 捕获所有异常（包括 BaseException）
        return f"Failed to execute. Error: {repr(e)}"
    
    # 格式化成功的结果
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    
    # 提示 Agent 完成任务后应返回 FINAL ANSWER
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

# 创建工具节点
"""
ToolNode: 工具执行节点
------------------------
ToolNode 是 LangGraph 的预构建节点，用于执行工具调用
它会自动：
1. 解析 Agent 的工具调用请求
2. 执行相应的工具函数
3. 返回工具执行结果
"""
from langgraph.prebuilt import ToolNode

# 定义系统中所有可用的工具列表
tools = [add_sale, delete_sale, update_sale, query_sales, python_repl]

# 创建工具执行器节点
tool_executor = ToolNode(tools)


print("=======================agent 构建=======================")
"""
Agent 构建 - 创建协作的 AI 代理
==========================================
本系统包含两个专业化的 Agent：
1. db_manager (数据库管理员): 负责数据库操作
2. code_generator (代码生成器): 负责执行 Python 代码

每个 Agent 都有：
- 专用的 LLM 模型
- 特定的工具集
- 定制的系统提示词
"""
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_agent(llm, tools, system_message: str):
    """
    创建一个 Agent
    
    参数：
    - llm: 语言模型实例
    - tools: 该 Agent 可以使用的工具列表
    - system_message: 该 Agent 的特定系统提示词
    
    返回：
    - 绑定了工具的提示词链（Prompt | LLM）
    
    工作原理：
    1. 创建提示词模板，包含系统消息和消息占位符
    2. 将工具列表绑定到 LLM
    3. 返回可调用的链
    """
    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # 系统提示词 - 定义 Agent 的行为和协作规则
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            # 消息占位符 - 用于插入对话历史
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # 部分应用系统消息（将特定 Agent 的提示词填入模板）
    prompt = prompt.partial(system_message=system_message)
    
    # 部分应用工具名称列表
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    # 返回：提示词模板 | LLM（绑定工具）
    # 这个链会：1) 格式化提示词 2) 调用 LLM 3) LLM 可以选择调用工具
    return prompt | llm.bind_tools(tools)

# 创建数据库管理员 Agent
"""
db_manager Agent
------------------
职责：处理所有数据库相关操作
工具：add_sale, delete_sale, update_sale, query_sales
特点：提供准确的数据供代码生成器使用
"""
db_agent = create_agent(
    db_llm,
    [add_sale, delete_sale, update_sale, query_sales],  # 数据库操作工具
    system_message="You should provide accurate data for the code_generator to use.  and source code shouldn't be the final answer",
)

# 创建代码生成器 Agent
"""
code_generator Agent
----------------------
职责：执行 Python 代码，生成图表和输出结果
工具：python_repl
特点：可以运行任意 Python 代码进行数据分析和可视化
"""
code_agent = create_agent(
    coder_llm,
    [python_repl],  # Python 代码执行工具
    system_message="Run python code to display diagrams or output execution results",
)


# Agent 节点包装器
"""
agent_node 函数 - 将 Agent 包装成图节点
--------------------------------------------
作用：
1. 调用 Agent 执行任务
2. 将 Agent 的输出转换为统一格式
3. 跟踪消息发送者，以便路由到正确的下一个节点

这个函数是连接 Agent 和 StateGraph 的桥梁
"""
import functools
from langchain_core.messages import AIMessage

def agent_node(state, agent, name):
    """
    Agent 节点函数
    
    参数：
    - state: 当前的图状态（包含消息历史和发送者信息）
    - agent: 要调用的 Agent 实例
    - name: Agent 的名称（用于标识和路由）
    
    返回：
    - 包含新消息和发送者信息的状态更新字典
    
    工作流程：
    1. 使用当前状态调用 Agent
    2. Agent 返回响应（可能是工具调用或文本响应）
    3. 将响应转换为 AIMessage 格式
    4. 更新状态中的消息列表和发送者信息
    """
    # 调用 Agent，传入当前状态（包含所有历史消息）
    result = agent.invoke(state)
    
    # 将代理输出转换为适合附加到全局状态的格式
    if isinstance(result, ToolMessage):
        # 如果结果已经是 ToolMessage，直接使用
        pass
    else:
        # 创建一个 AIMessage 实例
        # 使用 result.dict() 获取所有数据，但排除 type 和 name 字段
        # 然后设置 name 为当前 Agent 的名称
        # 这样可以追踪哪个 Agent 发送了这条消息
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    return {
        "messages": [result],  # 添加新消息到消息列表
        "sender": name,  # 记录发送者，用于路由决策
    }

# 使用 functools.partial 创建特定的节点函数
"""
functools.partial 的作用
--------------------------
partial 函数可以"冻结"函数的某些参数，创建一个新函数

例如：
原函数：agent_node(state, agent, name)
使用 partial 后：db_node(state)

这样在图中调用节点时，只需要传入 state 参数即可
agent 和 name 参数已经被预先设置好了
"""
# 创建数据库管理员节点
db_node = functools.partial(agent_node, agent=db_agent, name="db_manager")

# 创建代码生成器节点
code_node = functools.partial(agent_node, agent=code_agent, name="code_generator")


print("=======================定义路由=======================")
"""
路由逻辑 - 控制工作流的流向
==========================================
router 函数决定下一步应该执行什么操作：
1. 调用工具 (call_tool)
2. 继续传递给另一个 Agent (continue)
3. 结束流程 (END)

这是 LangGraph 中实现动态工作流的核心机制
"""
from typing import Literal
from langgraph.graph import END

def router(state):
    """
    路由函数 - 智能决策下一步动作
    
    参数：
    - state: 当前状态（包含消息历史）
    
    返回：
    - "call_tool": 如果上一个 Agent 请求调用工具
    - END: 如果任务已完成
    - "continue": 如果需要传递给另一个 Agent
    
    决策逻辑：
    1. 检查最后一条消息是否包含工具调用
       -> 如果是，返回 "call_tool"
    2. 检查最后一条消息是否包含 "FINAL ANSWER"
       -> 如果是，返回 END（结束流程）
    3. 否则返回 "continue"（继续协作）
    
    工作流示例：
    User Input -> db_manager -> router -> 
    - 如果需要查询数据 -> call_tool (query_sales) -> db_manager
    - 如果数据准备好 -> continue -> code_generator
    - 如果代码需要执行 -> call_tool (python_repl) -> code_generator
    - 如果任务完成 -> END
    """
    # 获取所有消息
    messages = state["messages"]
    
    # 获取最后一条消息（最新的 Agent 响应）
    last_message = messages[-1]
    
    # 检查是否有工具调用请求
    if last_message.tool_calls:
        # tool_calls 是 LangChain 自动解析的工具调用列表
        # 如果存在，说明 Agent 想要使用工具
        return "call_tool"
    
    # 检查是否包含最终答案标记
    if "FINAL ANSWER" in last_message.content:
        # Agent 通过在响应中包含 "FINAL ANSWER" 来表示任务完成
        # 这是协作 Agent 之间约定的信号
        return END
    
    # 默认情况：继续传递给另一个 Agent
    return "continue"


print("=======================定义状态和图=======================")
"""
状态定义和图构建
==========================================
这是 LangGraph 的核心部分：
1. 定义状态结构（AgentState）
2. 构建状态图（StateGraph）
3. 添加节点和边
4. 编译成可执行的图

状态图架构：
                    START
                      ↓
                [db_manager]
                      ↓
                  (router)
                /     |     \
      call_tool  continue   END
           ↓         ↓
    [call_tool] [code_generator]
           ↓         ↓
      (sender)   (router)
           ↓     /   |   \
          ...  ...  ...  END
"""
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Agent 状态定义
    ----------------
    这个类定义了在图中传递的状态结构
    
    字段说明：
    - messages: 消息列表，存储所有的对话历史
      * 使用 Annotated 添加元数据
      * operator.add 表示新消息会被追加到列表末尾（而不是替换）
      * Sequence[BaseMessage] 表示这是一个消息序列
    
    - sender: 字符串，记录最后一条消息的发送者
      * 用于路由决策（决定下一个节点）
      * 可能的值："db_manager" 或 "code_generator"
    
    为什么使用 operator.add？
    - 在 LangGraph 中，状态更新可以指定合并策略
    - operator.add 表示累加（append）策略
    - 这样每个节点返回的新消息会被添加到现有消息列表中
    - 而不是替换整个消息列表
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

from langgraph.graph import END, StateGraph

# 初始化状态图
"""
StateGraph 是 LangGraph 的核心类
- 参数：AgentState - 定义了状态的结构
- 作用：管理节点之间的状态流转
"""
workflow = StateGraph(AgentState)

# 添加节点
"""
节点是图中的执行单元，可以是：
- Agent 节点：执行 AI 决策
- Tool 节点：执行具体工具
- 自定义函数节点

每个节点：
1. 接收当前状态
2. 执行某些操作
3. 返回状态更新
"""
workflow.add_node("db_manager", db_node)  # 数据库管理员节点
workflow.add_node("code_generator", code_node)  # 代码生成器节点
workflow.add_node("call_tool", tool_executor)  # 工具执行节点

# 添加条件边 - 实现智能路由
"""
条件边 (Conditional Edges)
----------------------------
条件边根据状态动态决定下一个节点

add_conditional_edges 参数：
1. source_node: 源节点名称
2. condition_function: 条件函数（返回路由键）
3. mapping: 路由键到目标节点的映射

工作原理：
当源节点执行完毕后 -> 调用条件函数 -> 根据返回值选择下一个节点
"""

# db_manager 的条件边
"""
db_manager 执行后的路由逻辑：
- "continue" -> code_generator（数据准备好了，交给代码生成器）
- "call_tool" -> call_tool（需要执行数据库操作）
- END -> 结束（如果 db_manager 判断任务已完成）

典型流程：
User: "查询 sales_id=1 的销售额"
-> db_manager 决定调用 query_sales 工具
-> router 返回 "call_tool"
-> 执行 query_sales
-> 回到 db_manager
-> db_manager 将结果传递给 code_generator
"""
workflow.add_conditional_edges(
    "db_manager",  # 从 db_manager 节点出发
    router,  # 使用 router 函数决定下一步
    {"continue": "code_generator", "call_tool": "call_tool", END: END},  # 路由映射
)

# code_generator 的条件边
"""
code_generator 执行后的路由逻辑：
- "continue" -> db_manager（需要更多数据，返回给数据库管理员）
- "call_tool" -> call_tool（需要执行 Python 代码）
- END -> 结束（代码执行完毕，任务完成）

典型流程：
code_generator 收到数据后
-> 决定生成柱状图代码
-> router 返回 "call_tool"
-> 执行 python_repl
-> 回到 code_generator
-> code_generator 返回 "FINAL ANSWER"
-> router 返回 END
"""
workflow.add_conditional_edges(
    "code_generator",  # 从 code_generator 节点出发
    router,  # 使用 router 函数决定下一步
    {"continue": "db_manager", "call_tool": "call_tool", END: END},  # 路由映射
)

# call_tool 的条件边
"""
call_tool 执行后的路由逻辑：
根据 sender 字段决定返回哪个 Agent

lambda x: x["sender"] 是一个简单的函数，返回状态中的 sender 值

为什么需要这个？
- 工具执行完毕后，需要将结果返回给调用它的 Agent
- sender 字段记录了是哪个 Agent 请求的工具调用
- 这样可以形成：Agent -> Tool -> Agent 的循环

例如：
db_manager 调用 query_sales 
-> sender = "db_manager"
-> 工具执行完毕
-> 根据 sender 返回到 db_manager
-> db_manager 接收工具结果，继续处理
"""
workflow.add_conditional_edges(
    "call_tool",  # 从 call_tool 节点出发
    lambda x: x["sender"],  # 根据 sender 字段路由
    {
        "db_manager": "db_manager",  # 如果是 db_manager 调用的，返回给它
        "code_generator": "code_generator",  # 如果是 code_generator 调用的，返回给它
    },
)

# 设置入口点
"""
设置图的起始节点
用户的输入会首先进入这个节点

为什么选择 db_manager 作为入口？
- 大多数查询需要先获取数据
- db_manager 可以判断是否需要代码生成器
- 如果不需要数据，db_manager 可以直接传递给 code_generator
"""
workflow.set_entry_point("db_manager")

# 编译图
"""
编译 (compile) 操作
---------------------
将定义好的工作流编译成可执行的图

编译做了什么？
1. 验证图的结构（检查是否有孤立节点、死循环等）
2. 优化执行路径
3. 创建可调用的图对象

编译后的 graph 对象可以：
- 使用 invoke() 同步执行
- 使用 stream() 流式执行
- 使用 ainvoke() 异步执行
"""
graph = workflow.compile()


print("=======================可视化图=======================")
"""
图可视化 - 生成工作流结构图
==========================================
LangGraph 提供了强大的可视化功能
可以直观地查看图的结构和节点关系

get_graph() 参数：
- xray=True: 显示详细信息，包括内部节点
- xray=False: 只显示主要节点

draw_mermaid_png(): 生成 Mermaid 格式的 PNG 图片
Mermaid 是一种流程图绘制语法
"""
# 获取图的 Mermaid PNG 表示
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()

# 将 PNG 字节数据写入文件
with open("graph_51.png", "wb") as f:
    f.write(png_bytes)

print("图结构已保存到 graph_51.png")
print("\n图的结构说明：")
print("START -> db_manager -> router -> {code_generator | call_tool | END}")
print("code_generator -> router -> {db_manager | call_tool | END}")
print("call_tool -> {db_manager | code_generator} (根据 sender)")
print("="*60)


print("=======================功能测试=======================")
"""
功能测试 - 验证多代理协作系统
==========================================
这部分展示了如何使用已构建的图来处理不同类型的请求

graph.stream() 方法说明：
- 参数1: 初始状态（包含用户消息）
- 参数2: 配置字典
  * recursion_limit: 最大递归次数（防止无限循环）
- 参数3: stream_mode
  * 'values': 返回每个节点执行后的完整状态
  * 'updates': 只返回状态更新
  
返回值：
- 一个生成器，产生图执行过程中的每个状态快照
"""

print("******************** test01 ********************")
"""
测试用例 1: 数据查询 + 可视化
---------------------------------
用户请求：根据 sales_id 使用折线图显示前5名销售的销售总额

预期流程：
1. db_manager 收到请求
2. db_manager 决定查询前5名销售数据（调用 query_sales 工具）
3. 工具返回数据给 db_manager
4. db_manager 将数据传递给 code_generator
5. code_generator 生成 Python 代码（使用 matplotlib 绘制折线图）
6. 执行 python_repl 工具
7. 代码执行成功，生成图表
8. code_generator 返回 "FINAL ANSWER"
9. 流程结束

这个测试展示了完整的协作流程：数据获取 -> 数据处理 -> 可视化
"""
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="根据sales_id使用折线图显示前5名销售的销售总额")]}, 
#     {"recursion_limit": 50},  # 允许最多50次递归
#     stream_mode='values'):  # 返回完整状态
#     print(chunk)  # 打印每个状态快照

print("******************** test02 ********************")
"""
测试用例 2: 数据删除操作
--------------------------
用户请求：帮我删除销售 id 是 5 的这名销售信息

预期流程：
1. db_manager 收到请求
2. db_manager 识别出需要删除操作
3. db_manager 调用 delete_sale 工具（sales_id=5）
4. 工具执行删除操作
5. 返回删除成功消息
6. db_manager 返回 "FINAL ANSWER"
7. 流程结束

这个测试展示了纯数据库操作，不需要 code_generator 参与
"""
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="帮我删除销售id 是 5 的这名销售信息")]}, 
#     {"recursion_limit": 20}, 
#     stream_mode='values'):
#     print(chunk)

print("******************** test03 ********************")
"""
测试用例 3: 简单数据查询
--------------------------
用户请求：帮我找出数据库中 sales_id 为 18 的销售额是多少

预期流程：
1. db_manager 收到请求
2. db_manager 调用 query_sales 工具（sales_id=18）
3. 工具返回销售记录（包含销售额）
4. db_manager 提取 amount 字段
5. db_manager 返回答案和 "FINAL ANSWER"
6. 流程结束

这个测试展示了简单的查询操作，直接返回结果
"""
# for chunk in graph.stream(
#     {"messages": [HumanMessage(content="帮我找出数据库中sales_id为18的销售额是多少")]}, 
#     {"recursion_limit": 20}, 
#     stream_mode='values'):
#     print(chunk)


print("******************** test04 ********************")
"""
测试用例 4: 复杂查询 + 数据可视化（避免假数据）
-----------------------------------------------
用户请求：帮我根据前10个销售记录id，生成对应的销售额柱状图，不要给我搞模拟的假数据

关键点：
- "不要给我搞模拟的假数据" - 强调使用真实数据库数据
- "前10个销售记录id" - 需要查询 sales_id 1-10
- "柱状图" - 使用 matplotlib 的 bar 图

预期流程：
1. db_manager 收到请求
2. db_manager 循环调用 query_sales（sales_id=1 到 10）
3. 收集所有销售记录的 amount 数据
4. db_manager 将数据传递给 code_generator
5. code_generator 生成 Python 代码：
   - 使用 matplotlib.pyplot
   - 创建柱状图（bar chart）
   - X 轴：sales_id，Y 轴：amount
6. 执行 python_repl 工具
7. 代码执行，生成并显示/保存柱状图
8. code_generator 返回 "FINAL ANSWER"
9. 流程结束

pretty_print() 方法：
- 格式化打印消息内容
- 更易读的输出格式
- 区分不同类型的消息（AI、Human、Tool）
"""
for chunk in graph.stream(
    {"messages": [HumanMessage(content="帮我根据前10个销售记录id，生成对应的销售额柱状图，不要给我搞模拟的假数据")]}, 
    {"recursion_limit": 20},  # 限制最大递归次数
    stream_mode='values'):  # 返回完整状态值
    # 获取最新的消息并格式化打印
    chunk["messages"][-1].pretty_print()

print("\n" + "="*60)
print("测试完成！")
print("="*60)

"""
总结：多代理协作系统的优势
==========================================
1. 职责分离：
   - db_manager 专注于数据操作
   - code_generator 专注于代码执行
   
2. 灵活协作：
   - Agent 之间可以相互传递信息
   - 动态路由根据任务需要选择合适的 Agent
   
3. 可扩展性：
   - 易于添加新的 Agent（如分析师、报告生成器等）
   - 易于添加新的工具
   
4. 容错性：
   - 工具调用失败会返回错误消息
   - Agent 可以尝试替代方案
   
5. 透明性：
   - 可以追踪完整的执行流程
   - 易于调试和优化

潜在改进方向：
1. 添加更多工具（如数据分析、报表生成）
2. 引入记忆机制（记住之前的对话）
3. 添加人类审批节点（关键操作需要确认）
4. 实现并行执行（同时查询多个数据源）
5. 添加错误重试机制
"""


