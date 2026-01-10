
"""
构建混合知识库检索多代理系统
中文提示词

不再使用 with_structured_output 来解析主管输出的下一个节点，而是使用 json.loads 来解析，因为前者错误了太高了

当前脚本，8个测试全部通过，使用的模型是 qwen3-max
测试了deepseek-chat，同样8个测试也都通过了（有时候第4个会出错）

"""



import os
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_community.graphs import Neo4jGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from typing import Literal
from faker import Faker
import json
import random
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_community.chat_models.tongyi import ChatTongyi

# llm = init_chat_model(
#         model="deepseek-chat",
#         model_provider="deepseek", # 该参数可选，留空时自动推断
#         api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
#         base_url=os.getenv("DEEPSEEK_URL"),
#         temperature=0,
#     )
llm = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    temperature=0,
)

# llm = init_chat_model(
#         model="gpt-4o",
#         model_provider="openai",  # 使用 OpenAI 的 GPT-4o
#         api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取
#         base_url=os.getenv("OPENAI_BASE_URL"),
#         temperature=0,
#     )

print("=======================MySQL数据库相关配置=======================")
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



print("--------------------------向MySQL数据库插入/更新模拟数据-----------------------------")
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
    """向数据库新增一条销售记录。"""
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
    """从数据库中删除指定销售记录。"""
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
    """更新数据库中的销售记录。"""
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
    """查询数据库中的销售记录。"""
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

# matplotlib 初始化代码：使用 Agg 后端避免线程问题，图表保存为文件
MATPLOTLIB_INIT_CODE = """
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 Tkinter 线程问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
"""

@tool
def python_repl(
    code: Annotated[str, "执行以生成图表或输出结果的 Python 代码。"],
):
    """使用该工具运行 Python 代码；如需查看变量输出，请显式调用 `print(...)`，结果会展示给用户。"""
    try:
        # 先执行 matplotlib 初始化，再执行用户代码
        full_code = MATPLOTLIB_INIT_CODE + "\n" + code
        result = repl.run(full_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str



print("=======================所有节点构建=======================")
class AgentState(MessagesState):
    next: str


print("--------------------------milvus检索节点构建--------------------------")
# 初始化 Embeddings（必须与创建时使用的相同）
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
# 连接到已存在的 Milvus collection（不创建新的）
vectorstore = Milvus(
    collection_name="company_rag_milvus",
    embedding_function=embeddings,
    connection_args={
        "uri": str(os.getenv("uri")),
        "token": str(os.getenv("token")),
    }
)
def vec_kg(state: AgentState):

    messages = state["messages"][-1]
    prompt = PromptTemplate(
        template="""你是一名问答助手。
        请使用下面检索到的上下文回答问题；如果不知道答案，就直接说明不知道。
        回答最多三句话，并保持简洁：
        问题：{question}
        上下文：{context}
        回答：
        """,
        input_variables=["question", "context"],
    )
    # 构建传统的RAG Chain
    rag_chain = prompt | llm | StrOutputParser()
    # 运行
    question = messages.content
    
    # 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # 执行检索
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    
    final_response = [HumanMessage(content="vec_kg回答结果："+ generation, name="vec_kg")]   # 这里要添加名称
    
    return {"messages": final_response}


print("--------------------------neo4j检索节点构建--------------------------")
# 连接到已存在的 Neo4j 图数据库
graph = Neo4jGraph(
    url=str(os.getenv("url_neo4j")),
    username=str(os.getenv("username_neo4j")),
    password=str(os.getenv("password_neo4j")),
    database=str(os.getenv("database_neo4j"))
)
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True, # Validate relationship directions
    verbose=True,
    allow_dangerous_requests=True
)
def graph_kg(state: AgentState):
    messages = state["messages"][-1]
    response = cypher_chain.invoke(messages.content) 
    final_response = [HumanMessage(content="graph_kg回答结果："+ response["result"], name="graph_kg")]   # 这里要添加名称
    return {"messages": final_response}


print("--------------------------MySQL数据库操作节点构建--------------------------")
db_agent = create_react_agent(
    llm, 
    tools=[add_sale, delete_sale, update_sale, query_sales], 
    state_modifier="你负责执行数据库的增删改查任务，并向 code_node 提供准确的数据。"
)
def db_node(state: AgentState):
    result = db_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [
            HumanMessage(content="db_node回答结果："+ result["messages"][-1].content, name="db_node")
        ]
    }


print("--------------------------python代码创建并执行节点构建--------------------------")
code_agent = create_react_agent(
    llm, 
    tools=[python_repl], 
    state_modifier="通过运行 Python 代码来绘制图表或输出执行结果。绘图时请使用 plt.savefig('output.png') 保存图表，不要使用 plt.show()。"
)
def code_node(state: AgentState):
    result = code_agent.invoke({"messages": state["messages"]})
    return {
        "messages": [HumanMessage(content="code_node回答结果："+result["messages"][-1].content, name="code_node")]
    }


print("--------------------------聊天节点构建--------------------------")
def chat(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content="chat回答结果："+ model_response.content, name="chat")]
    return {"messages": final_response}


print("--------------------------主管节点构建--------------------------")
members = ["graph_kg", "vec_kg", "db_node", "code_node", "chat"]
options = ["graph_kg", "vec_kg", "db_node", "code_node", "chat", "FINISH"]

class Router(TypedDict):
    """用于指派下一位成员；若无需继续执行，则返回 FINISH。"""

    next: Literal[*options]

def supervisor(state: AgentState):
    system_prompt = (
        "你是一个主管，负责协调以下成员之间的对话："
        f"{members}。\n\n"
        "每位成员的职责如下：\n"
        "- chat：用自然语言直接回应用户输入。\n"
        "- graph_kg：图数据库检索，基于图谱知识库保存市场与公司信息，擅长回答宏观、全面的问题。当问题是宏观问题时，可选用此节点\n"
        "- vec_kg：向量数据库检索，基于语义检索知识库保存市场与公司信息，擅长回答细节类问题。当问题是细节问题时，可选用此节点\n"
        "- db_node：关系型数据库操作，执行销售数据的增删改查等数据库操作，当问题是关于销售数据时，可选用此节点\n"
        "- code_node：生成并运行 Python 代码以绘图或输出执行结果，当问题是关于绘图或输出执行结果时，可选用此节点\n\n"
        "必须遵循以下规则：\n"
        "1. 你只能从下列列表中选择下一位成员："
        f"{options}。\n"
        "2. 输出必须是严格的 JSON，格式为 {\"next\": \"<选中的成员>\"}。\n"
        "3. 不要输出任何解释性文本；若无法判断，请返回 {\"next\": \"chat\"} 作为兜底。\n"
        "4. 你必须返回一个 JSON 对象，其中只有一个键 'next'，其值必须是以下之一：chat、graph_kg、vec_kg、db_node、code_node、FINISH。 \n"
        "5. 绝不能连续两次选择同一位工作人员。如果上一条回复来自 “graph_kg”，则不能再次选择 “graph_kg” \n"
        "6. 当用户请求有答案或无法满足时，请选择 FINISH。\n"
        "7. 仔细分析对话历史后再做出决定。\n"
        "8. 有的任务需要多个工作人员协同串行完成，请根据任务需求选择合适的工作人员。但每次输出必须选择一个工作人员，不能同时选择多个工作人员。\n"
        "9. 输出必须是严格的 JSON，要纯json格式，不要加```json 标记，格式为 {\"next\": \"<选中的成员>\"}，候选成员有：chat、graph_kg、vec_kg、db_node、code_node、FINISH，只能选一个\n"
    )
    # print("=====消息记录======", state["messages"])
    messages = [{"role": "system", "content": system_prompt},] + state["messages"]
    raw_resp = llm.invoke(messages)
    print("Supervisor raw:", raw_resp.content)
    # response = llm.with_structured_output(Router).invoke(messages)
    response = json.loads(raw_resp.content)
    if not response or response.get("next") is None:
       raise ValueError(f"Supervisor 未返回有效 next，原始响应：{response}")

    next_ = response["next"]
    
    if next_ == "FINISH":
        next_ = END
    
    return {"next": next_}



print("=====================图构建=======================")
builder = StateGraph(AgentState)

# builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor)
builder.add_node("chat", chat)
builder.add_node("db_node", db_node)
builder.add_node("code_node", code_node)
builder.add_node("graph_kg", graph_kg)
builder.add_node("vec_kg", vec_kg)


for member in members:
    # 我们希望我们的工人在完成工作后总是向主管“汇报”
    builder.add_edge(member, "supervisor")


builder.add_conditional_edges("supervisor", lambda state: state["next"])

# 添加开始和节点
builder.add_edge(START, "supervisor")

# 编译图
graph = builder.compile()

print("=======================可视化图=======================")
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_66.png", "wb") as f:
    f.write(png_bytes)


print("=====================测试图=======================")



print("\n\n\n")
print("*********************** 第1次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="都有哪些公司在我的向量数据库中。")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第2次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="你好，请你介绍一下你自己")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第3次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="你好，帮我生成一个二分查找的Python代码")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# 失败，当需要执行多个节点完成任务时，deepseek不能在第一次按要求只输出一个节点
print("\n\n\n")
print("*********************** 第4次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="帮我根据前10名的销售记录，id为1-10，可以逐个查询或批量查询，生成对应的销售额柱状图")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第5次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="帮我删除 第 33条销售数据")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第6次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="小米科技有限责任公司推出了哪些创新技术？")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第7次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="华为技术有限公司与哪些教育机构建立了合作？")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("\n\n\n")
print("*********************** 第8次测试 ***********************")
for chunk in graph.stream({"messages": [HumanMessage(content="苹果公司推出了哪些产品？")]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

"""
# 防止重复输出版本
last_id = None
for chunk in graph.stream({"messages": [HumanMessage(content="你好，请你介绍一下你自己")]}, stream_mode="values"):
    msg = chunk["messages"][-1]
    if id(msg) == last_id:  # 或比较 content/name/timestamp
        continue
    last_id = id(msg)
    msg.pretty_print()

"""