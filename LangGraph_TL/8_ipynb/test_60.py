"""
集成于 LangChain 中的 基于 Neo4j 数据库的 图构建与检索测试

"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)


print("=======================   打印1   =======================")
print("\n\n\n")
# 打开文件，并赋予读取模式 'r'
with open('D:\\agent\langgraph\8_ipynb\company.txt', 'r', encoding="utf-8") as file:
    # 读取文件的全部内容
    content = file.read()
    print(content)


from langchain_core.documents import Document

documents = [Document(page_content=content)]

print("=======================   打印2   =======================")
print("\n\n\n")
print(documents)


# GraphRAG Setup
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model


# 创建图数据库示例
graph = Neo4jGraph(url= str(os.getenv("url_neo4j")),  # 替换为自己的
                  username=str(os.getenv("username_neo4j")),  # 替换为自己的
                  password=str(os.getenv("password_neo4j")), #替换为自己的
                  database=str(os.getenv("database_neo4j")) # 替换为自己的
                  )  


graph_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


# 图转换器配置
graph_transformer = LLMGraphTransformer(
    llm=graph_llm,
    allowed_nodes=["公司", "产品", "技术", "市场", "活动", "合作伙伴"],    # 可以自定义节点
    allowed_relationships=["推出", "参与", "合作", "位于", "开发"],       # 可以自定义关系
)


# graph_transformer = LLMGraphTransformer(llm=graph_llm)

graph_documents = graph_transformer.convert_to_graph_documents(documents)

graph.add_graph_documents(graph_documents)

print("=======================   打印3   =======================")
print("\n\n\n")
print(f"Graph documents: {len(graph_documents)}")
print(f"Nodes from 1st graph doc:{graph_documents[0].nodes}")
print(f"Relationships from 1st graph doc:{graph_documents[0].relationships}")



from langchain.chains import GraphCypherQAChain

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

print("=======================   打印4   =======================")
print("\n\n\n")
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True, # Validate relationship directions
    verbose=True,
    allow_dangerous_requests=True
)
cypher_chain.invoke("小米科技有限责任公司推出了哪些创新技术？")




print("=======================   打印5   =======================")
print("\n\n\n")
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True, # Validate relationship directions
    verbose=True,
    allow_dangerous_requests=True
)
cypher_chain.invoke("华为技术有限公司与哪些教育机构建立了合作？")



print("=======================   打印6   =======================")
print("\n\n\n")
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True, # Validate relationship directions
    verbose=True,
    allow_dangerous_requests=True
)
cypher_chain.invoke("都有哪些公司在我的数据库中？")











