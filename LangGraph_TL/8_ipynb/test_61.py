"""
基于 Neo4j 数据库的图检索脚本（仅检索，不创建图）

连接到已存在的 Neo4j 图数据库进行检索
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

# 调试：打印环境变量
print("="*60)
print("调试信息：检查 Neo4j 环境变量")
print("="*60)
print(f"URL: {repr(os.getenv('url_neo4j'))}")
print(f"Username: {repr(os.getenv('username_neo4j'))}")
print(f"Password: {repr(os.getenv('password_neo4j')[:10] + '...' if os.getenv('password_neo4j') else None)}")
print(f"Database: {repr(os.getenv('database_neo4j'))}")
print("="*60)

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import init_chat_model


# 连接到已存在的 Neo4j 图数据库
graph = Neo4jGraph(
    url=str(os.getenv("url_neo4j")),
    username=str(os.getenv("username_neo4j")),
    password=str(os.getenv("password_neo4j")),
    database=str(os.getenv("database_neo4j"))
)


# 初始化 LLM
llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)

# 创建 GraphCypherQAChain
cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True,
    verbose=True,
    allow_dangerous_requests=True
)


# ===================== 测试检索 =====================

print("\n" + "="*60)
print("测试1：查询小米的创新技术")
print("="*60 + "\n")
result1 = cypher_chain.invoke("小米科技有限责任公司推出了哪些创新技术？")
print(f"\n回答: {result1['result']}")


print("\n" + "="*60)
print("测试2：查询华为的教育合作")
print("="*60 + "\n")
result2 = cypher_chain.invoke("华为技术有限公司与哪些教育机构建立了合作？")
print(f"\n回答: {result2['result']}")


print("\n" + "="*60)
print("测试3：查询数据库中的所有公司")
print("="*60 + "\n")
result3 = cypher_chain.invoke("都有哪些公司在我的数据库中？")
print(f"\n回答: {result3['result']}")


print("\n" + "="*60)
print("测试4：查询苹果公司的产品")
print("="*60 + "\n")
result4 = cypher_chain.invoke("苹果公司推出了哪些产品？")
print(f"\n回答: {result4['result']}")


print("\n" + "="*60)
print("测试5：查询公司间的合作关系")
print("="*60 + "\n")
result5 = cypher_chain.invoke("哪些公司与微软有合作关系？")
print(f"\n回答: {result5['result']}")

