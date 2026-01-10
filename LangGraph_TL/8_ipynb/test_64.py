
"""
Milvus 向量数据库检索脚本（仅检索，不创建）

连接到已存在的 collection 进行检索
"""

import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv(override=True)

# 调试：打印环境变量
print("="*60)
print("调试信息：检查环境变量")
print("="*60)
print(f"URI: {repr(os.getenv('uri'))}")
print(f"Token: {repr(os.getenv('token')[:20] + '...' if os.getenv('token') else None)}")
print("="*60)

# 初始化 LLM
graph_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_URL"),
    temperature=0,
)

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

# 构建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG 提示模板
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Please answer in Chinese.
    
    Question: {question} 
    Context: {context} 
    Answer: 
    """,
    input_variables=["question", "context"],
)

# 数据预处理
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG Chain
rag_chain = prompt | graph_llm | StrOutputParser()


# ===================== 测试检索 =====================

print("\n" + "="*60)
print("测试1：检索知识库中的公司信息")
print("="*60)

question1 = "我的知识库中都有哪些公司信息"
docs1 = retriever.invoke(question1)
print(f"\n问题: {question1}")
print(f"\n检索到 {len(docs1)} 个文档:")
for i, doc in enumerate(docs1):
    print(f"\n--- 文档 {i+1} ---")
    print(doc.page_content[:200] + "...")

# 使用 RAG Chain 回答
context1 = format_docs(docs1)
answer1 = rag_chain.invoke({"question": question1, "context": context1})
print(f"\n回答: {answer1}")


print("\n" + "="*60)
print("测试2：查询小米的技术创新")
print("="*60)

question2 = "小米有哪些技术创新"
docs2 = retriever.invoke(question2)
print(f"\n问题: {question2}")
print(f"\n检索到 {len(docs2)} 个文档:")
for i, doc in enumerate(docs2):
    print(f"\n--- 文档 {i+1} ---")
    print(doc.page_content[:200] + "...")

context2 = format_docs(docs2)
answer2 = rag_chain.invoke({"question": question2, "context": context2})
print(f"\n回答: {answer2}")


print("\n" + "="*60)
print("测试3：查询华为的合作伙伴")
print("="*60)

question3 = "华为与哪些大学有合作关系"
docs3 = retriever.invoke(question3)
print(f"\n问题: {question3}")
print(f"\n检索到 {len(docs3)} 个文档:")
for i, doc in enumerate(docs3):
    print(f"\n--- 文档 {i+1} ---")
    print(doc.page_content[:200] + "...")

context3 = format_docs(docs3)
answer3 = rag_chain.invoke({"question": question3, "context": context3})
print(f"\n回答: {answer3}")
