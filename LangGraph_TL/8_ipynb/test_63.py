
"""
创建传统 RAG Agent ，Milvus数据库的构建和检索测试

"""

import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)

print("\n\n\n")
print("=======================   打印1   =======================")
# 打开文件，并赋予读取模式 'r'
with open('D:\\agent\langgraph\8_ipynb\company.txt', 'r', encoding="utf-8") as file:
    # 读取文件的全部内容
    content = file.read()
    print(content)


from langchain_core.documents import Document

documents = [Document(page_content=content)]

print("\n\n\n")
print("=======================   打印2   =======================")
print(documents)



graph_llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 250
chunk_overlap = 40
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
splits = text_splitter.split_documents(documents)

print("\n\n\n")
print("=======================   打印3   =======================")
print(splits)


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_milvus import Milvus


# Add to Milvus (Zilliz Cloud Serverless 只需要 uri 和 token)
vectorstore = Milvus.from_documents(
    documents=splits,
    collection_name="company_rag_milvus",
    embedding=embeddings,
    drop_old=True,  # 删除旧 collection 并重新创建
    connection_args={
        "uri": str(os.getenv("uri")),
        "token": str(os.getenv("token")),
    }
)


from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 提示
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Context: {context} 
    Answer: 
    """,
    input_variables=["question", "document"],
)


# 数据预处理
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# 构建传统的RAG Chain
rag_chain = prompt | graph_llm | StrOutputParser()

# 运行
question = "我的知识库中都有哪些公司信息"

# 构建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 执行检索
docs = retriever.invoke("question")

print("\n\n\n")
print("=======================   打印4   =======================")
print(docs)



generation = rag_chain.invoke({"context": docs, "question": question})
print("\n\n\n")
print("=======================   打印5   =======================")
print(generation)


