


"""
将 集成于 LangChain 中的 基于 Milvus 数据库的 “向量检索” 封装成多代理系统中的一个工具

这里的函数供后续集成系统使用，本脚本不运行

"""


def vec_kg(state: AgentState):

    messages = state["messages"][-1]
    
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


    # 构建传统的RAG Chain
    rag_chain = prompt | graph_llm | StrOutputParser()
    # 运行
    question = "我的知识库中都有哪些公司信息"
    
    # 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # 执行检索
    docs = retriever.invoke("question")
    generation = rag_chain.invoke({"context": docs, "question": question})
    
    final_response = [HumanMessage(content=generation, name="vec_kg")]   # 这里要添加名称
    
    return {"messages": final_response}











