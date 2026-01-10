
"""
将 集成于 LangChain 中的 基于 Neo4j 数据库的 “图检索” 封装成多代理系统中的一个工具

这里的函数供后续集成系统使用，本脚本不运行

"""


from langgraph.graph import StateGraph, MessagesState, START, END

class AgentState(MessagesState):
    next: str

def graph_kg(state: AgentState):
    messages = state["messages"][-1]
    cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True,
    allow_dangerous_requests=True
    )
    response = cypher_chain.invoke(messages.content) 
    final_response = [HumanMessage(content=response["result"], name="graph_kg")]   # 这里要添加名称
    return {"messages": final_response}









