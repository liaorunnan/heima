from re import search
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from langchain_core.tools import tool
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import math

from conf import settings


from rag.match_keyword import Yinyutl
from rag.llm import chat
from rag.reranker import rank
from rag.indexing import VecIndex
from rag.indexing_fqa import VecIndex as VecIndexFqa
from rag.radis import QA, Migrator, save_qa

from prompts import *

logger.add("rag_langchain.log", rotation="500 MB", encoding="utf-8")


Yinyutl_Index = VecIndex("children")
Fqa_Index = VecIndexFqa("fqa")
Yinyutl_rag = Yinyutl()
Migrator().run()

class State(TypedDict):
    messages: List[Dict[str, str]]

def get_percent(num_docs):
    top_percent = 0.20  
    return max(1, math.ceil(num_docs * top_percent))  # 至少选1个


def get_num(n):
    result = list(range(0,n,2)) + list(range(n-1,0,-2))
    return result

#缓存
def search_cache(state:dict):
    query = state["messages"][0]["content"]
    logger.info(f"缓存查询问题: {query}")
    response = QA.find(QA.query == query).all()
    if response:
        logger.info(f"缓存查询结果: {response[0].answer}")
        return {"messages": [{"content": response[0].answer, "from_cache": True}]}
    state["messages"].append({"role": "assistant", "content": "no_cache", "from_cache": False})
    return state

def search_faq(state:dict):
    query = state["messages"][0]["content"]
    fqa_docs = Fqa_Index.search(query)
    logger.info(f"faq查询问题: {query}")
    if float(fqa_docs[0].score) >0.91:
        logger.info(f"faq查询结果: {fqa_docs}")
        save_qa(QA, query, fqa_docs[0].answer)
        state["messages"].append({"role": "assistant", "content": fqa_docs[0].answer, "from_faq": True})
        return state
    state["messages"].append({"role": "assistant", "content": "no_faq", "from_faq": False})
    return state

def route_function(state:dict):
    message = state["messages"][-1]
    update_query = state["messages"][0]["content"]

    if message.get("from_cache", False) or message.get("from_faq", False):
   
        return END

    if "from_cache" in message:
        return "faq"

    if "from_faq" in message:
        return "answer" if llm.invoke(SEARCH_JUDGE_PROMPT.format(update_query=update_query)).content.strip().lower() =='false' else "search"

    if "from_search" in message:
        return "answer_ref"

    return "answer"

def answer_node(state:dict,user_flag:bool):
    query = state["messages"][0]["content"]
    message = state["messages"][-1]
    logger.info(f"查询问题: {query}")
    if all([user_flag, "from_search" in message, (reference := message["content"])]):
        prompt = SMART_SPEAKER_WITH_REFERENCE_PROMPT.format(reference=reference, update_query=query)
        logger.info(f"引用回答问题: {prompt}")
    else:
        prompt = SMART_SPEAKER_PROMPT.format(update_query=query)
        logger.info(f"普通回答问题: {prompt}")
    answer = llm.invoke(prompt)
    QA(query=query, answer=answer.content).save()
    logger.info(f"回答: {answer.content}")
    state["messages"].append({"role": "assistant", "content": answer.content})
    return state

def search_node(state:dict):
    update_query = state["messages"][0]["content"]
    logger.info(f"检索查询问题: {update_query}")
  
    strategy = llm.invoke(STRATEGY_PROMPT.format(update_query=update_query)).content.strip()
    query = update_query
    queries = {"假设问题检索": [llm.invoke(HYDE_PROMPT.format(update_query=update_query)).content],
               "子查询检索": [q.strip() for q in llm.invoke(SUBQUERY_PROMPT.format(update_query=update_query)).content.split("\n") if q.strip()],
               "回溯问题检索": [llm.invoke(BACKTRACKING_PROMPT.format(update_query=update_query)).content]
               }.get(strategy, [update_query])
    logger.info(f"原始问题：{update_query}")
    logger.info(f"检索策略：{strategy}")
    logger.info(f"修改后的问题：{queries}")
    all_docs = set()
    for update_query in queries:
        logger.info(f"修改后的问题：{update_query}")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(Yinyutl_rag.query, update_query)
            future2 = executor.submit(Yinyutl_Index.search, update_query)
            docs1 = future1.result()
            docs2 = future2.result()
        logger.info(f"关键词召回：{docs1}")
        logger.info(f"向量召回：{docs2}")
        docs = set(docs1 + docs2)
        all_docs.update(docs)
    for doc in all_docs:
        doc.score = rank(doc.parent, query)
    docs_ranked = sorted([doc for doc in all_docs if doc.score > 0.7], key=lambda x: x.score, reverse=True)
    reference = "\n\n".join([f"{doc.source}\n{doc.parent}\n\n" for doc in docs_ranked[:get_percent(len(docs_ranked))]]) 
    state["messages"].append({"role": "assistant", "content": reference, "from_search": True})
    return state
    


llm = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)


workflow = StateGraph(State)
workflow.add_node("cache", search_cache)
workflow.add_node("faq", search_faq)
workflow.add_node("search", search_node)
workflow.add_node("answer", lambda s: answer_node(s,user_flag=False))
workflow.add_node("answer_ref", lambda s: answer_node(s,user_flag=True))


workflow.set_entry_point("cache")


workflow.add_conditional_edges("cache", route_function)
workflow.add_conditional_edges("faq", route_function)
workflow.add_conditional_edges("search", route_function)
workflow.add_edge("answer", END)
workflow.add_edge("answer_ref", END)

app = workflow.compile()

if __name__ == '__main__':

    while True:
        query = input("请输入问题：")
        result = app.invoke({"messages": [{"role": "user", "content": query}]})
        


