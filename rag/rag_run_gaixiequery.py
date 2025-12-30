from rag.match_keyword import Yinyutl
from rag.llm import chat
from rag.reranker import rank
from rag.indexing import VecIndex
from rag.indexing_fqa import VecIndex as VecIndexFqa
from rag.radis import QA, Migrator, save_qa
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from refine_emb import RefineEmbedding
import math

from rag.prompts import *

import time



Yinyutl_Index = VecIndex("children")
Fqa_Index = VecIndexFqa("fqa")
Yinyutl_rag = Yinyutl()

# 运行一次 Migrator 即可
Migrator().run()

logger.add("rag_gaixie.log", rotation="500 MB", encoding="utf-8")



def get_percent(num_docs):
    top_percent = 0.20  
    return max(1, math.ceil(num_docs * top_percent))  # 至少选1个


def get_num(n):
    result = list(range(0,n,2)) + list(range(n-1,0,-2))
    return result

history = []
while True:
    update_query = input("请输入你的问题：")

    start_time = time.time()  # 开始计时
    print("开始时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

    response = QA.find(QA.query == update_query).all()
    print('2')

    if response :
        # print(response[0].answer)
        logger.info(f"缓存：问题：{update_query}，答案：{response[0].answer}")
        continue
    

    fqa_docs = Fqa_Index.search(update_query)

    

    if float(fqa_docs[0].score) >0.91:
        # print(f"FAQ:"+fqa_docs[0].answer)
        logger.info(f"FAQ：问题：{query}，答案：{fqa_docs[0].answer}")
        save_qa(QA, update_query, fqa_docs[0].answer)
        continue

    print('123')
    
    se_res = chat(SEARCH_JUDGE_PROMPT.format(update_query=update_query), history).lower()

    
    
    if  se_res == "false":
        response = chat(SMART_SPEAKER_PROMPT.format(update_query=update_query), history)
        logger.info(f"普通回复：{response}")
        continue
    
    

    strategy = chat(STRATEGY_PROMPT.format(update_query=update_query), history, system_prompt="你是一个有用的助手，用于判断用户的问题需要采用哪种检索策略。").strip()
    query = update_query
    queries = {
        "假设问题检索": [chat(HYDE_PROMPT.format(update_query=update_query), history)],
        "子查询检索": [q.strip() for q in chat(SUBQUERY_PROMPT.format(update_query=update_query), history).split("\n") if q.strip()],
        "回溯问题检索": [chat(BACKTRACKING_PROMPT.format(update_query=update_query), history)]
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
    

    reference = "".join([f"{doc.source}\n{doc.parent}\n\n" for doc in docs_ranked[:get_percent(len(docs_ranked))]]) 

    print("召回文档的程度：", len(reference))

    reference = RefineEmbedding(reference)

    print("精炼后的参考内容：", reference)
    print("精炼后的参考内容长度：", len(reference))

   
    response = chat(SMART_SPEAKER_WITH_REFERENCE_PROMPT.format(reference=reference, update_query=update_query), history)
    
    history.append({"role": "user", "content": update_query})
    history.append({"role": "assistant", "content": response})

    # save_qa(QA, update_query, response)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"最终回答：{response}")
    logger.info(f"处理耗时：{elapsed_time:.2f} 秒")

    # print(response)





    
