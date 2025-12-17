from rag.match_keyword import Yinyutl
from rag.llm import chat
from rag.llm_stream import chat as chat_stream
from rag.reranker import rank
from rag.indexing import VecIndex
from rag.indexing_fqa import VecIndex as VecIndexFqa
from rag.radis import QA, Migrator, save_qa
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import time




Yinyutl_Index = VecIndex("children")
Fqa_Index = VecIndexFqa("fqa")

# 运行一次 Migrator 即可
Migrator().run()



def get_num(n):
    result = list(range(0,n,2)) + list(range(n-1,0,-2))
    return result

history = []

async def stream_answer(answer,session_id,quert_type,start_time):
    for char in answer:
        yield {"token": char, "session_id": session_id, "query_type": quert_type}
    yield {"token": "","complete": True,"end_time": time.time()-start_time}

async def stream_llm(prompt, history, session_id, query_type, start_time, update_query):
    chunks = []
  
    stream = await chat_stream(prompt, history)

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            chunks.append(token)
            yield {"token": token, "session_id": session_id, "query_type": query_type}

    QA(query=update_query, answer=''.join(chunks)).save()
    yield {"token": "", "complete": True, "end_time": time.time() - start_time}

async def rag_stream_run(query,history,session_id):
  

    promptquery = f"请修改我的问题：{query}，使其更加清晰明确，如果是查询单词的意思，请直接返回单词，不要掺杂别的话语，如果是查询如何撰写一篇文章，请要求在知识库里面查找模版并加强描述需求，如果要查询文章的范文或有关文章的内容，请直接返回问题。请注意，不要直接回答问题，而是返回修改后的问题。并且只返回修改后的问题，不要返回其他内容。询问单词时，请直接返回单词，其余问题返回修改后的问题。"
    update_query = chat(promptquery, [])

    response = QA.find(QA.query == update_query).all()

    if response :
        
        logger.info(f"缓存：问题：{query}，答案：{response[0].answer}")
        async for chunk in stream_answer(response[0].answer,session_id,'qa cache',time.time()):
            yield chunk
        return

    fqa_docs = Fqa_Index.search(update_query)



    if float(fqa_docs[0].score) >0.91:
        # print(f"FAQ:"+fqa_docs[0].answer)
        logger.info(f"FAQ：问题：{query}，答案：{fqa_docs[0].answer}")
        save_qa(QA, update_query, fqa_docs[0].answer)
        async for chunk in stream_answer(fqa_docs[0].answer,session_id,'faq',time.time()):
            yield chunk
        return
        

    is_search_database = chat(f"请判断我的问题是否需要查询知识库，我的知识库是关于英语作文和单词的，如果查询单词意思或者作文写法以及英语学习相关内容的时候，需要查询知识库，问题如下：{update_query}，如果需要查询知识库，请返回True，否则返回False。不要带多余的语句",history)
    
    prompt = f"请根据我们之前的对话回答问题：{update_query}\n"

    respone_type = 'llm'
    
    if is_search_database.lower() == "true":
        
    
        Yinyutl_rag = Yinyutl()
        
        logger.info(f"修改后的问题：{update_query}")
        

        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(Yinyutl_rag.query, update_query)
            future2 = executor.submit(Yinyutl_Index.search, update_query)
            docs1 = future1.result()
            docs2 = future2.result()

        logger.info(f"关键词召回：{docs1}")
        
     

        logger.info(f"向量召回：{docs2}")


        docs = set(docs1 + docs2)

        docs = sorted(docs, key=lambda x: rank(x.parent, update_query), reverse=True)
        logger.info(f"搜索结果：{docs}")
        

        reference = " ".join([",".join(docs[i].source) + "\n" + docs[i].parent for i in get_num(len(docs))]) 

        prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{update_query}\n答案："

        respone_type = 'database'


    async for chunk in stream_llm(prompt, history, session_id, respone_type, time.time(), update_query):
        # print(chunk)
        yield chunk
   
        


  

