from b_rag.xiaokui.cache.cache import QA
from b_rag.xiaokui.process.indexing import VecIndex
from b_rag.xiaokui.process.es_query import search_documents

from b_rag.xiaokui.faq.indexing import VecIndex as FAQVecIndex
from b_rag.xiaokui.tool.embedding import get_embedding
from b_rag.xiaokui.tool.llm import chat
from b_rag.xiaokui.tool.reranker import rank

question = "英语单词word的信息是什么鬼意思？？"
history = []
while True:
    query = input("请输入问题：")

    results = QA.find(QA.query == query).all() # 读取缓存
    answer = None
    if results:
        answer = results[0].answer
        print("回答：", answer)
    else:
        query_vec = get_embedding(query)
        match_query = FAQVecIndex("faq").search(query_vec, topk=1)[0]
        if match_query.score > 0.82:
            answer = match_query.answer
            print("FAQ回答：", answer)
        else:
            prompt = f"请判断用户的问题是否需要搜索知识库，知识库中包含英语单词，音标，词性和中文解释，如果用户问题需要这些，请返回true，否则返回false。不要回复其他内容。这是用户的问题：{query}"
            if_need_search = chat(prompt, history)
            print("是否需要搜索知识库：", if_need_search)
            if if_need_search.lower() == "true":
                query_vec = get_embedding(query)
                docs1 = VecIndex("word").search(query_vec)
                # print("搜索结果1：", docs1)

                # docs2 = Book.query(query)
                docs2 = search_documents(query)
                # print("搜索结果2：", docs2)


                docs = docs1 + docs2
                for doc in docs:
                    print(doc)
                    doc.score = rank(doc.meaning, query)
                    print("搜索结果：", doc.score)


                filtered_docs = [doc for doc in docs if doc.score > 0.5]
                print("筛选后搜索结果：", filtered_docs)
                # print(docs)
                # docs_ranked = sorted(docs, key=lambda x: rank(x['score'], query), reverse=True)
                docs_ranked = sorted(filtered_docs, key=lambda x: rank(x.meaning, query), reverse=True)
                # print("搜索结果：", docs_ranked)

                length = len(docs_ranked)
                step = 1 if length % 2 == 0 else 2
                alist = list(range(0,length,2)) + list(range(length-step,0,-2))
                reference = " ".join([docs_ranked[i].pronunciation + "\n" + docs_ranked[i].meaning for i in alist])
                # reference = docs_ranked[0].meaning
                prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
            else:
                prompt = query
            answer = chat(prompt, history)
            print("回答：", answer)
            QA(query=query, answer=answer).save()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
