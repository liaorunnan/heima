
from b_rag.day02.llm import chat
from b_rag.day02.match_keyword import Law
from b_rag.day03.embedding import get_embedding
from b_rag.day03.indexing import VecIndex
from b_rag.day04.faq.indexing import VecIndex as FAQVecIndex

from b_rag.day03.reranker import rank
from b_rag.day04.cache.cache import QA

history = []
while True:
    query = input("请输入问题：")
    results = QA.find(QA.query == query).all()
    answer = None
    prompt = query
    if results:
        answer = results[0].answer
        print("缓存回答：", answer)
    else:
        query_vec = get_embedding(query)
        query = FAQVecIndex("faq").search(query_vec, topk=1)[0]
        # 分数大于0.95，直接返回
        if query.score > 0.82:
            print("FAQ回答：", query.answer)
        else:
            _prompt = f"请判断用户的问题是否需要搜索知识库，知识库中包含各个法律的内容，如果用户问题需要这些法律知识，请返回true，否则返回false。不要回复其他内容。这是用户的问题：{query}"
            if_need_search = chat(_prompt, history)
            print("是否需要搜索知识库：", if_need_search)
            if if_need_search.lower() == "true":
                docs1 = VecIndex("law").search(query_vec)
                docs2 = Law.query(query)
                docs = set(docs1 + docs2)
                for doc in docs:
                    doc.score = rank(doc.embedding_text, query)
                filtered_docs = [doc for doc in docs if doc.score > 0.7]
                docs_ranked = sorted(filtered_docs, key=lambda x: x.score, reverse=True)
                if  docs_ranked:
                    print("搜索结果：", docs_ranked)
                    length = len(docs_ranked)
                    alist = list(range(0, length, 2)) + list(range(length - 1, 0, -2))
                    reference = " ".join([docs_ranked[i].source + "\n" + docs_ranked[i].embedding_text for i in alist])
                    prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
            answer = chat(prompt, history,system_prompt="你是一个ai法律咨询师，主打解决用户法律疑问的功能，只说你的回答")
            QA(query=query, answer=answer).save()
            print("回答：", answer)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
