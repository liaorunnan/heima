from b_rag.day02.match_keyword import Law
from b_rag.day02.llm import chat
from b_rag.day03.embedding import get_embedding
from b_rag.day03.indexing import VecIndex
from b_rag.day03.reranker import rank

history = []
while True:
    query = input("请输入问题：")
    prompt = f"请判断用户的问题是否需要搜索知识库，知识库中包含各个法律的内容，如果用户问题需要这些法律知识，请返回true，否则返回false。不要回复其他内容。这是用户的问题：{query}"
    if_need_search = chat(prompt, history)
    print("是否需要搜索知识库：", if_need_search)
    if if_need_search.lower() == "true":
        query_vec = get_embedding(query)
        docs1 = VecIndex("laws").search(query_vec)
        docs2 = Law.query(query)
        docs = docs1 + docs2
        docs_ranked = sorted(docs, key=lambda x: rank(query,x.embedding_text), reverse=True)
        print("搜索结果：", docs_ranked)
        alist=list(range(0, len(docs_ranked), 2)) + list(range(len(docs_ranked) - 1, 0, -2))
        reference = " ".join([docs_ranked[i].source + "\n" + docs_ranked[i].parent for i in alist])
        prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
    else:
        prompt = query
    answer = chat(prompt, history)
    print("回答：", answer)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
