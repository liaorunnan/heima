from xiaokui.es_table import Word
from llm import chat
from embedding import get_embedding
from indexing import VecIndex
from ranker import rank
from redis_om import Migrator




Migrator().run()
history = []
while True:
    query = input("请输入问题：")
    prompt = f"请判断用户的问题是否需要搜索知识库，知识库中包含各个年级的教材，如果用户问题需要查询这些课本知识或者查询单个单词的话，请返回true，否则返回false。不要回复其他内容。这是用户的问题：{query}"
    if_need_search = chat(prompt, history)
    print("是否需要搜索知识库：", if_need_search)
    # print("是否需要搜索知识库：", if_need_search)
    if if_need_search.lower() == "true":
        query_vec = get_embedding(query)
        docs1 = VecIndex("word").search(query_vec)
        print("docs1", docs1)
        print("*"*30)
        docs2 = Word.query(query)
        print("docs2", docs2)
        print("*" * 30)
        docs = docs1 + docs2

        print("docs",docs)
        print("*" * 30)
        docs_ranked = sorted(docs, key=lambda x: rank(x.mean + x.pronunciation + x.word, query), reverse=True)



        print("搜索结果：", docs_ranked)
        print("*" * 30)
        length = len(docs_ranked)
        step = 1 if length % 2 == 0 else 2
        alist = list(range(0, length, 2)) + list(range(length - step, 0, -2))
        reference = " ".join([docs_ranked[i].mean + "\n" + docs_ranked[i].pronunciation + "\n" + docs_ranked[i].word  for i in alist])
        prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
    else:
        prompt = query
    answer = chat(prompt, history, system_prompt="你是一个智能英语助手，名叫小文同学，主打英语教育的功能，将用户提到的英语单词翻译成中文,并展示音标和意义")
    print("回答：", answer)
    print("*" * 30)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})