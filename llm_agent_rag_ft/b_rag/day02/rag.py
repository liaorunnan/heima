from b_rag.day02.demo_es import Article
from b_rag.day02.llm import chat
#天地出现在哪篇文章里
history = []
while True:
    query = input("请输入问题：")
    docs = Article().query(query)
    #print(docs)
    reference = " ".join([docs[i]['child'] + "\n" + docs[i]['parents'] for i in [0, 2, 4, 3, 1]])
    prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
    answer = chat(prompt, history)
    print("回答：", answer)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
