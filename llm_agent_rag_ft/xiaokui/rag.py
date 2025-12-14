from es_table import Word
from llm import chat
#天地出现在哪篇文章里
history = []
while True:
    query = input("请输入问题：")

    docs = Word().query(query)

    # exit()
    #print(docs)
    length = len(docs)
    step = 1 if length % 2 == 0 else 2
    alist = list(range(0, length, 2)) + list(range(length - step, 0, -2))
    reference = " ".join([docs[i].pronunciation + "\n" + docs[i].mean for i in alist])
    prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
    answer = chat(prompt, history)
    print("回答：", answer)
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
