from b_rag.day02.match_keyword import Law
from b_rag.day02.llm import chat

history=[]
while True:
    query = input("请输入问题：")
    docs=Law.query(query)
    print("找到以下法律条文：",docs)
    length = len(docs)
    alist=list(range(0,length,2)) + list(range(length-1,0,-2))
    reference=" ".join([docs[i].source + "\n" + docs[i].parent for i in alist])
    prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{query}\n答案："
    answer = chat(prompt,history)
    history.append({"role":"user","content":query})
    history.append({"role":"assistant","content":answer})
    print("答案：",answer)