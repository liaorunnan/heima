
from rag.llm import chat
from rag.reranker import rank
from rag.indexing_fqa import VecIndex as VecIndexFqa


Yinyutl_Index = VecIndex("children")
Fqa_Index = VecIndexFqa("fqa")

# 运行一次 Migrator 即可
Migrator().run()



def get_num(n):
    result = list(range(0,n,2)) + list(range(n-1,0,-2))
    return result

history = []
while True:
    query = input("请输入你的问题：")

    promptquery = f"请修改我的问题：{query}，使其更加清晰明确，如果是查询单词的意思，请直接返回单词，不要掺杂别的话语，如果是查询如何撰写一篇文章，请要求在知识库里面查找模版并加强描述需求，如果要查询文章的范文或有关文章的内容，请直接返回问题。请注意，不要直接回答问题，而是返回修改后的问题。并且只返回修改后的问题，不要返回其他内容。询问单词时，请直接返回单词，其余问题返回修改后的问题。"
    update_query = chat(promptquery, [])

    response = QA.find(QA.query == update_query).all()

    if response :
        # print(response[0].answer)
        logger.info(f"缓存：问题：{query}，答案：{response[0].answer}")
        continue

    fqa_docs = Fqa_Index.search(update_query)



    if float(fqa_docs[0].score) >0.91:
        # print(f"FAQ:"+fqa_docs[0].answer)
        logger.info(f"FAQ：问题：{query}，答案：{fqa_docs[0].answer}")
        save_qa(QA, update_query, fqa_docs[0].answer)
        continue

    is_search_database = chat(f"请判断我的问题是否需要查询知识库，我的知识库是关于英语作文和单词的，如果查询单词意思或者作文写法以及英语学习相关内容的时候，需要查询知识库，问题如下：{update_query}，如果需要查询知识库，请返回True，否则返回False。不要带多余的语句",history)