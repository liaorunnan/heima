from rag.match_keyword import Yinyutl
from rag.llm import chat
from rag.reranker import rank
from rag.indexing import VecIndex
from rag.indexing_fqa import VecIndex as VecIndexFqa
from rag.radis import QA, Migrator, save_qa
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

Yinyutl_Index = VecIndex("children")
Fqa_Index = VecIndexFqa("fqa")

# 运行一次 Migrator 即可，但确保在模块加载时运行一次
Migrator().run()

def get_num(n):
    """辅助函数，用于调整结果顺序"""
    result = list(range(0, n, 2)) + list(range(n-1, 0, -2))
    return result

def rag_query(query: str, history: list = None,return_doc=False) -> str:


    """
    处理单个RAG查询，返回回答。
    :param query: 用户查询字符串
    :param history: 对话历史列表，每个元素为{"role": "user" or "assistant", "content": "..."}
    :return: 回答字符串
    """
    if history is None:
        history = []
    
    # 第一步：优化查询
    promptquery = f"请修改我的问题：{query}，使其更加清晰明确，如果是查询单词的意思，请直接返回单词，不要掺杂别的话语，如果是查询如何撰写一篇文章，请要求在知识库里面查找模版并加强描述需求，如果要查询文章的范文或有关文章的内容，请直接返回问题。请注意，不要直接回答问题，而是返回修改后的问题。并且只返回修改后的问题，不要返回其他内容。询问单词时，请直接返回单词，其余问题返回修改后的问题。"
    update_query = chat(promptquery, [])

    # 第二步：检查缓存
    # response = QA.find(QA.query == update_query).all()
    # if response:
    #     return response[0].answer

    # 第三步：FAQ检索
    # fqa_docs = Fqa_Index.search(update_query)
    # if fqa_docs and float(fqa_docs[0].score) > 0.91:
    #     answer = f"FAQ:" + fqa_docs[0].answer
    #     save_qa(QA, update_query, answer)
    #     return answer

    # 第四步：判断是否需要查询知识库
    is_search_database = chat(f"请判断我的问题是否需要查询知识库，我的知识库是关于英语作文和单词的，如果查询单词意思或者作文写法以及英语学习相关内容的时候，需要查询知识库，问题如下：{update_query}，如果需要查询知识库，请返回True，否则返回False。不要带多余的语句", history)
    
    prompt = f"请根据我们之前的对话回答问题：{update_query}\n"

    docs=''
    
    if is_search_database.lower() == "true":
        Yinyutl_rag = Yinyutl()
        
        # 并行检索
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(Yinyutl_rag.query, update_query)
            future2 = executor.submit(Yinyutl_Index.search, update_query)
            docs1 = future1.result()
            docs2 = future2.result()

        docs = set(docs1 + docs2)
        docs = sorted(docs, key=lambda x: rank(x.parent, update_query), reverse=True)
        
        # 构建参考内容
        reference = " ".join([",".join(docs[i].source) + "\n" + docs[i].parent for i in get_num(len(docs))]) 
        prompt = f"请根据以下参考内容回答问题：\n{reference}\n问题：{update_query}\n答案："

    # 第五步：生成最终回答
    response = chat(prompt, history)
    
    # 更新历史
    history.append({"role": "user", "content": update_query})
    history.append({"role": "assistant", "content": response})
    
    # 保存到缓存
    save_qa(QA, update_query, response)

    if return_doc:
        return response, docs
    else:
    
        return response,''

if __name__ == "__main__":
    # 测试代码
    test_query = "如何写一篇英语作文？"
    answer = rag_query(test_query)
    print(f"Query: {test_query}")
    print(f"Answer: {answer}")