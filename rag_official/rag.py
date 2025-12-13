"""
正式 RAG 系统 - 混合检索 + 重排序
功能：
  1. LLM 判断是否需要检索知识库
  2. 双路召回：向量检索（Milvus）+ 关键字检索（Elasticsearch）
  3. 重排序：使用 BGE-reranker 对召回结果精排
  4. 支持法律条文和裁判文书两种模式
"""

from data_process.insert_to_es import ESLawChunk, ESWritChunk
from rag_official.embedding import get_embedding
from rag_official.indexing import LawVecIndex, WritVecIndex
from rag_official.reranker import rank
from rag_simple.llm import chat
from rag_official.cache import QA


def choose_mode():
    """
    选择检索模式
    
    返回:
        str: "1" 表示法律条文模式，"2" 表示裁判文书模式，"q" 表示退出
    """
    while True:
        mode = input("请选择模式：1) 法律相关问答  2) 起诉状/答辩状文书  (输入q退出)：")
        if mode in ("1", "2", "q", "quit"):
            return mode
        print("请输入 1 / 2 / q")


def check_need_search(query, history):
    """
    使用 LLM 判断用户问题是否需要检索知识库
    
    参数:
        query: 用户问题
        history: 对话历史
    
    返回:
        bool: True 表示需要检索，False 表示不需要
    """
    prompt = (
        f"请判断用户的问题是否需要搜索法律知识库，知识库中包含各类法律条文和裁判文书。"
        f"如果用户问题需要这些法律知识，请返回 true，否则返回 false。"
        f"不要回复其他内容。这是用户的问题：{query}"
    )
    
    response = chat(prompt, history)
    print(f"[检索判断] {response}")
    
    # 判断响应中是否包含 true（不区分大小写）
    return "true" in response.lower()


# 法律法规的混合索引
def hybrid_search_law(query, vec_topk=10, es_topk=10):
    """
    混合检索：法律条文模式
    结合向量检索和关键字检索，召回更全面的结果
    
    参数:
        query: 用户查询
        vec_topk: 向量检索返回数量
        es_topk: ES 关键字检索返回数量
    
    返回:
        List[LawItem]: 去重后的检索结果列表
    """
    # 1. 向量检索（语义相似度）
    query_vec = get_embedding(query)
    vec_results = LawVecIndex().search(query_vec, topk=vec_topk)
    print(f"[向量召回] 召回 {len(vec_results)} 条法律条文")
    
    # 2. 关键字检索（精确匹配）
    es_results = ESLawChunk.query(query, size=es_topk)
    print(f"[关键字召回] 召回 {len(es_results)} 条法律条文")
    
    # 3. 合并去重（根据 id）
    # seen_ids = set()
    # merged_results = []
    #
    # for item in vec_results + es_results:
    #     if item.id not in seen_ids:
    #         seen_ids.add(item.id)
    #         merged_results.append(item)
    merged_results = list(set(vec_results + es_results))
    
    print(f"[合并去重] 共 {len(merged_results)} 条候选结果")
    return merged_results


# 文书的混合索引
def hybrid_search_writ(query, vec_topk=10, es_topk=10):
    """
    混合检索：裁判文书模式
    结合向量检索和关键字检索
    
    参数:
        query: 用户查询
        vec_topk: 向量检索返回数量
        es_topk: ES 关键字检索返回数量
    
    返回:
        List[WritItem]: 去重后的检索结果列表
    """
    # 1. 向量检索
    query_vec = get_embedding(query)
    vec_results = WritVecIndex().search(query_vec, topk=vec_topk)
    print(f"[向量召回] 召回 {len(vec_results)} 条裁判文书")
    
    # 2. 关键字检索
    es_results = ESWritChunk.query(query, size=es_topk)
    print(f"[关键字召回] 召回 {len(es_results)} 条裁判文书")
    
    # 3. 合并去重
    # seen_ids = set()
    # merged_results = []
    #
    # for item in vec_results + es_results:
    #     if item.id not in seen_ids:
    #         seen_ids.add(item.id)
    #         merged_results.append(item)

    merged_results = list(set(vec_results + es_results))
    
    print(f"[合并去重] 共 {len(merged_results)} 条候选结果")
    return merged_results


def rerank_results(query, docs, mode, top_n=5):
    """
    使用重排序模型对检索结果进行精排
    
    参数:
        query: 用户查询
        docs: 检索到的文档列表（LawItem 或 WritItem）
        mode: "1" 表示法律模式，"2" 表示文书模式
        top_n: 返回前 N 个结果
    
    返回:
        List: 重排序后的前 N 个文档
    """
    if not docs:
        return []
    
    # 根据模式提取文本内容
    if mode == "1":
        texts = [doc.embedding_text for doc in docs]
    else:
        texts = [doc.context for doc in docs]
    
    # 计算每个文档与查询的相关性得分
    scored_docs = []
    for i, (doc, text) in enumerate(zip(docs, texts)):
        score = rank(query, text)
        scored_docs.append((doc, score))
        if i < 3:  # 打印前 3 个得分用于调试
            print(f"  [{i+1}] 重排序得分: {score:.4f}")
    
    # 按得分降序排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前 N 个文档
    top_docs = [doc for doc, score in scored_docs[:top_n]]
    print(f"[重排序] 从 {len(docs)} 条结果中选得分最高的 {len(top_docs)} 条")
    
    return top_docs


def build_reference_law(docs):
    """
    构建法律条文的参考文本
    采用交错策略：优先级高的文档穿插在不同位置
    
    参数:
        docs: 重排序后的 LawItem 列表
    
    返回:
        str: 格式化的参考文本
    """
    if not docs:
        return ""
    
    parts = []
    for i, doc in enumerate(docs, 1):
        # 优先使用 content_text，其次 embedding_text
        text = getattr(doc, "content_text", None) or getattr(doc, "embedding_text", "")
        if text:
            parts.append(f"【参考{i}】{text}")
    
    # 交错排列：[0, 2, 4, 3, 1] 的策略
    length = len(parts)
    step = 2 if length % 2 !=0 else 1
    indices = list(range(0, length, 2)) + list(range(length - step, 0, -2))
    reordered = [parts[i] for i in indices if i < length]
    
    return "\n\n".join(reordered)


def build_reference_writ(docs):
    """
    构建裁判文书的参考文本
    
    参数:
        docs: 重排序后的 WritItem 列表
    
    返回:
        str: 格式化的参考文本
    """
    if not docs:
        return ""
    
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"【案例{i}：{doc.indexbytitle}】\n{doc.context}")
    
    return "\n\n".join(parts)


def main():
    """
    主函数：启动交互式 RAG 问答系统
    """
    print("=" * 60)
    print("欢迎使用法律 RAG 问答系统（正式版）")
    # print("特性：LLM 智能判断 + 混合检索 + 重排序")
    print("=" * 60)
    
    history = []
    mode = choose_mode()
    
    if mode in ("q", "quit"):
        return

    while True:
        print("\n" + "-" * 60)
        query = input("请输入问题（输入 q/quit 退出，输入 switch 切换模式）: ")
        
        if query in ("q", "quit"):
            print("感谢使用，再见！")
            break
        
        if query == "switch":
            mode = choose_mode()
            if mode in ("q", "quit"):
                print("感谢使用，再见！")
                break
            continue
        
        # 1. 先查缓存，命中直接返回
        cached = QA.find(QA.query == query).all()
        answer = None

        if cached:
            answer = cached[0].answer
            print("\n[缓存命中] 直接返回缓存回答")
        else:
            # 2. 判断是否需要检索知识库
            need_search = check_need_search(query, history)
            
            if need_search:
                print("\n[开始检索知识库]")
                
                # 2.1 混合检索
                if mode == "1":
                    docs = hybrid_search_law(query, vec_topk=10, es_topk=10)
                else:
                    docs = hybrid_search_writ(query, vec_topk=10, es_topk=10)
                
                if not docs:
                    print("未检索到相关内容，将直接回答")
                    prompt = query
                else:
                    # 2.2 重排序
                    print("\n[重排序]")
                    top_docs = rerank_results(query, docs, mode, top_n=5)
                    
                    # 2.3 构建参考文本
                    if mode == "1":
                        reference = build_reference_law(top_docs)
                    else:
                        reference = build_reference_writ(top_docs)
                    
                    print(f"\n[参考资料]\n{reference[:300]}..." if len(reference) > 300 else f"\n[参考资料]\n{reference}")
                    
                    # 2.4 构建最终 prompt
                    prompt = f"请根据以下参考内容回答问题：\n\n{reference}\n\n问题：{query}\n\n答案："
            else:
                print("\n[无需检索] 直接使用 LLM 回答")
                prompt = query
            
            # 3. 生成回答并写入缓存
            print("\n[生成回答中...]")
            answer = chat(prompt, history)
            QA(query=query, answer=answer).save()
            print("[缓存写入完成]")
        
        print(f"\n【回答】\n{answer}")
        
        # 4. 更新对话历史
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
