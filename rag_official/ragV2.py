"""
RAG V2 - 增强版检索流程
特性：
  1) Redis 缓存命中直接返回
  2) FAQ 相似问答命中：Milvus 中 FAQ 向量检索，分数>0.8 直接返回答案
  3) 正常 RAG 流程：关键词 + 向量召回，并对得分进行阈值过滤
"""

from data_process.insert_to_es import ESLawChunk, ESWritChunk
from rag_official.embedding import get_embedding
from rag_official.indexing import LawVecIndex, WritVecIndex, QAVecIndex
from rag_official.reranker import rank
from rag_simple.llm import chat
from rag_official.cache import QA


def choose_mode():
    """选择检索模式"""
    while True:
        mode = input("请选择模式：1) 法律相关问答  2) 起诉状/答辩状文书  (输入q退出)：")
        if mode in ("1", "2", "q", "quit"):
            return mode
        print("请输入 1 / 2 / q")


def check_need_search(query, history):
    """判断是否需要检索知识库"""
    prompt = (
        f"请判断用户的问题是否需要搜索法律知识库，知识库中包含各类法律条文和裁判文书。"
        f"如果用户问题需要这些法律知识，请返回 true，否则返回 false。"
        f"不要回复其他内容。这是用户的问题：{query}"
    )
    response = chat(prompt, history)
    print(f"[检索判断] {response}")
    return "true" in response.lower()


# 混合检索：法律条文
def hybrid_search_law(query, vec_topk=10, es_topk=10, score_threshold=0):
    # 向量召回
    query_vec = get_embedding(query)
    vec_results = LawVecIndex().search(query_vec, topk=vec_topk)
    # 过滤得分
    vec_results = [item for item in vec_results if getattr(item, "score", 0) >= score_threshold]
    print(f"[向量召回] 过滤后 {len(vec_results)} 条法律条文")

    # 关键词召回
    es_results = ESLawChunk.query(query, size=es_topk)
    es_results = [item for item in es_results if getattr(item, "score", 0) >= score_threshold]
    print(f"[关键字召回] 过滤后 {len(es_results)} 条法律条文")

    merged_results = list(set(vec_results + es_results))
    print(f"[合并去重] 共 {len(merged_results)} 条候选结果")
    return merged_results


# 混合检索：裁判文书
def hybrid_search_writ(query, vec_topk=10, es_topk=10, score_threshold=0):
    query_vec = get_embedding(query)
    vec_results = WritVecIndex().search(query_vec, topk=vec_topk)
    vec_results = [item for item in vec_results if getattr(item, "score", 0) >= score_threshold]
    print(f"[向量召回] 过滤后 {len(vec_results)} 条裁判文书")

    es_results = ESWritChunk.query(query, size=es_topk)
    es_results = [item for item in es_results if getattr(item, "score", 0) >= score_threshold]
    print(f"[关键字召回] 过滤后 {len(es_results)} 条裁判文书")

    merged_results = list(set(vec_results + es_results))
    print(f"[合并去重] 共 {len(merged_results)} 条候选结果")
    return merged_results


def rerank_results(query, docs, mode, top_n=5):
    """重排序"""
    if not docs:
        return []

    if mode == "1":
        texts = [doc.embedding_text for doc in docs]
    else:
        texts = [doc.context for doc in docs]

    scored_docs = []
    for i, (doc, text) in enumerate(zip(docs, texts)):
        score = rank(query, text)
        scored_docs.append((doc, score))
        if i < 3:
            print(f"  [{i+1}] 重排序得分: {score:.4f}")

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored_docs[:top_n]]
    print(f"[重排序] 从 {len(docs)} 条结果中选得分最高的 {len(top_docs)} 条")
    return top_docs


def build_reference_law(docs):
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        text = getattr(doc, "content_text", None) or getattr(doc, "embedding_text", "")
        if text:
            parts.append(f"【参考{i}】{text}")
    length = len(parts)
    step = 2 if length % 2 != 0 else 1
    indices = list(range(0, length, 2)) + list(range(length - step, 0, -2))
    reordered = [parts[i] for i in indices if i < length]
    return "\n\n".join(reordered)


def build_reference_writ(docs):
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"【案例{i}：{doc.indexbytitle}】\n{doc.context}")
    return "\n\n".join(parts)


def try_faq_hit(query):
    """先在 Milvus FAQ 中尝试命中高相似问答"""
    query_vec = get_embedding(query)
    if query_vec is None:
        return None
    hits = QAVecIndex().search(query_vec, topk=3)
    if not hits:
        return None
    top_hit = hits[0]
    if getattr(top_hit, "score", 0) >= 0.8:
        print("[FAQ 命中] 相似度 >= 0.8，直接返回已有答案")
        return top_hit.answer
    return None


def main():
    print("=" * 60)
    print("欢迎使用法律 RAG 问答系统（V2 版）")
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

        # 1) 先查缓存
        cached = QA.find(QA.query == query).all()
        answer = None
        if cached:
            answer = cached[0].answer
            print("\n[缓存命中] 直接返回缓存回答")
        else:
            # 2) FAQ 高相似度命中
            answer = try_faq_hit(query)

        if answer is None:
            # 3) 判断是否需要检索
            need_search = check_need_search(query, history)
            if need_search:
                print("\n[开始检索知识库]")
                if mode == "1":
                    docs = hybrid_search_law(query, vec_topk=10, es_topk=10)
                else:
                    docs = hybrid_search_writ(query, vec_topk=10, es_topk=10)

                if not docs:
                    print("未检索到相关内容，将直接回答")
                    prompt = query
                else:
                    print("\n[重排序]")
                    top_docs = rerank_results(query, docs, mode, top_n=5)

                    if mode == "1":
                        reference = build_reference_law(top_docs)
                    else:
                        reference = build_reference_writ(top_docs)

                    if len(reference) > 300:
                        print(f"\n[参考资料]\n{reference[:300]}...")
                    else:
                        print(f"\n[参考资料]\n{reference}")

                    prompt = f"请根据以下参考内容回答用户问题问题：\n\n{reference}\n\n问题：{query}\n\n答案："
            else:
                print("\n[无需检索] 直接使用 LLM 回答")
                prompt = query

            print("\n[生成回答中...]")
            answer = chat(prompt, history)
            QA(query=query, answer=answer).save()
            print("[缓存写入完成]")

        print(f"\n【回答】\n{answer}")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
