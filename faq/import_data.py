import simple_pickle as sp
from b_rag.day04.faq.indexing import VecIndex
from b_rag.day02.items import FAQItem
from b_rag.day03.embedding import get_embedding  # 只导入单条函数也行
# from b_rag.day03.embedding import get_embeddings_batch  # 如果你实现了批量更好！

# 1. 读取数据
qas = sp.read_pickle("../../day04/data/qas.pkl")
print(f"共加载 {len(qas)} 条 Q&A 数据")

# 2. 先只构建不带 embedding 的 items（query_embedding_text 先留空或填原文本）
items = [
    FAQItem(
        id=f"{i}",
        query=qa["query"],
        answer=qa["answer"],
        query_embedding_text=qa["query"]  # 先临时放 query 原文，后面会覆盖成向量（可选）
    )
    for i, qa in enumerate(qas, 1)
]

print(f"构建了 {len(items)} 条 FAQItem 对象")

# 3. 直接调用优化后的 load 方法（它内部会批量处理 embedding + 插入）
VecIndex("faq").load(items)

print("所有数据导入完成！")