from pymilvus import connections, Collection

# 1. 连接 Milvus
connections.connect(host="127.0.0.1", port="19530")

# 2. 获取 Collection
collection_name = "fqa"
collection = Collection(collection_name)

# 3. 使用 query 接口拉取数据
# 注意：limit 有最大限制（通常是 16384），如果数据量大需要分批次(offset)拉取
# 或者使用迭代器（Milvus 2.3+ 支持 iterator）
res = collection.query(
    expr="id >= 0",  # 这是一个匹配所有的表达式，视具体 schema 而定
    output_fields=["qid","vec","query","answer","source"], # 指定要导出的字段
    limit=1000 # 演示用，实际可能需要循环拉取
)

# 4. 保存为 JSON 或 CSV
import pandas as pd
df = pd.DataFrame(res)
df.to_csv(f"{collection_name}_dump.csv", index=False)
print("导出完成")