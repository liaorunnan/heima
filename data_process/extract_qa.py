import os
import simple_pickle as sp
from datasets import load_dataset
import random
import rag_official.embedding as embedding
from tqdm import tqdm
import numpy as np

# 设置随机种子以确保结果可重复
random.seed(42)

# 测试vLLM服务连接
print("正在测试vLLM服务连接...")
if not embedding.test_vllm_connection():
    print("vLLM服务不可用，请确保服务正在运行在http://localhost:6006")
    exit(1)

# 下载Hugging Face法律问答数据集
print("正在下载法律问答数据集...")
dataset = load_dataset("Kuugo/chinese_law_ft_dataset")

# 从训练集中随机抽样2w条问答对
train_data = dataset["train"]
sampled_data = random.sample(list(train_data), 20000)
print(f"成功抽样{len(sampled_data)}条问答对")

# 将数据转换为需要的格式：[{"query":"","answer":""}]
print("正在处理问答对并生成嵌入向量...")
qas = []
processed_count = 0
failed_count = 0

for item in tqdm(sampled_data):
    # 获取嵌入向量
    embedding_vector = embedding.get_embedding(item["instruction"])
    
    if embedding_vector is not None:
        # 将numpy数组转换为列表，确保pickle兼容性
        embedding_list = embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
        
        qas.append({
            "query": item["instruction"],
            "answer": item["output"],
            "query_embedding_text": embedding_list,
        })
        processed_count += 1
    else:
        failed_count += 1
        print(f"处理失败: {item['instruction'][:50]}...")

# 保存为pickle文件
save_path = "../../day04/data/qas.pkl"
sp.write_pickle(qas, save_path)

print(f"\n处理完成！")
print(f"成功生成嵌入并保存: {processed_count}条")
print(f"生成嵌入失败: {failed_count}条")
print(f"最终保存{len(qas)}条法律问答对到{save_path}")
