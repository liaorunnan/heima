import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import simple_pickle as sp
from typing import List, Dict, Any

# ========================
# 1. 初始化 OpenAI 客户端（你的配置）
# ========================
from heima.conf import settings  # 替换为你的实际 settings 导入路径

client = OpenAI(
    base_url=f"{settings.vllm_index_url}/v1",
    api_key="token-abc123"  # vLLM 通常忽略，但需提供
)

def get_embeddings_from_vllm(texts: List[str], model: str = "bge-m3", batch_size: int = 256) -> np.ndarray:
    """
    通过 vLLM 的 OpenAI 兼容接口批量获取 embeddings
    texts: 文本列表
    returns: 归一化的 numpy array [N, D]
    """
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Fetching embeddings from vLLM"):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            # 提取 embeddings 并转为 numpy
            embeddings = np.array([item.embedding for item in response.data])
            # 归一化（BGE 要求）
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"Batch {i//batch_size} failed: {e}")
            # 可选：fallback to single retry or pad with zeros
            raise e
    return np.vstack(all_embeddings)


# ========================
# 2. 提取唯一文本（同前）
# ========================
def collect_unique_texts(lines: List[str]):
    queries = []
    pos_texts = []
    neg_texts = []

    query_list = []
    pos_list_of_lists = []
    neg_list_of_lists = []

    for line in lines:
        qa = json.loads(line.strip())
        query = qa["query"]
        pos = qa["pos"]
        neg = qa["neg"]

        query_list.append(query)
        pos_list_of_lists.append(pos)
        neg_list_of_lists.append(neg)

        queries.append(query)
        pos_texts.extend(pos)
        neg_texts.extend(neg)

    all_texts = list(set(queries + pos_texts + neg_texts))
    text_to_idx = {text: i for i, text in enumerate(all_texts)}
    return all_texts, text_to_idx, query_list, pos_list_of_lists, neg_list_of_lists


# ========================
# 3. 主流程：高效计算相似度
# ========================
def text_similarity_optimized(lines: List[str], model: str = "bge-m3", batch_size: int = 256):
    print("Collecting unique texts...")
    all_texts, text_to_idx, queries, pos_lists, neg_lists = collect_unique_texts(lines)

    print(f"Total unique texts: {len(all_texts)}")
    print("Sending batch requests to vLLM for embeddings...")
    all_embeddings = get_embeddings_from_vllm(all_texts, model=model, batch_size=batch_size)

    print("Building text-to-embedding map...")
    text_to_emb = {text: all_embeddings[idx] for text, idx in text_to_idx.items()}

    print("Computing similarity scores...")
    pos_neg_scores = []
    for query, pos_list, neg_list in tqdm(zip(queries, pos_lists, neg_lists), total=len(queries)):
        q_vec = text_to_emb[query]
        pos_scores = [float(np.dot(q_vec, text_to_emb[p])) for p in pos_list]
        neg_scores = [float(np.dot(q_vec, text_to_emb[n])) for n in neg_list]
        pos_neg_scores.append({
            "query": query,
            "pos": pos_scores,
            "neg": neg_scores
        })

    sp.write_pickle(pos_neg_scores, "pos_neg_scores.pkl")
    print("✅ Saved pos_neg_scores.pkl")


# ========================
# 4. 指标计算（保持不变）
# ========================
def calculate_metrics(threshold, pos_neg_scores):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for item in pos_neg_scores:
        pos_scores = item["pos"]
        neg_scores = item["neg"]

        pos_predictions = [score >= threshold for score in pos_scores]
        neg_predictions = [score >= threshold for score in neg_scores]

        tp_in_query = sum(pos_predictions)
        fn_in_query = len(pos_predictions) - tp_in_query
        fp_in_query = sum(neg_predictions)
        tn_in_query = len(neg_predictions) - fp_in_query

        true_positives += tp_in_query
        false_negatives += fn_in_query
        false_positives += fp_in_query
        true_negatives += tn_in_query

    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1


def find_optimal_threshold(pos_neg_scores, threshold_range=(0, 1), step=0.01):
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    best_threshold = 0
    best_f1 = 0
    best_metrics = None
    all_metrics = []

    print("搜索最佳阈值...")
    print(f"{'阈值':<8} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 50)

    for threshold in thresholds:
        accuracy, precision, recall, f1 = calculate_metrics(threshold, pos_neg_scores)
        metrics = {
            'threshold': float(threshold),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        all_metrics.append(metrics)

        print(f"{threshold:<8.2f} {accuracy:<8.4f} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_metrics = metrics

    print("-" * 50)
    print(f"\n最佳阈值: {best_threshold:.2f}")
    print(f"最佳指标:")
    print(f"  准确率: {best_metrics['accuracy']:.4f}")
    print(f"  精确率: {best_metrics['precision']:.4f}")
    print(f"  召回率: {best_metrics['recall']:.4f}")
    print(f"  F1分数: {best_metrics['f1']:.4f}")

    return best_threshold, best_metrics, all_metrics


# ========================
# 5. 主程序
# ========================
if __name__ == "__main__":
    with open("./data_negs.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 调用优化版流程
    text_similarity_optimized(lines, model="bge-m3", batch_size=256)

    pos_neg_scores = sp.read_pickle("pos_neg_scores.pkl")
    best_threshold, best_metrics, all_metrics = find_optimal_threshold(pos_neg_scores)

    result = {
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'all_metrics': all_metrics
    }

    sp.write_pickle(result, "threshold_results.pkl")
    print(f"\n✅ 结果已保存到 threshold_results.pkl")
