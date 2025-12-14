import json
from tqdm import tqdm
import numpy as np
import simple_pickle as sp

from b_rag.xiaokui.tool.embedding import get_embedding


def text_similarity(lines):
    pos_neg_scores = []
    for line in tqdm(lines):
        qa = json.loads(line)
        query_vec = get_embedding(qa["query"])
        pos_vec = [get_embedding(p) for p in qa["pos"]]
        neg_vec = [get_embedding(n) for n in qa["neg"]]

        query_vec = query_vec / np.linalg.norm(query_vec)
        pos_vec = [p / np.linalg.norm(p) for p in pos_vec]
        neg_vec = [n / np.linalg.norm(n) for n in neg_vec]
        pos_score = [np.dot(query_vec, p) for p in pos_vec]
        neg_score = [np.dot(query_vec, n) for n in neg_vec]

        pos_neg_scores.append({"query": qa["query"], "pos": pos_score, "neg": neg_score})

    sp.write_pickle(pos_neg_scores, "../../data/pos_neg_scores.pkl")


def calculate_metrics(threshold, pos_neg_scores):
    """
    根据给定阈值计算准确率、召回率、F1分数

    Args:
        threshold: 相似度阈值
        pos_neg_scores: 包含正负样本分数的列表

    Returns:
        accuracy, recall, f1, precision
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for item in pos_neg_scores:
        pos_scores = item["pos"]
        neg_scores = item["neg"]

        # 计算该query下正样本的预测结果
        pos_predictions = [score >= threshold for score in pos_scores]
        neg_predictions = [score >= threshold for score in neg_scores]

        # 统计正样本
        tp_in_query = sum(pos_predictions)
        fn_in_query = len(pos_predictions) - tp_in_query

        # 统计负样本
        fp_in_query = sum(neg_predictions)
        tn_in_query = len(neg_predictions) - fp_in_query

        true_positives += tp_in_query
        false_negatives += fn_in_query
        false_positives += fp_in_query
        true_negatives += tn_in_query

    # 计算指标
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return accuracy, precision, recall, f1


def find_optimal_threshold(pos_neg_scores, threshold_range=(0, 1), step=0.01):
    """
    搜索最佳阈值

    Args:
        pos_neg_scores: 包含正负样本分数的列表
        threshold_range: 阈值搜索范围
        step: 搜索步长

    Returns:
        best_threshold, best_metrics, all_metrics
    """
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
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        all_metrics.append(metrics)

        print(f"{threshold:<8.2f} {accuracy:<8.4f} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = metrics

    print("-" * 50)
    print(f"\n最佳阈值: {best_threshold:.2f}")
    print(f"最佳指标:")
    print(f"  准确率: {best_metrics['accuracy']:.4f}")
    print(f"  精确率: {best_metrics['precision']:.4f}")
    print(f"  召回率: {best_metrics['recall']:.4f}")
    print(f"  F1分数: {best_metrics['f1']:.4f}")

    return best_threshold, best_metrics, all_metrics


if __name__ == "__main__":
    lines = sp.read_data("./data_negs.jsonl")
    text_similarity(lines)

    pos_neg_scores = sp.read_pickle("../../data/pos_neg_scores.pkl")
    best_threshold, best_metrics, all_metrics = find_optimal_threshold(pos_neg_scores)

    result = {
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'all_metrics': all_metrics
    }

    sp.write_pickle(result, "../../data/threshold_results.pkl")
    print(f"\n结果已保存到 threshold_results.pkl")
