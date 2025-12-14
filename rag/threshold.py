import json
from tqdm import tqdm
import numpy as np
import simple_pickle as sp
from embedding import get_embedding


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

    sp.write_pickle(pos_neg_scores, "pos_neg_scores.pkl")


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
    lines = sp.read_data("./data/neg_data_output.jsonl")
    text_similarity(lines)

    pos_neg_scores = sp.read_pickle("pos_neg_scores.pkl")
    best_threshold, best_metrics, all_metrics = find_optimal_threshold(pos_neg_scores)

    result = {
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'all_metrics': all_metrics
    }

    sp.write_pickle(result, "threshold_results.pkl")
    print(f"\n结果已保存到 threshold_results.pkl")

#     阈值       准确率      精确率      召回率      F1分数    
# --------------------------------------------------
# 0.00     0.1667   0.1667   1.0000   0.2857  
# 0.01     0.1667   0.1667   1.0000   0.2857  
# 0.02     0.1667   0.1667   1.0000   0.2857  
# 0.03     0.1667   0.1667   1.0000   0.2857  
# 0.04     0.1667   0.1667   1.0000   0.2857  
# 0.05     0.1667   0.1667   1.0000   0.2857  
# 0.06     0.1667   0.1667   1.0000   0.2857  
# 0.07     0.1667   0.1667   1.0000   0.2857  
# 0.08     0.1667   0.1667   1.0000   0.2857  
# 0.09     0.1667   0.1667   1.0000   0.2857  
# 0.10     0.1667   0.1667   1.0000   0.2857  
# 0.11     0.1667   0.1667   1.0000   0.2857  
# 0.12     0.1667   0.1667   1.0000   0.2857  
# 0.13     0.1667   0.1667   1.0000   0.2857  
# 0.14     0.1667   0.1667   1.0000   0.2857  
# 0.15     0.1667   0.1667   1.0000   0.2857  
# 0.16     0.1667   0.1667   1.0000   0.2857  
# 0.17     0.1667   0.1667   1.0000   0.2857  
# 0.18     0.1667   0.1667   1.0000   0.2857  
# 0.19     0.1667   0.1667   1.0000   0.2857  
# 0.20     0.1667   0.1667   1.0000   0.2857  
# 0.21     0.1667   0.1667   1.0000   0.2857  
# 0.22     0.1667   0.1667   1.0000   0.2857  
# 0.23     0.1667   0.1667   1.0000   0.2857  
# 0.24     0.1667   0.1667   1.0000   0.2857  
# 0.25     0.1667   0.1667   1.0000   0.2857  
# 0.26     0.1667   0.1667   1.0000   0.2857  
# 0.27     0.1667   0.1667   1.0000   0.2857  
# 0.28     0.1667   0.1667   1.0000   0.2857  
# 0.29     0.1667   0.1667   1.0000   0.2857  
# 0.30     0.1667   0.1667   1.0000   0.2857  
# 0.31     0.1667   0.1667   1.0000   0.2857  
# 0.32     0.1667   0.1667   1.0000   0.2857  
# 0.33     0.1667   0.1667   1.0000   0.2857  
# 0.34     0.1667   0.1667   1.0000   0.2857  
# 0.35     0.1667   0.1667   1.0000   0.2857  
# 0.36     0.1667   0.1667   1.0000   0.2857  
# 0.37     0.1667   0.1667   1.0000   0.2857  
# 0.38     0.1667   0.1667   1.0000   0.2857  
# 0.39     0.1667   0.1667   1.0000   0.2857  
# 0.40     0.1667   0.1667   1.0000   0.2857  
# 0.41     0.1667   0.1667   1.0000   0.2857  
# 0.42     0.1667   0.1667   1.0000   0.2857  
# 0.43     0.1667   0.1667   1.0000   0.2857  
# 0.44     0.1667   0.1667   1.0000   0.2857  
# 0.45     0.1667   0.1667   1.0000   0.2857  
# 0.46     0.1667   0.1667   1.0000   0.2857  
# 0.47     0.1667   0.1667   1.0000   0.2857  
# 0.48     0.1667   0.1667   1.0000   0.2857  
# 0.49     0.1667   0.1666   0.9998   0.2857  
# 0.50     0.1668   0.1667   0.9997   0.2857  
# 0.51     0.1674   0.1668   0.9997   0.2858  
# 0.52     0.1686   0.1670   0.9997   0.2861  
# 0.53     0.1713   0.1674   0.9995   0.2868  
# 0.54     0.1756   0.1681   0.9995   0.2878  
# 0.55     0.1837   0.1695   0.9992   0.2898  
# 0.56     0.1954   0.1715   0.9992   0.2928  
# 0.57     0.2143   0.1749   0.9991   0.2977  
# 0.58     0.2412   0.1800   0.9991   0.3050  
# 0.59     0.2752   0.1868   0.9989   0.3148  
# 0.60     0.3125   0.1950   0.9988   0.3263  
# 0.61     0.3530   0.2047   0.9988   0.3397  
# 0.62     0.3977   0.2166   0.9984   0.3559  
# 0.63     0.4488   0.2320   0.9981   0.3764  
# 0.64     0.5131   0.2548   0.9981   0.4060  
# 0.65     0.5788   0.2832   0.9980   0.4413  
# 0.66     0.6462   0.3199   0.9975   0.4845  
# 0.67     0.7108   0.3654   0.9975   0.5348  
# 0.68     0.7689   0.4188   0.9971   0.5898  
# 0.69     0.8169   0.4765   0.9963   0.6447  
# 0.70     0.8556   0.5359   0.9953   0.6967  
# 0.71     0.8817   0.5854   0.9946   0.7370  
# 0.72     0.9019   0.6305   0.9943   0.7717  
# 0.73     0.9172   0.6696   0.9941   0.8002  
# 0.74     0.9290   0.7032   0.9935   0.8235  
# 0.75     0.9372   0.7286   0.9927   0.8404  
# 0.76     0.9446   0.7534   0.9921   0.8565  
# 0.77     0.9500   0.7726   0.9915   0.8685  
# 0.78     0.9548   0.7907   0.9908   0.8795  
# 0.79     0.9586   0.8060   0.9899   0.8885  
# 0.80     0.9617   0.8195   0.9881   0.8959  
# 0.81     0.9640   0.8295   0.9870   0.9014  
# 0.82     0.9667   0.8419   0.9854   0.9080  
# 0.83     0.9688   0.8517   0.9840   0.9131  
# 0.84     0.9707   0.8615   0.9822   0.9179  
# 0.85     0.9726   0.8719   0.9794   0.9225  
# 0.86     0.9748   0.8847   0.9758   0.9280  
# 0.87     0.9772   0.8996   0.9715   0.9341  
# 0.88     0.9792   0.9149   0.9653   0.9394  
# 0.89     0.9813   0.9321   0.9577   0.9447  
# 0.90     0.9834   0.9491   0.9513   0.9502  
# 0.91     0.9842   0.9630   0.9414   0.9521  
# 0.92     0.9844   0.9753   0.9297   0.9520  
# 0.93     0.9841   0.9860   0.9175   0.9505  
# 0.94     0.9829   0.9933   0.9034   0.9462  
# 0.95     0.9791   0.9972   0.8770   0.9332  
# 0.96     0.9689   0.9992   0.8140   0.8972  
# 0.97     0.9453   1.0000   0.6716   0.8036  
# 0.98     0.9187   1.0000   0.5123   0.6775  
# 0.99     0.9041   1.0000   0.4248   0.5963  
# --------------------------------------------------

# 最佳阈值: 0.91
# 最佳指标:
#   准确率: 0.9842
#   精确率: 0.9630
#   召回率: 0.9414
#   F1分数: 0.9521

# 结果已保存到 threshold_results.pkl
