"""
重排序模块
功能：计算查询文本与候选文本的相关性得分
使用场景：
  1. 从向量检索的初筛结果中进行精排
  2. 提高最终返回结果与用户查询的相关性
使用 BGE-reranker-v2-m3 模型
"""

import requests


def rank(text_1, text_2, model = "bge-reranker-v2-m3"):
    """
    计算两个文本的相关性得分（单对评分）
    
    参数:
        text_1: 查询文本（用户问题）
        text_2: 候选文本（检索到的文档内容）
        model: 重排序模型名称，默认 bge-reranker-v2-m3
    
    返回:
        float: 相关性得分分数越高越相关
    
    使用示例:
        query = "刑法第一百三十三条"
        candidate = "交通肇事罪相关条文"
        score = rank(query, candidate)
    """
    url = "http://127.0.0.1:8000/score"
    data = {
        "model": model,
        "encoding_format": "float",
        "text_1": text_1,
        "text_2": text_2
    }

    try:
        response = requests.post(url, json=data, timeout=10)
        score = response.json()['data'][0]['score']
        return score
    except Exception as e:
        print(f"重排序评分失败: {e}")
        return 0.0


def rank_batch(query, candidates, model = "bge-reranker-v2-m3"):
    """
    批量计算查询与多个候选文本的相关性得分
    
    参数:
        query: 查询文本（用户问题）
        candidates: 候选文本列表（检索到的多个文档）
        model: 重排序模型名称
    
    返回:
        List[Tuple[int, float]]: [(索引, 得分), ...] 按得分降序排列
    
    使用示例:
        query = "刑法第一百三十三条"
        docs = ["条文1", "条文2", "条文3"]
        ranked_results = rank_batch(query, docs)
        # 返回: [(1, 8.5), (0, 7.2), (2, 3.1)]
    """
    scores = []
    for idx, candidate in enumerate(candidates):
        score = rank(query, candidate, model)
        scores.append((idx, score))
    
    # 按得分降序排序
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


if __name__ == '__main__':
    # 测试单对评分
    result = rank(
        "What is the capital of France?",
        "The capital of France is Paris."
    )
    print(f"相关性得分: {result}")
    
    # 测试批量评分
    query = "刑法相关规定"
    candidates = [
        "刑法第一百三十三条规定交通肇事罪",
        "民法典关于合同的规定",
        "刑事诉讼法的司法解释"
    ]
    batch_results = rank_batch(query, candidates)
    print("\n批量评分结果:")
    for idx, score in batch_results:
        print(f"  文档{idx}: {score:.4f} - {candidates[idx][:20]}...")