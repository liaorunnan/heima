import requests
from b_rag.xiaokui.tool.embedding import get_embedding
"""
模型类型: 从名称 bge-reranker-v2-m3 来看，这是一个重排序模型（reranker）
功能用途: 用于计算两个文本之间的相似度得分
技术背景: BGE (Bidirectional Guided Encoder) 系列通常是用于文本检索和语义相似度计算的模型
"""

def rank(text1, text2, model='bge-reranker-v2-m3'):
    url = f"http://127.0.0.1:8000/score"
    data = {
        'model': model,
        'encoding_format': 'float',
        'text_1': text1,
        'text_2': text2
    }
    response = requests.post(url, json=data)
    # print(response.json())
    return response.json()['data'][0]['score']

if __name__ == '__main__':
    text1="1+1=2"
    text2="1+1=999"
    print(rank(text1, text2))


