"""
向量嵌入模块
功能：将文本转换为向量表示，支持单文本和批量文本处理
使用场景：
  1. 将用户查询转换为向量用于检索
  2. 将文档内容转换为向量用于存储到 Milvus
"""

from openai import OpenAI
import numpy as np
import requests
import logging

# 禁用 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def test_vllm_connection():
    """
    测试 vLLM 向量服务连接状态
    
    返回:
        bool: 连接成功返回 True，失败返回 False
    """
    try:
        response = requests.get("http://localhost:6006/v1/models", timeout=5)
        models = response.json()
        print("vLLM 服务可用模型:")
        for model in models.get('data', []):
            print(f"  - {model.get('id', 'Unknown')}")
        return True
    except Exception as e:
        print(f"vLLM 服务连接失败: {e}")
        return False


# 初始化 OpenAI 客户端，连接到本地 vLLM 服务
# vLLM 作为推理后端，提供兼容 OpenAI API 的接口
client = OpenAI(
    base_url="http://localhost:6006/v1",
    api_key="token-abc123"  # vLLM 默认不需要认证，但需要提供任意值
)


def get_embedding(text, model = "bge-m3"):
    """
    获取文本的向量嵌入表示
    
    参数:
        text: 单个文本字符串或文本列表
        model: 向量模型名称，默认使用 bge-m3 (1024维中文向量模型)
    
    返回:
        np.ndarray: 单文本时返回向量数组
        List[np.ndarray]: 多文本时返回向量列表
        None: 获取失败时返回 None
    
    使用示例:
        # 单文本
        vec = get_embedding("法律条文内容")
        
        # 批量文本
        vecs = get_embedding(["条文1", "条文2", "条文3"])
    """
    try:
        response = client.embeddings.create(model=model, input=text)
        
        # 单文本直接返回向量
        if isinstance(text, str):
            return np.array(response.data[0].embedding)
        
        # 多文本返回向量列表
        return [np.array(item.embedding) for item in response.data]
    
    except Exception as e:
        print(f"获取嵌入向量失败: {e}")
        return None


if __name__ == '__main__':
    if not test_vllm_connection():
        exit(1)

    text = "这是一个测试句子"
    print(f"处理文本: {text}")

    embedding = get_embedding(text)
    if embedding is not None:
        print(f"嵌入维度: {embedding.shape}")
        print(f"前10个维度: {embedding[:10]}")

        # 测试多个文本
        texts = ["人工智能", "机器学习", "深度学习"]
        print("\n批量处理多个文本:")
        for i, txt in enumerate(texts):
            emb = get_embedding(txt)
            if emb is not None:
                print(f"  {i + 1}. {txt}: {emb.shape}")
    else:
        print("无法获取嵌入向量")
