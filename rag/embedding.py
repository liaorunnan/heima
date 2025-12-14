from openai import OpenAI
import numpy as np
import requests
from conf import settings




def test_vllm_connection():
    try:
        response = requests.get(settings.Emb_url, timeout=5)
        models = response.json()
        print("vLLM 服务可用模型:")
        for model in models.get('data', []):
            print(f"  - {model.get('id', 'Unknown')}")
        return True
    except Exception as e:
        print(f"vLLM 服务连接失败: {e}")
        return False


client = OpenAI(
    base_url=settings.Emb_url.replace('/models', ''),
    api_key=settings.Emb_token  # vLLM 默认不需要认证，但需要提供任意值
)


def get_embedding(text, model="bge-m3"):

    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)


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
