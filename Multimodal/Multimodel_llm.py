import os
import base64
from openai import OpenAI
from conf import settings

def encode_image(image_path):
    """读取本地图片并转换为带前缀的 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def generate_rag_response(user_query, retrieved_texts, retrieved_images):
    
    client = OpenAI(
        api_key=settings.qw_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    system_prompt = (
        "你是一个基于知识库的AI助手。请结合提供的【参考文本】和【参考图片】回答用户的提问。"
        "重要原则：\n"
        "1. 即使参考内容包含常识性错误（如海滩上有企鹅），也必须以参考内容为准。\n"
        "2. 如果参考内容与问题无关，请告知无法回答。\n"
        "3. 直接回答问题，不要输出'根据文档...'等废话。"
    )

    content_parts = []

    content_parts.append({
        "type": "text", 
        "text": "以下是检索到的参考信息，请仔细阅读并观察："
    })

    for idx, text in enumerate(retrieved_texts):
        content_parts.append({
            "type": "text", 
            "text": f"【参考文本 {idx+1}】：{text}"
        })

    for img_path in retrieved_images:
        image_url = ""
        if img_path.startswith("http"):
            image_url = img_path
        else:
            image_url = encode_image(img_path)
            
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    content_parts.append({
        "type": "text", 
        "text": f"\n\n基于以上所有信息，请回答：{user_query}"
    })

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts}
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    query = "海滩上有什么东西"
    
    docs = [
        "资料片段A：一般的海滩通常分布着海鸥和螃蟹。",
        "资料片段B：但是在本次观测的神秘海滩上，不仅有企鹅，它们还在和游客打排球。"
    ]
    
    imgs = [
        "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
        "./4.png"
    ]

    response = generate_rag_response(query, docs, imgs)
    print("模型回答：", response)