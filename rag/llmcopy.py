import requests
from typing import List, Dict, Any
import time



url = "https://uu823409-a578-d1232007.bjb2.seetacloud.com:8443/chat"


def chat(query, history, system_prompt="主角李火旺分不清虚拟和现实，体内还有很多疯狂的人格，所以一直处于痛苦和挣扎中，请用主角李火旺多样化的疯言疯语进行回答,注意，你就是李火旺，使用第一人称与我对话", lora_name="default"):
    """
    聊天函数，调用本地部署的NLP LoRA服务
    
    Args:
        query: 用户查询内容
        history: 聊天历史
        system_prompt: 系统提示词
        lora_name: 要使用的LoRA模型名称
        
    Returns:
        模型生成的响应内容
    """
    
    # 构建消息列表
    messages = []
    
    # 添加系统提示
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 添加聊天历史
    messages.extend(history)
    
    # 添加当前查询
    messages.append({"role": "user", "content": query})
    
    # 构建请求体
    payload = {
        "model": "qwen3-06b",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 1024,
        "lora_name": lora_name  # 自定义参数，用于选择LoRA模型
    }
    
    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }
    
    # 发送HTTP POST请求
    response = requests.post(url, json=payload, headers=headers)
    
    # 检查响应状态
    response.raise_for_status()
    
    # 解析响应
    result = response.json()
    return result

if __name__ == "__main__":
    history = []
    query = "工作996，身体越来越差，但不敢辞职"
    start_time = time.time()
    response = chat(query, history)
    end_time = time.time()
    print(response['response'])
    print(f"响应时间: {end_time - start_time} 秒")
