import openai
from conf import settings
from typing import List, Dict, Any
import torch

# 全局变量，用于缓存本地模型和分词器
_local_model = None
_local_tokenizer = None

client = openai.Client(base_url=settings.base_url, api_key=settings.api_key)

def load_local_model():
    """
    加载本地微调模型和分词器
    """
    global _local_model, _local_tokenizer
    
    if _local_model is None or _local_tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading local fine-tuned model...")
        model_path = settings.local_model_path
        
        # 加载分词器
        _local_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _local_tokenizer.pad_token = _local_tokenizer.eos_token
        
        # 加载模型
        device = settings.local_model_device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            _local_model = _local_model.to(device)
        
        _local_model.eval()
        print("Local model loaded successfully")
    
    return _local_model, _local_tokenizer

def generate_local_response(query: str, history: List[Dict[str, str]], system_prompt: str) -> str:
    """
    使用本地模型生成回复
    """
    model, tokenizer = load_local_model()
    
    # 构建消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": query}
    ]
    
    # 将消息列表转换为模型所需的格式（这里以Qwen2.5-Instruct为例）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=settings.local_model_max_length)
    
    # 移动到模型所在的设备
    device = settings.local_model_device
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    else:
        inputs = {k: v for k, v in inputs.items()}
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出，跳过输入部分
    input_length = inputs["input_ids"].shape[1]
    response_ids = outputs[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

def chat(query, history, system_prompt="你是一位英语老师，请根据我的提问，帮我寻找对应的听力文章或者回答我关于英语文章内容的问题。如果是在查询英文单词，请详细列出音标和解释，并附上几条例句。如果没有在参考问题中找到相关内容，请告诉我。整体回答结构分明，意思清晰，不要重复。"):
    """
    聊天函数，根据配置选择使用API还是本地模型
    """
    if settings.use_local_model:
        try:
            return generate_local_response(query, history, system_prompt)
        except Exception as e:
            print(f"Local model error: {e}, falling back to API")
            # 如果本地模型出错，回退到API
            
    # 使用API
    response = client.chat.completions.create(
        model = settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content
