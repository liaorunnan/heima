import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置参数
BASE_MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"  # 基础NLP模型路径

# LoRA模型映射，键为名称，值为路径
LORA_MAPPING = {
    "default": "/root/autodl-tmp/output/qwen25-7b-lora/v0-20260126-211639/checkpoint-1000",  # 默认LoRA模型
    "medical": "/root/autodl-tmp/output/qwen25-7b-lora-medical/checkpoint-500",  # 医疗领域LoRA
    "finance": "/root/autodl-tmp/output/qwen25-7b-lora-finance/checkpoint-800",  # 金融领域LoRA
    # 可添加更多LoRA模型
}

# 全局变量，存储已加载的LoRA模型
LORA_MODELS = {}

# 初始化基础模型和tokenizer
print(f"正在加载基础模型: {BASE_MODEL_PATH} ...")
try:
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    print("基础模型和tokenizer加载完成！")
except Exception as e:
    print(f"基础模型加载失败: {e}")
    base_model = None
    tokenizer = None

# 创建FastAPI应用
app = FastAPI(title="Qwen2.5 NLP API with Dynamic LoRA")


# --- 定义请求数据结构 ---
class Message(BaseModel):
    """聊天消息结构"""
    role: str  # 'system', 'user', 'assistant'
    content: str  # 消息内容

class ChatRequest(BaseModel):
    """聊天请求结构"""
    messages: List[Message]  # 聊天历史
    max_new_tokens: int = 1024  # 最大生成token数
    temperature: float = 0.7  # 温度参数
    top_p: float = 0.8  # top-p参数
    lora_name: str = "default"  # 指定要使用的LoRA名称


# --- 加载指定LoRA模型 ---
def load_lora_model(lora_name: str) -> PeftModel:
    """
    加载指定名称的LoRA模型
    
    Args:
        lora_name: LoRA模型名称，必须在LORA_MAPPING中定义
        
    Returns:
        加载了LoRA权重的模型实例
        
    Raises:
        ValueError: 如果LoRA名称不在映射中
        HTTPException: 如果LoRA加载失败
    """
    if lora_name not in LORA_MAPPING:
        raise ValueError(f"LoRA名称 '{lora_name}' 未在配置中找到")
    
    lora_path = LORA_MAPPING[lora_name]
    
    # 如果模型已加载，直接返回
    if lora_name in LORA_MODELS:
        logger.info(f"LoRA模型 '{lora_name}' 已加载，直接使用")
        return LORA_MODELS[lora_name]
    
    logger.info(f"加载LoRA模型: {lora_name} -> {lora_path}")
    
    try:
        # 加载LoRA权重
        lora_model = PeftModel.from_pretrained(
            base_model, 
            lora_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        lora_model.eval()  # 设置为评估模式
        
        # 存储到全局字典
        LORA_MODELS[lora_name] = lora_model
        return lora_model
    except Exception as e:
        logger.error(f"加载LoRA模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA加载失败: {str(e)}")


# --- 定义OpenAI兼容的请求模型 ---
class OpenAIChatRequest(BaseModel):
    """OpenAI兼容的聊天请求结构"""
    model: str  # 模型名称，可忽略
    messages: List[Message]  # 聊天历史
    max_tokens: int = 1024  # OpenAI API使用max_tokens
    temperature: float = 0.7  # 温度参数
    top_p: float = 0.8  # top-p参数
    lora_name: str = "default"  # 指定要使用的LoRA名称

# --- API 接口 ---
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """
    OpenAI兼容的聊天API接口，支持动态切换LoRA模型
    
    Args:
        request: 符合OpenAI API格式的聊天请求
        
    Returns:
        符合OpenAI API格式的响应
        
    Raises:
        HTTPException: 如果基础模型未加载或推理失败
    """
    if base_model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="基础模型未加载")
    
    try:
        # 1. 加载指定的LoRA模型
        lora_model = load_lora_model(request.lora_name)
        
        # 2. 格式化输入数据
        messages_list = [m.model_dump(exclude_none=True) for m in request.messages]
        
        # 3. 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 4. 转换为Tensor
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移至GPU
        inputs = inputs.to("cuda")
        
        # 5. 配置生成参数
        generation_config = GenerationConfig(
            max_new_tokens=request.max_tokens,  # 使用OpenAI API的max_tokens参数
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 6. 推理生成
        with torch.no_grad():
            generated_ids = lora_model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 7. 获取生成的token（去除输入的token）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 8. 解码为文本
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 9. 构造OpenAI兼容的响应格式
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            },
            "lora_used": request.lora_name
        }
    
    except Exception as e:
        import traceback
        logger.error(f"推理错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 保留原有的/chat接口以兼容旧调用方式
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    聊天API接口，支持动态切换LoRA模型（兼容旧版本）
    
    Args:
        request: 聊天请求，包含消息历史、生成参数和LoRA名称
        
    Returns:
        包含生成响应和使用的LoRA名称的字典
        
    Raises:
        HTTPException: 如果基础模型未加载或推理失败
    """
    if base_model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="基础模型未加载")
    
    try:
        # 1. 加载指定的LoRA模型
        lora_model = load_lora_model(request.lora_name)
        
        # 2. 格式化输入数据
        messages_list = [m.model_dump(exclude_none=True) for m in request.messages]
        
        # 3. 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 4. 转换为Tensor
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 移至GPU
        inputs = inputs.to("cuda")
        
        # 5. 配置生成参数
        generation_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 6. 推理生成
        with torch.no_grad():
            generated_ids = lora_model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 7. 获取生成的token（去除输入的token）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 8. 解码为文本
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return {
            "response": output_text,
            "lora_used": request.lora_name,
            "generated_tokens": len(generated_ids_trimmed[0])
        }
    
    except Exception as e:
        import traceback
        logger.error(f"推理错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "base_model_loaded": base_model is not None,
        "available_lora_models": list(LORA_MAPPING.keys()),
        "loaded_lora_models": list(LORA_MODELS.keys())
    }


# 获取可用LoRA模型列表
@app.get("/lora/models")
async def list_lora_models():
    """获取可用的LoRA模型列表"""
    return {
        "available_lora_models": [
            {
                "name": name,
                "path": path,
                "is_loaded": name in LORA_MODELS
            }
            for name, path in LORA_MAPPING.items()
        ]
    }


if __name__ == "__main__":
    # 启动FastAPI服务
    uvicorn.run(app, host="0.0.0.0", port=6008)