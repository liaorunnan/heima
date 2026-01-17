import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



BASE_MODEL_PATH = "/root/autodl-tmp/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"


LORA_MAPPING = {
    "default": "/root/autodl-tmp/output/qwen2.5-vl-7b-lora/v9-20260102-202609/checkpoint-63",  # 添加更多LoRA模型
    # "other_lora": "/path/to/other_lora",
}

# 全局变量，存储已加载的LoRA模型
LORA_MODELS = {}

# 初始化基础模型
print(f"正在加载基础模型: {BASE_MODEL_PATH} ...")
try:
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    print("基础模型加载完成！")
except Exception as e:
    print(f"基础模型加载失败: {e}")
    base_model = None
    processor = None

app = FastAPI(title="Qwen2.5-VL API with Dynamic LoRA")

# --- 定义请求数据结构 ---
class MessageContent(BaseModel):
    type: str  # 'text' or 'image'
    text: Optional[str] = None
    image: Optional[str] = None  # 支持本地路径或 URL

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    lora_name: str = "default"  # 指定要使用的LoRA名称

# --- 加载指定LoRA模型 ---
def load_lora_model(lora_name: str):
    """加载指定的LoRA模型，如果尚未加载"""
    if lora_name not in LORA_MAPPING:
        raise ValueError(f"LoRA名称 '{lora_name}' 未在配置中找到")
    
    lora_path = LORA_MAPPING[lora_name]
    
    if lora_name in LORA_MODELS:
        logger.info(f"LoRA模型 '{lora_name}' 已加载，直接使用")
        return LORA_MODELS[lora_name]
    
    logger.info(f"加载LoRA模型: {lora_name} -> {lora_path}")
    
    try:
        # 加载LoRA权重
        lora_model = PeftModel.from_pretrained(
            base_model, 
            lora_path,
            torch_dtype=torch.bfloat16
        )
        lora_model.eval()  # 设置为评估模式
        
        # 存储到全局字典
        LORA_MODELS[lora_name] = lora_model
        return lora_model
    except Exception as e:
        logger.error(f"加载LoRA模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA加载失败: {str(e)}")

# --- API 接口 ---
@app.post("/chat")
async def chat(request: ChatRequest):
    if base_model is None or processor is None:
        raise HTTPException(status_code=500, detail="基础模型未加载")
    
    try:
        # 1. 加载指定的LoRA模型
        lora_model = load_lora_model(request.lora_name)
        
        # 2. 格式化输入数据
        messages_list = [m.model_dump(exclude_none=True) for m in request.messages]
        
        # 3. 预处理文本
        text = processor.apply_chat_template(
            messages_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 4. 预处理图像/视频
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # 5. 转换为 Tensor
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 移至 GPU
        inputs = inputs.to("cuda")
        
        # 6. 推理生成
        with torch.no_grad():
            generated_ids = lora_model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
        
        # 7. 获取生成的 token (去除输入的 token)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 8. 解码为文本
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return {"response": output_text, "lora_used": request.lora_name}
    
    except Exception as e:
        import traceback
        logger.error(f"推理错误: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    