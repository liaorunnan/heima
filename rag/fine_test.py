import os

# 设置 Hugging Face 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BitsAndBytesConfig,TrainingArguments,Trainer,AutoModelForCausalLM

import numpy as np
import evaluate
import torch

from datasets import load_from_disk
from peft import get_peft_model, LoraConfig

print(1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
print(2)
# metric = evaluate.load("accuracy")

model_path = "Qwen/Qwen2.5-7B-Instruct"
train_file = "./data1"
eval_file = "./data2"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4位量化
    bnb_4bit_quant_type="nf4",  # NF4量化
    bnb_4bit_compute_dtype=float16,  # 计算时使用float16
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步减小模型大小
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    gradient_checkpointing_enable=True
)

training_args = TrainingArguments(
    output_dir="test_trainer", 
    eval_strategy="epoch",
    logging_steps=10, 
    report_to="none",
    num_train_epochs=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=3,
    warmup_steps=100,
    save_steps=500,
    learning_rate=2e-5,
    fp16=True,

)

print(4)
lm_datasets = load_from_disk(train_file)
lm_eval_file_datasets = load_from_disk(eval_file)
# 1. 显式开启梯度检查点（减少显存）
model.gradient_checkpointing_enable()

# 2. 准备模型（这会处理 LayerNorm 的精度转换并允许输入梯度）
model = prepare_model_for_kbit_training(model)
print(5)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"], # Qwen 建议微调所有线性层效果更好
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    
)
print(6)
# 应用 LoRA
model = get_peft_model(model, lora_config)
print(7)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = lm_datasets,
    eval_dataset = lm_eval_file_datasets,
    compute_metrics=compute_metrics,
) 
print(8)
train_result = trainer.train()
trainer.save_model()
trainer.save_metrics("train", train_result.metrics)
print(9)
