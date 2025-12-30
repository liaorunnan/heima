import os

# 设置 Hugging Face 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig
)
# 1. 引入 prepare_model_for_kbit_training
from peft import get_peft_model, LoraConfig

# ==============================================================================
# 模型和数据集路径配置
# ==============================================================================
model_name_or_path = "Qwen/Qwen3-0.6B"
train_file = "./data1"

# ==============================================================================
# 训练参数配置
# ==============================================================================

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    save_steps=500,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,          # 建议调小以便观察
    save_total_limit=3,        # 限制保存的 checkpoint 数量，防爆硬盘
    gradient_checkpointing=True # 在这里开启梯度检查点
)

# ==============================================================================
# 加载分词器
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    trust_remote_code=True
)
# Qwen2.5 可能没有默认 pad_token，显式指定（通常使用 eos 或特定的 pad token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==============================================================================
# 加载预训练模型 (4-bit 量化)
# ==============================================================================
# 配置量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config, # 使用上面定义的配置
    trust_remote_code=True,
    device_map="auto"
)

# ==============================================================================
# [关键修复] 预处理模型以进行 k-bit 训练
# ==============================================================================
# 1. 显式开启梯度检查点（减少显存）
model.gradient_checkpointing_enable()

# 2. 准备模型（这会处理 LayerNorm 的精度转换并允许输入梯度）
model = prepare_model_for_kbit_training(model)

# ==============================================================================
# LoRA 配置
# ==============================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Qwen 建议微调所有线性层效果更好
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数量，确认配置成功
model.print_trainable_parameters()

# 注意：不要在 4-bit 模式下调用 resize_token_embeddings，除非你非常清楚你在做什么
# model.resize_token_embeddings(len(tokenizer)) 
# model.config.use_cache = False # prepare_model_for_kbit_training 通常会自动处理，但保留也无妨

# ==============================================================================
# 加载数据集并初始化训练器
# ==============================================================================
lm_datasets = load_from_disk(train_file)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    tokenizer=tokenizer, # 传入 tokenizer，Trainer 会自动处理 pad_token
    data_collator=default_data_collator
)

# ==============================================================================
# 开始训练
# ==============================================================================
# 如果之前中断过，可以设置 resume_from_checkpoint=True
train_result = trainer.train()
trainer.save_model()
trainer.save_metrics("train", train_result.metrics)