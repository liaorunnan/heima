#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微调训练脚本
使用LoRA方法对Qwen2.5-7B-Instruct模型进行微调
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, List
import pandas as pd


def load_and_prepare_data(data_path: str = "../train.csv") -> Dataset:
    """
    加载和准备训练数据
    
    Args:
        data_path: 训练数据路径
        
    Returns:
        处理后的数据集
    """
    # 加载原始数据
    df = pd.read_csv(data_path, header=None)
    
    # 构建训练样本
    samples = []
    for idx, row in df.iterrows():
        if len(row) >= 2:
            dialogue = row[0]
            summary = row[1]
            
            # 构建英语教学相关的提示
            instruction = "You are an English teacher. Based on the dialogue, provide teaching points and explanations."
            
            # 构建训练样本
            text = f"<|im_start|>system\n{instruction}\n<|im_end|>\n<|im_start|>user\n{dialogue}\n<|im_end|>\n<|im_start|>assistant\n{summary}\n<|im_end|>"
            
            samples.append({"text": text})
        
        if len(samples) >= 1000:  # 限制训练数据量
            break
    
    # 转换为Hugging Face数据集
    dataset = Dataset.from_list(samples)
    
    return dataset


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """
    对数据集进行分词处理
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    return dataset.map(tokenize_function, batched=True)


def train():
    """
    主训练函数
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 准备模型用于k-bit训练（如果有的话）
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA秩
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载和准备数据
    print("Loading and preparing data...")
    dataset = load_and_prepare_data()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # 划分训练集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",
        push_to_hub=False,
        report_to="none"
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    
    # 保存LoRA配置
    lora_config.save_pretrained("./finetuned_model")
    
    print("Training completed! Model saved to './finetuned_model'")


def convert_to_instruction_format():
    """
    将训练数据转换为指令格式
    """
    df = pd.read_csv("../train.csv", header=None)
    
    instruction_data = []
    for idx, row in df.iterrows():
        if len(row) >= 2:
            dialogue = row[0]
            summary = row[1]
            
            # 构建英语教学指令数据
            instruction_data.append({
                "instruction": "Analyze this English dialogue and provide teaching points",
                "input": dialogue,
                "output": summary
            })
        
        if len(instruction_data) >= 100:
            break
    
    # 保存为JSON格式
    with open("instruction_data.json", "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(instruction_data)} samples to instruction format")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "convert"],
                       help="Mode: train or convert")
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        convert_to_instruction_format()
    else:
        train()
