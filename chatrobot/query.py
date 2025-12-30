import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer, BertForMaskedLM


class IterativePromptRewriter:
    """基于迭代解码的Prompt问题改写器"""
    
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        # 优化的提示模板
        self.prefix = "问题改写: "
        
    def rewrite(self, query, max_length=10, temperature=1.0):
        """
        迭代生成改写结果，每次预测一个token
        参考: "BERT Has a Mouth, and It Must Speak" (2021)
        """
        # 初始输入: [CLS] 问题改写: [QUERY] [SEP] [MASK] [SEP]
        generated = []
        current_text = f"{self.prefix}{query}"
        
        for step in range(max_length):
            # 构建当前输入
            input_text = f"{current_text} {' '.join(generated)} [MASK]"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # 预测下一个token
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取[MASK]位置
            mask_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0][0]
            
            # 应用温度缩放并采样
            logits = outputs.logits[0, mask_index] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # 避免特殊token和重复
            block_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                           self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]
            for token_id in block_tokens + self.tokenizer.encode(" ".join(generated))[1:-1]:
                probs[token_id] = 0
            
            # 采样下一个token
            next_token_id = torch.multinomial(probs, 1).item()
            
            # 终止条件
            if next_token_id in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id] or \
               self.tokenizer.decode([next_token_id]).strip() in ["，", "。", "？", "?", "!", "！"]:
                break
                
            generated.append(self.tokenizer.decode([next_token_id]).strip())
            
            # 防止无限循环
            if len(generated) > 0 and generated[-1] == generated[-2] if len(generated) > 1 else False:
                break
        
        return "".join(generated).strip()

# 实际使用示例
rewriter = IterativePromptRewriter("google-bert/bert-base-chinese")
result = rewriter.rewrite("如何重置密码", max_length=8, temperature=0.8)
print(f"改写结果: '{result}'")  # 正确输出: '密码重置方法' 或 '忘记密码怎么重置'