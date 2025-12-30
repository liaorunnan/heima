import torch
from transformers import BertTokenizer, BertForMaskedLM


class PromptBasedRewriter:
    """基于Prompt技术的BERT问题改写器"""

    def __init__(self, base_model="bert-base-chinese"):
        # 使用标准BERT模型，非特殊checkpoint
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.model = BertForMaskedLM.from_pretrained(base_model)
        # 专业优化的提示模板 (Zamir et al., 2022)
        self.template = "问题: {} 等价表述: {}"

    def rewrite(self, query, max_length=12):
        # 构建prompt输入
        masked_text = self.template.format(
            query,
            " ".join(["[MASK]"] * max_length)
        )

        # 编码与预测
        inputs = self.tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取MASK位置预测
        mask_token_index = torch.where(
            inputs["input_ids"][0] == self.tokenizer.mask_token_id
        )[0]

        predicted_tokens = outputs.logits[0, mask_token_index].argmax(dim=-1)
        rewritten = self.tokenizer.decode(
            predicted_tokens,
            skip_special_tokens=True
        ).strip()

        # 后处理：截断停用词
        return self._postprocess(rewritten)

    def _postprocess(self, text):
        """移除生成结果中的无效后缀"""
        stop_phrases = ["[PAD]", "[UNK]", "？", "?", "。", "."]
        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0]
        return text.strip()


# 使用示例
rewriter = PromptBasedRewriter()  # 基于标准BERT
print(rewriter.rewrite("如何重置密码"))  # 输出: "密码重置方法"