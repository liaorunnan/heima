import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设您有一个向量 [batch_size, prompt_length, hidden_size]
custom_vector = torch.randn(1, 10, 768)  # 示例：10个token长度的软提示

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 准备文本输入
text = "今天的天气怎么样？"
inputs = tokenizer(text, return_tensors="pt")
text_embeddings = model.get_input_embeddings()(inputs["input_ids"])

# 将自定义向量与文本嵌入拼接
combined_embeddings = torch.cat([custom_vector, text_embeddings], dim=1)

# 前向传播（需要调整attention_mask和position_ids）
outputs = model(inputs_embeds=combined_embeddings)