

import openai

client = openai.Client(base_url="https://uu823409-b96d-6263d673.westb.seetacloud.com:8443/v1", api_key='EMPTY')
system_prompt = """
你是一个数据提取助手。请提取用户信息并以 JSON 格式输出。

# 步骤要求
1. 首先，在 <thinking> 标签内进行一步步推理。分析用户的意图，检查缺失信息，并纠正潜在的逻辑错误。
2. 然后，在 ```json 代码块中输出最终结果。

# 输出格式示例
<thinking>
用户提到了日期“下周五”，今天是2023-10-20，所以下周五是...
</thinking>
```json
{
  "date": "2023-10-27",
  ...
}
"""

query = "下周五的日期是什么？"

response = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ],
    temperature=0.5,
)

print(response.choices[0].message.content)