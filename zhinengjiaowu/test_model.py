from openai import OpenAI
import base64

# 1. 配置客户端
# 如果您的代理地址不是 localhost，请替换 base_url
# 例如：如果 AutoDL 提供了类似 http://region-x.autodl.com:port 的地址，填入那个地址 + /v1
client = OpenAI(
    api_key="EMPTY",  # 自托管服务通常不需要真实 Key
    base_url="http://localhost:16008/v1", 
)

# 列出所有可用模型
models = client.models.list()
for m in models:
    print(f"可用模型名称: {m.id}")

# 2. 准备模型名称
# 这必须与服务端启动时指定的模型名称一致
# 如果不确定，可以通过 client.models.list() 查看
model_name = "qwen3vl-8b"

# 3. 准备输入数据
# API 模式下，图片通常通过 URL 或 Base64 传递
image_url = "./zhinengjiaowu/jiemianpng.png"

with open(image_url, "rb") as image_file:
    # 读取二进制数据并转为 base64 字符串
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    encoded_string = f"data:image/png;base64,{encoded_string}"

messages = [{
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': '描述这幅图',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                    encoded_string,
            },
        }],
    }]

# messages = [
#     {
#         "role": "user",
#         "content": "你好，请介绍一下你自己。" # 只发文字，不带 image_url
#     }
# ]

print("正在发送请求到服务端...")

try:
    # 4. 发送请求 (流式或非流式)
    # 这里演示非流式 (等待生成完一次性返回)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=512,  # 控制生成长度
        temperature=0.0, # 控制随机性
    )
    

    # 5. 获取并打印结果
    content = response.choices[0].message.content
    print("-" * 30)
    print(content)
    print("-" * 30)

except Exception as e:
    print(f"连接失败: {e}")
    print("请检查：\n1. 服务端是否在 AutoDL 上成功启动？\n2. SSH 隧道/代理是否已连接到 8000 端口？")