from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage,SystemMessage,HumanMessage
from conf import settings


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent




model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to Chinese."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "我喜欢编程。"},
    {"role": "user", "content": "Translate: I love building applications."}
]

result = model.invoke(conversation)

print(result.content)
# 我喜欢开发应用程序。

message = [
    SystemMessage("You are a helpful assistant that translates English to Chinese."),
    HumanMessage("Translate: I love programming."),
    AIMessage("我喜欢编程。"),
    HumanMessage("Translate: I love building applications."),
]

print("*"*30)

result = model.invoke(message)

print(result.content)

print("*"*30)

# 同步方式 (当前使用的方式)
full = None  # None | AIMessageChunk
for chunk in model.stream("天空是什么颜色的？"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

# print(full.content_blocks)

print("*"*30)
print("以下是异步")


# 异步方式示例 (需要在异步环境中运行)
import asyncio
from langchain.modelstart import message

async def async_stream_example():
    full = None
    async for chunk in model.astream("天空是什么颜色的？"):
        full = chunk if full is None else full + chunk
        print(full.text)

# 在异步环境中运行
asyncio.run(async_stream_example())
