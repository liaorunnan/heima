"""
langchain 流式输出

""" 


import os
import asyncio
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(override=True)


# 生成模型实例
llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )



async def stream_function():
    chunks = []
    async for chunk in llm.astream("你好，请你详细的介绍一下你自己。"):
        chunks.append(chunk)
        print(chunk.content, end="|", flush=True)

    print(chunks[0], end="\n")
    print(chunks[1], end="\n")
    print(chunks[2], end="\n")
    print(chunks[3], end="\n")
    print(chunks[4], end="\n")
    print(chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4], end="\n")


asyncio.run(stream_function())



