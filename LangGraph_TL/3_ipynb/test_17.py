"""
提示工程
langchain 中使用 ChatPromptTemplate 提示词来格式化 json 输出
"""


import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json`",
        ),
        ("human", "{query}"),
    ]
)

chain = prompt | llm 

ans = chain.invoke({"query": "我叫木羽，今年28岁，邮箱地址是snow@gmial.com，电话是1234567052"})

print(ans.content)




