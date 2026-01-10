"""
提示工程
langchain 中使用 ChatPromptTemplate 提示词来格式化 json 输出
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
import json
import re
from typing import List

def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # 定义正则表达式模式来匹配JSON块
    pattern = r"```json(.*?)```"

    # 在字符串中查找模式的所有非重叠匹配
    matches = re.findall(pattern, text, re.DOTALL)

    # 返回匹配的JSON字符串列表，去掉任何开头或结尾的空格
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")


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

chain = prompt | llm | extract_json

ans = chain.invoke({"query": "我叫木羽，今年18岁，邮箱地址是snow@gmial.com，电话是1234567052"})

print(ans)




