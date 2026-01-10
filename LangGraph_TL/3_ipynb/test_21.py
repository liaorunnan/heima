
"""
第三种结构化输出：
    使用 Json Schema 做结构化输出.
    在调用模型时，使用 with_structured_output 方法来指定输出结构。

"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import Optional
from typing_extensions import Annotated, TypedDict
from langchain.chat_models import init_chat_model

# 定义 JSON Schema
json_schema = {
    "title": "user_info",
    "description": "Extracted user information",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The user's name",
        },
        "age": {
            "type": "integer",
            "description": "The user's age",
            "default": None,
        },
        "email": {
            "type": "string",
            "description": "The user's email address",
        },
        "phone": {
            "type": "string",
            "description": "The user's phone number",
            "default": None,
        },
    },
    "required": ["name", "email"],
}


llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


structured_llm = llm.with_structured_output(json_schema)

# 从非结构化文本中提取用户信息
extracted_user_info = structured_llm.invoke("我叫木羽，今年28岁，邮箱地址是snow@gmial.com，电话是1234567052")

# isinstance 函数用于判断一个对象是否是一个已知的类型，或者是该类型的子类的实例，
# TypedDict 不支持使用 isinstance() 进行实例检查。这里注释掉，不需要
# if isinstance(extracted_user_info, UserInfo):
#     print("执行节点A的逻辑")
# else:
#     print("执行节点B的逻辑")

print(extracted_user_info)



