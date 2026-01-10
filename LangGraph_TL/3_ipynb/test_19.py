"""
第一种结构化输出：
    使用 Pydantic 做结构化输出.
    在调用模型时，使用 with_structured_output 方法来指定输出结构。

"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# 定义 Pydantic 模型
class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email address of the user")
    phone: Optional[str] = Field(description="The phone number of the user")



llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )


structured_llm = llm.with_structured_output(UserInfo)

# 从非结构化文本中提取用户信息
extracted_user_info = structured_llm.invoke("我叫木羽，今年28岁，邮箱地址是snow@gmial.com，电话是1234567052")

# isinstance 函数用于判断一个对象是否是一个已知的类型，或者是该类型的子类的实例
if isinstance(extracted_user_info, UserInfo):
    print("执行节点A的逻辑")
else:
    print("执行节点B的逻辑")

print(extracted_user_info)




