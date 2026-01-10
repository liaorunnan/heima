"""
将输出格式化为 数据库表中的几个字段的格式 或 正常llm回答的格式
"""


from typing import Union, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# 定义数据库插入的用户信息模型
class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name: str = Field(description="The name of the user")
    age: Optional[int] = Field(description="The age of the user")
    email: str = Field(description="The email address of the user")
    phone: Optional[str] = Field(description="The phone number of the user")


# 定义正常生成模型回复的模型
class ConversationalResponse(BaseModel):
    """Respond to the user's query in a conversational manner. Be kind and helpful."""
    response: str = Field(description="A conversational response to the user's query")


# 定义最终响应模型，可以是用户信息或一般响应
class FinalResponse(BaseModel):
    final_output: Union[UserInfo, ConversationalResponse]

llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek", # 该参数可选，留空时自动推断
        api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取
        base_url=os.getenv("DEEPSEEK_URL"),
        temperature=0,
    )

structured_llm = llm.with_structured_output(FinalResponse)

# 从非结构化文本中提取用户信息或进行一般对话响应
extracted_user_info_1 = structured_llm.invoke("你好")

print("测试1：", extracted_user_info_1)

extracted_user_info_2 = structured_llm.invoke("我叫木羽，今年28岁，邮箱地址是snow@gmial.com，电话是1234567052")

print("测试2：", extracted_user_info_2.final_output)




