
import openai
import json
from typing import List, Dict
from conf import settings
from tuijianxiton.prompt import AI_TAG_PROMPT

# 初始化 OpenAI 客户端
client = openai.Client(base_url=settings.base_url, api_key=settings.api_key)

def extract_user_tags(chat_history: str, tag_library: List[Dict]) -> str:
    """
    使用大模型根据聊天记录和标签库提取用户标签。

    Args:
        chat_history (str): 用户的聊天记录字符串。
        tag_library (List[Dict]): 标准画像标签库。

    Returns:
        str: 大模型输出的 JSON 格式标签提取结果。
    """
    # 使用 format 方法将标签库内容注入到 TAG_LIST 占位符中
    # 将标签库转换为 JSON 字符串格式，确保中文不被转义
    tag_list_str = json.dumps(tag_library, ensure_ascii=False, indent=2)
    system_prompt = AI_TAG_PROMPT.format(TAG_LIST=tag_list_str)

    # 调用大模型
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chat_history}
        ],
        temperature=0.0, # 设置为 0 以获得更确定的结果
        response_format={"type": "json_object"} # 强制输出 JSON
    )

    return response.choices[0].message.content
