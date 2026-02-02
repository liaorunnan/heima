import json
from typing import List, Dict
import openai
from conf import settings
from tuijianxiton.prompt import AI_TAG_PROMPT

# 初始化 OpenAI 客户端
client = openai.Client(base_url=settings.base_url, api_key=settings.api_key)

def parse_llm_json(text: str) -> str:
    """
    清洗大模型返回的 JSON 字符串，去掉 Markdown 标记。
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def extract_user_tags(chat_history: str, tag_library: List[Dict]) -> str:
    """
    使用大模型根据聊天记录和标签库提取用户标签。

    Args:
        chat_history (str): 用户的聊天记录字符串。
        tag_library (List[Dict]): 标准画像标签库。

    Returns:
        str: 大模型输出的 JSON 格式标签提取结果。
    """
    # 将标签库转换为 JSON 字符串格式
    tag_list_str = json.dumps(tag_library, ensure_ascii=False, indent=2)
    system_prompt = AI_TAG_PROMPT.format(TAG_LIST=tag_list_str)

    # 调用大模型
    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_history}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # 强制输出 JSON
        )
        
        content = response.choices[0].message.content
        # 使用 parse_llm_json 进行清洗，确保 JSON 解析成功
        return parse_llm_json(content)
    except Exception as e:
        print(f"提取标签失败: {e}")
        return "{}"
