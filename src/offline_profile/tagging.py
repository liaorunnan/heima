import json
from typing import Dict, List, Any, Optional

from src.utils.llm_utils import get_default_llm
from src.workflow.offline_profile.config1 import logger
from src.prompts.prompts_tag import AI_TAG_PROMPT


def parse_llm_json(text):
    # 1. 去掉首尾空白
    text = text.strip()
    
    # 2. 去掉 Markdown 代码块标记
    if text.startswith("```json"):
        text = text[7:]  # 去掉 ```json
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    # 3. 再次去掉可能存在的空白
    text = text.strip()

    return text


class TagExtractor:
    """标签提取器"""
    
    def __init__(self, standard_tags):
        self.llm = get_default_llm()
        tag_list_str = json.dumps(standard_tags, ensure_ascii=False) if standard_tags else "[]"
        self.system_prompt = AI_TAG_PROMPT.format(TAG_LIST=tag_list_str)

    def extract_tags(self, summary: str, user_id: str) -> Dict[str, Any]:
        """从摘要中提取标签
        
        Args:
            summary: 用户需求摘要
            user_id: 用户ID
            
        Returns:
            结构化的标签数据 (global_traits, intents, instructions)
        """
        prompt = f"请从以下用户需求摘要中提取标签：\n{summary}"
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.invoke(messages).content
            
            # 解析JSON响应
            data = json.loads(parse_llm_json(response))
            logger.info(f"大模型提取结果: {data}")
            return data
            
        except Exception as e:
            logger.error(f"标签提取失败: {e}")
            return {}


def analyze_user_intent(summaries: List[Dict[str, Any]], standard_tags: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """分析用户意图并提取标签
    
    Args:
        summaries: 用户需求摘要列表
        standard_tags: 标准标签库（可选）
        user_id: 用户ID
        
    Returns:
        提取的结构化数据
    """
    extractor = TagExtractor(standard_tags)
    # 合并所有摘要
    combined_summary = " ".join([summary.get('summary', '') for summary in summaries if summary.get('summary')])
    
    if not combined_summary:
        return {}
    
    # 提取标签
    extracted_data = extractor.extract_tags(combined_summary, user_id)
    
    return extracted_data
