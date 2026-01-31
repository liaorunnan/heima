import json
from typing import Dict, List, Any, Optional

from src.tools.porstgreDB_tools import get_user_tags_db
from src.utils.llm_utils import get_default_llm
from langchain.agents import create_agent

from src.workflow.offline_profile.config1 import LLM_CONFIG, INTENT_SCORE_MAP, logger
from src.workflow.offline_profile.utils import normalize_tag_name, is_valid_tag
from src.prompts.prompts_tag import AI_TAG_PROMPT


class TagExtractor:
    """标签提取器"""
    
    def __init__(self,standard_tags):
        self.llm = get_default_llm()



        self.system_prompt = AI_TAG_PROMPT.format(TAG_LIST=json.dumps(standard_tags, ensure_ascii=False))
        
        self.agent = create_agent(
            model=self.llm,
            name="tag_extractor",
            system_prompt=self.system_prompt,
        )

        self.TierArray = {
            "Tier A":100,
            "Tier B":75,
            "Tier C":50,
            "Tier D":25,
        }

    
    def extract_tags(self, summary: str,user_id: str) -> List[Dict[str, Any]]:
        """从摘要中提取标签
        
        Args:
            summary: 用户需求摘要
            
        Returns:
            标签列表，每个标签包含tag_name和intent_level
        """
        prompt = f"请从以下用户需求摘要中提取标签并分配意图等级：\n{summary}"
        
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
            )['messages'][-1].content

            
            
            # 解析JSON响应
            data = json.loads(parse_llm_json(response))

            logger.info(f"大模型打分结果: {data}")

           

            
            # 验证标签格式
            valid_tags = []

            # 先把人工的加入进来
            user_tags_db = get_user_tags_db(str(user_id))

            logger.info(f"用户 {user_id} 人工打标标签: {user_tags_db}")

            for tag in user_tags_db:
                if tag['create_type'] == 'user':
                    valid_tags.append({
                        'id': tag['tag_id'],
                        'tag_name': tag['tag_name'],
                        'intent_level': 100, # 人工打标默认为Tier A
                    })

            
            for tag in data:
                if 'name' in tag and 'tier' in tag:
                    valid_tags.append({
                        'id': tag.get('id', 0),
                        'tag_name': tag.get('name', 0),
                        'intent_level': self.TierArray.get(tag.get('tier', 0), 0),
                    })
            
            return valid_tags
        except Exception as e:
            logger.error(f"标签提取失败: {e}")
            return []

    

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
class TagNormalizer:
    """标签归一化器"""
    
    def __init__(self, standard_tags: Optional[List[Dict[str, Any]]] = None):
       
        self.standard_tags = standard_tags

        self.tag_id_map = {tag['id']: tag for tag in standard_tags} if standard_tags else {}
        
    
   
    
    def normalize_tag(self, extracted_tag: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将提取的标签归一化到标准标签库
        
        Args:
            extracted_tag: 提取的标签，包含id、tag_name和intent_level
            
        Returns:
            归一化后的标签，如果不存在于标准标签库中则返回None
        """
        tag_id = extracted_tag.get('id', 0)
        tag_name = extracted_tag.get('tag_name', '')
        intent_level = extracted_tag['intent_level']
        
     
        standard_tag = self.tag_id_map.get(tag_id)
        if standard_tag and standard_tag['name'] == tag_name:
            return {
                'id': standard_tag['id'],
                'name': standard_tag['name'],
                'intent_level': intent_level,
                'score': INTENT_SCORE_MAP.get(intent_level, 0.1),
                'description': standard_tag.get('description', '')
            }
        
        return None
    
    def normalize_tags(self, extracted_tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """归一化多个标签
        
        Args:
            extracted_tags: 提取的标签列表
            
        Returns:
            过滤后的标签列表，只包含存在于standard_tags中的标签，结构与extracted_tags相同
        """
        filtered_tags = []
        seen_tag_ids = set()
        
        for tag in extracted_tags:
            # 检查标签是否存在于standard_tags中
            normalized_tag = self.normalize_tag(tag)
            if normalized_tag:
                # 保留原始标签结构，只需要确保标签存在于standard_tags中
                original_tag = {
                    'id': tag.get('id', 0),
                    'tag_name': tag.get('tag_name', ''),
                    'intent_level': tag.get('intent_level', 0)
                }
                # 去重：确保相同id的标签只保留一个
                tag_id = tag.get('id', 0)
                if tag_id not in seen_tag_ids:
                    filtered_tags.append(original_tag)
                    seen_tag_ids.add(tag_id)
        
        return filtered_tags


def analyze_user_intent(summaries: List[Dict[str, Any]], standard_tags: Optional[List[Dict[str, Any]]] = None,user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """分析用户意图并提取标签
    
    Args:
        summaries: 用户需求摘要列表
        standard_tags: 标准标签库（可选）
        
    Returns:
        提取并归一化后的标签列表
    """
    extractor = TagExtractor(standard_tags)
    # 合并所有摘要
    combined_summary = " ".join([summary['summary'] for summary in summaries if summary['summary']])
    
    if not combined_summary:
        return []
    
    # 提取标签
    
    extracted_tags = extractor.extract_tags(combined_summary,user_id)
    
    if not extracted_tags:
        return []

    logger.info(f"提取标签: {extracted_tags}")
    
    # 归一化标签
    normalizer = TagNormalizer(standard_tags=standard_tags)
    normalized_tags = normalizer.normalize_tags(extracted_tags)
    
    logger.info(f"提取并归一化后的标签数量: {len(normalized_tags)}")
    return normalized_tags





def calculate_tag_score(tag: Dict[str, Any], days_since_last_update: int = 0) -> float:
    """计算标签得分
    
    Args:
        tag: 标签字典
        days_since_last_update: 距离上次更新的天数
        
    Returns:
        计算后的得分
    """
    # 基础得分
    base_score = tag.get('score', 0.1)
    
    # 时间衰减
    decay_factor = 0.95 ** days_since_last_update
    
    return base_score * decay_factor


    
    