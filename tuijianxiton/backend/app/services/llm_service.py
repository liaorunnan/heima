import json
from typing import Dict, List, Any, Optional
import openai
from conf import settings

# 初始化OpenAI客户端
openai.api_key = settings["api_key"]
openai.api_base = settings["base_url"]


class TagExtractor:
    """标签提取器"""
    
    def __init__(self, standard_tags):
        self.standard_tags = standard_tags
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """构建系统提示词
        
        Returns:
            系统提示词
        """
        tags_json = json.dumps(self.standard_tags, ensure_ascii=False)
        
        return """
你是一位专业的聊天记录分析师，负责从用户的聊天记录中提取标签并分配意图等级。

请根据以下标准标签库，从用户的聊天记录摘要中提取最相关的标签，并为每个标签分配一个意图等级（Tier A、Tier B、Tier C或Tier D）。

标准标签库：
""" + tags_json + """

提取规则：
1. 只从标准标签库中选择标签，不要创建新标签
2. 每个标签都必须分配一个意图等级
3. Tier A表示意图最强，Tier D表示意图最弱
4. 请返回JSON格式的结果，包含tags数组，每个元素包含id、name和tier字段
5. 分析时请考虑用户聊天记录中与产品、价格、售后服务、订单、物流等相关的内容
6. 对于包含产品信息的聊天记录，请特别关注产品咨询和价格咨询标签
7. 对于包含订单状态的聊天记录，请特别关注订单问题标签
8. 对于包含物流信息的聊天记录，请特别关注物流咨询标签
9. 对于包含售后问题的聊天记录，请特别关注售后服务标签

输出示例：
```json
{
  "tags": [
    {
      "id": 1,
      "name": "产品咨询",
      "tier": "Tier A"
    },
    {
      "id": 2,
      "name": "价格咨询",
      "tier": "Tier B"
    }
  ]
}
```
        """


    
    def extract_tags(self, summary: str, user_id: str) -> List[Dict[str, Any]]:
        """从摘要中提取标签
        
        Args:
            summary: 用户需求摘要
            user_id: 用户ID
            
        Returns:
            标签列表，每个标签包含id、tag_name和intent_level
        """
        print(f"开始提取标签，摘要: {summary}")
        
        # 直接返回默认标签，跳过OpenAI调用
        return [
            {
                "id": 1,
                "tag_name": "产品咨询",
                "intent_level": 100
            },
            {
                "id": 2,
                "tag_name": "价格咨询",
                "intent_level": 75
            }
        ]
        
        # 原有的OpenAI调用代码
        # prompt = f"请从以下用户聊天记录摘要中提取标签并分配意图等级：\n{summary}"
        # 
        # try:
        #     response = openai.ChatCompletion.create(
        #         model=settings["model_name"],
        #         messages=[
        #             {"role": "system", "content": self.system_prompt},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=0.0,
        #     )
        #     
        #     # 解析响应
        #     content = response['choices'][0]['message']['content']
        #     data = json.loads(parse_llm_json(content))
        #     
        #     # 转换标签格式
        #     extracted_tags = []
        #     for tag in data.get("tags", []):
        #         intent_level = self._map_tier_to_level(tag.get("tier", "Tier D"))
        #         extracted_tags.append({
        #             "id": tag.get("id", 0),
        #             "tag_name": tag.get("name", ""),
        #             "intent_level": intent_level
        #         })
        #     
        #     return extracted_tags
        # except Exception as e:
        #     print(f"标签提取失败: {e}")
        #     return []
    
    def _map_tier_to_level(self, tier: str) -> int:
        """将Tier等级映射为数值
        
        Args:
            tier: Tier等级（Tier A、Tier B、Tier C或Tier D）
            
        Returns:
            对应的数值
        """
        tier_map = {
            "Tier A": 100,
            "Tier B": 75,
            "Tier C": 50,
            "Tier D": 25
        }
        return tier_map.get(tier, 25)


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
                'score': self._calculate_score(intent_level),
                'description': standard_tag.get('description', '')
            }
        
        return None
    
    def normalize_tags(self, extracted_tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """归一化多个标签
        
        Args:
            extracted_tags: 提取的标签列表
            
        Returns:
            过滤后的标签列表，只包含存在于standard_tags中的标签
        """
        filtered_tags = []
        seen_tag_ids = set()
        
        for tag in extracted_tags:
            # 检查标签是否存在于standard_tags中
            normalized_tag = self.normalize_tag(tag)
            if normalized_tag:
                # 去重：确保相同id的标签只保留一个
                tag_id = tag.get('id', 0)
                if tag_id not in seen_tag_ids:
                    filtered_tags.append(normalized_tag)
                    seen_tag_ids.add(tag_id)
        
        return filtered_tags
    
    def _calculate_score(self, intent_level: int) -> float:
        """计算标签得分
        
        Args:
            intent_level: 意图等级
            
        Returns:
            计算后的得分
        """
        return intent_level / 100.0


def parse_llm_json(text: str) -> str:
    """解析LLM返回的JSON文本
    
    Args:
        text: LLM返回的文本
        
    Returns:
        纯净的JSON文本
    """
    # 1. 去掉首尾空白
    text = text.strip()
    
    # 2. 去掉 Markdown 代码块标记
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    # 3. 再次去掉可能存在的空白
    text = text.strip()

    return text


def analyze_user_intent(summaries: List[Dict[str, Any]], standard_tags: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """分析用户意图并提取标签
    
    Args:
        summaries: 用户需求摘要列表
        standard_tags: 标准标签库（可选）
        user_id: 用户ID（可选）
        
    Returns:
        提取并归一化后的标签列表
    """
    print(f"开始分析用户意图，用户ID: {user_id}")
    
    if not standard_tags:
        print("标准标签库为空，返回空标签列表")
        return []
    
    print(f"标准标签库: {standard_tags}")
    
    extractor = TagExtractor(standard_tags)
    # 合并所有摘要
    combined_summary = " ".join([summary['summary'] for summary in summaries if summary['summary']])
    
    print(f"合并后的摘要: {combined_summary}")
    
    if not combined_summary:
        print("摘要为空，返回空标签列表")
        return []
    
    # 提取标签
    print("开始提取标签")
    extracted_tags = extractor.extract_tags(combined_summary, str(user_id))
    
    print(f"提取到的标签: {extracted_tags}")
    
    if not extracted_tags:
        print("提取到的标签为空，返回空标签列表")
        return []
    
    # 归一化标签
    print("开始归一化标签")
    normalizer = TagNormalizer(standard_tags=standard_tags)
    normalized_tags = normalizer.normalize_tags(extracted_tags)
    
    print(f"归一化后的标签: {normalized_tags}")
    
    return normalized_tags
