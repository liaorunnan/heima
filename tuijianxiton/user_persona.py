import time
import math
import json
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统
    基于 'Long-short term Interest Split' (长短期兴趣分离) 理论构建。
    """
    
    def __init__(self):
        # 短期意图 (Intent): 侧重当下想买什么，包含类目约束
        # 结构: { category: { tag_name: { value, score, ts } } }
        self.short_term_intents: Dict[str, Dict[str, Any]] = {} 
        
        # 长期属性 (Trait/Preference): 侧重用户是什么样的人
        # 结构: { tag_name: { value, score, ts } }
        self.long_term_traits: Dict[str, Dict[str, Any]] = {}
        
        # 定义衰减系数 (Lambda)
        self.ALPHA_INTENT = 0.2  # 意图衰减极快 (小时)
        self.ALPHA_TRAIT = 0.005 # 属性衰减极慢 (小时)

    def _calculate_decay(self, last_update_ts: float, lambda_factor: float) -> float:
        """计算指数衰减因子"""
        if not last_update_ts: return 1.0
        hours_diff = (time.time() - last_update_ts) / 3600
        return math.exp(-lambda_factor * max(0, hours_diff))

    def _convert_tier_to_score(self, tier: str) -> float:
        tier_map = {"Tier S": 1.0, "Tier A": 0.8, "Tier B": 0.5}
        return tier_map.get(tier, 0.5)

    def update_persona(self, llm_output: Dict[str, Any]):
        """
        根据 LLM 结构化输出更新画像
        llm_output 格式: { "target_category": "...", "tags": [...] }
        """
        current_time = time.time()
        target_category = llm_output.get("target_category", "unknown")
        tags = llm_output.get("tags", [])

        # 定义哪些标签属于“短期意图”(与商品直接相关)
        # 只有这些标签会进入 short_term_intents 并带有类目约束
        INTENT_TAG_NAMES = [
            "极致性价比", "参数党", "物流焦虑", "价格敏感度", 
            "决策风格", "品牌倾向", "促销反应", "颜值主义", "功能实用派"
        ]

        print(f"--- 正在处理类目: [{target_category}] 的新信号 (Tags: {len(tags)}) ---")

        # 1. 更新短期意图 (Short-term Intent)
        if target_category != "unknown":
            if target_category not in self.short_term_intents:
                self.short_term_intents[target_category] = {}
            
            cat_intents = self.short_term_intents[target_category]
            
            # 衰减旧意图
            for t_name, data in list(cat_intents.items()):
                decay = self._calculate_decay(data['ts'], self.ALPHA_INTENT)
                data['score'] *= decay
                if data['score'] < 0.1: del cat_intents[t_name]

            # 插入新意图 (过滤非意图标签)
            for item in tags:
                tag_name = item.get('tag_name')
                attr_value = item.get('attribute_value')
                base_score = self._convert_tier_to_score(item.get('tier', 'Tier B'))
                
                if tag_name in INTENT_TAG_NAMES:
                    cat_intents[tag_name] = {
                        "value": attr_value,
                        "score": base_score * 1.5,
                        "ts": current_time
                    }

        # 2. 更新长期特质 (Long-term Trait)
        # 所有标签都进入长期特质，作为用户画像的基石
        for item in tags:
            tag_name = item.get('tag_name')
            attr_value = item.get('attribute_value')
            base_score = self._convert_tier_to_score(item.get('tier', 'Tier B'))

            if tag_name in self.long_term_traits:
                data = self.long_term_traits[tag_name]
                decay = self._calculate_decay(data['ts'], self.ALPHA_TRAIT)
                new_score = (data['score'] * decay) + (base_score * 0.1)
                self.long_term_traits[tag_name] = {
                    "value": attr_value,
                    "score": min(new_score, 5.0),
                    "ts": current_time
                }
            else:
                self.long_term_traits[tag_name] = {
                    "value": attr_value,
                    "score": base_score * 0.2,
                    "ts": current_time
                }

    def get_final_persona(self) -> Dict[str, Any]:
        return {
            "short_term_intents": self.short_term_intents,
            "long_term_traits": self.long_term_traits
        }

    def debug_print(self):
        import pprint
        print("\n=== 用户画像当前状态 ===")
        print(">> 短期意图 (Intents by Category):")
        pprint.pprint(self.short_term_intents)
        print("\n>> 长期特质 (User Traits):")
        pprint.pprint(self.long_term_traits)
        print("========================\n")

if __name__ == "__main__":
    # 测试代码
    system = UserPersonaSystem()

    # 模拟场景：
    # 1. 历史数据：用户长期关注"极致性价比"和"参数党" (假设一天前更新)
    print("Initialize with history...")
    yesterday = time.time() - 86400
    system.long_term_profile = {
        '极致性价比': {'score': 4.5, 'ts': yesterday},
        '参数党': {'score': 3.0, 'ts': yesterday}
    }
    
    # 2. 实时数据：用户今天突然表现出"颜值主义"和"冲动消费"
    # 假设这是从 llm_tagger.py 或 extracted_tags.json 读入的数据
    new_tags = [
        {"name": "颜值主义", "tier": "Tier S"},
        {"name": "冲动消费", "tier": "Tier A"},
        # 用户同时也提到了性价比，但这次是实时的
        {"name": "极致性价比", "tier": "Tier B"} 
    ]
    
    system.update_persona(new_tags)
    
    # 3. 打印结果
    # 预期：
    # - "颜值主义" 在短期画像中分数很高 (1.5)，长期画像中刚起步 (0.2)。
    # - "极致性价比" 在短期画像中有分数 (0.75)，在长期画像中依然很高 (但会略微衰减后累加)。
    system.debug_print()
