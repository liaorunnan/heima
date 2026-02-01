import time
import math
import json
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统
    实现了全局特质与品类意图的分离存储，并支持多意图并行更新。
    """
    
    def __init__(self):
        # 全局特质 (Global Traits): 长期稳定的属性，如身份、性格、整体消费观
        # 结构: { tag_code: { value, score, ts, is_instruction } }
        self.global_traits: Dict[str, Dict[str, Any]] = {}
        
        # 品类意图 (Category Intents): 针对特定品类的短期需求
        # 结构: { category: { tag_code: { value, score, ts } } }
        self.category_intents: Dict[str, Dict[str, Any]] = {}
        
        # 衰减系数
        self.ALPHA_GLOBAL = 0.005 # 全局特质衰减慢 (小时)
        self.ALPHA_INTENT = 0.2   # 品类意图衰减极快 (小时)

    def _calculate_decay(self, last_update_ts: float, lambda_factor: float) -> float:
        """计算指数衰减因子"""
        if not last_update_ts: return 1.0
        hours_diff = (time.time() - last_update_ts) / 3600
        return math.exp(-lambda_factor * max(0, hours_diff))

    def _convert_tier_to_score(self, tier: str) -> float:
        """将 Tier 等级转换为数值分数"""
        tier_map = {"Tier S": 1.0, "Tier A": 0.8, "Tier B": 0.5}
        return tier_map.get(tier, 0.5)

    def update_persona(self, llm_output: Dict[str, Any]):
        """
        根据 LLM 的多维度结构化输出更新画像
        llm_output 格式: { "user_profile_update": { "global_traits": [...] }, "intents": [...] }
        """
        current_time = time.time()
        
        # 1. 更新全局特质
        profile_update = llm_output.get("user_profile_update", {})
        new_global_traits = profile_update.get("global_traits", [])
        
        for item in new_global_traits:
            code = item.get("tag_code")
            val = item.get("value")
            tag_type = item.get("type", "concern")
            conf = item.get("confidence", "Tier B")
            is_inst = item.get("is_instruction", False)
            
            # 基础分数受 Type 影响
            # requirement 给更高分，inquiry 给较低分
            type_weight = {"requirement": 1.2, "concern": 1.0, "inquiry": 0.6}.get(tag_type, 1.0)
            base_score = self._convert_tier_to_score(conf) * type_weight
            
            if code in self.global_traits:
                data = self.global_traits[code]
                decay = self._calculate_decay(data['ts'], self.ALPHA_GLOBAL)
                new_score = (data['score'] * decay) + (base_score * 0.1)
                self.global_traits[code] = {
                    "value": val,
                    "score": min(new_score, 5.0),
                    "ts": current_time,
                    "type": tag_type,
                    "is_instruction": is_inst
                }
            else:
                self.global_traits[code] = {
                    "value": val,
                    "score": base_score * 0.2,
                    "ts": current_time,
                    "type": tag_type,
                    "is_instruction": is_inst
                }
            
            if is_inst:
                print(f"  [指令识别] 发现运营指令: {code} -> {val}")

        # 2. 更新品类意图
        new_intents = llm_output.get("intents", [])
        for intent in new_intents:
            cat = intent.get("category", "unknown")
            if cat == "unknown": continue
            
            if cat not in self.category_intents:
                self.category_intents[cat] = {}
            
            cat_data = self.category_intents[cat]
            for code, data in list(cat_data.items()):
                decay = self._calculate_decay(data['ts'], self.ALPHA_INTENT)
                data['score'] *= decay
                if data['score'] < 0.1: del cat_data[code]
            
            specific_tags = intent.get("specific_tags", [])
            for tag in specific_tags:
                code = tag.get("tag_code")
                val = tag.get("value")
                tag_type = tag.get("type", "concern")
                
                # 意图权重受 Type 影响
                intent_weight = {"requirement": 2.0, "concern": 1.5, "inquiry": 0.8}.get(tag_type, 1.5)
                
                cat_data[code] = {
                    "value": val,
                    "score": intent_weight,
                    "ts": current_time,
                    "type": tag_type
                }
            print(f"  [意图更新] 已更新品类 [{cat}] 的意图标签 (Count: {len(specific_tags)})")

    def get_final_persona(self) -> Dict[str, Any]:
        """获取最终合并画像"""
        return {
            "global_traits": self.global_traits,
            "category_intents": self.category_intents
        }

    def debug_print(self):
        """调试打印画像状态"""
        import pprint
        print("\n=== 用户画像深度侧写 ===")
        print(">> 全局特质 (Global Traits):")
        pprint.pprint(self.global_traits)
        print("\n>> 品类意图 (Category Intents):")
        pprint.pprint(self.category_intents)
        print("========================\n")

if __name__ == "__main__":
    # 模拟运行
    system = UserPersonaSystem()
    
    # 模拟一次更新
    mock_llm_output = {
        "user_profile_update": {
            "global_traits": [
                {"tag_code": "life_stage", "value": "second_child_mom", "confidence": "Tier S"},
                {"tag_code": "logistics_preference", "value": "sf_express", "is_instruction": True}
            ]
        },
        "intents": [
            {
                "category": "smart_camera",
                "action": "buy",
                "specific_tags": [
                    {"tag_code": "security_concern", "value": "data_privacy"}
                ]
            },
            {
                "category": "electric_kettle",
                "action": "buy",
                "specific_tags": [
                    {"tag_code": "material_preference", "value": "316_stainless_steel"}
                ]
            }
        ]
    }
    
    system.update_persona(mock_llm_output)
    system.debug_print()
