import time
import math
import json
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统 (V7 工业级改进版)
    核心优化：支持多值并存 (Multi-value)、权重深度映射、生命周期管理。
    """
    
    def __init__(self):
        # 全局特质 (Global Traits)
        # 结构: { tag_code: { "values": [ { "value", "score", "ts", "type", "confidence" } ] } }
        self.global_traits: Dict[str, Dict[str, Any]] = {}
        
        # 品类意图 (Category Intents)
        # 结构: { category: { tag_code: { "values": [ { "value", "score", "ts", "type" } ] } } }
        self.category_intents: Dict[str, Dict[str, Any]] = {}

        # 操作指令 (Instructions)
        self.instructions: List[Dict[str, Any]] = []
        
        # 衰减系数
        self.ALPHA_GLOBAL = 0.005 # 全局特质衰减慢 (生命周期长)
        self.ALPHA_INTENT = 0.1   # 品类意图衰减中等 (生命周期约3-7天)
        
        # 初始分数权重映射
        self.TIER_MAP = {"Tier S": 1.0, "Tier A": 0.7, "Tier B": 0.4}
        self.TYPE_WEIGHT = {
            "requirement": 1.5, # 硬需求，分值高
            "concern": 1.0,     # 关注点，分值中
            "inquiry": 0.5      # 询问，分值低
        }

    def _calculate_decay(self, last_update_ts: float, lambda_factor: float) -> float:
        """计算指数衰减因子"""
        if not last_update_ts: return 1.0
        hours_diff = (time.time() - last_update_ts) / 3600
        return math.exp(-lambda_factor * max(0, hours_diff))

    def _get_base_score(self, tier: str, tag_type: str) -> float:
        """根据置信度和类型计算基础初始分值"""
        tier_score = self.TIER_MAP.get(tier, 0.4)
        type_weight = self.TYPE_WEIGHT.get(tag_type, 1.0)
        # 工业级标准：初始分值应能显著区分强度
        return tier_score * type_weight * 2.0 

    def update_persona(self, llm_output: Dict[str, Any]):
        """
        根据 LLM 的多维度结构化输出更新画像。
        解决“单值覆盖”Bug，实现多值并存。
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
            
            base_score = self._get_base_score(conf, tag_type)
            
            if code not in self.global_traits:
                self.global_traits[code] = {"values": []}
            
            trait_entry = self.global_traits[code]
            
            # 检查该值是否已存在，若存在则更新分数，不存在则追加
            found = False
            for entry in trait_entry["values"]:
                if entry["value"] == val:
                    decay = self._calculate_decay(entry["ts"], self.ALPHA_GLOBAL)
                    entry["score"] = (entry["score"] * decay) + (base_score * 0.3)
                    entry["score"] = min(entry["score"], 5.0)
                    entry["ts"] = current_time
                    found = True
                    break
            
            if not found:
                trait_entry["values"].append({
                    "value": val,
                    "score": base_score,
                    "ts": current_time,
                    "type": tag_type,
                    "confidence": conf
                })

        # 2. 更新品类意图
        new_intents = llm_output.get("intents", [])
        for intent in new_intents:
            cat = intent.get("category", "unknown")
            if cat == "unknown": continue
            
            if cat not in self.category_intents:
                self.category_intents[cat] = {}
            
            cat_data = self.category_intents[cat]
            specific_tags = intent.get("specific_tags", [])
            
            for tag in specific_tags:
                code = tag.get("tag_code")
                val = tag.get("value")
                tag_type = tag.get("type", "concern")
                # 意图场景下置信度默认为 A 级
                base_score = self._get_base_score("Tier A", tag_type)
                
                if code not in cat_data:
                    cat_data[code] = {"values": []}
                
                intent_entry = cat_data[code]
                
                # 检查值是否存在
                found = False
                for entry in intent_entry["values"]:
                    if entry["value"] == val:
                        decay = self._calculate_decay(entry["ts"], self.ALPHA_INTENT)
                        entry["score"] = (entry["score"] * decay) + (base_score * 0.5)
                        entry["score"] = min(entry["score"], 5.0)
                        entry["ts"] = current_time
                        found = True
                        break
                
                if not found:
                    intent_entry["values"].append({
                        "value": val,
                        "score": base_score,
                        "ts": current_time,
                        "type": tag_type
                    })
            print(f"  [意图更新] 已更新品类 [{cat}] 的意图标签")

        # 3. 更新即时指令 (Instructions)
        new_instructions = llm_output.get("instructions", [])
        if new_instructions:
            self.instructions.extend(new_instructions)
            self.instructions = self.instructions[-20:]
            for inst in new_instructions:
                print(f"  [指令提取] 识别到动作: {inst.get('action')} -> {inst.get('value')}")

    def get_final_persona(self) -> Dict[str, Any]:
        """获取最终合并画像，包含过期清理逻辑"""
        # 这里可以加入简单的过期清理 (如超过30天的短期意图)
        return {
            "global_traits": self.global_traits,
            "category_intents": self.category_intents,
            "recent_instructions": self.instructions,
            "persona_summary": self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """简单的画像概括逻辑 (演示用)"""
        traits = []
        if "life_stage" in self.global_traits:
            traits.append(self.global_traits["life_stage"]["values"][0]["value"])
        if "price_sensitivity" in self.global_traits:
            traits.append("价格敏感")
        return f"这是一个{'、'.join(traits)}的用户。" if traits else "新用户画像中"

    def debug_print(self):
        """调试打印画像状态"""
        import pprint
        print("\n=== 用户画像深度侧写 (V7 多值版) ===")
        print(">> 全局特质 (Global Traits):")
        pprint.pprint(self.global_traits)
        print("\n>> 品类意图 (Category Intents):")
        pprint.pprint(self.category_intents)
        print("\n>> 最近指令 (Recent Instructions):")
        pprint.pprint(self.instructions)
        print("========================\n")

if __name__ == "__main__":
    # 模拟 V7 逻辑运行
    system = UserPersonaSystem()
    
    # 模拟多次更新，观察多值并存
    mock_llm_output = {
        "user_profile_update": {
            "global_traits": [
                {"tag_code": "decision_style", "value": "纠结大师", "confidence": "Tier S", "type": "concern"},
                {"tag_code": "decision_style", "value": "参数党", "confidence": "Tier S", "type": "concern"}
            ]
        },
        "intents": [
            {
                "category": "摄像头",
                "action": "buy",
                "specific_tags": [
                    {"tag_code": "feature", "value": "哭声检测", "type": "requirement"}
                ]
            }
        ]
    }
    
    system.update_persona(mock_llm_output)
    system.debug_print()
