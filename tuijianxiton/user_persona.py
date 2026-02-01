import time
import math
import json
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统 (V8 专家改进版)
    核心优化：语义归一化、冲突检测 (Value Seeker)、权重对齐、生命周期/过期管理。
    """
    
    def __init__(self):
        # 全局特质 (Global Traits)
        self.global_traits: Dict[str, Dict[str, Any]] = {}
        # 品类意图 (Category Intents)
        self.category_intents: Dict[str, Dict[str, Any]] = {}
        # 操作指令 (Instructions)
        self.instructions: List[Dict[str, Any]] = []
        
        # 衰减系数
        self.ALPHA_GLOBAL = 0.005 
        self.ALPHA_INTENT = 0.1   
        
        # 权重映射
        self.TIER_MAP = {"Tier S": 1.0, "Tier A": 0.7, "Tier B": 0.4}
        self.TYPE_WEIGHT = {"requirement": 1.5, "concern": 1.0, "inquiry": 0.5}
        
        # 语义归一化映射表
        self.TAG_NORMALIZATION = {
            "保守主义": "low_risk_tolerance",
            "cautious": "low_risk_tolerance",
            "风险厌恶": "low_risk_tolerance",
            "砍价高手": "bargaining_expert",
            "极致性价比": "price_sensitive_high"
        }

    def _calculate_decay(self, last_update_ts: float, lambda_factor: float) -> float:
        """计算指数衰减因子"""
        if not last_update_ts: return 1.0
        hours_diff = (time.time() - last_update_ts) / 3600
        return math.exp(-lambda_factor * max(0, hours_diff))

    def _get_base_score(self, tier: Optional[str], tag_type: str) -> float:
        """计算对齐后的基础分值"""
        # 补全缺失的置信度 (针对 Issue #1)
        if not tier:
            tier = "Tier A" if tag_type == "requirement" else "Tier B"
            
        tier_score = self.TIER_MAP.get(tier, 0.4)
        type_weight = self.TYPE_WEIGHT.get(tag_type, 1.0)
        return tier_score * type_weight * 2.0 

    def _normalize_value(self, val: str) -> str:
        """标签归一化 (针对 Issue #2)"""
        return self.TAG_NORMALIZATION.get(val, val)

    def update_persona(self, llm_output: Dict[str, Any]):
        """
        更新画像，包含归一化、冲突检测和分值对齐逻辑。
        """
        current_time = time.time()
        
        # 1. 更新全局特质
        profile_update = llm_output.get("user_profile_update", {})
        new_global_traits = profile_update.get("global_traits", [])
        
        for item in new_global_traits:
            code = item.get("tag_code")
            val = self._normalize_value(item.get("value")) # 归一化
            tag_type = item.get("type", "concern")
            conf = item.get("confidence")
            
            base_score = self._get_base_score(conf, tag_type)
            
            if code not in self.global_traits:
                self.global_traits[code] = {"values": []}
            
            trait_entry = self.global_traits[code]
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
                    "confidence": conf or "Tier B",
                    "expiry": None # 全局特质永不过期
                })

        # 2. 更新品类意图
        new_intents = llm_output.get("intents", [])
        for intent in new_intents:
            cat = intent.get("category", "unknown")
            if cat == "unknown": continue
            
            if cat not in self.category_intents:
                self.category_intents[cat] = {}
            
            cat_data = self.category_intents[cat]
            for tag in intent.get("specific_tags", []):
                code = tag.get("tag_code")
                val = self._normalize_value(tag.get("value"))
                tag_type = tag.get("type", "concern")
                conf = tag.get("confidence")
                
                base_score = self._get_base_score(conf, tag_type)
                
                if code not in cat_data:
                    cat_data[code] = {"values": []}
                
                intent_entry = cat_data[code]
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
                        "type": tag_type,
                        "expiry": current_time + (86400 * 30) # 30天过期 (针对 Issue #5)
                    })

        # 3. 更新即时指令
        new_instructions = llm_output.get("instructions", [])
        for inst in new_instructions:
            inst["expiry"] = current_time + 3600 # 1小时过期 (会话级别)
            self.instructions.append(inst)
        self.instructions = self.instructions[-20:]

        # 4. 冲突检测与衍生标签 (针对 Issue #4)
        self._detect_conflicts(current_time)

    def _detect_conflicts(self, current_time: float):
        """冲突检测：极致性价比追求者识别"""
        has_high_price_sensitivity = False
        has_high_quality_requirement = False
        
        # 检查价格敏感度
        if "price_sensitivity" in self.global_traits:
            for v in self.global_traits["price_sensitivity"]["values"]:
                if v["value"] == "high" and v["score"] > 1.5:
                    has_high_price_sensitivity = True
        
        # 检查质量要求 (从意图或全局中寻找 requirement 类型的高分标签)
        for cat in self.category_intents.values():
            for tags in cat.values():
                for v in tags["values"]:
                    if v["type"] == "requirement" and v["score"] > 1.5:
                        has_high_quality_requirement = True
        
        if has_high_price_sensitivity and has_high_quality_requirement:
            if "trait_derivative" not in self.global_traits:
                self.global_traits["trait_derivative"] = {"values": []}
            
            found = False
            for v in self.global_traits["trait_derivative"]["values"]:
                if v["value"] == "value_seeker":
                    v["score"] = min(v["score"] + 0.5, 5.0)
                    v["ts"] = current_time
                    found = True
                    break
            
            if not found:
                self.global_traits["trait_derivative"]["values"].append({
                    "value": "value_seeker",
                    "score": 2.0,
                    "ts": current_time,
                    "type": "concern",
                    "confidence": "Tier S",
                    "reason": "同时具备高价格敏感度和高品质要求"
                })

    def get_final_persona(self) -> Dict[str, Any]:
        """获取最终合并画像，包含过期清理逻辑"""
        current_time = time.time()
        
        # 清理过期指令
        self.instructions = [i for i in self.instructions if not i.get("expiry") or i["expiry"] > current_time]
        
        return {
            "global_traits": self.global_traits,
            "category_intents": self.category_intents,
            "recent_instructions": self.instructions,
            "persona_summary": self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """专家级画像摘要 (针对 Issue #3)"""
        summary_parts = []
        
        # 1. 身份阶段
        if "life_stage" in self.global_traits:
            val = self.global_traits["life_stage"]["values"][0]["value"]
            summary_parts.append(f"一位[{val}]用户")
            
        # 2. 核心矛盾
        if "trait_derivative" in self.global_traits:
            for v in self.global_traits["trait_derivative"]["values"]:
                if v["value"] == "value_seeker":
                    summary_parts.append("典型的[极致性价比追求者]，对品质有硬要求但对价格高度敏感")

        # 3. 风险偏好
        if "after_sales_service" in self.global_traits:
            for v in self.global_traits["after_sales_service"]["values"]:
                if v["value"] == "low_risk_tolerance":
                    summary_parts.append("具有明显的[风险厌恶]特征，极其关注售后保障和品牌信任")

        if not summary_parts: return "用户特征分析中..."
        return "。".join(summary_parts) + "。"

    def debug_print(self):
        """调试打印画像状态"""
        import pprint
        print("\n=== 用户画像深度侧写 (V8 专家版) ===")
        print(">> 全局特质 (Global Traits):")
        pprint.pprint(self.global_traits)
        print("\n>> 品类意图 (Category Intents):")
        pprint.pprint(self.category_intents)
        print("\n>> 最近指令 (Recent Instructions):")
        pprint.pprint(self.instructions)
        print("\n>> 画像摘要:")
        print(self._generate_summary())
        print("========================\n")

if __name__ == "__main__":
    # 模拟运行
    system = UserPersonaSystem()
    
    # 模拟多次更新
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
