import time
import math
import json
import datetime
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统
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
        self.ALPHA_INTENT = 0.015   #24h 剩余约 70%
        
        # 权重映射
        self.TIER_MAP = {"Tier S": 1.0, "Tier A": 0.7, "Tier B": 0.2} 
        self.TYPE_WEIGHT = {
            "requirement": 1.5, 
            "concern": 1.0, 
            "inquiry": 0.5,
            "attribute": 1.2  #长期稳定的用户属性 (如性别、年龄)
        }
        
        # 语义归一化映射表
        self.TAG_NORMALIZATION = {
            "保守主义": "low_risk_tolerance",
            "cautious": "low_risk_tolerance",
            "风险厌恶": "low_risk_tolerance",
            "砍价高手": "bargaining_expert",
            "极致性价比": "price_sensitive_high"
        }

    def load_from_profile(self, profile: Dict[str, Any]):
        """从现有画像加载数据"""
        self.global_traits = profile.get('global_traits', {})
        self.category_intents = profile.get('category_intents', {})
        self.instructions = profile.get('recent_instructions', [])

    def _get_current_time_str(self) -> str:
        """获取当前时间字符串 (YYYY-MM-DD HH:MM:SS)"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def _parse_time_str(self, time_str: str) -> float:
        """解析时间字符串为时间戳"""
        try:
            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return dt.timestamp()
        except (ValueError, TypeError):
            # 如果解析失败（兼容旧数据是float的情况），尝试直接转换
            try:
                return float(time_str)
            except (ValueError, TypeError):
                return 0.0

    def _calculate_decay(self, last_update_ts: Any, lambda_factor: float) -> float:
        """计算指数衰减因子"""
        if not last_update_ts: return 1.0
        
        # 转换时间格式
        if isinstance(last_update_ts, str):
            ts = self._parse_time_str(last_update_ts)
        else:
            ts = float(last_update_ts)
            
        hours_diff = (time.time() - ts) / 3600
        return math.exp(-lambda_factor * max(0, hours_diff))

    def _get_base_score(self, tier: Optional[str], tag_type: str) -> float:
        """计算对齐后的基础分值"""
        # 补全缺失的置信度
        if not tier:
            tier = "Tier A" if tag_type == "requirement" else "Tier B"
            
        tier_score = self.TIER_MAP.get(tier, 0.4)
        type_weight = self.TYPE_WEIGHT.get(tag_type, 1.0)
        return round(tier_score * type_weight * 2.0, 2)

    def _normalize_value(self, val: str) -> str:
        """标签归一化"""
        return self.TAG_NORMALIZATION.get(val, val)

    def _suppress_conflicting_values(self, tag_entry: Dict[str, Any], current_val: str):
        """
        抑制同维度下的冲突值 (Negative Suppression)
        当用户更新某个维度的值时（如颜色选了黑色），对该维度下其他值（如红色）进行降权。
        """
        for entry in tag_entry["values"]:
            if entry["value"] != current_val:
                entry["score"] = round(entry["score"] * 0.5, 2) # 惩罚因子

    def _resolve_conflicts_and_clean_expired(self):
        """
        解决标签冲突并清理过期数据
        策略：
        1. 清理过期数据
        2. 对于同一个 Tag Code，如果存在多个 Value，保留时间最新的一个（假设属性更新逻辑）
        3. 清理低分噪音数据 (score < 0.1)
        """
        current_ts = time.time()

        # 清理 Global Traits
        for tag_code in list(self.global_traits.keys()):
            values = self.global_traits[tag_code].get("values", [])
            valid_values = []
            
            # 1. 过滤过期和噪音
            for v in values:
                # 噪音清理
                if v.get("score", 0) < 0.1:
                    continue

                expiry = v.get("expiry")
                if expiry:
                    expiry_ts = self._parse_time_str(expiry)
                    if expiry_ts <= current_ts:
                        continue
                valid_values.append(v)
            
            # 2. 解决冲突 (只保留最新的)
            if valid_values:
                # 按时间戳倒序排序
                valid_values.sort(key=lambda x: self._parse_time_str(x.get("ts", "")), reverse=True)
                # 只保留最新的一个
                self.global_traits[tag_code]["values"] = [valid_values[0]]
            else:
                # 如果没有有效值，移除该 Tag
                del self.global_traits[tag_code]

        # 清理 Category Intents
        for category in list(self.category_intents.keys()):
            cat_data = self.category_intents[category]
            for tag_code in list(cat_data.keys()):
                values = cat_data[tag_code].get("values", [])
                valid_values = []
                
                # 1. 过滤过期和噪音
                for v in values:
                    # 噪音清理
                    if v.get("score", 0) < 0.1:
                        continue

                    expiry = v.get("expiry")
                    if expiry:
                        expiry_ts = self._parse_time_str(expiry)
                        if expiry_ts <= current_ts:
                            continue
                    valid_values.append(v)
                
                # 2. 解决冲突 (只保留最新的)
                if valid_values:
                    # 按时间戳倒序排序
                    valid_values.sort(key=lambda x: self._parse_time_str(x.get("ts", "")), reverse=True)
                    # 只保留最新的一个
                    cat_data[tag_code]["values"] = [valid_values[0]]
                else:
                    del cat_data[tag_code]
            
            # 如果该分类下没有 Tag 了，移除该分类
            if not cat_data:
                del self.category_intents[category]

    def _deduplicate_instructions(self):
        """对指令进行去重，保留最新的"""
        unique_map = {}
        for inst in self.instructions:
            # 使用 (type, action, value) 作为唯一键
            key = (inst.get("type"), inst.get("action"), inst.get("value"))
            unique_map[key] = inst
        
        self.instructions = list(unique_map.values())

    def update_persona(self, llm_output: Dict[str, Any]):
        """
        更新画像，包含归一化、冲突检测和分值对齐逻辑。
        """
        current_time_str = self._get_current_time_str()
        current_ts = time.time()
        
        # 1. 更新全局特质
        profile_update = llm_output.get("user_profile_update", {})
        new_global_traits = profile_update.get("global_traits", [])
        
        for item in new_global_traits:
            code = item.get("tag_code")
            val = self._normalize_value(item.get("value")) # 归一化
            
            # 修复 Key-Value 混合问题 (e.g., "性别 - 男" -> "性别")
            if code and val and f" - {val}" in code:
                code = code.replace(f" - {val}", "").strip()
            
            tag_type = item.get("type", "concern")
            conf = item.get("confidence")
            source_quote = item.get("source_quote")
            
            base_score = self._get_base_score(conf, tag_type)
            
            if code not in self.global_traits:
                self.global_traits[code] = {"values": []}
            
            trait_entry = self.global_traits[code]
            found = False
            for entry in trait_entry["values"]:
                if entry["value"] == val:
                    decay = self._calculate_decay(entry["ts"], self.ALPHA_GLOBAL)
                    
                    # Noise Control: < 5 mins check
                    time_diff = current_ts - self._parse_time_str(entry["ts"])
                    if time_diff < 300:
                         # 短时间内重复输入，只做微小更新或取最大值，防止刷分
                         entry["score"] = max(entry["score"] * decay, base_score)
                    else:
                        entry["score"] = (entry["score"] * decay) + (base_score * 0.3)
                    
                    entry["score"] = min(round(entry["score"], 2), 5.0)
                    entry["ts"] = current_time_str
                    # Update metadata to latest
                    entry["confidence"] = conf or entry["confidence"]
                    entry["type"] = tag_type
                    if source_quote:
                        entry["source_quote"] = source_quote
                    
                    found = True
                    break
            
            if not found:
                trait_entry["values"].append({
                    "value": val,
                    "score": base_score, # 冷启动
                    "ts": current_time_str,
                    "type": tag_type,
                    "confidence": conf or "Tier B",
                    "expiry": None, # 全局特质永不过期
                    "source_quote": source_quote
                })
            
            # Apply Conflict Resolution
            self._suppress_conflicting_values(trait_entry, val)

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
                
                # 修复 Key-Value 混合问题
                if code and val and f" - {val}" in code:
                    code = code.replace(f" - {val}", "").strip()
                
                tag_type = tag.get("type", "concern")
                conf = tag.get("confidence")
                source_quote = tag.get("source_quote")
                
                base_score = self._get_base_score(conf, tag_type)
                
                if code not in cat_data:
                    cat_data[code] = {"values": []}
                
                intent_entry = cat_data[code]
                found = False
                for entry in intent_entry["values"]:
                    if entry["value"] == val:
                        decay = self._calculate_decay(entry["ts"], self.ALPHA_INTENT)
                        
                        # Noise Control: < 5 mins check
                        time_diff = current_ts - self._parse_time_str(entry["ts"])
                        if time_diff < 300:
                             entry["score"] = max(entry["score"] * decay, base_score)
                        else:
                            entry["score"] = (entry["score"] * decay) + (base_score * 0.5)
                            
                        entry["score"] = min(round(entry["score"], 2), 5.0)
                        entry["ts"] = current_time_str
                        # Update metadata to latest
                        entry["confidence"] = conf or entry["confidence"]
                        entry["type"] = tag_type
                        if source_quote:
                            entry["source_quote"] = source_quote
                        
                        found = True
                        break
                
                if not found:
                    # 计算过期时间 (30天)
                    expiry_ts = current_ts + (86400 * 30)
                    expiry_str = datetime.datetime.fromtimestamp(expiry_ts).strftime("%Y-%m-%d %H:%M:%S")
                    
                    intent_entry["values"].append({
                        "value": val,
                        "score": base_score, # 冷启动
                        "ts": current_time_str,
                        "type": tag_type,
                        "expiry": expiry_str, # 30天过期
                        "confidence": conf or "Tier B",
                        "source_quote": source_quote
                    })
                
                # Apply Conflict Resolution
                self._suppress_conflicting_values(intent_entry, val)

        # 3. 更新即时指令
        new_instructions = llm_output.get("instructions", [])
        for inst in new_instructions:
            # 计算过期时间 (1小时)
            expiry_ts = current_ts + 3600
            expiry_str = datetime.datetime.fromtimestamp(expiry_ts).strftime("%Y-%m-%d %H:%M:%S")
            
            inst["expiry"] = expiry_str # 1小时过期 (会话级别)
            self.instructions.append(inst)
        
        # Deduplicate instructions
        self._deduplicate_instructions()
        self.instructions = self.instructions[-20:]

        # 4. 冲突检测与衍生标签
        self._detect_conflicts(current_time_str)

    def _detect_conflicts(self, current_time_str: str):
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
                    v["score"] = min(round(v["score"] + 0.5, 2), 5.0)
                    v["ts"] = current_time_str
                    found = True
                    break
            
            if not found:
                self.global_traits["trait_derivative"]["values"].append({
                    "value": "value_seeker",
                    "score": 2.0,
                    "ts": current_time_str,
                    "type": "concern",
                    "confidence": "Tier S",
                    "reason": "同时具备高价格敏感度和高品质要求"
                })

    def get_final_persona(self) -> Dict[str, Any]:
        """获取最终合并画像，包含过期清理逻辑"""
        current_ts = time.time()
        
        # 1. 解决冲突并清理过期数据 (Global Traits & Category Intents)
        self._resolve_conflicts_and_clean_expired()
        
        # 2. 清理过期指令 (Recent Instructions)
        valid_instructions = []
        for i in self.instructions:
            expiry = i.get("expiry")
            if not expiry:
                valid_instructions.append(i)
                continue
                
            # 解析过期时间
            expiry_ts = self._parse_time_str(expiry)
            if expiry_ts > current_ts:
                valid_instructions.append(i)
                
        self.instructions = valid_instructions
        
        return {
            "global_traits": self.global_traits,
            "category_intents": self.category_intents,
            "recent_instructions": self.instructions
        }

    def debug_print(self):
        """调试打印画像状态"""
        import pprint
        print("\n=== 用户画像深度侧写 ===")
        print(">> 全局特质 (Global Traits):")
        pprint.pprint(self.global_traits)
        print("\n>> 品类意图 (Category Intents):")
        pprint.pprint(self.category_intents)
        print("\n>> 最近指令 (Recent Instructions):")
        pprint.pprint(self.instructions)
 
        print("========================\n")
