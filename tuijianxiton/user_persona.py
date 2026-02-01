import time
import math
import json
from typing import Dict, List, Any, Optional

class UserPersonaSystem:
    """
    用户画像管理系统
    基于 'Long-short term Interest Split' (长短期兴趣分离) 理论构建。
    
    Attributes:
        short_term_profile (Dict): 短期画像 (Intent)，存储在内存/Redis，衰减快。
        long_term_profile (Dict): 长期画像 (Preference)，存储在HBase/DB，衰减慢。
        ALPHA_SHORT (float): 短期衰减系数。
        ALPHA_LONG (float): 长期衰减系数。
    """
    
    def __init__(self):
        # 模拟存储结构
        # 实际生产中，这些应该存 Redis (short) 和 HBase (long)
        self.short_term_profile: Dict[str, Dict[str, Any]] = {} 
        self.long_term_profile: Dict[str, Dict[str, Any]] = {}
        
        # 定义衰减系数 (Lambda)
        # 短期衰减快 (单位: 小时)，例如 0.1 表示每小时衰减约 10% (e^-0.1 ≈ 0.90)
        self.ALPHA_SHORT = 0.1  
        # 长期衰减慢 (单位: 小时)，例如 0.005 表示每小时衰减约 0.5% (e^-0.005 ≈ 0.995)
        self.ALPHA_LONG = 0.005 

    def _calculate_decay(self, last_update_ts: float, lambda_factor: float) -> float:
        """
        计算时间衰减系数 (0~1之间)
        
        Formula: Score(t) = Score_initial * e^(-lambda * delta_t)
        
        Args:
            last_update_ts (float): 上次更新的时间戳。
            lambda_factor (float): 衰减系数。
            
        Returns:
            float: 衰减因子。
        """
        if not last_update_ts:
            return 1.0
        
        current_time = time.time()
        # 避免时间回溯导致的问题
        if current_time < last_update_ts:
            return 1.0
            
        hours_diff = (current_time - last_update_ts) / 3600
        # 指数衰减公式
        decay = math.exp(-lambda_factor * hours_diff)
        return decay

    def _convert_tier_to_score(self, tier: str) -> float:
        """
        将 Tier 等级转换为数值分数。
        """
        tier_map = {
            "Tier S": 1.0,
            "Tier A": 0.8,
            "Tier B": 0.5
        }
        return tier_map.get(tier, 0.5)

    def update_persona(self, llm_tags_output: List[Dict[str, Any]]):
        """
        核心方法：接收 LLM 的标签，更新画像
        
        Args:
            llm_tags_output (List[Dict]): LLM 提取的标签列表，格式参考 extracted_tags.json
            例如: [{"name": "极致性价比", "tier": "Tier A", ...}, ...]
        """
        current_time = time.time()
        
        print(f"--- 接收到新信号 (Tags Count: {len(llm_tags_output)}) ---")

        # ===========================
        # 1. 更新短期画像 (侧重意图捕捉)
        # 策略：激进更新，甚至直接覆盖
        # ===========================
        
        # 1.1 先对现有短期标签做一次衰减
        tags_to_remove = []
        for tag, data in self.short_term_profile.items():
            decay = self._calculate_decay(data['ts'], self.ALPHA_SHORT)
            data['score'] *= decay
            # 如果分数太低，直接清洗掉 (阈值可调)
            if data['score'] < 0.1:
                tags_to_remove.append(tag)
        
        for tag in tags_to_remove:
            del self.short_term_profile[tag]

        # 1.2 插入新标签 (短期画像直接给高权重)
        for item in llm_tags_output:
            tag_name = item.get('name')
            if not tag_name:
                continue
                
            # 获取基础分数
            base_score = item.get('score')
            if base_score is None:
                base_score = self._convert_tier_to_score(item.get('tier', 'Tier B'))
            
            # 短期策略：新来的意图，权重直接拉满，覆盖旧意图
            # 这里乘以 1.5 是为了强调当前的实时意图
            self.short_term_profile[tag_name] = {
                'score': base_score * 1.5, 
                'ts': current_time,
                'tier': item.get('tier', 'Unknown'),
                'source': 'realtime_chat'
            }

        # ===========================
        # 2. 更新长期画像 (侧重累积偏好)
        # 策略：平滑累加，不会剧烈波动
        # ===========================
        
        # 2.1 先衰减
        for tag, data in self.long_term_profile.items():
            decay = self._calculate_decay(data['ts'], self.ALPHA_LONG)
            data['score'] *= decay
        
        # 2.2 累加新标签
        for item in llm_tags_output:
            tag_name = item.get('name')
            if not tag_name:
                continue
            
            base_score = item.get('score')
            if base_score is None:
                base_score = self._convert_tier_to_score(item.get('tier', 'Tier B'))
            
            if tag_name in self.long_term_profile:
                # 长期策略：旧分值 + 新分值 * 权重 (平滑更新)
                # 0.2 的系数意味着单次行为对长期画像影响较小
                new_score = self.long_term_profile[tag_name]['score'] + (base_score * 0.2)
                # 封顶 5.0 分，防止无限膨胀
                self.long_term_profile[tag_name]['score'] = min(new_score, 5.0) 
                self.long_term_profile[tag_name]['ts'] = current_time
            else:
                # 新兴趣进入长期画像时，起步分要低
                self.long_term_profile[tag_name] = {
                    'score': base_score * 0.2, 
                    'ts': current_time,
                    'first_seen': current_time
                }

    def get_final_persona(self) -> Dict[str, Dict[str, Any]]:
        """
        获取当前用于推荐的混合画像
        可以在这里实现 Conflict Resolution (冲突处理)
        """
        return {
            "short_term_intent": self.short_term_profile,
            "long_term_preference": self.long_term_profile
        }

    def debug_print(self):
        """辅助打印当前画像状态"""
        import pprint
        print("\n=== Current Persona State ===")
        print(">> Short Term (Intent):")
        pprint.pprint(self.short_term_profile)
        print("\n>> Long Term (Preference):")
        pprint.pprint(self.long_term_profile)
        print("=============================\n")

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
