from typing import List, Dict
from src.sales_agent_score.models.data_models import MessageGroup
from src.sales_agent_score.utils.score_utils import normalize_score
from rag.llm import chat
from src.prompt.prompt_score import SCORE_PROMPT


class AISemanticScorer:
    """处理AI语义评分（权重50%）"""
    
    def __init__(self):
        # 正向维度加分规则
        self.positive_rules = {
            "需求具体": 10.0,
            "支付暗示": 15.0,
            "紧迫感": 12.0
        }
        
        # 负向维度扣分规则
        self.negative_rules = {
            "价格抗性": -15.0,
            "负面情绪": -12.0,
            "无效沟通": -10.0
        }
    
    def score(self, conversation_history) -> float:
        """计算AI语义评分
        
        Args:
            conversation_history: 会话历史消息组列表
            
        Returns:
            AI语义评分（0-100）
        """
        query = "根据以下会话历史，计算客户意图的语义分（0-100）：\n"
        query += str(conversation_history)

        response = chat(query,[],SCORE_PROMPT)
    
        
        
        # 归一化到0-100范围
        return response
    
    def _check_positive_dimensions(self, content: str) -> float:
        """检查正向维度并计算加分
        
        Args:
            content: 消息内容
            
        Returns:
            正向维度加分数
        """
        score = 0.0
        
        # 检查需求具体
        if self._has_specific_requirement(content):
            score += self.positive_rules["需求具体"]
        
        # 检查支付暗示
        if self._has_payment_hint(content):
            score += self.positive_rules["支付暗示"]
        
        # 检查紧迫感
        if self._has_urgency(content):
            score += self.positive_rules["紧迫感"]
        
        return score
    
    def _check_negative_dimensions(self, content: str) -> float:
        """检查负向维度并计算扣分
        
        Args:
            content: 消息内容
            
        Returns:
            负向维度扣分数
        """
        score = 0.0
        
        # 检查价格抗性
        if self._has_price_resistance(content):
            score += self.negative_rules["价格抗性"]
        
        # 检查负面情绪
        if self._has_negative_emotion(content):
            score += self.negative_rules["负面情绪"]
        
        # 检查无效沟通
        if self._has_invalid_communication(content):
            score += self.negative_rules["无效沟通"]
        
        return score
    
    def _has_specific_requirement(self, content: str) -> bool:
        """判断是否有具体需求
        
        Args:
            content: 消息内容
            
        Returns:
            是否有具体需求
        """
        # 这里实现具体需求检测逻辑
        # 示例：提到具体参数、场景、竞品对比
        keywords = ["参数", "场景", "对比", "竞品", "规格", "型号", "配置"]
        return any(keyword in content for keyword in keywords)
    
    def _has_payment_hint(self, content: str) -> bool:
        """判断是否有支付暗示
        
        Args:
            content: 消息内容
            
        Returns:
            是否有支付暗示
        """
        # 这里实现支付暗示检测逻辑
        # 示例：询问支付方式、货币、是否有税费
        keywords = ["支付", "付款", "货币", "税费", "价格", "费用"]
        return any(keyword in content for keyword in keywords)
    
    def _has_urgency(self, content: str) -> bool:
        """判断是否有紧迫感
        
        Args:
            content: 消息内容
            
        Returns:
            是否有紧迫感
        """
        # 这里实现紧迫感检测逻辑
        # 示例：询问发货时效、是否现货
        keywords = ["时效", "现货", "发货", "快递", "多久", "快吗"]
        return any(keyword in content for keyword in keywords)
    
    def _has_price_resistance(self, content: str) -> bool:
        """判断是否有价格抗性
        
        Args:
            content: 消息内容
            
        Returns:
            是否有价格抗性
        """
        # 这里实现价格抗性检测逻辑
        # 示例：明确表示"太贵了"、"买不起"
        keywords = ["太贵", "买不起", "价格高", "性价比", "优惠", "折扣"]
        return any(keyword in content for keyword in keywords)
    
    def _has_negative_emotion(self, content: str) -> bool:
        """判断是否有负面情绪
        
        Args:
            content: 消息内容
            
        Returns:
            是否有负面情绪
        """
        # 这里实现负面情绪检测逻辑
        # 示例：表达不满、愤怒、嘲讽、不信任
        keywords = ["不满", "愤怒", "嘲讽", "不信任", "骗子", "垃圾", "不好"]
        return any(keyword in content for keyword in keywords)
    
    def _has_invalid_communication(self, content: str) -> bool:
        """判断是否为无效沟通
        
        Args:
            content: 消息内容
            
        Returns:
            是否为无效沟通
        """
        # 这里实现无效沟通检测逻辑
        # 示例：持续发送无关内容、骚扰信息
        # 简单实现：检测是否包含无关关键词或内容过短
        irrelevant_keywords = ["你好", "在吗", "测试", "123", "哈哈哈"]
        return any(keyword in content for keyword in irrelevant_keywords) or len(content) < 2

if __name__ == "__main__":
    scorer = AISemanticScorer()
    # 生成的问答对数据示例

    # 场景1：电子产品（手机）
    history1 = [
            "user: 这款手机内存多大？", 
            "agent: 12GB运行内存，256GB存储空间。", 
            "user: 支持5G吗？", 
            "agent: 是的，支持全网通5G。", 
            "user: 那我要黑色的，帮我下单。"
        ]

    # 场景2：家电（冰箱）
    history2 = [
            "user: 这个冰箱容量是多少？", 
            "agent: 450升，多门设计。", 
            "user: 耗电吗？", 
            "agent: 一级能效，每天不到一度电。", 
            "user: 那我考虑一下，明天回复你。"
        ]

    # 场景3：服装（外套）
    history3 = [
            "user: 这件外套是什么材质的？", 
            "agent: 90%鸭绒，保暖性很好。", 
            "user: 有XL码吗？", 
            "agent: 有的，颜色有黑色、灰色和藏青色。", 
            "user: 我要黑色XL的，麻烦发货快点。"
        ]

    # 场景4：数码配件（耳机）
    history4 = [
            "user: 这款耳机续航多久？", 
            "agent: 单次使用8小时，充电盒可充3次。", 
            "user: 支持降噪吗？", 
            "agent: 支持主动降噪，效果很棒。", 
            "user: 好的，我买一个。"
        ]

    # 场景5：家居用品（沙发）
    history5 = [
            "user: 这个沙发尺寸是多少？", 
            "agent: 三人位，长2.1米，宽0.9米。", 
            "user: 可以拆洗吗？", 
            "agent: 沙发套可以拆洗，很方便。", 
            "user: 那我要灰色的，什么时候能送货？"
        ]

    # 场景6：美妆产品（护肤品）
    history6 = [
            "user: 这款面霜适合什么肤质？", 
            "agent: 适合所有肤质，尤其是干性肌肤。", 
            "user: 有没有过敏风险？", 
            "agent: 经过敏感肌测试，低刺激配方。", 
            "user: 那我买两瓶，有优惠吗？"
        ]

    # 场景7：运动器材（跑步机）
    history7 = [
            "user: 这个跑步机噪音大吗？", 
            "agent: 采用静音电机，噪音低于60分贝。", 
            "user: 可以折叠吗？", 
            "agent: 是的，支持一键折叠，节省空间。", 
            "user: 那我要一台，帮我安排安装。"
        ]

    # 场景8：食品（咖啡）
    history8 = [
            "user: 这个咖啡豆是中度烘焙吗？", 
            "agent: 是的，中度烘焙，口感醇厚。", 
            "user: 保质期多久？", 
            "agent: 12个月，密封保存。", 
            "user: 那我买两袋，送朋友一袋。"
        ]

    # 场景9：图书（教材）
    history9 = [
            "user: 这本书是最新版吗？", 
            "agent: 是的，2025年最新修订版。", 
            "user: 包含习题答案吗？", 
            "agent: 是的，附录有详细答案和解析。", 
            "user: 那我要一本，包邮吗？"
        ]

    # 场景10：汽车用品（坐垫）
    history10 = [
            "user: 这个坐垫适合冬季使用吗？", 
            "agent: 是的，毛绒材质，保暖舒适。", 
            "user: 安装复杂吗？", 
            "agent: 简单安装，配有说明书和工具。", 
            "user: 那我要一套，颜色和我的车内饰匹配吗？"
        ]

    # 测试
    for history in [history1, history2, history3, history4, history5, history6, history7, history8, history9, history10]:
        score = scorer.score(history)
        print(f"Semantic Score: {score}")


