from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp

class TFIDFIntentClassifier:
    def __init__(self, min_df=1, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            tokenizer=lambda x: x.split(),
            lowercase=False,
            dtype=np.float32  # 显式指定数据类型
        )
        self.intent_vectors = None
        self.intents = []
        self.intent_samples = {}  # 存储每个意图的样本
    
    def train(self, training_data):
        """训练TF-IDF意图分类器"""
        # 按意图分组样本
        self.intent_samples = {}
        for text, intent in training_data:
            if intent not in self.intent_samples:
                self.intent_samples[intent] = []
            self.intent_samples[intent].append(text)
        
        self.intents = list(self.intent_samples.keys())
        
        # 收集所有文本用于训练vectorizer
        all_texts = [text for text, _ in training_data]
        
        # 训练TF-IDF向量化器
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # 为每个意图创建原型向量
        intent_vectors_list = []
        for intent in self.intents:
            # 获取该意图的所有样本
            samples = self.intent_samples[intent]
            
            # 为这些样本创建TF-IDF向量
            sample_vectors = self.vectorizer.transform(samples)
            
            # 计算平均向量（正确的稀疏矩阵平均方式）
            intent_vector = sp.csr_matrix(sample_vectors.mean(axis=0))
            intent_vectors_list.append(intent_vector)
        
        # 合并所有意图向量
        self.intent_vectors = sp.vstack(intent_vectors_list)
        
        # 调试信息
        print(f"训练完成! 意图数量: {len(self.intents)}")
        print(f"特征维度: {self.intent_vectors.shape[1]}")
        print(f"意图列表: {self.intents}")
        
        # 验证向量是否有效
        print("\n意图向量验证:")
        for i, intent in enumerate(self.intents):
            vec_norm = np.linalg.norm(self.intent_vectors[i].toarray())
            sample_count = len(self.intent_samples[intent])
            print(f"- {intent}: 样本数={sample_count}, 向量范数={vec_norm:.4f}")
    
    def predict(self, text, threshold=0.3):
        """预测输入文本的意图"""
        # 转换输入文本
        input_vec = self.vectorizer.transform([text])
        
        # 调试：检查输入向量
        input_norm = np.linalg.norm(input_vec.toarray())
        if input_norm < 1e-8:
            print(f"警告: 输入向量范数接近零 ({input_norm:.6f}), 文本可能未包含训练词汇")
        
        # 计算与各意图的相似度
        similarities = cosine_similarity(input_vec, self.intent_vectors)[0]
        
        # 调试：打印原始相似度
        # print(f"原始相似度: {dict(zip(self.intents, similarities))}")
        
        # 获取最高相似度的意图
        max_idx = np.argmax(similarities)
        confidence = float(similarities[max_idx])
        
        # 应用阈值过滤
        if confidence < threshold:
            return "nlu_fallback", confidence
        
        return self.intents[max_idx], confidence
    
    def get_top_intents(self, text, top_k=3, threshold=0.1):
        """获取前K个可能意图"""
        input_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(input_vec, self.intent_vectors)[0]
        
        # 获取排序后的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 生成(意图, 置信度)列表
        results = []
        for idx in sorted_indices:
            conf = float(similarities[idx])
            if conf < threshold or len(results) >= top_k:
                break
            results.append((self.intents[idx], conf))
        
        return results
    
    def debug_input(self, text):
        """调试方法：分析输入文本的特征"""
        input_vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 获取非零特征
        non_zero_indices = input_vec.nonzero()[1]
        non_zero_features = [(feature_names[i], input_vec[0, i]) for i in non_zero_indices]
        
        print(f"\n输入文本: '{text}'")
        print(f"非零特征数量: {len(non_zero_features)}")
        print("主要特征(TF-IDF值最高):")
        for feature, value in sorted(non_zero_features, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {feature}: {value:.4f}")

# 测试函数
def test_classifier():
    # 训练数据（增加更多样例提高效果）
    training_data = [
        # 密码重置意图
        ("如何重置密码", "reset_password"),
        ("忘记密码怎么办", "reset_password"),
        ("怎么修改密码", "reset_password"),
        ("密码重置步骤", "reset_password"),
        ("密码忘记了", "reset_password"),
        
        # 余额查询意图
        ("查询账户余额", "check_balance"),
        ("我的账户还有多少钱", "check_balance"),
        ("查看余额", "check_balance"),
        ("账户余额是多少", "check_balance"),
        ("我有多少钱", "check_balance"),
        
        # 转账意图
        ("转账给张三", "transfer_money"),
        ("给李四汇款500元", "transfer_money"),
        ("转100元给王五", "transfer_money"),
        ("我要转账", "transfer_money"),
        ("转账操作", "transfer_money"),
        
        # 客服意图
        ("如何联系客服", "contact_support"),
        ("客服电话是多少", "contact_support"),
        ("人工客服", "contact_support"),
        ("找客服帮忙", "contact_support"),
        ("客服联系方式", "contact_support")
    ]
    
    # 初始化并训练分类器
    classifier = ChineseTFIDFIntentClassifier(min_df=1, max_features=1000)
    classifier.train(training_data)
    
    # 测试样本
    test_queries = [
        "找客服帮忙",
        "我想知道账户余额",
        "给王五转账200元",
        "这根本不是你们业务范围内的问题"
    ]
    
    # 预测结果
    print("\n" + "="*60)
    print("预测结果:")
    print("="*60)
    
    for query in test_queries:
        classifier.debug_input(query)  # 调试输入特征
        intent, confidence = classifier.predict(query, threshold=0.2)  # 降低阈值
        top_intents = classifier.get_top_intents(query, top_k=3, threshold=0.1)
        
        print(f"\n查询: '{query}'")
        print(f"→ 主要意图: '{intent}' (置信度: {confidence:.3f})")
        print(f"→ 候选意图: {top_intents}")
        print("-" * 50)

import jieba

class ChineseTFIDFIntentClassifier(TFIDFIntentClassifier):
    def __init__(self, min_df=1, max_features=5000):
        super().__init__(min_df, max_features)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            min_df=min_df,
            max_features=max_features,
            tokenizer=jieba.lcut,
            lowercase=True,
            dtype=np.float32
        )

if __name__ == "__main__":
    test_classifier()