from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp

class TFIDFIntentClassifier:
    def __init__(self, min_df=2, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            tokenizer=lambda x: x.split(),  # 中文需替换为jieba分词
            lowercase=False
        )
        self.intent_vectors = None  # 将存储为稀疏矩阵
        self.intents = []
    
    def train(self, training_data):
        """
        training_data格式: [("文本1", "意图1"), ("文本2", "意图2"), ...]
        """
        texts = [text for text, _ in training_data]
        self.intents = list({intent for _, intent in training_data})
        
        # 生成TF-IDF矩阵 (返回scipy稀疏矩阵)
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # 为每个意图创建原型向量
        intent_vectors_list = []
        for intent in self.intents:
            # 获取属于当前意图的样本索引
            indices = [i for i, (_, label) in enumerate(training_data) if label == intent]
            # 计算意图原型向量（样本均值）
            if indices:
                # 确保使用稀疏矩阵操作
                intent_vector = sp.csr_matrix(tfidf_matrix[indices].mean(axis=0))
                intent_vectors_list.append(intent_vector)
        
        # 使用scipy.sparse.vstack处理稀疏矩阵
        self.intent_vectors = sp.vstack(intent_vectors_list)
    
    def predict(self, text, threshold=0.3):
        """
        预测意图并返回置信度
        """
        # 转换输入文本 (保持稀疏格式)
        input_vec = self.vectorizer.transform([text])
        
        # 计算与各意图的相似度 (自动处理稀疏矩阵)
        similarities = cosine_similarity(input_vec, self.intent_vectors)[0]
        
        # 获取最高相似度的意图
        max_idx = np.argmax(similarities)
        confidence = float(similarities[max_idx])  # 转换为标准Python float
        
        # 应用阈值过滤
        if confidence < threshold:
            return "nlu_fallback", confidence
        
        return self.intents[max_idx], confidence
    
    def get_top_intents(self, text, top_k=3, threshold=0.1):
        """获取前K个可能意图"""
        input_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(input_vec, self.intent_vectors)[0]
        
        # 获取排序后的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 生成(意图, 置信度)列表
        results = []
        for idx in top_indices:
            conf = float(similarities[idx])  # 转换为标准float
            if conf < threshold:
                break
            results.append((self.intents[idx], conf))
        
        return results

# 使用示例
if __name__ == "__main__":
    # 训练数据
    training_data = [
        ("如何重置密码", "reset_password"),
        ("忘记密码怎么办", "reset_password"),
        ("怎么修改密码", "reset_password"),
        ("查询账户余额", "check_balance"),
        ("我的账户还有多少钱", "check_balance"),
        ("查看余额", "check_balance"),
        ("转账给张三", "transfer_money"),
        ("给李四汇款500元", "transfer_money"),
        ("如何联系客服", "contact_support"),
        ("客服电话是多少", "contact_support")
    ]
    
    # 初始化并训练分类器
    classifier = TFIDFIntentClassifier(min_df=1)
    classifier.train(training_data)
    
    # 测试样本
    test_queries = [
        "密码忘记了怎么重置",
        "我想知道账户余额",
        "给王五转账200元",
        "这根本不是你们业务范围内的问题"
    ]
    
    # 预测结果
    for query in test_queries:
        intent, confidence = classifier.predict(query)
        top_intents = classifier.get_top_intents(query, top_k=2)
        print(f"查询: '{query}'")
        print(f"→ 主要意图: '{intent}' (置信度: {confidence:.3f})")
        print(f"→ 候选意图: {top_intents}")
        print("-" * 50)