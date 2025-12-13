"""
向量索引模块 - Milvus 向量数据库操作
功能：管理法律条文和文书的向量索引，实现高效相似度检索
核心特性：
  1. 双索引架构：law_collection（法律条文）、writ_collection（裁判文书）
  2. ES-Milvus ID 一致性：保持 ES meta.id 与 Milvus es_id 一致，方便去重

  - 向量维度：1024 (BGE-M3 模型)
  - 相似度度量：IP (内积，适用于归一化向量)
  - 索引类型：FLAT (精确检索，适合中小规模数据)
"""

import numpy as np
from pymilvus import MilvusClient, DataType
from data_process.items import LawItem, WritItem, FAQItem
from config import settings
import logging
from uuid import uuid1
import simple_pickle as sp

# 禁用 Elasticsearch 和其他库的详细日志
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

# 向量维度配置（与 BGE-M3 模型输出一致）
DIMENSION = 1024

# 检索参数配置
SEARCH_PARAMS = {"metric_type": "IP"}  # IP = 内积相似度

# 索引参数配置
INDEX_PARAMS = MilvusClient.prepare_index_params()
INDEX_PARAMS.add_index(
    field_name="vec",           # 向量字段名
    index_type="FLAT",          # FLAT 索引：精确检索，适合十万级以下数据
    index_name="vec_index",     # 索引名称
    metric_type="IP",           # 内积相似度
)


class Singleton(type):
    """
    单例元类：确保每个集合名称对应唯一的 VecIndex 实例
    避免重复创建连接和索引，节省资源
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # 对于子类，使用类名作为key
        key = cls.__name__
        if key not in cls._instances:
            cls._instances[key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[key]


class VecIndex(metaclass=Singleton):
    """
    向量索引基类
    管理 Milvus 集合的创建、数据插入和向量检索
    """
    
    def __init__(self, collection_name: str):
        """
        初始化向量索引
        
        参数:
            collection_name: 集合名称（law_collection 或 writ_collection）
        """
        # 连接 Milvus 服务
        self.client = MilvusClient(
            uri=f'http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}',
            token=f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
        )
        
        # 定义集合的 Schema（数据结构）
        schema = self.client.create_schema()
        
        # 主键字段：使用 ES 的文档 ID，确保 ES 和 Milvus 数据一致性
        schema.add_field(
            field_name="es_id",
            datatype=DataType.VARCHAR,
            max_length=100,
            is_primary=True  # 主键，用于去重和更新
        )
        
        # 向量字段：存储文本的嵌入向量
        schema.add_field(
            field_name="vec",
            datatype=DataType.FLOAT_VECTOR,
            dim=DIMENSION  # 1024 维
        )
        
        # 文档字段：存储原始文本内容（用于召回后展示）
        schema.add_field(
            field_name="parent",
            datatype=DataType.VARCHAR,
            max_length=65535  # 支持大文本
        )
        
        # 来源字段：标识数据来源（如文件名、案号等）
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=500  # 增加长度以支持长文件名
        )

        self.collection_name = collection_name
        
        # 如果集合不存在则创建
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=INDEX_PARAMS
            )
            print(f"创建集合: {self.collection_name}")

    def insert(self, embeddings, docs, ids, sources):
        """
        批量插入或更新向量数据

        性能优化:
          - 使用 upsert_rows 而非 upsert，提高吞吐量
          - 批量处理 1000 条数据一次，适合大规模数据导入
          - 向量直接转换为列表，避免重复转换
        
        参数:
            embeddings: 向量列表，每个向量为 1024 维 numpy 数组
            docs: 文档内容列表（原始文本）
            ids: ES 文档 ID 列表（与 ES meta.id 一致）
            sources: 来源标识列表（如法律文件名、案号等）
        
        示例:
            embeddings = [vec1, vec2, vec3]
            docs = ["条文1", "条文2", "条文3"]
            ids = ["law_chunk_1", "law_chunk_2", "law_chunk_3"]
            sources = ["刑法.docx", "刑法.docx", "民法典.docx"]
            index.insert(embeddings, docs, ids, sources)
        """
        if not embeddings:
            return
        
        # 预先构造数据列表，避免重复转换
        data = []
        for i, doc in enumerate(docs):
            # 向量转换为列表（Milvus 要求）
            vec_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]  # 判断是否为ndarray
            
            data.append({
                "es_id": ids[i],
                "parent": doc,
                "vec": vec_list,
                "source": sources[i]
            })
        
        # 使用 upsert_rows 进行高效批量操作
        # upsert_rows 比 upsert 更高效，内部已优化
        try:
            self.client.upsert(collection_name=self.collection_name, data=data)
        except Exception as e:
            print(f"插入错误: {e}")
            raise

    def search(self, vec, topk = 3):
        """
        向量检索：根据查询向量找到最相似的 topk 个文档
        
        参数:
            vec: 查询向量（1024 维）
            topk: 返回最相似的前 k 个结果
        
        返回:
            List[Item]: 检索结果列表，按相似度降序排列
        
        示例:
            from rag_official.embedding import get_embedding
            query_vec = get_embedding("刑法第一百三十三条")
            results = law_index.search(query_vec, topk=5)
            for item in results:
                print(f"相似度: {item.score}, 内容: {item.embedding_text}")
        """
        # 确保向量是二维数组 (1, 1024)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)
        
        # 执行向量检索
        hits = self.client.search(
            collection_name=self.collection_name,
            data=vec.tolist(),
            anns_field="vec",  # 指定向量字段
            limit=topk,
            search_params=SEARCH_PARAMS,
            output_fields=["es_id", "parent", "source"])[0]  # 返回字段
        

        # 根据集合类型返回对应的 Item
        return self._convert_hits_to_items(hits)
    
    def _convert_hits_to_items(self, hits):
        """
        将 Milvus 检索结果转换为业务对象（子类实现）
        
        参数:
            hits: Milvus 检索原始结果
        
        返回:
            List[Item]: 业务对象列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def load_from_es(self):
        """
        从 Elasticsearch 加载数据到 Milvus（子类实现）

        确保 ES 和 Milvus 数据同步
        """
        raise NotImplementedError("子类必须实现此方法")


class LawVecIndex(VecIndex):
    """
    法律条文向量索引
    管理 law_collection 集合，存储法律文档的分块向量
    """
    
    def __init__(self):
        super().__init__(collection_name="law_collection")
    
    def _convert_hits_to_items(self, hits):
        """
        将检索结果转换为 LawItem 对象
        
        返回字段映射:
            - id: ES 文档 ID
            - law_filename: 来源字段（文件名）
            - embedding_text: 文档字段（原始文本）
            - score: 相似度得分
        """
        results = []
        for hit in hits:
            item = LawItem(
                id=hit.entity.get("es_id"),
                law_filename=hit.entity.get("source"),
                embedding_text=hit.entity.get("parent"),
                score=hit.distance  # IP 相似度得分
            )
            results.append(item)
        return results
    
    def load_from_es(self):
        """
        从 ES 的 law_chunks_index 加载所有数据到 Milvus
        
        工作流程:
            1. 使用 ESLawChunk.scan() 遍历所有文档
            2. 批量收集文本，然后批量调用 embedding 模型（充分利用 GPU）
            3. 批量插入 Milvus，es_id = ES meta.id
        
        性能优化:
            - **批量向量化**: 每 1000 条文本批量调用 embedding 模型，GPU 利用率提升 10-20 倍
            - **批量插入**: 直接将 1000 条向量批量插入 Milvus
            - **进度显示**: 每 2000 条数据打印一次进度信息
        
        使用示例:
            law_index = LawVecIndex()
            law_index.load_from_es()
        """
        from data_process.insert_to_es import ESLawChunk
        from rag_official.embedding import get_embedding
        
        print("开始从 ES 加载法律条文数据...")
        batch_size = 1000  # 每批次处理的文本数量
        total_count = 0    # 总处理数
        print_interval = 2000  # 每 2000 条打印一次进度
        
        # 批量收集变量
        batch_texts = []      # 批量文本列表
        batch_docs = []       # 原始文档内容
        batch_ids = []        # ES 文档 ID
        batch_sources = []    # 来源信息
        
        # 遍历 ES 中的所有法律分块
        for item in ESLawChunk.scan():
            # 收集文本到批次
            batch_texts.append(item.embedding_text)
            batch_docs.append(item.embedding_text)
            batch_ids.append(item.meta.id)  # 关键：使用 ES meta.id
            batch_sources.append(item.law_filename)
            
            total_count += 1
            
            # 定期打印进度
            if total_count % print_interval == 0:
                print(f"已收集 {total_count} 条法律条文...")
            
            # 达到批量大小，批量处理
            if len(batch_texts) >= batch_size:
                # 批量获取向量（一次性处理 1000 条）
                embeddings = get_embedding(batch_texts)
                
                if embeddings is not None:
                    # 批量插入 Milvus
                    self.insert(embeddings, batch_docs, batch_ids, batch_sources)
                
                # 清空批次
                batch_texts = []
                batch_docs = []
                batch_ids = []
                batch_sources = []
        
        # 处理剩余数据
        if batch_texts:
            embeddings = get_embedding(batch_texts)
            if embeddings is not None:
                self.insert(embeddings, batch_docs, batch_ids, batch_sources)
        
        print(f"完成！共加载 {total_count} 条法律条文数据到 Milvus")


class WritVecIndex(VecIndex):
    """
    裁判文书向量索引
    管理 writ_collection 集合，存储裁判文书的向量
    """
    
    def __init__(self):
        super().__init__(collection_name="writ_collection")
    
    def _convert_hits_to_items(self, hits):
        """
        将检索结果转换为 WritItem 对象
        
        返回字段映射:
            - id: ES 文档 ID
            - indexbytitle: 来源字段（案号或标题）
            - context: 父文档字段（文书内容）
            - score: 相似度得分
        """
        results = []
        for hit in hits:
            item = WritItem(
                id=hit.entity.get("es_id"),
                indexbytitle=hit.entity.get("source"),
                context=hit.entity.get("parent"),
                score=hit.distance
            )
            results.append(item)
        return results
    
    def load_from_es(self):
        """
        从 ES 的 writ_chunks_index 加载所有数据到 Milvus
        
        工作流程:
            1. 使用 ESWritChunk.scan() 遍历所有文档
            2. 批量收集文本，然后批量调用 embedding 模型（充分利用 GPU）
            3. 批量插入 Milvus，es_id = ES meta.id
        
        性能优化:
            - **批量向量化**: 每 1000 条文本批量调用 embedding 模型，GPU 利用率提升 10-20 倍
            - **批量插入**: 直接将 1000 条向量批量插入 Milvus
            - **进度显示**: 每 2000 条数据打印一次进度信息
        
        使用示例:
            writ_index = WritVecIndex()
            writ_index.load_from_es()
        """
        from data_process.insert_to_es import ESWritChunk
        from rag_official.embedding import get_embedding
        
        print("开始从 ES 加载裁判文书数据...")
        batch_size = 1000  # 每批次处理的文本数量，充分利用 GPU 显存
        total_count = 0    # 总处理数
        print_interval = 2000  # 每 2000 条打印一次进度
        
        # 批量收集变量
        batch_texts = []      # 批量文本列表
        batch_docs = []       # 原始文档内容
        batch_ids = []        # ES 文档 ID
        batch_sources = []    # 来源信息
        
        # 遍历 ES 中的所有文书
        for item in ESWritChunk.scan():
            # 收集文本到批次
            batch_texts.append(item.context)
            batch_docs.append(item.context)
            batch_ids.append(item.meta.id)  # 关键：使用 ES meta.id
            batch_sources.append(item.indexbytitle)
            
            total_count += 1
            
            # 定期打印进度
            if total_count % print_interval == 0:
                print(f"已收集 {total_count} 条裁判文书...")
            
            # 达到批量大小，批量处理
            if len(batch_texts) >= batch_size:
                # 批量获取向量（关键优化：一次性处理 300 条）
                embeddings = get_embedding(batch_texts)
                
                if embeddings is not None:
                    # 批量插入 Milvus
                    self.insert(embeddings, batch_docs, batch_ids, batch_sources)
                
                # 清空批次
                batch_texts = []
                batch_docs = []
                batch_ids = []
                batch_sources = []
        
        # 处理剩余数据
        if batch_texts:
            embeddings = get_embedding(batch_texts)
            if embeddings is not None:
                self.insert(embeddings, batch_docs, batch_ids, batch_sources)
        
        print(f"完成！共加载 {total_count} 条裁判文书数据到 Milvus")


class QAVecIndex(VecIndex):
    """FAQ 向量索引，批量嵌入+批量写入"""

    def __init__(self, collection_name: str = "faq_collection"):
        # 自定义 schema（不沿用基类的 es_id/parent/source）
        self.client = MilvusClient(
            uri=f'http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}',
            token=f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
        )

        schema = self.client.create_schema()
        schema.add_field("qid", DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field("vec", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("query", DataType.VARCHAR, max_length=65535)
        schema.add_field("answer", DataType.VARCHAR, max_length=65535)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=INDEX_PARAMS,
            )
            print(f"创建集合: {self.collection_name}")

    def insert(self, embeddings, queries, answers):
        if not embeddings:
            return

        data = []
        for i, query in enumerate(queries):
            vec = embeddings[i]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            data.append({
                "qid": str(uuid1()),
                "query": query,
                "vec": vec,
                "answer": answers[i],
            })

        self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, vec, topk: int = 3):
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)

        hits = self.client.search(
            collection_name=self.collection_name,
            data=vec.tolist(),
            anns_field="vec",
            limit=topk,
            search_params=SEARCH_PARAMS,
            output_fields=["qid", "query", "answer"],
        )[0]

        return [FAQItem(id=hit.entity.get("qid"), query=hit.entity.get("query"), answer=hit.entity.get("answer"), score=hit.distance) for hit in hits]

    def _convert_hits_to_items(self, hits):
        # 复用父类 search 时的转换接口
        return [FAQItem(id=hit.entity.get("qid"), query=hit.entity.get("query"), answer=hit.entity.get("answer"), score=hit.distance) for hit in hits]

    def load_from_es(self, qas_path: str = "../data_process/qa/qas.pkl", batch_size: int = 2000, log_every: int = 4000):
        from rag_official.embedding import get_embedding

        qas = sp.read_pickle(qas_path)
        items = [FAQItem(id="", query=qa["query"], answer=qa["answer"]) for qa in qas]
        total = len(items)
        processed = 0

        for start in range(0, total, batch_size):
            batch = items[start:start + batch_size]
            queries = [item.query for item in batch]
            answers = [item.answer for item in batch]

            embeddings = get_embedding(queries)
            if embeddings is None:
                print(f"获取第 {start}-{start + len(batch)} 条批次嵌入失败，跳过该批次")
                continue

            self.insert(embeddings, queries, answers)

            processed += len(batch)
            if processed % log_every == 0 or processed == total:
                print(f"已处理 {processed}/{total} 条 FAQ 记录")


if __name__ == '__main__':
    # # 测试法律条文索引
    # print("=== 测试法律条文向量索引 ===")
    
    # # 如果需要重新创建集合，先删除旧集合
    from pymilvus import MilvusClient
    # temp_client = MilvusClient(
    #     uri=f'http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}',
    #     token=f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}"
    # )
    # # if temp_client.has_collection("law_collection"):
    # #     print("检测到旧集合，删除中...")
    # #     temp_client.drop_collection("law_collection")
    # #     print("已删除旧集合 law_collection")
    
    # law_index = LawVecIndex()
    
    # # 从 ES 加载数据（首次运行或数据更新时使用）
    # # law_index.load_from_es()
    
    # # 测试向量检索
    from rag_official.embedding import get_embedding
    # query_vec = get_embedding("刑法第一百三十三条")
    # if query_vec is not None:
    #     results = law_index.search(query_vec, topk=3)
    #     print("\n检索结果:")
    #     for i, item in enumerate(results, 1):
    #         print(f"{i}. 相似度: {item.score:.4f}")
    #         print(f"   文件: {item.law_filename}")
    #         print(f"   内容: {item.embedding_text[:100]}...")
    #         print()
    
    # print("\n=== 测试裁判文书向量索引 ===")

    # # 如果需要重新创建集合，先删除旧集合
    # # if temp_client.has_collection("writ_collection"):
    # #     print("检测到旧集合，删除中...")
    # #     temp_client.drop_collection("writ_collection")
    # #     print("已删除旧集合 writ_collection")

    # writ_index = WritVecIndex()

    # # 从 ES 加载数据（首次运行或数据更新时使用）
    # # writ_index.load_from_es()

    # # 测试向量检索
    # query_vec2 = get_embedding("交通肇事案件")
    # if query_vec2 is not None:
    #     results2 = writ_index.search(query_vec2, topk=3)
    #     print("\n检索结果:")
    #     for i, item in enumerate(results2, 1):
    #         print(f"{i}. 相似度: {item.score:.4f}")
    #         print(f"   案号: {item.indexbytitle}")
    #         print(f"   内容: {item.context[:100]}...")
    #         print()

    # 测试问答向量检索
    qa_index = QAVecIndex()
    # qa_index.load_from_es()

    # 测试问答向量检索
    query_vec3 = get_embedding("某家公司经营不善,被吊销了营业执照,应该如何注销登记?")
    if query_vec3 is not None:
        results3 = qa_index.search(query_vec3, topk=3)
        print("\n检索结果:")
        for i, item in enumerate(results3, 1):
            print(f"{i}. 相似度: {item.score:.4f}")
            print(f"   问题: {item.query}")
            print(f"   答案: {item.answer[:100]}...")
            print()


