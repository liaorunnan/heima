from typing import List
import numpy as np
from pymilvus import MilvusClient, DataType
from b_rag.day02.items import FAQItem
from conf import settings
from b_rag.day03.embedding import get_embedding  # 假设你有单条函数
# 如果你的 embedding 支持批量，最好加个 get_embeddings_batch

dimension = 1024
search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vec",
    index_type="IVF_FLAT",
    index_name="inverted_index",
    metric_type="IP",
    params={"nlist": 128},
)

class Singleton(type):
    _instances = {}
    def __call__(cls, name):
        k = name
        if k not in cls._instances:
            cls._instances[k] = super(Singleton, cls).__call__(name)
        return cls._instances[k]

class VecIndex(metaclass=Singleton):
    def __init__(self, collection_name):
        self.client = MilvusClient(
            uri=f'http://{settings.milvus_host}:{settings.milvus_port}',
            token=f"{settings.milvus_user}:{settings.milvus_password}"
        )
        schema = self.client.create_schema(
            auto_id=False,                  # 我们自己提供 id
            enable_dynamic_field=False      # 推荐关闭，性能更好
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="query", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="query_embedding_text", datatype=DataType.VARCHAR, max_length=65535)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"创建集合: {self.collection_name}")

    def insert(self, embeddings, querys, answers, ids, embeddings_text=None):
        """批量插入"""
        if embeddings_text is None:
            embeddings_text = [""] * len(embeddings)

        data = [
            {
                "id": ids[i],
                "vec": embeddings[i],
                "query": querys[i],
                "answer": answers[i],
                "query_embedding_text": embeddings_text[i]
            }
            for i in range(len(embeddings))
        ]

        if data:
            res = self.client.upsert(collection_name=self.collection_name, data=data)
            print(f"批量插入 {len(data)} 条数据成功，影响行数: {res['upsert_count']}")

    def search(self, vec, topk=3):
        if vec is None:
            print("查询向量为 None，返回空结果")
            return []

        # 确保二维格式 [[vec]]
        if isinstance(vec, np.ndarray):
            query_vec = vec.reshape(1, -1).tolist()
        elif isinstance(vec, list):
            query_vec = [vec]
        else:
            raise ValueError("向量格式不支持")

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                data=query_vec,
                anns_field="vec",
                limit=topk,
                search_params=search_params,
                output_fields=["id", "query", "answer", "query_embedding_text"]
            )[0]  # hits 是 list of Hit

            if not hits:
                print("未找到匹配结果")
                return []

            results = []
            for hit in hits:
                entity = hit.entity
                results.append(FAQItem(
                    id=entity.get("id"),
                    query=entity.get("query"),
                    answer=entity.get("answer"),
                    query_embedding_text=entity.get("query_embedding_text"),
                    score=hit.distance  # Milvus 用 hit.distance 作为相似度分数（IP 越大越相似）
                ))
            return results

        except Exception as e:
            print(f"Milvus search 异常: {e}")
            import traceback
            traceback.print_exc()
            return []

    def load(self, items: List[FAQItem]):
        if not items:
            print("无数据可加载")
            return

        total = len(items)
        batch_size = 500  # 每批 500 条，可根据你的机器调整（300-1000 都行）
        print(f"开始分批加载 {total} 条 FAQ 数据（每批 {batch_size} 条）...")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_items = items[start:end]
            batch_num = start // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"\n--- 处理第 {batch_num}/{total_batches} 批 ({start + 1}-{end}/{total}) ---")

            # 提取本批数据
            queries = [item.query for item in batch_items]
            ids = [item.id for item in batch_items]
            answers = [item.answer for item in batch_items]
            query_texts = [item.query_embedding_text or "" for item in batch_items]

            # 本批计算 embedding
            print(f"  正在计算本批 {len(queries)} 条 embedding...")
            try:
                # 如果你有批量函数，最优
                from b_rag.day03.embedding import get_embeddings_batch
                batch_embeddings = get_embeddings_batch(queries)
            except ImportError:
                # 降级：逐条计算，但加进度条
                from tqdm import tqdm
                batch_embeddings = [
                    get_embedding(q).tolist()
                    for q in tqdm(queries, desc=f"批 {batch_num} embedding", leave=False)
                ]

            # 本批插入 Milvus
            print(f"  正在插入本批 {len(batch_embeddings)} 条数据到 Milvus...")
            self.insert(
                embeddings=batch_embeddings,
                querys=queries,
                answers=answers,
                ids=ids,
                embeddings_text=query_texts
            )

        print(f"\n🎉 所有 {total} 条数据分批导入完成！")