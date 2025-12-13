from uuid import uuid1

import numpy as np

from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import joblib
from b_rag.day02.items import BookItem, LawItem, FAQItem
from conf import settings

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
        schema = self.client.create_schema()
        schema.add_field(field_name="es_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="parent", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=100)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, docs, ids, source):
        data = []
        for i, doc in enumerate(docs):
            data.append({
                "es_id": ids[i],
                "parent": doc,
                "vec": embeddings[i],
                "source": source[i]
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, vec, topk=3):
        vec = np.expand_dims(vec, axis=0)
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["es_id", "parent", "source"])[0]
        # print(hits)
        return [BookItem(id=hit.entity.get("es_id"), parent=hit.entity.get("parent"), source=hit.entity.get("source"), score=hit.distance) for hit in hits]
        # return hits

    def load(self):
        from b_rag.day02.match_keyword import Book
        from b_rag.day03.embedding import get_embedding

        for item in Book.scan():
            embed = get_embedding(item.child)
            VecIndex(self.collection_name).insert([embed.tolist()], [item.parent], [item.meta.id], [item.source])
            print(item.meta.id, item.child, item.parent, item.source)


class VecIndexLaw(metaclass=Singleton):
    def __init__(self, collection_name):
        self.client = MilvusClient(
            uri=f'http://{settings.milvus_host}:{settings.milvus_port}',
            token=f"{settings.milvus_user}:{settings.milvus_password}"
        )
        # 临时添加删除集合的代码
        # self.client.drop_collection(collection_name)

        schema = self.client.create_schema()
        schema.add_field(field_name="es_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="embedding_text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="law_title", datatype=DataType.VARCHAR, max_length=10000)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, docs, ids, source):
        data = []
        for i, doc in enumerate(docs):
            data.append({
                "es_id": ids[i],
                "embedding_text": doc,
                "vec": embeddings[i],
                "law_title": source[i]
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, vec, topk=3):
        vec = np.expand_dims(vec, axis=0)
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["es_id", "embedding_text", "law_title"])[0]
        # print(hits)
        return [LawItem(id=hit.entity.get("es_id"), embedding_text=hit.entity.get("embedding_text"), law_title=hit.entity.get("law_title"), score=hit.distance) for hit in hits]
        # return hits

    def load(self, batch_size=100):
        from b_rag.day02.match_keyword import Law
        from b_rag.day03.embedding import get_embedding

        items = list(Law.scan())  # 先转成列表以便分批（如果数据量极大，可改用生成器+缓冲）
        total = len(items)
        print(f"📥 准备加载 {total} 条法律条文到集合 '{self.collection_name}'，batch_size={batch_size}")

        for i in tqdm(range(0, total, batch_size)):
            batch_items = items[i:i + batch_size]

            # 批量获取 embeddings 和字段
            embeddings = []
            docs = []
            ids = []
            law_titles = []

            for item in batch_items:
                embed = get_embedding(item.embedding_text)
                embeddings.append(embed.tolist())
                docs.append(item.embedding_text)
                ids.append(item.meta.id)
                law_titles.append(item.law_title)

            # 批量插入
            self.insert(embeddings, docs, ids, law_titles)

            print(f"✅ 已插入 [{i + 1} ~ {min(i + batch_size, total)}] / {total}")


class VecIndexFaq(metaclass=Singleton):
    def __init__(self, collection_name):
        self.client = MilvusClient(
            uri=f'http://{settings.milvus_host}:{settings.milvus_port}',
            token=f"{settings.milvus_user}:{settings.milvus_password}"
        )
        # 临时添加删除集合的代码
        # self.client.drop_collection(collection_name)

        schema = self.client.create_schema()
        schema.add_field(field_name="es_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, question, answer):
        data = []
        for i, doc in enumerate(question):
            data.append({
                "es_id": str(uuid1()),
                "question": doc,
                "vec": embeddings[i],
                "answer": answer[i]
            })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, vec, topk=3):
        vec = np.expand_dims(vec, axis=0)
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["es_id", "answer", "question"])[0]
        # print(hits)
        return [FAQItem(id=hit.entity.get("es_id"), answer=hit.entity.get("answer"), question=hit.entity.get("question"), score=hit.distance) for hit in hits]
        # return hits

    def load(self, batch_size=100):
        from b_rag.day03.embedding import get_embedding

        items = joblib.load("../day05/data/qas.pkl")
        items = [FAQItem(question=item['query'], answer=item['answer']) for item in items]
        total = len(items)
        print(f"📥 准备加载 {total} 条法律条文到集合 '{self.collection_name}'，batch_size={batch_size}")

        for i in tqdm(range(0, total, batch_size)):
            batch_items = items[i:i + batch_size]

            # 批量获取 embeddings 和字段
            answers = []
            questions = []
            embeddings = []

            for item in batch_items:
                embed = get_embedding(item.question)
                embeddings.append(embed.tolist())
                questions.append(item.question)
                answers.append(item.answer)

            # 批量插入
            self.insert(embeddings, questions, answers)

            print(f"✅ 已插入 [{i + 1} ~ {min(i + batch_size, total)}] / {total}")


if __name__ == '__main__':
    from b_rag.day03.embedding import get_embedding

    vec = get_embedding("某合伙企业在清算期间，其中一个合伙人私自开展了与清算无关的经营活动，违反了合伙企业法的相关规定。其他合伙人该如何应对？")
    hits = VecIndexFaq("laws_faq").search(vec)
    # hits = VecIndexLaw("laws").search(vec)
    # print(hits)
    for hit in hits:
        print(hit.answer, hit.score, hit.question)
    #     print(f"ID: {hit.id}, Distance: {hit.score}, Parent: {hit.parent}")
    #     print(f"ID: {hit.id}, Distance: {hit.score}, Embedding_text: {hit.embedding_text}")
    # vec2 = get_embedding("这是一个测试2")
    # VecIndex("children").insert([vec, vec2], ["测试1", "测试2"])

    # VecIndex("children").load()
    # VecIndexLaw("laws").load()
    # VecIndexFaq('laws_faq').load()
