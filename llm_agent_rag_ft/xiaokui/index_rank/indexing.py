import numpy as np

from pymilvus import MilvusClient, DataType

from xiaokui.items import BookItem
from conf import settings

dimension = 1024
search_params = {"metric_type": "IP"}
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vec",
    index_type="FLAT",
    index_name="inverted_index",
    metric_type="IP",

)


class Singleton(type): # 单例模式
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
        schema.add_field(field_name="word", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="pronunciation", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="mean", datatype=DataType.VARCHAR, max_length=1000)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings,  pronunciation, mean, ids,word):
        data = []

        data.append({
            "es_id": ids,
            "word":word,
            "vec": embeddings,
            "pronunciation": pronunciation,
            "mean": mean
        })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self,vec, topk=3):
        word = np.expand_dims(vec, axis=0)
        hits = self.client.search(collection_name=self.collection_name,
                                  data=word.tolist(),
                                  anns_field="vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["word", "pronunciation", "mean"])[0]

        # return [BookItem(id=hit.entity.get("es_id"), pronunciation=hit.entity.get("pronunciation"), mean=hit.entity.get("mean"), score=hit.distance) for hit in hits]
        return [BookItem(id=hit.entity.get("es_id"), pronunciation=hit.entity.get("pronunciation"), word=hit.entity.get("word"),  mean=hit.entity.get("mean") ) for hit in hits]



    def load(self):
        from xiaokui.es_table import Word
        from embedding import get_embedding

        for item in Word.scan():
            embed = get_embedding(item.word)
            print(embed)
            VecIndex("word").insert(embed.tolist(), item.pronunciation, item.mean, item.meta.id, item.word)
            print(item.meta.id, item.word, item.pronunciation, item.mean)


if __name__ == '__main__':
    from embedding import get_embedding
    #
    vec = get_embedding("abandon")
    print(vec)
    hits = VecIndex("word").search(vec)
    print(hits)
    # for hit in hits:
    #     print(hit.parent)
    # print(f"ID: {hit.entity.get('es_id')}, Distance: {hit.distance}, Parent: {hit.entity.get('parent')}")
    # vec2 = get_embedding("这是一个测试2")
    # VecIndex("children").insert([vec, vec2], ["测试1", "测试2"])

    # VecIndex("word").load()
