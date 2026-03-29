import csv

import numpy as np

from pymilvus import MilvusClient, DataType

from b_rag.xiaokui.tool.items import EnglishItem
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
        schema.add_field(field_name="meaning", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="pronunciation", datatype=DataType.VARCHAR, max_length=100)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, docs, ids, pronunciation):
        data = []
        for i, doc in enumerate(docs):
            data.append({
                "es_id": ids[i],
                "meaning": doc,
                "vec": embeddings[i],
                "pronunciation": pronunciation[i]

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
                                  output_fields=["es_id", "meaning", "pronunciation"])[0]



        # for hit in hits:
        #     print(hit)
        return [EnglishItem(id=hit["entity"].get("es_id"), meaning=hit["entity"].get("meaning"), pronunciation=hit["entity"].get("pronunciation"), score=hit["distance"]) for hit in hits]


    # 实现把数据导入Milvus

    def load(self, path="words_output2.csv"):
        """简化的加载方法（逐条插入）"""
        try:
            from b_rag.day03.embedding import get_embedding

            print(f"开始从CSV文件加载数据: {path}")

            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                for i, row in enumerate(reader):
                    if len(row) < 4:
                        continue

                    seq, word, pronunciation, meaning = row

                    try:
                        # 生成向量
                        embed = get_embedding(word)

                        # 插入到Milvus（单条插入）
                        self.insert(
                            [embed.tolist() if hasattr(embed, 'tolist') else embed],
                            [meaning],
                            [seq],
                            [pronunciation]
                        )

                        if (i + 1) % 100 == 0:
                            print(f"已插入 {i + 1} 条数据")

                    except Exception as e:
                        print(f"处理第 {i + 1} 行时出错: {e}")
                        continue

            print("CSV数据导入完成！")

        except Exception as e:
            print(f"加载数据时出错: {e}")



if __name__ == '__main__':
    from b_rag.day03.embedding import get_embedding

    vec = get_embedding("word")
    hits = VecIndex("word").search(vec)
    for hit in hits:
        print(hit.meaning)
        print(f"ID: {hit.id}, Distance: {hit.score}, meaning: {hit.meaning}")
    vec2 = get_embedding("这是一个测试2")
    VecIndex("word").insert([vec, vec2], ["测试1", "测试2"], ["这是一个测试1", "这是一个测试2"], ["test1", "test2"])


    # VecIndex("word").load()
