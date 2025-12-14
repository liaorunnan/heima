import numpy as np



from pymilvus import MilvusClient, DataType

from items import QaItem

from conf import settings
from embedding import get_embedding
from typing import List

dimension = 1024
search_params = {"metric_type": "IP"}#, "params": {"nprobe": 10}}
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vec",
    # index_type="IVF_FLAT",
    index_type="FLAT",
    index_name="inverted_index",
    metric_type="IP",
    # params={"nlist": 128},
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
        schema.add_field(field_name="qid", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="query", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=100)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings,query, answer, ids, source):
        data = []
   
    
    
        data.append({
            "qid": ids,
            "query": query,
            "answer": answer,
            "vec": embeddings,
            "source": ''
        })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, query, topk=10):
        vec = get_embedding(query)
        vec = np.expand_dims(vec, axis=0)
      
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["query","answer","source"])


        
        return [QaItem(qid=hit[0]['qid'], query=hit[0]['entity']["query"], answer=hit[0]['entity']["answer"], source=[hit[0]['entity']["source"]], score=hit[0]['distance']) for hit in hits]

    def load(self,data: List[QaItem]):

        from embedding import get_embedding
       
        num = 0
        for item in data:
            embed = get_embedding(item.query)
            self.insert([embed.tolist()], item.query, item.answer, [item.qid], item.source)
            num += 1
            print(num)
      


if __name__ == '__main__':
  
    VecIndex("fqa").load()
    # print(VecIndex("children").search("freely"))

