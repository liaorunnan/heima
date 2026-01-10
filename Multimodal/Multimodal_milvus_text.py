import numpy as np
import json



from pymilvus import MilvusClient, DataType

from rag.items import MuilPhotoItem, MuilTextItem

from conf import settings
from Multimodal.Multimodel_photo_emb import get_image_embedding
from typing import List

dimension = 768
search_params = {"metric_type": "IP"}#, "params": {"nprobe": 10}}
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="text_vec",
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
        schema.add_field(field_name="text_vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="text_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="text_meta", datatype=DataType.VARCHAR, max_length=65535)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, text_name, text_meta, ids):
        data = []
   
        data.append({
            "qid": ids,
            "text_vec": embeddings,
            "text_name": text_name,
            "text_meta": text_meta
        })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, textquery, topk=10):
        vec = get_image_embedding(text=textquery)
        vec = np.expand_dims(vec, axis=0)
      
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="text_vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["text_name","text_meta"])



        
        return [MuilTextItem(qid=hit[0]['qid'], text_name=hit[0]['entity']["text_name"], text_meta=hit[0]['entity']["text_meta"], score=hit[0]['distance']) for hit in hits]

    def load(self,data: List[MuilTextItem]):

    
       
        num = 0
        for item in data:
            embed = get_image_embedding(item.text_name)
            self.insert([embed.tolist()], item.text_name, item.text_meta, [item.qid])
            num += 1
            print(num)
      


if __name__ == '__main__':

    # data = {
    #     "qid": "1",
    #     "text_name": "1234",
    #     "text_meta": {"text_type": "test"}
    # }
    # text_emb = get_image_embedding(text=data["text_name"])
    # print(text_emb.shape)
    # VecIndex("text").insert(embeddings=text_emb.tolist(), text_name=data["text_name"], text_meta=json.dumps(data["text_meta"]), ids=data["qid"])   
                           
  
    # VecIndex("fqa").load()
    print(VecIndex("text").search("1234"))

