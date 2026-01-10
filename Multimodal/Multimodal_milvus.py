import numpy as np
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'



from pymilvus import MilvusClient, DataType

from rag.items import MuilPhotoItem, MuilTextItem

from conf import settings
from Multimodal.Multimodel_photo_emb import get_image_embedding
from typing import List

dimension = 768
search_params = {"metric_type": "IP"}#, "params": {"nprobe": 10}}
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="photo_vec",
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
        schema.add_field(field_name="photo_vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="photo_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="photo_meta", datatype=DataType.VARCHAR, max_length=65535)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings, photo_name, photo_meta, ids):
        data = []
   
        data.append({
            "qid": ids,
            "photo_vec": embeddings,
            "photo_name": photo_name,
            "photo_meta": photo_meta
        })

        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)

    def search(self, query, topk=10):
        vec = get_image_embedding(text=query)
        vec = np.expand_dims(vec, axis=0)
      
        hits = self.client.search(collection_name=self.collection_name,
                                  data=vec.tolist(),
                                  anns_field="photo_vec",
                                  limit=topk,
                                  search_params=search_params,
                                  output_fields=["photo_name","photo_meta"])
        print(hits)
        exit()


        
        return [MuilPhotoItem(qid=hit[0]['qid'], photo_name=hit[0]['entity']["photo_name"], photo_meta=hit[0]['entity']["photo_meta"], score=hit[0]['distance']) for hit in hits]

    def load(self,data: List[MuilPhotoItem]):

    
       
        num = 0
        for item in data:
            embed = get_image_embedding(item.photo_name)
            self.insert([embed.tolist()], item.photo_name, item.photo_meta, [item.qid])
            num += 1
            print(num)
      


if __name__ == '__main__':

    # data = {
    #     "qid": "1",
    #     "photo_name": "./1.png",
    #     "photo_meta": {"photo_type": "test"}
    # }
    # photo_emb = get_image_embedding(data["photo_name"])
    # print(photo_emb.shape)
    # VecIndex("photo").insert(embeddings=photo_emb.tolist(),
    #                          photo_name=data["photo_name"],
    #                          photo_meta=json.dumps(data["photo_meta"]),
    #                          ids=data["qid"])
  

    print(VecIndex("photo").search("ldihaosehwfoiaweofi"))

