import numpy as np



from pymilvus import MilvusClient, DataType

from items import YinyutlItem

from conf import settings
from embedding import get_embedding

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
        schema.add_field(field_name="es_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="child", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="parent", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=100)

        self.collection_name = collection_name
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)

    def insert(self, embeddings,child, parent, ids, source):
        data = []
   
    
    
        data.append({
            "es_id": ids[0],
            "child": child,
            "parent": parent,
            "vec": embeddings[0],
            "source": source[0][0]
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
                                  output_fields=["es_id", "child", "parent", "source"])[0]
     
     
       
        
        return [YinyutlItem(id=hit.get("es_id"), child=hit.entity.get("child"), parent=hit.entity.get("parent"), source=[hit.entity.get("source")], score=hit.distance) for hit in hits]

    def load(self):

        from match_keyword import Yinyutl
        from embedding import get_embedding
       
        num = 0
        for item in Yinyutl.scan():
            embed = get_embedding(item.child)
            self.insert([embed.tolist()], item.child, item.parent, [item.meta.id], [item.source])
            num += 1
            print(num)
      


if __name__ == '__main__':
  
    VecIndex("children").load()
    # print(VecIndex("children").search("freely"))

