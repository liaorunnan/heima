

from conf import settings
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections,DenseVector

connections.create_connection(hosts=settings.es_host, http_auth=(settings.es_user, settings.es_password),
                              verify_certs=False, ssl_assert_hostname=False)

from rag.items import YinyutlItem
from elasticsearch_dsl.query import Script


import urllib3
import warnings
import tqdm
from pdf import PDF
from bs4 import BeautifulSoup
from pathlib import Path
import os

from rag.embedding import get_embedding




# 屏蔽 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Connecting to 'https://localhost:9200' using TLS")

from text_splitter import split_text, get_html

from pydantic import BaseModel
from typing import List


class TestEmbItem(BaseModel):
    id: str
    child: str
    parent: str
    source: List[str]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TestEmbItem):
            return self.id == other.id
        
        return False



class TestEmb(Document):
    child = Text(analyzer="smartcn")
    parent = Text(analyzer="smartcn")
    vector = DenseVector(dims=1024, index=True, similarity="cosine")
    source = Keyword()

    class Index:
        name = 'testemb'
        settings = {
            "number_of_shards": 2,
        }

    def query(self, item,type=''):
  
        items = self.search().query("match", child=item)[:10].execute()
        return [TestEmbItem(id=item.meta.id, child=item.child, parent=item.parent, source=item.source) for item in items]

    def query_by_vector(self, query_embedding, k=10):
        s = self.search()
        s = s.knn(field='vector', k=k, num_candidates=10, query_vector=vector)
        items = s.execute()

        return [
            TestEmbItem(
                id=item.meta.id, 
                child=item.child, 
                parent=item.parent, 
                source=item.source
                # 如果需要查看相似度分数，可以使用 score=item.meta.score
            ) 
            for item in items
        ]


    @classmethod
    def scan(self):
        s = self.search()
      
        for item in s.scan():
        
            yield item


if __name__ == '__main__':

    num = 2

    child= 'child'
    parent = 'parent'
    source = ['四级英语听力']

    vector = get_embedding(child).tolist()

    TestEmb.init()

    TestEmb_obj = TestEmb()

    print(TestEmb_obj.query_by_vector(vector))
  

    # TestEmb = TestEmb(meta={'id': num}, child=child, parent=parent, vector=vector, source=source)
    # TestEmb.save()   
    # print('已保存 1 条数据')



              
        





