from datetime import datetime
from conf import settings
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections
from sqlmodel import Session, select
from conf import settings
from duihua import Duihua, engine  # 替换 your_main_file 为你的实际文件名（不含 .py）
# Define a default Elasticsearch client

def read_duihua():
    with Session(engine) as session:
        # 查询所有记录
        duanhua_list = session.exec(select(Duihua)).all()



        print(f"共读取 {len(duanhua_list)} 条记录：\n")
        sentences = []
        for d in duanhua_list:
            print(f"ID: {d.id} | 内容: {d.duanluo[:50]}...")
            sentences.append(d.duanluo)
        return sentences  # 返回列表
class Article(Document):
    sentence = Text()

    class Index:
        name = 'sentence'
        settings = {
          "number_of_shards": 2,
        }
    def save(self, ** kwargs):

        return super(Article, self).save(** kwargs)
    def is_published(self):
        return datetime.now() > self.published_from

if __name__ == "__main__":
    sentences = read_duihua()


    connections.create_connection(hosts=settings.es_host,
                                  verify_certs=False,
                                  ssl_assert_hostname=False,
                                  http_auth=(settings.es_user, settings.es_password),
                                  )


    num =1
    # create the mappings in elasticsearch
    for sentence in sentences:
        Article.init()
        # create and save and article
        article = Article(meta={'id': num}, sentence='d.duanluo')
        article.sentence = sentence
        article.save()
        num +=1





