from datetime import datetime
from conf import settings
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections
from sqlmodel import Session, select
from conf import settings
from get_database import Wenda, engine  # 替换 your_main_file 为你的实际文件名（不含 .py）
# Define a default Elasticsearch client

def read_wenda():
    with Session(engine) as session:
        # 查询所有记录
        wenda_list = session.exec(select(Wenda)).all()



        # print(wenda_list)
        q_a_list = []
        for d in wenda_list:
            print(f"ID: {d.id} | 内容: {d.wenda}...")
            q_a_list.append(d.wenda)
        return q_a_list  # 返回列表
class Article(Document):
    q_a = Text()

    class Index:
        name = 'wenda'
        settings = {
          "number_of_shards": 2,
        }
    def save(self, ** kwargs):

        return super(Article, self).save(** kwargs)
    def is_published(self):
        return datetime.now() > self.published_from

if __name__ == "__main__":
    q_a_list1 = read_wenda()


    connections.create_connection(hosts=settings.es_host,
                                  verify_certs=False,
                                  ssl_assert_hostname=False,
                                  http_auth=(settings.es_user, settings.es_password),
                                  )


    num =1
    # create the mappings in elasticsearch
    for q_a in q_a_list1:
        Article.init()
        # create and save and article
        article = Article(meta={'id': num}, q_a='d.wenda')
        article.q_a = q_a
        article.save()
        num +=1