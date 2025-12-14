from heima.conf import settings
from elasticsearch_dsl import Document,Text,connections,Keyword
from heima.lidekui.day02.items import BookItem, LawItem


connections.create_connection(hosts=settings.es_host,
                              http_auth=(settings.es_user, settings.es_password),
                              verify_certs=False,  # 忽略证书验证
                              ssl_show_warn=False  # 可选，禁止SSL警告
                              )


class Book(Document):
    child = Text(analyzer='smartcn')
    parent = Text(analyzer='smartcn')
    source = Keyword()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Index:
        name = 'articles'
        settings = {
            "number_of_shards": 2,
        }

    @classmethod
    def query(self, text):
        items = self.search().query("match", child=text)[:10].execute()
        return [BookItem(child=item.child, parent=item.parent, source=item.source) for item in items]

    @classmethod
    def scan(self):
        s = self.search()
        for item in s.scan():
            yield item


class Law(Document):
    law_title = Text(analyzer='ik_max_word', search_analyzer='ik_smart')
    embedding_text = Text(analyzer='ik_max_word', search_analyzer='ik_smart')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Index:
        name = 'laws'
        settings = {
            "number_of_shards": 2,
        }

    @classmethod
    def query(cls, title):
        items = cls.search().query("match", law_title=title)[:10].execute()
        return [LawItem(law_title=item.law_title, embedding_text=item.embedding_text) for item in items]

    @classmethod
    def scan(self):
        s = self.search()
        for item in s.scan():
            yield item
