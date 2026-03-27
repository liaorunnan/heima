from conf import settings
from elasticsearch_dsl import Document, Text, connections, Keyword
from b_rag.day02.items import BookItem

connections.create_connection(hosts=settings.es_host,
                              http_auth=(settings.es_user, settings.es_password),
                              verify_certs=False,  # 忽略证书验证
                              ssl_show_warn=False  # 可选，禁止SSL警告
                              )


class Word(Document):
    word = Text(analyzer='smartcn')
    pronunciation= Text(analyzer='smartcn')
    mean = Text(analyzer='smartcn')


    class Index:
        name = 'word1'
        settings = {
            "number_of_shards": 2,
        }

    @classmethod
    def query(self, text):
        items = self.search().query("match", word=text)[:10].execute()




        return [BookItem(id=item.meta.id, word = item.word,pronunciation=item.pronunciation, mean=item.mean) for item in items]

    @classmethod
    def scan(self):
        s = self.search()
        for item in s.scan():
            yield item


if __name__ == '__main__':
    Word.init()
    print(connections.get_connection().cluster.health())
    import simple_pickle as sp

    datas = sp.read_pickle("./data/大学英语四级词汇带音标_乱序版.pdf")
    for data in datas:
        word_test= Word(word=data['word'], prononciation=data["prononciation"], mean=data['mean'],score=data['score'])
        word_test.save()

    items =Word.query("aboard")

    # for item in Word.scan():
    #     print(item.meta.id, item.word, item.pronunciation, item.mean)
