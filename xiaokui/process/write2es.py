import warnings
from urllib3.exceptions import InsecureRequestWarning

# 在连接之前添加,关闭警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)
import csv
from elasticsearch_dsl import Document, Text, connections, Keyword
from conf import settings

####################################

connections.create_connection(
        hosts=settings.es_host,
        verify_certs=False,
        ssl_assert_hostname=False,
        http_auth=(settings.es_user, settings.es_password)
    )
# 2. 只定义最核心的字段
def write2es():
    class SimpleDoc(Document):
        # word: 用Keyword便于精确查找和去重
        word = Keyword()
        pronunciation = Text()  # 音标，通常不需要分词
        # meaning: 存储词性和中文意思，使用smartcn便于搜索释义
        meaning = Text(analyzer='smartcn')

        class Index:
            name = "english"  # 索引名
    # 3. 如果索引不存在，则创建映射
    SimpleDoc.init()

##############################

    with open("../data/words_output2.csv", mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            seq, word, pronunciation, meaning = row
            print(seq, word, pronunciation, meaning)
            doc = SimpleDoc(meta={'id': seq}, pronunciation=pronunciation, word=word, meaning=meaning)
            doc.save()  # 保存，id会自动生成



if __name__ == '__main__':
    write2es()