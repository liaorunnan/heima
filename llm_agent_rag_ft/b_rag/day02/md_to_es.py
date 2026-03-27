import warnings
from urllib3.exceptions import InsecureRequestWarning

# 在连接之前添加,关闭警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

import os
from text_splitter import split_text
from elasticsearch_dsl import Document, Text, connections, analyzer, Keyword
from conf import settings

####################################

connections.create_connection(
        hosts=settings.es_host,
        verify_certs=False,
        ssl_assert_hostname=False,
        http_auth=(settings.es_user, settings.es_password)
    )
# 2. 只定义最核心的字段
class SimpleDoc(Document):
    child = Text(analyzer='smartcn')
    parent = Text(analyzer='smartcn')
    source = Keyword()



    class Index:
        name = "book"  # 索引名
# 3. 如果索引不存在，则创建映射
SimpleDoc.init()

##############################
def read_txt(path):
    """
    读取所有txt文件，进行分词后输入到你定义的es中
    :param path:
    :return:
    """
    i = -1
    # 输出路径path中所有文件的文本内容
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            with open(file_path, "r", encoding="utf-8") as file:

                text = file.read()

                text = split_text(text)

                for chile_parent in text:
                    i = i + 1
                    # print(chile_parent)
                    # print(i)
                #     i += 1
                #     # doc = SimpleDoc(meta={'id': 1}, chile_parents=chile_parents)
                #     # 将child和parents拆分出来，分别输入到es中
                    doc = SimpleDoc(meta={'id': i}, child=chile_parent['child'], parent=chile_parent['parent'])
                    print(i)
                    doc.save()  # 保存，id会自动生成
                #     break
                # print("----"*20)


if __name__ == '__main__':
    read_txt("../data/results")