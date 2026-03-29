from datetime import datetime
from elasticsearch_dsl import Document, Text, connections, Keyword

from b_rag.xiaokui.tool.items import EnglishItem
from conf import settings

import warnings
from urllib3.exceptions import InsecureRequestWarning
# 在连接之前添加,关闭警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# 1. 连接ES (不变)
connections.create_connection(
    hosts=settings.es_host,
    verify_certs=False,
    ssl_assert_hostname=False,
    http_auth=(settings.es_user, settings.es_password)
)

# 2. 只定义最核心的字段
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


def search_documents(query="hello", size=2):
    """
    查询ES中与指定query最相关的前N个文档
    :param query: 查询关键词
    :param size: 返回文档数量，默认10
    :return: 查询结果列表
    """
    # 使用search_result对对象进行一个查找
    import re
    english_words = re.findall(r'[a-zA-Z]+', query)
    search_term = english_words[0] if english_words else query
    # 组合精确匹配和模糊匹配
    search_result = SimpleDoc.search() \
        .query("term", word=search_term) \
        .extra(size=size)
    # 如果没有精确匹配结果，再进行模糊搜索
    if not search_result.execute().hits:
        search_result = SimpleDoc.search() \
            .query("multi_match", query=query, fields=['word', 'meaning']) \
            .extra(size=size)
    # exact_query = SimpleDoc.search() \
    #     .query("term", word=query)
    # search_result = SimpleDoc.search() \
    #     .query("multi_match", query=query, fields=['word', 'meaning']) \
    #     .extra(size=size) # 返回文档数量

    # 将结果转换为BookItem对象列表
    book_items = []
    for hit in search_result:
        # 创建BookItem对象
        book_item = EnglishItem(
            id=hit.meta.id,
            word=hit.word,
            meaning=hit.meaning,
            pronunciation=hit.pronunciation,
            score=hit.meta.score  # 使用归一化后的分数
        )
        book_items.append(book_item)

    return book_items


    # 返回前十篇的parent
    # return [result['parent'] for result in results]
    #
    # return  results[0]['parent']

# def query_search(query="hello", size=10):
#     for i, result in enumerate(search_documents(query, size), 1):
#         print(f"{i}. {result['word']} - {result['meaning']}")
#         print(f"   Score: {result['score']}")

# 使用示例
if __name__ == '__main__':
    # 查询与"你好"相关的前十个文档
    print(search_documents("英语单词word是什么意思"))


