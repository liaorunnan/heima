from datetime import datetime
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections
from conf import settings
import joblib


# Define a default Elasticsearch client
connections.create_connection(hosts=settings.es_host,
                              verify_certs=False,
                              ssl_assert_hostname=False,
                              http_auth=(settings.es_user, settings.es_password),
                              )


class Article(Document):
    # 定义索引名称
    class Index:
        name = 'articles'  # Elasticsearch 索引名称
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }

    # 定义文档字段
    child = Text(analyzer='ik_max_word', search_analyzer='ik_smart')
    parent = Text(analyzer='ik_max_word', search_analyzer='ik_smart')
    source = Keyword()
    created_at = Date()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def save_md2article():
    # 在保存数据前，确保索引存在

    md_contents = joblib.load("./data/md_contents.pkl")
    for i, md_content in enumerate(md_contents):
        # 构建 parent 内容
        parent_content = ''
        if i > 0:
            parent_content += md_contents[i - 1]
        parent_content += md_content
        if i < len(md_contents) - 1:
            parent_content += md_contents[i + 1]

        # 创建文档并保存
        article = Article(
            child=md_content,
            parent=parent_content,
            source='义务教育教科书·语文一年级上册',
            created_at=datetime.now()
        )
        # 保存到 Elasticsearch
        article.save()


if __name__ == '__main__':
    save_md2article()
