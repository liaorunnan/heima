from datetime import datetime
from elasticsearch_dsl import Document, Text, connections
from conf import settings

# 1. 连接ES (不变)
connections.create_connection(
    hosts=settings.es_host,
    verify_certs=False,
    ssl_assert_hostname=False,
    http_auth=(settings.es_user, settings.es_password)
)

# 2. 只定义最核心的字段
class SimpleDoc(Document):
    english = Text()
    chinese = Text()

    class Index:
        name = 'english_chinese'  # 索引名

# 3. 如果索引不存在，则创建映射
SimpleDoc.init()



# 4. 创建并保存文档 (无需手动计算lines等字段)
doc = SimpleDoc(meta={'id': 1},english="Hello world", chinese="你好世界")
doc.save()  # 保存，id会自动生成

print(f"文档保存成功，ID: {doc.meta.id}")