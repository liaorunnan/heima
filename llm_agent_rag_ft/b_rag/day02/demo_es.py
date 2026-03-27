from datetime import datetime  # 用于处理日期和时间
from conf import settings  # 假设这是包含Elasticsearch配置的自定义模块
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text, connections  # Elasticsearch DSL组件

# 创建一个默认的Elasticsearch客户端连接
connections.create_connection(
    hosts=settings.es_host,  # Elasticsearch服务器地址
    verify_certs=False,  # 不验证SSL证书
    ssl_assert_hostname=False,  # 不验证主机名
    http_auth=(settings.es_user, settings.es_password),  # HTTP认证信息
)

# 定义Article文档类，继承自Document
class Article(Document):
    # 定义字段类型及分析器

    child = Text(analyzer='snowball',)  # 子字段，附带原始关键词子字段 = Keyword()  # 标签字段
    parents =Text(analyzer='snowball')  # 行数字段

    class Index:
        name = 'db_es'  # 索引名称
        settings = {
            "number_of_shards": 2,  # 主分片数量
        }

    # 重写save方法，计算并设置行数属性
    def query(self, query_text):
        from elasticsearch_dsl import Q

        # 构建 multi_match 查询（注意：是 "multi_match"，不是 "mutil_match"）
        s = Article.search()
        s = s.query(
            "multi_match",
            query=query_text,
            fields=['child', 'parents'],  # 指定要搜索的字段
            type="best_fields",  # 默认类型，也可用 "most_fields", "cross_fields" 等
            fuzziness="AUTO"  # 可选：允许模糊匹配（如拼写容错）
        )

        # 执行查询
        response = s.execute()

        # 打印或返回结果
        results = []
        for hit in response:
            results.append({
                'id': hit.meta.id,
                'score': hit.meta.score,
                'child': getattr(hit, 'child', None),
                'parents': getattr(hit, 'parents', None)

            })
            # 如果你想直接打印：
            # print(f"ID: {hit.meta.id}, Score: {hit.meta.score}, source: {hit.source}")

        return results

if __name__ == '__main__':
    Article.init()
    article = Article()
    print(article.query("天地"))