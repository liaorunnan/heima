from tqdm import tqdm

from b_rag.day02.database import LawChunk
from conf import settings
from elasticsearch_dsl import Document, Text, connections, Keyword, Integer, analyzer
import json
from b_rag.day02.items import LawsItem

connections.create_connection(
    hosts=settings.es_host,  # 集群节点地址
    verify_certs=False,
    ssl_assert_hostname=False,
    http_auth=(settings.es_user,settings.es_password),
)

class Law(Document):
    # === 主键与元数据 ===
    document_id = Integer()  # 原始文档ID
    law_filename = Keyword()  # 如：人民检察院公益诉讼办案规则_2023
    law_title = Keyword()  # 法律标题
    part_name = Keyword()  # 编（如：第一编 总则）
    chapter_name = Keyword()  # 章（如：第一章 总则）
    section_name = Keyword()  # 节（如：第一节 管辖）
    article_id = Keyword()  # 第X条、第X款（最核心！）


    content_text = Text(analyzer="smartcn")
    embedding_vector = Keyword()
    embedding_text = Text(analyzer="smartcn")  # 存原始用于生成向量的文本（可选）
    token_count = Integer()  # 统计 token 数（用于成本控制）


    class Index:
        name = 'laws_chunks'
        settings = {
          "number_of_shards": 2,
        }

    @classmethod
    def query(cls, text: str):
        """关键词搜索，返回 LawsItem 列表"""
        search = cls.search()
        results = search.query("match", content_text=text)[:10].execute()

        items = []
        for hit in results:
            # hit 是一个 Law 对象，直接转成 dict 再构造 LawsItem
            item = LawsItem(
                id=hit.meta.id,  # ES 的 _id
                document_id=hit.document_id,
                law_filename=hit.law_filename,
                law_title=hit.law_title,
                part_name=getattr(hit, 'part_name', None),
                chapter_name=getattr(hit, 'chapter_name', None),
                section_name=getattr(hit, 'section_name', None),
                article_id=hit.article_id,
                content_text=hit.content_text,
                embedding_text=getattr(hit, 'embedding_text', None),
                token_count=getattr(hit, 'token_count', 0),
            )
            items.append(item)
        return items

    @classmethod
    def scan(self):
        s = self.search()
        for item in s.scan():
            yield item
Law.init()
print("Elasticsearch 索引已准备就绪")
# ============ 一键全量同步（核心代码）============
def sync_all_to_es():
    # 1. 从数据库查出所有数据（你原来的 query_all）
    db_chunks = LawChunk.query_all()
    print(f"从 MySQL 查出 {len(db_chunks)}  条法律条文，开始同步到 Elasticsearch...")

    # 2. 遍历 + 创建 Law 对象 + save()（100% 模仿官方示例）
    for db_chunk in tqdm(db_chunks, desc="同步进度", unit="条"):
        # 处理向量（现在是摆设，后面再填）
        vector = None
        if db_chunk.embedding_vector:
            try:
                vector = json.loads(db_chunk.embedding_vector)
            except:
                pass

        # 模仿官方示例的写法：创建对象 → 设属性 → save()
        law = Law(meta={'id': db_chunk.id})   # 指定 _id，和数据库保持一致
        law.document_id = db_chunk.document_id
        law.law_filename = db_chunk.law_filename
        law.law_title = db_chunk.law_title
        law.part_name = db_chunk.part_name or ""
        law.chapter_name = db_chunk.chapter_name or ""
        law.section_name = db_chunk.section_name or ""
        law.article_id = db_chunk.article_id
        law.content_text = db_chunk.content_text
        law.embedding_text = db_chunk.embedding_text or db_chunk.content_text
        law.token_count = db_chunk.token_count or 0
        law.embedding_vector = vector           # 现在是 None，后面补上就行

        law.save()   # ← 完全等价于官方示例的 article.save()

    print("全量同步完成！所有法条已成功存入 Elasticsearch！")

# ============ 运行 ============
if __name__ == "__main__":
    sync_all_to_es()

    # 可选：打印集群健康状态（和官方示例一模一样）
    print("集群状态:", connections.get_connection().cluster.health())

