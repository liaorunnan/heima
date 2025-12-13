from elasticsearch_dsl import Document, Text, connections, Keyword
from sqlmodel import Session, create_engine, select
from config import settings
from data_process.split_data_tomysql import LawChunk   # 导入数据库模型
from data_process.split_data_tomysql import WritRecord
from data_process.items import LawItem, WritItem

# 定义 Elasticsearch 连接
connections.create_connection(
    hosts=settings.ES_HOST,
    verify_certs=False,  # # 忽略证书验证
    ssl_show_warn=False,  # 可选，禁止SSL警告
    ssl_assert_hostname=False,
    http_auth=(settings.ES_USER, settings.ES_PASSWORD),
)

# 定义 Elasticsearch 文档模型
class ESLawChunk(Document):
    """
    对应 law_chunks 表的数据结构，用于存储到 Elasticsearch
    """
    chunk_id = Keyword()   # 精确匹配字段
    document_id = Keyword()
    law_filename = Text()  # 全文搜索字段
    law_title = Text()
    part_name = Text()
    chapter_name = Text()
    section_name = Text()
    article_id = Text()
    content_text = Text(analyzer='smartcn')
    embedding_text = Text(analyzer='smartcn') # 可根据需要调整分词器

    class Index:
        name = 'law_chunks_index'
        settings = {
            "number_of_shards": 2,
        }
    @classmethod
    def query(cls, text, size=10):
        """
        关键字搜索法律条文
        
        参数:
            text: 搜索关键字
            size: 返回结果数量，默认 10 条
        
        返回:
            List[LawItem]: 法律条文列表
        """
        items = cls.search().query("match", content_text=text)[:size].execute()
        return [LawItem(id=item.meta.id, law_filename=item.law_filename, embedding_text=item.embedding_text) for item in items]

    @classmethod
    def scan(cls):
        s = cls.search()
        for item in s.scan():
            yield item


class ESWritChunk(Document):
    """对应 writ 表，用于存储到 Elasticsearch"""

    indexbytitle = Text(analyzer='smartcn')
    context = Text(analyzer='smartcn')

    class Index:
        name = 'writ_chunks_index'
        settings = {
            "number_of_shards": 2,
        }

    @classmethod
    def query(cls, text, size=10):
        """
        关键字搜索裁判文书
        
        参数:
            text: 搜索关键字
            size: 返回结果数量，默认 10 条
        
        返回:
            List[WritItem]: 裁判文书列表
        """
        items = cls.search().query("match", indexbytitle=text)[:size].execute()
        return [WritItem(id=item.meta.id, indexbytitle=item.indexbytitle, context=item.context) for item in items]

    @classmethod
    def scan(cls):
        s = cls.search()
        for item in s.scan():
            yield item

def get_sqlmodel_engine():
    """根据 config 构造 SQLModel Engine"""
    return create_engine(settings.url, echo=False)  # echo 不输出SQL 相关日志

def fetch_chunks_from_db(engine):
    """使用 SQLModel 从 MySQL 数据库获取所有 law_chunks 数据"""
    chunks = []
    try:
        with Session(engine) as session:
            # 使用 SQLModel 的 select API 查询所有数据
            statement = select(LawChunk)
            results = session.exec(statement)
            for db_chunk in results:
                chunks.append({
                    'id': db_chunk.id,
                    'document_id': db_chunk.document_id,
                    'law_filename': db_chunk.law_filename,
                    'law_title': db_chunk.law_title,
                    'part_name': db_chunk.part_name,
                    'chapter_name': db_chunk.chapter_name,
                    'section_name': db_chunk.section_name,
                    'article_id': db_chunk.article_id,
                    'content_text': db_chunk.content_text,
                    'embedding_text': db_chunk.embedding_text,
                })
        print(f"使用 SQLModel 从数据库获取到 {len(chunks)} 条数据")
    except Exception as e:
        print(f"从数据库获取数据时出错: {e}")
    return chunks


def fetch_writs_from_db(engine):
    """使用 SQLModel 从 writ 表获取数据"""
    items = []
    try:
        with Session(engine) as session:
            statement = select(WritRecord)
            results = session.exec(statement)
            for row in results:
                items.append({
                    'id': row.id,
                    'indexbytitle': row.indexbytitle,
                    'context': row.context,
                })
        print(f"使用 SQLModel 从数据库获取到 {len(items)} 条 writ 记录")
    except Exception as e:
        print(f"从数据库获取 writ 数据时出错: {e}")
    return items

# 把法条插入到ES
def sync_law_to_es():
    """同步 law_chunks -> Elasticsearch"""
    # 如果索引结构有变化，可考虑先删除旧索引
    ESLawChunk.init()
    print("Elasticsearch law 索引初始化完成")

    engine = get_sqlmodel_engine()
    chunks_data = fetch_chunks_from_db(engine)

    success_count = 0
    error_count = 0
    batch_size = 500

    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i + batch_size]
        actions = []
        for item in batch:
            try:
                doc = ESLawChunk(
                    meta={'id': item['id']},
                    chunk_id=str(item['id']),
                    document_id=str(item['document_id']),
                    law_filename=item['law_filename'],
                    law_title=item['law_title'],
                    part_name=item['part_name'] or '',
                    chapter_name=item['chapter_name'] or '',
                    section_name=item['section_name'] or '',
                    article_id=item['article_id'],
                    content_text=item['content_text'],
                    embedding_text=item['embedding_text']
                )
                actions.append(doc)
            except Exception as e:
                print(f"准备 law 文档时出错 (ID: {item.get('id')}): {e}")
                error_count += 1

        if actions:
            try:
                for doc in actions:
                    doc.save()
                success_count += len(actions)
                print(f"成功同步 law 批次 {i // batch_size + 1}, 数量: {len(actions)}")
            except Exception as e:
                print(f"批量同步 law 到 Elasticsearch 时出错: {e}")
                error_count += len(actions)

    print(f"law 数据同步完成。成功: {success_count}, 失败: {error_count}")


# 把文书插入到ES
def sync_writ_to_es():
    """同步 writ -> Elasticsearch"""
    ESWritChunk.init()
    print("Elasticsearch writ 索引初始化完成")

    engine = get_sqlmodel_engine()
    writs_data = fetch_writs_from_db(engine)

    success_writ = 0
    error_writ = 0
    batch_size = 500

    for i in range(0, len(writs_data), batch_size):
        batch = writs_data[i:i + batch_size]
        actions = []
        for item in batch:
            try:
                doc = ESWritChunk(
                    meta={'id': item['id']},
                    indexbytitle=item['indexbytitle'],
                    context=item['context'],
                )
                actions.append(doc)
            except Exception as e:
                print(f"准备 writ 文档时出错 (ID: {item.get('id')}): {e}")
                error_writ += 1

        if actions:
            try:
                for doc in actions:
                    doc.save()
                success_writ += len(actions)
                print(f"成功同步 writ 批次 {i // batch_size + 1}, 数量: {len(actions)}")
            except Exception as e:
                print(f"批量同步 writ 到 Elasticsearch 时出错: {e}")
                error_writ += len(actions)

    print(f"writ 数据同步完成。成功: {success_writ}, 失败: {error_writ}")

def sync_data_to_es():
    """同步 law 与 writ，默认先 law 再 writ"""
    sync_law_to_es()
    sync_writ_to_es()

if __name__ == "__main__":
    sync_data_to_es()