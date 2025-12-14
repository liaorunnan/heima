import numpy as np  # 导入 numpy 库，用于处理向量（embedding 通常是 numpy array）

from pymilvus import MilvusClient, DataType  # 从 pymilvus 导入新版客户端和数据类型常量

from b_rag.day02.items import  LawsItem  # 导入自定义的数据类 BookItem，用于结构化返回搜索结果
from conf import settings  # 从项目配置中导入 settings，里面存放 Milvus 的 host、port、user、password 等信息

dimension = 1024  # 定义向量维度为 1024（对应 BGE-large、BGE-m3 等模型的输出维度）

search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # 定义搜索参数：使用内积（IP）作为相似度度量，搜索时探查 10 个聚类单元（nprobe 越大越准但越慢）

index_params = MilvusClient.prepare_index_params()  # 准备索引参数对象
index_params.add_index(  # 为 vec 字段添加索引
    field_name="vec",  # 要建立索引的字段名（向量字段）
    index_type="IVF_FLAT",  # 索引类型：倒排文件 + 平坦（精确搜索，不压缩）
    index_name="inverted_index",  # 索引名称（自定义）
    metric_type="IP",  # 相似度度量方式：内积（BGE 系列推荐）
    params={"nlist": 128},  # IVF 参数：聚类中心数量为 128（值越大索引越细，内存占用越大）
)

class Singleton(type):  # 定义一个元类，用于实现单例模式
    _instances = {}  # 类变量，保存已经创建的实例

    def __call__(cls, name):  # 当类被调用时执行（即 VecIndex("children") 时）
        k = name  # 用 collection_name 作为 key
        if k not in cls._instances:  # 如果这个 collection_name 还没创建过实例
            cls._instances[k] = super(Singleton, cls).__call__(name)  # 创建新实例
        return cls._instances[k]  # 返回已存在的实例（保证单例）


class VecIndex(metaclass=Singleton):  # 主类，使用上面定义的 Singleton 元类，确保同一个 collection_name 只创建一个实例
    def __init__(self, collection_name):  # 构造函数，参数是集合名称
        self.client = MilvusClient(  # 创建 Milvus 客户端连接
            uri=f'http://{settings.milvus_host}:{settings.milvus_port}',  # Milvus 服务地址（http 模式）
            token=f"{settings.milvus_user}:{settings.milvus_password}"  # 认证 token，格式为 username:password
        )
        schema = self.client.create_schema()

        # 1. 主键（必须唯一，推荐用 ES 的 _id 或自定义 UUID）
        schema.add_field(field_name="es_id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)

        # 2. 向量字段（核心！存储 embedding_text 的向量）
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=1024)  # dim 根据你的模型调整，如 BGE-m3 是 1024

        # 3. 存储实际用于生成 embedding 的文本（embedding_text）
        schema.add_field(field_name="embedding_text", datatype=DataType.VARCHAR, max_length=65535)
        # 5. 元数据字段（用于过滤、展示、溯源）
        schema.add_field(field_name="document_id", datatype=DataType.INT64)  # 文档 ID
        schema.add_field(field_name="law_filename", datatype=DataType.VARCHAR, max_length=512)  # 如：人民检察院公益诉讼办案规则_2023
        schema.add_field(field_name="law_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="article_id", datatype=DataType.VARCHAR, max_length=512)  # 第X条
        schema.add_field(field_name="part_name", datatype=DataType.VARCHAR, max_length=512)  # 编（可选）
        schema.add_field(field_name="chapter_name", datatype=DataType.VARCHAR, max_length=512)  # 章（可选）
        schema.add_field(field_name="section_name", datatype=DataType.VARCHAR, max_length=512)  # 节（可选）


        self.collection_name = collection_name  # 保存当前集合名称
        if not self.client.has_collection(self.collection_name):  # 检查该集合是否已存在
            self.client.create_collection(collection_name=self.collection_name,  # 如果不存在，则创建集合
                                          schema=schema,  # 使用上面定义的 schema
                                          index_params=index_params)  # 并应用前面准备好的索引参数（IVF_FLAT）

    def insert(self, embeddings, items, ids):
        """
        批量插入法律 chunk 数据到 Milvus

        参数:
            embeddings: List[np.ndarray] 或 List[list]，每个 chunk 的向量（已转为 list）
            items: List[LawsItem]，法律条目对象列表（包含所有元数据和文本）
            ids: List[str]，每个 chunk 的唯一 ID（推荐用 ES 的 _id 或数据库主键）
        """
        data = []  # 准备要插入的数据列表（Milvus upsert 格式）

        for i, item in enumerate(items):  # 遍历每个 LawsItem 对象
            data.append({  # 构造一条记录
                "es_id": ids[i],  # 主键（ES _id 或数据库 ID，字符串）
                "vec": embeddings[i],  # 向量（必须是 list 格式，长度=1024）

                # 用于生成向量的原始 chunk 文本（调试/重算向量时有用）
                "embedding_text": item.embedding_text ,
                # 元数据字段（用于过滤、展示、溯源）
                "document_id": item.document_id,  # 整篇文档 ID
                "law_filename": item.law_filename,  # 如：人民检察院公益诉讼办案规则_2023
                "law_title": item.law_title,  # 法律标题
                "article_id": item.article_id,  # 第X条（核心定位）
                "part_name": item.part_name or "",  # 编（可选）
                "chapter_name": item.chapter_name or "",  # 章（可选）
                "section_name": item.section_name or "",  # 节（可选）
            })

        if data:  # 如果有数据
            self.client.upsert(collection_name=self.collection_name, data=data)  # 批量 upsert（存在则更新）
            print(f"成功插入/更新 {len(data)} 条法律 chunk 到 Milvus 集合 '{self.collection_name}'")

    def search(self, vec, topk=3):  # 向量搜索函数
        vec = np.expand_dims(vec, axis=0)  # 将一维向量扩展为二维（batch=1），Milvus search 要求这样
        hits = self.client.search(collection_name=self.collection_name,  # 执行搜索
                                  data=vec.tolist(),  # 查询向量（转为 list）
                                  anns_field="vec",   # 在 vec 字段上搜索
                                  limit=topk,         # 返回前 topk 个结果
                                  search_params=search_params,  # 使用前面定义的搜索参数（IP + nprobe）
                                  output_fields=["es_id", "embedding_text", "document_id","law_filename", "law_title", "article_id", "part_name", "chapter_name", "section_name"])[0]  # 返回指定的输出字段；[0] 是因为返回的是 list of list（batch）
        # 将搜索结果转换为 BookItem 对象列表，便于后续使用
        return [LawsItem(id=hit.entity.get("es_id"), embedding_text=hit.entity.get("embedding_text"), document_id=hit.entity.get("document_id"), law_filename=hit.entity.get("law_filename"), law_title=hit.entity.get("law_title"), article_id=hit.entity.get("article_id"), part_name=hit.entity.get("part_name"), chapter_name=hit.entity.get("chapter_name"), section_name=hit.entity.get("section_name"), score=hit.distance) for hit in hits]

    def load(self):  # 批量导入数据的函数（把所有书籍数据向量化后插入 Milvus）
        from b_rag.day02.match_keyword import Law  # 导入 Book 类（可能是 Scrapy Item 或数据库模型）
        from b_rag.day03.embedding import get_embedding  # 导入获取 embedding 的函数（调用 BGE 等模型）
        # 用于批量处理的临时列表
        batch_size = 500  # 每 100 条插入一次（可根据内存和速度调整，100~500 合适）
        embeddings_batch = []
        items_batch = []
        ids_batch = []
        print("开始从 Elasticsearch 加载法律 chunk 并向量化插入 Milvus...")
        for item in Law.scan():  # 遍历所有 Book 数据（scan 可能是迭代器，类似 yield 所有条目）
            embed = get_embedding(item.embedding_text)  # 对子块文本（item.child）生成向量 embedding
            embeddings_batch.append(embed.tolist())  # Milvus 需要 list
            # 2. 保存 LawsItem 对象（后续 insert 需要所有字段）
            # 这里直接用 hit 转 dict 再构造 LawsItem，或直接传 hit（取决于你的 LawsItem 定义）
            laws_item = LawsItem(
                id=item.meta.id,
                document_id=getattr(item, 'document_id', None),
                law_filename=item.law_filename,
                law_title=getattr(item, 'law_title', ''),
                part_name=getattr(item, 'part_name', None),
                chapter_name=getattr(item, 'chapter_name', None),
                section_name=getattr(item, 'section_name', None),
                article_id=item.article_id,
                embedding_text=item.embedding_text,
            )
            items_batch.append(laws_item)
            # 3. 主键 ID（ES 的 _id，必须是字符串）
            ids_batch.append(str(item.meta.id))
            # 4. 达到 batch_size 或最后一批时插入
            if len(embeddings_batch) >= batch_size:
                VecIndex("laws").insert(embeddings_batch, items_batch, ids_batch)
                print(f"已插入 {len(embeddings_batch)} 条，共累计 {len(ids_batch)} 条")
                # 清空 batch
                embeddings_batch = []
                items_batch = []
                ids_batch = []
                # 打印进度（可选）
                # 处理剩余的数据（最后一批）
        if embeddings_batch:
            VecIndex("laws").insert(embeddings_batch, items_batch, ids_batch)
            print(f"最后一批插入 {len(embeddings_batch)} 条，总完成！")

        print("所有法律 chunk 已成功向量化并插入 Milvus！")


if __name__ == '__main__':  # 当直接运行这个文件时执行下面的代码
    # 下面的代码是测试示例，已被注释掉
    # from b_rag.day03.embedding import get_embedding
    #
    # vec = get_embedding("猴子")
    # hits = VecIndex("children").search(vec)
    # for hit in hits:
    #     print(hit.parent)
    # print(f"ID: {hit.entity.get('es_id')}, Distance: {hit.distance}, Parent: {hit.entity.get('parent')}")
    # vec2 = get_embedding("这是一个测试2")
    # VecIndex("children").insert([vec, vec2], ["测试1", "测试2"])

    VecIndex("laws").load()