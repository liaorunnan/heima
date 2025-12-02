from neo4j import GraphDatabase
from conf import settings
from tqdm import tqdm
import pandas as pd



class KnowledgeGraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def batch_insert(self, triples, batch_size=10000):
        """
        使用 UNWIND 进行批量导入
        triples: [(sub, pred, obj), ...]
        """
        with self.driver.session() as session:
            # 分批处理，防止内存溢出
            for i in tqdm(range(0, len(triples), batch_size)):
                batch = triples[i: i + batch_size]
                # 将元组转换为字典列表，方便 Cypher 处理
                batch_data = [{"s": s, "p": p, "o": o} for s, p, o in batch]

                session.execute_write(self._create_nodes_and_relationships, batch_data)
                print(f"已处理批次: {i} - {i + len(batch)}")

    @staticmethod
    def _create_nodes_and_relationships(tx, batch_data):


        # 方案 1：假设已经安装 APOC 插件 (推荐)
        query_apoc = """
        UNWIND $batch AS row
        MERGE (s:Entity {name: row.s})
        MERGE (o:Entity {name: row.o})
        WITH s, o, row
        CALL apoc.create.relationship(s, row.p, {}, o) YIELD rel
        RETURN count(*)
        """

        # 方案 2：纯 Cypher (仅当关系类型是有限的几种时，或者使用通用关系名)
        # 这里演示将关系类型作为属性存储，这样最灵活，但可视化稍弱
        # query_pure = """
        # UNWIND $batch AS row
        # MERGE (s:Entity {name: row.s})
        # MERGE (o:Entity {name: row.o})
        # MERGE (s)-[:RELATION {type: row.p}]->(o)
        # """

        tx.run(query_apoc, batch=batch_data)


# --- 使用示例 ---
if __name__ == "__main__":
    # 配置 Neo4j 连接
    importer = KnowledgeGraphImporter("bolt://localhost:"+settings.neo4j_port, settings.neo4j_username, settings.neo4j_password)

    file = pd.read_csv('./data/test.csv', header=None,  encoding='utf-8')
    print(file.head())
    data_array = []
    for row in tqdm(file.itertuples(),index=):
        print(row[''])
        tmp_tuple = tuple(array[:3])
        data_array.append(tmp_tuple)
    exit()


    with open('./data/Disease.csv', 'r', encoding='utf-8') as file:

        data = file.readlines()

    data_array = []

    for row in data[62000:]:
        # print(row)
        array = row.strip().split(",")
        if len(array) >= 3:
            # 只取前三个元素：parts[0], parts[1], parts[2]
            # parts[:3] 会截取列表的前3个
            tmp_tuple = tuple(array[:3])
            data_array.append(tmp_tuple)





    try:
        importer.batch_insert(data_array)
        print("Neo4j 导入完成")
    finally:
        importer.close()