from neo4j import GraphDatabase
import re
from conf import settings

class RuleBasedBot:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_neo4j(self, cypher):
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]

    def answer(self, question):
        # 规则 1：询问某人的所有关系 (例如：Elon Musk 有什么关系？)
        # 正则逻辑：匹配 "XX 有什么关系" 或 "XX 的关系"
        pattern1 = re.compile(r"(.*)有什么关系|intro (.*)")
        match1 = pattern1.search(question)
        

        if match1:
            name = match1.group(1) or match1.group(2)
            name = name.strip()

            # 对应的 Cypher 模板
            sql = f"MATCH (n:Entity)-[r]->(m) WHERE n.name CONTAINS '{name}' RETURN type(r) as relation, m.name as target LIMIT 10"
            data = self.query_neo4j(sql)



            if not data:
                return f"抱歉，没有找到关于 {name} 的信息。"

            # 格式化输出
            reply = f"关于 {name} 的信息如下：\n"
            for item in data:
                reply += f"- {item['relation']}: {item['target']}\n"
            return reply

        # 规则 2：询问两个实体之间的关系 (例如：Apple 和 Samsung 是什么关系？)
        pattern2 = re.compile(r"(.*)和(.*)是什么关系")
        match2 = pattern2.search(question)

        if match2:
            entity_a = match2.group(1).strip()
            entity_b = match2.group(2).strip()

            sql = f"""
            MATCH (a:Entity), (b:Entity)
            WHERE a.name CONTAINS '{entity_a}' AND b.name CONTAINS '{entity_b}'
            MATCH p=shortestPath((a)-[*]-(b))
            RETURN p
            """
            # 注意：这里处理 Path 结果比较复杂，简单起见我们只查直接关系
            simple_sql = f"""
            MATCH (a:Entity)-[r]-(b:Entity)
            WHERE a.name CONTAINS '{entity_a}' AND b.name CONTAINS '{entity_b}'
            RETURN type(r) as rel
            """
            data = self.query_neo4j(simple_sql)
            if data:
                return f"{entity_a} 和 {entity_b} 的关系是: {data[0]['rel']}"
            else:
                return "暂未发现他们之间有直接关系。"

        return "抱歉，我不理解这个问题。请尝试问：'XX有什么关系' 或 'A和B是什么关系'"


# --- 测试 ---
if __name__ == "__main__":
    bot = RuleBasedBot("bolt://localhost:" + settings.neo4j_port, settings.neo4j_username,
                                      settings.neo4j_password)

    print(bot.answer("百日咳[疾病]和儿科是什么关系"))
