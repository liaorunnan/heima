from redis_om import HashModel, get_redis_connection, Field, Migrator

from conf import settings

# 设置带认证信息的 Redis 连接
redis_conn = get_redis_connection(
    url=f"redis://default:{settings.redis_password}@localhost:6379"
)

# 确保模型使用该连接
HashModel.Meta.database = redis_conn


class QA(HashModel):
    query: str = Field(index=True)
    answer: str

    class Meta:
        database = redis_conn


Migrator().run()

if __name__ == '__main__':
    Migrator().run()
    andrew = QA(query="Andrew", answer="Brookins")
    print(andrew.pk)
    andrew.save()
    # andrew.expire(120)
    # assert QA.get(andrew.pk) == andrew
    results = QA.find(QA.query == "Andrew").all()
    print(results)
