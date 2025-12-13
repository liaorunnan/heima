from redis_om import HashModel, get_redis_connection, Field, Migrator

from config import settings

redis = get_redis_connection(
    host=settings.RE_HOST,
    port=settings.RE_PORT,
    password=settings.RE_PASSWORD,
    decode_responses=True
)


class QA(HashModel):
    query: str = Field(index=True)
    answer: str

    class Meta:
        database = redis


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
