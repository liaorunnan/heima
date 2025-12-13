from redis_om import HashModel, get_redis_connection, Field, Migrator

from conf import settings

redis = get_redis_connection(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    decode_responses=True
)


class QA(HashModel):
    query: str = Field(index=True)
    answer: str

    class Meta:
        database = redis

def get_redis_answer(query):
    qa = QA.find(QA.query == query).all()
    if qa:
        return qa[0].answer
    else:
        return None

def set_redis_answer(query, answer):
    qa = QA(query=query, answer=answer)
    qa.save()

def run_migrate():
    Migrator().run()

if __name__ == '__main__':
    # Migrator().run()
    # andrew = QA(query="Andrew", answer="Brookins")
    # print(andrew.pk)
    # andrew.save()
    # # andrew.expire(120)
    # # assert QA.get(andrew.pk) == andrew
    # results = QA.find(QA.query == "Andrew").first()
    # print(results.answer)
    print(get_redis_answer("Andrew"))
