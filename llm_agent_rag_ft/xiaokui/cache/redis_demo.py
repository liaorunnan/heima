import datetime
from typing import Optional

from pydantic import EmailStr

from redis_om import HashModel, get_redis_connection

from conf import settings

redis = get_redis_connection(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    decode_responses=True
)


class Customer(HashModel):
    first_name: str
    last_name: str
    email: EmailStr
    join_date: datetime.date
    age: int
    bio: Optional[str] = None

    class Meta:
        database = redis


andrew = Customer(
    first_name="Andrew",
    last_name="Brookins",
    email="andrew.brookins@example.com",
    join_date=datetime.date.today(),
    age=38,
    bio="Python developer, works at Redis, Inc."
)

print(andrew.pk)
andrew.save()
andrew.expire(120)
cus = Customer.get(andrew.pk)
print(cus.first_name)
