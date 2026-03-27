from sqlmodel import Field, Session, SQLModel, create_engine,select
from conf import settings

engine=create_engine(settings.url)

class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    secret_name: str
    age: int | None = None

    @classmethod
    def query(self,name):
        with Session(engine) as session:
            statement = select(self).where(self.name == name)
            hero = session.exec(statement).first()
        return hero


    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()
        return self

    @classmethod
    def query_all(self):
        with Session(engine) as session:
            statement = select(self)
            heroes = session.exec(statement).all()
        return heroes

if __name__ == '__main__':
    SQLModel.metadata.create_all(engine)

    Hero(name="Deadpond", secret_name="Dive Wilson").insert()

    hero = Hero.query("Deadpond")
    print(hero)
    heros = Hero.query_all()
    print(heros)

