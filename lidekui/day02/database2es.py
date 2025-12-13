from sqlmodel import Field,Session,SQLModel,create_engine,select
from conf import settings
from typing import Optional
from elasticsearch_dsl import Document, Text, connections
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
engine = create_engine(settings.project_database_url)
connections.create_connection(hosts=settings.es_host,
                              verify_certs=False,
                              ssl_assert_hostname=False,
                              http_auth=(settings.es_user, settings.es_password),
                              )


class law_chunks(SQLModel, table=True):
    __tablename__ = "law_chunks"
    id: Optional[int] = Field(default=None, primary_key=True)
    law_title: str
    embedding_text: str

    @classmethod
    def query(cls, title):
        with Session(engine) as session:
            statement = select(cls).where(law_chunks.law_title == title)
            law = session.exec(statement).first()
        return law

    @classmethod
    def query_all(cls):
        with Session(engine) as session:
            statement = select(cls)
            laws = session.exec(statement).all()
        return laws


class Law(Document):
    # 定义索引名称
    class Index:
        name = 'laws'  # Elasticsearch 索引名称
        settings = {
            'number_of_shards': 1,
            'number_of_replicas': 0
        }

    # 定义文档字段
    law_title = Text(analyzer='ik_max_word', search_analyzer='ik_smart')
    embedding_text = Text(analyzer='ik_max_word', search_analyzer='ik_smart')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def law2es():
    laws = law_chunks.query_all()
    for law in tqdm(laws):
        law = Law(law_title=law.law_title, embedding_text=law.embedding_text)
        law.save()


if __name__ == '__main__':
    SQLModel.metadata.create_all(engine)
    law2es()
