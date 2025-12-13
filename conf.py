import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import List


PATH = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    api_key: str
    model_name: str
    base_url: str
    user: str
    password: str
    host: str
    port: str
    database: str
    es_host: List[str]
    # es_port: str
    es_user: str
    es_password: str

    project_host: str
    project_port: str
    project_user: str
    project_password: str
    project_database: str

    milvus_host: str
    milvus_port: str
    milvus_user: str
    milvus_password: str

    vllm_index_url: str
    vllm_rerank_url: str

    redis_host: str
    redis_port: str
    redis_password: str

    model_config = ConfigDict(
        extra='allow',
        env_file=f'{PATH}/.env',
        case_sensitive=False
    )

    @property
    def database_url(self):
        return f'mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'

    @property
    def project_database_url(self):
        return f'mysql+pymysql://{self.project_user}:{self.project_password}@{self.project_host}:{self.project_port}/{self.project_database}'


settings = Settings()
if __name__ == '__main__':
    print(settings.model_name)
