import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import List
PATH = os.path.dirname(os.path.abspath(__file__))

class Settings(BaseSettings):
    api_key:str
    model_name:str
    base_url:str

    # mysql_user: str
    # mysql_password: str
    # mysql_database: str
    # mysql_host: str
    # mysql_port: int

    es_host:List[str]
    es_user:str
    es_password:str

    milvus_host:str
    milvus_port:str
    milvus_user:str
    milvus_password:str

    redis_host:str
    redis_port:str
    redis_password:str

    model_config = ConfigDict(
        extra='allow',
        env_file=f'{PATH}/.env',
        case_sensitive=False
    )

    @property
    def url(self):
        mysql_url=f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return mysql_url

settings = Settings()

if __name__ == '__main__':
    print(settings.model_name)