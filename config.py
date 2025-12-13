from typing import List

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

import os.path

PATH = os.path.dirname(os.path.abspath(__file__))


# __file__ : 当前文件
# os.path.abspath : 当前文件的绝对路径
# dirname : 上级路径

class Settings(BaseSettings):
    # LLM相关配置
    API_KEY: str
    MODEL_NAME: str
    BASE_URL: str
    # 数据库的配置文件
    USENAME: str
    PASSWORD: str
    HOST: str
    PORT: str
    DATABASE: str
    # ES 的配置文件
    ES_HOST: List[str]
    ES_USER: str
    ES_PASSWORD: str
    # REDIS的配置文件
    RE_PASSWORD: str
    RE_PORT: str
    RE_HOST: str
    # Milvus 的配置文件
    MILVUS_USER: str
    MILVUS_HOST: str
    MILVUS_PORT: str
    MILVUS_PASSWORD: str

    model_config = ConfigDict(
        extra='allow',
        env_file=f'{PATH}/.env',
        case_sensitive=False,
    )

    @property
    def url(self):
        mysql_url = f"mysql+pymysql://{self.USENAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}?charset=utf8mb4"
        return mysql_url


settings = Settings()

if __name__ == '__main__':
    # print(settings.base_url)
    # print(settings.model_name)
    # print(settings.database)
    print(settings.USENAME)
    print(settings.url)
