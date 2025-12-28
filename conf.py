import os
from typing import List

from pydantic import ConfigDict
from pydantic_settings import BaseSettings

PATH=os.path.dirname(os.path.abspath(__file__))
# dirname：在拿上级目录
# __file__：当前文件
# os.path.abspath：当前绝对路径


class Settings(BaseSettings):
    api_key:str
    model_name:str
    base_url:str
    # url:str

    es_host: List[str] #???
    es_user : str
    es_password : str


    redis_name : str
    redis_password : str
    redis_port : int



    milvus_host: str
    milvus_port: str

    milvus_user : str
    milvus_password : str


    model_config = ConfigDict(
        extra="allow",
        env_file=f'{PATH}/.env',
        case_senstive=False,
    ) # 从env中映射到settings中的api_key这些文件

    @property
    def url(self):
        mysql_url = f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return mysql_url


settings = Settings()

if __name__ == '__main__':
    print(settings.base_url)