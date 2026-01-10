import os
import signal
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import List



PATH = os.path.dirname(os.path.abspath(__file__))


class Setting(BaseSettings):

    api_key: str
    model_name: str
    base_url: str

    # 数据库配置
    qw_model: str
    qw_api_key: str
    qw_api_url: str

    autodl_qw_model: str
    autodl_qw_api_key: str
    autodl_qw_api_url: str
    

    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

    es_host: List[str]
    es_user: str
    es_password: str


    milvus_host: str
    milvus_port: int
    milvus_user: str
    milvus_password: str

    Emb_url: str
    Emb_token: str
    Rank_url: str
    Mu_Rank_url: str

    redis_host: str
    redis_password: str
    redis_host: str
    
    SERPAPI_API_KEY: str
    OPENAI_API_KEY: str

    # 微调模型配置
    use_local_model: bool = False
    local_model_path: str = "./finetuned_model"
    local_model_device: str = "cpu"
    local_model_max_length: int = 512


    refin_emb_url: str
    signal_emb_url: str

    model_config = ConfigDict(
        extra='allow',
        env_file=f'{PATH}/.env',
        case_sensitive=False,
    )

    def url(self):
        mysql_url = f'mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        return mysql_url

settings = Setting()

if __name__ == '__main__':
    print(settings.model_name)