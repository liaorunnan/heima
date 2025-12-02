import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

PATH = os.path.dirname(os.path.abspath(__file__))


class Setting(BaseSettings):

    Host: str
    Port: str
    db_username: str
    db_password: str
    db_database: str
    
    MODEL_NAME: str
    API_KEY: str
    BASE_URL: str

    model_config = ConfigDict(
        extra='allow',
        env_file=f'{PATH}/.env',
        case_sensitive=False,
    )

settings = Setting()

if __name__ == '__main__':
    print(settings.db_username)