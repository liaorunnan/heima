from conf import settings
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model=settings.model_name,
                   openai_api_base=settings.base_url,
                   openai_api_key=settings.api_key)