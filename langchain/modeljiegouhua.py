from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import json
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()

from conf import settings
model = ChatOpenAI(temperature=0.7, model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="电影上映年份")
    director: str = Field(..., description="电影导演")
    rating: float = Field(..., description="豆瓣评分")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception",config={"callbacks": [callback]})
print(type(response)) 
print(response.title)

# 将 Movie 实例转换为 JSON 字符串
response_json = response.model_dump_json(indent=2)  # indent=2 使 JSON 格式化更易读

# 打印 JSON
print(response_json)
print(callback.usage_metadata)
