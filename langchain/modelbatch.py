from langchain_openai import ChatOpenAI
from langchain.messages import AIMessage,SystemMessage,HumanMessage
from conf import settings


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent




model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)
responses = model.batch([
    "为什么会有彩色的羽毛？",
    "飞机为什么会飞？",
    "什么是绿色的？"
])
for response in responses:
    print(response)
