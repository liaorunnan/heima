from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from conf import settings



@tool
def get_name():
    """
    获取用户姓名
    """
    print("调用了get_name工具")
    return "张三"

@tool
def get_age():
    """
    获取用户年龄
    """
    print("调用了get_age工具")
    return 30

@tool
def get_gender():
    """
    获取用户性别
    """
    print("调用了get_gender工具")
    return "男"


tools = [get_name, get_age, get_gender]

model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)

agent = create_agent(model=model, tools=tools,system_prompt="你是一个智能助手，能够根据用户需求调用工具完成任务")

message = {
    "messages": [
        {"role": "user", "content": "请帮我获取用户姓名、年龄"}
    ]
}
result = agent.invoke(message)

print(result)

print(result['messages'][-1].content)

