from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import  tool
from langchain_openai import ChatOpenAI
from conf import settings

model = ChatOpenAI(temperature=0.1, model_name=settings.qw_model, api_key=settings.qw_api_key, base_url=settings.qw_api_url,timeout=1)



@tool
def get_user_info():
    """
    获取用户信息
    """
    print("获取用户信息")
    time.sleep(2)
    print("获取用户信息完成")
    return {
        "name": "李四",
    }

agent = create_agent(
    model=model,
    tools=[get_user_info],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

message = agent.invoke(
    {"messages": [{"role": "user", "content": "你好,我的名字是什么"}]},
)
print(message['messages'][-1].content) # 包含第一次一共四次调用了