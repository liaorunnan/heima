from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import  tool
from langchain_openai import ChatOpenAI
from conf import settings

model = ChatOpenAI(temperature=0.1, model_name=settings.qw_model, api_key=settings.qw_api_key, base_url=settings.qw_api_url,timeout=1)# 超时时间1秒，在工具中沉睡2秒，模拟工具调用耗时，这样子设置可以让系统认为工具调用超时，从而触发重试机制



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

# 工具重试
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

# 模型重试
agent = create_agent(
    model=model,#可以设置为国内访问不到的模型
    tools=[get_user_info],
    middleware=[
        ModelRetryMiddleware(
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