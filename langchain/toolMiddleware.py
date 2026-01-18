from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware,ModelRetryMiddleware,LLMToolEmulator
from langchain.tools import  tool
from langchain_openai import ChatOpenAI
import time

from conf import settings

model = ChatOpenAI(temperature=0.1, model_name=settings.qw_model, api_key=settings.qw_api_key, base_url=settings.qw_api_url)# 超时时间1秒，在工具中沉睡2秒，模拟工具调用耗时，这样子设置可以让系统认为工具调用超时，从而触发重试机制



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
# agent = create_agent(
#     model=model,#可以设置为国内访问不到的模型
#     tools=[get_user_info],
#     middleware=[
#         ModelRetryMiddleware(
#             max_retries=3,
#             backoff_factor=2.0,
#             initial_delay=1.0,
#         ),
#     ],
# )

# message = agent.invoke(
#     {"messages": [{"role": "user", "content": "你好,我的名字是什么"}]},
# )
# print(message['messages'][-1].content) # 包含第一次一共四次调用了
# exit()


model2 = ChatOpenAI(temperature=0.1, model_name=settings.qw_model, api_key=settings.qw_api_key, base_url=settings.qw_api_url)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"



# 模拟调用，并不是正式调用
agent2 = create_agent(
    model=model2,
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model=model2, tools=["get_weather"])],
)

# Use custom model for emulation
# agent4 = create_agent(
#     model=model2,
#     tools=[get_weather, send_email],
#     middleware=[LLMToolEmulator(model="claude-sonnet-4-5-20250929")],
# )

result = agent2.invoke({"messages": [{"role": "user", "content": "今天天气怎么样"}]})
print(result['messages'][-1].content)
