
"""
LangGraph React智能体记忆管理与多轮对话方法

"""

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import os
import requests
import json

from dotenv import load_dotenv 
load_dotenv(override=True)

class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")

@tool(args_schema = WeatherQuery)
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": os.getenv("OPENWEATHER_API_KEY"),    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)



checkpointer = InMemorySaver()

tools = [get_weather]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  
agent = create_react_agent(model=model, 
                           tools=tools,
                           checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1"     # thread 翻译为 线程
    }
}


response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫陈明，好久不见！"}]},
    config
)


print("=================================================    1    ===============================================================")
print(response["messages"])
"""
response["messages"] 的结构：
[
HumanMessage(content='你好，我叫陈明，好久不见！', additional_kwargs={}, response_metadata={}, id='1948bbe1-4cd8-47f0-a918-6aec42432342'), 
AIMessage(content='你好陈明！很高兴再次见到你！有什么我可以帮助你的吗？比如查询天气信息或者其他问题？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 267, 'total_tokens': 289, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 11}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '981a7dcd-e98e-4343-a76e-9af881ce44e5', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--d80962b9-b25a-4df9-a0ed-1681bb9231b0-0', usage_metadata={'input_tokens': 267, 'output_tokens': 22, 'total_tokens': 289, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})
]
"""

print("=================================================    2    ===============================================================")
print(response["messages"][-1].content)


# 此时记忆就自动保存在当前Agent和线程中，如下方式查看
latest = agent.get_state(config)
print("=================================================    3    ===============================================================")
print(latest)
"""
StateSnapshot(
                values={
                        'messages': [
                                        HumanMessage(content='你好，我叫陈明，好久不见！', additional_kwargs={}, response_metadata={}, id='3499ad34-6b84-46d0-a505-e9fe30c82593'), 
                                        AIMessage(content='你好陈明！很高兴再次见到你！有什么我可以帮助你的吗？比如查询天气信息或者其他问题？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 267, 'total_tokens': 289, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 11}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '56287ccc-e5fa-42ff-9a07-88675aa54e39', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--735d86c1-0b42-438b-a765-fbbc4bedc95a-0', usage_metadata={'input_tokens': 267, 'output_tokens': 22, 'total_tokens': 289, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})
                                    ]
                        }, 
                next=(), 
                config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f096b1a-729a-625f-8001-3f72ecd7a48a'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2025-09-21T06:10:20.638678+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f096b1a-4899-686a-8000-67c8390b4101'}}, tasks=(), interrupts=()
                
             )
"""







# 当再次进行对话时，直接带入线程ID，即可带入此前对话记忆：
response = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请问你还记得我叫什么名字么？"}]},
    config
)

print("=================================================    4    ===============================================================")
print(response["messages"])
"""
response["messages"] 的结构：
[
HumanMessage(content='你好，我叫陈明，好久不见！', additional_kwargs={}, response_metadata={}, id='f4609b18-408a-4bf6-968d-30afd9b0dcc6'), 
AIMessage(content='你好陈明！很高兴再次见到你！有什么我可以帮助你的吗？比如查询天气信息或者其他问题？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 267, 'total_tokens': 289, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 11}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'e5860e4c-03f1-4e02-ae9e-a1924e7296a4', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--d5f5b3d4-32bf-461f-99f3-690f9293cabd-0', usage_metadata={'input_tokens': 267, 'output_tokens': 22, 'total_tokens': 289, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}}), 
HumanMessage(content='你好，请问你还记得我叫什么名字么？', additional_kwargs={}, response_metadata={}, id='92360dee-fd3c-4e5f-97e4-bcf652243245'), 
AIMessage(content='当然记得！你刚才告诉我你叫陈明。很高兴再次和你交流！有什么需要我帮助的吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 303, 'total_tokens': 325, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 47}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '9a884053-abc0-4d63-b860-dd8e072cb1bc', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--2e8b44f4-0f06-4b44-84f8-7b88587b69b7-0', usage_metadata={'input_tokens': 303, 'output_tokens': 22, 'total_tokens': 325, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})
]
"""

print("=================================================    5    ===============================================================")
print(response["messages"][-1].content)

# 查看记忆
latest = agent.get_state(config)
print("=================================================    6    ===============================================================")
print(latest)
"""
StateSnapshot(
                values={
                        'messages': [
                                        HumanMessage(content='你好，我叫陈明，好久不见！', additional_kwargs={}, response_metadata={}, id='f4609b18-408a-4bf6-968d-30afd9b0dcc6'), 
                                        AIMessage(content='你好陈明！很高兴再次见到你！有什么我可以帮助你的吗？比如查询天气信息或者其他问题？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 267, 'total_tokens': 289, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 11}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'e5860e4c-03f1-4e02-ae9e-a1924e7296a4', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--d5f5b3d4-32bf-461f-99f3-690f9293cabd-0', usage_metadata={'input_tokens': 267, 'output_tokens': 22, 'total_tokens': 289, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}}), 
                                        HumanMessage(content='你好，请问你还记得我叫什么名字么？', additional_kwargs={}, response_metadata={}, id='92360dee-fd3c-4e5f-97e4-bcf652243245'), 
                                        AIMessage(content='当然记得！你刚才告诉我你叫陈明。很高兴再次和你交流！有什么需要我帮助的吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 303, 'total_tokens': 325, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 47}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '9a884053-abc0-4d63-b860-dd8e072cb1bc', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--2e8b44f4-0f06-4b44-84f8-7b88587b69b7-0', usage_metadata={'input_tokens': 303, 'output_tokens': 22, 'total_tokens': 325, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})
                                    ]
                        }, 
                next=(), 
                config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f096c37-645f-668b-8004-9eebf683a17d'}}, metadata={'source': 'loop', 'step': 4, 'parents': {}}, created_at='2025-09-21T08:17:49.557108+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f096c37-3d48-6ff6-8003-f8c35c1fba60'}}, tasks=(), interrupts=())
"""






# 如果更新线程ID，则会重新开启对话：
config2 = {
    "configurable": {
        "thread_id": "2"  
    }
}
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，你还记得我叫什么名字么？"}]},
    config2
)
print("=================================================    7    ===============================================================")
print(response2["messages"])

print("=================================================    8    ===============================================================")
print(response2["messages"][-1].content)

print("=================================================    9    ===============================================================")
latest2 = agent.get_state(config2)
print(latest2)












