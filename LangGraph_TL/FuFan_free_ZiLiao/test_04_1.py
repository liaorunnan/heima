
"""
预构建智能体 API 就是 create_react_agent 的使用

并联  调用：借助LangGraph React Agent，无需任何额外设置，即可完成多工具串联和并联调用。

"""


import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv 
load_dotenv(override=True)
import requests,json

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





from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# 封装外部函数列表
tools = [get_weather]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  
agent = create_react_agent(model=model, tools=tools)

response = agent.invoke({"messages": [{"role": "user", "content": "请问北京和杭州今天哪里更热？"}]})

print("=================================================    1    ===============================================================")
print(response)

print("=================================================    2    ===============================================================")
print(response["messages"])


print("=================================================    3    ===============================================================")
print(response["messages"][-1].content)

print("=================================================    4    ===============================================================")
print(response["messages"][0])  # 用户输入 HumanMessage
print(response["messages"][1])  # 模型输出 AIMessage （ Function calling 的结果 ）
print(response["messages"][2])  # 工具输出 ToolMessage
print(response["messages"][3])  # 模型输出 AIMessage


print("=================================================    8    ===============================================================")



"""
response["messages"] 的结构：


[
HumanMessage(content='请问北京和杭州今天哪里更热？', additional_kwargs={}, response_metadata={}, id='78a828e4-c6ef-4722-a21d-0190cfbf21b0'), 
AIMessage(content='我来帮您查询北京和杭州的天气情况，然后比较哪里更热。', additional_kwargs={'tool_calls': [{'id': 'call_00_f262tZaRS6WwlIjt3co9jO2U', 'function': {'arguments': '{"loc": "Beijing"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0}, {'id': 'call_01_7Sa3enQEBVhLjlvMGxIdqhtT', 'function': {'arguments': '{"loc": "Hangzhou"}', 'name': 'get_weather'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 267, 'total_tokens': 312, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 11}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '3dc4d983-4184-441d-b7fa-beae31ce1d40', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--1ca278e5-bced-4607-8d8d-601c801d2796-0', tool_calls=[{'name': 'get_weather', 'args': {'loc': 'Beijing'}, 'id': 'call_00_f262tZaRS6WwlIjt3co9jO2U', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'loc': 'Hangzhou'}, 'id': 'call_01_7Sa3enQEBVhLjlvMGxIdqhtT', 'type': 'tool_call'}], usage_metadata={'input_tokens': 267, 'output_tokens': 45, 'total_tokens': 312, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}}), 
ToolMessage(content='{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 803, "main": "Clouds", "description": "\\u591a\\u4e91", "icon": "04d"}], "base": "stations", "main": {"temp": 26.64, "feels_like": 26.64, "temp_min": 26.64, "temp_max": 26.64, "pressure": 1020, "humidity": 43, "sea_level": 1020, "grnd_level": 1014}, "visibility": 10000, "wind": {"speed": 2.15, "deg": 150, "gust": 3.03}, "clouds": {"all": 76}, "dt": 1758429943, "sys": {"country": "CN", "sunrise": 1758405645, "sunset": 1758449671}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}', name='get_weather', id='94b7e090-6c96-4bc3-b3dd-33c96b0ef2d9', tool_call_id='call_00_f262tZaRS6WwlIjt3co9jO2U'), 
ToolMessage(content='{"coord": {"lon": 120.1614, "lat": 30.2937}, "weather": [{"id": 804, "main": "Clouds", "description": "\\u9634\\uff0c\\u591a\\u4e91", "icon": "04d"}], "base": "stations", "main": {"temp": 24.3, "feels_like": 24.77, "temp_min": 24.3, "temp_max": 24.3, "pressure": 1015, "humidity": 76, "sea_level": 1015, "grnd_level": 1013}, "visibility": 10000, "wind": {"speed": 2.76, "deg": 33, "gust": 5.19}, "clouds": {"all": 100}, "dt": 1758429625, "sys": {"country": "CN", "sunrise": 1758404816, "sunset": 1758448693}, "timezone": 28800, "id": 1808926, "name": "Hangzhou", "cod": 200}', name='get_weather', id='d26d2dd7-6714-4a09-93bd-988d420f4111', tool_call_id='call_01_7Sa3enQEBVhLjlvMGxIdqhtT'), 
AIMessage(content='根据查询结果，我来为您比较一下北京和杭州的温度：\n\n**北京天气情况：**\n- 温度：26.6°C\n- 体感温度：26.6°C  \n- 湿度：43%\n- 天气状况：多云\n\n**杭州天气情况：**\n- 温度：24.3°C\n- 体感温度：24.8°C\n- 湿度：76%\n- 天气状况：阴，多云\n\n**结论：北京比杭州更热一些。** 北京的温度为26.6°C，而杭州的温度为24.3°C，北京比杭州高了约2.3°C。虽然杭州的湿度较高（76% vs 43%），但实际温度还是北京更高。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 159, 'prompt_tokens': 812, 'total_tokens': 971, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 556}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': '41140b0a-f47d-41af-87f9-f795ee20c781', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--0093394e-17e1-4b85-afc4-f82702ebac52-0', usage_metadata={'input_tokens': 812, 'output_tokens': 159, 'total_tokens': 971, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}})
]


"""





