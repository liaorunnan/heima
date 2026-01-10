
"""
预构建智能体 API 就是 create_react_agent 的使用

串联  调用：借助LangGraph React Agent，无需任何额外设置，即可完成多工具串联和并联调用。

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



class Write_Query(BaseModel):
    content: str = Field(description="需要写入文档的具体内容")

@tool(args_schema = Write_Query)
def write_file(content: str) -> str:
    """
    将指定内容写入本地文件。
    :param content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。
    :return：是否成功写入
    """
    return "已成功写入本地文件。"





from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# 封装外部函数列表
tools = [get_weather, write_file]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  
agent = create_react_agent(model=model, tools=tools)

response = agent.invoke({"messages": [{"role": "user", "content": "你好，请帮我查询北京和杭州的天气，并将其写入本地文件中。"}]})

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
HumanMessage(content='你好，请帮我查询北京和杭州的天气，并将其写入本地文件中。', additional_kwargs={}, response_metadata={}, id='2199b374-edb4-480b-9228-409c9e2f329f'), 
AIMessage(content='我来帮您查询北京和杭州的天气，然后写入本地文件。', additional_kwargs={'tool_calls': [{'id': 'call_00_4CR3vvFrOVQpFpa8Jc6Suwp6', 'function': {'arguments': '{"loc": "Beijing"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0}, {'id': 'call_01_xQQx3TwJ9HGDAHd8hlwyRIh9', 'function': {'arguments': '{"loc": "Hangzhou"}', 'name': 'get_weather'}, 'type': 'function', 'index': 1}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 352, 'total_tokens': 395, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 32}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'd5efd8da-a2b5-4852-92b6-a2ad8751b2ee', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--e8ada4b9-d97b-4167-8ebb-a957215cba9e-0', tool_calls=[{'name': 'get_weather', 'args': {'loc': 'Beijing'}, 'id': 'call_00_4CR3vvFrOVQpFpa8Jc6Suwp6', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'loc': 'Hangzhou'}, 'id': 'call_01_xQQx3TwJ9HGDAHd8hlwyRIh9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 352, 'output_tokens': 43, 'total_tokens': 395, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}}), 
ToolMessage(content='{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 803, "main": "Clouds", "description": "\\u591a\\u4e91", "icon": "04d"}], "base": "stations", "main": {"temp": 26.51, "feels_like": 26.51, "temp_min": 26.51, "temp_max": 26.51, "pressure": 1020, "humidity": 44, "sea_level": 1020, "grnd_level": 1014}, "visibility": 10000, "wind": {"speed": 2.15, "deg": 150, "gust": 3.03}, "clouds": {"all": 76}, "dt": 1758429001, "sys": {"country": "CN", "sunrise": 1758405645, "sunset": 1758449671}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}', name='get_weather', id='f9eec4a9-4b0a-4c87-8c07-9269eb3a6cdd', tool_call_id='call_00_4CR3vvFrOVQpFpa8Jc6Suwp6'), 
ToolMessage(content='{"coord": {"lon": 120.1614, "lat": 30.2937}, "weather": [{"id": 804, "main": "Clouds", "description": "\\u9634\\uff0c\\u591a\\u4e91", "icon": "04d"}], "base": "stations", "main": {"temp": 24.3, "feels_like": 24.77, "temp_min": 24.3, "temp_max": 24.3, "pressure": 1015, "humidity": 76, "sea_level": 1015, "grnd_level": 1013}, "visibility": 10000, "wind": {"speed": 2.76, "deg": 33, "gust": 5.19}, "clouds": {"all": 100}, "dt": 1758429625, "sys": {"country": "CN", "sunrise": 1758404816, "sunset": 1758448693}, "timezone": 28800, "id": 1808926, "name": "Hangzhou", "cod": 200}', name='get_weather', id='4fb9c983-c509-4d65-9167-56fa29c4867e', tool_call_id='call_01_xQQx3TwJ9HGDAHd8hlwyRIh9'), 
AIMessage(content='现在我将天气信息整理并写入本地文件：', additional_kwargs={'tool_calls': [{'id': 'call_00_IXjFIk3CoczWGEkMMPSxzkne', 'function': {'arguments': '{"content": "\\u5929\\u6c14\\u67e5\\u8be2\\u62a5\\u544a\\n==================\\n\\n\\u5317\\u4eac\\u5929\\u6c14\\u60c5\\u51b5\\uff1a\\n- \\u57ce\\u5e02\\uff1a\\u5317\\u4eac (Beijing)\\n- \\u5929\\u6c14\\u72b6\\u51b5\\uff1a\\u591a\\u4e91\\n- \\u5f53\\u524d\\u6e29\\u5ea6\\uff1a26.51\\u00b0C\\n- \\u4f53\\u611f\\u6e29\\u5ea6\\uff1a26.51\\u00b0C\\n- \\u6e7f\\u5ea6\\uff1a44%\\n- \\u6c14\\u538b\\uff1a1020 hPa\\n- \\u98ce\\u901f\\uff1a2.15 m/s\\n- \\u80fd\\u89c1\\u5ea6\\uff1a10000\\u7c73\\n- \\u4e91\\u91cf\\uff1a76%\\n\\n\\u676d\\u5dde\\u5929\\u6c14\\u60c5\\u51b5\\uff1a\\n- \\u57ce\\u5e02\\uff1a\\u676d\\u5dde (Hangzhou)\\n- \\u5929\\u6c14\\u72b6\\u51b5\\uff1a\\u9634\\uff0c\\u591a\\u4e91\\n- \\u5f53\\u524d\\u6e29\\u5ea6\\uff1a24.3\\u00b0C\\n- \\u4f53\\u611f\\u6e29\\u5ea6\\uff1a24.77\\u00b0C\\n- \\u6e7f\\u5ea6\\uff1a76%\\n- \\u6c14\\u538b\\uff1a1015 hPa\\n- \\u98ce\\u901f\\uff1a2.76 m/s\\n- \\u80fd\\u89c1\\u5ea6\\uff1a10000\\u7c73\\n- \\u4e91\\u91cf\\uff1a100%\\n\\n\\u67e5\\u8be2\\u65f6\\u95f4\\uff1a2025\\u5e741\\u670820\\u65e5\\n\\n\\u5907\\u6ce8\\uff1a\\u5317\\u4eac\\u6e29\\u5ea6\\u7565\\u9ad8\\u4e8e\\u676d\\u5dde\\uff0c\\u4f46\\u6e7f\\u5ea6\\u8f83\\u4f4e\\uff1b\\u676d\\u5dde\\u6e7f\\u5ea6\\u8f83\\u9ad8\\uff0c\\u4f53\\u611f\\u5e73\\u6da6\\u5ea6\\u8f83\\u597d\\u3002"}', 'name': 'write_file'}, 'type': 'function', 'index': 0}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 694, 'prompt_tokens': 895, 'total_tokens': 1589, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 384}, 'prompt_cache_hit_tokens': 384, 'prompt_cache_miss_tokens': 511}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'c06cd1ea-b0ae-4dec-a6a6-512c6782515c', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--5eb5c540-96cb-4451-a847-37c0092970e3-0', tool_calls=[{'name': 'write_file', 'args': {'content': '天气查询报告\n==================\n\n北京天气情况：\n- 城市：北京 (Beijing)\n- 天气状况：多云\n- 当前温度：26.51°C\n- 体感温度：26.51°C\n- 湿度：44%\n- 气压：1020 hPa\n- 风速：2.15 m/s\n- 能见度：10000米\n- 云量：76%\n\n杭州天气情况：\n- 城市：杭州 (Hangzhou)\n- 天气状况：阴，多云\n- 当前温度：24.3°C\n- 体感温度：24.77°C\n- 湿度：76%\n- 气压：1015 hPa\n- 风速：2.76 m/s\n- 能见度：10000米\n- 云量：100%\n\n查询时间：2025年1月20日\n\n备注：北京温度略高于杭州，但湿度较低；杭州湿度较高，体感平润度较好。'}, 'id': 'call_00_IXjFIk3CoczWGEkMMPSxzkne', 'type': 'tool_call'}], usage_metadata={'input_tokens': 895, 'output_tokens': 694, 'total_tokens': 1589, 'input_token_details': {'cache_read': 384}, 'output_token_details': {}}), 
ToolMessage(content='已成功写入本地文件。', name='write_file', id='175005e1-00ce-4ccc-bf4c-18ff9db61af8', tool_call_id='call_00_IXjFIk3CoczWGEkMMPSxzkne'), 
AIMessage(content='完成！我已经成功查询了北京和杭州的天气信息，并将详细的气象数据写入本地文件中。\n\n**主要发现：**\n- 北京：26.51°C，多云，湿度44%\n- 杭州：24.3°C，阴天多云，湿度76%\n\n文件包含了完整的天气信息，包括温度、湿度、气压、风速等详细数据，方便您查阅和比较两个城市的天气状况。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 87, 'prompt_tokens': 1152, 'total_tokens': 1239, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 896}, 'prompt_cache_hit_tokens': 896, 'prompt_cache_miss_tokens': 256}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_08f168e49b_prod0820_fp8_kvcache', 'id': 'dd796e25-3d7c-4aa4-a1f8-8e8c1738e14a', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--f4c82e7e-a447-4d35-a8b8-00acf65bdcaf-0', usage_metadata={'input_tokens': 1152, 'output_tokens': 87, 'total_tokens': 1239, 'input_token_details': {'cache_read': 896}, 'output_token_details': {}})
]

"""





