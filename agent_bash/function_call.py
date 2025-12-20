import json
from openai import OpenAI
from conf import settings

def get_weather(city: str) -> str:
    """
    获取城市的天气信息
    :param city: 城市名称
    :return: 天气信息字符串
    """
    dummy_weather = {
        "北京": "晴朗",
        "上海": "多云",
        "广州": "阴",
        "深圳": "雨",
    }
    # 这里简单返回一个模拟的天气信息
    return f"{city}的天气是{dummy_weather.get(city, '未知')}"

def dress_advice(weather: str) -> str:
    """
    根据天气建议穿衣
    :param weather: 天气信息字符串
    :return: 穿衣建议字符串
    """
    if "晴" in weather:
        return "建议不带伞"
    elif "多云" in weather:
        return "建议不带伞"
    elif "雨" in weather:
        return "建议带伞"
    elif "阴" in weather:
        return "建议带伞"
    else:
        return "建议不带伞"

def get_city():
    """
    获取用户所在的城市
    :return: 城市名称字符串
    """
    return "广州"

tools = [
    {
        "type": "function",
        "function":{
            "name": "get_weather",
            "description": "获取城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                    },
                },
                "required": ["city"],
            },
        }
    },
    {
        "type": "function",
        "function":{
            "name": "dress_advice",
            "description": "根据天气建议穿衣",
            "parameters": {
                "type": "object",
                "properties": {
                    "weather": {
                        "type": "string",
                        "description": "天气信息",
                    },
                },
                "required": ["weather"],
            },
        }
    }
]



tool_map = {
    "get_weather": get_weather,
    "dress_advice": dress_advice
}

client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def find_weather(query):
 

    message = [{
        "role": "system",
        "content": "你是一个智能助手，能够根据用户需求调用工具完成任务"
        },
        {
            "role": "user",
            "content": query
        }]

    
    while True:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=message,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        
        )

        result = response.choices[0].message

    
        if not result.tool_calls:
            break


        for tool_call in result.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_name in tool_map:
                tool_output = tool_map[tool_name](**args)  # 直接调用对应函数
            else:
                tool_output = f"未知工具：{tool_name}"
            message.append({
                "role": "assistant",
                "tool_calls":[tool_call]
            })
            message.append({
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tool_call.id,
                "content": tool_output
            })
            
    return result.content
        

print(find_weather("帮我查一下广州、上海的天气，然后根据天气给我一些穿衣建议"))
        

    

