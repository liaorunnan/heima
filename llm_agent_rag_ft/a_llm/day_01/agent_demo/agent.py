import json
from a_llm.day_01.agent_demo.tool_desc import list_tools, call_tool
import openai
from conf import settings

client = openai.Client(api_key=settings.api_key,base_url=settings.base_url)

system_prompt="你是一个助手,根据用户的指令,使用你的工具包一步步计算,并输出结果"

history=[
    {"role": "user", "content": "4+5*6-2"},
    {"role": "system", "content": json.dumps({"tool_name":"mul_tool","tool_param":{"a":5,"b":6},"is_end":False})},
    {"role": "user", "content": json.dumps({"tool_name": "mul_tool", "tool_param": {"a": 5, "b": 6}, "tool_result": 30})},
    {"role": "assistant", "content": json.dumps({"tool_name": "add_tool", "tool_param": {"a": 4, "b": 30}, "is_end": False})},
    {"role": "user", "content": json.dumps({"tool_name": "add_tool", "tool_param": {"a": 4, "b": 30}, "tool_result": 34})},
    {"role": "assistant", "content": json.dumps({"tool_name": "sub_tool", "tool_param": {"a": 34, "b": 2}, "is_end": False})},
    {"role": "user", "content": json.dumps({"tool_name": "sub_tool", "tool_param": {"a": 34, "b": 2}, "tool_result": 32})},
    {"role": "assistant", "content": json.dumps({"tool_name": None, "tool_param": None, "is_end": True})},
]

user_prompt="6/3+2*2"

messages=[
    {"role":"system","content":system_prompt},
    *history,
    {"role":"user","content":user_prompt}
]

response=client.chat.completions.create(
    model=settings.model_name,
    messages=messages,
    temperature=0,
)
result = json.loads(response.choices[0].message.content)

