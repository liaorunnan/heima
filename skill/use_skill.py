import openai
import json
import importlib

from conf import settings

shuoming_dir = "skill/manifest.json"

prompt = "你是一个智能助手。你可以使用以下工具来解决用户问题：\n\n"

with open(shuoming_dir, "r", encoding="utf-8") as f:
    manifest = json.load(f)

        
for skill in manifest['skill']:
    # 【这里是重点】：LLM 就是靠读这一行 description 来决定选谁的！
    prompt += f"- 工具名: {skill['name']}\n"
    prompt += f"  描述: {skill['description']}\n"
    prompt += f"  参数: {skill['schemas']}\n"
    prompt += "\n"
            
prompt += "请根据用户输入，只返回 JSON 格式结果，格式如：{\"tool\": \"工具名\", \"args\": {参数字典}}\n"


client = openai.Client(api_key=settings.api_key,base_url=settings.base_url)

result = client.chat.completions.create(
    model=settings.model_name,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": "请利用skill完成以下任务：我想知道中国宋代的诗歌"}
    ],
    max_tokens=100,
    temperature=0.5,
)

def run_skill(**kwargs):

    tool_name =  kwargs['tool'].replace("Skill", "").lower()

    full_module_path = f"skill.actions.{tool_name}"

    module = importlib.import_module(full_module_path)

    ToolClass = getattr(module, kwargs['tool'])

    tool_instance = ToolClass(**kwargs['args'])

    print(tool_instance.execute())


doct = json.loads(result.choices[0].message.content)
run_skill(**doct)

# print(result.choices[0].message.content)
