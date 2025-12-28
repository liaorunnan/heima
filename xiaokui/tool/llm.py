import openai
from conf import settings

client = openai.OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key
)


def chat(query, history=[], system_prompt="You are a helpful assistant."):
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    text = "简单来说，这个错误是阿里云IDE插件的一个bug，与你自己的项目代码无关。它会导致你在进行一些UI操作（如选择文本）时，后台抛出错误（虽然可能不影响主要功能，但会污染日志并可能引起卡顿）。建议你首先尝试【更新插件和IDE】。 如果不行，再按照流程图进行禁用排查。如果你需要帮助查找具体的插件名称或操作步骤，可以告诉我你使用的IntelliJ IDEA版本和操作系统（如 Windows 11, macOS等），我可以提供更详细的指引。"
    result = chat(query=text, system_prompt="你能够对下面的文章进行一个摘要提取，获取20字左右的摘要：")
    print(result)