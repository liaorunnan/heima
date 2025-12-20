import openai
from conf import settings

client = openai.Client(base_url=settings.base_url, api_key=settings.api_key)
client = openai.Client(base_url=settings.qw_api_url, api_key=settings.qw_api_key)
client = openai.Client(base_url=settings.autodl_qw_api_url, api_key=settings.autodl_qw_api_key)



def chat(query,history, system_prompt="你是一位英语老师，请根据我的提问，帮我寻找对应的听力文章或者回答我关于英语文章内容的问题。如果是在查询英文单词，请详细列出音标和解释，并附上几条例句；如果是在要写文章，请根据我的要求，写一篇符合要求的文章并标记出引用的文章模版；如果没有在参考问题中找到相关内容，请告诉我。整体回答结构分明，意思清晰，不要重复。"):
    response = client.chat.completions.create(
        model = settings.autodl_qw_model,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print(chat("你好,你是谁",[]))
