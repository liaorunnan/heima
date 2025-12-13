import openai
from conf import settings

client = openai.Client(api_key=settings.api_key, base_url=settings.base_url)

def chat(query, history=[], system_prompt='你是一个小学一年级的语文老师，我会给你一篇篇的markdown文档，请帮我取出其中的文档，因为是'
                                          '要做智能音箱，所以提取出来的文本是可以直接读的，而且读着顺口的。注意不要有多余的废话，不要给'
                                          '音效，要写的有趣，不要描述画面，为防止拼音读成英文，请注意单个字母换成同音的字，千万不要写单'
                                          '个字母，不要写注解、提示'):
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
