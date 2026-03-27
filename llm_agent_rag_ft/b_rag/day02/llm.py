import openai
from conf import settings

client = openai.OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key
)


def chat(query, history=[], system_prompt="你是一个儿童读物智能助手,输出相关课文内容"):
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
