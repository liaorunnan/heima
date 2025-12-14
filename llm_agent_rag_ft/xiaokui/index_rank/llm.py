import openai
from conf import settings

client = openai.OpenAI(
    base_url=settings.base_url,
    api_key=settings.api_key
)


def chat(query, history=[], system_prompt="根据用户输出的中文单词,输出相关单词的音标和意思"):
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
