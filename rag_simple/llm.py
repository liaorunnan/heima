import openai
from config import settings

client = openai.OpenAI(
    base_url=settings.BASE_URL,
    api_key=settings.API_KEY
)


def chat(query, history=[], system_prompt="你是一个智能AI法律咨询小助手"):
    response = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query}
        ],
        temperature=0
    )
    return response.choices[0].message.content
