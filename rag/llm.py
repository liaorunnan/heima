import openai
from conf import settings
client = openai.Client(api_key=settings.api_key,base_url=settings.base_url)

def chat(query,history =[],system_prompt="You are a helpful assistant."):
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
