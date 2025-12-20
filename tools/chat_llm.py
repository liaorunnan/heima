import openai
from conf import settings


def chat(query,history, system_prompt="",temperature=0.0,base_url=settings.base_url, api_key=settings.api_key,model=settings.model_name):
    client = openai.Client(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query}
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print(chat("你好,你是谁",[]))
