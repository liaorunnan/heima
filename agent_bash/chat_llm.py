import openai



def chat(query,history, system_prompt="",temperature=0.0,base_url='https://api.deepseek.com', api_key="sk-621e417ec1a640cf8f990b76b20d05e2",model="deepseek-chat"):
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
