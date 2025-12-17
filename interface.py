import chainlit as cl, requests, json, asyncio
from chainlit.cli import run_chainlit

API_URL = "http://localhost:8006"


@cl.on_chat_start
async def start():
    await cl.Message("你好，我是英语AI助手，有什么可以帮助你的吗？").send()
    cl.user_session.set("history", [])


@cl.on_message
async def handle_msg(message: cl.Message):
    msg = cl.Message("")
    await msg.send()

    history = cl.user_session.get("history", [])
    response = requests.post(f"{API_URL}/api/ragapi",
                             json={"query": message.content, "history": history},
                             stream=True, timeout=30)


    answer = ""
    for line in response.iter_lines():

        
       
     
        if line and (s := line.decode()).startswith('data:'):
          
            chunk = json.loads(s[5:])
           
            
        
            
            token = chunk.get('token', '')
            if token:
                await msg.stream_token(token)
                answer += token
            elif chunk.get('complete'):
                break
                # await msg.stream_token(answer)

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)


if __name__ == '__main__':
    run_chainlit(target=__file__, port=9001)