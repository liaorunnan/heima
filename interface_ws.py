import chainlit as cl, requests, json, asyncio
from chainlit.cli import run_chainlit
import websockets

API_URL = "ws://localhost:8003/api/wsragapi"


@cl.on_chat_start
async def start():
    await cl.Message("你好，我是英语AI助手，有什么可以帮助你的吗1111？").send()
    cl.user_session.set("history", [])


@cl.on_message
async def handle_msg(message: cl.Message):
    msg = cl.Message("")
    await msg.send()

    history = cl.user_session.get("history", [])
    async with websockets.connect(API_URL) as websocket:
        await websocket.send(json.dumps({"query": message.content, "history": history}))

   
        answer = ""
        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "token":
                token = data.get("token", "")
                if token:
                    answer += token
                    await msg.stream_token(token)
            elif data.get("type") == "end":
                break

        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", history)


