
import uuid
from fastapi import FastAPI, Request,WebSocket
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import random

from mianshi import get_random_question_logic

from translate_baidu import baidu_ai_translate
from wenzhang.wenzhang import SummarizeAgent
from chat.chat import chat_text
from rag.rag_api import rag_query
from rag.rag_run_steam import rag_stream_run
from zhinengjiaowu.jwrag_api import jwrag_query

from pydantic import BaseModel # 1. 引入 Pydantic
import json # 引入 json 用于解析 AI 返回的结果
from typing import List, Dict


app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArticleRequest(BaseModel):
    article: str

class JW_ChatRequest(BaseModel):
    question: str
    image: str = ""
    history: List[Dict[str, str]] = [] 

class ChatRequest(BaseModel):
    question: str
    scenario: str = "general"
    history: List[Dict[str, str]] = [] 


class QueryRequest(BaseModel):
    query: str
    history: list = []





    


@app.get("/")
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)



@app.get("/mianshifont")
def read_root():
    file_path = os.path.join(os.path.dirname(__file__), "mianshi.html")
    return FileResponse(file_path)

@app.get("/mianshi")
def interview_api(types: str = ""):

    result_data = get_random_question_logic(types)
    
    return result_data

@app.get("/translatefont")
def translate_font():
    file_path = os.path.join(os.path.dirname(__file__), "translate.html")
    return FileResponse(file_path)
@app.get("/translate")
def translate_api(query_text: str = ""):
    result_data = baidu_ai_translate(query_text = query_text)

    return result_data

@app.get("/zaiyaofont")
def zaiyao_font():
    file_path = os.path.join(os.path.dirname(__file__), "wenzhang/wenzhang.html")
    return FileResponse(file_path)
@app.post("/zaiyao")
def zaiyao_api(item: ArticleRequest): # 3. 使用模型作为参数
    SummarizeAgent_model = SummarizeAgent()
    ai_response_str = SummarizeAgent_model.run(article=item.article)

    
    try:
        # 尝试将 AI 返回的字符串转为字典
        ai_data = json.loads(ai_response_str)
        summary_content = ai_data.get("summary", ai_response_str)
    except:
        # 如果 AI 返回的不是标准 JSON，直接用原字符串
        summary_content = ai_response_str

    # 返回符合前端逻辑的结构
    return {
        "success": True,
        "summary": summary_content
    }

@app.get("/chatfont")
def chat_font():
    file_path = os.path.join(os.path.dirname(__file__), "chat/chat.html")
    return FileResponse(file_path)

@app.post("/chat")
def chat_api(item: ChatRequest): 
    user_msg = item.question
    user_scenario = item.scenario
    user_history = item.history
    chat_content = chat_text(question=user_msg, scenario=user_scenario, history=user_history)

    

    # try:
    #     # 尝试将 AI 返回的字符串转为字典
    #     ai_data = json.loads(ai_response_str)
    #     chat_content = ai_data.get("chat", ai_response_str)
    # except:
    #     # 如果 AI 返回的不是标准 JSON，直接用原字符串
    #     chat_content = ai_response_str

    # 返回符合前端逻辑的结构
    return chat_content


@app.get("/ragfont")
def rag_font():
    file_path = os.path.join(os.path.dirname(__file__), "rag/rag_chat.html")
    return FileResponse(file_path)

@app.post("/api/ragapi") #SSE
async def rag_api_stream(item: QueryRequest):
    sid = str(uuid.uuid4())
    return StreamingResponse((f"data:{json.dumps(chunk)}\n\n" async for chunk in rag_stream_run(query=item.query, history=item.history, session_id=sid)), 
                             media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.websocket("/api/wsragapi") #SSE
async def rag_api_stream_ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        request = json.loads(data)
        query = request.get("query","").strip()
        if not query:
            await websocket.send_json(json.dumps({"type": "error", "message": "Query is empty"}))
            continue

        sid = str(uuid.uuid4())
        await websocket.send_json({"type": "start",  "session_id": sid})

        async for answer in rag_stream_run(query,history=request.get("history",[]), session_id=sid):
            if answer.get("complete"):
                await websocket.send_json({"type":"end", "complete": True, "end_time": answer.get("end_time",0)})
            else:
                await websocket.send_json({"type":"token","token": answer.get("token", ""), "session_id": answer.get("session_id", ""), "query_type": answer.get("query_type", "unknown")})
    
    

@app.post("/rag")
def rag_api(item: ChatRequest): 
    user_msg = item.question
    user_history = item.history
    # 注意：rag_query 目前没有使用 scenario 参数，但我们传递时忽略即可
    answer = rag_query(query=user_msg, history=user_history)
    # 返回结构需要与前端匹配，前端期望一个包含 answer 的 JSON 对象
    return {"answer": answer}

@app.get("/listeningfont")
def listening_font():
    file_path = os.path.join(os.path.dirname(__file__), "listening/listening.html")
    return FileResponse(file_path)


@app.get("/jiaowufont")
def jiaowu_font():
    file_path = os.path.join(os.path.dirname(__file__), "zhinengjiaowu/zhinengjiaowu.html")
    
    return FileResponse(file_path)

@app.post("/jiaowuapi")
def jiaowu_api(item: JW_ChatRequest): 

    user_msg = item.question
    user_history = item.history
    # 注意：rag_query 目前没有使用 scenario 参数，但我们传递时忽略即可
    answer = jwrag_query(query=user_msg, history=user_history)
    # 返回结构需要与前端匹配，前端期望一个包含 answer 的 JSON 对象
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)

