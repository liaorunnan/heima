import asyncio
import time
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()


# --- 模拟打标签的耗时函数 ---
async def tag_user_task(user_id: int):
    print(f"Start: 正在给用户 {user_id} 打标签...")
    await asyncio.sleep(3)  # 模拟耗时 3 秒的操作（比如写数据库、调外部API）
    print(f"Finish: 用户 {user_id} 标签打完了！")



# --- 场景 2：正确做法（用户无感，后台执行） ---
@app.post("/chat_fast")
async def chat_fast(user_id: int, background_tasks: BackgroundTasks):
    """
    background_tasks: FastAPI 注入的专门处理后台任务的对象
    """
    
    # 1. 把任务添加到后台队列中
    # 注意：这里只传函数名和参数，不要加 await，也不要括号调用
    background_tasks.add_task(tag_user_task, user_id)
    
    # 2. 立即返回响应给用户
    # 此时，tag_user_task 甚至可能还没开始运行，或者刚开始运行
    return {"message": "这条消息瞬间发送！标签正在后台静默处理。"}

if __name__ == "__main__":
    import uvicorn
    # 11434
    uvicorn.run("yibu:app", host="127.0.0.1", port=8090)