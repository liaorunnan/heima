import json
import os
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, Query, BackgroundTasks
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from typing import List

from src.services.file_response_service import (
    extract_text_with_pdfnumber,
    extract_text_from_docx,
    hybrid_chunk,
    generate_embeddings
)
from src.utils.logger_utils import setup_logger
from src.utils.request_storage import (
    store_request_body,
    delete_request_body
)
from src.workflow.workflow_main_agent import WorkflowMainAgent
from src.tools.milvus_tools import MilvusManager, get_milvus_manager

from tag_mcp import run_intent_analysis

from pydantic import BaseModel

router = APIRouter()
workflow = WorkflowMainAgent()

logger = setup_logger(__name__)

# 创建全局 MilvusManager 实例
_milvus_manager = None




def get_handoff_intent_prompt(exit_keywords: List[str]) -> str:
    """生成转人工意图检测的提示词

    Args:
        exit_keywords: 退出关键词列表

    Returns:
        格式化后的提示词字符串
    """
    return (
        "你是客服质检助手，请判断用户是否需要转人工客服。"
        "如果当前用户输入满足exitKeyword中的条件，则返回handoff=true，否则返回handoff=false。"
        f"exitKeyword: {exit_keywords}"
        "请仅输出 JSON：{\"handoff\": true|false}。"
    )
@router.post("/")
async def stream_response_api(request: Request):
    """流式响应 API 端点，返回 Server-Sent Events"""
    logger.info("[stream_response_api] 收到 POST 请求")

    try:
        body = await request.json()
        logger.info("[stream_response_api] 请求体解析成功")
    except Exception as exc:
        logger.error("[stream_response_api] JSON 解析失败: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid JSON body: {exc}"}
        )

    # 验证必需字段
    required_fields = ["session_id", "user_id", "bot_id", "app_id", "user_input"]
    missing = [field for field in required_fields if field not in body]
    if missing:
        logger.warning("[stream_response_api] 缺少必需字段: %s", missing)
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing fields: {', '.join(missing)}"}
        )

    session_id = str(body["session_id"])
    user_id = str(body["user_id"])
    bot_id = str(body["bot_id"])
    app_id = str(body["app_id"])
    tenant_outer_id_raw = body.get("tenant_outer_id")
    tenant_outer_id = str(tenant_outer_id_raw) if tenant_outer_id_raw is not None else ""
    user_input = str(body["user_input"])
    equity = body.get("info")

    # 存储请求体，供其他方法获取，返回唯一的请求ID
    request_id = store_request_body(user_id, session_id, body)
    logger.info(f"[stream_response_api] 已存储请求体: user_id={user_id}, session_id={session_id}, request_id={request_id}")

    # 打印参数
    logger.info(f"[stream_response_api打印参数] session_id={session_id}, user_id={user_id},bot_id={bot_id},app_id={app_id},tenant_outer_id={tenant_outer_id},user_input={user_input},equity={equity}")

    if equity is not None and not isinstance(equity, dict):
        logger.warning("[stream_response_api] equity 格式错误，返回 400")
        # 删除已存储的请求体
        delete_request_body(user_id, session_id, request_id)
        return JSONResponse(
            status_code=400,
            content={"error": "equity must be a JSON object if provided"}
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        """生成 Server-Sent Events 格式的字符串"""
        try:
            logger.info("[stream_response_api] 开始异步流式处理")
            async for event in workflow.start_process_async(
                request_id=request_id
            ):
                # 添加时间戳
                payload_with_timestamp = {**event, "timestamp": time.time()}
                # 转换为 JSON 字符串
                text = json.dumps(payload_with_timestamp, ensure_ascii=False)
                # 生成 SSE 格式的数据
                yield f"data: {text}\n\n"
            
            logger.info("[stream_response_api] 异步流式处理完成")
        except Exception as exc:
            logger.exception("[stream_response_api] 流式处理失败")
            error_event = {
                "event": "error",
                "message": str(exc),
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            # 处理完成后删除请求体（使用 request_id 精确删除）
            delete_request_body(user_id, session_id, request_id)
            logger.info(f"[stream_response_api] 已删除请求体: user_id={user_id}, session_id={session_id}, request_id={request_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream;charset=UTF-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        }
    )

class TagRequest(BaseModel):
    conversation_id: str
    session_id: str
    user_id: str
    
@router.post("/customer_tagging")
async def customer_tagging(request: TagRequest, background_tasks: BackgroundTasks):
    """
    客户标签分析端点（后台异步处理）
    
    使用 BackgroundTasks 将耗时操作放入后台执行，立即返回响应给用户，
    让用户无感知等待标签分析过程。
    """
    conversation_id = request.conversation_id
    session_id = request.session_id
    user_id = request.user_id

    # 将耗时任务添加到后台队列中
    # 注意：这里只传函数名和参数，不要加 await，也不要括号调用
    background_tasks.add_task(run_intent_analysis, conversation_id=conversation_id, session_id=session_id, user_id=user_id)
    
    # 立即返回响应给用户，任务在后台静默处理
    logger.info(f"客户标签分析任务已添加到后台队列 - 会话ID: {conversation_id}, 用户ID: {user_id}")
    
    return {
        "message": "客户标签分析任务已启动，正在后台处理中",
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": user_id,
        "status": "processing"
    }