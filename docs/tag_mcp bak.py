import json
import logging
import threading
import time
from concurrent.futures import as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from bot_mcp import call_tool as _base_call_tool
from config.settings import TAG_MCP_SSE_URL
from core.thread_pool import get_thread_pool
from src.tools import (
    get_app_id_from_session,
    get_conversation_with_cache,
    list_chat_messages,
    scan_chat_conversations,
)
from Responder import llm_generic
# from scheduler import new_scheduler, remove_scheduler, list_schedulers, scheduler
from mcp.server.fastmcp import FastMCP

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确认调度器状态
logger.info("=" * 60)
logger.info("Tag MCP 模块初始化中...")
logger.info(f"调度器状态: {'运行中' if scheduler.running else '未运行'}")
logger.info(f"调度器任务数量: {len(scheduler.get_jobs())}")
if scheduler.get_jobs():
    for job in scheduler.get_jobs():
        logger.info(f"  任务 ID: {job.id}, 下次执行时间: {job.next_run_time}")
logger.info("=" * 60)

# MCP 服务器实例（用于暴露标签相关工具）
mcp = FastMCP("Tag Server", host="0.0.0.0", port=3005)

# 任务执行状态管理（参照 scorer.py 的实现方式）
_label_task_running = False
_label_task_lock = threading.Lock()
_LABEL_JOB_ID = "hourly_label_task"

# 系统标签缓存
_label_cache_lock = threading.Lock()
_label_cache: Dict[str, Dict[str, Any]] = {}
_LABEL_CACHE_TTL = 10 * 60  # 10 分钟缓存


def call_tool(
    tool_name: str,
    params: Dict[str, Any],
    url: str = TAG_MCP_SSE_URL,
    timeout: Optional[float] = 180.0,
) -> Any:
    """封装标签相关 Dify MCP 的 call_tool，默认连接到独立的 SSE 服务."""
    return _base_call_tool(tool_name, params, url=url, timeout=timeout)


def _flatten_label_nodes(nodes: Sequence[Dict[str, Any]], parent_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """将树形结构的标签节点扁平化为列表"""
    flattened: List[Dict[str, Any]] = []
    if not nodes:
        return flattened
    for node in nodes:
        if not isinstance(node, dict):
            continue
        entry = {
            "id": node.get("id"),
            "name": node.get("name"),
            "description": node.get("description"),
            "parent": parent_name,
        }
        flattened.append(entry)
        children = node.get("children")
        if isinstance(children, list) and children:
            flattened.extend(_flatten_label_nodes(children, parent_name=node.get("name")))
    return flattened


def _extract_mcp_data(raw: Any) -> Optional[Dict[str, Any]]:
    """从 MCP 返回结果中提取数据字典"""
    if hasattr(raw, "structuredContent"):
        return raw.structuredContent
    
    if isinstance(raw, dict):
        if "structuredContent" in raw:
            return raw["structuredContent"]
        if "content" in raw:
            content_list = raw.get("content", [])
            if content_list and isinstance(content_list[0], dict) and "text" in content_list[0]:
                try:
                    text = content_list[0]["text"]
                    return json.loads(text) if isinstance(text, str) else text
                except (json.JSONDecodeError, TypeError):
                    pass
        return raw
    
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    
    return None


def _get_valid_labels(app_id: str) -> List[Dict[str, Any]]:
    """获取指定应用的有效系统标签列表，带缓存机制"""
    # 强制使用 "aiwa-admin" 作为 app_id 获取标签定义
    admin_app_id = "aiwa-admin"
    
    now = time.time()
    with _label_cache_lock:
        cached = _label_cache.get(admin_app_id)
        if cached and cached.get("expires_at", 0) > now:
            return cached.get("labels", [])

    try:
        # 使用 aiwa-admin 获取标签
        raw = call_tool("get_valid_labels", params={"appId": admin_app_id})
        data = _extract_mcp_data(raw)
        
        if not isinstance(data, dict):
            logger.warning("无法解析 MCP 返回数据")
            return []
            
    except Exception as exc:
        logger.error("获取系统标签失败: %s", exc, exc_info=True)
        return []

    # 从解析后的数据中提取 labels.system
    system_labels = None
    if "labels" in data and isinstance(data["labels"], dict):
        system_labels = data["labels"].get("system")
    elif "system" in data:
        system_labels = data["system"]
    
    flattened = _flatten_label_nodes(system_labels or [])
    logger.info(f"扁平化后的标签数量: {len(flattened)}")

    with _label_cache_lock:
        _label_cache[admin_app_id] = {"labels": flattened, "expires_at": now + _LABEL_CACHE_TTL}

    return flattened


def _group_labels_by_parent(labels: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按顶级父类分组标签，每个顶级父类及其所有子类为一组
    
    返回格式: {父类名称: [父类标签, 子类1, 子类2, ...]}
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    
    # 第一步：找出所有顶级父类（parent 为 None 的标签）
    top_level_parents = []
    for label in labels:
        if label.get("parent") is None:
            parent_name = label.get("name")
            if parent_name:
                top_level_parents.append(parent_name)
                # 初始化组，包含父类本身
                groups[parent_name] = [label]
    
    # 第二步：将每个父类的所有子类添加到对应的组
    for label in labels:
        parent_name = label.get("parent")
        if parent_name and parent_name in groups:
            # 这是某个顶级父类的子类
            groups[parent_name].append(label)
    
    return groups


def _build_label_prompt_for_parent(parent_name: Optional[str], labels: Sequence[Dict[str, Any]]) -> str:
    """为单个父类组构建 prompt"""
    label_lines = []
    for item in labels:
        try:
            label_id = item.get("id")
            if label_id is None:
                continue
            name = (item.get("name") or "").strip()
            if not name:
                continue
            description = (item.get("description") or "未提供").strip()
            parent = item.get("parent")
            label_path = f"{parent}>{name}" if parent else name
            label_lines.append(f"- ID={label_id} | {label_path} | {description}")
        except Exception:
            continue

    label_section = "\n".join(label_lines) if label_lines else "（无可用标签，请直接返回空标签结果）"
    parent_desc = f"【{parent_name}】类别" if parent_name else "【顶级】类别"

    return (
        f"你是企业客户画像标签助手，需要基于会话聊天记录为用户选择最匹配的{parent_desc}系统标签。\n"
        "标签说明：\n"
        f"{label_section}\n\n"
        "输出要求：\n"
        "1. 仅根据标签定义判断，允许选择 0、1 或多条标签。\n"
        "2. 优先使用更细粒度（子级）标签，除非只有父级可用。\n"
        "3. 输出 JSON，字段为 labelIds (int 数组) 与 reason (string 简述选择理由)。\n"
        "示例输出：{\"labelIds\": [2,7], \"reason\": \"...\"}\n"
        "4. 如无合适标签，返回空数组并说明原因。\n"
    )


def _build_history_context(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """构建用于 LLM 的聊天历史上下文"""
    history: List[Dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        if role not in ("user", "assistant", "system"):
            role = "assistant" if role == "agent" else "user"
        ts = msg.get("datetime") or ""
        content = msg.get("content") or ""
        history.append({
            "role": role,
            "content": f"[{ts}] {content}".strip(),
        })
    return history


def _safe_json_loads(payload: Any) -> Optional[Dict[str, Any]]:
    """安全地解析 JSON 字符串或对象"""
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            logger.warning("解析 JSON 失败: %s", payload)
    return None


def _make_json_serializable(obj: Any) -> Any:
    """将对象转换为可 JSON 序列化的格式"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    # 对于不可序列化的对象，转换为字符串
    try:
        # 尝试调用对象的 __dict__ 或转换为字符串
        if hasattr(obj, "__dict__"):
            return _make_json_serializable(obj.__dict__)
        return str(obj)
    except Exception:
        return str(obj)


def _extract_label_ids(model_response: Optional[Dict[str, Any]]) -> List[int]:
    """从 LLM 响应中提取标签 ID 列表"""
    if not isinstance(model_response, dict):
        return []
    candidates = model_response.get("labelIds") or model_response.get("labels") or []
    label_ids: List[int] = []
    if isinstance(candidates, (list, tuple, set)):
        for item in candidates:
            try:
                value = int(item)
                if value not in label_ids:
                    label_ids.append(value)
            except (ValueError, TypeError):
                continue
    return label_ids


def _update_customer_labels(app_id: str, user_id: str, label_ids: Sequence[int]) -> Dict[str, Any]:
    """更新客户标签到 CRM 系统"""
    params = {
        "appId": app_id,
        "labelIds": list(label_ids),
        "type": 1,
        "outerId": user_id,
    }
    try:
        raw = call_tool("update_customer_label", params=params)
        
        # 处理 MCP 返回格式：可能是 CallToolResult 对象或字典
        result_data = None
        if hasattr(raw, "structuredContent"):
            # 优先使用 structuredContent（标准 MCP 格式）
            result_data = raw.structuredContent
        elif hasattr(raw, "content"):
            # 尝试从 content 中提取
            content = raw.content
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                if hasattr(first_item, "text"):
                    try:
                        result_data = json.loads(first_item.text) if isinstance(first_item.text, str) else first_item.text
                    except (json.JSONDecodeError, TypeError):
                        pass
        elif isinstance(raw, dict):
            # 字典格式：优先 structuredContent，否则使用原始字典
            if "structuredContent" in raw:
                result_data = raw["structuredContent"]
            elif "content" in raw:
                content_list = raw.get("content", [])
                if isinstance(content_list, list) and len(content_list) > 0:
                    first_item = content_list[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        try:
                            text_content = first_item["text"]
                            result_data = json.loads(text_content) if isinstance(text_content, str) else text_content
                        except (json.JSONDecodeError, TypeError):
                            pass
            else:
                result_data = raw
        elif isinstance(raw, str):
            # 字符串格式：尝试解析 JSON
            try:
                result_data = json.loads(raw)
            except json.JSONDecodeError:
                result_data = {"raw_text": raw}
        else:
            # 无法解析的对象，转换为字符串描述
            result_data = {"raw_type": type(raw).__name__, "raw_str": str(raw)}
        
        # 确保返回的是可序列化的字典
        if isinstance(result_data, dict):
            return result_data
        else:
            return {"result": str(result_data)}
            
    except Exception as exc:
        logger.error("更新客户标签失败: %s", exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}


def _label_single_parent_group(
    parent_name: Optional[str],
    labels: List[Dict[str, Any]],
    history_context: List[Dict[str, str]],
    session_state: Dict[str, Any],
) -> Dict[str, Any]:
    """为单个父类组生成标签"""
    try:
        prompt = _build_label_prompt_for_parent(parent_name, labels)
        user_instruction = (
            "请基于完整聊天记录判断该用户的兴趣、需求或画像特征，并给出最合适的标签 ID 列表。"
        )
        
        llm_output = llm_generic(
            full_prompt=prompt,
            user_input=user_instruction,
            history_context=history_context,
            session_state=session_state,
            force_json=True,
            input_label=f"标签任务-{parent_name or '顶级'}",
        )
        
        parsed = _safe_json_loads(llm_output)
        label_ids = _extract_label_ids(parsed)
        
        return {
            "parent": parent_name,
            "label_ids": label_ids,
            "llm_raw": parsed,
            "success": True,
        }
    except Exception as exc:
        logger.error(f"父类组 {parent_name} 标签生成失败: %s", exc, exc_info=True)
        return {
            "parent": parent_name,
            "label_ids": [],
            "llm_raw": None,
            "success": False,
            "error": str(exc),
        }


def _label_conversation_once(conversation_id: str, session_id: str, user_id: str, app_id: Optional[str] = None, message_limit: int = 50) -> Dict[str, Any]:
    """为单个会话生成标签（内部实现）"""
    # 如果未提供 app_id，尝试从会话数据中获取
    if not app_id:
        fetched_app_id = get_app_id_from_session(conversation_id, session_id, user_id)
        if fetched_app_id:
            app_id = fetched_app_id
            logger.info(f"从会话数据中获取到 app_id: {app_id}")
        else:
            # 如果从会话数据中也获取不到，返回状态而不是抛出异常
            logger.warning(f"无法获取 app_id：未提供且从会话数据中获取失败 (conversation_id={conversation_id}, session_id={session_id}, user_id={user_id})")
            return {"status": "no_app_id", "conversation_id": conversation_id, "session_id": session_id, "user_id": user_id}
    
    labels = _get_valid_labels(app_id)
    if not labels:
        return {"status": "no_labels_available"}

    history_json = list_chat_messages(conversation_id, session_id, user_id, limit=message_limit)
    messages = json.loads(history_json) if history_json else []
    if not messages:
        return {"status": "no_messages"}

    history_context = _build_history_context(messages)

    convo_meta_raw = get_conversation_with_cache(user_id=user_id, session_id=session_id)
    session_state = _safe_json_loads(convo_meta_raw) or {}
    session_state.update({
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": user_id,
    })

    # 按父类分组标签
    label_groups = _group_labels_by_parent(labels)
    logger.info(f"标签按父类分组，共 {len(label_groups)} 个父类组")

    # 并行处理所有父类组
    executor = get_thread_pool(namespace="label_conversation")
    future_to_parent = {}
    
    for parent_name, parent_labels in label_groups.items():
        future = executor.submit(
            _label_single_parent_group,
            parent_name=parent_name,
            labels=parent_labels,
            history_context=history_context,
            session_state=session_state,
        )
        future_to_parent[future] = parent_name

    # 收集所有结果
    all_label_ids = []
    all_results = []
    failed_groups = []
    
    for future in as_completed(future_to_parent.keys()):
        parent_name = future_to_parent[future]
        try:
            result = future.result()
            all_results.append(result)
            if result.get("success"):
                all_label_ids.extend(result.get("label_ids", []))
            else:
                failed_groups.append(parent_name)
                logger.warning(f"父类组 {parent_name} 标签生成失败")
        except Exception as exc:
            logger.exception(f"父类组 {parent_name} 处理异常: %s", exc)
            failed_groups.append(parent_name)

    # 去重 label_ids
    all_label_ids = list(set(all_label_ids))

    if not all_label_ids:
        return {
            "status": "no_prediction",
            "llm_raw": all_results,
            "failed_groups": failed_groups,
        }

    update_result = _update_customer_labels(app_id=app_id, user_id=user_id, label_ids=all_label_ids)
    return {
        "status": "updated",
        "label_ids": all_label_ids,
        "llm_raw": all_results,
        "update_result": update_result,
        "failed_groups": failed_groups,
    }


def compute_conversation_labels(conversation_id: str, session_id: str, user_id: str, app_id: Optional[str] = None, message_limit: int = 50) -> Dict[str, Any]:
    """为会话生成标签并更新到 CRM 系统"""
    result = _label_conversation_once(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        app_id=app_id,
        message_limit=message_limit,
    )
    logger.info(
        "标签任务完成 conversation=%s session=%s user=%s result=%s",
        conversation_id,
        session_id,
        user_id,
        result.get("status"),
    )
    # 确保返回结果可以 JSON 序列化
    return _make_json_serializable(result)


@mcp.tool(description="Generate customer labels for a single conversation and push them to the CRM system. If app_id is not provided, it will be automatically fetched from session data. If fetch fails, an error will be raised.")
def tag_conversation(
    conversation_id: str,
    session_id: str,
    user_id: str,
    app_id: Optional[str] = None,
    message_limit: int = 50,
) -> str:
    """为单个会话生成标签并更新到 CRM 系统"""
    result = compute_conversation_labels(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        app_id=app_id,
        message_limit=message_limit,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(description="Scan score table conversations in batches and assign labels via LLM, similar to scorer.py.")
def scan_label_conversations(
    app_id: Optional[str] = None,
    before_datetime: str = None,
    after_datetime: str = None,
    limit: int = 100,
    message_limit: int = 50,
) -> str:
    """批量扫描 score 表中的会话记录并生成标签"""
    total_processed = 0
    success = 0
    failed = 0
    batch_number = 0
    
    # 确定时间窗口
    # 如果未指定时间，默认扫描过去 1 小时的数据（(Now-1h, Now]）
    if before_datetime is None and after_datetime is None:
        # 需要显式导入 BEIJING_TZ，因为在函数内部可能访问不到顶层的导入（如果有的话，或者顶层没导入）
        from core.time_utils import BEIJING_TZ
        now = datetime.now(BEIJING_TZ)
        before_datetime = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        one_hour_ago = now - timedelta(hours=1)
        after_datetime = one_hour_ago.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger.info("未指定时间范围，默认查询过去1小时的记录（%s ~ %s）", after_datetime, before_datetime)
    
    # 初始化游标为起始时间
    last_datetime = after_datetime

    # 限制并发数，避免数据库连接池耗尽
    # 连接池大小为 20，留一些余量给其他操作，设置并发数为 15
    executor = get_thread_pool(namespace="label_conversation", max_workers=15)

    while True:
        batch_number += 1
        result_json = scan_chat_conversations(limit=limit, last_datetime=last_datetime, before_datetime=before_datetime)
        result_data = json.loads(result_json) if isinstance(result_json, str) else result_json
        records = result_data.get("records", [])
        has_more = bool(result_data.get("has_more"))
        last_datetime = result_data.get("last_datetime")

        if not records:
            logger.info("没有更多记录需要打标签，总计处理 %s 条", total_processed)
            break

        futures = []
        record_info = []
        for i, record in enumerate(records, 1):
            futures.append(
                executor.submit(
                    compute_conversation_labels,
                    conversation_id=record.get("conversation_id"),
                    session_id=record.get("session_id"),
                    user_id=record.get("user_id"),
                    app_id=app_id,  # 如果为 None，会在内部自动从会话数据获取
                    message_limit=message_limit,
                )
            )
            record_info.append(record)

        batch_success = 0
        batch_failed = 0
        for future in as_completed(futures):
            try:
                future.result()
                batch_success += 1
            except Exception as exc:
                batch_failed += 1
                # 特别处理连接池耗尽错误
                if "connection pool exhausted" in str(exc) or "PoolError" in str(type(exc).__name__):
                    logger.error("标签任务失败: 数据库连接池耗尽，请检查连接池配置和并发数设置")
                    logger.error("建议：1) 增加数据库连接池大小 2) 减少并发任务数 3) 检查是否有连接泄漏")
                else:
                    logger.exception("标签任务失败: %s", exc)

        success += batch_success
        failed += batch_failed
        total_processed += len(records)

        logger.info(
            "批次 %s 完成: 成功 %s 条, 失败 %s 条 (累计成功 %s, 失败 %s)",
            batch_number,
            batch_success,
            batch_failed,
            success,
            failed,
        )

        if not has_more:
            logger.info("label 扫描已到最后一批, 总计 %s 条", total_processed)
            break

    return json.dumps({
        "status": "completed",
        "total_processed": total_processed,
        "batches_processed": batch_number,
        "before_datetime": before_datetime,
        "success": success,
        "failed": failed,
    }, ensure_ascii=False)


def _scheduled_label_task(app_id: Optional[str] = None):
    """
    定时执行的标签任务，扫描所有会话并生成标签。
    
    参数:
        app_id: 应用 ID。如果为 None，每个会话会从会话数据中自动获取自己的 app_id。
    """
    global _label_task_running
    logger.info("=" * 60)
    logger.info("[%s] 定时标签任务被触发", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(f"  传入的 app_id: {app_id}")
    logger.info(f"  调度器运行状态: {'运行中' if scheduler.running else '未运行'}")
    
    with _label_task_lock:
        if _label_task_running:
            logger.warning("[%s] 标签任务正在运行，跳过本次执行", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            logger.info("=" * 60)
            return
        _label_task_running = True
        logger.info("  任务状态已设置为运行中")

    try:
        logger.info("[%s] 开始执行定时标签任务", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        result = scan_label_conversations(app_id=app_id)  # 传入 None 时，每个会话会自己获取 app_id
        result_data = json.loads(result) if isinstance(result, str) else result
        logger.info("[%s] 定时标签任务完成", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("  处理结果: 总计 %s 条, 成功 %s 条, 失败 %s 条, 批次 %s",
                   result_data.get("total_processed", 0),
                   result_data.get("success", 0),
                   result_data.get("failed", 0),
                   result_data.get("batches_processed", 0))
        logger.info("  时间范围: %s", result_data.get("before_datetime"))
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 定时标签任务完成: 总计 {result_data.get('total_processed', 0)} 条, 成功 {result_data.get('success', 0)} 条, 失败 {result_data.get('failed', 0)} 条")
    except Exception as exc:
        logger.exception("定时标签任务执行出错: %s", exc)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 定时标签任务执行出错: {exc}")
    finally:
        with _label_task_lock:
            _label_task_running = False
            logger.info("  任务状态已重置为未运行")
        logger.info("=" * 60)


def start_label_scheduler(cron_expression: str = "0 * * * *", app_id: Optional[str] = None):
    """启动定时标签扫描任务，每小时执行一次"""
    try:
        # 创建每小时执行一次的定时任务
        job_id = new_scheduler(cron_expression, _scheduled_label_task, job_id=_LABEL_JOB_ID, app_id=app_id)
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 定时标签任务已启动，任务ID: {job_id}")
        logger.info("任务计划: 每小时执行一次 scan_label_conversations 函数")
        return job_id
    except Exception as e:
        logger.error(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 启动定时标签任务失败: {e}")
        return None


@mcp.tool(description="Start scheduled label scan task (hourly, similar to scorer scheduler). Each conversation will automatically fetch its own app_id if not provided.")
def start_label_scheduler_tool(cron_expression: str = "0 * * * *", app_id: Optional[str] = None) -> str:
    """启动定时标签扫描任务（MCP 工具）"""
    try:
        logger.info("=" * 60)
        logger.info("收到启动标签调度器请求")
        logger.info(f"  Cron 表达式: {cron_expression}")
        logger.info(f"  App ID: {app_id}")
        logger.info(f"  任务 ID: {_LABEL_JOB_ID}")
        logger.info(f"  调度器当前状态: {'运行中' if scheduler.running else '未运行'}")
        logger.info(f"  调度器当前任务数: {len(scheduler.get_jobs())}")
        
        job_id = new_scheduler(cron_expression, _scheduled_label_task, job_id=_LABEL_JOB_ID, app_id=app_id)
        
        # 获取任务信息
        jobs = list_schedulers()
        label_jobs = [job for job in jobs if job['id'] == _LABEL_JOB_ID]
        next_run = label_jobs[0]['next_run_time'] if label_jobs else None
        
        logger.info(f"  任务创建成功，任务 ID: {job_id}")
        logger.info(f"  下次执行时间: {next_run}")
        logger.info(f"  调度器当前任务数: {len(scheduler.get_jobs())}")
        logger.info("=" * 60)
        
        return json.dumps({
            "status": "started",
            "job_id": job_id,
            "cron": cron_expression,
            "app_id": app_id,
            "next_run_time": next_run,
        }, ensure_ascii=False)
    except Exception as exc:
        logger.exception("=" * 60)
        logger.exception("启动标签定时任务失败: %s", exc)
        logger.exception("=" * 60)
        return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)


@mcp.tool(description="Get scheduled label scan status.")
def get_label_scheduler_status() -> str:
    """获取定时标签扫描任务的状态"""
    try:
        logger.info("查询标签调度器状态...")
        logger.info(f"  调度器运行状态: {'运行中' if scheduler.running else '未运行'}")
        logger.info(f"  任务执行中: {_label_task_running}")
        logger.info(f"  调度器总任务数: {len(scheduler.get_jobs())}")
        
        jobs = list_schedulers()
        label_jobs = [job for job in jobs if job['id'] == _LABEL_JOB_ID]
        
        logger.info(f"  标签任务数量: {len(label_jobs)}")
        for job in label_jobs:
            logger.info(f"    任务 ID: {job['id']}, 下次执行: {job.get('next_run_time')}, 触发器: {job.get('trigger')}")
        
        status = {
            "scheduler_running": scheduler.running,
            "task_running": _label_task_running,
            "scheduled_jobs": label_jobs,
            "has_scheduled_task": len(label_jobs) > 0,
            "next_run_time": label_jobs[0]['next_run_time'] if label_jobs else None,
            "total_jobs": len(scheduler.get_jobs()),
        }
        return json.dumps(status, ensure_ascii=False)
    except Exception as exc:
        logger.exception("查询调度器状态失败: %s", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool(description="Stop scheduled label scan task.")
def stop_label_scheduler() -> str:
    """停止定时标签扫描任务"""
    try:
        logger.info("=" * 60)
        logger.info("收到停止标签调度器请求")
        logger.info(f"  任务 ID: {_LABEL_JOB_ID}")
        logger.info(f"  调度器当前状态: {'运行中' if scheduler.running else '未运行'}")
        logger.info(f"  停止前调度器任务数: {len(scheduler.get_jobs())}")
        
        success = remove_scheduler(_LABEL_JOB_ID)
        
        logger.info(f"  停止操作结果: {'成功' if success else '失败（未找到任务）'}")
        logger.info(f"  停止后调度器任务数: {len(scheduler.get_jobs())}")
        logger.info("=" * 60)
        
        if success:
            return json.dumps({"status": "success", "message": "定时标签任务已停止"}, ensure_ascii=False)
        return json.dumps({"status": "failed", "message": "未找到定时标签任务"}, ensure_ascii=False)
    except Exception as exc:
        logger.exception("=" * 60)
        logger.exception("停止标签定时任务失败: %s", exc)
        logger.exception("=" * 60)
        return json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False)


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("正在启动 Tag MCP 服务器...")
        logger.info(f"服务器配置: host=0.0.0.0, port=3005")
        logger.info(f"调度器状态: {'运行中' if scheduler.running else '未运行'}")
        logger.info(f"当前调度器任务数: {len(scheduler.get_jobs())}")
        
        # 列出所有已注册的工具（手动列出，因为 list_tools() 是异步方法）
        registered_tools = [
            "tag_conversation",
            "scan_label_conversations", 
            "start_label_scheduler_tool",
            "get_label_scheduler_status",
            "stop_label_scheduler"
        ]
        logger.info(f"已注册的 MCP 工具数量: {len(registered_tools)}")
        logger.info(f"工具列表: {', '.join(registered_tools)}")
        
        # 启动定时标签任务
        logger.info("=" * 60)
        start_label_scheduler("*/5 * * * *")
        logger.info("=" * 60)
        
        logger.info("=" * 60)
        logger.info("服务器启动成功，等待连接...")
        logger.info("按 Ctrl+C 停止服务器")
        logger.info("=" * 60)
        
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("=" * 60)
        logger.info("收到中断信号，正在优雅关闭服务器...")
        logger.info(f"关闭前调度器状态: {'运行中' if scheduler.running else '未运行'}")
        logger.info("服务器已关闭")
        logger.info("=" * 60)
        print("服务器已关闭")
    except Exception as e:
        logger.exception("=" * 60)
        logger.exception("服务器运行出错: %s", e)
        logger.exception("=" * 60)
        raise

