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
from src.tools.porstgreDB_tools import (
    get_app_id_from_session,
    get_conversation_with_cache,
    list_chat_messages,
    scan_chat_conversations,
)
from Responder import llm_generic
from mcp.server.fastmcp import FastMCP

from src.utils.llm_utils import get_default_llm
from langchain.agents import create_agent
from src.tools.redis_tools import redis_get, redis_set



# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



llm_model = get_default_llm()



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
            "alias": node.get("alias") or node.get("tag_alias") or [],  # 支持别名字段
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


def _build_label_prompt_for_parent(
    parent_name: Optional[str],
    labels: Sequence[Dict[str, Any]],
    block_log: Optional[List[Dict[str, Any]]] = None,
    manual_labels: Optional[List[int]] = None,
) -> str:
    """
    为单个父类组构建 prompt
    
    Args:
        parent_name: 父类名称
        labels: 标签列表
        block_log: block_log 列表（被手动删除的标签记录）
        manual_labels: 手动添加的标签ID列表
    """
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
            
            # 标记手动标签
            is_manual = manual_labels and int(label_id) in manual_labels
            manual_mark = " [手动标签-不可移除]" if is_manual else ""
            
            label_lines.append(f"- ID={label_id} | {label_path} | {description}{manual_mark}")
        except Exception:
            continue

    label_section = "\n".join(label_lines) if label_lines else "（无可用标签，请直接返回空标签结果）"
    parent_desc = f"【{parent_name}】类别" if parent_name else "【顶级】类别"
    
    # 构建 block_log 提示信息
    block_log_section = ""
    if block_log:
        blocked_label_ids = [entry.get("label_id") for entry in block_log if entry.get("label_id")]
        if blocked_label_ids:
            block_log_section = (
                f"\n\n【重要提示：人工删除记录】\n"
                f"以下标签曾经被用户手动删除过：{', '.join(blocked_label_ids)}\n"
                f"请谨慎判断是否应该再次选择这些标签。只有在会话内容明确支持的情况下，才可以选择这些标签。\n"
                f"如果会话内容不足以支持，请不要选择这些标签。\n"
            )
    
    # 构建手动标签提示信息
    manual_labels_section = ""
    if manual_labels:
        manual_labels_str = ", ".join(str(lid) for lid in manual_labels)
        manual_labels_section = (
            f"\n\n【重要提示：手动标签保护】\n"
            f"以下标签是用户手动添加的（标签ID: {manual_labels_str}），这些标签必须保留，不能被移除。\n"
            f"即使这些标签不在你的推荐列表中，也必须包含在最终输出中。\n"
        )

    return (
        f"你是企业客户画像标签助手，需要基于会话聊天记录为用户选择最匹配的{parent_desc}系统标签。\n"
        "标签说明：\n"
        f"{label_section}\n"
        f"{block_log_section}"
        f"{manual_labels_section}"
        "\n输出要求：\n"
        "1. 仅根据标签定义判断，允许选择 0、1 或多条标签。\n"
        "2. 优先使用更细粒度（子级）标签，除非只有父级可用。\n"
        "3. 输出 JSON，字段为 labelIds (int 数组) 与 reason (string 简述选择理由)。\n"
        "4. 必须包含所有手动标签（标记为[手动标签-不可移除]的标签），即使它们不在你的推荐中。\n"
        "5. 对于曾经被删除的标签，请谨慎判断，只有在会话内容明确支持时才选择。\n"
        "示例输出：{\"labelIds\": [2,7], \"reason\": \"...\"}\n"
        "6. 如无合适标签，返回空数组并说明原因（但手动标签仍必须包含）。\n"
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


def _build_candidate_recall_prompt(labels: List[Dict[str, Any]]) -> str:
    """构建候选集召回的 prompt，包含所有标签的ID、名称、别名、描述"""
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
            alias_list = item.get("alias") or []
            if isinstance(alias_list, str):
                alias_list = [alias_list]
            elif not isinstance(alias_list, list):
                alias_list = []
            alias_str = "、".join([str(a).strip() for a in alias_list if a]) if alias_list else "无"
            parent = item.get("parent")
            label_path = f"{parent}>{name}" if parent else name
            
            label_lines.append(
                f"- ID={label_id} | 名称={label_path} | 别名={alias_str} | 描述={description}"
            )
        except Exception as exc:
            logger.warning(f"构建标签信息时出错: {exc}")
            continue

    label_section = "\n".join(label_lines) if label_lines else "（无可用标签）"
    
    return (
        "你是企业客户画像标签召回助手，需要基于会话聊天记录从大量标签中筛选出最相关的候选标签。\n\n"
        "标签列表：\n"
        f"{label_section}\n\n"
        "任务要求：\n"
        "1. 仔细分析会话记录中的用户表达、需求、兴趣点等信息。\n"
        "2. 根据标签的**名称**、**别名**（关键词）和**描述**进行匹配判断。\n"
        "3. 如果标签的别名、名称或描述与会话内容有语义关联，则将该标签纳入候选集。\n"
        "4. 候选集应该包含所有可能相关的标签，宁可多召回也不要漏掉（后续会进行精排）。\n"
        "5. 输出 JSON 格式，字段为 candidateLabelIds (int 数组) 和 reason (string 简述召回理由)。\n"
        "示例输出：{\"candidateLabelIds\": [1, 5, 12, 23], \"reason\": \"根据会话中的价格敏感、预算不足等关键词，召回相关标签\"}\n"
        "6. 如果没有任何相关标签，返回空数组并说明原因。\n"
    )


def recall_candidate_labels(
    labels: List[Dict[str, Any]],
    history_context: List[Dict[str, str]],
    session_state: Optional[Dict[str, Any]] = None,
    max_candidates: int = 30,
) -> Dict[str, Any]:
    """
    阶段二：候选集召回
    
    目的：解决商户标签过多（如 >50个）导致 LLM 上下文溢出或精度下降的问题。
    基于标签名/别名/标签描述，以及会话记录给LLM做匹配判断，返回候选标签ID列表。
    
    Args:
        labels: 所有标签列表（可能很多，>50个）
        history_context: 会话记录上下文（由 _build_history_context 构建）
        session_state: 会话状态信息（可选）
        max_candidates: 最大候选标签数量，默认30个
    
    Returns:
        Dict包含:
            - candidate_label_ids: List[int] 候选标签ID列表
            - reason: str 召回理由
            - success: bool 是否成功
            - llm_raw: Dict LLM原始响应
    """
    if not labels:
        logger.warning("标签列表为空，无法进行候选集召回")
        return {
            "candidate_label_ids": [],
            "reason": "标签列表为空",
            "success": False,
            "llm_raw": None,
        }
    
    if not history_context:
        logger.warning("会话记录为空，无法进行候选集召回")
        return {
            "candidate_label_ids": [],
            "reason": "会话记录为空",
            "success": False,
            "llm_raw": None,
        }
    
    try:
        # 构建召回 prompt
        prompt = _build_candidate_recall_prompt(labels)
        user_instruction = (
            f"请基于完整聊天记录，从 {len(labels)} 个标签中筛选出最相关的候选标签（最多 {max_candidates} 个）。"
            "重点关注标签的别名、名称和描述与会话内容的语义关联。"
        )
        
        # 参数验证
        if not isinstance(history_context, list):
            logger.error("history_context 参数类型错误: %s", type(history_context))
            return {
                "candidate_label_ids": [],
                "reason": "参数验证失败: history_context 必须是列表类型",
                "success": False,
                "llm_raw": None,
            }
            
        # 使用 llm_generic 调用大模型
        llm_output = None
        try:
            llm_output = llm_generic(
                full_prompt=prompt,
                user_input=user_instruction,
                history_context=history_context,
                session_state=session_state or {},
                force_json=True,
                input_label="候选集召回",
            )
        except Exception as llm_exc:
            logger.error(f"调用 llm_generic 失败: {str(llm_exc)}")
            return {
                "candidate_label_ids": [],
                "reason": f"调用大模型失败: {str(llm_exc)}",
                "success": False,
                "llm_raw": None,
            }
        
        # 检查 llm_output 是否为 None
        if llm_output is None:
            logger.warning("llm_generic 返回了 None")
            return {
                "candidate_label_ids": [],
                "reason": "大模型返回结果为空",
                "success": False,
                "llm_raw": None,
            }
        
        # 解析 LLM 响应
        parsed = _safe_json_loads(llm_output)
        if not parsed:
            logger.warning("LLM 响应解析失败，原始输出: %s", llm_output)
            return {
                "candidate_label_ids": [],
                "reason": "LLM响应解析失败",
                "success": False,
                "llm_raw": llm_output,
            }
        
        # 提取候选标签ID
        candidate_ids = parsed.get("candidateLabelIds") or parsed.get("candidate_label_ids") or []
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        
        # 转换为整数列表并去重
        candidate_label_ids: List[int] = []
        for item in candidate_ids:
            try:
                label_id = int(item)
                if label_id not in candidate_label_ids:
                    candidate_label_ids.append(label_id)
            except (ValueError, TypeError):
                continue
        
        # 限制最大候选数量
        if len(candidate_label_ids) > max_candidates:
            logger.info(f"候选标签数量 {len(candidate_label_ids)} 超过限制 {max_candidates}，进行截断")
            candidate_label_ids = candidate_label_ids[:max_candidates]
        
        reason = parsed.get("reason") or parsed.get("reasoning") or "未提供理由"

        logger.info(f"reason: {reason}")
        
        logger.info(
            f"候选集召回完成: 从 {len(labels)} 个标签中召回 {len(candidate_label_ids)} 个候选标签"
        )
        
        return {
            "candidate_label_ids": candidate_label_ids,
            "reason": reason,
            "success": True,
            "llm_raw": parsed,
        }
        
    except Exception as exc:
        logger.error(f"候选集召回失败: %s", exc, exc_info=True)
        return {
            "candidate_label_ids": [],
            "reason": f"召回过程异常: {str(exc)}",
            "success": False,
            "llm_raw": None,
            "error": str(exc),
        }


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


def _get_block_log_key(app_id: str, user_id: str) -> str:
    """生成 block_log 的 Redis key"""
    return f"label_block_log:{app_id}:{user_id}"


def record_label_block_log(app_id: str, user_id: str, label_id: int, deleted_by: Optional[str] = None) -> bool:
    """
    记录用户手动删除标签的操作到 block_log
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
        label_id: 被删除的标签ID
        deleted_by: 删除操作者（可选，如 "user" 或 "admin"）
    
    Returns:
        是否成功记录
    """
    try:
        key = _get_block_log_key(app_id, user_id)
        existing_log = redis_get(key)
        
        # 解析现有的 block_log
        block_log = []
        if existing_log:
            try:
                block_log = json.loads(existing_log)
                if not isinstance(block_log, list):
                    block_log = []
            except (json.JSONDecodeError, TypeError):
                block_log = []
        
        # 检查是否已存在该标签的记录
        label_id_str = str(label_id)
        existing_entry = next((entry for entry in block_log if entry.get("label_id") == label_id_str), None)
        
        if existing_entry:
            # 更新现有记录的时间戳
            existing_entry["deleted_at"] = datetime.now(timezone.utc).isoformat()
            if deleted_by:
                existing_entry["deleted_by"] = deleted_by
        else:
            # 添加新记录
            block_log.append({
                "label_id": label_id_str,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "deleted_by": deleted_by or "user",
            })
        
        # 保存到 Redis（不过期，永久保存）
        redis_set(key, json.dumps(block_log, ensure_ascii=False))
        logger.info(f"记录 block_log: app_id={app_id}, user_id={user_id}, label_id={label_id}")
        return True
        
    except Exception as exc:
        logger.error(f"记录 block_log 失败: %s", exc, exc_info=True)
        return False


def get_label_block_log(app_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    获取用户的 block_log（被手动删除的标签列表）
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
    
    Returns:
        block_log 列表，格式: [{"label_id": "123", "deleted_at": "...", "deleted_by": "user"}, ...]
    """
    try:
        key = _get_block_log_key(app_id, user_id)
        log_data = redis_get(key)
        
        if not log_data:
            return []
        
        block_log = json.loads(log_data)
        if not isinstance(block_log, list):
            return []
        
        return block_log
        
    except Exception as exc:
        logger.error(f"获取 block_log 失败: %s", exc, exc_info=True)
        return []


def get_customer_manual_labels(app_id: str, user_id: str) -> List[int]:
    """
    获取客户的手动添加的标签ID列表
    
    注意：此函数需要根据实际的 CRM 系统接口来实现。
    如果 CRM 系统支持区分手动标签和 AI 标签，应该通过 MCP 工具获取。
    目前返回空列表，需要根据实际情况实现。
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
    
    Returns:
        手动标签ID列表
    """
    try:
        # TODO: 根据实际 CRM 系统接口实现
        # 例如：通过 call_tool("get_customer_manual_labels", params={"appId": app_id, "outerId": user_id})
        # 如果 CRM 系统不支持区分，可以返回空列表，表示所有标签都可能是手动的
        # 这种情况下，需要在更新时保留所有现有标签
        
        # 示例实现（需要根据实际情况修改）：
        # raw = call_tool("get_customer_labels", params={"appId": app_id, "outerId": user_id, "type": "manual"})
        # data = _extract_mcp_data(raw)
        # if isinstance(data, dict) and "labelIds" in data:
        #     return [int(lid) for lid in data["labelIds"] if str(lid).isdigit()]
        
        return []
    except Exception as exc:
        logger.error(f"获取手动标签失败: %s", exc, exc_info=True)
        return []


def get_customer_current_labels(app_id: str, user_id: str) -> List[int]:
    """
    获取客户当前的标签ID列表（包括手动和AI标签）
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
    
    Returns:
        当前标签ID列表
    """
    try:
        # TODO: 根据实际 CRM 系统接口实现
        # 例如：通过 call_tool("get_customer_labels", params={"appId": app_id, "outerId": user_id})
        # raw = call_tool("get_customer_labels", params={"appId": app_id, "outerId": user_id})
        # data = _extract_mcp_data(raw)
        # if isinstance(data, dict) and "labelIds" in data:
        #     return [int(lid) for lid in data["labelIds"] if str(lid).isdigit()]
        
        return []
    except Exception as exc:
        logger.error(f"获取当前标签失败: %s", exc, exc_info=True)
        return []


def _update_customer_labels(
    app_id: str,
    user_id: str,
    label_ids: Sequence[int],
    manual_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    更新客户标签到 CRM 系统
    
    注意：此函数会确保手动标签不会被移除（人工 > AI 原则）
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
        label_ids: 要设置的标签ID列表
        manual_labels: 手动添加的标签ID列表（这些标签必须保留）
    """
    # 确保手动标签被包含（人工 > AI 原则）
    final_label_ids = list(set(label_ids))
    if manual_labels:
        for manual_label_id in manual_labels:
            if manual_label_id not in final_label_ids:
                final_label_ids.append(manual_label_id)
                logger.info(f"保留手动标签 {manual_label_id}，确保不被移除")
    
    params = {
        "appId": app_id,
        "labelIds": final_label_ids,
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
    block_log: Optional[List[Dict[str, Any]]] = None,
    manual_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    为单个父类组生成标签
    
    Args:
        parent_name: 父类名称
        labels: 标签列表
        history_context: 会话历史上下文
        session_state: 会话状态
        block_log: block_log 列表（被手动删除的标签记录）
        manual_labels: 手动添加的标签ID列表
    """
    try:
        prompt = _build_label_prompt_for_parent(
            parent_name=parent_name,
            labels=labels,
            block_log=block_log,
            manual_labels=manual_labels,
        )
        user_instruction = (
            "请基于完整聊天记录判断该用户的兴趣、需求或画像特征，并给出最合适的标签 ID 列表。"
            "请特别注意手动标签和曾经被删除的标签的处理规则。"
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
        
        # 确保手动标签被包含（人工 > AI 原则）
        if manual_labels:
            for manual_label_id in manual_labels:
                if manual_label_id not in label_ids:
                    label_ids.append(manual_label_id)
                    logger.info(f"自动添加手动标签 {manual_label_id} 到输出结果")
        
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
            "label_ids": manual_labels or [],  # 即使失败，也要返回手动标签
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

    # 获取 block_log（被手动删除的标签记录）
    block_log = get_label_block_log(app_id=app_id, user_id=user_id)
    if block_log:
        logger.info(f"获取到 block_log，包含 {len(block_log)} 条删除记录")
    
    # 获取手动标签（用户手动添加的标签，不能被移除）
    manual_labels = get_customer_manual_labels(app_id=app_id, user_id=user_id)
    if manual_labels:
        logger.info(f"获取到 {len(manual_labels)} 个手动标签: {manual_labels}")

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
            block_log=block_log,  # 传递 block_log
            manual_labels=manual_labels,  # 传递手动标签
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
    
    # 确保手动标签被包含（人工 > AI 原则）
    if manual_labels:
        for manual_label_id in manual_labels:
            if manual_label_id not in all_label_ids:
                all_label_ids.append(manual_label_id)
                logger.info(f"自动添加手动标签 {manual_label_id} 到最终结果")

    if not all_label_ids:
        return {
            "status": "no_prediction",
            "llm_raw": all_results,
            "failed_groups": failed_groups,
        }

    # 更新客户标签，确保手动标签不会被移除
    update_result = _update_customer_labels(
        app_id=app_id,
        user_id=user_id,
        label_ids=all_label_ids,
        manual_labels=manual_labels,  # 传递手动标签，确保不被移除
    )
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


@mcp.tool(description="Record a label deletion event to block_log when a user manually deletes an AI-recommended label. This implements the cooldown mechanism for data governance.")
def record_label_deletion(
    app_id: str,
    user_id: str,
    label_id: int,
    deleted_by: Optional[str] = None,
) -> str:
    """
    记录用户手动删除标签的操作到 block_log
    
    当用户手动删除了 AI 推荐的标签时，调用此函数记录删除事件。
    之后在 LLM 打标签时，会告知 AI 该标签曾经被删除过，让 AI 进行判断。
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
        label_id: 被删除的标签ID
        deleted_by: 删除操作者（可选，如 "user" 或 "admin"）
    
    Returns:
        JSON 格式的结果
    """
    success = record_label_block_log(
        app_id=app_id,
        user_id=user_id,
        label_id=label_id,
        deleted_by=deleted_by,
    )
    
    result = {
        "success": success,
        "app_id": app_id,
        "user_id": user_id,
        "label_id": label_id,
        "message": "已记录标签删除事件到 block_log" if success else "记录失败",
    }
    
    return json.dumps(result, ensure_ascii=False)


@mcp.tool(description="Get the block_log (deletion history) for a customer. Returns labels that were manually deleted by users.")
def get_customer_block_log(
    app_id: str,
    user_id: str,
) -> str:
    """
    获取客户的 block_log（被手动删除的标签记录）
    
    Args:
        app_id: 应用ID
        user_id: 用户ID
    
    Returns:
        JSON 格式的 block_log 列表
    """
    block_log = get_label_block_log(app_id=app_id, user_id=user_id)
    
    result = {
        "app_id": app_id,
        "user_id": user_id,
        "block_log": block_log,
        "count": len(block_log),
    }
    
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



# 上下文切片
def context_licing(conversation_id: str, session_id: str, user_id: str,  message_limit: int = 30) -> Dict[str, Any]:
    """
    输入：获取最近 30条会话记录。
    预处理：过滤掉无意义短语（如“你好”、“在吗”、“收到”），保留信息密度高的文本。
    :param conversation_id: 会话 ID
    :param session_id: 会话会话 ID
    :param user_id: 用户 ID
    :param message_limit: 消息限制数量
    :return: 过滤后的会话记录列表
    """
    history_json = list_chat_messages(conversation_id, session_id, user_id, limit=message_limit)
    messages = json.loads(history_json) if history_json else []

    return {
        "filtered_messages": messages
    }


    #预处理：过滤掉无意义短语（如“你好”、“在吗”、“收到”），保留信息密度高的文本。
    
    text_refinement_prompt="""你是一个文本精炼助手。你的任务是：  
            当用户提供一段文本时，识别并移除其中无实际信息量的客套话、问候语、确认语或填充性短语（例如：“你好”、“在吗”、“收到”、“好的”、“谢谢”、“嗯嗯”、“看到了”、“已阅”等），仅保留包含具体事实、请求、问题、数据、指令或实质性内容的部分。

            要求：
            1. 如果整句话都是无意义短语，返回空字符串 ""。
            2. 如果句子混合了无意义短语和有效信息，请只输出有效信息部分，保持原意不变。
            3. 不要添加任何解释、前缀或后缀，只输出过滤后的文本。
            4. 保留原始语言（如中文）和关键标点（如问号、句号）。
            5. 处理包含多个无意义短语的句子，只保留有效信息。
            6. 去除所有的emjoy表情符号。


            请处理以下输入："""


    text_refinement_agent = create_agent(
        model=llm_model,
        name="text_refinement_agent",
        system_prompt=text_refinement_prompt,
    )
    filtered_messages = []
    for msg in messages:
       
        msg['content'] = text_refinement_agent.invoke(
             {"messages": [{"role": "user", "content": msg['content']}]},
        )['messages'][-1].content
       
        if msg['content'].strip():
            filtered_messages.append(msg)
    
    return {
        "filtered_messages": filtered_messages
    }

def recall_text(conversation_id, session_id, user_id):
    fetched_app_id = get_app_id_from_session(conversation_id, session_id, user_id)
    if fetched_app_id:
        app_id = fetched_app_id
        logger.info(f"从会话数据中获取到 app_id: {app_id}")
    else:
        # 如果从会话数据中也获取不到，返回状态而不是抛出异常
        logger.warning(f"无法获取 app_id：未提供且从会话数据中获取失败 (conversation_id={conversation_id}, session_id={session_id}, user_id={user_id})")
        return {"status": "no_app_id", "conversation_id": conversation_id, "session_id": session_id, "user_id": user_id}

    labels = _get_valid_labels(app_id)

    return labels

def analyze_customer_intent(chat_history: List[Dict], candidate_tags: List[Dict]) -> List[Dict]:
    """
    分析客户意图并返回高置信度标签
    
    Args:
        chat_history: 最近20条对话历史（格式：[{"role": "user", "content": "..."}, ...]）
        candidate_tags: 候选标签列表（格式：[{"id": "1", "name": "...", "desc": "..."}, ...]）
    
    Returns:
        符合要求的JSON数组（confidence_score >= 75）
    """
    # 格式化对话历史
    formatted_history = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'agent'}: {msg['content'].strip()}"
        for msg in chat_history
    )
    
    # 格式化候选标签
    formatted_tags = json.dumps(candidate_tags, ensure_ascii=False, indent=2)
    
    # 构建优化后的提示词
    intent_prompt = f"""你是一个专业的客户意图分析专家。

任务：分析提供的对话历史，从候选标签中识别适用于该客户的标签。

约束：
1. 依据优先级：判断主要依据是标签描述（tag_desc），其次是标签名称（tag_name）。
2. 严格模式：只有当对话中有明确证据支持时，才可打标。严禁由于用户“没有提到某事”而打上负面标签。
3. 输出限制：仅输出confidence_score >= 75的标签。



候选标签列表：
{formatted_tags}

输出要求：
1. 仅输出JSON数组，每个元素包含：
   - "tag_id": 标签ID（字符串）
   - "confidence_score": 置信度分数（整数，0-100）
   - "reason": 简要说明理由（引用对话中的具体证据）
2. 仅当confidence_score >= 75时才输出。
3. 依据优先级：判断主要依据是 [tag_description]，其次是 [tag_name]。
4. 严格模式：只有当对话中有明确证据支持时，才可打标。严禁由于用户“没有提到某事”而打上负面标签。
5. 不要输出任何其他信息。
6. 如果存在互斥标签，请根据对话内容选择最符合的一个

# Data Inputs
Chat History: {formatted_history}
Candidate Tags: {formatted_tags}

# Output Format (JSON)
[
  {{"tag_id": "1",
    "confidence_score": 85,
    "reason": "用户在第3条消息明确询问'有没有折扣'，并在第5条消息表示'预算有限'。"
  }}
]

注意：请确保输出是有效的JSON，不要包含任何Markdown格式。
"""
    
    # 调用大模型
    llm_intent_agent = create_agent(
        model=llm_model,
        name="llm_intent_agent",
        system_prompt=intent_prompt,
    )

    response_text = llm_intent_agent.invoke(
        {"messages": [{"role": "user", "content": intent_prompt}]},
    )['messages'][-1].content
    
    # 解析JSON响应
    try:
        result = json.loads(response_text)
        # 过滤掉confidence_score < 75的标签
        filtered_result = [tag for tag in result if tag["confidence_score"] >= 75]
        return filtered_result
    except json.JSONDecodeError:
        # 处理模型输出异常
        return []





if __name__ == "__main__":


    

    chat_history = context_licing(conversation_id='1835', session_id='45', user_id='199')
    candidate_tags = recall_text(conversation_id='1835', session_id='45', user_id='199')
    result = analyze_customer_intent(chat_history['filtered_messages'], candidate_tags)
    print(result)
    
   
    
