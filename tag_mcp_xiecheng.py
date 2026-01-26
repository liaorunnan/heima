import asyncio
import json
import logging
import threading
import time
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from bot_mcp import call_tool as _base_call_tool
from config.settings import TAG_MCP_SSE_URL
from src.tools.porstgreDB_tools import (
    get_app_id_from_session,
    list_chat_messages,
    get_user_tags_db
)
from mcp.server.fastmcp import FastMCP
from src.utils.llm_utils import get_default_llm
from langchain.agents import create_agent
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from pydantic import BaseModel

class Tag(BaseModel):
    tag_id: str | int
    tag_name: str | int
    tag_alias: str | int
    tag_desc: str | None = ""

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


async def call_tool_async(
    tool_name: str,
    params: Dict[str, Any],
    url: str = TAG_MCP_SSE_URL,
    timeout: Optional[float] = 180.0,
) -> Any:
    """异步调用 MCP 工具（用于后台任务）"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: _base_call_tool(tool_name, params, url=url, timeout=timeout)
    )


tmp_data =[
    {
        "label_id": "175",
        "label_name": "性别 - 女",
        "alias": "女",
        "reason": "用户自述为女，或提及“我男朋友/老公/闺蜜”等具有明显女性画像的语义。",
        "parent": "173",
    },
    {
        "label_id": "207",
        "label_name": "综合价值等级 - 低",
        "alias": "浅层尝试",
        "reason": "用户仅尝试性对话几句，未进入核心业务流程即离开。",
        "parent": 204,
    }
]


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
            "alias": node.get("alias") or node.get("tag_alias") or "",  # 支持别名字段
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


async def _update_customer_labels(app_id: str, user_id: str, label_ids: Sequence[int]) -> Dict[str, Any]:
    """更新客户标签到 CRM 系统"""
    params = {
        "appId": app_id,
        "labelIds": list(label_ids),
        "type": 1,
        "outerId": user_id,
    }
    try:
        raw = await call_tool_async("update_customer_label", params=params)

        
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
            return {"error":0,"result": str(result_data)}
            
    except Exception as exc:
        logger.error("更新客户标签失败: %s", exc, exc_info=True)
        return {"error": str(exc), "error_type": type(exc).__name__}

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
    message_count = len(messages)
    logger.info(f"上下文切片完成，会话ID: {conversation_id}，用户ID: {user_id}，获取到 {message_count} 条原始消息")

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
    """
    召回文本标签
    
    Args:
        conversation_id: 会话 ID
        session_id: 会话会话 ID
        user_id: 用户 ID
    
    Returns:
        有效标签列表（格式：[{"id": "1", "name": "...", "description": "..."}, ...]）
    """
    fetched_app_id = get_app_id_from_session(conversation_id, session_id, user_id)
    if fetched_app_id:
        app_id = fetched_app_id
        logger.info(f"从会话数据中获取到 app_id: {app_id}")
    else:
        # 如果从会话数据中也获取不到，返回状态而不是抛出异常
        logger.warning(f"无法获取 app_id：未提供且从会话数据中获取失败 (conversation_id={conversation_id}, session_id={session_id}, user_id={user_id})")
        return {"status": "no_app_id", "conversation_id": conversation_id, "session_id": session_id, "user_id": user_id}

  
    app_id = "aiwa-admin"
    labels = _get_valid_labels(app_id)
   
    logger.info(f"从会话数据中获取到 {len(labels)} 个有效标签")
  
    # 检查标签池是否为空，如果为空则直接返回，不消耗LLM Token
    if not labels:
        logger.warning(f"标签池为空，无法进行标签召回 (conversation_id={conversation_id}, session_id={session_id}, user_id={user_id})")
        return []

    logger.info(f"标签召回准备完成，会话ID: {conversation_id}，用户ID: {user_id}，准备返回 {len(labels)} 个候选标签")
    return labels,app_id

def filtered_user_tags(user_tags):
    """过滤用户标签并返回分类结果
    
    Args:
        user_tags: 用户已打标签列表
        
    Returns:
        曾经被删除blocklog_tags列表
    """
    blocklog_tags = []
    
    for tag in user_tags:
        tag_type = tag.get('type', 0)
        if tag_type == 3:
            blocklog_tags.append(str(tag.get('label_id', '')))
    
    # 去重处理
 
    blocklog_tags = list(set(blocklog_tags))
    
    return {
        'blocklog_tags': blocklog_tags
    }

def analyze_customer_intent(chat_history: List[Dict], candidate_tags: List[Dict], user_tags: List[Dict]) -> List[Tag]:
    """
    分析客户意图并返回高置信度标签
    
    Args:
        chat_history: 最近20条对话历史（格式：[{"role": "user", "content": "..."}, ...]）
        candidate_tags: 候选标签列表（格式：[{"id": "1", "name": "...", "desc": "..."}, ...]）
        user_tags: 用户已打标签列表（格式：[{"type": "1", "label_id": "2", "is_active": "1"}, ...]）

    
    Returns:
        符合要求的Tag对象列表，包含大模型生成的标签和customer_tags，已去重
    """
    logger.info(f"开始客户意图分析，输入：聊天历史 {len(chat_history)} 条，候选标签 {len(candidate_tags)} 个，用户现有标签 {len(user_tags)} 个")
    
    # 检查候选标签是否为空，如果为空则直接返回，不消耗LLM Token
    if not candidate_tags:
        logger.warning("候选标签列表为空，无法进行客户意图分析")
        return []


    user_tags_data = filtered_user_tags(user_tags)
    logger.info(f"用户标签过滤完成，blocklog_tags数量：{len(user_tags_data.get('blocklog_tags', []))}")

    
    candidate_tag_ids = set()
    for tag in candidate_tags:
        tag_id = str(tag.get("id")) if tag.get("id") is not None else None
        if tag_id:
            candidate_tag_ids.add(tag_id)
  
    
    # 格式化对话历史
    formatted_history = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'agent'}: {msg['content'].strip()}"
        for msg in chat_history
    )
    
    # 格式化候选标签
    formatted_tags = json.dumps(candidate_tags, ensure_ascii=False, indent=2)
    
    
    
    intent_prompt = f"""你是一个专业的客户意图分析专家。

任务：分析提供的对话历史，从候选标签中识别适用于该客户的标签。

约束：
1. 依据优先级：判断主要依据是标签描述（tag_desc），其次是标签名称（tag_name）。
2. 严格模式：只有当对话中有明确证据支持时，才可打标。严禁由于用户“没有提到某事”而打上负面标签。
3. 输出限制：仅输出confidence_score >= 75的标签。

【重要提示：人工删除记录】
以下标签曾经被用户手动删除过：{', '.join(user_tags_data['blocklog_tags'])}
请谨慎判断是否应该再次选择这些标签。只有在会话内容明确支持的情况下，才可以选择这些标签。
如果会话内容不足以支持，请不要选择这些标签。

候选标签列表：
{formatted_tags}

输出要求：
1. 仅输出JSON数组，每个元素包含：
   - "tag_id": 标签ID（字符串）
   - "tag_name": 标签名称（字符串）
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

# Example
[
    {{
        "input":"我想给我的 咖啡店 采购一批桌椅，大概要 50套。",
        "output": [
            {{
                "tag_id": "101",
                "tag_name": "B端客户/批发",
                "confidence_score": 95,
                "reason": "用户明确自述了商业场景（咖啡店）和采购量级（50套），证据确凿，无歧义。"
            }},
        ]
    }},
    {{
        "input":"（前序对话询问了发货地）...“那 发到新疆 这一单大概要几天？运费能包吗？",
        "output": [
            {{
                "tag_id": "102",
                "tag_name": "偏远地区客户",
                "confidence_score": 80,
                "reason": "虽然用户没说“我住在新疆”，但询问发往新疆的时效和运费，在业务逻辑上大概率指向该地区，逻辑推导合理。"
            }},
        ]
    }},
    {{
        "input":"多少钱？",
        "output": [
            {{
                "tag_id": "103",
                "tag_name": "价格敏感",
                "confidence_score": 30,
                "reason": "询问价格是购买流程中的标准动作，不能因为客户问了价格就判定为“价格敏感”，除非后续有嫌贵行为。证据不足。"
            }},
        ]
    }},
    
]
# Output Format (JSON)
[
  {{"tag_id": "1",
    "tag_name": "标签名称",
    "confidence_score": 85,
    "reason": "用户在第3条消息明确询问'有没有折扣'，并在第5条消息表示'预算有限'。"
  }},
  
]

注意：请确保输出是有效的JSON，不要包含任何Markdown格式。
"""
    
    # 调用大模型
    logger.info("开始调用大模型进行意图分析")
    llm_intent_agent = create_agent(
        model=llm_model,
        name="llm_intent_agent",
        system_prompt=intent_prompt,
    )

    response_text = llm_intent_agent.invoke(
        {"messages": [{"role": "user", "content": intent_prompt}]},
    )['messages'][-1].content
    
    logger.info(f"大模型调用完成，响应长度：{len(response_text)} 字符")

    final_tags = []
    seen_tag_ids = set()
    
    try:
        result = json.loads(response_text)
        logger.info(f"模型输出解析成功，包含 {len(result)} 个标签")
        
        # 1. 处理大模型生成的标签
        for tag in result:
            if tag.get("confidence_score", 0) >= 75:
                tag_id = str(tag.get("tag_id")) if tag.get("tag_id") is not None else None
                if tag_id and tag_id in candidate_tag_ids:
                    if tag_id not in seen_tag_ids:
                        seen_tag_ids.add(int(tag_id))
                        # 创建Tag对象
                        final_tag = Tag(
                            tag_id=tag.get("tag_id",""),
                            tag_name=tag.get("tag_name", ""),
                            tag_alias=tag.get("alias", ""),
                            tag_desc=tag.get("reason", "")
                        )
                        final_tags.append(final_tag)
                else:
                    logger.warning(f"过滤掉不在候选标签列表中的标签: tag_id={tag_id}")

        # 2. 处理用户已打标签
        logger.info(f"大模型生成标签处理完成，共处理 {len(result)} 个标签，保留 {len(seen_tag_ids)} 个标签")
        
        return list(seen_tag_ids)
    except json.JSONDecodeError:
        # 处理模型输出异常
        logger.error(f"模型输出解析错误: {response_text}")
        
        return []

class IntentAnalysisState(TypedDict):
    """客户意图分析工作流状态"""
    conversation_id: str
    session_id: str
    user_id: str
    app_id: str
    chat_history: Optional[Dict] = None
    candidate_tags: Optional[List[Dict]] = None
    intent_result: Optional[List[Dict]] = None
    user_tags: Optional[List[Dict]] = None
    update_result: Optional[Dict] = None

def get_chat_history_node(state: IntentAnalysisState) -> IntentAnalysisState:
    """获取并过滤会话历史"""
    chat_history = context_licing(
        conversation_id=state["conversation_id"],
        session_id=state["session_id"],
        user_id=state["user_id"]
    )
    message_count = len(chat_history.get("filtered_messages", []))
    logger.info(f"获取聊天历史完成，会话ID: {state['conversation_id']}，用户ID: {state['user_id']}，获取到 {message_count} 条消息")
    return {**state, "chat_history": chat_history}


def recall_candidate_tags_node(state: IntentAnalysisState) -> IntentAnalysisState:
    """召回候选标签"""
    candidate_tags,app_id = recall_text(
        conversation_id=state["conversation_id"],
        session_id=state["session_id"],
        user_id=state["user_id"]
    )
    tag_count = len(candidate_tags) if isinstance(candidate_tags, list) else 0
    logger.info(f"标签召回完成，会话ID: {state['conversation_id']}，用户ID: {state['user_id']}，召回 {tag_count} 个候选标签，APP_ID: {app_id}")
    return {**state, "candidate_tags": candidate_tags, "app_id": app_id}


def analyze_intent_node(state: IntentAnalysisState) -> IntentAnalysisState:
    """分析客户意图"""
    result = analyze_customer_intent(
        chat_history=state["chat_history"]["filtered_messages"],
        candidate_tags=state["candidate_tags"],
        user_tags=state["user_tags"]
    )
    intent_count = len(result) if isinstance(result, list) else 0
    logger.info(f"意图分析完成，会话ID: {state['conversation_id']}，用户ID: {state['user_id']}，识别到 {intent_count} 个意图标签")
    return {**state, "intent_result": result}


def get_user_tags(
    user_id: str
) -> List[Dict]:
    """获取用户标签"""

    return get_user_tags_db(user_id)
    
def get_user_tags_node(state: IntentAnalysisState) -> IntentAnalysisState:
    """获取用户标签"""
    user_tags = get_user_tags(
        user_id=state["user_id"]
    )
    tag_count = len(user_tags)
    logger.info(f"获取用户现有标签完成，用户ID: {state['user_id']}，现有标签数量: {tag_count}")
    return {**state, "user_tags": user_tags}


# 更新到数据库
def update_user_tags_node(state: IntentAnalysisState) -> IntentAnalysisState:
    intent_result=state["intent_result"]
    user_id=state["user_id"]
    app_id=state["app_id"]
    conversation_id=state["conversation_id"]
    
    logger.info(f"开始更新用户标签，会话ID: {conversation_id}，用户ID: {user_id}，APP_ID: {app_id}，要更新的标签ID列表: {intent_result}")

    update_result = _update_customer_labels(app_id=app_id, user_id=user_id, label_ids=intent_result)
    logger.info(f"标签更新API调用完成，结果: {update_result}")

    if update_result.get("error","1") == "1":
        logger.info(f"用户 {user_id} 标签更新成功，会话ID: {conversation_id}，更新的标签数量: {len(intent_result) if isinstance(intent_result, list) else 0}")
        update_result['success'] = True
    else:
        logger.error(f"用户 {user_id} 标签更新失败，会话ID: {conversation_id}，错误信息: {update_result.get('msg', '未知错误')}，API返回: {update_result}")
        update_result['success'] = False
    return {**state, "update_result": update_result}






# ==========================================
# 主执行函数（推荐使用工具方式）
# ==========================================

async def run_intent_analysis(
    conversation_id: str,
    session_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    运行客户意图分析（后台异步处理）
    
    按照工作流的顺序执行，保持逻辑一致性。
    这个函数设计为在后台运行，不会阻塞用户请求。
    """
    logger.info(f"开始执行客户意图分析（后台异步），会话ID: {conversation_id}，会话会话ID: {session_id}，用户ID: {user_id}")
    
    try:
        # 步骤1: 获取聊天历史
        chat_history = context_licing(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id
        )
        message_count = len(chat_history.get("filtered_messages", []))
        logger.info(f"获取聊天历史完成，会话ID: {conversation_id}，用户ID: {user_id}，获取到 {message_count} 条消息")
        
        # 步骤2: 召回候选标签
        recall_result = recall_text(
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id
        )
        
        # 检查是否成功获取到标签
        if isinstance(recall_result, dict) and recall_result.get("status") == "no_app_id":
            logger.warning(f"无法获取 app_id，跳过标签分析 - 会话ID: {conversation_id}，用户ID: {user_id}")
            return {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "user_id": user_id,
                "status": "skipped",
                "reason": "no_app_id"
            }
        
        candidate_tags, app_id = recall_result
        tag_count = len(candidate_tags) if isinstance(candidate_tags, list) else 0
        logger.info(f"标签召回完成，会话ID: {conversation_id}，用户ID: {user_id}，召回 {tag_count} 个候选标签，APP_ID: {app_id}")
        
        # 检查候选标签是否为空
        if not candidate_tags:
            logger.warning(f"候选标签列表为空，跳过标签分析 - 会话ID: {conversation_id}，用户ID: {user_id}")
            return {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "user_id": user_id,
                "app_id": app_id,
                "status": "skipped",
                "reason": "no_candidate_tags"
            }
        
        # 步骤3: 获取用户现有标签
        user_tags = get_user_tags(user_id=user_id)
        logger.info(f"获取用户现有标签完成，用户ID: {user_id}，现有标签数量: {len(user_tags)}")
        
        # 步骤4: 分析客户意图
        intent_result = analyze_customer_intent(
            chat_history=chat_history.get("filtered_messages", []),
            candidate_tags=candidate_tags,
            user_tags=user_tags
        )
        logger.info(f"意图分析完成，会话ID: {conversation_id}，用户ID: {user_id}，识别到 {len(intent_result)} 个意图标签")
        
        # 步骤5: 更新用户标签（异步调用）
        update_result = await _update_customer_labels(app_id=app_id, user_id=user_id, label_ids=intent_result)
        logger.info(f"标签更新API调用完成，结果: {update_result}")
        
        result = {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "app_id": app_id,
            "chat_history": chat_history,
            "candidate_tags": candidate_tags,
            "intent_result": intent_result,
            "user_tags": user_tags,
            "update_result": update_result,
            "status": "completed"
        }
        
        logger.info(f"客户意图分析（后台异步）执行完成，会话ID: {conversation_id}，用户ID: {user_id}，最终结果: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"客户意图分析执行失败 - 会话ID: {conversation_id}，用户ID: {user_id}，错误: {exc}", exc_info=True)
        return {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__
        }




if __name__ == "__main__":

    pass
    # result = run_intent_analysis(
    #     conversation_id='1911', 
    #     session_id='95', 
    #     user_id='252'
    # )
    # print("最终结果的输出：")
    # print(result['update_result'])
