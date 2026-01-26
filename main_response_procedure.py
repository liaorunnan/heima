from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, Union, List, Callable, Tuple
# import porstgreDB_server as db
import src.tools.porstgreDB_tools as db
import json
from datetime import datetime, timezone, timedelta
import prompts.prompts as prompts
import re
import uuid
from bot_mcp import call_tool, get_bot_config, get_products_with_cache
from dify_mcp import retrieve_dataset, extract_dataset_snippets
import router_server as router
from Responder import llm_generic
# from core.redis_client import redis_get, redis_set
from config.settings import DEFAULT_BOT_LLM_CONFIG
import logging
from cors_handler import add_cors_support
from concurrent.futures import ThreadPoolExecutor, Future
from scorer import generate_conversation_scores
from src.tools.redis_tools import redis_get, redis_set

# 配置 logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

mcp = FastMCP("Main Response Server", host="0.0.0.0", port=3003)
# 添加 CORS 支持：处理 OPTIONS 预检请求
add_cors_support(mcp)

# 线程池执行器，用于异步生成 info
_INFO_EXECUTOR = ThreadPoolExecutor(max_workers=4)
# 线程池执行器，用于异步计算成交意愿分
_SCORE_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def stage_switch(conversation_id: str, session_id: str, user_id: str, state_order: list[str]) -> str:
    """根据最新一条 session_states 决定是否切换到下一个阶段。

    逻辑：
    - 取最新记录的 current_state 以及对应列（lower(current_state)）的 payload。
    - 若该 payload 已存在且非空，则按 state_order 中的顺序切换到下一阶段；
      - 当前阶段在列表中：返回下一个；若已是最后一个则返回当前阶段；
      - 当前阶段不在列表中：返回列表第一个；
    - 若 payload 为空或不存在：返回当前阶段（不切）。
    """
    latest = json.loads(db.get_latest_session_state_payload(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
    ))
    cur_state = latest.get("current_state")
    payload = latest.get("stage_payload")

    has_target = payload is not None and payload != "" and payload != {}
    if not has_target:
        return cur_state

    if not state_order:
        return cur_state

    if cur_state in state_order:
        idx = state_order.index(cur_state)
        return state_order[idx + 1] if idx + 1 < len(state_order) else cur_state

    return state_order[0]

def _ensure_prompt_text(segment) -> str:
    """将 prompts 段落标准化为字符串。
    - 若为 set/list/tuple，拼接为单一字符串；
    - 其余类型转为 str。
    """
    if isinstance(segment, set):
        return "".join(segment)
    if isinstance(segment, (list, tuple)):
        return "\n\n".join([str(x) for x in segment])
    return str(segment)


def _cleanup_llm_json_str(raw: str) -> str:
    """移除 LLM 返回结果中的 ```json ``` 样式包裹。"""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            first = lines[0].strip()
            if first.startswith("```"):
                lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_to_single_id(raw_value: Any) -> str:
    """将商品ID或权益ID规范化为单个ID字符串。
    
    如果输入是数组，取第一个元素；如果是字符串，先尝试解析为 JSON 数组。
    如果是 JSON 字符串形式的数组（如 '["289","290","291"]'），解析后取第一个元素。
    如果是其他类型，转换为字符串。
    
    参数:
        raw_value: 原始值（可能是字符串、列表或其他类型）
    
    返回:
        单个ID字符串（如果为空或无效，返回空字符串）
    """
    if raw_value is None:
        return ""
    
    if isinstance(raw_value, list):
        # 如果是数组，取第一个非空元素
        for item in raw_value:
            if item is not None:
                return str(item).strip()
        return ""
    
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        # 如果字符串看起来像 JSON 数组（以 [ 开头，以 ] 结尾），尝试解析
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    # 解析成功且是列表，取第一个非空元素
                    for item in parsed:
                        if item is not None:
                            return str(item).strip()
                    return ""
            except (json.JSONDecodeError, ValueError, TypeError):
                # JSON 解析失败，继续使用原始字符串
                pass
        return stripped
    
    # 其他类型直接转换为字符串
    return str(raw_value).strip()


def parse_exit_keywords(structured_content: Any) -> List[str]:
    """从 bot 配置中提取 exitKeyword 列表（兼容字符串与列表）。"""
    keywords = structured_content.get('exitKeyword')
    return keywords


def detect_handoff_intent(
    *,
    exit_keywords: List[str],
    user_input: str,
    history_context: Any,
    session_state: Dict[str, Any],
    bot_llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    """调用 LLM 判断是否满足转人工条件。"""
    if not exit_keywords:
        return {"handoff": False, "reason": "no keywords"}

    system_prompt = (
        "你是客服质检助手，请判断用户是否需要转人工客服。"
        "如果当前用户输入满足exitKeyword中的条件，则返回handoff=true，否则返回handoff=false。"
        f"exitKeyword: {exit_keywords}"
        "请仅输出 JSON：{\"handoff\": true|false}。"
    )
    system_prompt = f"{system_prompt}"

    #keywords_text = ", ".join(exit_keywords)

    llm_result = llm_generic(
        full_prompt=system_prompt,
        user_input=user_input,
        history_context=history_context,
        session_state=session_state,
        botLLMConfig=bot_llm_config,
        prompt_without_character=system_prompt,
        input_label="用户最新输入",
    )

    raw_text: str
    if isinstance(llm_result, dict):
        raw_text = llm_result.get("response") or llm_result.get("text") or json.dumps(llm_result, ensure_ascii=False)
    else:
        raw_text = str(llm_result)
    cleaned = _cleanup_llm_json_str(raw_text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"handoff": False, "raw": cleaned}


def judge_after_sales_intent(
    *,
    user_input: str,
    history_context: Any,
    bot_llm_config: Dict[str, Any],
) -> bool:
    """判断用户意图是否为售后。"""
    system_prompt = (
        "你是一个电商客服助手。请判断用户的最新输入是否涉及售后服务（如：退款、退货、换货、物流查询、投诉、订单状态、发票问题等）。"
        "如果是售后相关问题，请返回 true，否则返回 false。"
        "请仅输出 JSON：{\"is_after_sales\": true|false}。"
    )

    llm_result = llm_generic(
        full_prompt=system_prompt,
        user_input=user_input,
        history_context=history_context,
        session_state={},
        botLLMConfig=bot_llm_config,
        prompt_without_character=system_prompt,
        input_label="用户最新输入",
    )

    raw_text: str
    if isinstance(llm_result, dict):
        raw_text = llm_result.get("response") or llm_result.get("text") or json.dumps(llm_result, ensure_ascii=False)
    else:
        raw_text = str(llm_result)
    cleaned = _cleanup_llm_json_str(raw_text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed.get("is_after_sales", False)
    except json.JSONDecodeError:
        pass
    return False


def process_handoff_detection(
    *,
    conversation_id: str,
    turn_uuid: str,
    fn: Optional[Callable[[Dict[str, Any]], None]],
    threshold: int,
    reply_text: str,
    ttl_seconds: int,
) -> bool:
    """记录转人工触发次数，达到阈值后推送提示并清零计数。"""
    redis_key = f"handoff:{conversation_id}"
    count = 0
    cached = redis_get(redis_key)
    if cached:
        try:
            cached_data = json.loads(cached)
            count = int(cached_data.get("count", 0))
        except Exception:
            count = 0

    count += 1
    timestamp = datetime.now(timezone.utc).isoformat()
    redis_set(
        redis_key,
        json.dumps({"count": count, "latest_ts": timestamp}, ensure_ascii=False),
        expired=ttl_seconds,
    )

    if count >= threshold:
        logger.info(
            "========== [handoff] 达到转人工阈值 (%d) ==========",
            threshold,
        )
        redis_set(
            redis_key,
            json.dumps({"count": 0, "latest_ts": timestamp}, ensure_ascii=False),
            expired=ttl_seconds,
        )
        if fn:
            try:
                fn(
                    {
                        "event": "reply",
                        "data": {
                            "info": {"customer_service": True},
                            "response": reply_text,
                            "turn_uuid": turn_uuid,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[handoff] 推送转人工提示失败: %s", exc)
        return True

    logger.info(
        "========== [handoff] 当前 conversation=%s 转人工累计次数: %d ==========",
        conversation_id,
        count,
    )
    return False


def build_stage_prompt(current_state: str, current_sub_stage: Optional[str] = None) -> str:
    """根据 current_state + current_sub_stage 取对应模块 prompt，并与角色 prompt 组装。

    规则：
    - 角色 prompt 使用 prompts.Character_Prompts
    - 模块名解析：
      - 若提供 substage，则属性名 = f"{current_state}_{current_sub_stage}_Prompts"
      - 否则属性名 = f"{current_state}_Prompts"
    - 不做兜底：若属性不存在将抛出 AttributeError（符合"让错误直接暴露"的约定）。
    """
    role_seg = _ensure_prompt_text(prompts.Character_Prompts)
    attr_name = f"{current_state}_{current_sub_stage}_Prompts" if current_sub_stage else f"{current_state}_Prompts"
    module_seg = getattr(prompts, attr_name)  # 若不存在，直接抛异常
    module_text = _ensure_prompt_text(module_seg)
    return "\n\n".join([role_seg, module_text])


def extract_conversational_content_from_info(results: dict) -> dict:
    """从 results 的 info 字段中提取对话内容，合并到 response 中。

    判断标准：
    - 字段值长度 > 30 字符
    - 包含中文标点符号（。，！？、等）
    - 不是纯数字或纯ID
    - 可能包含表情符号或对话语气词

    参数:
        results: LLM 返回的结果字典，包含 info 和 response 字段

    返回:
        修改后的 results 字典
    """
    if not isinstance(results, dict) or "info" not in results:
        return results

    info = results.get("info", {})
    if not isinstance(info, dict):
        return results

    # 中文标点符号
    chinese_punctuation = "。，！？、；：""''（）【】《》"
    # 对话语气词/表情符号常见字符
    conversational_indicators = ["吗", "呢", "吧", "啊", "呀", "哦", "嗯", "✨", "💫", "😊", "😄", "？", "！"]
    punctuation_set = set(chinese_punctuation + "，。！？!?.,;:~…—-、；：'\"（）【】《》·")

    sentence_enders = set("。！？!?")

    extracted_content = []
    fields_to_check = []

    # 遍历 info 中的所有字段
    for key, value in info.items():
        if not isinstance(value, (str, int, float)):
            continue

        value_str = str(value).strip()

        # 跳过仅由重复标点组成的内容（例如 "......"、"！！！"）
        compact_value = re.sub(r"\s+", "", value_str)
        if compact_value:
            first_char = compact_value[0]
            if all(ch == first_char for ch in compact_value) and first_char in punctuation_set:
                continue

        # 跳过空值、纯数字、纯ID（长度短且无标点）
        if not value_str or len(value_str) < 10:
            continue

        # 跳过纯数字或纯ID（如商品ID）
        if value_str.isdigit() or (len(value_str) < 10 and not any(c in value_str for c in chinese_punctuation + "，。！？")):
            continue

        # 判断是否包含对话特征
        has_chinese_punct = any(c in value_str for c in chinese_punctuation)
        has_conversational = any(indicator in value_str for indicator in conversational_indicators)
        has_sentence_end = any(c in sentence_enders for c in value_str)
        has_chinese_chars = bool(re.search(r'[\u4e00-\u9fff]', value_str))

        # 如果包含对话特征，提取出来
        if has_chinese_chars and (has_conversational or has_sentence_end):
            extracted_content.append(value_str)
            fields_to_check.append(key)

    # 如果有提取到内容，合并到 response
    if extracted_content:
        current_response = results.get("response", "").strip()

        # 合并提取的内容（用空格分隔）
        extracted_text = " ".join(extracted_content)

        # 检查 response 中是否已经包含提取的内容（避免重复）
        should_add = True
        if current_response:
            # 检查提取的内容是否已经在 response 中存在
            # 使用简单的包含检查，如果提取内容的主要部分已经在 response 中，则不添加
            extracted_words = set(re.findall(r'[\u4e00-\u9fff]+', extracted_text))
            response_words = set(re.findall(r'[\u4e00-\u9fff]+', current_response))

            # 如果提取内容的主要词汇大部分都在 response 中，则认为已存在
            if extracted_words and len(extracted_words.intersection(response_words)) / len(extracted_words) > 0.7:
                should_add = False
                logger.info(f"[Extract Conversational Content] Content already exists in response, skipping. Fields: {fields_to_check}")

        if should_add:
            if current_response:
                # 将提取的内容放在 response 前面
                results["response"] = f"{extracted_text} {current_response}".strip()
            else:
                # 如果 response 不存在，直接设置
                results["response"] = extracted_text

            # 记录日志
            logger.info(f"[Extract Conversational Content] Extracted from fields: {fields_to_check}")
            logger.info(f"[Extract Conversational Content] Merged content length: {len(extracted_text)} characters")
            logger.info(f"[Extract Conversational Content] Content placed before existing response")

    return results

def join_prompts(prompt: dict) -> str:
    """将单个模块 prompt 字典拼接为规范文本：

    输入示例：
    {
      "purpose": "根据会话状态向用户推荐牌阵",
      "name": ["牌阵名称", "所需牌数", "response"],
      "expect": ["牌阵名称", "牌阵所需牌数", "一段自然对话来告知客户牌阵，需要抽几张牌；并请客户在内心聚焦待解的问题，依照牌阵的顺序，从1至78中抽取所需数量的数字"],
      "operation": ["1. 输出牌阵名称", "2. 输出牌阵所需牌数", "3. 输出一段自然对话……"]
    }

    输出：
    [任务]\n<purpose>\n\n[步骤]\n<按行拼接 operation>\n\n[结果]\n<按行编号拼接 expect>\n\n【输出格式要求】+ JSON 结构，其中 info 由 name/expect 成对映射，若 name 为 'response'（不区分大小写），则映射到顶级 "response"。
    """
    purpose = str(prompt.get("purpose", "")).strip()
    names = prompt.get("name") or []
    expects = prompt.get("expect") or []
    operations = prompt.get("operation") or []

    # 步骤
    steps_block = "\n".join(str(x) for x in operations)

    # 结果（编号）
    result_lines = []
    for i, exp in enumerate(expects, start=1):
        result_lines.append(f"{i}. {str(exp)}")
    results_block = "\n".join(result_lines)

    # 输出格式要求：info 与 response 的映射
    info_pairs = []
    response_value = None

    # 收集所有名为 response 的期望值，并将其合并到顶级 response
    response_values = []
    for idx, n in enumerate(names):
        key = str(n)
        val = str(expects[idx]) if idx < len(expects) else ""
        if key.strip().lower() == "response":
            # 收集到列表，稍后统一合并为顶级 response
            if val:
                response_values.append(val)
        else:
            info_pairs.append((key, val))

    # 将所有 response 合并为一个顶级 response（保持顺序）
    if response_values:
        response_value = " ".join(response_values)

    # 组装 JSON 片段（以文本形式返回，保持中文与引号）
    info_lines = []
    for k, v in info_pairs:
        info_lines.append(f'        "{k}": "{v}"')
    info_block = (",\n".join(info_lines)) if info_lines else ""

    # 顶级 response 行（可选）
    response_line = f'      "response": "{response_value}"' if response_value is not None else None

    parts = []
    parts.append("[任务]")
    parts.append(purpose)
    parts.append("")
    parts.append("[步骤]")
    parts.append(steps_block)
    parts.append("")
    parts.append("[结果]")
    parts.append(results_block)
    parts.append("")
    parts.append("    【输出格式要求}")
    parts.append("")
    parts.append("    输出必须严格遵循以下JSON结构：")
    parts.append("    {")
    parts.append("      \"info\": {")
    if info_block:
        parts.append(info_block)
    parts.append("      }")
    if response_line is not None:
        parts.append("      ,")
        parts.append(response_line)
    parts.append("    }")

    return "\n".join(parts)


def build_response_prompt_only(prompt: dict) -> str:
    """根据模块 prompt 生成仅输出 response 的提示词。"""
    purpose = str(prompt.get("purpose", "")).strip()
    operations = prompt.get("operation") or []
    expects = prompt.get("expect") or []
    names = prompt.get("name") or []

    steps_block = "\n".join(str(x) for x in operations)

    response_keywords = ("对话", "回复", "自然语言", "礼貌", "response")
    response_expectations = []
    for idx, exp in enumerate(expects):
        exp_str = str(exp)
        name_str = (str(names[idx]) if idx < len(names) else "").strip().lower()
        if name_str == "response":
            response_expectations.append(exp_str)
            continue
        if any(keyword in exp_str for keyword in response_keywords):
            response_expectations.append(exp_str)

    if not response_expectations:
        response_expectations = ["发送对话回复"]

    result_lines = [f"{i}. {text}" for i, text in enumerate(response_expectations, start=1)]
    response_block = "\n".join(result_lines)

    parts = [
        "[声明]",
        "",
        "你只负责输出自然语言回复response用以回复用户，不参与任何其他任务。",
        "",
        "[任务]",
        purpose,
        "",
        "[步骤]",
        steps_block,
        "",
        "[结果]",
        response_block,
        "",
        # "    【输出格式要求】",
        # "",
        # "    输出必须严格遵循以下JSON结构：",
        # "    {",
        # '      "response": "发送对话回复"',
        # "    }",
    ]

    return "\n".join(parts)


def build_info_prompt_only(prompt: dict) -> str:
    """根据模块 prompt 生成仅输出 info 的提示词。"""
    purpose = str(prompt.get("purpose", "")).strip()
    operations = prompt.get("operation") or []
    expects = prompt.get("expect") or []
    names = prompt.get("name") or []

    steps_block = "\n".join(str(x) for x in operations)

    info_entries = []
    for idx, exp in enumerate(expects):
        name_str = (str(names[idx]) if idx < len(names) else "").strip()
        if name_str.lower() == "response":
            continue
        exp_str = str(exp)
        info_key = name_str if name_str else f"field_{idx + 1}"
        info_entries.append((info_key, exp_str))

    if not info_entries:
        info_entries = [("field_1", "结构化字段")]

    result_lines = [f"{i}. {text}" for i, (_, text) in enumerate(info_entries, start=1)]
    results_block = "\n".join(result_lines)

    info_lines = [f'        "{key}": "{value}"' for key, value in info_entries]
    info_block = ",\n".join(info_lines)

    parts = [
        "[声明]",
        "",
        "你只负责根据上下文与Assistant本轮回复内容，输出 info 所需的结构化字段，不生成任何自然语言回复内容，但需参与任何除生成自然语言回复以外的任务。你的结果需来源于上下文与Assistant本轮回复内容。",
        "",
        "[任务]",
        purpose,
        "",
        "[步骤]",
        steps_block,
        "",
        "[结果]",
        results_block,
        "",
        "    【输出格式要求】",
        "",
        "    输出必须严格遵循以下JSON结构：",
        "    {",
        '      "info": {',
        f"{info_block}",
        "      }",
        "    }",
    ]

    return "\n".join(parts)


def route_current_stage(
    conversation_id: str,
    session_id: str,
    user_id: str,
    user_input: str,
    bot_id: str,
    app_id: str,
    current_state: str,
    current_sub_stage: str,
):
    """调用 Router 并解析 stage/substage 结果，返回 router 响应与最终路由值。"""
    router_response = router.route_and_store(
        conversation_id,
        session_id,
        user_id,
        user_input,
        bot_id,
        app_id,
        current_state,
    )
    parsed = router_response.get("llm_output")
    logger.info("[Router] route_current_stage 输出: %s", json.dumps(router_response, ensure_ascii=False, indent=2))

    routed_current_state = ""
    routed_current_sub_stage = ""
    if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list) and parsed["queries"]:
        first_query = parsed["queries"][0]
        if isinstance(first_query, dict):
            routed_current_state = first_query.get("stage", "")
            routed_current_sub_stage = first_query.get("substage", "")

    if not routed_current_state:
        routed_current_state = current_state
    if not routed_current_sub_stage:
        routed_current_sub_stage = current_sub_stage

    logger.info("[Router] route_current_stage 结果: state=%s, sub_stage=%s", routed_current_state, routed_current_sub_stage)
    return router_response, routed_current_state, routed_current_sub_stage


def _select_stage_module(route_state_prompt_map: Dict[str, List[dict]], stage: str, sub_stage: str) -> dict:
    """根据阶段和子阶段编号选择对应的模块字典。"""
    if not isinstance(route_state_prompt_map, dict):
        return {}
    modules = route_state_prompt_map.get(stage) or []
    if not isinstance(modules, list) or not modules:
        return {}
    module_idx = 0
    if isinstance(sub_stage, str) and "_" in sub_stage:
        num_part = sub_stage.split("_")[-1]
        if num_part.isdigit():
            module_idx = max(int(num_part) - 1, 0)
    if module_idx >= len(modules):
        module_idx = 0
    module = modules[module_idx]
    return module if isinstance(module, dict) else {}


def normalize_substage_name(state: str, sub_stage: str) -> str:
    """确保返回 {stage}_{xx} 格式的子阶段名称。"""
    if sub_stage and "_" in sub_stage:
        if sub_stage.count("_") >= 2:
            return sub_stage
        num_part = sub_stage.split("_")[-1]
        return f"{state}_{num_part}"
    return f"{state}_01"


def _generate_info_content(
    *,
    character_prompt: str,
    route_state_prompt_map: Dict[str, List[dict]],
    routed_current_state: str,
    routed_current_sub_stage: str,
    input_text: str,
    history_context: Any,
    session_state: Dict[str, Any],
    bot_llm_config: Dict[str, Any],
    input_label: str = "用户本轮",
) -> Dict[str, Any]:
    """内部函数：调用 info LLM 生成 info 内容（核心逻辑）。
    
    参数:
        input_text: 输入文本（可以是 user_input 或 response_content）
        input_label: 输入标签（用于 prompt 显示）
    
    返回:
        info_content 字典
    """
    stage_module = _select_stage_module(route_state_prompt_map, routed_current_state, routed_current_sub_stage)
    info_prompt = build_info_prompt_only(stage_module) if stage_module else ""
    prompt_parts = [segment for segment in (character_prompt, info_prompt) if segment]
    full_prompt = "\n\n".join(prompt_parts)

    llm_result = llm_generic(
        full_prompt=full_prompt,
        user_input=input_text,
        history_context=history_context,
        session_state=session_state,
        botLLMConfig=bot_llm_config,
        prompt_without_character=info_prompt,
        input_label=input_label,
    )

    if isinstance(llm_result, str):
        try:
            llm_result = json.loads(_cleanup_llm_json_str(llm_result))
        except json.JSONDecodeError:
            llm_result = {"info": {"value": llm_result}}

    info_content: Dict[str, Any] = {}
    if isinstance(llm_result, dict):
        info_field = llm_result.get("info")
        if isinstance(info_field, dict):
            info_content = info_field
        elif info_field is not None:
            info_content = {"value": info_field}
    else:
        info_content = {"value": llm_result}
    
    return info_content


def generate_info_payload(
    *,
    character_prompt: str,
    route_state_prompt_map: Dict[str, List[dict]],
    routed_current_state: str,
    routed_current_sub_stage: str,
    turn_uuid: str,
    conversation_id: str,
    session_id: str,
    user_id: str,
    user_input: str,
    history_context: Any,
    session_state: Dict[str, Any],
    bot_llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    """调用 info LLM，更新 pending_turn，并返回包含 info 的 payload。"""
    info_content = _generate_info_content(
        character_prompt=character_prompt,
        route_state_prompt_map=route_state_prompt_map,
        routed_current_state=routed_current_state,
        routed_current_sub_stage=routed_current_sub_stage,
        input_text=user_input,
        history_context=history_context,
        session_state=session_state,
        bot_llm_config=bot_llm_config,
        input_label="用户本轮",
    )

    substage_col = normalize_substage_name(routed_current_state, routed_current_sub_stage)
    stage_payload = {substage_col: {"info": info_content}}
    try:
        db.update_pending_turn_state(
            turn_uuid=turn_uuid,
            routed_current_state=routed_current_state,
            routed_current_sub_stage=routed_current_sub_stage,
            stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("[generate_info_payload] 更新 pending_turn info 失败: %s", exc)

    return {
        "info": info_content,
        "turn_uuid": turn_uuid,
        "routed_current_state": routed_current_state,
        "routed_current_sub_stage": routed_current_sub_stage,
    }


def generate_product_analysis_payload(
    *,
    character_prompt: str,
    route_state_prompt_map: Dict[str, List[dict]],
    routed_current_state: str,
    routed_current_sub_stage: str,
    user_input: str,
    history_context: Any,
    session_state: Dict[str, Any],
    bot_llm_config: Dict[str, Any],
    turn_uuid: str,
    product_list_text: str,
    fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """调用商品分析 LLM，分析是否需要推荐/售卖商品，更新 pending_turn，并返回结果。

    参数:
        character_prompt: 角色设定 prompt
        route_state_prompt_map: 路由状态 prompt 映射，用于获取当前阶段的目标提示词
        routed_current_state: 路由后的阶段
        routed_current_sub_stage: 路由后的子阶段
        user_input: 用户输入
        history_context: 历史对话上下文
        session_state: 会话状态
        bot_llm_config: LLM 配置
        turn_uuid: 回合 UUID
        product_list_text: 商品列表文本（已格式化）
        fn: 可选的回调函数，用于推送事件给前端

    返回:
        包含 product_id_promoted / product_equity_promoted / product_id_sales / product_equity_sales 和 turn_uuid 的字典
    """
    # 构建商品分析 prompt
    product_analysis_prompt = "\n".join([
        "你是一个智能销售助手，负责根据当前任务和对话判断是否需要推销或售卖商品，或两种皆有。请严格遵循以下规则：",
        "",
        "输入：",
        "- 阶段任务：当前销售目标",
        "- 上下文：历史对话与会话信息（由系统提供）",
        "- 用户本轮：用户当前回复",
        "- 商品列表：可用商品信息，包含商品ID、介绍、权益等",
        "",
        "判断逻辑：",
        "1. 推销（product_id_promoted / product_equity_promoted）：",
        "   - 当需要主动推荐商品或权益时使用，例如：",
        "     * 阶段任务要求'推销商品'或'推荐商品'",
        "     * 用户询问商品功能、价格、特点等",
        "     * 历史聊天用户明确表示想了解某类商品或要求推荐商品",
        "",
        "2. 售卖（product_id_sales / product_equity_sales）：",
        "   - 当用户明确表达购买意图时使用，例如：",
        "     * 用户说'我想购买'、'我要买'、'帮我下单'、'购买XX'、'订阅XX'等",
        "     * 用户说'给我发链接'、'发购买链接'、'怎么买'等购买相关请求",
        "     * 阶段任务为'完成销售'或'促成交易'",
        "     * 历史聊天已达成购买意向，用户确认要购买",
        "   - 重要：只要用户明确表达购买意图，就必须设置 product_id_sales 和 product_equity_sales，即使阶段任务不是销售",
        "",
        "3. 同时存在：",
        "   - 可以同时设置推销和售卖字段，例如：用户想购买A商品，同时推荐B商品",
        "",
        "4. 无行为：",
        "   - 若无需推销或售卖，所有字段均保持空字符串",
        "",
        "注意事项：",
        "- 商品ID与权益key必须来自输入的商品列表，确保准确性",
        "- 如果涉及多个商品或权益，可使用 JSON 数组（如 [\"p1\", \"p2\"]）",
        "- 权益key需根据用户需求填写相应的权益key",
        "- 优先识别购买意图：如果用户明确表达购买，优先设置 workflow 字段",
        "",
        "输出要求：",
        "- 仅输出JSON，不要其他内容",
        "- 严格遵循以下格式：",
        "{",
        '  "product_id_promoted": "商品ID或空字符串",',
        '  "product_equity_promoted": "权益key或空字符串",',
        '  "product_id_sales": "商品ID或空字符串",',
        '  "product_equity_sales": "权益key或空字符串"',
        "}",
        "",
        "根据实际输入分析并输出JSON。",
    ])

    # 将阶段“任务”部分拼接进商品分析 prompt
    stage_module = _select_stage_module(route_state_prompt_map, routed_current_state, routed_current_sub_stage)
    stage_task_text = ""
    if isinstance(stage_module, dict):
        stage_purpose = str(stage_module.get("purpose", "")).strip()
        if stage_purpose:
            stage_task_text = f"【阶段任务】：{stage_purpose}"

    product_list_block = ""
    if product_list_text and product_list_text.strip():
        product_list_block = "【商品列表】\n" + product_list_text.strip()

    # 组装完整 prompt
    prompt_parts = [segment for segment in (stage_task_text, product_analysis_prompt, product_list_block) if segment]
    prompt_without_character = "\n\n".join(prompt_parts)
    full_prompt = prompt_without_character

    # 用户输入保持仅包含用户消息
    user_input_with_products = f"用户本轮：{user_input}"

    # 调用 LLM
    llm_result = llm_generic(
        full_prompt=full_prompt,
        user_input=user_input_with_products,
        history_context=history_context,
        session_state=session_state,
        botLLMConfig=bot_llm_config,
        prompt_without_character=full_prompt,
    )

    # 解析 LLM 结果
    product_id_promoted = ""
    product_equity_promoted = ""
    product_id_sales = ""
    product_equity_sales = ""

    if isinstance(llm_result, str):
        try:
            llm_result = json.loads(_cleanup_llm_json_str(llm_result))
        except json.JSONDecodeError:
            logger.warning("[generate_product_analysis_payload] LLM 返回结果不是有效 JSON: %s", llm_result)
            llm_result = {}

    if isinstance(llm_result, dict):
        # 规范化商品ID和权益ID为单个字符串（数组取第一个元素）
        product_id_promoted = _normalize_to_single_id(llm_result.get("product_id_promoted", ""))
        product_equity_promoted = _normalize_to_single_id(llm_result.get("product_equity_promoted", ""))
        product_id_sales = _normalize_to_single_id(llm_result.get("product_id_sales", ""))
        product_equity_sales = _normalize_to_single_id(llm_result.get("product_equity_sales", ""))

    # 更新 pending_turn（存入商品分析结果）
    try:
        product_analysis_payload = {
            "product_id_promoted": product_id_promoted,
            "product_equity_promoted": product_equity_promoted,
            "product_id_sales": product_id_sales,
            "product_equity_sales": product_equity_sales,
        }
        db.update_pending_turn_state(
            turn_uuid=turn_uuid,
            routed_current_state=routed_current_state,
            routed_current_sub_stage=routed_current_sub_stage,
            stage_payload_draft=json.dumps(
                {f"{routed_current_sub_stage}.product_analysis": {"info": product_analysis_payload}},
                ensure_ascii=False
            ),
        )
    except Exception as exc:
        logger.warning("[generate_product_analysis_payload] 更新 pending_turn 商品分析结果失败: %s", exc)

    result_payload = {
        "product_id_promoted": product_id_promoted,
        "product_equity_promoted": product_equity_promoted,
        "product_id_sales": product_id_sales,
        "product_equity_sales": product_equity_sales,
        "turn_uuid": turn_uuid,
    }

    # 如果提供了 fn 回调，立即推送商品分析事件
    if fn:
        try:
            fn({
                "event": "reply",
                "data": {
                    "info": {
                        "product_id_promoted": product_id_promoted,
                        "product_equity_promoted": product_equity_promoted,
                        "product_id_sales": product_id_sales,
                        "product_equity_sales": product_equity_sales,
                    },
                    "turn_uuid": turn_uuid,
                },
            })
        except Exception as exc:
            logger.warning("[generate_product_analysis_payload] 推送商品分析事件失败: %s", exc)

    return result_payload


def build_prev_next_step_texts(
    *,
    routed_current_state: str,
    routed_current_sub_stage: str,
    route_state_prompt_map: Dict[str, List[dict]],
    stages_complete: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str]:
    """构建上一步和下一步任务的提示词文本。
    
    参数:
        routed_current_state: 路由后的阶段
        routed_current_sub_stage: 路由后的子阶段
        route_state_prompt_map: 路由状态 prompt 映射
        stages_complete: 所有阶段完成状态字典（可选）
    
    返回:
        (prev_step_text, next_step_text) 元组
    """
    prev_step_text = ""
    next_step_text = ""
    
    if not stages_complete:
        return prev_step_text, next_step_text
    
    try:
        # 计算当前完整 substage 名称（如 cognition_01）
        if routed_current_sub_stage and '_' in routed_current_sub_stage:
            if routed_current_sub_stage.count('_') >= 2:
                current_substage_full = routed_current_sub_stage
            else:
                num_part = routed_current_sub_stage.split('_')[-1]
                current_substage_full = f"{routed_current_state}_{num_part}"
        else:
            current_substage_full = f"{routed_current_state}_01"
        
        stages_list = list(stages_complete.keys()) if isinstance(stages_complete, dict) else []
        
        # 上一步任务：取全局顺序中当前子阶段之前的第一个合法子阶段
        try:
            prev_module = None
            if current_substage_full in stages_list:
                cur_idx = stages_list.index(current_substage_full)
                for j in range(cur_idx - 1, -1, -1):
                    substage_name = stages_list[j]
                    stage_name = substage_name.rsplit('_', 1)[0] if '_' in substage_name else substage_name
                    if stage_name in ("questions", "after_sales"):
                        continue
                    stage_modules = route_state_prompt_map.get(stage_name, [])
                    if stage_modules:
                        prev_module = stage_modules[0]
                        break
            if isinstance(prev_module, dict):
                prev_purpose = str(prev_module.get("purpose", "")).strip()
                if prev_purpose:
                    prev_step_text = "\n\n".join(["[上一步任务]", prev_purpose])
        except Exception:
            prev_step_text = ""

        # 下一步任务：从全局 stages_complete 顺序中选取"当前子阶段之后"的下一个子阶段
        try:
            next_module = None
            if current_substage_full in stages_list:
                cur_idx = stages_list.index(current_substage_full)
                for j in range(cur_idx + 1, len(stages_list)):
                    substage_name = stages_list[j]
                    # 过滤不可选阶段
                    stage_name = substage_name.rsplit('_', 1)[0] if '_' in substage_name else substage_name
                    if stage_name in ("questions", "after_sales"):
                        continue
                    stage_modules = route_state_prompt_map.get(stage_name, [])
                    if stage_modules:
                        next_module = stage_modules[0]
                        break
            # 取 purpose 并拼接
            if isinstance(next_module, dict):
                next_purpose = str(next_module.get("purpose", "")).strip()
                if next_purpose:
                    next_step_text = "\n\n".join(["[下一步任务]", next_purpose])
        except Exception:
            next_step_text = ""
    except Exception:
        # 不影响主流程
        prev_step_text = ""
        next_step_text = ""
    
    return prev_step_text, next_step_text


def generate_response_payload(
    *,
    character_prompt: str,
    route_state_prompt_map: Dict[str, List[dict]],
    routed_current_state: str,
    routed_current_sub_stage: str,
    turn_uuid: str,
    conversation_id: str,
    session_id: str,
    user_id: str,
    user_input: str,
    history_context: Any,
    session_state: Dict[str, Any],
    bot_llm_config: Dict[str, Any],
    product_analysis_result: Optional[Dict[str, Any]] = None,
    prev_step_text: str = "",
    next_step_text: str = "",
    marketing_context: str = "",
    formatted_products_text: Dict[str, Any],
    fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """调用 response LLM，生成自然语言回复，更新 pending_turn，并通过 fn 推送事件。

    参数:
        character_prompt: 角色设定 prompt
        route_state_prompt_map: 路由状态 prompt 映射
        routed_current_state: 路由后的阶段
        routed_current_sub_stage: 路由后的子阶段
        turn_uuid: 回合 UUID
        conversation_id: 会话 ID
        session_id: 会话 ID
        user_id: 用户 ID
        user_input: 用户输入
        history_context: 历史对话上下文
        session_state: 会话状态
        bot_llm_config: LLM 配置
        product_analysis_result: 商品分析结果（可选）
        prev_step_text: 上一步任务文本（可选）
        next_step_text: 下一步任务文本（可选）
        marketing_context: 营销话术文本（可选）
        formatted_products_text: 商品列表文本（可选）
        fn: 回调函数，用于推送事件给前端

    返回:
        包含 response 和 turn_uuid 的字典
    """
    stage_module = _select_stage_module(route_state_prompt_map, routed_current_state, routed_current_sub_stage)
    response_prompt = build_response_prompt_only(stage_module) if stage_module else ""
    # 注意顺序：当前阶段 -> 上一步任务 -> 下一步任务
    system_context_parts: List[str] = []
    marketing_context_text = ""
    if isinstance(marketing_context, str) and marketing_context.strip():
        marketing_context_text = marketing_context.strip()
        system_context_parts.append("【营销话术】\n" + marketing_context_text)
    # 移除商品列表拼接 - 不再将商品列表拼接到回复prompt中
    # formatted_block = ""
    # if isinstance(formatted_products_text, str) and formatted_products_text.strip():
    #     formatted_block = formatted_products_text.strip()
    #     if formatted_block.startswith("商品列表"):
    #         system_context_parts.append(formatted_block)
    #     else:
    #         system_context_parts.append("【商品知识】\n" + formatted_block)

    base_prompt_parts = [p for p in [character_prompt, response_prompt, prev_step_text, next_step_text] if p]

    # 用户本轮输入
    user_input_sections = [f"用户本轮：{user_input}"]

    # 仅拼接【推广商品信息】，不再拼接【商品分析结果】
    if product_analysis_result:
        promo_details = product_analysis_result.get("promoted_product_details")
        if promo_details:
            system_context_parts.append(f"【Related products/关联商品】\n{promo_details}")
            logger.info(
                "[generate_response_payload] 已附加推广商品信息到 system prompt",
            )
        # 从 product_analysis_result 中读取知识库内容（纯知识库文本，不含商品其他信息）
        knowledge_content = product_analysis_result.get("knowledgeContent")
        if isinstance(knowledge_content, str) and knowledge_content.strip():
            system_context_parts.append(f"【商品知识】\n{knowledge_content.strip()}")
            logger.info(
                "[generate_response_payload] 已附加商品知识到 system prompt，长度=%d",
                len(knowledge_content),
            )

    if marketing_context_text:
        logger.info(
            "[generate_response_payload] 已附加营销话术到 system prompt，长度=%d",
            len(marketing_context_text),
        )
    # 移除商品列表日志
    # if formatted_block:
    #     logger.info(
    #         "[generate_response_payload] 已附加商品列表到 system prompt，长度=%d",
    #         len(formatted_block),
    #     )

    context_prompt_part = "\n\n".join(system_context_parts) if system_context_parts else ""
    full_prompt_parts = list(base_prompt_parts)
    if context_prompt_part:
        full_prompt_parts.append(context_prompt_part)
    full_prompt = "\n\n".join(full_prompt_parts)

    user_input_for_llm = "\n\n".join(user_input_sections)

    llm_result = llm_generic(
        full_prompt=full_prompt,
        user_input=user_input_for_llm,
        history_context=history_context,
        session_state=session_state,
        botLLMConfig=bot_llm_config,
        prompt_without_character=full_prompt,
    )

    # 解析 LLM 结果
    logger.info("[generate_response_payload] LLM 原始返回结果类型: %s", type(llm_result).__name__)
    if isinstance(llm_result, str):
        logger.info("[generate_response_payload] LLM 原始返回字符串: %s", llm_result[:500] if len(llm_result) > 500 else llm_result)
    
    response_content = ""
    if isinstance(llm_result, str):
        try:
            llm_result = json.loads(_cleanup_llm_json_str(llm_result))
        except json.JSONDecodeError:
            # 如果不是 JSON，直接作为 response
            response_content = llm_result

    if isinstance(llm_result, dict):
        response_field = llm_result.get("response")
        if isinstance(response_field, str):
            response_content = response_field
        elif response_field is not None:
            response_content = str(response_field)

    # 更新 pending_turn（存入 response）
    substage_col = normalize_substage_name(routed_current_state, routed_current_sub_stage)
    stage_payload = {substage_col: {"response": response_content}}
    try:
        db.update_pending_turn_state(
            turn_uuid=turn_uuid,
            routed_current_state=routed_current_state,
            routed_current_sub_stage=routed_current_sub_stage,
            stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("[generate_response_payload] 更新 pending_turn response 失败: %s", exc)

    result_payload = {
        "response": response_content,
        "turn_uuid": turn_uuid,
        "routed_current_state": routed_current_state,
        "routed_current_sub_stage": routed_current_sub_stage,
    }

    # 如果提供了 fn 回调，立即推送 response 事件
    if fn:
        try:
            fn({
                "event": "reply",
                "data": {
                    "response": response_content,
                    "turn_uuid": turn_uuid,
                },
            })
        except Exception as exc:
            logger.warning("[generate_response_payload] 推送 response 事件失败: %s", exc)

    return result_payload


def get_all_stages_and_substages(conversation_id: str, bot_config: Any) -> Dict[str, bool]:
    """从 bot_config 中按顺序提取所有 stage 和 substage 的名称，返回字典表示是否完成。

    参数:
        conversation_id: 会话标识（用于缓存键）
        bot_config: get_bot_config 返回的配置对象

    返回:
        Dict[str, bool]: 格式为 {"cognition_01": False, "interest_01": False, ...}
                        键为 "{stage}_{index:02d}" 格式，值为是否完成的布尔值（默认为 False）

    注意:
        - 结果按会话缓存，缓存有效期 90 天
        - 缓存键格式: "stages_substages:{conversation_id}"
    """
    cache_key = f"stages_substages:{conversation_id}"

    # 尝试从缓存读取
    cached = redis_get(cache_key)
    if cached:
        try:
            obj = json.loads(cached)
            # 确保返回的是正确的格式
            if isinstance(obj, dict):
                # 转换字符串键的布尔值为真正的布尔值（redis 返回的是字符串）
                result = {}
                for k, v in obj.items():
                    if isinstance(v, bool):
                        result[k] = v
                    elif isinstance(v, str):
                        result[k] = v.lower() in ('true', '1', 'yes')
                    else:
                        result[k] = bool(v)
                return result
        except Exception:
            # 缓存解析失败，继续执行计算逻辑
            pass

    # 计算所有 stage 和 substage
    structuredContent = bot_config.structuredContent

    # 优先使用 routeStateStrategies，如果没有则使用 routeStatePrompt
    route_state_map = structuredContent.get('routeStateStrategies') or structuredContent.get('routeStatePrompt')

    if not isinstance(route_state_map, dict):
        return {}

    result = {}

    # 按指定顺序遍历阶段；未包含的阶段按原顺序附加在后
    desired_order = ["cognition", "interest", "decision_making", "compliance", "after_sales"]
    all_stage_names = list(route_state_map.keys())
    ordered_stage_names = [s for s in desired_order if s in route_state_map]
    # 附加剩余未出现在 desired_order 中的阶段，保持原顺序
    ordered_stage_names.extend([s for s in all_stage_names if s not in desired_order])

    for stage_name in ordered_stage_names:
        modules = route_state_map.get(stage_name)
        if not isinstance(modules, list):
            continue

        # 遍历每个模块，生成 substage 名称
        for idx, module in enumerate(modules, start=1):
            substage_name = f"{stage_name}_{idx:02d}"
            result[substage_name] = False  # 默认未完成

    # 写入缓存（90 天，与 bot_config 缓存时长一致）
    try:
        redis_set(cache_key, json.dumps(result, ensure_ascii=False), expired=7_776_000)
    except Exception:
        # 缓存失败不影响主流程
        pass

    return result

def set_all_stages_and_substages(conversation_id: str, substage_name: str, is_complete: bool, bot_config: Any = None) -> bool:
    """更新 Redis 缓存中指定 substage 的完成状态。

    参数:
        conversation_id: 会话标识（用于缓存键）
        substage_name: 要更新的 substage 名称，格式如 "cognition_01", "interest_02" 等
        is_complete: 完成状态（True 表示完成，False 表示未完成）
        bot_config: 可选的 bot_config 对象，如果缓存不存在且提供此参数，会先初始化缓存

    返回:
        bool: 是否更新成功

    注意:
        - 如果缓存不存在，且提供了 bot_config，会先调用 get_all_stages_and_substages 初始化缓存
        - 如果缓存不存在且未提供 bot_config，会返回 False（更新失败）
    """
    cache_key = f"stages_substages:{conversation_id}"

    # 尝试从缓存读取现有数据
    cached = redis_get(cache_key)
    stages_dict = {}

    if cached:
        try:
            obj = json.loads(cached)
            if isinstance(obj, dict):
                # 转换字符串布尔值为真正的布尔值
                for k, v in obj.items():
                    if isinstance(v, bool):
                        stages_dict[k] = v
                    elif isinstance(v, str):
                        stages_dict[k] = v.lower() in ('true', '1', 'yes')
                    else:
                        stages_dict[k] = bool(v)
        except Exception:
            # 缓存解析失败，如果提供了 bot_config 则初始化
            pass

    # 如果缓存不存在且提供了 bot_config，先初始化缓存
    if not stages_dict and bot_config is not None:
        stages_dict = get_all_stages_and_substages(conversation_id, bot_config)

    # 如果仍然为空，说明无法获取数据，返回 False
    if not stages_dict:
        return False

    # 检查 substage_name 是否存在
    if substage_name not in stages_dict:
        # 如果不存在，可以选择添加它（为了容错）或者返回 False
        # 这里选择添加新的 substage
        pass

    # 更新对应 substage 的状态
    stages_dict[substage_name] = is_complete

    # 写回缓存（保持 90 天有效期）
    try:
        redis_set(cache_key, json.dumps(stages_dict, ensure_ascii=False), expired=7_776_000)
        return True
    except Exception:
        return False

def _has_content(val) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    if isinstance(val, (list, tuple, set)):
        return len(val) > 0
    if isinstance(val, dict):
        return len(val) > 0
    return True  # 其他类型按有值处理

def add_conversation_id_uuid_and_cache(results: dict, conversation_id: str, session_id: str, user_id: str, turn_uuid: str) -> None:
    """为 results 添加 turn_uuid 并将 response 缓存到 Redis（使用 turn_uuid 作为键）。

    参数:
        results: 包含 "response" 字段的字典
        conversation_id: 会话标识
        session_id: 会话标识
        user_id: 用户标识
        turn_uuid: 回合 UUID（用于配对 user 和 agent 消息）

    注意:
        - 如果 results 不是字典或不包含 "response" 字段，则不执行任何操作
        - 缓存键格式：response:{turn_uuid}，有效期 24 小时
        - 缓存失败不影响主流程
    """
    if not isinstance(results, dict) or "response" not in results:
        return

    # 在 results 对象中添加 turn_uuid 字段（用于后续 store_response_by_uuid 调用）
    results["turn_uuid"] = turn_uuid

    # 将 turn_uuid 和 response 的映射存入 Redis（临时存储，等待外部调用存储）
    # 缓存键格式：response:{turn_uuid}，有效期 24 小时
    cache_key = f"response:{turn_uuid}"
    response_data = {
        "response": results["response"],
        "conversation_id": conversation_id,
        "session_id": session_id,
        "user_id": user_id,
        "turn_uuid": turn_uuid
    }
    try:
        redis_set(cache_key, json.dumps(response_data, ensure_ascii=False), expired=86400)  # 24小时
    except Exception:
        # 缓存失败不影响主流程
        pass


def _extract_products_list(product_result: Any) -> List[Dict[str, Any]]:
    # 解析输入数据
    if isinstance(product_result, str):
        try:
            data = json.loads(product_result)
        except json.JSONDecodeError:
            return []
    else:
        data = product_result

    # 提取 products 列表（支持多种格式）
    products = []
    if isinstance(data, dict):
        # 如果包含 structuredContent，从中提取
        if "structuredContent" in data:
            structured = data.get("structuredContent", {})
            products = structured.get("products", [])
        # 如果直接包含 products
        elif "products" in data:
            products = data.get("products", [])
        # 如果 structuredContent 是对象属性
        elif hasattr(data, "structuredContent"):
            structured = getattr(data, "structuredContent", {})
            if isinstance(structured, dict):
                products = structured.get("products", [])
    elif isinstance(data, list):
        products = data
    else:
        # 尝试从对象属性获取
        if hasattr(data, "products"):
            products = getattr(data, "products", [])
        elif hasattr(data, "structuredContent"):
            structured = getattr(data, "structuredContent", {})
            if isinstance(structured, dict):
                products = structured.get("products", [])
        else:
            return []
    return products


def format_product_list_for_llm(
    product_result: Any,
    exclude_equity_keys: Optional[List[str]] = None,
    *,
    include_product_id: bool = True,
    include_equity_key: bool = True,
    filter_product_ids: Optional[List[str]] = None,
    add_header: bool = True,
) -> str:
    """将商品列表数据格式化为 LLM 易读的字符串格式。

    参数:
        product_result: 商品列表数据
        exclude_equity_keys: 需要排除的权益key列表
    """
    products = _extract_products_list(product_result)
    if not products:
        return "暂无商品信息"

    # 根据权益key过滤商品
    if exclude_equity_keys:
        filtered_products = []
        for item in products:
            product_equities = item.get("productEquities", [])
            # 检查商品是否包含需要排除的权益
            has_excluded_equity = any(
                equity.get("productEquityKey") in exclude_equity_keys
                for equity in product_equities
            )
            if not has_excluded_equity:
                filtered_products.append(item)
        products = filtered_products

    # 可选：只保留指定商品 ID（用于“只拼推荐商品”等场景）
    if filter_product_ids:
        normalized_ids = {
            str(pid).strip() for pid in filter_product_ids if str(pid).strip()
        }
        if normalized_ids:
            filtered_products = []
            for item in products:
                product = item.get("product", {})
                pid = str(product.get("outerId", "")).strip()
                if pid in normalized_ids:
                    filtered_products.append(item)
            products = filtered_products

    if not products:
        return "暂无商品信息"

    # 商品类型映射
    type_map = {
        1: "实物",
        2: "服务订阅",
        3: "人工服务",
        4: "单次服务"
    }

    # 权益类型映射
    equity_type_map = {
        0: "不限次",
        1: "次数"
    }

    formatted_lines: List[str] = []
    if add_header:
        formatted_lines.append("商品列表：")
        formatted_lines.append("=" * 60)

    for idx, item in enumerate(products, 1):
        product = item.get("product", {})
        product_id = product.get("outerId", "")
        product_extra = item.get("productExtra", {})
        product_equities = item.get("productEquities", [])

        # 基本信息
        product_name = product.get("name", "")
        product_type_code = product.get("type", 0)
        product_type = type_map.get(product_type_code, f"未知类型({product_type_code})")
        description = product_extra.get("description", "")
        currency = product.get("currency", "")

        # 价格处理
        price_range = product_extra.get("priceRangeString")
        if price_range and isinstance(price_range, list) and len(price_range) > 0:
            # 有价格范围
            if len(price_range) == 2:
                price_str = f"{price_range[0]} - {price_range[1]} {currency}"
            else:
                price_str = f"{price_range[0]} {currency}"
        else:
            # 使用单一价格
            price = product.get("price", 0)
            price_str = f"{price / 100 if price else 0:.2f} {currency}" if currency else f"{price / 100 if price else 0:.2f}"

        # 服务时间（仅服务类型显示）
        service_time = ""
        is_service = product_type_code in [2, 3, 4]  # 服务订阅、人工服务、单次服务
        if is_service:
            service_time_date = product_extra.get("serviceTimeDate")
            if service_time_date:
                if isinstance(service_time_date, list) and len(service_time_date) >= 2:
                    service_time = f"{service_time_date[0]} 至 {service_time_date[1]}"
                elif isinstance(service_time_date, list) and len(service_time_date) == 1:
                    service_time = str(service_time_date[0])
                else:
                    service_time = str(service_time_date)

        # 格式化商品信息
        formatted_lines.append(f"\n商品 {idx}:")
        if include_product_id:
            formatted_lines.append(f"  商品ID: {product_id}")
        formatted_lines.append(f"  商品名: {product_name}")
        formatted_lines.append(f"  商品类型: {product_type}")
        if description:
            formatted_lines.append(f"  商品介绍: {description}")

        knowledge_summary = product_extra.get("knowledgeContent")
        if isinstance(knowledge_summary, str) and knowledge_summary.strip():
            formatted_lines.append("  商品知识库要点:")
            for line in knowledge_summary.strip().splitlines():
                formatted_lines.append(f"    {line.strip()}")
        formatted_lines.append(f"  价格: {price_str}")

        # 服务时间（仅服务类型）
        if is_service and service_time:
            formatted_lines.append(f"  服务时间: {service_time}")

        # 商品权益（改为编号列表：1. 名称 - 次数，可选是否暴露权益Key）
        if product_equities:
            formatted_lines.append(f"  商品权益:")
            formatted_lines.append("")  # 空行分隔
            idx_counter = 1
            for equity in product_equities:
                equity_name = equity.get("name", "")
                equity_type_code = equity.get("type", 0)  # 0: 不限次, 1: 次数
                equity_amount = equity.get("amount", 0)
                equity_key = equity.get("productEquityKey", "")
                if equity_name:
                    if include_equity_key and equity_key:
                        equity_prefix = f"{equity_name} (权益Key: {equity_key})"
                    else:
                        equity_prefix = equity_name

                    if equity_type_code == 0:
                        formatted_lines.append(f"    {idx_counter}. {equity_prefix} - 不限次")
                    else:
                        formatted_lines.append(f"    {idx_counter}. {equity_prefix} - {equity_amount}次")
                    idx_counter += 1

        formatted_lines.append("-" * 60)

    return "\n".join(formatted_lines)


def build_promotion_product_snippet(product_result: Any, target_ids: List[str]) -> str:
    """
    根据商品 ID 列表构建推广商品信息片段。

    要求（用于 response LLM）：
    - **样式**：与 format_product_list_for_llm 中的商品块保持一致
    - **范围**：仅包含推荐的商品（target_ids）
    - **隐私**：不暴露商品ID和权益Key
    """
    if not target_ids:
        return ""

    return format_product_list_for_llm(
        product_result,
        exclude_equity_keys=None,
        include_product_id=False,
        include_equity_key=False,
        filter_product_ids=target_ids,
        add_header=False,
    )


# ===== Compliance helpers =====
def _get_compliance_modules(route_state_prompt_map: Any) -> list[Any]:
    if isinstance(route_state_prompt_map, dict):
        modules = route_state_prompt_map.get("compliance")
        if isinstance(modules, list):
            return modules
    return []


def find_compliance_module_index_by_product(route_state_prompt_map: Any, equity_key: Optional[str]) -> Optional[int]:
    """根据 equity_key（如 product_equity_23_4）在 compliance 模块列表中查找对应的模块索引。"""
    if not equity_key:
        return None
    modules = _get_compliance_modules(route_state_prompt_map)
    for idx, module in enumerate(modules):
        if isinstance(module, dict) and module.get("product") == equity_key:
            return idx
    return None


# 注意：产品缓存逻辑已迁移到 bot_mcp.get_products_with_cache


def skip_completed_stage(routed_current_state: str, routed_current_sub_stage: str, stages_complete: Dict[str, bool], bot_config: Any, conversation_id: str, allowed_stages: list[str] = None) -> tuple[str, str]:
    """如果路由到的阶段已完成，则跳到下一个未完成的阶段。

    参数:
        routed_current_state: 路由到的当前阶段（如 "cognition"）
        routed_current_sub_stage: 路由到的当前子阶段（如 "cognition_01"）
        stages_complete: 所有阶段的完成状态字典
        allowed_stages: 需要使用该机制的阶段列表（如 ["cognition", "interest"]）
        bot_config: bot 配置对象
        conversation_id: 会话标识（用于刷新 stages_complete）

    返回:
        tuple[str, str]: (更新后的 routed_current_state, routed_current_sub_stage)
    """
    # 如果 allowed_stages 为 None，使用默认值
    if allowed_stages is None:
        allowed_stages = ["cognition", "interest"]

    # 如果当前阶段不在允许列表中，直接返回原路由结果
    if routed_current_state not in allowed_stages:
        return routed_current_state, routed_current_sub_stage

    # 确保 stages_complete 是最新的（如果为空则重新获取）
    if not stages_complete:
        stages_complete = get_all_stages_and_substages(conversation_id, bot_config)

    # 构建完整的 substage 名称（如 "cognition_01"）
    if routed_current_sub_stage and '_' in routed_current_sub_stage:
        # 如果已经是完整格式，直接使用
        if routed_current_sub_stage.count('_') >= 2:
            routed_substage_full = routed_current_sub_stage
        else:
            # 提取数字部分并组合
            num_part = routed_current_sub_stage.split('_')[-1]
            routed_substage_full = f"{routed_current_state}_{num_part}"
    else:
        # 如果没有 substage，使用默认格式
        routed_substage_full = f"{routed_current_state}_01"

    # 检查路由到的阶段是否已完成
    if routed_substage_full in stages_complete and stages_complete.get(routed_substage_full, False):
        # 已完成，寻找下一个未完成的阶段
        stages_list = list(stages_complete.keys())

        # 找到当前路由阶段的索引
        if routed_substage_full in stages_list:
            current_idx = stages_list.index(routed_substage_full)
            # 从当前位置往后找第一个未完成的阶段
            found_next = False
            for idx in range(current_idx + 1, len(stages_list)):
                next_substage = stages_list[idx]
                # 检查该阶段是否也需要在 allowed_stages 中
                next_stage_name = next_substage.rsplit('_', 1)[0] if '_' in next_substage else next_substage
                # 如果下一个阶段不在允许列表中，跳过（不跳转到不允许的阶段）
                if next_stage_name not in allowed_stages:
                    continue
                if not stages_complete.get(next_substage, False):
                    # 找到未完成的阶段，更新路由结果
                    routed_substage_full = next_substage
                    routed_current_state = next_stage_name
                    routed_current_sub_stage = next_substage
                    found_next = True
                    break

            # 如果没找到后续未完成的阶段（可能所有后续阶段都已完成或不在允许列表中），则保持原路由结果
            if not found_next:
                # 保持原路由结果
                pass
        # 如果路由到的阶段不在列表中，保持原路由结果

    # 更新路由结果（确保格式一致）
    if routed_substage_full and '_' in routed_substage_full:
        routed_current_state = routed_substage_full.rsplit('_', 1)[0]
        routed_current_sub_stage = routed_substage_full

    return routed_current_state, routed_current_sub_stage


def validate_query_result(query: dict) -> bool:
    """
    规则：
    - 若存在 results.substage_results（且为 dict）：检查该 dict 的所有 value，任一为空 → False；否则 True
    - 否则：检查 results（dict）的所有 value，任一为空 → False；否则 True
    """
    results = query.get("results")
    if not isinstance(results, dict):
        return False

    substage_results = results.get("substage_results")
    if isinstance(substage_results, dict):
        for v in substage_results.values():
            if not _has_content(v):
                return False
        return True

    # 无 substage_results，检查 results 内所有值
    for v in results.values():
        if not _has_content(v):
            return False
    return True

# @mcp.tool(description="Main response procedure: process user input, route, optionally skip decision_making based on equity, and return JSON.")
# def main_response_procedure(
#     session_id: str,
#     user_id: str,
#     bot_id: str,
#     app_id: str,
#     user_input: Optional[str] = None,
#     equity: Optional[Union[str, Dict[str, Any]]] = None
# ) -> str:
#     """
#     主流程：用户输入 -> 分解 -> 响应
#     """
#     # 延迟导入，避免仅为了使用 join_prompts 等函数而触发 LLM 客户端初始化
#     # get_conversation_with_cache 会创建 conversation（如果不存在），所以 conversation_id 总是存在
#     conversation_json = db.get_conversation_with_cache(user_id, session_id)

#     # 解析 conversation JSON，提取 conversation_id
#     conversation_id = ""
#     try:
#         conv_data = json.loads(conversation_json)
#         if isinstance(conv_data, dict) and "id" in conv_data:
#             conversation_id = str(conv_data["id"])
#     except (json.JSONDecodeError, TypeError, KeyError):
#         # 解析失败，理论上不应该发生（get_conversation_with_cache 总是会创建 conversation）
#         # 但为了代码健壮性，保留错误处理
#         conversation_id = ""

#     # 从数据库读取 next_stage 和 next_sub_stage（通过 get_latest_session_state_payload）
#     # 注意：即使有 conversation_id，session_states 表中也可能没有记录（首次使用场景）
#     # get_latest_session_state_payload 在无记录时返回 "{}"
#     session_state_json = db.get_latest_session_state_payload(conversation_id, session_id, user_id)

#     # 处理 session_state_json 为空或无效的情况
#     # 可能的情况：
#     # 1. conversation_id 为空，session_state_json 为 ""（空字符串）
#     # 2. conversation_id 有值但 session_states 表中没有记录，返回 "{}"（空字典 JSON）
#     # 3. session_states 表有记录，返回包含数据的 JSON
#     session_state = {}
#     if session_state_json:
#         try:
#             parsed = json.loads(session_state_json)
#             if isinstance(parsed, dict):
#                 session_state = parsed
#         except (json.JSONDecodeError, TypeError):
#             # JSON 解析失败，使用空字典
#             session_state = {}

#     # 从 session_state 读取 next_stage 和 next_substage，如果没有则使用默认值
#     # 如果 session_states 表中没有记录（首次使用），使用默认初始值
#     next_stage = session_state.get("next_stage") or "cognition"
#     next_substage = session_state.get("next_sub_stage") or "cognition_01"

#     # 使用读取的值作为 current_state 和 current_sub_stage（用于路由等逻辑）
#     current_state = next_stage
#     current_sub_stage = next_substage

#     #客户阶段一： 认知 - 新conversation_id
#     #if current_state == "cognition": #用户不输入目的只回复，则进入路由
#         #session_choise = router.route_and_store(conversation_id, session_id, user_id, user_input, table = "chat_messages", state_table = "session_states")
#     # 获取历史对话消息列表（返回 JSON 字符串格式，如 "[]" 或 "[{...}]"）
#     history_context = db.list_chat_messages(conversation_id, session_id, user_id)
#     # 解析 equity 参数（形如 {"info": {"商品A.权益1": 1, "商品A.权益2": 0}}）
#     equity_info = {}
#     if equity:
#         if isinstance(equity, str):
#             try:
#                 equity_obj = json.loads(equity)
#             except json.JSONDecodeError:
#                 equity_obj = {}
#         else:
#             equity_obj = equity
#         if isinstance(equity_obj, dict):
#             info_part = equity_obj.get("info")
#             if isinstance(info_part, dict):
#                 equity_info = info_part
#             else:
#                 # 兼容直接传入 info 字典的情况（无外层 {"info": ...}）
#                 candidate_keys = [
#                     k for k in equity_obj.keys()
#                     if isinstance(k, str) and ("." in k or "product_equity" in k)
#                 ]
#                 numeric_values = {
#                     k: v for k, v in equity_obj.items()
#                     if isinstance(v, (int, float))
#                 }
#                 if candidate_keys and numeric_values:
#                     equity_info = numeric_values
#     products_cached = get_products_with_cache(conversation_id, outer_id=bot_id, app_id=app_id)  # 开头获取商品列表并使用缓存（90天）
#     bot_config = get_bot_config(conversation_id, bot_id, app_id)
#     structuredContent = bot_config.structuredContent
#     # 兼容缺失或类型不符的情况，避免 KeyError/类型错误
#     route_state_prompt_map = structuredContent.get('routeStateStrategies') if isinstance(structuredContent, dict) else None
#     if not isinstance(route_state_prompt_map, dict):
#         route_state_prompt_map = {}
#         if isinstance(structuredContent, dict):
#             structuredContent['routeStateStrategies'] = route_state_prompt_map

#     # # 确保 'questions' 阶段存在，如果不存在则添加默认配置
#     # if 'questions' not in route_state_prompt_map:
#     #     route_state_prompt_map['questions'] = [{
#     #         "purpose": "处理用户查询，根据查询是否与业务相关（如服务、产品等）生成回答或委婉拒绝回复，并确保回复自然且不超过100字。",
#     #         "name": ["response"],
#     #         "expect": ["自然回复用户的内容"],
#     #         "operation": [
#     #             "1. 接收并分析用户查询内容",
#     #             "2. 判断查询是否与业务相关",
#     #             "3. 如果相关，检索业务信息并生成回答",
#     #             "4. 如果不相关，生成委婉拒绝消息",
#     #             "5. 优化回复确保自然流畅且不超过100字"
#     #         ]
#     #     }]

#     character_prompt = structuredContent['character']
#     stages_complete = get_all_stages_and_substages(conversation_id, bot_config)
#     # botLLMConfig 兜底：如果 key 不存在或值为空，则使用默认的 qwen 配置
#     botLLMConfig = structuredContent.get('botLLMConfig')
#     if not botLLMConfig or not isinstance(botLLMConfig, dict):
#         botLLMConfig = DEFAULT_BOT_LLM_CONFIG.copy()
#     # if purpose:
#     #     #储存dummy用户输入
#     #     dt = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
#     #     inserted = db.store_chat_message(session_id=session_id, user_id=user_id, conversation_id=conversation_id, content="用户目的：" + purpose, role="user", dt=dt)
#     #     msg = json.loads(inserted)
#     #     message_id = msg["id"]
#     #     #state储存目的
#     #     stage_payload = {"cognition_01": {"info": {"purpose": purpose}}}
#     #     db.store_session_state(message_id=message_id, conversation_id=conversation_id, session_id=session_id, user_id=user_id, current_state="cognition", stage_payload_json=json.dumps(stage_payload, ensure_ascii=False), dt=dt, table="session_states")
#     #     #知识库匹配牌阵/mcp获取牌阵?

#     #     #获取Prompt组装（使用 join_prompts）
#     #     cog_module = route_state_prompt_map["cognition"][0]
#     #     cognition_prompt = join_prompts(cog_module)
#     #     full_prompt = "\n\n".join([character_prompt, cognition_prompt])

#     #     # 记录 stage_prompt（purpose 分支）
#     #     logger.info(f"[Stage Prompt] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#     #     logger.info(f"[Stage Prompt] purpose={purpose}, current_state=cognition, current_sub_stage=cognition_01")
#     #     logger.info(f"[Stage Prompt] stage_prompt length={len(cognition_prompt)} characters")
#     #     logger.info(f"[Stage Prompt] stage_prompt content:\n{cognition_prompt}")

#     #     #LLM尝试进一步追问客户具体问题
#     #     results = llm_generic(full_prompt = full_prompt, user_input = "用户占卜类别：" + purpose, history_context = history_context, session_state = session_state, botLLMConfig=botLLMConfig)

#     #     # 解析 results（可能是 JSON 字符串）
#     #     if isinstance(results, str):
#     #         try:
#     #             results = json.loads(results)
#     #         except json.JSONDecodeError:
#     #             # 如果不是 JSON，保持原样
#     #             pass

#     #     # 为 results 添加 turn_uuid 并缓存 response（purpose 分支暂时保留原逻辑）
#     #     # : purpose 分支也需要改为使用 turn_uuid 机制
#     #     turn_uuid_purpose = str(uuid.uuid4())
#     #     add_conversation_id_uuid_and_cache(results, conversation_id, session_id, user_id, turn_uuid_purpose)

#     #     # 检查 results 的每一项是否都不为空或空字符串
#     #     if isinstance(results, dict):
#     #         all_filled = True
#     #         for key, value in results.items():
#     #             # 跳过 turn_uuid 字段
#     #             if key == "turn_uuid":
#     #                 continue
#     #             # 检查值是否为空或空字符串
#     #             if value is None or (isinstance(value, str) and value.strip() == ""):
#     #                 all_filled = False
#     #                 break
#     #             # 如果是字典，检查字典内的值
#     #             if isinstance(value, dict):
#     #                 for v in value.values():
#     #                     if v is None or (isinstance(v, str) and v.strip() == ""):
#     #                         all_filled = False
#     #                         break
#     #                 if not all_filled:
#     #                     break

#     #         # 如果所有字段都非空，设置当前阶段为完成，并设置下一阶段
#     #         if all_filled:
#     #             # 当前阶段的 substage（purpose 分支对应 cognition_01）
#     #             current_substage = "cognition_01"

#     #             # 设置当前阶段为完成
#     #             set_all_stages_and_substages(conversation_id, current_substage, True, bot_config)

#     #             # 获取所有 stages，找到下一个
#     #             stages_complete = get_all_stages_and_substages(conversation_id, bot_config)
#     #             stages_list = list(stages_complete.keys())

#     #             # 找到当前 substage 的索引
#     #             if current_substage in stages_list:
#     #                 current_idx = stages_list.index(current_substage)
#     #                 # 获取下一个 substage
#     #                 if current_idx + 1 < len(stages_list):
#     #                     next_substage = stages_list[current_idx + 1]
#     #                     # 提取 next_stage（数字之前的英文部分）
#     #                     next_stage = next_substage.rsplit('_', 1)[0] if '_' in next_substage else next_substage
#     #                 else:
#     #                     # 已经是最后一个，保持当前
#     #                     next_stage = current_substage.rsplit('_', 1)[0] if '_' in current_substage else current_substage
#     #                     next_substage = current_substage
#     #             else:
#     #                 # 如果找不到当前 substage，默认设置
#     #                 next_stage = "cognition"
#     #                 next_substage = "cognition_01"
#     #         else:
#     #             # 有字段为空，保持当前阶段
#     #             next_stage = "cognition"
#     #             next_substage = "cognition_01"

#     #     # 检查 stages_complete 的最后一项是否为 True（所有阶段都完成）
#     #     stages_list = list(stages_complete.keys())
#     #     if stages_list and stages_complete.get(stages_list[-1], False):
#     #         # 所有阶段都完成，结束当前会话并创建新会话
#     #         db.end_conversation_and_create(user_id=user_id, session_id=session_id)
#     #         # 返回到最开始
#     #         next_stage = "cognition"
#     #         next_substage = "cognition_01"

#     #     # 更新数据库中的 next_stage 和 next_sub_stage（只更新这两个字段，其他字段继承最新一条）
#     #     latest_state = json.loads(db.get_latest_session_state_payload(conversation_id, session_id, user_id))
#     #     if latest_state and "id" in latest_state:
#     #         with db._connect() as conn:
#     #             with conn.cursor() as cur:
#     #                 cur.execute(
#     #                     'UPDATE "session_states" SET next_stage = %s, next_substage = %s WHERE id = %s',
#     #                     [next_stage, next_substage, latest_state["id"]]
#     #                 )
#     #             conn.commit()

#     #     # 转换为 JSON 字符串（MCP tool 需要返回字符串，不再包含 next_stage 和 next_substage）
#     #     result_dict = results if isinstance(results, dict) else {"response": results}
#     #     return json.dumps(result_dict, ensure_ascii=False)
#     if user_input: #用户不输入目的只回复，则进入路由
#         # 入口处：生成 UUID 并写入 pending_turns
#         dt_user = str(int(datetime.now(timezone.utc).timestamp()))
#         turn_uuid = str(uuid.uuid4())
#         user_message_uuid = str(uuid.uuid4())

#         # 写入 pending_turns 基本信息
#         db.store_pending_turn(
#             turn_uuid=turn_uuid,
#             user_message_uuid=user_message_uuid,
#             conversation_id=conversation_id,
#             session_id=session_id,
#             user_id=user_id,
#             dt_user=dt_user,
#             user_content=user_input,
#             bot_id=bot_id,
#             app_id=app_id
#         )

#         # 同时缓存到 Redis（冗余）
#         try:
#             pending_cache_key = f"pending:{turn_uuid}"
#             pending_cache_data = {
#                 "turn_uuid": turn_uuid,
#                 "user_message_uuid": user_message_uuid,
#                 "conversation_id": conversation_id,
#                 "session_id": session_id,
#                 "user_id": user_id,
#                 "dt_user": dt_user,
#                 "user_content": user_input,
#                 "bot_id": bot_id,
#                 "app_id": app_id
#             }
#             redis_set(pending_cache_key, json.dumps(pending_cache_data, ensure_ascii=False), expired=86400)  # 24小时
#         except Exception:
#             pass

#         # 记录输入给 router 的 current_state
#         logger.info(f"[Router Input] ========== 输入给 Router 的状态 ==========")
#         logger.info(f"[Router Input] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#         logger.info(f"[Router Input] current_state={current_state}, current_sub_stage={current_sub_stage}")
#         logger.info(f"[Router Input] user_input: {user_input}")
#         logger.info(f"[Router Input] ==========================================")

#         router_response = router.route_and_store(conversation_id, session_id, user_id, user_input, bot_id, app_id, current_state)
#         parsed = router_response.get("llm_output")

#         # 记录 router 响应
#         logger.info(f"[Router Response] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#         logger.info(f"[Router Response] router_response: {json.dumps(router_response, ensure_ascii=False, indent=2)}")

#         # 计算 current_state / current_sub_stage 以及动态 stage 列（支持多模块写入）
#         # route_and_store 返回的 llm_output 格式为 {"queries": [{"stage": "...", "substage": "..."}]}
#         routed_current_state = ""
#         routed_current_sub_stage = ""

#         if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list) and parsed["queries"]:
#             # 取第一个 query 的 stage 和 substage
#             first_query = parsed["queries"][0]
#             if isinstance(first_query, dict):
#                 routed_current_state = first_query.get("stage", "")
#                 routed_current_sub_stage = first_query.get("substage", "")

#         # 如果解析失败，使用默认值
#         if not routed_current_state:
#             routed_current_state = current_state
#         if not routed_current_sub_stage:
#             routed_current_sub_stage = current_sub_stage

#         # 如果路由到 compliance，检查是否有对应子阶段的权益剩余
#         def _handle_compliance_no_equity_flow(current_state_hint: str, substage_hint: str) -> str:
#             """当没有剩余权益时，执行收尾流程并返回结束话术。"""
#             nonlocal stages_complete

#             if substage_hint and '_' in substage_hint:
#                 if substage_hint.count('_') >= 2:
#                     compliance_substage = substage_hint
#                 else:
#                     num_part = substage_hint.split('_')[-1]
#                     compliance_substage = f"{current_state_hint}_{num_part}"
#             else:
#                 compliance_substage = "compliance_01"

#             try:
#                 set_all_stages_and_substages(conversation_id, compliance_substage, True, bot_config)
#             except Exception:
#                 pass

#             try:
#                 stage_payload = {compliance_substage: {"info": {"status": "no_equity_remaining"}}}
#                 db.update_pending_turn_state(
#                     turn_uuid=turn_uuid,
#                     routed_current_state="compliance",
#                     routed_current_sub_stage=compliance_substage,
#                     stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False)
#                 )
#                 db.update_pending_turn_state(
#                     turn_uuid=turn_uuid,
#                     next_stage="compliance",
#                     next_substage=compliance_substage
#                 )
#             except Exception:
#                 pass

#             try:
#                 db.end_conversation_and_create(user_id=user_id, session_id=session_id)
#             except Exception:
#                 pass

#             no_equity_prompt = "\n".join([
#                 "[任务]",
#                 "当前订阅权益已经使用完毕。请以亲切、感谢的语气，向用户说明权益已结清，真诚致谢这段陪伴，并邀请对方日后如有需要随时再来交流或续订服务。",
#                 "",
#                 "[补充要求]",
#                 "1. 保持角色设定，语气温暖、柔和。",
#                 "2. 不提及系统或技术细节，只谈服务体验。",
#                 "3. 鼓励用户在需要时继续提问或选择新的服务。"
#             ])
#             farewell_prompt = "\n\n".join([character_prompt, no_equity_prompt])
#             try:
#                 farewell_raw = llm_generic(
#                     full_prompt=farewell_prompt,
#                     user_input="",
#                     history_context=history_context,
#                     session_state=session_state,
#                     botLLMConfig=botLLMConfig
#                 )
#             except Exception:
#                 farewell_raw = ""

#             farewell_text = ""
#             if isinstance(farewell_raw, dict):
#                 farewell_text = str(farewell_raw.get("response") or "").strip()
#             elif isinstance(farewell_raw, str):
#                 try:
#                     parsed_farewell = json.loads(farewell_raw)
#                     if isinstance(parsed_farewell, dict):
#                         farewell_text = str(parsed_farewell.get("response") or "").strip()
#                     else:
#                         farewell_text = farewell_raw.strip()
#                 except (json.JSONDecodeError, TypeError):
#                     farewell_text = farewell_raw.strip()

#             if not farewell_text:
#                 farewell_text = "本次订阅权益已全部使用完毕，聊聊就先到这里。若需要继续服务，欢迎随时再来。"

#             results = {
#                 "response": farewell_text,
#                 "info": {"status": "no_equity_remaining"}
#             }
#             add_conversation_id_uuid_and_cache(results, conversation_id, session_id, user_id, turn_uuid)
#             return json.dumps(results, ensure_ascii=False)

#         if routed_current_state == "compliance":
#             # 解析子阶段号（如 "compliance_02" → idx=1）
#             target_idx = None
#             if routed_current_sub_stage and isinstance(routed_current_sub_stage, str) and "_" in routed_current_sub_stage:
#                 try:
#                     num_part = routed_current_sub_stage.split("_")[-1]
#                     target_idx = int(num_part) - 1  # compliance_01 → 0, compliance_02 → 1
#                 except (ValueError, IndexError):
#                     target_idx = None

#             compliance_modules = _get_compliance_modules(route_state_prompt_map)

#             # 优先：根据当前 substage 所对应模块的 product 字段查找 equity
#             equity_key = None
#             if (
#                 target_idx is not None
#                 and 0 <= target_idx < len(compliance_modules)
#                 and isinstance(equity_info, dict)
#             ):
#                 module = compliance_modules[target_idx]
#                 if isinstance(module, dict):
#                     module_product = module.get("product")
#                     amount = equity_info.get(module_product) if module_product else None
#                     if isinstance(amount, (int, float)) and amount > 0:
#                         equity_key = module_product

#             # 兜底：遍历所有 equity key，找到与模块 product 对应且仍有剩余次数的第一条
#             if not equity_key and isinstance(equity_info, dict):
#                 for key, amount in equity_info.items():
#                     if not isinstance(amount, (int, float)) or amount <= 0:
#                         continue
#                     module_idx = find_compliance_module_index_by_product(route_state_prompt_map, key)
#                     if module_idx is not None:
#                         equity_key = key
#                         routed_current_sub_stage = f"compliance_{module_idx + 1:02d}"
#                         break

#             # 如果找到有效的 equity_key，直接调用 service_compliance_response 并返回
#             if equity_key:
#                 try:
#                     return service_compliance_response(
#                         session_id=session_id,
#                         user_id=user_id,
#                         bot_id=bot_id,
#                         app_id=app_id,
#                         turn_uuid=turn_uuid,
#                         equity_key=equity_key
#                     )
#                 except Exception as _e:
#                     return json.dumps({
#                         "error": f"failed to invoke service_compliance_response: {str(_e)}",
#                         "turn_uuid": turn_uuid
#                     }, ensure_ascii=False)
#             else:
#                 return _handle_compliance_no_equity_flow(routed_current_state, routed_current_sub_stage)

#         # 在进入 decision_making 之前：若 equity.info 任一项非 0，则跳过 decision_making，直接进入 compliance，
#         # 同时将 decision_making 视为已完成（标记首个 decision_making 子阶段为完成）
#         should_skip_to_compliance = False
#         equity_selections = []
#         if equity_info:
#             for v in equity_info.values():
#                 # 只要有任意一个值为非零数字，则触发跳过
#                 if isinstance(v, (int, float)) and v != 0:
#                     should_skip_to_compliance = True
#                     break
#             # 收集非零权益项，作为仅用于透传的临时数据（不入 session_state / 不进入 LLM 上下文）
#             for k, v in equity_info.items():
#                 if isinstance(v, (int, float)) and v != 0:
#                     equity_selections.append({
#                         "key": k,
#                         "amount": v
#                     })
#         if should_skip_to_compliance:
#             # 仅当本轮路由到了 decision_making 时才判断并跳过
#             if routed_current_state == "decision_making":
#                 # 标记 decision_making 阶段完成（所有 decision_making_* 子阶段）
#                 if stages_complete:
#                     for key in list(stages_complete.keys()):
#                         if isinstance(key, str) and key.startswith("decision_making_"):
#                             try:
#                                 set_all_stages_and_substages(conversation_id, key, True, bot_config)
#                             except Exception:
#                                 # 容错，不阻断主流程
#                                 pass
#                 # 跳到 compliance
#                 routed_current_state = "compliance"
#                 routed_current_sub_stage = "compliance_01"
#                 # 强制路由至履约：直接调用 service_compliance_response 并返回，不再在本流程内调用 LLM
#                 try:
#                     # 选择一个用于匹配策略的 equity_key（非零的第一条）
#                     equity_key = None
#                     if equity_selections:
#                         for _it in equity_selections:
#                             if isinstance(_it, dict) and isinstance(_it.get("amount"), (int, float)) and _it.get("amount") != 0:
#                                 equity_key = _it.get("key")
#                                 if isinstance(equity_key, str) and equity_key:
#                                     break
#                     return service_compliance_response(
#                         session_id=session_id,
#                         user_id=user_id,
#                         bot_id=bot_id,
#                         app_id=app_id,
#                         turn_uuid=turn_uuid,
#                         equity_key=equity_key
#                     )
#                 except Exception as _e:
#                     return json.dumps({
#                         "error": f"failed to invoke service_compliance_response: {str(_e)}",
#                         "turn_uuid": turn_uuid
#                     }, ensure_ascii=False)

#         # 检查路由到的阶段是否已完成，如果已完成则跳到下一个未完成的阶段
#         # 只有 cognition 和 interest 阶段使用此机制（默认）
#         # routed_current_state, routed_current_sub_stage = skip_completed_stage(
#         #     routed_current_state=routed_current_state,
#         #     routed_current_sub_stage=routed_current_sub_stage,
#         #     stages_complete=stages_complete,
#         #     bot_config=bot_config,
#         #     conversation_id=conversation_id
#         # )

#         # 若路由结果为 compliance，则直接打断当前流程，调用履约服务并返回
#         if routed_current_state == "compliance" and not routed_current_sub_stage:
#             # 基于传入的 equity 信息选择一个用于策略匹配的 key（非零的第一条）；无则传 None
#             equity_key = None
#             equity_has_remaining = False
#             if equity_info:
#                 for _k, _v in equity_info.items():
#                     if isinstance(_v, (int, float)) and _v != 0:
#                         equity_key = _k
#                         equity_has_remaining = True
#                         break
#             if not equity_has_remaining:
#                 return _handle_compliance_no_equity_flow(routed_current_state, routed_current_sub_stage)
#             try:
#                 return service_compliance_response(
#                     session_id=session_id,
#                     user_id=user_id,
#                     bot_id=bot_id,
#                     app_id=app_id,
#                     turn_uuid=turn_uuid,
#                     equity_key=equity_key
#                 )
#             except Exception as _e:
#                 return json.dumps({
#                     "error": f"failed to invoke service_compliance_response: {str(_e)}",
#                     "turn_uuid": turn_uuid
#                 }, ensure_ascii=False)

#         #获取Prompt组装（使用 join_prompts）
#         # 从 substage 中提取编号（例如 "cognition_01" -> 01 -> 索引 0）
#         if routed_current_sub_stage and '_' in routed_current_sub_stage:
#             # 提取数字部分（例如从 "cognition_01" 提取 "01"）
#             num_part = routed_current_sub_stage.split('_')[-1]
#             # 转换为索引（从 1 开始，所以要减 1）
#             module_idx = int(num_part) - 1
#         else:
#             # 如果没有 substage 或格式不对，默认使用第一个模块
#             module_idx = 0

#         # 获取对应 stage 的模块列表，然后获取对应索引的模块
#         modules = route_state_prompt_map.get(routed_current_state, [])
#         if modules and module_idx < len(modules):
#             cog_module = modules[module_idx]
#         else:
#             # 如果索引超出范围，使用第一个模块
#             if modules:
#                 cog_module = modules[0]
#             else:
#                 # 如果获取结果为空，兜底使用 questions_01
#                 questions_modules = route_state_prompt_map.get('questions', [])
#                 if questions_modules:
#                     cog_module = questions_modules[0]
#                     logger.warning(f"[兜底] 未找到阶段 '{routed_current_state}' 的模块，使用 questions_01 作为兜底")
#                 else:
#                     cog_module = {}
#                     logger.error(f"[错误] 未找到阶段 '{routed_current_state}' 的模块，且 questions 模块也不存在")

#         stage_prompt = join_prompts(cog_module)
#         # 追加上一步/下一步任务 purpose（若可推断）
#         prev_step_text = ""
#         next_step_text = ""
#         try:
#             # 计算当前完整 substage 名称（如 cognition_01）
#             if routed_current_sub_stage and '_' in routed_current_sub_stage:
#                 if routed_current_sub_stage.count('_') >= 2:
#                     current_substage_full = routed_current_sub_stage
#                 else:
#                     num_part = routed_current_sub_stage.split('_')[-1]
#                     current_substage_full = f"{routed_current_state}_{num_part}"
#             else:
#                 current_substage_full = f"{routed_current_state}_01"
#             stages_list = list(stages_complete.keys()) if isinstance(stages_complete, dict) else []
#             # 上一步任务：取全局顺序中当前子阶段之前的第一个合法子阶段
#             try:
#                 prev_module = None
#                 if current_substage_full in stages_list:
#                     cur_idx = stages_list.index(current_substage_full)
#                     for j in range(cur_idx - 1, -1, -1):
#                         substage_name = stages_list[j]
#                         stage_name = substage_name.rsplit('_', 1)[0] if '_' in substage_name else substage_name
#                         if stage_name in ("questions", "after_sales"):
#                             continue
#                         stage_modules = route_state_prompt_map.get(stage_name, [])
#                         if stage_modules:
#                             prev_module = stage_modules[0]
#                             break
#                 if isinstance(prev_module, dict):
#                     prev_purpose = str(prev_module.get("purpose", "")).strip()
#                     if prev_purpose:
#                         prev_step_text = "\n\n".join(["[上一步任务]", prev_purpose])
#             except Exception:
#                 prev_step_text = ""

#             # 从全局 stages_complete 顺序中选取"当前子阶段之后"的下一个子阶段
#             # 在 stages_complete 顺序中找到下一个未必完成的子阶段，挑第一个可用模块
#             next_module = None
#             if current_substage_full in stages_list:
#                 cur_idx = stages_list.index(current_substage_full)
#                 for j in range(cur_idx + 1, len(stages_list)):
#                     substage_name = stages_list[j]
#                     # 过滤不可选阶段
#                     stage_name = substage_name.rsplit('_', 1)[0] if '_' in substage_name else substage_name
#                     if stage_name in ("questions", "after_sales"):
#                         continue
#                     stage_modules = route_state_prompt_map.get(stage_name, [])
#                     if stage_modules:
#                         next_module = stage_modules[0]
#                         break
#             # 取 purpose 并拼接
#             if isinstance(next_module, dict):
#                 next_purpose = str(next_module.get("purpose", "")).strip()
#                 if next_purpose:
#                     next_step_text = "\n\n".join(["[下一步任务]", next_purpose])
#         except Exception:
#             # 不影响主流程
#             prev_step_text = ""
#             next_step_text = ""
#         # 注意顺序：当前阶段 -> 上一步任务 -> 下一步任务
#         full_prompt = "\n\n".join([p for p in [character_prompt, stage_prompt, prev_step_text, next_step_text] if p])

#         # 记录 stage_prompt
#         logger.info(f"[Stage Prompt] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#         logger.info(f"[Stage Prompt] routed_current_state={routed_current_state}, routed_current_sub_stage={routed_current_sub_stage}")
#         logger.info(f"[Stage Prompt] stage_prompt length={len(stage_prompt)} characters")
#         logger.info(f"[Stage Prompt] stage_prompt content:\n{stage_prompt}")
#         if prev_step_text and prev_step_text.strip() != "[上一步任务]":
#             logger.info(f"[Prev Step] length={len(prev_step_text)} characters")
#             logger.info(f"[Prev Step] content:\n{prev_step_text}")
#         # 仅在包含有效 purpose 时打印"下一步任务"，避免只打印标题
#         if next_step_text and next_step_text.strip() != "[下一步任务]":
#             logger.info(f"[Next Step] length={len(next_step_text)} characters")
#             logger.info(f"[Next Step] content:\n{next_step_text}")
#         # 原样输出：full_prompt 去掉 character_prompt 的部分（仅包含 stage_prompt 与下一步任务）
#         full_prompt_wo_character = "\n\n".join([p for p in [stage_prompt, prev_step_text, next_step_text] if p])
#         logger.info(f"[Full Prompt Without Character]\n{full_prompt_wo_character}")

#         # 检查 stage_prompt 是否提到商品列表，或是 decision_making 阶段，若是则拼接开头获取的商品列表
#         user_input_for_llm = user_input
#         if "商品列表" in stage_prompt or routed_current_state == "decision_making":
#             if products_cached:
#                 product_lists_str = format_product_list_for_llm(products_cached)
#                 user_input_for_llm = (
#                     f"用户本轮输入：{user_input}\n\n"
#                     f"【商品列表供参考】\n{product_lists_str}"
#                 )

#         # 调用 LLM 生成响应
#         results = llm_generic(full_prompt=full_prompt, user_input=user_input_for_llm, history_context=history_context, session_state=session_state, botLLMConfig=botLLMConfig)

#         # 解析 results（可能是 JSON 字符串）
#         if isinstance(results, str):
#             try:
#                 results = json.loads(results)
#             except json.JSONDecodeError:
#                 # 如果不是 JSON，保持原样
#                 pass

#         # 后处理：从 info 中提取对话内容并合并到 response
#         if isinstance(results, dict):
#             results = extract_conversational_content_from_info(results)
#         elif isinstance(results, str):
#             # 如果 results 是字符串，转换为字典格式
#             results = {"response": results}

#         # 为 results 添加 turn_uuid 并缓存 response
#         # 确保 results 是字典且包含 response 字段
#         if isinstance(results, dict):
#             if "response" not in results:
#                 # 如果没有 response 字段，尝试从字符串中提取或设置默认值
#                 results["response"] = str(results) if results else ""
#             add_conversation_id_uuid_and_cache(results, conversation_id, session_id, user_id, turn_uuid)

#         # 记录包含 turn_uuid 的最终 results
#         logger.info(f"[LLM Results] ========== LLM 返回的完整结果（含 turn_uuid）==========")
#         logger.info(f"[LLM Results] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#         logger.info(f"[LLM Results] routed_current_state={routed_current_state}, routed_current_sub_stage={routed_current_sub_stage}")
#         logger.info(f"[LLM Results] turn_uuid={results.get('turn_uuid') if isinstance(results, dict) else 'N/A'}")
#         logger.info(f"[LLM Results] results: {json.dumps(results, ensure_ascii=False, indent=2)}")
#         logger.info(f"[LLM Results] ==========================================")

#         # 如果 results 包含 info 字段，更新 pending_turns 的状态信息
#         if isinstance(results, dict) and "info" in results:
#             # 构建 substage 名称（如 "cognition_01"）
#             if routed_current_sub_stage and '_' in routed_current_sub_stage:
#                 # 如果已经是完整格式（如 "cognition_01"），直接使用
#                 if routed_current_sub_stage.count('_') >= 2:
#                     substage_col = routed_current_sub_stage
#                 else:
#                     # 提取数字部分并组合
#                     num_part = routed_current_sub_stage.split('_')[-1]
#                     substage_col = f"{routed_current_state}_{num_part}"
#             else:
#                 # 如果没有 substage，使用默认格式
#                 substage_col = f"{routed_current_state}_01"

#             # 构建 stage_payload 格式：{"cognition_01": {"info": {...}}}
#             stage_payload = {substage_col: {"info": results["info"]}}

#             # 标注位置：更新 pending_turns 的状态信息（不再直接写入 session_states）
#             db.update_pending_turn_state(
#                 turn_uuid=turn_uuid,
#                 routed_current_state=routed_current_state,
#                 routed_current_sub_stage=routed_current_sub_stage if routed_current_sub_stage else f"{routed_current_state}_01",
#                 stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False)
#             )

#         # 初始化 next_stage 和 next_substage（默认保持当前阶段）
#         next_stage = routed_current_state
#         next_substage = routed_current_sub_stage if routed_current_sub_stage else f"{routed_current_state}_01"

#         # 检查 results 的每一项是否都不为空或空字符串
#         if isinstance(results, dict):
#             all_filled = True
#             for key, value in results.items():
#                 # 跳过 turn_uuid 字段
#                 if key == "turn_uuid":
#                     continue
#                 # 检查值是否为空或空字符串
#                 if value is None or (isinstance(value, str) and value.strip() == ""):
#                     all_filled = False
#                     break
#                 # 如果是字典，检查字典内的值
#                 if isinstance(value, dict):
#                     for v in value.values():
#                         if v is None or (isinstance(v, str) and v.strip() == ""):
#                             all_filled = False
#                             break
#                     if not all_filled:
#                         break

#             # 如果所有字段都非空，设置当前阶段为完成，并设置下一阶段
#             if all_filled and routed_current_sub_stage:
#                 # 构建当前阶段的 substage 名称
#                 # 如果 routed_current_sub_stage 已经是完整格式（如 "interest_02"），直接使用
#                 # 否则组合成完整格式
#                 if '_' in routed_current_sub_stage and routed_current_sub_stage.count('_') >= 2:
#                     # 已经是完整格式，直接使用
#                     current_substage = routed_current_sub_stage
#                 else:
#                     # 从 routed_current_state 和 routed_current_sub_stage 构建
#                     # 提取数字部分（如果有）
#                     if '_' in routed_current_sub_stage:
#                         num_part = routed_current_sub_stage.split('_')[-1]
#                         current_substage = f"{routed_current_state}_{num_part}"
#                     else:
#                         # 如果没有数字部分，使用默认格式
#                         current_substage = f"{routed_current_state}_01"

#                 # 设置当前阶段为完成
#                 set_all_stages_and_substages(conversation_id, current_substage, True, bot_config)

#                 # 获取所有 stages，找到下一个
#                 stages_complete = get_all_stages_and_substages(conversation_id, bot_config)
#                 stages_list = list(stages_complete.keys())

#                 # 找到当前 substage 的索引
#                 if current_substage in stages_list:
#                     current_idx = stages_list.index(current_substage)
#                     # 获取下一个 substage
#                     if current_idx + 1 < len(stages_list):
#                         next_substage = stages_list[current_idx + 1]
#                         # 提取 next_stage（数字之前的英文部分）
#                         next_stage = next_substage.rsplit('_', 1)[0] if '_' in next_substage else next_substage
#                     else:
#                         # 已经是最后一个，保持当前
#                         next_stage = current_substage.rsplit('_', 1)[0] if '_' in current_substage else current_substage
#                         next_substage = current_substage

#         # 检查 stages_complete 的最后一项是否为 True（所有阶段都完成）
#         stages_list = list(stages_complete.keys())
#         if stages_list and stages_complete.get(stages_list[-2], False): #-2为compliance
#             # 所有阶段都完成，结束当前会话并创建新会话
#             db.end_conversation_and_create(user_id=user_id, session_id=session_id)
#             # 返回到最开始
#             next_stage = "cognition"
#             next_substage = "cognition_01"

#         # 更新 pending_turns 的 next_stage 和 next_substage
#         db.update_pending_turn_state(
#             turn_uuid=turn_uuid,
#             next_stage=next_stage,
#             next_substage=next_substage
#         )

#         # 记录最后设定的 next_stage 和 next_substage
#         logger.info(f"[Final Next Stage] ========== 最终设定的下一阶段 ==========")
#         logger.info(f"[Final Next Stage] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
#         logger.info(f"[Final Next Stage] next_stage={next_stage}, next_substage={next_substage}")
#         logger.info(f"[Final Next Stage] ==========================================")

#         # 不再更新 session_states 的 next_stage/next_substage，等 store_response_by_uuid 时统一处理
#         # latest_state = json.loads(db.get_latest_session_state_payload(conversation_id, session_id, user_id))
#         # if latest_state and "id" in latest_state:
#         #     with db._connect() as conn:
#         #         with conn.cursor() as cur:
#         #             cur.execute(
#         #                 'UPDATE "session_states" SET next_stage = %s, next_substage = %s WHERE id = %s',
#         #                 [next_stage, next_substage, latest_state["id"]]
#         #             )
#         #         conn.commit()

#         # 转换为 JSON 字符串（MCP tool 需要返回字符串，不再包含 next_stage 和 next_substage）
#         if isinstance(results, dict):
#             result_dict = results
#         else:
#             # 如果 results 不是字典，创建新字典并确保包含 turn_uuid
#             result_dict = {"response": results}
#             if "turn_uuid" not in result_dict:
#                 result_dict["turn_uuid"] = turn_uuid
#         # 确保 result_dict 包含 turn_uuid（防止遗漏）
#         if "turn_uuid" not in result_dict:
#             result_dict["turn_uuid"] = turn_uuid
#         return json.dumps(result_dict, ensure_ascii=False)
#     else:
#         # 既没有 purpose 也没有 user_input，返回错误
#         return json.dumps({
#             "error": "Either 'purpose' or 'user_input' must be provided"
#         }, ensure_ascii=False)


@mcp.tool(description="Store response to chat_messages by turn_uuid. Retrieves response from Redis cache and stores both user and agent messages, plus session_state.")
def store_response_by_uuid(turn_uuid: str, table: str = "chat_messages") -> str:
    """根据 turn_uuid 统一入库：将 user 消息、agent 消息和 session_state 一并存储。

    参数:
        turn_uuid: 回合 UUID（从 main_response_procedure 返回结果中获取）
        table: 表名，默认 "chat_messages"

    返回:
        JSON 字符串，例如 {"success": true, "user_message_id": 1001, "agent_message_id": 1002, "session_state_id": 1003}
        如果 turn_uuid 对应的数据不存在，返回错误信息

    流程:
        1. 从 pending_turns 读取用户消息数据
        2. 从 Redis 读取 response:{turn_uuid} 对应的数据
        3. 生成 agent_message_uuid
        4. 统一入库：chat_messages（user + agent）、session_states
        5. 标记 pending_turn 为 done
    """
    # 从 pending_turns 读取用户消息数据
    pending_data_json = db.get_pending_turn(turn_uuid)
    if not pending_data_json or pending_data_json == "{}":
        return json.dumps({
            "error": "Pending turn not found for the given turn_uuid",
            "turn_uuid": turn_uuid
        }, ensure_ascii=False)

    try:
        pending_data = json.loads(pending_data_json)
    except json.JSONDecodeError:
        return json.dumps({
            "error": "Invalid pending turn data format",
            "turn_uuid": turn_uuid
        }, ensure_ascii=False)

    # 从 Redis 缓存中查找对应的 response
    cache_key = f"response:{turn_uuid}"
    cached_data = redis_get(cache_key)

    if not cached_data:
        return json.dumps({
            "error": "Response not found for the given turn_uuid",
            "turn_uuid": turn_uuid
        }, ensure_ascii=False)

    try:
        response_data = json.loads(cached_data)
        if not isinstance(response_data, dict):
            return json.dumps({
                "error": "Invalid response data format",
                "turn_uuid": turn_uuid
            }, ensure_ascii=False)

        response_content = response_data.get("response", "")
        if not response_content:
            return json.dumps({
                "error": "Response content is empty",
                "turn_uuid": turn_uuid
            }, ensure_ascii=False)

        # 提取 pending_turns 中的数据
        user_message_uuid = pending_data.get("user_message_uuid")
        conversation_id = pending_data.get("conversation_id")
        session_id = pending_data.get("session_id")
        user_id = pending_data.get("user_id")
        dt_user = pending_data.get("dt_user")
        user_content = pending_data.get("user_content")
        routed_current_state = pending_data.get("routed_current_state")
        routed_current_sub_stage = pending_data.get("routed_current_sub_stage")
        stage_payload_draft = pending_data.get("stage_payload_draft")
        next_stage = pending_data.get("next_stage")
        next_substage = pending_data.get("next_substage")

        # 从products_draft中提取商品信息
        product_sales = []
        product_promote = []
        products_draft = pending_data.get("products_draft")
        if products_draft:
            try:
                products_data = json.loads(products_draft) if isinstance(products_draft, str) else products_draft
                if isinstance(products_data, dict):
                    # 确保返回有效的列表
                    product_id_sales = products_data.get("product_id_sales")
                    product_id_promoted = products_data.get("product_id_promoted")

                    if product_id_sales is not None:
                        if isinstance(product_id_sales, list):
                            product_sales = product_id_sales
                        elif isinstance(product_id_sales, str) and product_id_sales.strip():
                            product_sales = [product_id_sales]
                        else:
                            product_sales = []

                    if product_id_promoted is not None:
                        if isinstance(product_id_promoted, list):
                            product_promote = product_id_promoted
                        elif isinstance(product_id_promoted, str) and product_id_promoted.strip():
                            product_promote = [product_id_promoted]
                        else:
                            product_promote = []

            except (json.JSONDecodeError, TypeError):
                product_sales = []
                product_promote = []

        # 如果未获取到商品信息，尝试沿用同一 conversation/session/user 的最近一条商品值
        if not product_sales and not product_promote:
            def _normalize_products(raw):
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        return []
                return []

            try:
                latest_products_raw = db.get_latest_message_products(conversation_id, session_id, user_id)
                latest_products = json.loads(latest_products_raw) if isinstance(latest_products_raw, str) else latest_products_raw
                if isinstance(latest_products, dict):
                    product_sales = _normalize_products(latest_products.get("product_sales"))
                    product_promote = _normalize_products(latest_products.get("product_promote"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[store_response_by_uuid] 获取最近商品信息失败 turn_uuid=%s: %s", turn_uuid, exc)
                product_sales = product_sales or []
                product_promote = product_promote or []

        if not all([user_message_uuid, conversation_id, session_id, user_id, dt_user, user_content]):
            return json.dumps({
                "error": "Incomplete pending turn data",
                "turn_uuid": turn_uuid
            }, ensure_ascii=False)

        # 生成 agent_message_uuid
        agent_message_uuid = str(uuid.uuid4())

        # 生成 dt_response（如果没有则使用当前时间）
        dt_response = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 预先获取历史上下文，供 info 与成交意愿评分共用
        history_context: List[Dict[str, Any]] = []
        history_context_raw = db.list_chat_messages(conversation_id, session_id, user_id, limit=30)
        if history_context_raw:
            try:
                history_context = json.loads(history_context_raw) if isinstance(history_context_raw, str) else history_context_raw
                if not isinstance(history_context, list):
                    history_context = []
            except (json.JSONDecodeError, TypeError):
                history_context = []

        # 0. 异步生成 info（在存储消息的同时进行）
        info_future: Optional[Future] = None
        bot_id = pending_data.get("bot_id")
        app_id = pending_data.get("app_id")
        
        # 检查 bot_id 和 app_id 是否存在
        if not bot_id or not app_id:
            logger.warning(
                "[store_response_by_uuid] pending_data 中缺少 bot_id 或 app_id，跳过 info 生成。"
                "bot_id=%s, app_id=%s, pending_data_keys=%s",
                bot_id,
                app_id,
                list(pending_data.keys()) if isinstance(pending_data, dict) else "N/A"
            )
        
        if routed_current_state and bot_id and app_id:
            try:
                # 提前获取 bot 配置（用于异步任务）
                bot_config = get_bot_config(conversation_id, bot_id, app_id)
                structured_content = getattr(bot_config, "structuredContent", {}) or {}
                if not isinstance(structured_content, dict):
                    structured_content = {}
                
                route_state_prompt_map = structured_content.get('routeStateStrategies')
                if not isinstance(route_state_prompt_map, dict):
                    route_state_prompt_map = {}
                
                character_prompt = structured_content.get('character', '')
                bot_llm_config = structured_content.get('botLLMConfig')
                if not bot_llm_config or not isinstance(bot_llm_config, dict):
                    bot_llm_config = DEFAULT_BOT_LLM_CONFIG.copy()
                
                # history_context 转换给 info：role agent -> assistant
                history_context_for_info: List[Dict[str, Any]] = []
                for item in history_context:
                    if isinstance(item, dict):
                        copied = dict(item)
                        if copied.get("role") == "agent":
                            copied["role"] = "assistant"
                        history_context_for_info.append(copied)
                
                # 获取 session_state（用于异步任务）
                session_state_json = db.get_latest_session_state_payload(conversation_id, session_id, user_id)
                session_state_for_info: Dict[str, Any] = {}
                if session_state_json:
                    try:
                        parsed_state = json.loads(session_state_json)
                        if isinstance(parsed_state, dict):
                            session_state_for_info = parsed_state
                    except (json.JSONDecodeError, TypeError):
                        session_state_for_info = {}
                
                # 提交异步任务生成 info
                info_future = _INFO_EXECUTOR.submit(
                    _generate_info_content,
                    character_prompt=character_prompt,
                    route_state_prompt_map=route_state_prompt_map,
                    routed_current_state=routed_current_state,
                    routed_current_sub_stage=routed_current_sub_stage,
                    input_text=response_content,
                    history_context=history_context_for_info,
                    session_state=session_state_for_info,
                    bot_llm_config=bot_llm_config,
                    input_label="Assistant 本轮回复内容",
                )
                logger.info(
                    "[store_response_by_uuid] 已提交异步 info 生成任务，"
                    "turn_uuid=%s, conversation_id=%s, state=%s, sub_stage=%s",
                    turn_uuid,
                    conversation_id,
                    routed_current_state,
                    routed_current_sub_stage
                )
            except Exception as exc:
                logger.warning("[store_response_by_uuid] 准备异步 info 生成失败: %s", exc)
                info_future = None
        
        # 1. 存储用户消息到 chat_messages
        user_msg_result = db.store_chat_message(
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            content=user_content,
            role="user",
            table=table,
            dt=dt_user,
            uuid_id=user_message_uuid,
            turn_uuid=turn_uuid
        )
        user_msg_data = json.loads(user_msg_result)
        user_message_id = user_msg_data.get("id")

        # 1.1 异步计算成交意愿分（基于最新用户消息）

        def _calculate_and_store_intent_score(history_messages: List[Dict[str, Any]]) -> None:
            try:
                if not isinstance(history_messages, list):
                    logger.warning("[store_response_by_uuid] history_messages 非列表，跳过成交意愿计算。turn_uuid=%s", turn_uuid)
                    return
                score_value = generate_conversation_scores(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    user_id=user_id,
                    messages=history_messages
                )
                db.store_total_score(
                    message_id=user_message_id,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    user_id=user_id,
                    total_score=score_value,
                    uuid_id=user_message_uuid,
                    dt=dt_user
                )
                logger.info(
                    "[store_response_by_uuid] 成交意愿分计算并写入成功 turn_uuid=%s, score=%.2f, message_id=%s",
                    turn_uuid,
                    score_value,
                    user_message_id
                )
            except Exception as exc:
                logger.warning("[store_response_by_uuid] 成交意愿分计算失败 turn_uuid=%s: %s", turn_uuid, exc, exc_info=True)

        try:
            history_for_score: List[Dict[str, Any]] = [dict(item) for item in history_context if isinstance(item, dict)]
            history_for_score.append({
                "id": user_message_id,
                "uuid_id": user_message_uuid,
                "datetime": dt_user,
                "role": "user",
                "content": user_content
            })
            _SCORE_EXECUTOR.submit(_calculate_and_store_intent_score, history_for_score)
        except Exception as exc:
            logger.warning("[store_response_by_uuid] 提交成交意愿分异步任务失败 turn_uuid=%s: %s", turn_uuid, exc, exc_info=True)

        # 2. 存储 agent 消息到 chat_messages
        agent_msg_result = db.store_chat_message(
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            content=response_content,
            role="agent",
            table=table,
            dt=dt_response,
            uuid_id=agent_message_uuid,
            turn_uuid=turn_uuid
        )
        agent_msg_data = json.loads(agent_msg_result)
        agent_message_id = agent_msg_data.get("id")

        # 2.1 更新 agent 消息的商品字段
        if product_sales is not None or product_promote is not None:
            db.update_chat_message_products(
                message_id=agent_message_id,
                product_sales=product_sales,
                product_promote=product_promote,
                table=table
            )

        # 3. 等待异步 info 生成完成（如果已提交）
        info_content: Dict[str, Any] = {}
        if info_future:
            try:
                info_content = info_future.result()  # 等待异步任务完成
                logger.info(
                    "[store_response_by_uuid] 异步 info 生成成功，"
                    "turn_uuid=%sinfo_content=%s",
                    turn_uuid,
                    json.dumps(info_content, ensure_ascii=False) if info_content else "{}"
                )
            except Exception as exc:
                logger.warning(
                    "[store_response_by_uuid] 异步 info 生成失败，turn_uuid=%s: %s",
                    turn_uuid,
                    exc,
                    exc_info=True
                )
                info_content = {}
        
        # 4. 存储 session_state（仅存储 info，不包含 response）
        session_state_id = None
        stage_payload: Dict[str, Any] = {}
        
        if routed_current_state and info_content:
            # 无论 stage_payload_draft 中有什么，我们只关心构建包含 info 的 payload
            # 忽略 draft 中的 response 内容
            substage_col = normalize_substage_name(routed_current_state, routed_current_sub_stage)
            stage_payload = {substage_col: {"info": info_content}}
            
            logger.info(
                "[store_response_by_uuid] 构建仅包含 info 的 stage_payload，"
                "turn_uuid=%s, substage=%s",
                turn_uuid,
                substage_col
            )
        else:
            # 如果没有 info，尝试从 draft 中提取 info（如果有），否则为空
            if stage_payload_draft:
                try:
                    draft_payload = json.loads(stage_payload_draft) if isinstance(stage_payload_draft, str) else stage_payload_draft
                    if isinstance(draft_payload, dict):
                        # 遍历 draft，只保留 info 字段
                        for key, value in draft_payload.items():
                            if isinstance(value, dict) and "info" in value:
                                stage_payload[key] = {"info": value["info"]}
                except json.JSONDecodeError:
                    pass
            
            if not stage_payload and info_content:
                 logger.warning(
                    "[store_response_by_uuid] 生成了 info 但无法存储（缺少 routed_current_state），"
                    "turn_uuid=%s, info_content=%s",
                    turn_uuid,
                    json.dumps(info_content, ensure_ascii=False)
                )

        session_state_result = db.store_session_state(
            message_id=user_message_id,  # 使用 user_message_id（兼容旧逻辑）
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            current_state=routed_current_state,
            current_sub_stage=routed_current_sub_stage if routed_current_sub_stage else f"{routed_current_state}_01",
            stage_payload_json=json.dumps(stage_payload, ensure_ascii=False),
            dt=dt_user,
            table="session_states",
            uuid_id=user_message_uuid,
            turn_uuid=turn_uuid,
            next_stage=next_stage,
            next_substage=next_substage
        )
        session_state_data = json.loads(session_state_result)
        session_state_id = session_state_data.get("id")

        # 5. 标记 pending_turn 为 done
        db.mark_pending_turn_done(turn_uuid)

        # 6. 清理 Redis 缓存
        try:
            redis_set(cache_key, "", expired=1)  # 立即过期
            redis_set(f"pending:{turn_uuid}", "", expired=1)  # 立即过期
        except Exception:
            pass

        return json.dumps({
            "success": True,
            "message": "存储成功",
            "turn_uuid": turn_uuid,
            "user_message_id": user_message_id,
            "user_message_uuid": user_message_uuid,
            "agent_message_id": agent_message_id,
            "agent_message_uuid": agent_message_uuid,
            "session_state_id": session_state_id
        }, ensure_ascii=False)

    except json.JSONDecodeError:
        return json.dumps({
            "error": "Failed to parse response data from cache",
            "turn_uuid": turn_uuid
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to store response: {str(e)}",
            "turn_uuid": turn_uuid
        }, ensure_ascii=False)


@mcp.tool(description="Service compliance response: output compliance service content based on session context.")
def service_compliance_response(session_id: str, user_id: str, bot_id: str, app_id: str, turn_uuid: str, equity_key: str = None) -> str:
    """履约服务响应：根据用户输入和会话上下文输出履约服务内容。

    参数:
        session_id: 会话标识
        user_id: 用户标识
        bot_id: 机器人标识
        app_id: 应用标识
        turn_uuid: 回合 UUID（由调用方生成，用于定位上下文）

    返回:
        JSON 字符串，包含履约服务内容和 turn_uuid
    """
    if not turn_uuid:
        return json.dumps({"error": "turn_uuid is required"}, ensure_ascii=False)

    # 获取 conversation_id
    conversation_json = db.get_conversation_with_cache(user_id, session_id)
    conversation_id = ""
    try:
        conv_data = json.loads(conversation_json)
        if isinstance(conv_data, dict) and "id" in conv_data:
            conversation_id = str(conv_data["id"])
    except (json.JSONDecodeError, TypeError, KeyError):
        conversation_id = ""

    # 获取 bot 配置
    bot_config = get_bot_config(conversation_id, bot_id, app_id)
    structuredContent = bot_config.structuredContent

    # 入口处：生成时间戳并写入 pending_turns
    dt_user = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    user_message_uuid = str(uuid.uuid4())

    # 写入 pending_turns 基本信息
    db.store_pending_turn(
        turn_uuid=turn_uuid,
        user_message_uuid=user_message_uuid,
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        dt_user=dt_user,
        user_content="",
        bot_id=bot_id,
        app_id=app_id
    )

    # 同时缓存到 Redis（冗余）
    try:
        pending_cache_key = f"pending:{turn_uuid}"
        pending_cache_data = {
            "turn_uuid": turn_uuid,
            "user_message_uuid": user_message_uuid,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "dt_user": dt_user,
            "user_content": "",
            "bot_id": bot_id,
            "app_id": app_id
        }
        redis_set(pending_cache_key, json.dumps(pending_cache_data, ensure_ascii=False), expired=86400)  # 24小时
    except Exception:
        pass

    # 获取 compliance 阶段的 prompt
    route_state_prompt_map = structuredContent.get('routeStateStrategies') if isinstance(structuredContent, dict) else None
    if not isinstance(route_state_prompt_map, dict):
        route_state_prompt_map = {}

    # 获取 character prompt
    character_prompt = structuredContent.get('character', '')

    # 设置 compliance 阶段信息
    routed_current_state = "compliance"
    routed_current_sub_stage = "compliance_01"  # 将在选择模块后根据索引覆盖

    # 获取 compliance 阶段的模块，并基于 equity_key 直接匹配 product 字段确定模块索引
    compliance_modules = route_state_prompt_map.get('compliance', []) if isinstance(route_state_prompt_map, dict) else []
    selected_idx = 0
    matching_idx = find_compliance_module_index_by_product(route_state_prompt_map, equity_key)
    if matching_idx is not None:
        selected_idx = matching_idx
    compliance_module = compliance_modules[selected_idx] if compliance_modules else {}
    # 根据选择的索引确定具体 substage（例如 0 -> compliance_01）
    routed_current_sub_stage = f"compliance_{selected_idx + 1:02d}"
    # if not compliance_modules:
    #     # 如果没有配置，使用默认的 compliance prompt
    #     compliance_module = {
    #         "purpose": "根据用户需求和会话状态，输出履约服务内容，包括服务说明、使用方式、注意事项等。",
    #         "name": ["response"],
    #         "expect": ["履约服务内容的详细说明"],
    #         "operation": [
    #             "1. 分析用户需求和当前会话状态",
    #             "2. 提取相关的服务信息",
    #             "3. 生成详细的履约服务内容说明",
    #             "4. 确保内容清晰、完整、易于理解"
    #         ]
    #     }
    # else:
    #     compliance_module = compliance_modules[0]
    #     # 如果有多个模块，可以根据需要选择，这里使用第一个
    #     # 如果模块有编号，可以提取 substage 编号
    #     if len(compliance_modules) > 1:
    #         # 可以根据业务逻辑选择模块，这里默认使用第一个
    #         pass

    # 组装 prompt
    compliance_prompt = join_prompts(compliance_module)
    full_prompt = "\n\n".join([character_prompt, compliance_prompt])

    # botLLMConfig 兜底
    botLLMConfig = structuredContent.get('botLLMConfig')
    if not botLLMConfig or not isinstance(botLLMConfig, dict):
        botLLMConfig = DEFAULT_BOT_LLM_CONFIG.copy()

    # 以 turn_uuid 为节点，获取该时间点之前的历史上下文
    history_context = db.list_chat_messages_before_turn(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        dt_upper=dt_user,
        limit=50
    )
    session_state_json = db.get_latest_session_state_before_turn(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        dt_upper=dt_user
    )
    session_state = {}
    if session_state_json:
        try:
            parsed = json.loads(session_state_json)
            if isinstance(parsed, dict):
                session_state = parsed
        except (json.JSONDecodeError, TypeError):
            session_state = {}

    # 将本轮 pending 的用户输入拼到历史末尾（仅用于上下文），并做去重
    try:
        pending_json = db.get_pending_turn(turn_uuid)
        if pending_json:
            pending_obj = json.loads(pending_json)
            if isinstance(pending_obj, dict):
                pending_content = (pending_obj.get("user_content") or "").strip()
                pending_dt = pending_obj.get("dt_user")
                if pending_content:
                    hist = json.loads(history_context) if isinstance(history_context, str) else history_context
                    if isinstance(hist, list):
                        append_needed = True
                        if len(hist) > 0 and isinstance(hist[-1], dict):
                            last = hist[-1]
                            last_role = last.get("role")
                            last_content = (last.get("content") or "").strip()
                            if last_role == "user" and last_content == pending_content:
                                append_needed = False
                        if append_needed:
                            hist.append({
                                "id": None,
                                "datetime": pending_dt or "",
                                "role": "user",
                                "content": pending_content
                            })
                        history_context = json.dumps(hist, ensure_ascii=False)
    except Exception:
        # 不影响主流程
        pass

    # 调用 LLM 生成履约服务内容
    results = llm_generic(
        full_prompt=full_prompt,
        user_input="",
        history_context=history_context,
        session_state=session_state,
        botLLMConfig=botLLMConfig
    )

    # 解析 results（可能是 JSON 字符串）
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except json.JSONDecodeError:
            # 如果不是 JSON，保持原样
            pass

    # 后处理：从 info 中提取对话内容并合并到 response
    if isinstance(results, dict):
        results = extract_conversational_content_from_info(results)

    # 为 results 添加 turn_uuid 并缓存 response
    add_conversation_id_uuid_and_cache(results, conversation_id, session_id, user_id, turn_uuid)

    # 如果 results 包含 info 字段，更新 pending_turns 的状态信息
    if isinstance(results, dict) and "info" in results:
        # 构建 substage 名称（如 "compliance_01"）
        substage_col = routed_current_sub_stage

        # 构建 stage_payload 格式：{"compliance_01": {"info": {...}}}
        stage_payload = {substage_col: {"info": results["info"]}}

        # 更新 pending_turns 的状态信息
        db.update_pending_turn_state(
            turn_uuid=turn_uuid,
            routed_current_state=routed_current_state,
            routed_current_sub_stage=routed_current_sub_stage,
            stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False)
        )

    # 设置 next_stage 和 next_substage：使用 equity_key 对应的模块索引确定的 substage
    next_stage = routed_current_state
    next_substage = routed_current_sub_stage

    # 更新 pending_turns 的 next_stage 和 next_substage
    db.update_pending_turn_state(
        turn_uuid=turn_uuid,
        next_stage=next_stage,
        next_substage=next_substage
    )

    # 记录最后设定的 next_stage 和 next_substage（service_compliance_response）
    logger.info(f"[Final Next Stage - Compliance] conversation_id={conversation_id}, session_id={session_id}, user_id={user_id}")
    logger.info(f"[Final Next Stage - Compliance] next_stage={next_stage}, next_substage={next_substage}")

    # 转换为 JSON 字符串返回（包含 turn_uuid）
    result_dict = results if isinstance(results, dict) else {"response": results}
    result_dict["turn_uuid"] = turn_uuid

    # 在 info 中手动添加 equity_used 字段
    if "info" not in result_dict:
        result_dict["info"] = {}
    if not isinstance(result_dict["info"], dict):
        result_dict["info"] = {}
    result_dict["info"]["equity_used"] = equity_key if equity_key else ""

    return json.dumps(result_dict, ensure_ascii=False)


@mcp.tool(description="Calculate follow-up strategy: determine optimal follow-up timestamp (ISO format) based on conversation history and timing. Returns timestamp string directly.")
def calculate_follow_up_timestamp(
    app_id: str,
    bot_id: str,
    session_id: str,
    user_id: str,
    user_last_timestamp: int,
    agent_last_timestamp: int,
    current_timestamp: int,
) -> str:
    """追单策略：根据会话历史和时间信息计算合适的追问时机。

    参数:
        app_id: 应用标识
        bot_id: 机器人标识
        session_id: 会话标识
        user_id: 用户标识
        user_last_timestamp: 用户最后回复时间戳（Unix 秒级，int）
        agent_last_timestamp: Agent 最后回复时间戳（Unix 秒级，int）
        current_timestamp: 当前时间戳（Unix 秒级，int）

    返回:
        时间戳字符串（Unix 秒级，字符串形式），例如："1732427400"
        确保返回的时间戳 >= current_timestamp

    流程:
        1. 根据 session_id 和 user_id 获取最新 conversation_id 和历史对话
        2. 根据 app_id 和 bot_id 获取 LLM 配置（优先 Redis）
        3. 调用 LLM 分析历史对话和时间信息，给出合适的追问时机
        4. 确保返回的时间戳 >= current_timestamp
        5. 兜底逻辑：如果 LLM 返回格式错误或时间戳无效，使用默认策略（当前时间 + 15 分钟）
    """
    def _parse_timestamp(ts_value: int) -> datetime:
        """解析秒级 Unix 时间戳（仅限整数）。"""
        return datetime.fromtimestamp(ts_value, tz=timezone.utc)

    try:
        # 解析时间戳
        user_last_dt = _parse_timestamp(user_last_timestamp)
        agent_last_dt = _parse_timestamp(agent_last_timestamp)
        current_dt = _parse_timestamp(current_timestamp)

        # 获取 conversation_id
        conversation_json = db.get_conversation_with_cache(user_id, session_id)
        conversation_id = ""
        try:
            conv_data = json.loads(conversation_json)
            if isinstance(conv_data, dict) and "id" in conv_data:
                conversation_id = str(conv_data["id"])
        except (json.JSONDecodeError, TypeError, KeyError):
            conversation_id = ""

        if not conversation_id:
            return json.dumps({
                "error": "Conversation not found",
                "follow_up_timestamp": str(int(current_dt.timestamp())),
            }, ensure_ascii=False)

        # 获取历史对话（最近 20 条），并标准化角色
        history_context_str = db.list_chat_messages(conversation_id, session_id, user_id, limit=20)
        raw_history: List[Dict[str, Any]] = []
        if history_context_str:
            try:
                parsed_history = json.loads(history_context_str)
                if isinstance(parsed_history, list):
                    raw_history = parsed_history
            except (json.JSONDecodeError, TypeError):
                raw_history = []

        history_context: List[Dict[str, Any]] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role == "agent":
                role = "assistant"
            elif role not in {"assistant", "system", "user", "tool", "function"}:
                role = "user"
            history_context.append({
                "role": role,
                "content": item.get("content", ""),
            })

        # 获取 bot LLM 配置（优先 Redis）
        bot_config = get_bot_config(conversation_id, bot_id, app_id)
        structured_content = bot_config.structuredContent
        bot_llm_config = structured_content.get('botLLMConfig')
        if not bot_llm_config or not isinstance(bot_llm_config, dict):
            bot_llm_config = DEFAULT_BOT_LLM_CONFIG.copy()

        # 计算时间差（秒）
        user_silence_seconds = (current_dt - user_last_dt).total_seconds()
        agent_silence_seconds = (current_dt - agent_last_dt).total_seconds()

        # 构建 prompt（历史对话通过 history_context 传入，不在 prompt 中拼接）
        follow_up_prompt = f"""你是一个智能追单策略助手，需要根据会话历史和时间信息，判断合适的追问时机。

【任务】
请结合历史对话上下文和时间信息，判断合适的追问时机。考虑因素：
1. Agent 最后回复内容：如果 Agent 提出了问题或等待用户回复，可适当提前追问
2. 会话阶段：根据对话内容判断会话处于哪个阶段（初次接触、深入沟通、成交阶段等）
3. 用户参与度：根据历史对话判断用户参与度，参与度高可适当提前追问

【输出要求】
请直接输出期望执行追单的时间戳（ISO 格式，必须 >= 当前时间），不要输出其他内容。
时间戳格式示例：{current_dt.isoformat()}

请根据实际情况直接输出时间戳。"""

        # 构建用户输入（包含时间信息，历史对话通过 history_context 传入）
        user_input_with_timing = f"""请根据以下时间信息和历史对话上下文，判断合适的追问时机：

- 用户最后回复时间: {user_last_dt.isoformat()}
- Agent 最后回复时间: {agent_last_dt.isoformat()}
- 当前时间: {current_dt.isoformat()}
- 用户已沉默时长: {user_silence_seconds:.0f} 秒（约 {user_silence_seconds / 60:.1f} 分钟）
- Agent 已沉默时长: {agent_silence_seconds:.0f} 秒（约 {agent_silence_seconds / 60:.1f} 分钟）

请结合历史对话内容，直接输出时间戳（ISO 格式）。"""

        # 调用 LLM（历史对话作为上下文传入）
        llm_result = llm_generic(
            full_prompt=follow_up_prompt,
            user_input=user_input_with_timing,
            history_context=history_context,
            session_state={},
            botLLMConfig=bot_llm_config,
            prompt_without_character=follow_up_prompt,
        )

        # 解析 LLM 结果（直接提取时间戳字符串）
        follow_up_ts_str = ""
        if isinstance(llm_result, str):
            # 清理可能的 JSON 包裹或多余文本
            cleaned = _cleanup_llm_json_str(llm_result).strip()
            # 尝试提取时间戳（可能是 ISO 格式字符串）
            # 移除可能的引号
            cleaned = cleaned.strip('"').strip("'").strip()
            # 尝试从 JSON 中提取（兼容 LLM 可能仍返回 JSON 的情况）
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    follow_up_ts_str = parsed.get("follow_up_timestamp", "") or parsed.get("timestamp", "")
                elif isinstance(parsed, str):
                    follow_up_ts_str = parsed
            except json.JSONDecodeError:
                # 不是 JSON，直接使用清理后的字符串
                follow_up_ts_str = cleaned
        elif isinstance(llm_result, dict):
            follow_up_ts_str = llm_result.get("follow_up_timestamp", "") or llm_result.get("timestamp", "")

        # 解析并验证时间戳
        follow_up_dt = None
        try:
            if follow_up_ts_str:
                follow_up_dt = _parse_timestamp(follow_up_ts_str)
        except (ValueError, TypeError):
            follow_up_dt = None

        # 兜底逻辑：如果解析失败或为空，使用默认策略
        if follow_up_dt is None:
            # 使用默认策略：当前时间 + 15 分钟
            follow_up_dt = current_dt + timedelta(minutes=15)
            logger.warning(
                "[calculate_follow_up_timestamp] LLM 返回时间戳格式错误或为空，使用默认策略（当前时间 + 15 分钟）。原始返回: %s",
                llm_result[:200] if isinstance(llm_result, str) else str(llm_result)[:200]
            )

        # 确保时间戳 >= current_timestamp
        if follow_up_dt < current_dt:
            # 如果小于当前时间，使用当前时间 + 5 分钟作为最小间隔
            follow_up_dt = current_dt + timedelta(minutes=5)
            logger.warning(
                "[calculate_follow_up_timestamp] LLM 返回时间戳早于当前时间，调整至当前时间后 5 分钟"
            )

        # 返回秒级 Unix 时间戳
        return str(int(follow_up_dt.timestamp()))

    except Exception as e:
        logger.error(f"[calculate_follow_up_timestamp] 执行失败: {e}", exc_info=True)
        # 发生错误时，返回当前时间 + 15 分钟作为默认值（确保输出为时间戳格式）
        try:
            current_dt = _parse_timestamp(current_timestamp)
            default_dt = current_dt + timedelta(minutes=15)
            return str(int(default_dt.timestamp()))
        except Exception:
            try:
                fallback_dt = datetime.now(timezone.utc) + timedelta(minutes=15)
                return str(int(fallback_dt.timestamp()))
            except Exception:
                return str(int(datetime.now(timezone.utc).timestamp()))


@mcp.tool(description="Generate follow-up reply: produce proactive follow-up response (ISO prompt) and return response text with turn_uuid.")
def generate_follow_up_response(
    app_id: str,
    bot_id: str,
    session_id: str,
    user_id: str,
    tenant_outer_id: Optional[str] = None,
) -> str:
    """调用 LLM 生成追单回复，返回 response 与 turn_uuid（JSON 字符串）。

    - 自动获取 conversation_id、会话上下文与 bot 配置
    - 若提供 tenant_outer_id 且配置了 datasetId，则检索营销话术
    - 结果仅返回 {"response": "...", "turn_uuid": "..."}，供外部调用 store_response_by_uuid
    """
    try:
        conversation_json = db.get_conversation_with_cache(user_id, session_id)
        conversation_id = ""
        if conversation_json:
            try:
                conv_obj = json.loads(conversation_json)
                if isinstance(conv_obj, dict) and "id" in conv_obj:
                    conversation_id = str(conv_obj["id"])
            except (json.JSONDecodeError, TypeError, KeyError):
                conversation_id = ""
        if not conversation_id:
            return json.dumps({
                "error": "conversation_id not found",
            }, ensure_ascii=False)

        session_state_json = db.get_latest_session_state_payload(conversation_id, session_id, user_id)
        session_state: Dict[str, Any] = {}
        if session_state_json:
            try:
                parsed_state = json.loads(session_state_json)
                if isinstance(parsed_state, dict):
                    session_state = parsed_state
            except (json.JSONDecodeError, TypeError):
                session_state = {}

        routed_current_state = (
            session_state.get("next_stage")
            or session_state.get("current_state")
            or "decision_making"
        )
        routed_current_sub_stage = (
            session_state.get("next_sub_stage")
            or session_state.get("current_sub_stage")
            or f"{routed_current_state}_01"
        )

        history_context_str = db.list_chat_messages(conversation_id, session_id, user_id, limit=30)
        raw_history: List[Dict[str, Any]] = []
        if history_context_str:
            try:
                parsed_history = json.loads(history_context_str)
                if isinstance(parsed_history, list):
                    raw_history = parsed_history
            except (json.JSONDecodeError, TypeError):
                raw_history = []

        history_context: List[Dict[str, Any]] = []
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role == "agent":
                role = "assistant"
            elif role == "assistant":
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                role = "user"
            history_context.append({
                "role": role,
                "content": item.get("content", ""),
            })

        latest_user_message = ""
        for item in reversed(history_context):
            if isinstance(item, dict) and item.get("role") == "user":
                latest_user_message = str(item.get("content", "")).strip()
                if latest_user_message:
                    break

        bot_config = get_bot_config(conversation_id, bot_id, app_id)
        structured_content = getattr(bot_config, "structuredContent", {}) or {}
        if not isinstance(structured_content, dict):
            structured_content = {}
        character_prompt = structured_content.get("character", "")
        route_state_prompt_map = structured_content.get("routeStateStrategies")
        if not isinstance(route_state_prompt_map, dict):
            route_state_prompt_map = {}
        bot_llm_config = structured_content.get("botLLMConfig")
        if not isinstance(bot_llm_config, dict):
            bot_llm_config = DEFAULT_BOT_LLM_CONFIG.copy()

        marketing_snippet = ""
        dataset_id = str(structured_content.get("datasetId", "") or "").strip()
        if tenant_outer_id and dataset_id:
            query_text = latest_user_message or "追单营销话术"
            try:
                dataset_raw = retrieve_dataset(
                    tenant_outer_id=str(tenant_outer_id),
                    app_id=app_id,
                    dataset_id=dataset_id,
                    query=query_text,
                )
                marketing_snippet = extract_dataset_snippets(dataset_raw)
                if marketing_snippet:
                    logger.info("[generate_follow_up_response] 营销话术检索成功，长度=%d", len(marketing_snippet))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[generate_follow_up_response] 营销话术检索失败: %s", exc)

        dt_user = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        turn_uuid = str(uuid.uuid4())
        user_message_uuid = str(uuid.uuid4())

        db.store_pending_turn(
            turn_uuid=turn_uuid,
            user_message_uuid=user_message_uuid,
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            dt_user=dt_user,
            user_content="",
            bot_id=bot_id,
            app_id=app_id,
        )
        try:
            pending_cache_key = f"pending:{turn_uuid}"
            pending_cache_data = {
                "turn_uuid": turn_uuid,
                "user_message_uuid": user_message_uuid,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "user_id": user_id,
                "dt_user": dt_user,
                "user_content": "",
                "bot_id": bot_id,
                "app_id": app_id,
            }
            redis_set(pending_cache_key, json.dumps(pending_cache_data, ensure_ascii=False), expired=86400)
        except Exception:
            pass

        follow_up_brief = "\n".join([
            "[追单任务]",
            "用户阶段暂未结束，请以温暖、真诚且不过度打扰的口吻进行追单，鼓励用户回复。",
            "需要：",
            "1. 简短回顾用户关注点或痛点；",
            "2. 给出新的价值点或限时权益提示；",
            "3. 引导用户给出下一步（例如确认、提问或预约时间）。",
        ])

        system_parts = [character_prompt, follow_up_brief]
        if marketing_snippet:
            system_parts.append("【知识库】\n" + marketing_snippet)
        stage_module = _select_stage_module(route_state_prompt_map, routed_current_state, routed_current_sub_stage)
        if isinstance(stage_module, dict):
            mission = stage_module.get("purpose")
            if mission:
                system_parts.append("【阶段目标】\n" + str(mission))

        system_prompt = "\n\n".join([part for part in system_parts if part])

        user_input_payload = "\n".join([
            "【最新用户消息】",
            latest_user_message or "用户未留下额外信息，请从历史上下文提炼要点。",
            "",
            "【回复要求】",
            "- 以关怀开场，体现你仍然记得用户的诉求；",
            "- 如果适用，可提及营销话术要点；",
            "- 结尾提出具体邀请或可执行的下一步。",
        ])

        llm_result = llm_generic(
            full_prompt=system_prompt,
            user_input=user_input_payload,
            history_context=history_context,
            session_state=session_state,
            botLLMConfig=bot_llm_config,
            prompt_without_character=system_prompt,
        )

        response_text = ""
        if isinstance(llm_result, str):
            try:
                parsed_result = json.loads(_cleanup_llm_json_str(llm_result))
                if isinstance(parsed_result, dict):
                    response_text = str(parsed_result.get("response") or parsed_result.get("text") or "")
                else:
                    response_text = str(parsed_result)
            except json.JSONDecodeError:
                response_text = llm_result.strip()
        elif isinstance(llm_result, dict):
            response_value = llm_result.get("response") or llm_result.get("text")
            if isinstance(response_value, str):
                response_text = response_value.strip()
            else:
                response_text = str(response_value or "").strip()

        if not response_text:
            response_text = "想跟你确认一下，之前提到的服务还有任何疑问或需要帮助的地方吗？我在这里等你，随时可以继续。"

        info_payload = {"type": "follow_up", "auto": True}
        response_payload = {
            "response": response_text,
            "info": info_payload,
        }

        add_conversation_id_uuid_and_cache(
            response_payload,
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            turn_uuid=turn_uuid,
        )

        substage_col = normalize_substage_name(routed_current_state, routed_current_sub_stage)
        stage_payload = {substage_col: {"info": info_payload, "response": response_text}}
        try:
            db.update_pending_turn_state(
                turn_uuid=turn_uuid,
                routed_current_state=routed_current_state,
                routed_current_sub_stage=routed_current_sub_stage,
                stage_payload_draft=json.dumps(stage_payload, ensure_ascii=False),
            )
            db.update_pending_turn_state(
                turn_uuid=turn_uuid,
                next_stage=routed_current_state,
                next_substage=routed_current_sub_stage or substage_col,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[generate_follow_up_response] 更新 pending_turn 状态失败: %s", exc)

        return json.dumps({
            "response": response_text,
            "turn_uuid": turn_uuid,
        }, ensure_ascii=False)

    except Exception as e:  # noqa: BLE001
        logger.error("[generate_follow_up_response] 执行失败: %s", e, exc_info=True)
        return json.dumps({
            "error": str(e),
        }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="sse")



