import asyncio
import json
import time
import uuid
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, Dict, Generator, Callable, AsyncGenerator, Any, List, Tuple

# import porstgreDB_tools as db
import src.tools.porstgreDB_tools as db
from Responder import _get_llm_client_from_config, llm_generic
from bot_mcp import get_bot_config, get_products_with_cache
from config.settings import DEFAULT_BOT_LLM_CONFIG, FAST_REPLY_LLM_CONFIG
from dify_mcp import extract_dataset_snippets, retrieve_dataset
from main_response_procedure import judge_after_sales_intent, detect_handoff_intent, build_prev_next_step_texts, \
    get_all_stages_and_substages, _extract_products_list, generate_response_payload, generate_product_analysis_payload, \
    format_product_list_for_llm, find_compliance_module_index_by_product, add_conversation_id_uuid_and_cache, \
    set_all_stages_and_substages, _has_content, normalize_substage_name, route_current_stage, process_handoff_detection, \
    build_promotion_product_snippet

from src.tools.redis_tools import redis_set
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)
_QUICK_REPLY_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=8)
_HANDOFF_REDIS_TTL_SECONDS = 24 * 60 * 60
_HANDOFF_TRIGGER_THRESHOLD = 3
_HANDOFF_REPLY_TEXT = "已为您转接人工客服, 请稍后"

def get_fast_reply_client():
    """获取快速回复的 LLM 客户端和模型，使用固定配置进行初始化。"""
    logger.info("[get_fast_reply_client] 开始获取快速回复客户端")
    llm_client, llm_model = _get_llm_client_from_config(FAST_REPLY_LLM_CONFIG)
    logger.info("[get_fast_reply_client] 客户端获取成功，模型: %s", llm_model)
    return llm_client, llm_model


def _generate_quick_reply(user_input: str, bot_character: str, conversation_id: str, session_id: str, user_id: str,
                          fn: Callable[[Dict[str, object]], None] = None) -> Optional[str]:
    start_time = time.perf_counter()
    logger.info("[_generate_quick_reply] 开始生成快速回复，用户输入长度: %d", len(user_input))

    logger.info("[_generate_quick_reply] user_input: %s", user_input)
    logger.info("[_generate_quick_reply] bot_character: %s", bot_character)
    logger.info("[_generate_quick_reply] conversation_id: %s", conversation_id)
    logger.info("[_generate_quick_reply] session_id: %s", session_id)
    logger.info("[_generate_quick_reply] user_id: %s", user_id)
    try:
        logger.info("[_generate_quick_reply] 获取 LLM 客户端")
        llm_client, llm_model = get_fast_reply_client()

        logger.info("[_generate_quick_reply] 获取历史会话消息")
        history_context = db.list_chat_messages(conversation_id, session_id, user_id, limit=6)
        if isinstance(history_context, str):
            try:
                history_context = json.loads(history_context)
            except (json.JSONDecodeError, TypeError):
                history_context = []
        elif not isinstance(history_context, List):
            history_context = []

        # 确保 bot_character 是字符串类型
        if bot_character:
            if isinstance(bot_character, str):
                system_content = bot_character
            elif isinstance(bot_character, dict):
                # 如果是字典，转换为 JSON 字符串
                system_content = json.dumps(bot_character, ensure_ascii=False)
            else:
                # 其他类型转换为字符串
                system_content = str(bot_character)
        else:
            system_content = "根据用户的语言，用一句自然、轻松、不涉及业务的“收到/明白/稍等”类短回复来快速回应用户。回复***不能超过10个字***。"

        quick_messages = [
            {
                "role": "system",
                "content": system_content,
            },
        ]
        if history_context:
            for item in history_context:
                quick_messages.append({
                    "role": "assistant" if item["role"] != "user" else "user",
                    "content": item["content"],
                })
        quick_messages.append(
            {
                "role": "user",
                "content": f"根据用户的语言，用一句自然、轻松、不涉及业务的“收到/明白/稍等”类短回复来快速回应用户。回复***不能超过50个字***。我输入的内容为：{user_input}",
            }
        )
        logger.info("[_generate_quick_reply] 调用 LLM 生成快速回复（流式模式）")
        reply = llm_client.chat.completions.create(
            model=llm_model,
            messages=quick_messages,
        )

        # 累积流式响应内容
        # reply_parts = []
        # for chunk in stream:
        #     if chunk.choices and len(chunk.choices) > 0:
        #         msg = chunk.choices[0]
        #         delta = msg.delta
        #         if delta and delta.content:
        #             reply_parts.append(delta.content)
        #         if "finish_reason" in msg and msg.finish_reason == "stop":
        #             break

        # reply = "".join(reply_parts).strip()
        logger.info("[_generate_quick_reply] 快速回复生成成功，内容: %s", reply)
        logger.info("[_generate_quick_reply] 快速回复生成成功，长度: %d", len(reply))

        # 检查是否包含禁止字符
        # forbidden_chars = ['"', "'",  '“', '”', '‘', '’']
        # if any(char in reply for char in forbidden_chars):
        #     logger.warning("[_generate_quick_reply] 回复包含禁止字符，返回 None")
        #     elapsed = time.perf_counter() - start_time
        #     logger.info("[_generate_quick_reply] 快速回复生成完成，耗时: %.3f 秒", elapsed)
        #     return None

        # 检查是否包含冒号，如果包含则返回冒号后的内容
        # colon_chars = [':', '：']
        # for colon in colon_chars:
        #     if colon in reply:
        #         colon_index = reply.rfind(colon)  # 找到最后一个冒号的位置
        #         reply = reply[colon_index + 1:].strip()  # 取冒号后的内容
        #         logger.info("[_generate_quick_reply] 检测到冒号，截取后部分内容: %s", reply)
        #         break

        if fn and reply:
            # 检查前置逻辑执行时间，如果未超过2秒则等待到2秒
            elapsed_before_fn = time.perf_counter() - start_time
            min_wait_time = 1.0
            if elapsed_before_fn < min_wait_time:
                wait_time = min_wait_time - elapsed_before_fn
                logger.info("[_generate_quick_reply] 前置逻辑耗时 %.3f 秒，等待 %.3f 秒后执行回调", elapsed_before_fn,
                            wait_time)
                time.sleep(wait_time)
            else:
                logger.info("[_generate_quick_reply] 前置逻辑耗时 %.3f 秒，无需等待", elapsed_before_fn)

            logger.info("[_generate_quick_reply] 通过回调函数发送快速回复")
            fn({
                "event": "reply",
                "data": {
                    "text": reply,
                },
            })
        elapsed = time.perf_counter() - start_time
        logger.info("[_generate_quick_reply] 快速回复生成完成，耗时: %.3f 秒", elapsed)
        return reply
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start_time
        logger.warning("[_generate_quick_reply] 生成快速回复失败，耗时: %.3f 秒，错误: %s", elapsed, exc)
        return None


def _normalize_to_list(raw_value: Any) -> List[str]:
    """将字符串或列表规范为字符串列表。"""
    keys: List[str] = []
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if cleaned:
            keys.append(cleaned)
    elif isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    keys.append(cleaned)
    return keys


def _has_available_equity(equity_info: Dict[str, Any], equity_key: str) -> bool:
    """检查给定权益 key 在输入权益中是否存在且数量 > 0。"""
    if not isinstance(equity_key, str) or not equity_key.startswith("product_equity_"):
        return False
    amount = equity_info.get(equity_key) if isinstance(equity_info, dict) else None
    return isinstance(amount, (int, float)) and amount > 0


def _check_and_reset_session(
        conversation_id: str,
        session_id: str,
        user_id: str,
        current_state: str,
        current_sub_stage: str,
        session_state: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    检查会话是否需要重置（例如处于 compliance 并超过 24 小时），如需则结束旧会话并创建新会话。

    返回新的 (conversation_id, current_state, current_sub_stage, session_state)。
    """
    if current_state != "compliance":
        return conversation_id, current_state, current_sub_stage, session_state

    try:
        last_msg_json = db.get_latest_chat_message(conversation_id, session_id, user_id)
        if last_msg_json:
            last_msg = json.loads(last_msg_json)
            if isinstance(last_msg, dict):
                dt_str = last_msg.get("dt")
                if dt_str:
                    last_dt = None
                    try:
                        last_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                    except ValueError:
                        try:
                            last_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        except ValueError:
                            last_dt = None

                    if last_dt and (datetime.now(timezone.utc) - last_dt).total_seconds() > 24 * 3600:
                        logger.info(
                            "[stream_response_procedure] 触发会话重置逻辑：处于 compliance 阶段且超 24 小时无对话")
                        new_conv_json = db.end_conversation_and_create(user_id=user_id, session_id=session_id)
                        new_conv = json.loads(new_conv_json)
                        if isinstance(new_conv, dict) and "id" in new_conv:
                            new_id = str(new_conv["id"])
                            logger.info("[stream_response_procedure] 会话已重建，新 conversation_id: %s", new_id)
                            return new_id, "cognition", "cognition_01", {}
    except Exception as exc:
        logger.warning("[stream_response_procedure] 检查会话过期失败: %s", exc)

    return conversation_id, current_state, current_sub_stage, session_state


async def stream_response_procedure_async(
        session_id: str,
        user_id: str,
        bot_id: str,
        app_id: str,
        tenant_outer_id: Optional[str],
        user_input: str,
        equity: Optional[Dict[str, Dict[str, int]]] = None,
        fn: Optional[Callable[[Dict[str, object]], None]] = None,
) -> AsyncGenerator[Dict[str, object], None]:
    """异步包装器，将同步生成器转换为异步生成器"""
    import queue

    loop = asyncio.get_event_loop()

    # 使用线程安全的队列来在线程和异步循环之间传递事件
    event_queue: queue.Queue[Optional[Dict[str, object]]] = queue.Queue()

    def fn_wrapper(payload: Dict[str, object]) -> None:
        """包装 fn 回调，将事件添加到队列"""
        if fn:
            fn(payload)

    def run_sync_generator():
        """在线程中运行同步生成器"""
        try:
            for event in stream_response_procedure(
                    session_id=session_id,
                    user_id=user_id,
                    bot_id=bot_id,
                    app_id=app_id,
                    tenant_outer_id=tenant_outer_id,
                    user_input=user_input,
                    equity=equity,
                    fn=fn_wrapper,
            ):
                event_queue.put(event)
        except Exception as exc:
            logger.exception("[stream_response_procedure_async] 同步生成器执行失败: %s", exc)
            event_queue.put({"event": "error", "message": str(exc)})
        finally:
            event_queue.put(None)  # 结束标记

    # 在线程池中运行同步生成器
    loop.run_in_executor(None, run_sync_generator)

    # 异步消费事件
    while True:
        # 使用 asyncio.to_thread 来等待队列，避免阻塞事件循环
        event = await asyncio.to_thread(event_queue.get)

        if event is None:  # 结束标记
            break
        yield event


def stream_response_procedure(
        session_id: str,
        user_id: str,
        bot_id: str,
        app_id: str,
        tenant_outer_id: Optional[str],
        user_input: str,
        equity: Optional[Dict[str, Dict[str, int]]] = None,
        fn: Callable[[Dict[str, object]], None] = None,
) -> Generator[Dict[str, object], None, None]:
    """Yield progressive events while executing the main response procedure."""
    logger.info("[stream_response_procedure] 开始处理流式响应，session_id=%s, user_id=%s, bot_id=%s", session_id,
                user_id, bot_id)

    # 1. 告知客户端请求已受理，避免前端误判“无响应”
    yield {"event": "accepted", "message": "请求已接收，准备处理"}

    # 2. 查询（或创建）conversation_id，作为整个会话的主键
    logger.info("[stream_response_procedure] 获取 conversation_id")
    conversation_id = ""
    try:
        conversation_json = db.get_conversation_with_cache(user_id, session_id)
        if conversation_json:
            conv_obj = json.loads(conversation_json)
            if isinstance(conv_obj, dict) and "id" in conv_obj:
                conversation_id = str(conv_obj["id"])  # type: ignore[index]
                logger.info("[stream_response_procedure] conversation_id 获取成功: %s", conversation_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[stream_response_procedure] 获取 conversation_id 失败: %s", exc)

    # 3. 读取最新 session_state（当前阶段、子阶段、历史状态等）
    logger.info("[stream_response_procedure] 读取 session_state")
    session_state_json = db.get_latest_session_state_payload(conversation_id, session_id, user_id)
    session_state = {}
    if session_state_json:
        try:
            parsed_state = json.loads(session_state_json)
            if isinstance(parsed_state, dict):
                session_state = parsed_state
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] session_state 解析失败: %s", exc)
            session_state = {}

    current_state = session_state.get("next_stage") or "cognition"
    current_sub_stage = session_state.get("next_sub_stage") or "cognition_01"
    logger.info("[stream_response_procedure] 当前阶段: %s, 子阶段: %s", current_state, current_sub_stage)

    # 4. 拉取 bot 配置（角色 prompt、LLM 配置、快速回复开关等）
    logger.info("[stream_response_procedure] 获取 bot 配置")

    raw_bot_llm_config: Dict[str, str] | None = None
    botCharacter = ""
    fastReply = False
    bot_config = None
    structured_content: Dict[str, Any] = {}
    try:
        bot_config = get_bot_config(conversation_id, bot_id, app_id)
        structured_content = getattr(bot_config, "structuredContent", {}) or {}
        if isinstance(structured_content, dict):
            cfg = structured_content.get("botLLMConfig")
            botCharacter = structured_content.get("character")

            # 在原有角色设定后追加一条商品推荐约束提示
            if isinstance(botCharacter, str) and botCharacter.strip():
                botCharacter = botCharacter + "\n\n【商品推荐约束】\n你只能推荐【Related products/关联商品】中提到的商品"

            fastReply = structured_content.get("fastReply")
            marketing_dataset_id = structured_content.get("datasetId")
            if isinstance(cfg, dict):
                raw_bot_llm_config = cfg
                logger.info("[stream_response_procedure] bot 配置获取成功，模型: %s", cfg.get("model"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[stream_response_procedure] 获取 bot_config 失败: %s", exc)

    # 5. 准备路由阶段 prompt 映射，供后续路由与 LLM 调用使用
    route_state_prompt_map = {}
    if isinstance(structured_content, dict):
        rs_map = structured_content.get("routeStateStrategies")
        if isinstance(rs_map, dict):
            route_state_prompt_map = rs_map
    exit_keywords = structured_content.get('exitKeyword')
    logger.info("[stream_response_procedure] exit_keywords: %s", exit_keywords)

    # 快速回复
    if fastReply:
        logger.info("[stream_response_procedure] 提交快速回复任务到线程池")
        _QUICK_REPLY_EXECUTOR.submit(_generate_quick_reply, user_input=user_input, bot_character=botCharacter,
                                     conversation_id=conversation_id,
                                     session_id=session_id, user_id=user_id, fn=fn)

    if not raw_bot_llm_config:
        logger.info("[stream_response_procedure] 使用默认 bot 配置")
        raw_bot_llm_config = DEFAULT_BOT_LLM_CONFIG.copy()

    yield {
        "event": "progress",
        "stage": "bot_config_loaded",
        "detail": "已加载 bot 配置",
        "data": {
            "conversation_id": conversation_id,
            "botLLMConfig": raw_bot_llm_config,
        },
    }

    logger.info("[stream_response_procedure] 处理 equity 参数")
    payload_equity: Optional[str]
    if equity is None:
        payload_equity = None
    else:
        payload_equity = json.dumps(equity, ensure_ascii=False)

    # 解析权益信息（只保留前缀为 product_equity_ 的键）
    equity_info_map: Dict[str, Any] = {}
    if isinstance(equity, dict):
        info_part = equity.get("info")
        if isinstance(info_part, dict):
            equity_info_map = info_part
        else:
            equity_info_map = equity

    conversation_id, current_state, current_sub_stage, session_state = _check_and_reset_session(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        current_state=current_state,
        current_sub_stage=current_sub_stage,
        session_state=session_state,
    )
    logger.info(
        "[stream_response_procedure] 检查后状态: state=%s, sub_stage=%s, conv_id=%s",
        current_state,
        current_sub_stage,
        conversation_id,
    )

    def _is_stage_completed(response_payload: Any) -> bool:
        """判断阶段是否完成（仅基于 response，info 在 store_response_by_uuid 中生成）"""
        response_text = ""
        if isinstance(response_payload, dict):
            response_text = response_payload.get("response", "")
        elif isinstance(response_payload, str):
            response_text = response_payload
        return _has_content(response_text)

    yield {
        "event": "progress",
        "stage": "calling_main_procedure",
        "detail": "开始执行 main_response_procedure",
    }

    logger.info("[stream_response_procedure] 初始化 turn_uuid 与 pending_turn")
    # 生成 turn_uuid / user_message_uuid，并将用户输入写入 pending_turns，保证后续可追踪
    dt_user = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    turn_uuid = str(uuid.uuid4())
    user_message_uuid = str(uuid.uuid4())

    try:
        db.store_pending_turn(
            turn_uuid=turn_uuid,
            user_message_uuid=user_message_uuid,
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            dt_user=dt_user,
            user_content=user_input,
            bot_id=bot_id,
            app_id=app_id
        )
        pending_cache_key = f"pending:{turn_uuid}"
        pending_cache_data = {
            "turn_uuid": turn_uuid,
            "user_message_uuid": user_message_uuid,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "user_id": user_id,
            "dt_user": dt_user,
            "user_content": user_input,
            "bot_id": bot_id,
            "app_id": app_id
        }
        redis_set(pending_cache_key, json.dumps(pending_cache_data, ensure_ascii=False), expired=86400)
    except Exception as exc:
        logger.warning("[stream_response_procedure] 写入 pending_turn 失败: %s", exc)

    # ============================================================================
    # 阶段 1: Router - 路由阶段
    # ============================================================================

    # 提前加载历史消息
    history_context = db.list_chat_messages(conversation_id, session_id, user_id)

    # 异步判断是否为售后
    after_sales_future: Optional[Future] = None
    if raw_bot_llm_config:
        try:
            after_sales_future = _LLM_EXECUTOR.submit(
                judge_after_sales_intent,
                user_input=user_input,
                history_context=history_context,
                bot_llm_config=raw_bot_llm_config,
            )
        except Exception as exc:
            logger.warning("[stream_response_procedure] 提交售后检测任务失败: %s", exc)

    # 调用 router，确定当前轮次应该进入的阶段 / 子阶段
    logger.info("[stream_response_procedure] 调用 router 进行阶段路由")
    router_response, routed_current_state, routed_current_sub_stage = route_current_stage(
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        bot_id=bot_id,
        app_id=app_id,
        current_state=current_state,
        current_sub_stage=current_sub_stage,
    )

    # 检查售后判断结果
    is_after_sales = False
    early_response_result = None

    if after_sales_future:
        try:
            is_after_sales = after_sales_future.result()
            if is_after_sales:
                logger.info("[stream_response_procedure] 检测到售后意图")
                after_sales_strategy = None
                if isinstance(structured_content, dict):
                    after_sales_strategy = structured_content.get("routeStateStrategies", {}).get("after_sales")

                if after_sales_strategy:
                    logger.info("[stream_response_procedure] 使用售后策略生成回复")
                    character_prompt = botCharacter if isinstance(botCharacter, str) else ""
                    as_prompt_content = after_sales_strategy if isinstance(after_sales_strategy, str) else str(
                        after_sales_strategy)
                    full_prompt = f"{character_prompt}\n\n{as_prompt_content}" if character_prompt else as_prompt_content

                    as_response = llm_generic(
                        full_prompt=full_prompt,
                        user_input=user_input,
                        history_context=history_context,
                        session_state=session_state,
                        botLLMConfig=raw_bot_llm_config,
                        prompt_without_character=as_prompt_content,
                        input_label="用户最新输入",
                    )

                    if isinstance(as_response, dict):
                        # 强制移除 info 字段，只保留 response
                        as_response.pop("info", None)
                        early_response_result = as_response
                    else:
                        early_response_result = {"response": str(as_response)}

                    # 确保 response 字段存在
                    if "response" not in early_response_result:
                        if "text" in early_response_result:
                            early_response_result["response"] = early_response_result["text"]
                        else:
                            # 如果返回的是其他结构，尝试将其转换为字符串作为 response
                            early_response_result["response"] = json.dumps(early_response_result, ensure_ascii=False)

                    # 立即通过 fn 推送回复（模拟流式响应的最终结果）
                    if fn:
                        fn({
                            "event": "reply",
                            "data": early_response_result
                        })

                    # 保持本轮传入的 state
                    routed_current_state = current_state
                    routed_current_sub_stage = current_sub_stage
                else:
                    logger.info("[stream_response_procedure] 未找到售后策略配置，忽略售后意图")
                    is_after_sales = False
        except Exception as exc:
            logger.warning("[stream_response_procedure] 获取售后检测结果失败: %s", exc)

    character_prompt = botCharacter if isinstance(botCharacter, str) else ""

    handoff_future: Optional[Future] = None
    handoff_customer_service_flag = False
    if exit_keywords:
        logger.info("[stream_response_procedure] 检测到 exitKeyword 配置，启动转人工检测任务")
        try:
            handoff_future = _LLM_EXECUTOR.submit(
                detect_handoff_intent,
                exit_keywords=exit_keywords,
                user_input=user_input,
                history_context=history_context,
                session_state=session_state,
                bot_llm_config=raw_bot_llm_config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 提交转人工检测任务失败: %s", exc)
            handoff_future = None
    else:
        logger.info("[stream_response_procedure] 未配置 exitKeyword，跳过转人工检测")

    def _ensure_handoff_result() -> bool:
        """延迟获取转人工检测结果，避免阻塞主流程"""
        nonlocal handoff_customer_service_flag
        if not handoff_future:
            return False
        try:
            handoff_detection = handoff_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 转人工检测任务执行失败: %s", exc)
            return False
        if isinstance(handoff_detection, dict) and handoff_detection.get("handoff"):
            logger.info(
                "========== [handoff] 转人工检测命中 ==========\n关键词: %s\n原因: %s\n==========================================",
                handoff_detection.get("matched_keyword"),
                handoff_detection.get("reason"),
            )
            handoff_customer_service_flag = process_handoff_detection(
                conversation_id=conversation_id,
                turn_uuid=turn_uuid,
                fn=fn,
                threshold=_HANDOFF_TRIGGER_THRESHOLD,
                reply_text=_HANDOFF_REPLY_TEXT,
                ttl_seconds=_HANDOFF_REDIS_TTL_SECONDS,
            )
            return handoff_customer_service_flag
        else:
            logger.info("[stream_response_procedure] 转人工检测未命中或返回格式异常: %s", handoff_detection)
            return False

    # ============================================================================
    # 阶段 2: Product Analysis - 商品分析（线性流程，与 response 串行）
    # ============================================================================
    # 获取 stages_complete 用于构建上一步和下一步提示词
    stages_complete = None
    if bot_config:
        try:
            stages_complete = get_all_stages_and_substages(conversation_id, bot_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 获取 stages_complete 失败: %s", exc)

    # 构建上一步和下一步提示词
    prev_step_text, next_step_text = build_prev_next_step_texts(
        routed_current_state=routed_current_state,
        routed_current_sub_stage=routed_current_sub_stage,
        route_state_prompt_map=route_state_prompt_map,
        stages_complete=stages_complete,
    )

    # 判断是否存在商品，存在则执行商品分析 LLM
    product_analysis_result = None
    response_result = early_response_result
    response_state = routed_current_state
    response_sub_stage = routed_current_sub_stage
    equity_used_key: Optional[str] = None

    def _fetch_dataset_snippet(dataset_id: str) -> str:
        """
        根据 dataset_id 调用外部知识库检索，并返回抽取后的片段。

        这里不再捕获异常，让上层调用感知并在日志 / 监控中暴露问题，方便排查。
        """
        if not tenant_outer_id:
            return ""

        result = retrieve_dataset(
            tenant_outer_id=tenant_outer_id,
            app_id=app_id,
            dataset_id=dataset_id,
            query=user_input,
        )
        logger.info("[stream_response_procedure] 知识库检索结果: %s", result)
        snippet = extract_dataset_snippets(result)
        return snippet

    def _build_product_knowledge_content(
            products_payload: Any,
            product_ids: List[str],
    ) -> str:
        """为指定商品获取知识库内容，返回纯知识库文本（不包含商品其他信息）"""
        if not product_ids or not tenant_outer_id:
            return ""
        products_list = _extract_products_list(products_payload)
        if not products_list:
            return ""

        target_ids = [pid.strip() for pid in product_ids if pid.strip()]
        if not target_ids:
            return ""

        id_set = set(target_ids)
        knowledge_snippets: List[str] = []

        for item in products_list:
            product = item.get("product", {})
            pid = str(product.get("outerId", "")).strip()
            if pid not in id_set:
                continue
            product_extra = item.get("productExtra")
            if not isinstance(product_extra, dict):
                continue

            dataset_id = product_extra.get("datasetId", "")
            if isinstance(dataset_id, str):
                dataset_id = dataset_id.strip()
            else:
                dataset_id = ""

            if dataset_id:
                snippet = _fetch_dataset_snippet(dataset_id)
                if snippet and snippet.strip():
                    product_extra["knowledgeContent"] = snippet  # 写回商品数据，供格式化使用
                    knowledge_snippets.append(snippet.strip())

        # 返回所有知识库片段的拼接（仅知识库内容，不含商品信息）
        return "\n".join(knowledge_snippets) if knowledge_snippets else ""

    formatted_products_text = ""
    marketing_snippet: Optional[str] = None
    marketing_future: Optional[Future] = None
    if tenant_outer_id and isinstance(marketing_dataset_id, str) and marketing_dataset_id.strip():
        marketing_dataset_id = marketing_dataset_id.strip()
        try:
            marketing_future = _LLM_EXECUTOR.submit(_fetch_dataset_snippet, marketing_dataset_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 提交营销话术检索失败: %s", exc)

    try:
        # 获取商品列表
        products_cached = get_products_with_cache(conversation_id, outer_id=bot_id, app_id=app_id)
        if marketing_future:
            try:
                marketing_snippet = marketing_future.result()
                if marketing_snippet:
                    logger.info("[stream_response_procedure] 营销话术检索完成，长度=%d", len(marketing_snippet))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[stream_response_procedure] 营销话术检索失败: %s", exc)
                marketing_snippet = ""

        target_product_ids: List[str] = []

        if response_result is not None:
            logger.info("[stream_response_procedure] 已生成响应（如售后），跳过商品分析")
        elif not products_cached:
            logger.info("[stream_response_procedure] 未获取到商品列表，直接调用 response LLM")
            response_result = generate_response_payload(
                character_prompt=character_prompt,
                route_state_prompt_map=route_state_prompt_map,
                routed_current_state=response_state,
                routed_current_sub_stage=response_sub_stage,
                turn_uuid=turn_uuid,
                conversation_id=conversation_id,
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                history_context=history_context,
                session_state=session_state,
                bot_llm_config=raw_bot_llm_config,
                product_analysis_result=None,
                prev_step_text=prev_step_text,
                next_step_text=next_step_text,
                marketing_context=marketing_snippet or "",
                formatted_products_text=formatted_products_text,
                fn=fn,
            )
            logger.info(
                "[stream_response_procedure] response LLM 完成（无商品），response 长度: %d",
                len(response_result.get("response", "") if isinstance(response_result, dict) else ""),
            )
        else:  # 格式化商品列表，过滤掉已使用的权益对应的商品
            exclude_equity_keys = []
            if equity and isinstance(equity, dict):
                info = equity.get("info", {})
                if isinstance(info, dict):
                    exclude_equity_keys = list(info.keys())

            product_list_text = format_product_list_for_llm(products_cached, exclude_equity_keys)
            if product_list_text and product_list_text.strip() != "暂无商品信息":
                # 有商品，执行商品分析（线性流程，完成后推送事件）
                logger.info("[stream_response_procedure] 开始执行商品分析 LLM")
                product_analysis_result = generate_product_analysis_payload(
                    character_prompt=character_prompt,
                    route_state_prompt_map=route_state_prompt_map,
                    routed_current_state=routed_current_state,
                    routed_current_sub_stage=routed_current_sub_stage,
                    user_input=user_input,
                    history_context=history_context,
                    session_state=session_state,
                    bot_llm_config=raw_bot_llm_config,
                    turn_uuid=turn_uuid,
                    product_list_text=product_list_text,
                    # fn=fn,  # 传入 fn，商品分析完成后立即推送事件
                )
                # 1. 有推销内容时，直接推送
                if product_analysis_result.get("product_id_promoted"):
                    try:  # 推送商品推销素材事件
                        fn({
                            "event": "reply",
                            "data": {
                                "info": {
                                    "product_id_promoted": product_analysis_result.get("product_id_promoted"),
                                    "product_equity_promoted": product_analysis_result.get("product_equity_promoted"),
                                    # "product_id_sales": product_analysis_result.get("product_id_sales"),
                                    # "product_equity_sales": product_analysis_result.get("product_equity_sales"),
                                },
                                "product_info": {
                                    "sale_product_id": "",
                                    "sale_product_name": "",
                                    "promote_product_id": [product_analysis_result.get("product_id_promoted")],
                                    "promote_product_name": ""
                                },

                                "turn_uuid": turn_uuid,
                            },
                        })
                    except Exception as exc:
                        logger.warning("[generate_product_analysis_payload] 推送商品推销事件失败: %s", exc)

                logger.info(
                    "[stream_response_procedure] 商品分析完成: promoted=%s (equity=%s), workflow=%s (equity=%s)",
                    product_analysis_result.get("product_id_promoted"),
                    product_analysis_result.get("product_equity_promoted"),
                    product_analysis_result.get("product_id_sales"),
                    product_analysis_result.get("product_equity_sales"),
                )
                logger.debug(
                    "[stream_response_procedure] 商品分析原始结果: %s",
                    json.dumps(product_analysis_result, ensure_ascii=False),
                )
                sales_ids = _normalize_to_list(product_analysis_result.get("product_id_sales"))
                promo_ids = _normalize_to_list(product_analysis_result.get("product_id_promoted"))
                sales_keys = _normalize_to_list(product_analysis_result.get("product_equity_sales"))
                promo_keys = _normalize_to_list(product_analysis_result.get("product_equity_promoted"))

                # 存储商品分析结果到 pending_turns
                try:
                    products_draft = json.dumps({
                        "product_id_sales": product_analysis_result.get("product_id_sales"),
                        "product_equity_sales": product_analysis_result.get("product_equity_sales"),
                        "product_id_promoted": product_analysis_result.get("product_id_promoted"),
                        "product_equity_promoted": product_analysis_result.get("product_equity_promoted"),
                        "turn_uuid": turn_uuid
                    }, ensure_ascii=False)
                    db.update_pending_turn_state(
                        turn_uuid=turn_uuid,
                        products_draft=products_draft,
                    )
                    logger.info("[stream_response_procedure] 商品分析结果已存储到pending_turns products_draft: %s",
                                products_draft)
                except Exception as exc:
                    logger.warning("[stream_response_procedure] 存储商品分析结果失败: %s", exc)

                selected_compliance_key: Optional[str] = None
                selected_source: Optional[str] = None

                # 1. 确定用于合规跳转的权益 Key（优先使用 workflow，其次 promoted）
                # 注意：商品ID和权益ID已规范化，都是单个字符串，不再需要处理数组情况
                if sales_ids and sales_keys:
                    # 保持为单个字符串（已规范化）
                    selected_compliance_key = sales_keys[0] if sales_keys else ""
                    selected_source = "workflow"
                elif promo_ids and promo_keys:
                    # 保持为单个字符串（已规范化）
                    selected_compliance_key = promo_keys[0] if promo_keys else ""
                    selected_source = "promoted"

                # 检查权益余额：如果选中了权益但余额不足
                if selected_compliance_key:
                    if not _has_available_equity(equity_info_map, selected_compliance_key):
                        logger.info(
                            "[stream_response_procedure] 检测到权益 %s 但余额不足/为0",
                            selected_compliance_key
                        )

                        # 2. 有售卖内容，且确认用户没有该权益（余额不足）时，推送售卖信息
                        if product_analysis_result.get("product_id_sales"):
                            try:
                                fn({
                                    "event": "reply",
                                    "data": {
                                        "info": {
                                            # "product_id_promoted": product_analysis_result.get("product_id_promoted"),
                                            # "product_equity_promoted": product_analysis_result.get("product_equity_promoted"),
                                            "product_id_sales": product_analysis_result.get("product_id_sales"),
                                            "product_equity_sales": product_analysis_result.get("product_equity_sales"),
                                        },
                                        "product_info": {
                                            "sale_product_id": product_analysis_result.get("product_id_sales"),
                                            "sale_product_name": "",
                                            "promote_product_id": [],
                                            "promote_product_name": ""
                                        },
                                        "turn_uuid": turn_uuid,
                                    },
                                })
                            except Exception as exc:
                                logger.warning("[stream_response_procedure] 推送商品售卖事件失败: %s", exc)
                else:
                    try:
                        fn({
                            "event": "reply",
                            "data": {
                                "info": {
                                    # "product_id_promoted": product_analysis_result.get("product_id_promoted"),
                                    # "product_equity_promoted": product_analysis_result.get("product_equity_promoted"),
                                    "product_id_sales": product_analysis_result.get("product_id_sales"),
                                    "product_equity_sales": product_analysis_result.get("product_equity_sales"),
                                },
                                "product_info": {
                                    "sale_product_id": product_analysis_result.get("product_id_sales"),
                                    "sale_product_name": "",
                                    "promote_product_id": [],
                                    "promote_product_name": ""
                                },
                                "turn_uuid": turn_uuid,
                            },
                        })
                    except Exception as exc:
                        logger.warning("[stream_response_procedure] 推送商品售卖事件失败: %s", exc)

                        # 如果当前已在 compliance 阶段，这通常意味着权益刚用完或中断
                        if routed_current_state == "compliance":
                            logger.info("[stream_response_procedure] 处于 compliance 阶段且权益不足，执行会话结束流程")
                            try:
                                db.end_conversation_and_create(user_id=user_id, session_id=session_id)
                                logger.info("[stream_response_procedure] 会话已结束并重建")
                            except Exception as exc:
                                logger.error("[stream_response_procedure] 结束会话失败: %s", exc)

                            # 生成告别语
                            no_equity_prompt = "\n".join([
                                "[任务]",
                                "当前订阅权益已经使用完毕。请以亲切、感谢的语气，向用户说明权益已结清，真诚致谢这段陪伴，并邀请对方日后如有需要随时再来交流或续订服务。",
                                "",
                                "[补充要求]",
                                "1. 保持角色设定，语气温暖、柔和。",
                                "2. 不提及系统或技术细节，只谈服务体验。",
                                "3. 鼓励用户在需要时继续提问或选择新的服务。"
                            ])
                            farewell_prompt = "\n\n".join([character_prompt, no_equity_prompt])
                            try:
                                # 这里需要导入 llm_generic，或者使用已有的 client 调用
                                # 由于 main_response_with_stream.py 中没有直接导入 llm_generic，
                                # 我们使用已有的 _LLM_EXECUTOR 和 bot_llm_config 进行调用
                                # 或者为了方便，直接复用 generate_response_payload，但传入特殊的 prompt

                                # 为了简单且复用现有依赖，我们复用 main_response_procedure 中的 generate_response_payload 逻辑
                                # 但直接构造 prompt 可能更直接
                                farewell_raw = llm_generic(
                                    full_prompt=farewell_prompt,
                                    user_input="",
                                    history_context=history_context,
                                    session_state=session_state,
                                    botLLMConfig=raw_bot_llm_config,
                                    prompt_without_character=no_equity_prompt
                                )
                                if isinstance(farewell_raw, str):
                                    response_result = farewell_raw
                                else:
                                    response_result = str(farewell_raw)
                                logger.info("[stream_response_procedure] 已生成告别语")
                            except Exception as exc:
                                logger.warning("[stream_response_procedure] 生成告别语失败: %s", exc)
                                response_result = "感谢您的使用，本次服务已结束。欢迎随时再来！"

                            # 标记 selected_compliance_key 为 None，避免后续再次进入合规跳转逻辑
                            selected_compliance_key = None
                            selected_source = None
                        else:
                            # 如果不在 compliance 阶段，仅标记为无效权益，走正常流程（可能会被路由到其他阶段）
                            selected_compliance_key = None
                            selected_source = None

                # 2. 判断是否允许跳转合规阶段
                # 仅当当前处于 decision_making 阶段时才允许跳转
                is_decision_state = (
                        response_state == "decision_making"
                        or routed_current_state == "decision_making"
                )

                if selected_compliance_key and is_decision_state:
                    # 尝试在 bot 配置中查找该权益对应的合规模块
                    compliance_idx = find_compliance_module_index_by_product(
                        route_state_prompt_map,
                        selected_compliance_key,
                    )
                    if compliance_idx is not None:
                        # 找到合规模块：强制切换本轮 response 阶段为 compliance
                        response_state = "compliance"
                        response_sub_stage = f"compliance_{compliance_idx + 1:02d}"
                        equity_used_key = selected_compliance_key
                        logger.info(
                            "[stream_response_procedure] 使用 compliance prompt 处理权益 %s (来源=%s, sub_stage=%s)",
                            selected_compliance_key,
                            selected_source,
                            response_sub_stage,
                        )
                    else:
                        logger.info("[stream_response_procedure] 未找到合规模块，保持原阶段（权益=%s）",
                                    selected_compliance_key)
                elif selected_compliance_key and not is_decision_state:
                    logger.info(
                        "[stream_response_procedure] 检测到合规权益 %s 但当前阶段=%s，保持原阶段",
                        selected_compliance_key,
                        response_state,
                    )
                else:
                    logger.info("[stream_response_procedure] 无可用于合规跳转的销售/推广商品与权益，保持原阶段")

                # 3. 构建商品知识库上下文（用于 Prompt）
                promo_ids = _normalize_to_list(product_analysis_result.get("product_id_promoted"))
                if promo_ids and products_cached:
                    promo_snippet = build_promotion_product_snippet(products_cached, promo_ids)
                    if promo_snippet:
                        product_analysis_result["promoted_product_details"] = promo_snippet
                    target_product_ids.extend(promo_ids)
                sales_ids = _normalize_to_list(product_analysis_result.get("product_id_sales"))
                target_product_ids.extend(sales_ids)

                # 获取商品知识库内容（仅知识库文本，不含商品其他信息）
                if target_product_ids and products_cached:
                    knowledge_content = _build_product_knowledge_content(products_cached, target_product_ids)
                    if knowledge_content:
                        product_analysis_result["knowledgeContent"] = knowledge_content
                        logger.info(
                            "[stream_response_procedure] 已获取商品知识库内容，目标商品数=%d，知识库长度=%d",
                            len(set(target_product_ids)),
                            len(knowledge_content),
                        )

                response_result = generate_response_payload(
                    character_prompt=character_prompt,
                    route_state_prompt_map=route_state_prompt_map,
                    routed_current_state=response_state,
                    routed_current_sub_stage=response_sub_stage,
                    turn_uuid=turn_uuid,
                    conversation_id=conversation_id,
                    session_id=session_id,
                    user_id=user_id,
                    user_input=user_input,
                    history_context=history_context,
                    session_state=session_state,
                    bot_llm_config=raw_bot_llm_config,
                    product_analysis_result=product_analysis_result,
                    prev_step_text=prev_step_text,
                    next_step_text=next_step_text,
                    marketing_context=marketing_snippet or "",
                    formatted_products_text=formatted_products_text,
                    fn=fn,
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[stream_response_procedure] 商品分析执行失败: %s", exc)
        response_result = generate_response_payload(
            character_prompt=character_prompt,
            route_state_prompt_map=route_state_prompt_map,
            routed_current_state=response_state,
            routed_current_sub_stage=response_sub_stage,
            turn_uuid=turn_uuid,
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            history_context=history_context,
            session_state=session_state,
            bot_llm_config=raw_bot_llm_config,
            product_analysis_result=None,
            prev_step_text=prev_step_text,
            next_step_text=next_step_text,
            marketing_context=marketing_snippet or "",
            formatted_products_text=formatted_products_text,
            fn=fn,
        )

    # ============================================================================
    # 阶段 4: Response LLM - 生成自然语言回复（与 product_analysis 串行）
    # ============================================================================
    if response_result is None:
        response_result = generate_response_payload(
            character_prompt=character_prompt,
            route_state_prompt_map=route_state_prompt_map,
            routed_current_state=response_state,
            routed_current_sub_stage=response_sub_stage,
            turn_uuid=turn_uuid,
            conversation_id=conversation_id,
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            history_context=history_context,
            session_state=session_state,
            bot_llm_config=raw_bot_llm_config,
            product_analysis_result=product_analysis_result,
            prev_step_text=prev_step_text,
            next_step_text=next_step_text,
            marketing_context=marketing_snippet or "",
            formatted_products_text=formatted_products_text,
            fn=fn,
        )

    if isinstance(response_result, dict):
        response_payload_for_cache = dict(response_result)
    else:
        response_payload_for_cache = {"response": response_result}
    if "response" not in response_payload_for_cache:
        response_payload_for_cache["response"] = ""
    add_conversation_id_uuid_and_cache(
        response_payload_for_cache,
        conversation_id=conversation_id,
        session_id=session_id,
        user_id=user_id,
        turn_uuid=turn_uuid,
    )
    response_result = response_payload_for_cache

    # ============================================================================
    # 阶段完成 & 下一阶段计算
    # ============================================================================
    # 注意：info 在 store_response_by_uuid 中基于 response 生成，此处不再生成
    response_text_for_stage = ""
    if isinstance(response_result, dict):
        response_text_for_stage = response_result.get("response", "")
    elif isinstance(response_result, str):
        response_text_for_stage = response_result

    current_substage_full = normalize_substage_name(response_state, response_sub_stage)

    stage_payload_entry: Dict[str, Any] = {}
    if _has_content(response_text_for_stage):
        stage_payload_entry["response"] = response_text_for_stage

    # 如果是售后流程，强制写入一个占位 info，防止后续流程（如 store_response_by_uuid）错误地自动补全 info
    if early_response_result is not None and "response" in stage_payload_entry:
        # 这里假设 early_response_result 存在即意味着走了售后/特殊分支
        stage_payload_entry["info"] = {"after_sales": True}

    if stage_payload_entry:
        try:
            db.update_pending_turn_state(
                turn_uuid=turn_uuid,
                routed_current_state=response_state,
                routed_current_sub_stage=response_sub_stage or current_substage_full,
                stage_payload_draft=json.dumps({current_substage_full: stage_payload_entry}, ensure_ascii=False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 更新阶段 payload 失败: %s", exc)

    next_stage = response_state
    next_substage = response_sub_stage or current_substage_full
    stage_completed = _is_stage_completed(response_result)

    if stage_completed and bot_config is not None:
        try:
            set_all_stages_and_substages(conversation_id, current_substage_full, True, bot_config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 标记阶段完成失败: %s", exc)
        try:
            stages_complete = get_all_stages_and_substages(conversation_id, bot_config)
            stages_list = list(stages_complete.keys())
            if current_substage_full in stages_list:
                idx = stages_list.index(current_substage_full)
                if idx + 1 < len(stages_list):
                    next_substage_full = stages_list[idx + 1]
                    next_stage = next_substage_full.rsplit("_", 1)[
                        0] if "_" in next_substage_full else next_substage_full
                    next_substage = next_substage_full
        except Exception as exc:  # noqa: BLE001
            logger.warning("[stream_response_procedure] 计算下一阶段失败: %s", exc)

    try:
        db.update_pending_turn_state(
            turn_uuid=turn_uuid,
            next_stage=next_stage,
            next_substage=next_substage,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[stream_response_procedure] 更新下一阶段失败: %s", exc)

    # 延迟获取转人工检测结果（在 response 输出之后）
    _ensure_handoff_result()

    # 构建返回结果（info 在 store_response_by_uuid 中生成，此处只保留转人工标志）
    info_output: Dict[str, Any] = {}
    if handoff_customer_service_flag:
        info_output["customer_service"] = True
    if equity_used_key:
        info_output["equity_used"] = equity_used_key

    result_payload = {
        "info": info_output if info_output else None,  # 如果没有转人工标志，则不包含 info
        "turn_uuid": turn_uuid,
    }
    # 移除空的 info 字段
    if result_payload.get("info") is None:
        result_payload.pop("info", None)

    result_raw = json.dumps(result_payload, ensure_ascii=False)
    logger.info("[stream_response_procedure] 响应结果构建完成，长度: %d", len(result_raw))

    logger.info("[stream_response_procedure] 解析返回结果")
    try:
        result_parsed = json.loads(result_raw)
    except Exception:
        result_parsed = {"raw": result_raw}

    # 如果包含 info，通过 fn 发送（因为之前的 fn 调用只发了 response）
    if result_parsed.get("info") and fn:
        # 确保 response 也包含在内，以便前端显示
        reply_data = dict(result_parsed)
        if "response" not in reply_data and "response" in response_result:
            reply_data["response"] = response_result["response"]

        fn({
            "event": "reply",
            "data": reply_data
        })

    yield {
        "event": "reply",
        "data": result_parsed,
    }
    yield {"event": "end"}
    logger.info("[stream_response_procedure] 流式响应处理完成")
