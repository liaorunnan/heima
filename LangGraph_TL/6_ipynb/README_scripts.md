# 6\_ipynb 脚本知识速览

本目录的 5 个脚本围绕 **LangGraph 断点管控 + 工具调用代理** 展开，以下文档依场景总结关键实现与可复用做法，便于快速理解和迁移。

---

## 公共依赖与约定
- 环境变量：`DEEPSEEK_API_KEY`、`DEEPSEEK_URL`、`OPENWEATHER_API_KEY`、`google_serper_KEY`、可选 `OPENAI_API_KEY`。
- 模型封装：全部使用 `init_chat_model(..., model="deepseek-chat", temperature=0)`。
- 图编译：统一依赖 `StateGraph` + `MemorySaver`，并通过 `interrupt_before=[...]` 注入断点。
- 消息体：安全场景下用 `MessagesState`，自定义状态下用 `TypedDict`（`user_input`、`model_response`、`user_approval`）。

---

## `test_46.py` – 单轮审批流演示
- **目标**：当用户输入含“删除”时，自动在 `execute_users` 节点打断点，请求人工审批后再继续流水线。
- **节点设计**：
  - `call_model`：判定风险输入，直接给 `user_approval` 打上 “人工确认” or “直接运行” 标签。
  - `execute_users`：管理员根据 `user_approval` 结果给出最终提示语。
  - `translate_message`：将模型输出统一翻译成英文，便于对外输出。
- **断点机制**：`graph.compile(..., interrupt_before=["execute_users"])`，通过 `graph.stream(None, config)` 续跑。
- **示例流程**：脚本内展示 7 次调用（高风险删除×2、审批通过/拒绝、非敏感查询），串联了 `graph.get_state()`、`graph.update_state()`、`graph.stream(None, ...)` 的全生命周期操作。

**速用要点**：
1. `graph.stream({"user_input": ...}, config)` 只跑到断点。
2. `snapshot = graph.get_state(config)` 后可直接修改 `snapshot.values` 来注入人工审批结论。
3. 续跑时输入 `None`，图会从断点继续向下执行。

---

## `test_47.py` – 命令行对话版审批流
- **目标**：把 `test_46.py` 的逻辑封装进 CLI 循环，允许多轮人工输入与审批。
- **核心函数 `run_dialogue`**：
  1. 读取终端输入（输入 `退出` 结束）。
  2. 触发 `graph.stream({"user_input": ...}, ...)` 并收集 `chunk`。
  3. 若最新 `chunk` 内的 `user_approval` 包含 “请人工确认”，立刻提示管理员输入 “是/否”。
  4. 调用 `graph.update_state(config, {"user_approval": user_approval})`，然后通过 `graph.stream(None, ...)` 完成后续节点。
- **输出**：最终模型回复统一打印在 `人工智能助理：...` 行内。
- **差异点**：无翻译节点的特殊处理；主流程强调如何在循环中复用 `thread_id` 与 `all_chunks`。

---

## `test_48.py` – 工具代理的动态断点
- **目标**：在复杂代理中对“工具执行”阶段加断点，从而实时审阅工具调用。
- **工具集合**：
  - `get_weather`：OpenWeather 查询。
  - `fetch_real_time_info`：Serper 实时搜索。
- **模型绑定**：`llm.bind_tools(tools)`，输出结构自动包含 `tool_calls`。
- **图结构**：
  - `agent` 节点：调用 LLM。
  - `action` 节点：LangGraph 自带 `ToolNode`，实际执行工具。
  - 条件边 `should_continue`：根据 `last_message.tool_calls` 判断进入 `action` or `END`。
- **断点设置**：`interrupt_before=["action"]` 可让每次工具执行前停下。
- **示例**：脚本演示查询北京天气与搜索 OpenAI 新闻，强调 `graph.stream(None, config)` 多次续跑以覆盖多轮工具调用。

**应用提示**：若代理链路复杂、希望在真实调用外部 API 前做合规审批，可复用该模式把 `action` 或更细粒度节点设为断点。

---

## `test_49.py` – 带数据库的风险操作审批
- **功能矩阵**：
  - 工具层：`get_weather`（API 查询）、`insert_weather_to_db`、`query_weather_from_db`、`delete_weather_from_db`。
  - 数据库：SQLAlchemy + MySQL（表 `weather_333`，字段包含城市气象指标）。
- **风险控制**：
  - `should_continue` 将 `delete_weather_from_db` 路由到独立节点 `risk_tool`。
  - `risk_tool` 只注册高危工具，由人工决定是否执行；真实执行前 `interrupt_before=["risk_tool"]` 会强制中断。
- **交互脚本**：
  - 前 4 步演示独立命令（查天气、写库、比对多城市、触发删除审批）。
  - 第 5 步循环任务展示如何在审批通过/拒绝时修改 `graph` 状态：
    - 如果管理员允许：`graph.update_state(config=config, values=chunk)` 后 `graph.stream(None, ...)`。
    - 如果拒绝：构造 `role="tool"` 的替代消息，把 “管理员不允许执行该操作！” 注入到 `risk_tool`。
- **附加 API**：利用 `graph.get_state(config)` 查看 `state.tasks`、`state.next` 以判断当前断点位置。

---

## `test_50.py` – 多轮会话 + 动态审批 UI
- **继承 `test_49`** 的所有工具/数据库/图结构，但封装成更友好的多轮助手。
- **`run_multi_round_dialogue` 流程**：
  1. 循环询问用户意图，输入 `退出` 结束。
  2. 每轮调用 `graph.stream({"messages": user_input}, ...)`，随后检查 `state.tasks`。
  3. 如果任务队列首个节点是 `risk_tool`，进入人工审批分支。
  4. 审批通过：沿用图状态继续执行；审批拒绝：生成 `tool` 角色反馈阻断删除。
  5. 所有回复统一以 “人工智能助理：...” 格式输出。
- **价值**：提供“代理 + 数据库 + 人工监管”的端到端形态，便于直接对话体验或集成到客服后台。

---

## 迁移/复用建议
1. **断点选择**：`interrupt_before` 可以指向任意节点，若要更精细可针对具体工具调用创建独立节点（如 `risk_tool`）。
2. **状态注入**：通过 `graph.update_state(config, values, as_node=...)` 可在任意节点层面覆盖消息；结合 `graph.get_state` 能精确感知上下文。
3. **任务队列**：`state.tasks` 是判断还有哪些节点待执行的权威来源，写交互式审批 UI 时必须轮询它。
4. **工具安全**：对高危操作（删库、转账等）单独建节点，避免与普通 `ToolNode` 共用，便于插拔安全策略。
5. **线程隔离**：`config = {"configurable": {"thread_id": "<id>"}}` 控制并发上下文；多用户场景务必为每位用户分配唯一线程。

---

如需进一步扩展（例如新增审计日志、将审批写入数据库、把 CLI 改为 Web），可在现有图结构中增加新的节点或在 `risk_tool` 中加入自定义逻辑即可。

