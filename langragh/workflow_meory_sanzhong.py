from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from datetime import datetime
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# --- 配置导入 ---
from llm_input import model  # 确保这个 model 支持 bind_tools
from conf import settings

from rag.embedding import get_embedding

# 导入向量数据库操作类
from workflow_db import CoreMemory, ArchivalMemory, ChatLog

# 初始化向量数据库实例

core_memory_db = CoreMemory()
archival_memory_db = ArchivalMemory()
chat_log_db = ChatLog()

# ==========================================
# 2. 定义工具 (给模型管理记忆的能力)
# ==========================================

# 辅助函数：获取向量
def get_vector(content: str):
    """获取内容的向量表示"""
    return get_embedding(content).tolist()

@tool
def update_core_memory(content: str, thread_id: str):
    """
    【核心记忆工具】
    当用户提供了重要的个人属性（如姓名、职业、性格、偏好）时调用此工具。
    这将覆盖或追加更新用户的核心画像。
    content: 新的用户画像描述。
    """
    # 获取内容向量
    vector = get_vector(content)
    
    # 更新核心记忆
    core_memory_db.update_by_thread_id(thread_id, content, vector)
    return f"核心记忆已更新: {content}"

@tool
def add_archival_memory(content: str, thread_id: str):
    """
    【归档记忆工具】
    当对话中出现值得记录的独立事件、知识或经历时（非用户属性），调用此工具归档。
    content: 具体的事件或知识描述。
    """
    # 获取内容向量
    vector = get_vector(content)
    
    # 自动提取标签（简单实现，可根据需要改进）
    tags = []
    if "出差" in content:
        tags.append("出差")
    if "会议" in content:
        tags.append("会议")
    if "计划" in content:
        tags.append("计划")
    
    # 先检查是否已经存在相似的记忆
    # 使用内容作为查询，检查是否已有相似的记忆
    search_results = archival_memory_db.search_by_vector(thread_id, vector, k=1)
    
    # 如果存在相似的记忆（相似度较高的情况），则不再添加
    if search_results:
        # 检查内容是否基本相同（简单的字符串相似度检查）
        existing_content = search_results[0].content
        if content in existing_content or existing_content in content:
            print(f"【归档记忆工具】: 相似的记忆已存在，不再重复添加")
            return "相似的记忆已存在，不再重复添加。"
    
    # 添加归档记忆
    archival_memory_db.add(thread_id, content, vector, tags)
    print(f"【归档记忆工具】: 已添加新记忆")
    return "已添加至归档记忆。"

@tool
def search_archival_memory(query: str, thread_id: str):
    """
    【记忆检索工具】
    当需要回忆之前的具体细节或历史事件时调用。
    query: 搜索关键词。
    """
    print("【记忆检索工具】:")
    print(query)
    print(thread_id)
    
    # 获取查询向量
    query_vector = get_vector(query)
    
    # 使用向量检索归档记忆
    results = archival_memory_db.search_by_vector(thread_id, query_vector, k=10)
    
    if not results:
        # 如果向量检索没有结果，尝试关键词检索
        results = archival_memory_db.search_by_keyword(thread_id, query, k=10)
    
    if not results:
        return "未找到相关归档记忆。"
    
    return "\n".join([f"- {r.content}" for r in results])

# 绑定工具到模型
tools = [update_core_memory, add_archival_memory, search_archival_memory]
model_with_tools = model.bind_tools(tools)

# ==========================================
# 3. Graph 状态与逻辑
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    core_memory: str  # 专门存放读取到的核心记忆

# 节点：加载记忆上下文
def load_memories(state: AgentState):
    thread_id = state["thread_id"]
    
    # 1. 读取核心记忆
    core_mem_results = core_memory_db.query_by_thread_id(thread_id)
    core_content = core_mem_results[0].content if core_mem_results else "暂无用户核心画像。"
    
    # 2. 可以在这里做自动检索（可选），或者完全依赖 Agent 主动调用 search 工具
    # 这里我们只加载核心记忆进入 State
    return {"core_memory": core_content}

# 节点：调用模型
def call_model(state: AgentState):
    messages = state["messages"]
    core_memory = state["core_memory"]
    
    # 构造 System Prompt
    system_prompt = (
        "你是一个拥有长期记忆的智能助手。\n"
        "--- [核心记忆] ---\n"
        f"{core_memory}\n"
        "------------------\n"
        "你可以使用工具来：\n"
        "1. `update_core_memory`: 更新用户的核心画像（如改名、换工作）。\n"
        "2. `add_archival_memory`: 记录具体的历史事件。\n"
        "3. `search_archival_memory`: 搜索之前的对话细节。\n"
        "请根据对话内容自动判断是否需要操作记忆：\n"
        "- 如果核心记忆中已有答案，不必再查别的记忆\n"
        "- 如果在记忆中找到答案，绝对不要再次添加到归档记忆中\n"
        "- 只有当信息是新的且值得记住时，才使用 add_archival_memory 工具\n"
        "注意：调用工具时，必须传入当前的 thread_id。"
    )
    
    # 将 SystemMessage 插在最前面，或者如果最前面已经是 SystemMessage 则替换
    final_messages = [SystemMessage(content=system_prompt)] + messages
    
    response = model_with_tools.invoke(final_messages)
    return {"messages": [response]}

# 节点：记录日志（模拟回溯记忆的存储）
def log_history(state: AgentState):
    last_message = state["messages"][-1]
    thread_id = state["thread_id"]
    
    # 只记录 AI 的回复和 用户的输入（工具调用过程可选记录）
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # 获取内容向量
        vector = get_vector(last_message.content)
        
        # 添加到对话日志
        chat_log_db.add(thread_id, "ai", last_message.content, vector)
    
    # 注意：LangGraph 的 add_messages 已经处理了内存中的由 MemorySaver 负责的状态
    return {}

# ==========================================
# 4. 构建图
# ==========================================

workflow = StateGraph(AgentState)

# 定义节点
workflow.add_node("load_memories", load_memories)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # LangGraph 预置的工具执行节点
workflow.add_node("logger", log_history)

# 定义边
workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "agent")

def should_continue(state: AgentState) -> Literal["tools", "logger"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "logger"

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # 工具执行完，结果回传给 Agent 继续生成回复
workflow.add_edge("logger", END)

# 编译
memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)

# ==========================================
# 5. 运行测试
# ==========================================

def run_chat(user_input: str, thread_id: str):
    print(f"\n>>> 用户({thread_id}): {user_input}")
    config = {"configurable": {"thread_id": thread_id}}
    
    # 构造输入状态
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "thread_id": thread_id
    }
    
    # 记录用户输入到向量数据库
    user_vector = get_vector(user_input)
    chat_log_db.add(thread_id, "user", user_input, user_vector)

    # 流式运行
    final_response = ""
    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            if key == "agent":
                msg = value["messages"][0]
                if msg.content:
                    print(f"Agent: {msg.content}")
                    print("*"*30)
                if msg.tool_calls:
                    print(f"   [触发工具]: {msg.tool_calls[0]['name']}")
            elif key == "tools":
                msg = value["messages"][0]
                print(f"   [工具结果]: {msg.content}")

# --- 测试场景 ---

# 1. 第一次对话：建立核心记忆
run_chat("你好，我叫张三，我是一名在大厂工作的程序员。", "1")

# 2. 第二次对话：测试核心记忆的读取（不需要重新告诉它名字）
run_chat("我刚才说我是干什么的来着？", "1")

# 3. 第三次对话：测试归档记忆（记录细节）
run_chat("下周一我要去北京出差，帮我记一下。", "1")



# 5. 切换用户：确保记忆隔离
run_chat("你好，我是李四。", "2")
run_chat("我叫什么名字？", "2")

# 4. 第四次对话：测试归档记忆的检索
run_chat("我最近有什么出差计划吗？", "1")


# 4. 第四次对话：测试归档记忆的检索
run_chat("我叫什么名字", "1")