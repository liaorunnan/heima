from typing import Annotated, List, Literal
from typing_extensions import TypedDict
import json
import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode


from llm_input import model
from rag.embedding import get_embedding
# 导入向量数据库操作类 (假设这些类在你的项目中已定义好)
from workflow_db_yiwang import CoreMemory, ArchivalMemory, ChatLog



try:
    CoreMemory.init()
    ArchivalMemory.init()
    ChatLog.init()

except Exception as e:
    print(f"索引初始化跳过或出错 (通常是因为已存在): {e}")

core_memory_db = CoreMemory()
archival_memory_db = ArchivalMemory()
chat_log_db = ChatLog()

def get_vector(content: str):
    return get_embedding(content).tolist()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    core_memory: str
    
@tool
def update_core_memory(content: str, thread_id: str):
    """
    更新用户的【核心画像】。
    
    仅当用户提供了长期有效的、关键的个人属性（如姓名、职业、长期性格、家庭成员、重大偏好）时使用。
    不要用于记录临时的、琐碎的对话内容。
    
    Args:
        content (str): 完整的用户画像描述语句。
        thread_id (str): 当前对话的会话ID。
    """

    vector = get_vector(content)
    print(f"【系统】正在更新核心记忆: {content} (会话ID: {thread_id})")
    core_memory_db.update_by_thread_id(thread_id, content, vector)
    return f"核心记忆已更新: {content}"



@tool
def add_archival_memory(content: str, thread_id: str):
    """
    添加【归档记忆】。
    
    用于记录具体的独立事件、知识点、经历或具体的对话细节。
    注意：在调用此工具前，必须先调用 `search_archival_memory` 进行查重。
    
    Args:
        content (str): 要归档的事件或知识描述。
        thread_id (str): 当前对话的会话ID。
    """
    vector = get_vector(content)
    

    tags = []
    for keyword in ["出差", "会议", "计划", "报错", "需求"]:
        if keyword in content:
            tags.append(keyword)
    

    search_results = archival_memory_db.search_by_vector(thread_id, vector, k=1, decay_rate=0.1)
    if search_results:
        existing_content = search_results[0].content
        # 简单的相似度判断
        if content in existing_content or existing_content in content:
            print(f"【系统拦截】检测到重复记忆，跳过写入。")
            return "相似的记忆已存在，已跳过写入。"
    
    archival_memory_db.add(thread_id, content, vector, tags)
    print(f"【系统】已归档新记忆: {content}")
    return "已成功添加到归档记忆。"

@tool
def search_archival_memory(query: str, thread_id: str):
    """
    检索【归档记忆】。
    
    当需要回答关于过去的问题、回忆细节、或者在写入新记忆前进行查重时，必须调用此工具。
    返回结果已包含时间衰减权重（越近越重要），并带有记忆ID和时间戳。
    
    Args:
        query (str): 搜索关键词。
        thread_id (str): 当前对话的会话ID。
    """
    print(f"【系统】正在检索: {query}")
    query_vector = get_vector(query)
    

    results = archival_memory_db.search_by_vector(thread_id, query_vector, k=10)
    
    if not results:
        results = archival_memory_db.search_by_keyword(thread_id, query, k=10)
    
    if not results:
        return "未找到相关归档记忆。"
    
    return "\n".join([f"[ID: {r.id}] (时间: {r.created_at}) 内容: {r.content}" for r in results])

@tool
def delete_duplicate_memory(id_list: str):
    """
    删除指定的【归档记忆】。
    
    当发现记忆冲突、重复或错误时使用。
    
    Args:
        id_list (str): 要删除的记忆ID列表，多个ID之间必须用英文逗号分隔。例如: "uuid1,uuid2"
    """
    ids = id_list.split(",")
    print(f"【系统】正在删除记忆ID: {ids}")
    count = 0
    for mid in ids:
        mid = mid.strip()
        if mid:
            archival_memory_db.delete_by_id(mid)
            count += 1
    return f"已成功删除 {count} 条归档记忆。"

# 绑定工具
tools = [update_core_memory, add_archival_memory, search_archival_memory, delete_duplicate_memory]
model_with_tools = model.bind_tools(tools)




def get_system_prompt(core_memory_content, thread_id):

    print("*"*30)
    print("当前核心记忆内容："+core_memory_content)
    print("*"*30)
    return f"""
# Role & Objective
你是一个拥有长期记忆的智能助手。你的目标是维护一个准确、整洁且随时间演进的用户记忆库。
你拥有完全的记忆管理权限，请根据以下作业流程（SOP）主动管理记忆。

# Context: Core Memory (User Persona)
以下是你目前对用户的核心认知：
<core_memory>
{core_memory_content}
</core_memory>

# Current Session Information
当前会话的 thread_id 是: {thread_id}

# Methodology & SOP (记忆管理作业流程)

## 1. Retrieval Strategy (检索策略)
*   **核心优先**：回答问题前，首先检查 `<core_memory>`。如果答案已存在，**严禁**调用搜索工具。
*   **按需检索**：只有当 `<core_memory>` 无法回答，且问题涉及过去的历史细节时，才调用 `search_archival_memory`。

## 2. Upgrade Strategy (记忆升级策略 - 重要)
*   **提炼核心**：如果你在检索 `search_archival_memory` 的结果中，发现了用户的**长期属性**（如：职业、性格、家庭、长期目标、重大偏好），且这些信息尚未存在于 `<core_memory>` 中：
    *   **必须**调用 `update_core_memory` 将其提取并固化。
    *   这有助于减少未来的检索开销。

## 3. Storage Strategy (存储与去重策略)
*   **查重原则**：在决定调用 `add_archival_memory` 记录新事件前，**必须先进行心理自查或调用搜索**，确认该事件是否已被记录。
*   **冲突处理**：如果发现新信息与检索到的旧记忆（Old ID）冲突：
    1.  以当前的最新对话为准。
    2.  调用 `delete_duplicate_memory` 删除旧的、错误的记忆 ID。
    3.  记录新的记忆。

## 4. Constraint (限制)
*   不要重复记录 `<core_memory>` 中已有的信息到归档记忆中。
*   调用 `delete_duplicate_memory` 时，ID 必须严格来源于 `search_archival_memory` 的返回结果。
*   所有记忆操作（如写入、检索、删除）都必须基于当前会话的 thread_id(位于State中)。
*   **CRITICAL: 调用任何记忆相关工具时，必须明确指定 thread_id 参数，值为当前会话的 thread_id ({thread_id})。绝对禁止使用默认值"default"！**
""".strip()



def load_memories(state: AgentState):
    """加载阶段：只负责把核心数据注入 State，不负责 System Prompt 的拼接"""
    thread_id = state["thread_id"]
    print(thread_id)
    core_mem_results = core_memory_db.query_by_thread_id(thread_id)
    
    core_content = core_mem_results[0].content if core_mem_results else "（暂无用户核心画像，请在对话中收集）"
    state["core_memory"] = core_content
    print(state)
    return state

def call_model(state: AgentState):
    """执行阶段：动态合成 Prompt"""
    messages = state["messages"]
    core_memory = state["core_memory"]
    thread_id = state["thread_id"]

    
    system_message = SystemMessage(content=get_system_prompt(core_memory, thread_id))
    

    final_messages = [system_message] + [m for m in messages if not isinstance(m, SystemMessage)]
    
    response = model_with_tools.invoke(final_messages)
    return {"messages": [response]}

def log_history(state: AgentState):
    """日志记录阶段"""
    last_message = state["messages"][-1]
    thread_id = state["thread_id"]
    
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        vector = get_vector(last_message.content)
        chat_log_db.add(thread_id, "ai", last_message.content, vector)
    return {}



workflow = StateGraph(AgentState)

workflow.add_node("load_memories", load_memories)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("logger", log_history)

workflow.add_edge(START, "load_memories")
workflow.add_edge("load_memories", "agent")

def should_continue(state: AgentState) -> Literal["tools", "logger"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "logger"

workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "load_memories")  
workflow.add_edge("load_memories", "agent")  
workflow.add_edge("logger", END)

memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)


def run_chat(user_input: str, thread_id: str):
    print(f"\n>>> 用户({thread_id}): {user_input}")
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "thread_id": thread_id
    }
    
    # 记录用户输入日志
    user_vector = get_vector(user_input)
    chat_log_db.add(thread_id, "user", user_input, user_vector)

    print("\n--- 记忆库加载 ---")

    for event in app.stream(initial_state, config=config):
        for key, value in event.items():
            if key == "agent":
                msg = value["messages"][0]
                if msg.content:
                    print(f"Agent: {msg.content}")
                if msg.tool_calls:
                    print(f"   [触发工具]: {msg.tool_calls[0]['name']}")
            elif key == "tools":
                # 打印工具输出的前50个字符，防止刷屏
                tool_output = value["messages"][0].content
                print(f"   [工具结果]: {tool_output[:100]}...")
    print("\n--- 结束 ---")
                

# --- 测试用例 ---
if __name__ == "__main__":

    
    # # 1. 第一次对话：建立核心记忆
    # run_chat("你好，我叫张三，我是一名在大厂工作的程序员。", "1")

    # # 2. 第二次对话：测试核心记忆的读取（不需要重新告诉它名字）
    # run_chat("我刚才说我是干什么的来着？", "1")

    # # 3. 第三次对话：测试归档记忆（记录细节）
    # run_chat("下周一我要去北京出差，帮我记一下。", "1")



    # # 5. 切换用户：确保记忆隔离
    # run_chat("你好，我是李四。", "2")
    # run_chat("我叫什么名字？", "2")

    # # 4. 第四次对话：测试归档记忆的检索
    # run_chat("我最近有什么出差计划吗？", "1")

    # 测试对话
    thread_id = "3"
    
    # 第一次对话：建立核心记忆
    run_chat("你好，我叫张三，我是一名在大厂工作的程序员。", thread_id)
    
    # 第二次对话：测试核心记忆是否被保存和检索
    run_chat("我叫什么名字？", thread_id)