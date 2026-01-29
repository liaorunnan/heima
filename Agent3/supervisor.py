import operator
import json
from typing import Annotated, List, Literal, TypedDict, Union

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 假设 conf.settings 存在，替换为你实际的导入
from conf import settings 

# --- 1. 定义工具 (Tools) ---
@tool
def add_function(a: float, b: float) -> float:
    """计算 a + b"""
    return a + b

@tool
def multiply_function(a: float, b: float) -> float:
    """计算 a * b"""
    return a * b

@tool
def save_result(content: str):
    """用于将最终结果写入 result.txt 文件"""
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(str(content))
    return f"结果 {content} 已成功保存到 result.txt"

# --- 2. 核心修复：定义 Supervisor (使用 Parser 替代 structured_output) ---

# 定义输出结构
class RouteResponse(BaseModel):
    next: Literal["Adder", "Multiplier", "Saver", "FINISH"] = Field(
        ..., description="下一个执行任务的角色名称"
    )

# 初始化解析器
parser = PydanticOutputParser(pydantic_object=RouteResponse)

# --- 3. 定义图的状态 (State) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# --- 4. 初始化 LLM ---
llm = ChatOpenAI(
    model=settings.model_name,
    temperature=0,
    api_key=settings.api_key,
    base_url=settings.base_url
)

# --- 5. 创建 Worker 节点逻辑 ---
# 辅助函数：运行 LLM -> 检查是否要调工具 -> 执行工具 -> 返回结果
def create_worker_node(tools, system_prompt):
    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # 构建工具字典，方便后续查找执行
    tool_map = {t.name: t for t in tools}

    def worker_node(state):
        messages = state['messages']
        # 1. 构造 Prompt，包含 System 指令和历史
        prompt = [("system", system_prompt)] + messages
        
        # 2. 调用 LLM
        response = llm_with_tools.invoke(prompt)
        
        # 3. 如果 LLM 决定调用工具，我们必须在这里执行它！
        # (原代码缺少这一步，导致只会空转)
        results = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                selected_tool = tool_map[tool_call["name"]]
                # 执行工具
                tool_output = selected_tool.invoke(tool_call["args"])
                # 创建工具消息
                results.append(ToolMessage(
                    content=str(tool_output), 
                    tool_call_id=tool_call["id"]
                ))
            
            # 返回 LLM 的回答(包含调用请求) + 工具的执行结果
            return {"messages": [response] + results}
        else:
            # 如果没调工具，直接返回回答
            return {"messages": [response]}
            
    return worker_node

# 创建具体的 Worker 节点
add_node = create_worker_node(
    [add_function], 
    "你是一个加法助手。收到请求后，请务必调用 add_function 工具进行计算。"
)

multiply_node = create_worker_node(
    [multiply_function], 
    "你是一个乘法助手。收到请求后，请务必调用 multiply_function 工具进行计算。"
)

saver_node = create_worker_node(
    [save_result], 
    "你是一个记录员。收到数字后，请务必调用 save_result 工具保存。"
)

# --- 6. 核心修复：实现 Supervisor 节点 ---
def supervisor_node(state):
    members = ["Adder", "Multiplier", "Saver"]
    
    system_prompt = (
        "你是一个主管（Supervisor），负责管理以下工人：{members}。\n"
        "用户输入了一个数学任务。请根据对话历史，决定下一步由谁来行动。\n\n"
        "逻辑规则：\n"
        "1. 优先处理乘除法 (Multiplier)。\n"
        "2. 乘除法处理完后，处理加减法 (Adder)。\n"
        "3. 得到了最终数值结果后，必须交给 Saver 保存。\n"
        "4. 只有当 Saver 明确表示'已保存'后，才返回 FINISH。\n\n"
        "{format_instructions}"
    )
    
    # 将解析器的指令注入 Prompt，让模型输出 JSON
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "根据上述情况，下一步是谁？")
    ]).partial(
        members=", ".join(members),
        format_instructions=parser.get_format_instructions()
    )
    
    # 链：Prompt -> LLM -> Parser (纯文本处理，不依赖 API 特定功能)
    supervisor_chain = prompt | llm | parser
    
    try:
        response = supervisor_chain.invoke(state)
        return {"next": response.next}
    except Exception as e:
        print(f"Supervisor 解析错误: {e}, 尝试重试或结束")
        return {"next": "FINISH"}

# --- 7. 构建图 (Graph Construction) ---
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Adder", add_node)
workflow.add_node("Multiplier", multiply_node)
workflow.add_node("Saver", saver_node)

# 添加边：Worker 完成后回到 Supervisor
workflow.add_edge("Adder", "Supervisor")
workflow.add_edge("Multiplier", "Supervisor")
workflow.add_edge("Saver", "Supervisor")

# 添加条件边
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Adder": "Adder",
        "Multiplier": "Multiplier",
        "Saver": "Saver",
        "FINISH": END
    }
)

workflow.set_entry_point("Supervisor")
graph = workflow.compile()

# --- 8. 运行 ---
print("Start Workflow...")
initial_state = {
    "messages": [
        HumanMessage(content="1+4*2等于多少？请计算并将结果保存到文件。")
    ]
}

# 增加递归限制，防止无限循环
for s in graph.stream(initial_state, {"recursion_limit": 20}):
    if "__end__" not in s:
        # 打印当前执行的节点名称和结果摘要
        node_name = list(s.keys())[0]
        print(f"--- 节点: {node_name} ---")
        if "messages" in s[node_name]:
            # 打印最新的一条消息内容
            print(f"输出: {s[node_name]['messages'][-1].content}")
        elif "next" in s[node_name]:
            print(f"决策: {s[node_name]['next']}")
        print("----")

print("Workflow Finished. Check result.txt")