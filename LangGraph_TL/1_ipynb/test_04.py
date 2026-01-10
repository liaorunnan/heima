


from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph import START, END

# 定义输入的模式
class InputState(TypedDict):
    question: str

# 定义输出的模式
class OutputState(TypedDict):
    answer: str

# 将 InputState 和 OutputState 这两个 TypedDict 类型合并成一个更全面的字典类型。
class OverallState(InputState, OutputState):
    pass

def agent_node(state: InputState):
    print("我是一个AI Agent。")
    return {"question": state["question"]}

def action_node(state: InputState):
    print("我现在是一个执行者。")
    step = state["question"]
    return {"answer": f"我接收到的问题是：{step}，读取成功了！"}

# 明确指定它的输入和输出数据的结构或模式
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# 添加节点
builder.add_node("agent_node", agent_node)
builder.add_node("action_node", action_node)

# 添加便
builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "action_node")
builder.add_edge("action_node", END)

# 编译图
graph = builder.compile()

result = graph.invoke({"question":"今天的天气怎么样？"})
print("****************************************************************")
print(result)

















