
"""

使用 StateGraph 定义一个简单的图

"""

from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph import START, END


# 定义输入的模式
class InputState(TypedDict):
    question: str

# 定义输出的模式
class OutputState(TypedDict):
    answer: str

# 将 InputState 和 OutputState 这两个 TypedDict 类型合并成一个字典类型。
class OverallState(InputState, OutputState):
    pass

# 明确指定它的输入和输出数据的结构或模式
builder = StateGraph(OverallState, input=InputState, output=OutputState)


def agent_node(state:InputState):
    print("我是一个AI Agent。")
    return

def action_node(state:InputState):
    print("我现在是一个执行者。")
    return {"answer":"我现在执行成功了"}


builder.add_node("agent_node", agent_node)
builder.add_node("action_node", action_node)

builder.add_edge(START, "agent_node")
builder.add_edge("agent_node", "action_node")
builder.add_edge("action_node", END)

graph = builder.compile()


result = graph.invoke({"question":"你好"})
print("****************************************************************")
print(result)

