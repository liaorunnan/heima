
"""
`LangGraph`中，如果没有显示的指定，则对状态中某个键的所有更新都执行的是覆盖操作。
"""


from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END

def addition(state):
    print(state)
    return {"x": state["x"] + 1}

def subtraction(state):
    print(state)
    return {"y": state["x"] - 2}

class State(TypedDict):
    x: int
    y: int


# 构建图
builder = StateGraph(State) 

# 向图中添加两个节点
builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

# 构建节点之间的边
builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

graph = builder.compile()

# 定义一个初始化的状态
initial_state = {"x":10}

result = graph.invoke(initial_state)
print("****************************************************************")
print("最终输出结果为：", result)























