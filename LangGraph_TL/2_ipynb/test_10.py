
"""
`LangGraph`中，如果没有显示的指定，则对状态中某个键的所有更新都执行的是覆盖操作。
下面显示指定，不再使用覆盖操作，而是使用拼接操作。

"""

import operator
from typing import Annotated, TypedDict, List
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END


class State(TypedDict):
    messages: Annotated[List[str], operator.add]


def addition(state):
    print(state)
    msg = state['messages'][-1]
    response = {"x": msg["x"] + 1}
    return {"messages": [response]}

def subtraction(state):
    print(state)
    msg = state['messages'][-1]
    response = {"x": msg["x"] - 2}
    return {"messages": [response]}


# 构建图
builder = StateGraph(State) 

# 向图中添加两个节点
builder.add_node("node1", addition)
builder.add_node("node2", subtraction)

# 构建节点之间的边
builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile()


input_state = {'messages': [{"x": 10}]}



result = graph.invoke(input_state)

print("****************************************************************")
print("最终输出结果为：", result)




