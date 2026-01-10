"""
条件边复杂案例，根据状态决定流向。

使用`path_map`参数

"""

from langgraph.graph import START, StateGraph, END
from langgraph.graph import StateGraph

def node_a(state):
    return {"x": state["x"] + 1}

def node_b(state):
    return {"x": state["x"] - 2}

def node_c(state):
    return {"x": state["x"] + 1}

def routing_function(state):
    if state["x"] == 10:
        return True
    else:
        return False

builder = StateGraph(dict)

builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)

builder.set_entry_point("node_a")

# 构建节点之间的边
builder.add_conditional_edges("node_a", routing_function, path_map={True: "node_b", False: "node_c"})

builder.add_edge("node_b", END)
builder.add_edge("node_c", END)

graph = builder.compile()



# 可视化
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_03.png", "wb") as f:
    f.write(png_bytes)

























