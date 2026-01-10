

"""
前边定义图时使用的是 StateGraph ，为了标准化输入输出格式，传入的参数是 TypedDict 类型的 OverallState
这里不再固定输入输出格式，直接使用 dict ，这样的话，输入、输出可以灵活地定义
可视化图

"""

import os
from langgraph.graph import StateGraph
from langgraph.graph import START, END




# 构建图
builder = StateGraph(dict) 

print("所创建图的输入输出格式（模式）为：", builder.schema)
print("****************************************************************")

def addition(state):
    print(state)
    return {"x": state["x"] + 1}

def subtraction(state):
    print(state)
    return {"y": state["x"] - 2}



# 向图中添加两个节点
builder.add_node("addition", addition)
builder.add_node("subtraction", subtraction)

# 构建节点之间的边
builder.add_edge(START, "addition")
builder.add_edge("addition", "subtraction")
builder.add_edge("subtraction", END)

print("所创建图的边为：", builder.edges)
print("****************************************************************")
print("所创建图的节点为：", builder.nodes)
print("****************************************************************")


graph = builder.compile()


"""
# 可视化 graph 图用，在jupyter中运行可直接显示图片，这里需要先将其保存为图片再查看
from IPython.display import Image, display
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
"""

# 由于需要将输出的图先保存为图片，所以使用如下代码，而不是上面的两行
png_bytes = graph.get_graph(xray=True).draw_mermaid_png()
with open("graph_01.png", "wb") as f:
    f.write(png_bytes)




# 定义一个初始化的状态
initial_state = {"x":10}

result = graph.invoke(initial_state)


print("最终输出结果为：", result)
print("****************************************************************")




