
"""

langgraph 使用浏览器搜索工具进行搜索，并返回搜索结果
添加了 限制工具调用次数：
    对于任何全自动的代理，合理控制调用次数都是至关重要的一环，对于LangGraph React Agent来说，
    只需要在Agent运行的时候设置{"recursion_limit": X}，即可限制智能体自主执行任务时的步数。
    设置recursion_limit为 2 ，即可限制智能体自主执行任务时的步数为 2 。

"""

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

from langchain_tavily import TavilySearch

from dotenv import load_dotenv 
load_dotenv(override=True)



search_tool = TavilySearch(max_results=5, topic="general")

tools = [search_tool]
model = init_chat_model(model="deepseek-chat", model_provider="deepseek")  

search_agent = create_react_agent(model=model, tools=tools)

try:
    response = search_agent.invoke(
        {"messages": [{"role": "user", "content": "请问北京今天天气如何？"}]},
        {"recursion_limit": 2},
    )
except GraphRecursionError:
    print("Agent stopped due to max iterations.")


print("=================================================    1    ===============================================================")
print(response["messages"])

print("=================================================    2    ===============================================================")
print(response["messages"][-1].content)



"""
response["messages"] 的结构：

[

HumanMessage(content='请问北京今天天气如何？', additional_kwargs={}, response_metadata={}, id='dfb7af1b-562e-4d38-9ee6-c10b589daff2'), 

AIMessage(content='Sorry, need more steps to process this request.', additional_kwargs={}, response_metadata={}, id='run--86ddc3d0-4e0f-4d14-ac1f-bc3ac3a1f5db-0')

]

"""



























