from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.types import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage



from llm_input import model

import random



class EmailAgentState(TypedDict):
    email: str
    response: str = ""
    next_action: str = ""


def read_email(State: EmailAgentState):
    email = State["email"]
    print(f"正在读取邮件: {email}")
    return State

def classify_intent(State: EmailAgentState):
    print("正在分类意图...")
    number = random.randint(0, 1)
    if number == 0:
        State["next_action"] = "send_reply"
    else:
        State["next_action"] = "send_reply_error"
    return State
def classify_intent_node(State: EmailAgentState):
    return State["next_action"]

def search_documentation(State: EmailAgentState):
    print("正在搜索文档...")
    return State

def bug_tracking(State: EmailAgentState):
    print("正在跟踪问题...")
    return State

def draft_response(State: EmailAgentState):
    print("正在起草回复...")
    return State

def human_review(State: EmailAgentState):
    print("正在人类审核...")
    return State

def send_reply(State: EmailAgentState):
    print("正在发送回复...")
    return State

def send_reply_error(State: EmailAgentState):
    print("回复错误...")
    return State


def error_handler(State: EmailAgentState):


    message = [
        SystemMessage(content="你是一个专业的电子邮件助手，负责处理用户的电子邮件问题，根据我给你的字符串判断是否发送成功了，如果是send_reply便是成功，如果是send_reply_error便是失败，发送成功返回success，否则返回error"),
        HumanMessage(content=State["next_action"])
    ]
    
    response = model.invoke(message)
    State["response"] = response.content
    print(f"回复: {response.content}")
    return State
    


# 创建图
workflow = StateGraph(EmailAgentState)

# 添加带有适当错误处理的节点
workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)

# 为可能有瞬态故障的节点添加重试策略
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3)
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)
workflow.add_node("send_reply_error", send_reply_error)
workflow.add_node("error_handler", error_handler)


# 只添加必要的边
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    classify_intent_node
)

# workflow.add_conditional_edges("read_email", classify_intent)
workflow.add_edge("send_reply", error_handler.__name__)
workflow.add_edge("send_reply_error", error_handler.__name__)
workflow.add_edge("error_handler", END)


config1 = RunnableConfig(configurable={
    "thread_id": "12345"
})

# 使用检查点进行持久化编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
app.invoke({"email": "test@example.com", "next_action": "", "response": ""}, config=config1)