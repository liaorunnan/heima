from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.types import RunnableConfig
import random



class EmailAgentState(TypedDict):
    email: str
    intent: str
    documentation: str
    bug_tracking: str
    response: str
    human_review: str
    reply: str
    next_action: str


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
    return State['next_action']

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

# 只添加必要的边
workflow.add_edge(START, "read_email")
# workflow.add_edge("read_email", "classify_intent")

# workflow.add_conditional_edges(
#     "classify_intent",
#     lambda state: state["next_action"],
#     {
#         "send_reply": "send_reply",
#         "send_reply_error": "send_reply_error"
#     }
# )

workflow.add_conditional_edges("read_email", classify_intent)
workflow.add_edge("send_reply", END)
workflow.add_edge("send_reply_error", END)


config1 = RunnableConfig(configurable={
    "thread_id": "12345"
})

# 使用检查点进行持久化编译
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
app.invoke({"email": "test@example.com"}, config=config1)