from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from conf import settings
from langchain.tools import ToolRuntime
from langchain.messages import SystemMessage,AIMessage,HumanMessage

@dataclass
class UserContext:
    user_id: str


Usedata = [
    {"user_id": "user123", "name": "Bob Smith", "account_type": "Standard", "balance": 1200, "email": "bob@example.com"},
    {"user_id": "user456", "name": "Alice Johnson", "account_type": "Premium", "balance": 5000, "email": "alice@example.com"}
]

model = ChatOpenAI(temperature=0.7, model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

def get_account_info(runtime: ToolRuntime[UserContext]):
    """
    获取当前用户的账号信息
    """
    print(runtime.state)
    user_id = runtime.context.user_id
    for user_data in Usedata:
        if user_data["user_id"] == user_id:
            return f"账号信息: {user_data['name']}\n账号类型: {user_data['account_type']}\n余额: ${user_data['balance']}"
        
    return "User not found"

agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)


messages = {
    "messages":[{"role":"user","content":"我的余额是多少？"}],
    "context": UserContext(user_id="user123")
}

result = agent.invoke(messages)

print(result)