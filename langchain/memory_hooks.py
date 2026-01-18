from dataclasses import dataclass

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langgraph.runtime import Runtime

from langchain_openai import ChatOpenAI

from conf import settings


@dataclass
class Context:
    user_name: str

# Dynamic prompts
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name  
    
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

# Before model hook
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:  
    print(f"Processing request for user: {runtime.context.user_name}")  
    runtime.context.user_name = "Echo"
    return None

# After model hook
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:  
    print(f"Completed request for user: {runtime.context.user_name}")  
    return None

model = ChatOpenAI(temperature=0.7, model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

agent = create_agent(
    model=model,
    middleware=[dynamic_system_prompt, log_before_model, log_after_model],  
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)

print(result["messages"][-1].content)
