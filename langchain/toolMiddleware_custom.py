from langchain.agents.middleware import (
    before_model, after_model, before_agent, after_agent,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable
from langchain_openai import ChatOpenAI
from conf import settings

model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("agent运行之前")
    return None

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("agent运行之后")
    return None


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("模型运行之前")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("模型运行之后")
    return None



@wrap_model_call
def wrap_model_call(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse: 
    print("调用模型")
    # 调用实际的模型处理函数并返回结果
    return handler(request)

agent = create_agent(
    model=model,
    middleware=[log_before_agent, log_before_model, wrap_model_call, log_after_model, log_after_agent],
)

message = {"messages": [{"role": "user", "content": "你好,我的名字是什么"}]}
result = agent.invoke(message)
print(result["messages"][-1].content)