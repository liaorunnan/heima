from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain_openai import ChatOpenAI
from conf import settings


@dataclass
class Context:
    name: str


@dynamic_prompt
def context_prompt(request) -> str:
    """
    将运行时 context 注入到系统提示词。
    注意：context_schema 仅做校验，不会自动写入模型消息。
    """
    ctx = request.runtime.context
    name = None
    if ctx is not None:
        if hasattr(ctx, "name"):
            name = ctx.name
        elif isinstance(ctx, dict):
            name = ctx.get("name")
    if name:
        return f"你是一个助手。当前用户名字是{name}。回答时请记住。"
    return "你是一个助手。"


def main():
    model = ChatOpenAI(
        temperature=0.7,
        model_name=settings.qw_model,
        max_tokens=1024,
        timeout=60,
        api_key=settings.qw_api_key,
        base_url=settings.qw_api_url,
    )

agent = create_agent(
    model=model,
        context_schema=Context,
        middleware=[context_prompt],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is my name?"}]},
        context=Context(name="Echo"),
)

print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
