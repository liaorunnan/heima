from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain_openai import ChatOpenAI
from conf import settings

model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)


agent = create_agent(
    model=model,
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=3,
                    clear_tool_inputs=False,
                    exclude_tools=[],
                    placeholder="[cleared]",
                ),
            ],
        ),
    ],
)