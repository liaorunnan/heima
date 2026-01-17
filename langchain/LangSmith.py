from langchain_core.runnables import astream_events
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers import LangChainTracer

# 启用 LangSmith 跟踪
from langsmith import Client
client = Client()

# 创建跟踪器
tracer = LangChainTracer(project_name="agent_debug")

async def stream_with_tracing():
    async for event in astream_events(
        agent,
        input={"input": "计算 2+2*3"},
        config=RunnableConfig(
            callbacks=[tracer]
        )
    ):
        print(f"事件: {event['event']}")

# 运行
import asyncio
asyncio.run(stream_with_tracing())