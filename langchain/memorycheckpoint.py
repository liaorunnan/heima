from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI
from conf import settings


model = ChatOpenAI(temperature=0.7, model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

agent = create_agent(
    model=model,
    checkpointer=InMemorySaver()

)

message = {
    "messages": [{"role": "user", "content": "你好,我叫张三"}]
}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好,我叫张三"}]},
    {"configurable":{"thread_id":1}}
)

print(result['messages'][-1].content)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好,我叫李四"}]},
    {"configurable":{"thread_id":2}}
)
print(result['messages'][-1].content)


result = agent.invoke(
    {"messages": [{"role": "user", "content": "我刚刚说我叫什么？"}]},
    {"configurable":{"thread_id":1}}
)
print(result['messages'][-1].content)