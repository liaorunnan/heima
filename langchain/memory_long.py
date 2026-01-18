from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

from langchain_openai import ChatOpenAI
from conf import settings


@dataclass
class Context:
    user_id: str

# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production.
store = InMemoryStore() 

# Write sample data to the store using the put method
store.put( 
    ("users",),  # Namespace to group related data together (users namespace for user data)
    "user_123",  # Key within the namespace (user ID as key)
    {
        "name": "Echo",
        "language": "Chinese",
    }  # Data to store for the given user
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    # Access the store - same as that provided to `create_agent`
    store = runtime.store 
    user_id = runtime.context.user_id
    # Retrieve data from store - returns StoreValue object with value and metadata
    user_info = store.get(("users",), user_id) 
    return str(user_info.value) if user_info else "Unknown user"

model = ChatOpenAI(temperature=0.7, model_name=settings.qw_model,max_tokens=1024,timeout=60,api_key=settings.qw_api_key, base_url=settings.qw_api_url)

agent = create_agent(
    model=model,
    tools=[get_user_info],
    store=store, 
    context_schema=Context
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "查看我的信息"}]},
    context=Context(user_id="user_123") 
)

print(result['messages'][-1].content)