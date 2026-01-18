from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

from conf import settings

model = ChatOpenAI(temperature=0.7, model_name=settings.model_name,max_tokens=1024,timeout=60,api_key=settings.api_key, base_url=settings.base_url)



# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """
    从内存存储中查询用户信息
    
    Args:
        user_id (str): 用户的唯一标识符
        runtime (ToolRuntime): 工具运行时对象，包含内存存储的访问接口
        
    Returns:
        str: 查询到的用户信息字符串，如果用户不存在则返回"Unknown user"
    
    示例:
        # 调用示例
        result = get_user_info("abc123", runtime)
        # 返回: "{'name': 'Foo', 'age': 25, 'email': 'foo@langchain.dev'}"
    """
    # 从运行时获取内存存储实例
    store = runtime.store
    
    # 使用复合键("users",)和user_id从存储中获取用户信息
    # ("users",) 作为命名空间，user_id 作为具体的用户标识符
    user_info = store.get(("users",), user_id)
    
    # 如果找到用户信息，则返回其值的字符串表示；否则返回"Unknown user"
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """
    将用户信息保存到内存存储中
    
    Args:
        user_id (str): 用户的唯一标识符
        user_info (dict[str, Any]): 包含用户信息的字典
        runtime (ToolRuntime): 工具运行时对象，包含内存存储的访问接口
        
    Returns:
        str: 保存成功的提示信息
    
    示例:
        # 调用示例
        result = save_user_info(
            "abc123", 
            {"name": "Foo", "age": 25, "email": "foo@langchain.dev"}, 
            runtime
        )
        # 返回: "Successfully saved user info."
    """
    # 从运行时获取内存存储实例
    store = runtime.store
    
    # 使用复合键("users",)和user_id将用户信息保存到存储中
    # ("users",) 作为命名空间，user_id 作为具体的用户标识符
    store.put(("users",), user_id, user_info)
    
    # 返回保存成功的提示信息
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
result1= agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})
print(result1["messages"][-1].content)

# Second session: get user info
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
print(result2["messages"][-1].content)
