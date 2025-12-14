from a_llm.day_01.agent_demo import tools


def list_tools():
    return [{"name": name, "description":getattr(tools,name).__doc__} for name in dir(tools) if name.startswith("tool_")]

def call_tool(name, **kwargs):
    func=getattr(tools,name)
    return func(**kwargs)

if __name__ == '__main__':
    print([name for name in dir(tools) if name.startswith("tool_")])
    print(getattr(tools,'sub_tool').__doc__)
    print(getattr(tools,'sub_tool')(1,2))
    print(call_tool('mul_tool',a=1,b=2))
